"""Pinned NanoLLaVA keyframe semantics for the local Apple-silicon fallback.

This adapter is deliberately not called VideoLLaMA.  NanoLLaVA describes a
small deterministic set of still frames; it does not consume the soundtrack,
model continuous motion, or predict an audience outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Protocol, Sequence

from .torch_runtime import (
    runtime_request_from_env as torch_request_from_env,
    unavailable_reason as torch_unavailable_reason,
)
from ..worker_protocol import (
    WORKER_OUTPUT_SCHEMA_VERSION,
    BranchOutput,
    WorkerIdentity,
    validate_branch_output,
)


MODEL_ID = "mlx-community/nanoLLaVA-1.5-8bit"
MODEL_REVISION = "6e2bc13b87ab178668313552b9d69026af7d556f"
MODEL_WEIGHTS_SHA256 = (
    "f0a1d1517ad9e6e810bc6ac99956643c66e4b87a2f82bd5d1b5cb0966e5c5476"
)
MODEL_LICENSE = "Apache-2.0"
ADAPTER_ID = "nanollava-keyframe-mlx-fallback"
# Protocol v1 names this 40-character field ``codeRevision``. For this local
# non-Git wrapper it is the SHA-1 content identity of both executable sources.
STDIO_HELPER_PATH = Path(__file__).with_name("nanollava_stdio.py").resolve()


def _worker_code_revision() -> str:
    digest = hashlib.sha1()
    for path in (Path(__file__).resolve(), STDIO_HELPER_PATH):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


WORKER_CODE_REVISION = _worker_code_revision()
PREPROCESSING_ID = "ffmpeg-keyframes-6x384-png-center-sampled/1"
PREPROCESSING_SHA256 = (
    "ff850282d6471942fa9a41e26aa23cd3d223c4a6822781af47feffd7497ebbe7"
)
# The portable backend's artifact. A different repository from the MLX one on
# purpose: those weights are MLX-quantised and torch cannot read them.
TORCH_MODEL_ID = "qnguyen3/nanoLLaVA-1.5"
TORCH_BACKEND_ENV = "FORECAST_NANOLLAVA_BACKEND"
TORCH_REVISION_ENV = "FORECAST_NANOLLAVA_TORCH_REVISION"
TRUST_REMOTE_CODE_ENV = "FORECAST_NANOLLAVA_TRUST_REMOTE_CODE"
MLX_BACKEND_NAME = "mlx-vlm"
TORCH_BACKEND_NAME = "transformers"
NANOLLAVA_BACKENDS = (MLX_BACKEND_NAME, TORCH_BACKEND_NAME)
TORCH_ADAPTER_ID = "nanollava-keyframe-transformers"
# The reserved position the model substitutes its vision embedding into.
IMAGE_TOKEN_INDEX = -200
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


def _resolve_pinned_torch_snapshot(model_id: str, revision: str) -> str:
    """Resolve the pinned revision from the local cache, without any network."""

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_id, revision=revision, local_files_only=True)


SNAPSHOT_ENV = "FORECAST_NANOLLAVA_SNAPSHOT"
FFMPEG_ENV = "FORECAST_NANOLLAVA_FFMPEG_BINARY"
PYTHON_ENV = "FORECAST_NANOLLAVA_PYTHON"
SUBPROCESS_TIMEOUT_ENV = "FORECAST_NANOLLAVA_TIMEOUT_SECONDS"
KEYFRAME_COUNT = 6
# The six keyframes are sampled proportionally, so the first one always lands at
# duration/12 and the hook window's share of them shrinks as a clip gets longer:
# past ~36 seconds no proportional frame falls inside the first three seconds at
# all. Hook frames are therefore sampled at fixed absolute times, under their own
# preprocessing identity so the pinned six-frame contract is untouched.
# Six times, matching KEYFRAME_COUNT, so the isolated runtime contract that
# requires exactly six frames holds for the hook pass unchanged.
HOOK_FRAME_TIMES: tuple[float, ...] = (0.0, 0.4, 1.0, 1.8, 2.4, 3.0)
HOOK_PREPROCESSING_ID = "ffmpeg-hookframes-6x384-png-absolute/1"
HOOK_PASS_ENV = "FORECAST_NANOLLAVA_HOOK_PASS"
HOOK_FRAME_SPAN_SECONDS = 0.4


def _two_pass_preprocessing() -> tuple[str, str]:
    """Declare the combined contract when both passes run.

    The clip digest above is an audited constant and is never altered. When the
    hook pass is enabled the branch runs a second, differently pinned
    preprocessing contract, so provenance must describe both rather than claim
    only the first. The combined digest is derived — reproducibly — from the two
    contracts it covers.
    """

    combined_id = f"{PREPROCESSING_ID}+{HOOK_PREPROCESSING_ID}"
    payload = json.dumps(
        {
            "passes": [
                {"id": PREPROCESSING_ID, "sha256": PREPROCESSING_SHA256},
                {"id": HOOK_PREPROCESSING_ID, "times": list(HOOK_FRAME_TIMES)},
            ]
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return combined_id, hashlib.sha256(payload.encode("utf-8")).hexdigest()


TWO_PASS_PREPROCESSING_ID, TWO_PASS_PREPROCESSING_SHA256 = _two_pass_preprocessing()
MAX_RESPONSE_CHARACTERS = 12_000
MAX_BATCH_RESPONSE_BYTES = 512 * 1024
MAX_TOKENS = 320
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
BATCH_REQUEST_SCHEMA = "forecast-nanollava-stdio-request/1"
BATCH_RESPONSE_SCHEMA = "forecast-nanollava-stdio-response/1"

SYSTEM_BOUNDARY = (
    "Analyze only directly visible evidence in this one still frame. Do not "
    "infer identity, emotion, intent, speech, or unseen events. Do not predict "
    "attention, retention, engagement, virality, or any brain response."
)
FRAME_PROMPT = """Analyze the supplied image using only directly visible evidence.
Return only one valid JSON object, without markdown or commentary. It has
exactly five fields: scene says what is visible; action says what is visibly
happening or none; visibleText is null because OCR is unavailable; shot says
how the image is framed; uncertainties is null or a short array of visual
ambiguities. Fill scene, action, and shot from the actual image. Never copy or
restate these instructions. {boundary}
"""

PLACEHOLDER_ECHOES = (
    "short literal scene description",
    "visible action or none",
    "literal readable text",
    "literal framing/camera description",
    "anything not visually certain",
    "one short factual sentence naming visible people and objects",
    "one short factual sentence describing visible physical action",
    "one short factual phrase describing framing and camera view",
    "visibletext mustbe",
    "visibletext must be null",
)


class NanoLlavaUnavailable(RuntimeError):
    """The pinned local snapshot or optional MLX runtime is not executable."""


class NanoLlavaOutputError(RuntimeError):
    """NanoLLaVA returned no contract-valid keyframe observations."""


class FrameExtractor(Protocol):
    def extract(
        self, video_path: Path, duration_seconds: float, destination: Path
    ) -> Sequence[tuple[float, Path]]:
        ...


class SemanticBackend(Protocol):
    def describe(self, image_path: Path, prompt: str) -> str:
        ...


class BatchSemanticBackend(Protocol):
    def describe_many(self, image_paths: Sequence[Path], prompt: str) -> Sequence[str]:
        ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _default_snapshot() -> Path:
    huggingface_home = Path(
        os.environ.get(
            "HF_HOME",
            str(Path.home() / ".cache" / "huggingface"),
        )
    )
    return (
        huggingface_home
        / "hub"
        / "models--mlx-community--nanoLLaVA-1.5-8bit"
        / "snapshots"
        / MODEL_REVISION
    )


def hook_frame_times(
    duration_seconds: float, times: tuple[float, ...] = HOOK_FRAME_TIMES
) -> tuple[float, ...]:
    """Fixed absolute sample times inside the hook, clipped to the clip's length.

    Unlike the proportional pass, these do not move with duration: the opening
    frame of a 12-second clip and a 55-second clip are both sampled at t=0.
    """

    if (
        not isinstance(duration_seconds, (int, float))
        or isinstance(duration_seconds, bool)
        or not math.isfinite(float(duration_seconds))
        or duration_seconds <= 0
    ):
        raise ValueError("duration must be positive")
    duration = float(duration_seconds)
    selected: list[float] = []
    for candidate in times:
        moment = round(min(float(candidate), max(0.0, duration - 0.05)), 6)
        if moment not in selected:
            selected.append(moment)
    return tuple(selected)


def deterministic_sample_times(
    duration_seconds: float, count: int = KEYFRAME_COUNT
) -> tuple[float, ...]:
    if (
        not isinstance(duration_seconds, (int, float))
        or isinstance(duration_seconds, bool)
        or not math.isfinite(float(duration_seconds))
        or duration_seconds <= 0
        or count <= 0
    ):
        raise ValueError("duration and keyframe count must be positive")
    duration = float(duration_seconds)
    return tuple(round(duration * (index + 0.5) / count, 6) for index in range(count))


@dataclass(slots=True)
class FfmpegKeyframeExtractor:
    binary: str = "ffmpeg"
    timeout_seconds: float = 30.0
    count: int = KEYFRAME_COUNT

    def extract(
        self,
        video_path: Path,
        duration_seconds: float,
        destination: Path,
        sample_times: tuple[float, ...] | None = None,
    ) -> tuple[tuple[float, Path], ...]:
        """Decode frames at the given times, or at the proportional defaults."""

        destination.mkdir(parents=True, exist_ok=True)
        frames: list[tuple[float, Path]] = []
        planned = (
            sample_times
            if sample_times is not None
            else deterministic_sample_times(duration_seconds, self.count)
        )
        for index, sample_time in enumerate(planned):
            output = destination / f"frame-{index:02d}.png"
            command = [
                self.binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-ss",
                f"{sample_time:.6f}",
                "-i",
                str(video_path),
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-vf",
                (
                    "scale=384:384:force_original_aspect_ratio=decrease,"
                    "pad=384:384:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
                ),
                "-c:v",
                "png",
                "-y",
                str(output),
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise NanoLlavaUnavailable(
                    "Deterministic ffmpeg keyframe extraction is unavailable."
                ) from exc
            if completed.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
                raise NanoLlavaUnavailable(
                    "Deterministic ffmpeg keyframe extraction failed."
                )
            frames.append((sample_time, output))
        return tuple(frames)


class MlxVlmSemanticBackend:
    """Lazy optional bridge to the documented ``mlx_vlm`` load/generate API."""

    def __init__(self, snapshot: Path) -> None:
        self.snapshot = snapshot
        self._model: Any = None
        self._processor: Any = None
        self._generate: Any = None

    @staticmethod
    def installed() -> bool:
        return importlib.util.find_spec("mlx_vlm") is not None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            module = importlib.import_module("mlx_vlm")
            load = getattr(module, "load")
            generate = getattr(module, "generate")
            self._model, self._processor = load(str(self.snapshot))
            self._generate = generate
        except Exception as exc:
            raise NanoLlavaUnavailable(
                "The optional mlx_vlm runtime could not load the pinned local snapshot."
            ) from exc

    def describe(self, image_path: Path, prompt: str) -> str:
        self._load()
        formatted_prompt = f"<image>\n{prompt}"
        apply_template = getattr(self._processor, "apply_chat_template", None)
        if callable(apply_template):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            formatted_prompt = apply_template(messages, add_generation_prompt=True)

        kwargs: dict[str, Any] = {"verbose": False, "max_tokens": MAX_TOKENS}
        try:
            parameters = inspect.signature(self._generate).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "temperature" in parameters:
            kwargs["temperature"] = 0.0
        if "image" in parameters:
            kwargs["image"] = str(image_path)
            result = self._generate(
                self._model, self._processor, formatted_prompt, **kwargs
            )
        elif "images" in parameters:
            kwargs["images"] = [str(image_path)]
            result = self._generate(
                self._model, self._processor, formatted_prompt, **kwargs
            )
        else:
            result = self._generate(
                self._model,
                self._processor,
                formatted_prompt,
                str(image_path),
                **kwargs,
            )
        text = result if isinstance(result, str) else getattr(result, "text", None)
        if not isinstance(text, str):
            raise NanoLlavaOutputError("mlx_vlm returned no text result")
        return text


class TransformersVlmSemanticBackend:
    """NanoLLaVA through torch, for hosts where MLX does not exist.

    This is a *different pinned artifact* from the MLX backend, not the same one
    on other hardware: ``mlx-community/nanoLLaVA-1.5-8bit`` holds MLX-quantised
    weights torch cannot read, so the portable path names the upstream
    repository and carries its own model identity into the manifest.

    The upstream repository ships an ``auto_map``, so loading it executes model
    code from the snapshot. That is a real change in what this lane does, so it
    is refused unless the operator opts in explicitly. The pinned 40-character
    commit is the integrity anchor: the hub resolves that revision to exact
    content, and an unpinned revision keeps the lane unavailable.
    """

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        snapshot_resolver: Any = None,
        loader: Any = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self._snapshot_resolver = snapshot_resolver or _resolve_pinned_torch_snapshot
        self._loader = loader
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self.runtime: Any = None

    @staticmethod
    def installed() -> bool:
        return (
            importlib.util.find_spec("transformers") is not None
            and importlib.util.find_spec("torch") is not None
        )

    def unavailable_reason(self) -> str | None:
        """Why this backend could not run here, or ``None`` if it could."""

        if not COMMIT_RE.fullmatch(self.revision or ""):
            return (
                f"{TORCH_REVISION_ENV} must be a full 40-character commit SHA; "
                "mutable names such as 'main' are rejected."
            )
        if not self.trust_remote_code:
            return (
                f"{TRUST_REMOTE_CODE_ENV} is not enabled. The portable NanoLLaVA "
                "weights ship an auto_map, so loading them runs model code from "
                "the pinned snapshot. Enable it only if you accept that."
            )
        if self._loader is None and not self.installed():
            return "transformers and torch are not installed in this backend environment."
        if self._loader is None:
            blocked = torch_unavailable_reason(
                requested_device=self.device, requested_dtype=self.dtype
            )
            if blocked:
                return blocked
        try:
            self._snapshot_resolver(self.model_id, self.revision)
        except Exception:
            return (
                "The pinned NanoLLaVA revision is not present in the local snapshot "
                "cache. Fetch it once before running offline."
            )
        return None

    def _load(self) -> None:
        if self._model is not None:
            return
        blocked = self.unavailable_reason()
        if blocked:
            raise NanoLlavaUnavailable(blocked)
        snapshot = self._snapshot_resolver(self.model_id, self.revision)
        if self._loader is not None:
            self._model, self._tokenizer, self._torch = self._loader(snapshot)
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from .torch_runtime import resolve_runtime, torch_dtype

            runtime = resolve_runtime(
                requested_device=self.device,
                requested_dtype=self.dtype,
                torch_module=torch,
            )
            self.runtime = runtime
            model = AutoModelForCausalLM.from_pretrained(
                snapshot,
                torch_dtype=torch_dtype(torch, runtime.dtype),
                trust_remote_code=True,
            ).to(runtime.device)
            model.eval()
            self._model = model
            self._tokenizer = AutoTokenizer.from_pretrained(
                snapshot, trust_remote_code=True
            )
            self._torch = torch
        except Exception as exc:
            raise NanoLlavaUnavailable(
                "The optional transformers runtime could not load the pinned "
                f"NanoLLaVA snapshot: {type(exc).__name__}"
            ) from exc

    def describe(self, image_path: Path, prompt: str) -> str:
        """One frame, one description, following the model's documented interface."""

        self._load()
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model

        messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # The image occupies one reserved position in the token stream; the model
        # substitutes its own vision embedding at index -200.
        before, _, after = text.partition("<image>")
        head = tokenizer(before).input_ids
        tail = tokenizer(after).input_ids[1:]
        input_ids = torch.tensor(
            head + [IMAGE_TOKEN_INDEX] + tail, dtype=torch.long
        ).unsqueeze(0)

        from PIL import Image

        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            pixels = model.process_images([image], model.config)
        pixels = pixels.to(dtype=model.dtype, device=model.device)
        input_ids = input_ids.to(device=model.device)

        with torch.inference_mode():
            produced = model.generate(
                input_ids,
                images=pixels,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
                use_cache=True,
            )[0]
        text_out = tokenizer.decode(
            produced[input_ids.shape[1]:], skip_special_tokens=True
        )
        if not isinstance(text_out, str) or not text_out.strip():
            raise NanoLlavaOutputError("the transformers runtime returned no text result")
        return text_out.strip()


class IsolatedMlxVlmBatchBackend:
    """Run one six-frame request in an explicitly separate Python runtime.

    The configured executable is never resolved through the current process's
    import path.  The child receives only local, hash-pinned paths and is
    forced into the Hugging Face offline modes before it imports ``mlx_vlm``.
    """

    def __init__(
        self,
        python_executable: Path,
        snapshot: Path,
        *,
        timeout_seconds: float = 300.0,
        runner: Any = subprocess.run,
        configuration_error: str | None = None,
    ) -> None:
        # Keep the configured path verbatim. Resolving a venv's ``bin/python``
        # symlink would replace it with the base interpreter and lose the venv.
        self.python_executable = python_executable
        self.snapshot = snapshot.resolve()
        self.timeout_seconds = timeout_seconds
        self.runner = runner
        self.configuration_error = configuration_error

    def unavailable_reason(self) -> str | None:
        if self.configuration_error:
            return self.configuration_error
        if not self.python_executable.is_absolute():
            return (
                f"{PYTHON_ENV} must be an absolute path to the isolated "
                "environment's Python executable."
            )
        if (
            not self.python_executable.is_file()
            or not os.access(self.python_executable, os.X_OK)
        ):
            return (
                f"{PYTHON_ENV} does not point to an executable file in an "
                "isolated MLX-VLM environment."
            )
        if not STDIO_HELPER_PATH.is_file():
            return "The pinned NanoLLaVA isolated-runtime helper is unavailable."
        return None

    def _require_available(self) -> None:
        reason = self.unavailable_reason()
        if reason:
            raise NanoLlavaUnavailable(reason)

    @staticmethod
    def _child_environment() -> dict[str, str]:
        environment = dict(os.environ)
        for name in (
            "PYTHONPATH",
            "PYTHONHOME",
            "HF_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
            "MLX_TRUST_REMOTE_CODE",
        ):
            environment.pop(name, None)
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "PYTHONNOUSERSITE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        return environment

    def describe_many(
        self, image_paths: Sequence[Path], prompt: str
    ) -> tuple[str, ...]:
        self._require_available()
        if len(image_paths) != KEYFRAME_COUNT:
            raise NanoLlavaOutputError(
                "The isolated NanoLLaVA runtime requires exactly six keyframes."
            )
        frames: list[dict[str, Any]] = []
        expected_hashes: list[str] = []
        for index, raw_path in enumerate(image_paths):
            path = raw_path.resolve()
            if not path.is_file() or path.stat().st_size <= 0:
                raise NanoLlavaOutputError(
                    "An isolated NanoLLaVA keyframe is unavailable."
                )
            try:
                digest = _sha256_file(path)
            except OSError as exc:
                raise NanoLlavaOutputError(
                    "An isolated NanoLLaVA keyframe could not be read."
                ) from exc
            expected_hashes.append(digest)
            frames.append({"index": index, "path": str(path), "sha256": digest})

        request = {
            "schemaVersion": BATCH_REQUEST_SCHEMA,
            "snapshot": str(self.snapshot),
            "modelRevision": MODEL_REVISION,
            "modelWeightsSha256": MODEL_WEIGHTS_SHA256,
            "prompt": prompt,
            "maxTokens": MAX_TOKENS,
            "frames": frames,
        }
        request_bytes = json.dumps(
            request,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        command = [
            str(self.python_executable),
            "-I",
            str(STDIO_HELPER_PATH),
        ]
        try:
            completed = self.runner(
                command,
                input=request_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=self.timeout_seconds,
                env=self._child_environment(),
            )
        except subprocess.TimeoutExpired as exc:
            raise NanoLlavaUnavailable(
                "The isolated NanoLLaVA runtime exceeded its configured timeout."
            ) from exc
        except OSError as exc:
            raise NanoLlavaUnavailable(
                "The isolated NanoLLaVA runtime could not be started."
            ) from exc
        if completed.returncode != 0:
            raise NanoLlavaUnavailable(
                "The isolated NanoLLaVA runtime failed without producing evidence."
            )
        return _strict_batch_response(completed.stdout, expected_hashes)


def _strict_json_object(payload: str) -> dict[str, Any]:
    if not isinstance(payload, str) or not payload.strip():
        raise NanoLlavaOutputError("empty semantic response")
    if len(payload) > MAX_RESPONSE_CHARACTERS:
        raise NanoLlavaOutputError("semantic response exceeds the contract limit")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise NanoLlavaOutputError("semantic response contains duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(payload.strip(), object_pairs_hook=pairs)
    except (json.JSONDecodeError, TypeError) as exc:
        raise NanoLlavaOutputError("semantic response is not strict JSON") from exc
    if not isinstance(value, dict) or set(value) != {
        "scene",
        "action",
        "visibleText",
        "shot",
        "uncertainties",
    }:
        raise NanoLlavaOutputError("semantic response has an invalid object shape")

    def text_field(name: str, maximum: int) -> str:
        item = value[name]
        if (
            not isinstance(item, str)
            or not item.strip()
            or item != item.strip()
            or len(item) > maximum
            or any(ord(character) < 32 for character in item)
        ):
            raise NanoLlavaOutputError(f"semantic response {name} is invalid")
        return item

    def nullable_text_list(
        name: str, maximum_items: int, maximum_length: int
    ) -> list[str] | None:
        items = value[name]
        if items is None:
            return None
        if not isinstance(items, list) or len(items) > maximum_items:
            raise NanoLlavaOutputError(f"semantic response {name} is invalid")
        normalized: list[str] = []
        for item in items:
            if (
                not isinstance(item, str)
                or not item.strip()
                or item != item.strip()
                or len(item) > maximum_length
                or any(ord(character) < 32 for character in item)
            ):
                raise NanoLlavaOutputError(f"semantic response {name} is invalid")
            normalized.append(item)
        return normalized

    normalized = {
        "scene": text_field("scene", 600),
        "action": text_field("action", 400),
        "visibleText": nullable_text_list("visibleText", 12, 160),
        "shot": text_field("shot", 240),
        "uncertainties": nullable_text_list("uncertainties", 12, 240),
    }
    flattened = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).casefold()
    if any(marker in flattened for marker in PLACEHOLDER_ECHOES):
        raise NanoLlavaOutputError("semantic response echoed an instruction placeholder")
    return normalized


def _strict_batch_response(
    payload: bytes, expected_hashes: Sequence[str]
) -> tuple[str, ...]:
    if not isinstance(payload, bytes) or not payload:
        raise NanoLlavaOutputError(
            "The isolated NanoLLaVA runtime returned no JSON output."
        )
    if len(payload) > MAX_BATCH_RESPONSE_BYTES:
        raise NanoLlavaOutputError(
            "The isolated NanoLLaVA response exceeded the contract limit."
        )

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise NanoLlavaOutputError(
                    "The isolated NanoLLaVA response contains duplicate keys."
                )
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise NanoLlavaOutputError(
            "The isolated NanoLLaVA response is not strict UTF-8 JSON."
        ) from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"schemaVersion", "responses"}
        or value["schemaVersion"] != BATCH_RESPONSE_SCHEMA
        or not isinstance(value["responses"], list)
        or len(value["responses"]) != KEYFRAME_COUNT
        or len(expected_hashes) != KEYFRAME_COUNT
    ):
        raise NanoLlavaOutputError(
            "The isolated NanoLLaVA response has an invalid batch shape."
        )

    responses: list[str] = []
    for index, item in enumerate(value["responses"]):
        if (
            not isinstance(item, dict)
            or set(item) != {"index", "frameSha256", "text"}
            or item["index"] != index
            or item["frameSha256"] != expected_hashes[index]
            or not isinstance(item["text"], str)
            or not item["text"].strip()
            or item["text"] != item["text"].strip()
            or len(item["text"]) > MAX_RESPONSE_CHARACTERS
        ):
            raise NanoLlavaOutputError(
                "The isolated NanoLLaVA response failed frame provenance validation."
            )
        responses.append(item["text"])
    return tuple(responses)


def _observation_bounds(
    sample_times: Sequence[float], index: int, duration_seconds: float
) -> tuple[float, float]:
    start = (
        0.0
        if index == 0
        else (sample_times[index - 1] + sample_times[index]) / 2.0
    )
    end = (
        duration_seconds
        if index == len(sample_times) - 1
        else (sample_times[index] + sample_times[index + 1]) / 2.0
    )
    return round(start, 6), round(end, 6)


class NanoLlavaSemanticAdapter:
    """Local adapter with the same ``run`` surface as remote branch adapters."""

    branch = "semanticModel"

    def __init__(
        self,
        snapshot: Path,
        *,
        backend: SemanticBackend | BatchSemanticBackend | None = None,
        frame_extractor: FrameExtractor | None = None,
        hook_pass: bool = True,
    ) -> None:
        self.snapshot = snapshot.resolve()
        self.backend = backend
        self.frame_extractor = frame_extractor or FfmpegKeyframeExtractor()
        # A second described pass over the hook window. It doubles this lane's
        # model time, so an operator can turn it off; it is on by default because
        # without it the opening of a longer clip is never described at all.
        self.hook_pass = hook_pass

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "NanoLlavaSemanticAdapter":
        source = os.environ if environ is None else environ
        raw_snapshot = str(source.get(SNAPSHOT_ENV, "")).strip()
        snapshot = Path(raw_snapshot).expanduser() if raw_snapshot else _default_snapshot()
        binary = str(source.get(FFMPEG_ENV, "ffmpeg")).strip() or "ffmpeg"
        hook_pass = str(source.get(HOOK_PASS_ENV, "true")).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        kind = str(source.get(TORCH_BACKEND_ENV, MLX_BACKEND_NAME)).strip().lower()
        if kind not in NANOLLAVA_BACKENDS:
            raise ValueError(
                f"{TORCH_BACKEND_ENV} must be one of {list(NANOLLAVA_BACKENDS)}"
            )
        raw_python = str(source.get(PYTHON_ENV, "")).strip()
        backend: SemanticBackend | BatchSemanticBackend | None = None
        if kind == TORCH_BACKEND_NAME:
            device, dtype = torch_request_from_env(source)
            return cls(
                snapshot,
                backend=TransformersVlmSemanticBackend(
                    str(source.get("FORECAST_NANOLLAVA_TORCH_MODEL", TORCH_MODEL_ID)).strip()
                    or TORCH_MODEL_ID,
                    str(source.get(TORCH_REVISION_ENV, "")).strip().lower(),
                    device=device,
                    dtype=dtype,
                    trust_remote_code=str(
                        source.get(TRUST_REMOTE_CODE_ENV, "false")
                    ).strip().lower()
                    in {"1", "true", "yes", "on"},
                ),
                frame_extractor=FfmpegKeyframeExtractor(binary=binary),
                hook_pass=hook_pass,
            )
        if raw_python:
            raw_timeout = str(source.get(SUBPROCESS_TIMEOUT_ENV, "300")).strip()
            configuration_error: str | None = None
            try:
                timeout_seconds = float(raw_timeout)
            except (TypeError, ValueError):
                timeout_seconds = 300.0
                configuration_error = (
                    f"{SUBPROCESS_TIMEOUT_ENV} must be a finite number from 10 to 1800."
                )
            if (
                not configuration_error
                and (
                    not math.isfinite(timeout_seconds)
                    or not 10 <= timeout_seconds <= 1800
                )
            ):
                configuration_error = (
                    f"{SUBPROCESS_TIMEOUT_ENV} must be a finite number from 10 to 1800."
                )
            backend = IsolatedMlxVlmBatchBackend(
                Path(raw_python),
                snapshot,
                timeout_seconds=timeout_seconds,
                configuration_error=configuration_error,
            )
        return cls(
            snapshot,
            backend=backend,
            frame_extractor=FfmpegKeyframeExtractor(binary=binary),
            hook_pass=hook_pass,
        )

    def availability(self) -> dict[str, Any]:
        if isinstance(self.backend, TransformersVlmSemanticBackend):
            return self._torch_availability(self.backend)
        snapshot_present = self.snapshot.is_dir() and (
            self.snapshot / "model.safetensors"
        ).is_file()
        isolated_reason = (
            self.backend.unavailable_reason()
            if isinstance(self.backend, IsolatedMlxVlmBatchBackend)
            else None
        )
        runtime_present = (
            isolated_reason is None
            if isinstance(self.backend, IsolatedMlxVlmBatchBackend)
            else self.backend is not None or MlxVlmSemanticBackend.installed()
        )
        reason = None
        if not snapshot_present:
            reason = "The exact pinned NanoLLaVA snapshot is not present locally."
        elif isolated_reason:
            reason = isolated_reason
        elif not runtime_present:
            reason = (
                "The pinned NanoLLaVA snapshot is cached, but the optional mlx_vlm "
                "runtime is not installed."
            )
        return {
            "configured": snapshot_present,
            "executionAvailable": snapshot_present and runtime_present,
            "reason": reason,
            "role": "nanollava-keyframe-semantic-fallback",
            "isVideoLlama": False,
            "usesAudio": False,
            "usesLearnedModel": True,
            "provenance": {
                "modelId": MODEL_ID,
                "modelRevision": MODEL_REVISION,
                "modelWeightsSha256": MODEL_WEIGHTS_SHA256,
                "license": MODEL_LICENSE,
                "adapterId": ADAPTER_ID,
                "codeRevision": WORKER_CODE_REVISION,
                "preprocessingId": PREPROCESSING_ID,
                "preprocessingSha256": PREPROCESSING_SHA256,
            },
        }

    def _torch_availability(
        self, backend: "TransformersVlmSemanticBackend"
    ) -> dict[str, Any]:
        """Readiness for the portable backend, with its own model identity.

        The MLX weight digest is deliberately absent: this is a different
        artifact, anchored by its pinned commit rather than by that hash.
        """

        reason = backend.unavailable_reason()
        return {
            "configured": bool(COMMIT_RE.fullmatch(backend.revision or "")),
            "executionAvailable": reason is None,
            "reason": reason,
            "role": "nanollava-keyframe-semantic-fallback",
            "isVideoLlama": False,
            "usesAudio": False,
            "usesLearnedModel": True,
            "provenance": {
                "modelId": backend.model_id,
                "modelRevision": backend.revision or None,
                "license": MODEL_LICENSE,
                "adapterId": TORCH_ADAPTER_ID,
                "codeRevision": WORKER_CODE_REVISION,
                "preprocessingId": PREPROCESSING_ID,
                "preprocessingSha256": PREPROCESSING_SHA256,
                "trustRemoteCode": backend.trust_remote_code,
            },
        }

    def _verify_snapshot(self) -> None:
        if isinstance(self.backend, TransformersVlmSemanticBackend):
            # A different artifact with a different integrity anchor: the pinned
            # commit, checked when the backend resolves its snapshot. Applying the
            # MLX weight digest here would always fail, and faking it would be worse.
            blocked = self.backend.unavailable_reason()
            if blocked:
                raise NanoLlavaUnavailable(blocked)
            return
        required = (
            self.snapshot / "model.safetensors",
            self.snapshot / "config.json",
            self.snapshot / "tokenizer.json",
        )
        if any(not path.is_file() for path in required):
            raise NanoLlavaUnavailable(
                "The exact pinned NanoLLaVA snapshot is incomplete."
            )
        try:
            digest = _sha256_file(required[0])
        except OSError as exc:
            raise NanoLlavaUnavailable(
                "The pinned NanoLLaVA weights could not be read."
            ) from exc
        if digest != MODEL_WEIGHTS_SHA256:
            raise NanoLlavaUnavailable(
                "The cached NanoLLaVA weights failed the pinned SHA-256 check."
            )

    def _describe_pass(
        self,
        backend: Any,
        frames: tuple[tuple[float, Path], ...],
        prompt: str,
    ) -> tuple[str, ...]:
        describe_many = getattr(backend, "describe_many", None)
        if callable(describe_many):
            raw_responses = tuple(describe_many(tuple(item[1] for item in frames), prompt))
            if len(raw_responses) != len(frames):
                raise NanoLlavaOutputError(
                    "The batch semantic backend returned the wrong response count."
                )
            return raw_responses
        describe = getattr(backend, "describe", None)
        if not callable(describe):
            raise NanoLlavaUnavailable(
                "The configured NanoLLaVA semantic backend is invalid."
            )
        return tuple(describe(image_path, prompt) for _, image_path in frames)

    def run(
        self,
        *,
        video_path: Path,
        input_sha256: str,
        duration_seconds: float,
        context: Mapping[str, Any],
    ) -> BranchOutput:
        del context
        if not video_path.is_file():
            raise NanoLlavaUnavailable("The input video is unavailable.")
        if not SHA256_RE.fullmatch(input_sha256):
            raise ValueError("NanoLLaVA input hash is invalid")
        try:
            actual_input_sha256 = _sha256_file(video_path)
        except OSError as exc:
            raise NanoLlavaUnavailable("The input video could not be read.") from exc
        if actual_input_sha256 != input_sha256:
            raise ValueError("NanoLLaVA input hash does not match the uploaded video")
        if not 10 <= float(duration_seconds) <= 60:
            raise ValueError("NanoLLaVA input duration must be from 10 to 60 seconds")
        self._verify_snapshot()
        backend = self.backend or MlxVlmSemanticBackend(self.snapshot)
        if isinstance(backend, IsolatedMlxVlmBatchBackend):
            backend._require_available()
        started_at = _utc_now()
        observations: list[dict[str, Any]] = []
        warnings: list[str] = [
            "NanoLLaVA is a still-keyframe semantic fallback, not VideoLLaMA 2.1; it does not consume audio or model continuous motion.",
            "Keyframe descriptions are bounded model-derived observations and may be incorrect; they are not verified ground truth or audience-outcome evidence.",
            "visibleText is null by contract because this NanoLLaVA fallback is not treated as an OCR provider; null is unavailable evidence, not observed absence.",
        ]
        field_coverage = {
            "scene": 0,
            "action": 0,
            "visible_text": 0,
            "shot": 0,
            "uncertainties": 0,
        }
        prompt = FRAME_PROMPT.format(boundary=SYSTEM_BOUNDARY).strip()
        frames: tuple[tuple[float, Path], ...] = ()
        with tempfile.TemporaryDirectory(prefix="forecast-nanollava-") as directory:
            root = Path(directory)
            # Each pass gets its own directory. The extractor also creates it,
            # but a caller-supplied extractor need not.
            (root / "clip").mkdir(parents=True, exist_ok=True)
            (root / "hook").mkdir(parents=True, exist_ok=True)
            frames = tuple(
                self.frame_extractor.extract(video_path, float(duration_seconds), root / "clip")
            )
            expected_times = deterministic_sample_times(float(duration_seconds))
            if (
                len(frames) != len(expected_times)
                or any(
                    timestamp != expected_times[index]
                    or not image_path.is_file()
                    or image_path.stat().st_size <= 0
                    for index, (timestamp, image_path) in enumerate(frames)
                )
            ):
                raise NanoLlavaUnavailable(
                    "The keyframe extractor violated the deterministic frame contract."
                )

            # The clip pass samples proportionally, so on a clip longer than
            # about 36 seconds none of its frames land inside the first three
            # seconds. The hook pass samples fixed absolute times so the opening
            # is described on every clip length.
            hook_frames: tuple[tuple[float, Path], ...] = ()
            if self.hook_pass:
                try:
                    hook_frames = tuple(
                        self.frame_extractor.extract(
                            video_path,
                            float(duration_seconds),
                            root / "hook",
                            sample_times=hook_frame_times(float(duration_seconds)),
                        )
                    )
                except TypeError:
                    warnings.append(
                        "The configured keyframe extractor cannot sample the hook window, "
                        "so opening keyframes were not described."
                    )
                except NanoLlavaUnavailable:
                    warnings.append(
                        "The hook keyframe pass failed; only proportionally sampled "
                        "keyframes were described."
                    )

            passes = (("clip", frames, tuple(item[0] for item in frames)),)
            if hook_frames:
                passes = passes + (
                    ("hook", hook_frames, tuple(item[0] for item in hook_frames)),
                )

            for pass_name, pass_frames, pass_times in passes:
                raw_responses = self._describe_pass(backend, pass_frames, prompt)
                for index, raw_response in enumerate(raw_responses):
                    try:
                        parsed = _strict_json_object(raw_response)
                    except NanoLlavaOutputError:
                        warnings.append(
                            f"The {pass_name} pass omitted keyframe {index + 1} because its "
                            "model response failed the strict semantic JSON contract."
                        )
                        continue
                    if pass_name == "hook":
                        start = round(float(pass_times[index]), 6)
                        end = round(
                            min(float(duration_seconds), start + HOOK_FRAME_SPAN_SECONDS), 6
                        )
                        if end <= start:
                            end = round(min(float(duration_seconds), start + 0.001), 6)
                    else:
                        start, end = _observation_bounds(
                            pass_times, index, float(duration_seconds)
                        )
                    encoded_observation = json.dumps(
                        parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    if len(encoded_observation) > 4096:
                        warnings.append(
                            f"The {pass_name} pass omitted keyframe {index + 1} because its "
                            "validated semantic JSON exceeded the observation limit."
                        )
                        continue
                    field_coverage["scene"] += 1
                    field_coverage["action"] += 1
                    field_coverage["shot"] += 1
                    if parsed["visibleText"] is not None:
                        field_coverage["visible_text"] += 1
                    if parsed["uncertainties"] is not None:
                        field_coverage["uncertainties"] += 1
                    observations.append(
                        {
                            "kind": "nanollava-keyframe-semantics",
                            "startTime": start,
                            "endTime": end,
                            "text": encoded_observation,
                            "labels": [
                                "model-derived",
                                "single-keyframe",
                                "nanollava-fallback",
                                f"pass:{pass_name}",
                            ],
                        }
                    )
            ran_hook_pass = bool(hook_frames)
        if not frames or not observations:
            raise NanoLlavaOutputError(
                "NanoLLaVA returned no contract-valid keyframe observations."
            )

        identity = WorkerIdentity(
            branch=self.branch,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            model_weights_sha256=MODEL_WEIGHTS_SHA256,
            adapter_id=ADAPTER_ID,
            code_revision=WORKER_CODE_REVISION,
            preprocessing_id=(
                TWO_PASS_PREPROCESSING_ID if ran_hook_pass else PREPROCESSING_ID
            ),
            preprocessing_sha256=(
                TWO_PASS_PREPROCESSING_SHA256 if ran_hook_pass else PREPROCESSING_SHA256
            ),
        )
        output = {
            "schemaVersion": WORKER_OUTPUT_SCHEMA_VERSION,
            "branch": self.branch,
            "inputSha256": input_sha256,
            "startedAt": started_at,
            "completedAt": _utc_now(),
            "evidenceKind": "learned-keyframe-semantics",
            "features": {
                "nanollava.keyframes_requested": float(len(frames)),
                "nanollava.keyframes_described": float(len(observations)),
                "nanollava.keyframe_coverage_fraction": len(observations) / len(frames),
                "nanollava.field_coverage.scene_fraction": field_coverage["scene"]
                / len(frames),
                "nanollava.field_coverage.action_fraction": field_coverage["action"]
                / len(frames),
                "nanollava.field_coverage.visible_text_fraction": field_coverage[
                    "visible_text"
                ]
                / len(frames),
                "nanollava.field_coverage.shot_fraction": field_coverage["shot"]
                / len(frames),
                "nanollava.field_coverage.uncertainties_fraction": field_coverage[
                    "uncertainties"
                ]
                / len(frames),
            },
            "observations": observations,
            "warnings": warnings,
            "provenance": identity.public_value(),
            "behavioralOutcome": False,
        }
        return validate_branch_output(
            output,
            expected_branch=self.branch,
            expected_input_sha256=input_sha256,
            identity=identity,
        )
