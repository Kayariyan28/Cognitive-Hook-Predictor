"""On-screen-text evidence over the existing six deterministic keyframes.

Text recognition reports glyphs and where they sat on the frame. It reports no
meaning, no emphasis, and nothing about a reader. Apple Vision output varies by
operating-system version, so the branch records the macOS version and the engine
that actually ran alongside its other provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from .nanollava import (
    HOOK_FRAME_TIMES,
    KEYFRAME_COUNT,
    FfmpegKeyframeExtractor,
    NanoLlavaUnavailable,
    deterministic_sample_times,
    hook_frame_times,
)


SCHEMA_VERSION = "creator-forecast-ocr/1"
BRANCH = "ocr"
ADAPTER_ID = "on-screen-text-recognition"
EVIDENCE_KIND = "measured-on-screen-text"
# Two passes. The clip pass is NanoLLaVA's contract, deliberately: the same six
# frames, at the same times, at the same size, are read by both branches. The
# hook pass exists because that contract samples proportionally, so past roughly
# 36 seconds none of its frames land inside the first three seconds and on-screen
# hook text becomes invisible. Hook frames are sampled at fixed absolute times.
CLIP_PREPROCESSING_ID = "ffmpeg-keyframes-6x384-png-center-sampled/1"
HOOK_PREPROCESSING_ID = "ffmpeg-hookframes-6x384-png-absolute/1"
PREPROCESSING_ID = f"{CLIP_PREPROCESSING_ID}+{HOOK_PREPROCESSING_ID}"
HOOK_FRAME_SPAN_SECONDS = 0.4
ENGINES = ("ocrmac", "pytesseract")

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAXIMUM_BLOCKS_PER_FRAME = 32
MAXIMUM_BLOCK_CHARACTERS = 256

WARNING = (
    "This is on-screen text recognition. It records glyphs and their positions, not "
    "meaning, emphasis, or any claim about what a viewer reads or does."
)


class OcrUnavailable(RuntimeError):
    """No configured recognition engine could read this clip's keyframes."""


class TextRecognizer(Protocol):
    engine: str

    def recognize(self, image_path: Path) -> list[Mapping[str, Any]]:
        ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def macos_product_version(runner: Callable[..., Any] = subprocess.run) -> str | None:
    """Record the OS version, because Apple Vision output varies with it."""

    try:
        completed = runner(
            ["sw_vers", "-productVersion"],
            check=False,
            capture_output=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if getattr(completed, "returncode", 1) != 0:
        return None
    stdout = getattr(completed, "stdout", b"")
    text = stdout.decode("utf-8", "replace") if isinstance(stdout, bytes) else str(stdout)
    version = text.strip()
    return version if re.fullmatch(r"\d+(?:\.\d+){0,2}", version) else None


@dataclass(slots=True)
class OcrmacRecognizer:
    """Apple Vision through ocrmac, imported lazily."""

    engine: str = "ocrmac"

    def recognize(self, image_path: Path) -> list[Mapping[str, Any]]:
        from ocrmac import ocrmac

        annotations = ocrmac.OCR(str(image_path)).recognize()
        blocks: list[Mapping[str, Any]] = []
        for entry in annotations or ():
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                continue
            text, confidence, bbox = entry
            blocks.append({"text": text, "confidence": confidence, "bbox": bbox})
        return blocks


@dataclass(slots=True)
class PytesseractRecognizer:
    """The optional fallback; it records that it, not Vision, produced the text."""

    engine: str = "pytesseract"

    def recognize(self, image_path: Path) -> list[Mapping[str, Any]]:
        import pytesseract
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
            data = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT
            )
        blocks: list[Mapping[str, Any]] = []
        for index, text in enumerate(data.get("text", [])):
            if not isinstance(text, str) or not text.strip():
                continue
            try:
                confidence = float(data["conf"][index])
                left = float(data["left"][index])
                top = float(data["top"][index])
                block_width = float(data["width"][index])
                block_height = float(data["height"][index])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if confidence < 0:
                continue
            blocks.append(
                {
                    "text": text,
                    "confidence": confidence / 100.0,
                    "bbox": [
                        left / width,
                        top / height,
                        block_width / width,
                        block_height / height,
                    ],
                }
            )
        return blocks


def _validated_blocks(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise OcrUnavailable("The recognition engine returned a malformed block list.")
    if len(raw) > MAXIMUM_BLOCKS_PER_FRAME:
        raise OcrUnavailable("The recognition engine returned more blocks than the branch accepts.")
    blocks: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise OcrUnavailable("The recognition engine returned a malformed block.")
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        if len(text) > MAXIMUM_BLOCK_CHARACTERS:
            raise OcrUnavailable("The recognition engine returned a block longer than the branch accepts.")
        confidence = entry.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise OcrUnavailable("The recognition engine returned an out-of-range confidence.")
        raw_bbox = entry.get("bbox")
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            raise OcrUnavailable("The recognition engine returned a malformed bounding box.")
        bbox: list[float] = []
        for value in raw_bbox:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not -1.0 <= float(value) <= 2.0
            ):
                raise OcrUnavailable("The recognition engine returned an out-of-range bounding box.")
            bbox.append(round(float(value), 6))
        blocks.append(
            {"text": text.strip(), "confidence": round(float(confidence), 6), "bbox": bbox}
        )
    return blocks


@dataclass(frozen=True, slots=True)
class OcrOutput:
    input_sha256: str
    started_at: str
    completed_at: str
    features: Mapping[str, float]
    observations: tuple[Mapping[str, Any], ...]
    warnings: tuple[str, ...]
    engine: str
    macos_version: str | None
    evidence_kind: str = EVIDENCE_KIND

    def public_value(self) -> dict[str, Any]:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "branch": BRANCH,
            "inputSha256": self.input_sha256,
            "startedAt": self.started_at,
            "completedAt": self.completed_at,
            "evidenceKind": self.evidence_kind,
            "features": dict(self.features),
            "observations": [
                {
                    "kind": item["kind"],
                    "startTime": item["startTime"],
                    "endTime": item["endTime"],
                    "text": item["text"],
                    "labels": list(item["labels"]),
                }
                for item in self.observations
            ],
            "warnings": list(self.warnings),
            "provenance": {
                "adapterId": ADAPTER_ID,
                "preprocessingId": PREPROCESSING_ID,
                "engine": self.engine,
                "macOSVersion": self.macos_version,
                "usesLearnedModel": True,
            },
            "behavioralOutcome": False,
        }


def _observation_bounds(
    sample_times: tuple[float, ...], index: int, duration_seconds: float
) -> tuple[float, float]:
    start = sample_times[index]
    end = sample_times[index + 1] if index + 1 < len(sample_times) else duration_seconds
    if end <= start:
        end = min(duration_seconds, start + 0.001)
    return round(start, 6), round(end, 6)


def _frame_observation(
    index: int,
    start: float,
    end: float,
    blocks: list[dict[str, Any]],
    engine: str,
    pass_name: str,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "kind": "ocr-frame-text-blocks",
            "startTime": start,
            "endTime": end,
            "text": json.dumps(blocks, ensure_ascii=False, separators=(",", ":")),
            "labels": (
                "ocr",
                f"frame:{index}",
                f"engine:{engine}",
                f"pass:{pass_name}",
                "not-semantic",
            ),
        }
    )


class OcrScreenTextAdapter:
    """An on-screen-text branch that records which engine actually ran."""

    branch = BRANCH

    def __init__(
        self,
        recognizer: TextRecognizer | None = None,
        *,
        engine: str = "ocrmac",
        fallback_recognizer: TextRecognizer | None = None,
        extractor: Any = None,
        binary: str = "ffmpeg",
        module_probe: Callable[[str], bool] | None = None,
        version_probe: Callable[[], str | None] = macos_product_version,
    ) -> None:
        if engine not in ENGINES:
            raise ValueError(f"OCR engine must be one of {sorted(ENGINES)}")
        self.engine = engine
        self.recognizer = recognizer
        self.fallback_recognizer = fallback_recognizer
        self.extractor = extractor or FfmpegKeyframeExtractor(binary=binary)
        self.binary = binary
        self._module_probe = module_probe or _module_available
        self._version_probe = version_probe

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "OcrScreenTextAdapter":
        source = os.environ if environ is None else environ
        engine = str(source.get("INSIGHT_OCR_ENGINE", "ocrmac")).strip().lower() or "ocrmac"
        if engine not in ENGINES:
            engine = "ocrmac"
        binary = str(source.get("FORECAST_NANOLLAVA_FFMPEG_BINARY", "ffmpeg")).strip() or "ffmpeg"
        return cls(engine=engine, binary=binary)

    # -- readiness ---------------------------------------------------------

    def _engine_available(self, engine: str) -> bool:
        if engine == "ocrmac":
            return self._module_probe("ocrmac")
        return self._module_probe("pytesseract") and shutil.which("tesseract") is not None

    def _selected_engine(self) -> str | None:
        if self.recognizer is not None:
            return self.recognizer.engine
        if self._engine_available(self.engine):
            return self.engine
        if self.fallback_recognizer is not None:
            return self.fallback_recognizer.engine
        if self.engine == "ocrmac" and self._engine_available("pytesseract"):
            return "pytesseract"
        return None

    def availability(self) -> dict[str, Any]:
        engine = self._selected_engine()
        provenance = {
            "adapterId": ADAPTER_ID,
            "preprocessingId": PREPROCESSING_ID,
            "engine": engine,
            "macOSVersion": self._version_probe(),
        }
        common = {
            "role": "optional-measured-on-screen-text",
            "usesLearnedModel": True,
            "isBehavioralModel": False,
            "provenance": provenance,
        }
        if engine is None:
            return {
                "configured": False,
                "executionAvailable": False,
                "reason": (
                    "Neither ocrmac (Apple Vision) nor a pytesseract installation with the "
                    "tesseract binary is available in this backend environment."
                ),
                **common,
            }
        if not isinstance(self.extractor, FfmpegKeyframeExtractor) or shutil.which(self.binary):
            return {"configured": True, "executionAvailable": True, "reason": None, **common}
        return {
            "configured": True,
            "executionAvailable": False,
            "reason": "The configured ffmpeg executable is unavailable.",
            **common,
        }

    # -- execution ---------------------------------------------------------

    def _recognizer_for(self, engine: str) -> TextRecognizer:
        if self.recognizer is not None and self.recognizer.engine == engine:
            return self.recognizer
        if self.fallback_recognizer is not None and self.fallback_recognizer.engine == engine:
            return self.fallback_recognizer
        return OcrmacRecognizer() if engine == "ocrmac" else PytesseractRecognizer()

    def run(
        self,
        *,
        video_path: Path,
        input_sha256: str,
        duration_seconds: float,
        context: Mapping[str, Any],
    ) -> OcrOutput:
        del context
        if not video_path.is_file():
            raise OcrUnavailable("The input video is unavailable.")
        if not SHA256_RE.fullmatch(input_sha256):
            raise ValueError("on-screen-text input hash is invalid")
        if _sha256_file(video_path) != input_sha256:
            raise ValueError("on-screen-text input hash does not match the uploaded video")
        if not 10 <= float(duration_seconds) <= 60:
            raise ValueError("on-screen-text input duration must be from 10 to 60 seconds")
        engine = self._selected_engine()
        if engine is None:
            raise OcrUnavailable("No configured text-recognition engine is available.")

        started_at = _utc_now()
        duration = float(duration_seconds)
        clip_times = deterministic_sample_times(duration, KEYFRAME_COUNT)
        hook_times = hook_frame_times(duration)
        recognizer = self._recognizer_for(engine)
        observations: list[Mapping[str, Any]] = []
        block_count = 0
        frames_with_text = 0
        hook_blocks = 0
        hook_frames_with_text = 0

        def read(image_path: Path) -> list[dict[str, Any]]:
            try:
                return _validated_blocks(recognizer.recognize(image_path))
            except OcrUnavailable:
                raise
            except Exception as exc:
                raise OcrUnavailable(
                    f"The {engine} recognition engine failed: {type(exc).__name__}"
                ) from exc

        with tempfile.TemporaryDirectory(prefix="signalframe-ocr-") as workspace:
            root = Path(workspace)
            (root / "clip").mkdir(parents=True, exist_ok=True)
            (root / "hook").mkdir(parents=True, exist_ok=True)
            try:
                clip_frames = self.extractor.extract(video_path, duration, root / "clip")
                hook_frames = self.extractor.extract(
                    video_path, duration, root / "hook", sample_times=hook_times
                )
            except NanoLlavaUnavailable as exc:
                raise OcrUnavailable(
                    "Deterministic keyframe extraction is unavailable for this clip."
                ) from exc
            except TypeError as exc:  # an extractor without the two-pass contract
                raise OcrUnavailable(
                    "The configured keyframe extractor cannot sample the hook window."
                ) from exc
            if len(clip_frames) != KEYFRAME_COUNT:
                raise OcrUnavailable("The keyframe contract requires exactly six frames.")
            if not hook_frames:
                raise OcrUnavailable("The hook pass produced no frame.")

            for index, (_, image_path) in enumerate(clip_frames):
                blocks = read(image_path)
                start, end = _observation_bounds(clip_times, index, duration)
                block_count += len(blocks)
                if blocks:
                    frames_with_text += 1
                observations.append(
                    _frame_observation(index, start, end, blocks, engine, "clip")
                )

            for offset, (sample_time, image_path) in enumerate(hook_frames):
                blocks = read(image_path)
                index = len(clip_frames) + offset
                start = round(float(sample_time), 6)
                end = round(min(duration, start + HOOK_FRAME_SPAN_SECONDS), 6)
                if end <= start:
                    end = round(min(duration, start + 0.001), 6)
                block_count += len(blocks)
                hook_blocks += len(blocks)
                if blocks:
                    frames_with_text += 1
                    hook_frames_with_text += 1
                observations.append(
                    _frame_observation(index, start, end, blocks, engine, "hook")
                )

        features = MappingProxyType(
            {
                "ocr.frames_read": float(len(observations)),
                "ocr.frames_with_text": float(frames_with_text),
                "ocr.block_count": float(block_count),
                "ocr.hook_frames_read": float(len(hook_times)),
                "ocr.hook_frames_with_text": float(hook_frames_with_text),
                "ocr.hook_block_count": float(hook_blocks),
            }
        )
        return OcrOutput(
            input_sha256=input_sha256,
            started_at=started_at,
            completed_at=_utc_now(),
            features=features,
            observations=tuple(observations),
            warnings=(WARNING,),
            engine=engine,
            macos_version=self._version_probe(),
        )


def screen_text_document(output: OcrOutput) -> dict[str, Any]:
    """The branch's declared output schema: `{frames: [{frameIndex, blocks}]}`."""

    frames: list[dict[str, Any]] = []
    for index, observation in enumerate(output.observations):
        frames.append({"frameIndex": index, "blocks": json.loads(observation["text"])})
    return {"frames": frames}
