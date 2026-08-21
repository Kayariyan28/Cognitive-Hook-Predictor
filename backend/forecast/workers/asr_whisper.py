"""Transcript evidence from a pinned Whisper model.

A transcript records the words that were spoken and when. It records nothing
about who spoke, how they felt, or how anyone will respond. The branch reuses
the existing deterministic 16 kHz mono decode contract rather than adding a
second ffmpeg path, and it fails on its own: a missing model or an absent
runtime makes this one branch unavailable while the job still completes.

Two interchangeable runtimes sit behind one `Transcriber` protocol. `mlx-whisper`
stays the default because Apple silicon is the primary target; `transformers`
runs the same lane on CUDA, MPS or CPU so a Linux or Colab host is not simply
locked out. They are separate pinned artifacts with separate provenance — the
MLX repository holds MLX-quantised weights that torch cannot read, so the two
backends genuinely are different models and the manifest says which one ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import re
import shutil
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from .torch_runtime import runtime_request_from_env, unavailable_reason
from .measured_audio import (
    SAMPLE_RATE,
    FfmpegPcmDecoder,
    PcmDecoder,
)


SCHEMA_VERSION = "creator-forecast-asr/1"
BRANCH = "asr"
ADAPTER_ID = "mlx-whisper-transcript"
BACKEND_ENV = "INSIGHT_ASR_BACKEND"
MLX_BACKEND = "mlx-whisper"
TORCH_BACKEND = "transformers"
ASR_BACKENDS = (MLX_BACKEND, TORCH_BACKEND)
ADAPTER_IDS = {
    MLX_BACKEND: ADAPTER_ID,
    TORCH_BACKEND: "transformers-whisper-transcript",
}
BACKEND_MODULES = {MLX_BACKEND: "mlx_whisper", TORCH_BACKEND: "transformers"}
# What an operator would type to install it, which is not always the import name.
BACKEND_PACKAGES = {MLX_BACKEND: "mlx-whisper", TORCH_BACKEND: "transformers"}
EVIDENCE_KIND = "measured-speech-transcript"
# The decode contract is the measured-audio one, deliberately: one clip decodes
# once, the same way, for every branch that needs PCM.
PREPROCESSING_ID = "ffmpeg-f32le-mono-16khz/1"
DEFAULT_MODEL_ID = "mlx-community/whisper-large-v3-turbo"
# The torch runtime cannot load MLX-quantised weights, so the portable backend
# names the upstream repository instead. Neither default is usable until the
# operator pins a revision: an unpinned lane stays unavailable by design.
DEFAULT_TORCH_MODEL_ID = "openai/whisper-large-v3-turbo"
DEFAULT_MODEL_IDS = {
    MLX_BACKEND: DEFAULT_MODEL_ID,
    TORCH_BACKEND: DEFAULT_TORCH_MODEL_ID,
}

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LANGUAGE_RE = re.compile(r"^[a-z]{2,8}(?:-[a-z0-9]{2,8})?$", re.IGNORECASE)
MAXIMUM_SEGMENTS = 256
MAXIMUM_SEGMENT_CHARACTERS = 1024
TIMING_TOLERANCE_SECONDS = 0.5

WARNING = (
    "This is a machine transcript of spoken words with timings. It carries no speaker "
    "identity, no sentiment, and no claim about how an audience responds."
)


class AsrUnavailable(RuntimeError):
    """The pinned transcript model could not produce evidence for this clip."""


class Transcriber(Protocol):
    def transcribe(self, samples: np.ndarray) -> Mapping[str, Any]:
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


def _resolve_pinned_snapshot(model_id: str, revision: str) -> str:
    """Resolve the pinned revision from the local cache, without any network."""

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_id, revision=revision, local_files_only=True)


@dataclass(slots=True)
class MlxWhisperTranscriber:
    """Lazily imported mlx-whisper, pinned to an immutable revision."""

    model_id: str
    revision: str
    snapshot_resolver: Callable[[str, str], str] = _resolve_pinned_snapshot

    def transcribe(self, samples: np.ndarray) -> Mapping[str, Any]:
        snapshot = self.snapshot_resolver(self.model_id, self.revision)
        import mlx_whisper

        return mlx_whisper.transcribe(
            samples.astype(np.float32, copy=False),
            path_or_hf_repo=snapshot,
            word_timestamps=False,
            condition_on_previous_text=False,
        )


@dataclass(slots=True)
class TransformersWhisperTranscriber:
    """Whisper through torch, pinned to an immutable revision.

    Returns the same `{language, segments}` shape the MLX runtime does, so the
    branch's validator and output schema are shared rather than duplicated. The
    pipeline's final chunk can carry an open-ended timestamp; that is closed
    against the clip duration here, because a `None` end is a decoding detail
    and not something the evidence schema should learn about.
    """

    model_id: str
    revision: str
    device: str = "auto"
    dtype: str = "auto"
    snapshot_resolver: Callable[[str, str], str] = _resolve_pinned_snapshot
    pipeline_factory: Callable[..., Any] | None = None
    runtime: Any = None

    def transcribe(self, samples: np.ndarray) -> Mapping[str, Any]:
        snapshot = self.snapshot_resolver(self.model_id, self.revision)
        recognise = self.pipeline_factory or self._build_pipeline
        pipe = recognise(snapshot)
        audio = samples.astype(np.float32, copy=False)
        raw = pipe(
            {"raw": audio, "sampling_rate": SAMPLE_RATE},
            return_timestamps=True,
            chunk_length_s=30,
        )
        return self._normalised(raw, duration=len(audio) / float(SAMPLE_RATE))

    def _build_pipeline(self, snapshot: str) -> Any:
        from transformers import pipeline

        from .torch_runtime import resolve_runtime, torch_dtype

        import torch

        runtime = resolve_runtime(
            requested_device=self.device, requested_dtype=self.dtype, torch_module=torch
        )
        self.runtime = runtime
        return pipeline(
            "automatic-speech-recognition",
            model=snapshot,
            device=runtime.device,
            torch_dtype=torch_dtype(torch, runtime.dtype),
        )

    @staticmethod
    def _normalised(raw: Any, *, duration: float) -> Mapping[str, Any]:
        if not isinstance(raw, Mapping):
            raise AsrUnavailable("The transcriber returned no transcript object.")
        chunks = raw.get("chunks")
        if not isinstance(chunks, (list, tuple)):
            raise AsrUnavailable("The transcriber returned no segment list.")
        segments: list[dict[str, Any]] = []
        language: str | None = None
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                raise AsrUnavailable("The transcriber returned a malformed segment.")
            stamp = chunk.get("timestamp")
            if not isinstance(stamp, (list, tuple)) or len(stamp) != 2:
                raise AsrUnavailable("The transcriber returned a malformed segment time.")
            start, end = stamp
            if start is None:
                raise AsrUnavailable("The transcriber returned a segment with no start.")
            # An open final chunk closes at the clip's end, never past it.
            end = duration if end is None else end
            if isinstance(chunk.get("language"), str) and language is None:
                language = chunk["language"]
            segments.append(
                {"start": float(start), "end": float(end), "text": chunk.get("text")}
            )
        return {"language": language, "segments": segments}


def _validated_transcript(
    raw: Any, duration_seconds: float
) -> tuple[str | None, tuple[Mapping[str, Any], ...]]:
    """Read only language, timings, and text; ignore everything else offered."""

    if not isinstance(raw, Mapping):
        raise AsrUnavailable("The transcriber returned no transcript object.")
    language = raw.get("language")
    if language is not None and (
        not isinstance(language, str) or not LANGUAGE_RE.fullmatch(language.strip())
    ):
        raise AsrUnavailable("The transcriber returned an unsupported language tag.")
    normalized_language = language.strip().lower() if isinstance(language, str) else None

    raw_segments = raw.get("segments")
    if not isinstance(raw_segments, (list, tuple)):
        raise AsrUnavailable("The transcriber returned no segment list.")
    if len(raw_segments) > MAXIMUM_SEGMENTS:
        raise AsrUnavailable("The transcriber returned more segments than the branch accepts.")

    limit = float(duration_seconds) + TIMING_TOLERANCE_SECONDS
    segments: list[Mapping[str, Any]] = []
    previous_end = 0.0
    for entry in raw_segments:
        if not isinstance(entry, Mapping):
            raise AsrUnavailable("The transcriber returned a malformed segment.")
        start = entry.get("start")
        end = entry.get("end")
        text = entry.get("text")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
        ):
            raise AsrUnavailable("The transcriber returned a non-finite segment time.")
        start = float(start)
        end = float(end)
        if start < 0.0 or end <= start or end > limit or start < previous_end - TIMING_TOLERANCE_SECONDS:
            raise AsrUnavailable("The transcriber returned segment timings out of order or range.")
        if not isinstance(text, str) or not text.strip():
            continue
        if len(text) > MAXIMUM_SEGMENT_CHARACTERS:
            raise AsrUnavailable("The transcriber returned a segment longer than the branch accepts.")
        previous_end = end
        segments.append(
            MappingProxyType(
                {
                    "kind": "asr-transcript-segment",
                    "startTime": round(start, 6),
                    "endTime": round(min(end, limit), 6),
                    "text": text.strip(),
                    "labels": (
                        "asr",
                        f"language:{normalized_language}" if normalized_language else "language:unknown",
                        "no-speaker-identity",
                        "no-sentiment",
                    ),
                }
            )
        )
    if not segments:
        raise AsrUnavailable("The transcriber produced no spoken-word segment for this clip.")
    return normalized_language, tuple(segments)


@dataclass(frozen=True, slots=True)
class AsrOutput:
    input_sha256: str
    started_at: str
    completed_at: str
    features: Mapping[str, float]
    observations: tuple[Mapping[str, Any], ...]
    warnings: tuple[str, ...]
    model_id: str
    model_revision: str
    evidence_kind: str = EVIDENCE_KIND
    adapter_id: str = ADAPTER_ID
    backend: str = MLX_BACKEND
    runtime: Mapping[str, Any] | None = None

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
                "adapterId": self.adapter_id,
                "preprocessingId": PREPROCESSING_ID,
                "sampleRateHz": SAMPLE_RATE,
                "modelId": self.model_id,
                "modelRevision": self.model_revision,
                "usesLearnedModel": True,
                **(dict(self.runtime) if self.runtime else {}),
            },
            "behavioralOutcome": False,
        }


def _features(
    observations: tuple[Mapping[str, Any], ...], duration_seconds: float
) -> Mapping[str, float]:
    spoken = sum(item["endTime"] - item["startTime"] for item in observations)
    values = {
        "asr.segment_count": float(len(observations)),
        "asr.speech_seconds": float(round(spoken, 6)),
        "asr.speech_fraction": float(
            round(spoken / duration_seconds, 6) if duration_seconds > 0 else 0.0
        ),
        "asr.first_segment_start_seconds": float(observations[0]["startTime"]),
        "asr.character_count": float(sum(len(item["text"]) for item in observations)),
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise AsrUnavailable(f"transcript feature {name} is not finite")
    return MappingProxyType(values)


class AsrWhisperAdapter:
    """A transcript evidence branch that fails alone and claims nothing extra."""

    branch = BRANCH

    def __init__(
        self,
        transcriber: Transcriber | None = None,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        model_revision: str = "",
        decoder: PcmDecoder | None = None,
        binary: str = "ffmpeg",
        probe_binary: str = "ffprobe",
        module_probe: Callable[[str], bool] | None = None,
        snapshot_resolver: Callable[[str, str], str] | None = None,
        backend: str = MLX_BACKEND,
        device: str = "auto",
        dtype: str = "auto",
    ) -> None:
        if backend not in ASR_BACKENDS:
            raise ValueError(f"{BACKEND_ENV} must be one of {list(ASR_BACKENDS)}")
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self.model_revision = model_revision
        self.transcriber = transcriber
        self.decoder = decoder or FfmpegPcmDecoder(binary=binary, probe_binary=probe_binary)
        self.binary = binary
        self.probe_binary = probe_binary
        self._module_probe = module_probe or _module_available
        self._snapshot_resolver = snapshot_resolver or _resolve_pinned_snapshot

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "AsrWhisperAdapter":
        source = os.environ if environ is None else environ
        backend = str(source.get(BACKEND_ENV, MLX_BACKEND)).strip().lower() or MLX_BACKEND
        if backend not in ASR_BACKENDS:
            raise ValueError(f"{BACKEND_ENV} must be one of {list(ASR_BACKENDS)}")
        default_model = DEFAULT_MODEL_IDS[backend]
        model_id = str(source.get("INSIGHT_ASR_MODEL", default_model)).strip() or default_model
        device, dtype = runtime_request_from_env(source)
        revision = str(source.get("INSIGHT_ASR_MODEL_REVISION", "")).strip().lower()
        binary = str(source.get("FORECAST_AUDIO_FFMPEG_BINARY", "ffmpeg")).strip() or "ffmpeg"
        probe_binary = str(source.get("FORECAST_AUDIO_FFPROBE_BINARY", "ffprobe")).strip() or "ffprobe"
        return cls(
            model_id=model_id,
            model_revision=revision,
            binary=binary,
            probe_binary=probe_binary,
            backend=backend,
            device=device,
            dtype=dtype,
        )

    @property
    def revision_is_pinned(self) -> bool:
        return bool(COMMIT_RE.fullmatch(self.model_revision))

    def availability(self) -> dict[str, Any]:
        provenance = {
            "adapterId": ADAPTER_IDS[self.backend],
            "preprocessingId": PREPROCESSING_ID,
            "sampleRateHz": SAMPLE_RATE,
            "modelId": self.model_id,
            "modelRevision": self.model_revision or None,
        }
        common = {
            "role": "optional-measured-speech-transcript",
            "usesLearnedModel": True,
            "isSpeechTranscript": True,
            "isBehavioralModel": False,
            "provenance": provenance,
        }
        if not self.revision_is_pinned:
            return {
                "configured": False,
                "executionAvailable": False,
                "reason": (
                    "INSIGHT_ASR_MODEL_REVISION must be a full 40-character commit SHA; "
                    "mutable names such as 'main' are rejected."
                ),
                **common,
            }
        if self.transcriber is None:
            module = BACKEND_MODULES[self.backend]
            if not self._module_probe(module):
                return {
                    "configured": True,
                    "executionAvailable": False,
                    "reason": (
                        f"{BACKEND_PACKAGES[self.backend]} is not installed in this "
                        f"backend environment."
                    ),
                    **common,
                }
            if self.backend == TORCH_BACKEND:
                blocked = unavailable_reason(
                    requested_device=self.device, requested_dtype=self.dtype
                )
                if blocked:
                    return {
                        "configured": True,
                        "executionAvailable": False,
                        "reason": blocked,
                        **common,
                    }
                # Resolve the pin from the local cache, as the other portable
                # lanes do. Without this the status reads ready and the branch
                # then fails mid-job, which is a worse way to learn the same
                # thing. This reads the cache only; it never downloads.
                try:
                    self._snapshot_resolver(self.model_id, self.model_revision)
                except Exception:
                    return {
                        "configured": True,
                        "executionAvailable": False,
                        "reason": (
                            "The pinned transcript revision is not present in the "
                            "local snapshot cache. Fetch it once before running."
                        ),
                        **common,
                    }
        decoder_ready = not isinstance(self.decoder, FfmpegPcmDecoder) or (
            shutil.which(self.binary) is not None and shutil.which(self.probe_binary) is not None
        )
        if not decoder_ready:
            return {
                "configured": True,
                "executionAvailable": False,
                "reason": "The configured ffmpeg executable is unavailable.",
                **common,
            }
        return {
            "configured": True,
            "executionAvailable": True,
            "reason": None,
            **common,
        }

    def run(
        self,
        *,
        video_path: Path,
        input_sha256: str,
        duration_seconds: float,
        context: Mapping[str, Any],
    ) -> AsrOutput:
        del context
        if not video_path.is_file():
            raise AsrUnavailable("The input video is unavailable.")
        if not SHA256_RE.fullmatch(input_sha256):
            raise ValueError("transcript input hash is invalid")
        if _sha256_file(video_path) != input_sha256:
            raise ValueError("transcript input hash does not match the uploaded video")
        if not 10 <= float(duration_seconds) <= 60:
            raise ValueError("transcript input duration must be from 10 to 60 seconds")
        if not self.revision_is_pinned:
            raise AsrUnavailable("The transcript model revision is not pinned.")

        started_at = _utc_now()
        samples = self.decoder.decode(video_path)
        if not isinstance(samples, np.ndarray) or samples.ndim != 1:
            raise AsrUnavailable("The PCM decoder returned an invalid signal.")
        samples = samples[: math.ceil(float(duration_seconds) * SAMPLE_RATE)]
        transcriber = self.transcriber or self._build_transcriber()
        try:
            raw = transcriber.transcribe(samples)
        except AsrUnavailable:
            raise
        except Exception as exc:
            raise AsrUnavailable(
                f"The pinned transcript model failed: {type(exc).__name__}"
            ) from exc
        language, observations = _validated_transcript(raw, float(duration_seconds))
        runtime = getattr(transcriber, "runtime", None)
        return AsrOutput(
            input_sha256=input_sha256,
            started_at=started_at,
            completed_at=_utc_now(),
            features=_features(observations, float(duration_seconds)),
            observations=observations,
            warnings=(WARNING,),
            model_id=self.model_id,
            model_revision=self.model_revision,
            adapter_id=ADAPTER_IDS[self.backend],
            backend=self.backend,
            runtime=runtime.provenance() if runtime is not None else None,
        )

    def _build_transcriber(self) -> Transcriber:
        if self.backend == TORCH_BACKEND:
            return TransformersWhisperTranscriber(
                model_id=self.model_id,
                revision=self.model_revision,
                device=self.device,
                dtype=self.dtype,
            )
        return MlxWhisperTranscriber(
            model_id=self.model_id, revision=self.model_revision
        )


def transcript_document(output: AsrOutput) -> dict[str, Any]:
    """The branch's declared output schema: `{language, segments}`, nothing else."""

    language = None
    for observation in output.observations:
        for label in observation["labels"]:
            if label.startswith("language:"):
                candidate = label.split(":", 1)[1]
                language = None if candidate == "unknown" else candidate
                break
        break
    return {
        "language": language,
        "segments": [
            {
                "startSec": item["startTime"],
                "endSec": item["endTime"],
                "text": item["text"],
            }
            for item in output.observations
        ],
    }
