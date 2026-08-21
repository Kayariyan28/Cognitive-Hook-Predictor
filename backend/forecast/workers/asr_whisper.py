"""Transcript evidence from a pinned mlx-whisper model.

A transcript records the words that were spoken and when. It records nothing
about who spoke, how they felt, or how anyone will respond. The branch reuses
the existing deterministic 16 kHz mono decode contract rather than adding a
second ffmpeg path, and it fails on its own: a missing model or an absent MLX
runtime makes this one branch unavailable while the job still completes.
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

from .measured_audio import (
    SAMPLE_RATE,
    FfmpegPcmDecoder,
    PcmDecoder,
)


SCHEMA_VERSION = "creator-forecast-asr/1"
BRANCH = "asr"
ADAPTER_ID = "mlx-whisper-transcript"
EVIDENCE_KIND = "measured-speech-transcript"
# The decode contract is the measured-audio one, deliberately: one clip decodes
# once, the same way, for every branch that needs PCM.
PREPROCESSING_ID = "ffmpeg-f32le-mono-16khz/1"
DEFAULT_MODEL_ID = "mlx-community/whisper-large-v3-turbo"

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
                "sampleRateHz": SAMPLE_RATE,
                "modelId": self.model_id,
                "modelRevision": self.model_revision,
                "usesLearnedModel": True,
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
    ) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.transcriber = transcriber
        self.decoder = decoder or FfmpegPcmDecoder(binary=binary, probe_binary=probe_binary)
        self.binary = binary
        self.probe_binary = probe_binary
        self._module_probe = module_probe or _module_available

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "AsrWhisperAdapter":
        source = os.environ if environ is None else environ
        model_id = str(source.get("INSIGHT_ASR_MODEL", DEFAULT_MODEL_ID)).strip() or DEFAULT_MODEL_ID
        revision = str(source.get("INSIGHT_ASR_MODEL_REVISION", "")).strip().lower()
        binary = str(source.get("FORECAST_AUDIO_FFMPEG_BINARY", "ffmpeg")).strip() or "ffmpeg"
        probe_binary = str(source.get("FORECAST_AUDIO_FFPROBE_BINARY", "ffprobe")).strip() or "ffprobe"
        return cls(
            model_id=model_id,
            model_revision=revision,
            binary=binary,
            probe_binary=probe_binary,
        )

    @property
    def revision_is_pinned(self) -> bool:
        return bool(COMMIT_RE.fullmatch(self.model_revision))

    def availability(self) -> dict[str, Any]:
        provenance = {
            "adapterId": ADAPTER_ID,
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
        if self.transcriber is None and not self._module_probe("mlx_whisper"):
            return {
                "configured": True,
                "executionAvailable": False,
                "reason": "mlx-whisper is not installed in this backend environment.",
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
        transcriber = self.transcriber or MlxWhisperTranscriber(
            model_id=self.model_id, revision=self.model_revision
        )
        try:
            raw = transcriber.transcribe(samples)
        except AsrUnavailable:
            raise
        except Exception as exc:
            raise AsrUnavailable(
                f"The pinned transcript model failed: {type(exc).__name__}"
            ) from exc
        language, observations = _validated_transcript(raw, float(duration_seconds))
        return AsrOutput(
            input_sha256=input_sha256,
            started_at=started_at,
            completed_at=_utc_now(),
            features=_features(observations, float(duration_seconds)),
            observations=observations,
            warnings=(WARNING,),
            model_id=self.model_id,
            model_revision=self.model_revision,
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
