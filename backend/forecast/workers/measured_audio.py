"""Server-measured audio descriptors with no learned or behavioral claims."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import numpy as np


SCHEMA_VERSION = "creator-forecast-measured-audio/1"
PREPROCESSING_ID = "ffmpeg-f32le-mono-16khz-stft1024-hop512/1"
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 1024
HOP_SAMPLES = 512
SILENCE_RMS_THRESHOLD = 0.01
MAX_DECODE_SECONDS = 60.25
MAX_PCM_BYTES = int(MAX_DECODE_SECONDS * SAMPLE_RATE * 4) + 4
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_FEATURE_FRAGMENTS = (
    "virality",
    "retention",
    "completion",
    "rewatch",
    "engagement",
    "attention",
    "brain",
    "cortex",
    "tribe",
)


class MeasuredAudioUnavailable(RuntimeError):
    """ffmpeg could not produce bounded PCM evidence for this clip."""


class PcmDecoder(Protocol):
    def decode(self, video_path: Path) -> np.ndarray:
        ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class FfmpegPcmDecoder:
    binary: str = "ffmpeg"
    probe_binary: str = "ffprobe"
    timeout_seconds: float = 45.0

    def decode(self, video_path: Path) -> np.ndarray:
        probe_command = [
            self.probe_binary,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(video_path),
        ]
        try:
            probe = subprocess.run(
                probe_command,
                check=False,
                capture_output=True,
                timeout=min(self.timeout_seconds, 15.0),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MeasuredAudioUnavailable(
                "The configured ffprobe audio-stream check is unavailable."
            ) from exc
        if probe.returncode != 0:
            raise MeasuredAudioUnavailable(
                "ffprobe could not inspect the clip's audio streams."
            )
        if not probe.stdout.strip():
            return np.array([], dtype=np.float64)

        command = [
            self.binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(video_path),
            "-map",
            "0:a:0?",
            "-vn",
            "-t",
            f"{MAX_DECODE_SECONDS:.2f}",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "f32le",
            "pipe:1",
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MeasuredAudioUnavailable(
                "The configured ffmpeg PCM decoder is unavailable."
            ) from exc
        if completed.returncode != 0:
            raise MeasuredAudioUnavailable(
                "ffmpeg could not decode the clip's audio track."
            )
        if len(completed.stdout) > MAX_PCM_BYTES or len(completed.stdout) % 4:
            raise MeasuredAudioUnavailable(
                "ffmpeg returned an invalid or oversized PCM stream."
            )
        samples = np.frombuffer(completed.stdout, dtype="<f4").astype(np.float64)
        if samples.size and not np.isfinite(samples).all():
            raise MeasuredAudioUnavailable("Decoded PCM contains non-finite samples.")
        return samples


@dataclass(frozen=True, slots=True)
class MeasuredAudioOutput:
    input_sha256: str
    started_at: str
    completed_at: str
    features: Mapping[str, float]
    observations: tuple[Mapping[str, Any], ...]
    warnings: tuple[str, ...]
    evidence_kind: str = "measured-audio-descriptors"

    def public_value(self) -> dict[str, Any]:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "branch": "measuredAudio",
            "inputSha256": self.input_sha256,
            "startedAt": self.started_at,
            "completedAt": self.completed_at,
            "evidenceKind": self.evidence_kind,
            "features": dict(self.features),
            "observations": [dict(item) for item in self.observations],
            "warnings": list(self.warnings),
            "provenance": {
                "adapterId": "server-measured-audio-descriptors",
                "preprocessingId": PREPROCESSING_ID,
                "sampleRateHz": SAMPLE_RATE,
                "usesLearnedModel": False,
            },
            "behavioralOutcome": False,
        }


def _frames(samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return np.empty((0, WINDOW_SAMPLES), dtype=np.float64)
    if samples.size < WINDOW_SAMPLES:
        padded = np.pad(samples, (0, WINDOW_SAMPLES - samples.size))
        return padded.reshape(1, WINDOW_SAMPLES)
    count = 1 + (samples.size - WINDOW_SAMPLES) // HOP_SAMPLES
    shape = (count, WINDOW_SAMPLES)
    strides = (samples.strides[0] * HOP_SAMPLES, samples.strides[0])
    return np.lib.stride_tricks.as_strided(samples, shape=shape, strides=strides).copy()


def _finite_feature_map(value: Mapping[str, Any]) -> Mapping[str, float]:
    normalized: dict[str, float] = {}
    for name, raw in value.items():
        if (
            not isinstance(name, str)
            or not name.startswith("measured_audio.")
            or any(fragment in name.lower() for fragment in FORBIDDEN_FEATURE_FRAGMENTS)
            or not isinstance(raw, (int, float, np.integer, np.floating))
            or isinstance(raw, (bool, np.bool_))
            or not math.isfinite(float(raw))
        ):
            raise ValueError("measured audio produced an invalid scalar feature")
        normalized[name] = float(raw)
    return MappingProxyType(normalized)


def _measure(samples: np.ndarray) -> tuple[Mapping[str, float], tuple[Mapping[str, Any], ...]]:
    if samples.size == 0:
        return _finite_feature_map(
            {
                "measured_audio.present": 0.0,
                "measured_audio.decoded_seconds": 0.0,
                "measured_audio.sample_count": 0.0,
            }
        ), tuple()

    clipped = np.clip(samples, -1.0, 1.0)
    framed = _frames(clipped)
    rms_by_frame = np.sqrt(np.mean(np.square(framed), axis=1))
    window = np.hanning(WINDOW_SAMPLES)
    spectra = np.abs(np.fft.rfft(framed * window, axis=1))
    frequencies = np.fft.rfftfreq(WINDOW_SAMPLES, d=1.0 / SAMPLE_RATE)
    spectral_sums = spectra.sum(axis=1)
    safe_sums = np.maximum(spectral_sums, np.finfo(np.float64).eps)
    centroids = (spectra * frequencies).sum(axis=1) / safe_sums
    high_frequency_ratio = spectra[:, frequencies >= 4000].sum(axis=1) / safe_sums
    normalized_spectra = spectra / safe_sums[:, None]
    flux = (
        np.sqrt(np.square(np.diff(normalized_spectra, axis=0)).sum(axis=1))
        if normalized_spectra.shape[0] > 1
        else np.zeros(1, dtype=np.float64)
    )
    positive_spectra = np.maximum(spectra, np.finfo(np.float64).eps)
    flatness = np.exp(np.mean(np.log(positive_spectra), axis=1)) / np.maximum(
        np.mean(positive_spectra, axis=1), np.finfo(np.float64).eps
    )
    frame_db = 20.0 * np.log10(np.maximum(rms_by_frame, 1e-8))
    features = _finite_feature_map(
        {
            "measured_audio.present": 1.0,
            "measured_audio.decoded_seconds": samples.size / SAMPLE_RATE,
            "measured_audio.sample_count": samples.size,
            "measured_audio.rms": np.sqrt(np.mean(np.square(clipped))),
            "measured_audio.peak": np.max(np.abs(clipped)),
            "measured_audio.zero_crossings_per_second": (
                np.count_nonzero(np.signbit(clipped[1:]) != np.signbit(clipped[:-1]))
                / (samples.size / SAMPLE_RATE)
            ),
            "measured_audio.silent_window_fraction": np.mean(
                rms_by_frame < SILENCE_RMS_THRESHOLD
            ),
            "measured_audio.window_rms_p10": np.percentile(rms_by_frame, 10),
            "measured_audio.window_rms_p90": np.percentile(rms_by_frame, 90),
            "measured_audio.window_dynamic_range_db": (
                np.percentile(frame_db, 90) - np.percentile(frame_db, 10)
            ),
            "measured_audio.spectral_centroid_hz_mean": np.mean(centroids),
            "measured_audio.spectral_centroid_hz_std": np.std(centroids),
            "measured_audio.spectral_flatness_mean": np.mean(flatness),
            "measured_audio.high_frequency_energy_ratio_mean": np.mean(
                high_frequency_ratio
            ),
            "measured_audio.normalized_spectral_flux_mean": np.mean(flux),
        }
    )

    observations: list[Mapping[str, Any]] = []
    selected: list[int] = []
    minimum_gap = max(1, round(0.4 * SAMPLE_RATE / HOP_SAMPLES))
    for index in np.argsort(rms_by_frame)[::-1]:
        candidate = int(index)
        if any(abs(candidate - prior) < minimum_gap for prior in selected):
            continue
        selected.append(candidate)
        start = candidate * HOP_SAMPLES / SAMPLE_RATE
        end = min(samples.size / SAMPLE_RATE, (candidate * HOP_SAMPLES + WINDOW_SAMPLES) / SAMPLE_RATE)
        if end <= start:
            continue
        observations.append(
            MappingProxyType(
                {
                    "kind": "measured-short-window-energy-peak",
                    "startTime": round(start, 6),
                    "endTime": round(end, 6),
                    "text": "A relative short-window RMS peak was measured here; this is signal energy, not attention or engagement.",
                    "labels": ("measured", "relative-within-clip", "not-semantic"),
                }
            )
        )
        if len(observations) == 3:
            break
    return features, tuple(observations)


class MeasuredAudioAdapter:
    """Always-separate measured evidence; never impersonates ``audioModel``."""

    branch = "measuredAudio"

    def __init__(
        self,
        decoder: PcmDecoder | None = None,
        *,
        binary: str = "ffmpeg",
        probe_binary: str = "ffprobe",
    ) -> None:
        self.decoder = decoder or FfmpegPcmDecoder(
            binary=binary, probe_binary=probe_binary
        )
        self.binary = binary
        self.probe_binary = probe_binary

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "MeasuredAudioAdapter":
        source = os.environ if environ is None else environ
        binary = str(source.get("FORECAST_AUDIO_FFMPEG_BINARY", "ffmpeg")).strip()
        probe_binary = str(
            source.get("FORECAST_AUDIO_FFPROBE_BINARY", "ffprobe")
        ).strip()
        return cls(
            binary=binary or "ffmpeg",
            probe_binary=probe_binary or "ffprobe",
        )

    def availability(self) -> dict[str, Any]:
        executable = self.decoder is not None and (
            not isinstance(self.decoder, FfmpegPcmDecoder)
            or (
                shutil.which(self.binary) is not None
                and shutil.which(self.probe_binary) is not None
            )
        )
        return {
            "configured": executable,
            "executionAvailable": executable,
            "reason": None if executable else "The configured ffmpeg executable is unavailable.",
            "role": "server-measured-audio-descriptors",
            "usesLearnedModel": False,
            "isAudioModel": False,
            "provenance": {
                "adapterId": "server-measured-audio-descriptors",
                "preprocessingId": PREPROCESSING_ID,
                "sampleRateHz": SAMPLE_RATE,
            },
        }

    def run(
        self,
        *,
        video_path: Path,
        input_sha256: str,
        duration_seconds: float,
        context: Mapping[str, Any],
    ) -> MeasuredAudioOutput:
        del context
        if not video_path.is_file():
            raise MeasuredAudioUnavailable("The input video is unavailable.")
        if not SHA256_RE.fullmatch(input_sha256):
            raise ValueError("measured audio input hash is invalid")
        try:
            actual_input_sha256 = _sha256_file(video_path)
        except OSError as exc:
            raise MeasuredAudioUnavailable("The input video could not be read.") from exc
        if actual_input_sha256 != input_sha256:
            raise ValueError("measured audio input hash does not match the uploaded video")
        if not 10 <= float(duration_seconds) <= 60:
            raise ValueError("measured audio input duration must be from 10 to 60 seconds")
        started_at = _utc_now()
        samples = self.decoder.decode(video_path)
        if not isinstance(samples, np.ndarray) or samples.ndim != 1:
            raise MeasuredAudioUnavailable("The PCM decoder returned an invalid signal.")
        maximum_samples = math.ceil(float(duration_seconds) * SAMPLE_RATE)
        samples = samples[:maximum_samples]
        features, observations = _measure(samples.astype(np.float64, copy=False))
        warnings = (
            "These are deterministic PCM and spectral measurements, not learned speech, music, emotion, attention, or audience predictions.",
        )
        return MeasuredAudioOutput(
            input_sha256=input_sha256,
            started_at=started_at,
            completed_at=_utc_now(),
            features=features,
            observations=observations,
            warnings=warnings,
        )
