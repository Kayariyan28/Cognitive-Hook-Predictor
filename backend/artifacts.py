"""Validation and lossless serialization of TRIBE v2 cortical predictions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np

from .config import (
    TRIBE_CODE_REVISION,
    VIDEO_DEVICES,
    VIDEO_MODEL_ID,
    VIDEO_MODEL_REVISION,
    VIDEO_MODEL_WEIGHTS_SHA256,
    VIDEO_PRECISIONS,
)
from .errors import PredictionValidationError


SCHEMA_VERSION = "tribe-fsaverage5/1"
SURFACE_SPACE = "fsaverage5"
SURFACE_MAPPING_ID = (
    "fsaverage5-half:52eda3c221c9439b320f97663d441390c56a14e69c2cddb7a5073f6d550466cb"
)
VERTICES_PER_HEMISPHERE = 10_242
VERTEX_COUNT = VERTICES_PER_HEMISPHERE * 2
EXPECTED_HEMODYNAMIC_OFFSET_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class SegmentTiming:
    times_seconds: tuple[float, ...]
    durations_seconds: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SerializedFrames:
    path: Path
    shape: tuple[int, int]
    byte_length: int
    sha256: str


def prepare_predictions(predictions: Any) -> np.ndarray:
    """Return validated, C-contiguous, little-endian float32 predictions.

    This does not fabricate missing vertices, reorder hemispheres, normalize per
    video, or replace invalid values. Any mismatch fails closed.
    """

    array = np.asarray(predictions)
    if array.ndim != 2:
        raise PredictionValidationError(
            f"TRIBE predictions must be rank 2, received shape {array.shape!r}"
        )
    if array.shape[0] < 1:
        raise PredictionValidationError("TRIBE returned no prediction frames")
    if array.shape[1] != VERTEX_COUNT:
        raise PredictionValidationError(
            f"TRIBE output must contain exactly {VERTEX_COUNT} fsaverage5 vertices; "
            f"received {array.shape[1]}"
        )
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(
        array.dtype, np.number
    ):
        raise PredictionValidationError(
            f"TRIBE predictions must be a real numeric array, received {array.dtype}"
        )
    if np.issubdtype(array.dtype, np.complexfloating):
        raise PredictionValidationError("Complex TRIBE predictions are not supported")
    if not np.isfinite(array).all():
        raise PredictionValidationError("TRIBE predictions contain NaN or infinity")

    prepared = np.ascontiguousarray(array, dtype=np.dtype("<f4"))
    if not np.isfinite(prepared).all():
        raise PredictionValidationError(
            "TRIBE predictions overflowed while converting to float32"
        )
    return prepared


def calculate_display_scale(predictions: Any) -> dict[str, Any]:
    """Compute one official-style robust scale for the complete prediction.

    TRIBE's plotting utility uses the global 1st/99th percentiles. This function
    computes those bounds once for the entire clip; it never normalizes each
    frame independently and it never modifies the serialized prediction values.
    """

    prepared = prepare_predictions(predictions)
    lower, median, upper = (
        float(value) for value in np.percentile(prepared, [1.0, 50.0, 99.0])
    )
    if not lower < upper:
        raise PredictionValidationError(
            "TRIBE predictions have no non-degenerate global p1/p99 display range"
        )

    if lower < 0.0 < upper:
        center = 0.0
        center_method = "zero"
    elif lower < median < upper:
        center = median
        center_method = "global-median"
    else:
        center = lower + (upper - lower) / 2.0
        center_method = "p1-p99-midpoint"

    if not lower < center < upper:
        raise PredictionValidationError(
            "TRIBE display scale could not produce a strictly increasing domain"
        )
    return {
        "type": "robust-diverging-linear",
        "domain": [lower, center, upper],
        "percentiles": [1.0, 99.0],
        "centerMethod": center_method,
        "scope": "all-frames-all-vertices",
        "clamp": True,
        "units": "TRIBE-v2 predicted BOLD (training-target z-score units)",
    }


def _segment_value(segment: Any, field: str) -> Any:
    if isinstance(segment, dict):
        return segment.get(field)
    return getattr(segment, field, None)


def extract_segment_timing(
    segments: Sequence[Any], *, expected_frames: int
) -> SegmentTiming:
    """Extract authoritative frame timing from the segments returned by TRIBE."""

    if len(segments) != expected_frames:
        raise PredictionValidationError(
            f"TRIBE returned {expected_frames} frames but {len(segments)} segments"
        )

    starts: list[float] = []
    durations: list[float] = []
    for index, segment in enumerate(segments):
        raw_start = _segment_value(segment, "start")
        raw_duration = _segment_value(segment, "duration")
        if raw_start is None or raw_duration is None:
            raise PredictionValidationError(
                f"TRIBE segment {index} has no explicit start/duration timing"
            )
        try:
            start = float(raw_start)
            duration = float(raw_duration)
        except (TypeError, ValueError) as exc:
            raise PredictionValidationError(
                f"TRIBE segment {index} contains non-numeric timing"
            ) from exc
        if not math.isfinite(start) or not math.isfinite(duration):
            raise PredictionValidationError(
                f"TRIBE segment {index} contains non-finite timing"
            )
        if duration <= 0:
            raise PredictionValidationError(
                f"TRIBE segment {index} duration must be positive"
            )
        if starts and start <= starts[-1]:
            raise PredictionValidationError(
                "TRIBE segment starts must be strictly increasing for video playback"
            )
        starts.append(start)
        durations.append(duration)

    return SegmentTiming(tuple(starts), tuple(durations))


def write_frames(predictions: np.ndarray, destination: Path) -> SerializedFrames:
    """Atomically write raw little-endian time-major float32 frames."""

    prepared = prepare_predictions(predictions)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    data = memoryview(prepared).cast("B")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".frames-",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for offset in range(0, len(data), 8 * 1024 * 1024):
                chunk = data[offset : offset + 8 * 1024 * 1024]
                handle.write(chunk)
                digest.update(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return SerializedFrames(
        path=destination,
        shape=(int(prepared.shape[0]), int(prepared.shape[1])),
        byte_length=destination.stat().st_size,
        sha256=digest.hexdigest(),
    )


def build_manifest(
    *,
    result_id: str,
    model_id: str,
    model_revision: str,
    model_weights_sha256: str,
    inference_mode: str,
    modalities_used: tuple[str, ...],
    video_extractor: dict[str, str],
    frames: SerializedFrames,
    timing: SegmentTiming,
    hemodynamic_offset_seconds: float,
    display_scale: dict[str, Any],
) -> dict[str, Any]:
    """Build the stable frontend contract for an actual TRIBE result."""

    if not math.isclose(
        hemodynamic_offset_seconds,
        EXPECTED_HEMODYNAMIC_OFFSET_SECONDS,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise PredictionValidationError(
            "Loaded TRIBE configuration does not use the expected 5-second "
            "hemodynamic offset"
        )
    if frames.shape[0] != len(timing.times_seconds):
        raise PredictionValidationError(
            "Serialized frame count does not match authoritative segment timing"
        )
    expected_modalities = (
        ("video",) if inference_mode == "vision-only" else ("video", "audio", "text")
    )
    if inference_mode not in {"full", "vision-only"} or tuple(modalities_used) != expected_modalities:
        raise PredictionValidationError("Inference mode and modalities are inconsistent")
    expected_video_pin = {
        "id": VIDEO_MODEL_ID,
        "revision": VIDEO_MODEL_REVISION,
        "weightsSha256": VIDEO_MODEL_WEIGHTS_SHA256,
    }
    if any(video_extractor.get(key) != value for key, value in expected_video_pin.items()):
        raise PredictionValidationError(
            "The video extractor must use the pinned official V-JEPA2 snapshot"
        )
    if set(video_extractor) != {*expected_video_pin, "device", "precision"}:
        raise PredictionValidationError(
            "The video extractor provenance fields are incomplete or unsupported"
        )
    video_device = video_extractor.get("device")
    video_precision = video_extractor.get("precision")
    if video_device not in VIDEO_DEVICES or video_precision not in VIDEO_PRECISIONS:
        raise PredictionValidationError(
            "The video extractor device or precision is unsupported"
        )
    if video_precision == "autocast-fp16" and video_device != "mps":
        raise PredictionValidationError(
            "V-JEPA autocast-fp16 provenance requires the MPS device"
        )
    if model_weights_sha256 != "9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321":
        raise PredictionValidationError("The TRIBE checkpoint weight digest is not the audited value")
    domain = display_scale.get("domain")
    if (
        not isinstance(domain, list)
        or len(domain) != 3
        or not all(isinstance(value, (int, float)) for value in domain)
        or not float(domain[0]) < float(domain[1]) < float(domain[2])
        or display_scale.get("scope") != "all-frames-all-vertices"
    ):
        raise PredictionValidationError(
            "Display scale must have a strictly increasing global three-value domain"
        )

    return {
        "schemaVersion": SCHEMA_VERSION,
        "resultId": result_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "predictionType": "average-subject cortical BOLD response",
        "isViralityScore": False,
        "inferenceMode": inference_mode,
        "modalitiesUsed": list(modalities_used),
        "missingModalities": [
            modality
            for modality in ("video", "audio", "text")
            if modality not in modalities_used
        ],
        "extractors": {"video": dict(video_extractor)},
        "model": {
            "id": model_id,
            "revision": model_revision,
            "weightsSha256": model_weights_sha256,
            "codeRevision": TRIBE_CODE_REVISION,
            "license": "CC-BY-NC-4.0",
        },
        "surface": {
            "space": SURFACE_SPACE,
            "mappingId": SURFACE_MAPPING_ID,
            "vertexCount": VERTEX_COUNT,
            "hemispheres": [
                {
                    "id": "left",
                    "offset": 0,
                    "count": VERTICES_PER_HEMISPHERE,
                },
                {
                    "id": "right",
                    "offset": VERTICES_PER_HEMISPHERE,
                    "count": VERTICES_PER_HEMISPHERE,
                },
            ],
        },
        "frames": {
            "url": f"/api/tribe/v1/results/{result_id}/frames.f32",
            "dtype": "float32",
            "byteOrder": "little-endian",
            "layout": "time-major",
            "shape": [frames.shape[0], frames.shape[1]],
            "byteLength": frames.byte_length,
            "sha256": frames.sha256,
            "timesSeconds": list(timing.times_seconds),
            "durationsSeconds": list(timing.durations_seconds),
            "hemodynamicOffsetSeconds": hemodynamic_offset_seconds,
            "displayScale": dict(display_scale),
        },
    }
