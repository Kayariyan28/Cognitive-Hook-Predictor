"""Content-addressed, provenance-bound reuse of completed TRIBE results.

The cache never creates predictions. It can only return a previously completed
result whose manifest and raw float32 payload still satisfy the exact pinned
model contract and whose payload length and SHA-256 are revalidated from disk.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterator
from uuid import UUID

from .artifacts import (
    EXPECTED_HEMODYNAMIC_OFFSET_SECONDS,
    SCHEMA_VERSION,
    SURFACE_MAPPING_ID,
    SURFACE_SPACE,
    VERTEX_COUNT,
    VERTICES_PER_HEMISPHERE,
)
from .config import (
    PREPROCESSING_VERSION,
    TRIBE_CODE_REVISION,
    TRIBE_MODEL_WEIGHTS_SHA256,
)
from .thumbnails import (
    MAX_THUMBNAIL_BYTES,
    THUMBNAIL_CAPTURE_TIME_BASIS,
    THUMBNAIL_CONTRACT_VERSION,
    THUMBNAIL_FORMAT,
    THUMBNAIL_MAX_WIDTH_PIXELS,
    THUMBNAIL_OUTSIDE_SOURCE_REASON,
    THUMBNAIL_SELECTION,
    THUMBNAIL_UNAVAILABLE_REASON,
)


CACHE_IDENTITY_VERSION = "tribe-result-cache/1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RESULT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
HASH_CHUNK_BYTES = 8 * 1024 * 1024


class CacheValidationError(RuntimeError):
    """A newly stored artifact failed its own cache-integrity check."""


@dataclass(frozen=True, slots=True)
class CacheIdentity:
    key: str
    input_sha256: str
    payload: dict[str, Any]


def sha256_file(path: Path) -> str:
    """Hash a file without reading the complete upload/artifact into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def build_cache_identity(
    settings: Any,
    input_sha256: str,
    *,
    preprocessing_version: str = PREPROCESSING_VERSION,
) -> CacheIdentity:
    """Bind reuse to input bytes and every inference-relevant immutable pin."""

    normalized_digest = input_sha256.strip().lower()
    if not SHA256_RE.fullmatch(normalized_digest):
        raise ValueError("input_sha256 must be a lowercase 64-character SHA-256")
    if not preprocessing_version:
        raise ValueError("preprocessing_version cannot be empty")

    payload: dict[str, Any] = {
        "identityVersion": CACHE_IDENTITY_VERSION,
        "inputSha256": normalized_digest,
        "inferenceMode": settings.inference_mode,
        "preprocessingVersion": preprocessing_version,
        "tribe": {
            "modelId": settings.model_id,
            "modelRevision": settings.model_revision,
            "checkpointName": settings.checkpoint_name,
            "weightsSha256": TRIBE_MODEL_WEIGHTS_SHA256,
            "codeRevision": TRIBE_CODE_REVISION,
        },
        "videoExtractor": dict(settings.video_extractor),
        "surfaceMappingId": SURFACE_MAPPING_ID,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    key = hashlib.sha256(canonical).hexdigest()
    return CacheIdentity(key=key, input_sha256=normalized_digest, payload=payload)


def cache_manifest_metadata(identity: CacheIdentity) -> dict[str, str]:
    """Bind an artifact to a key without exposing the raw input digest."""

    return {
        "identityVersion": CACHE_IDENTITY_VERSION,
        "key": identity.key,
        "preprocessingVersion": str(identity.payload["preprocessingVersion"]),
        "thumbnailContractVersion": THUMBNAIL_CONTRACT_VERSION,
    }


def stable_staging_path(
    work_dir: Path, identity: CacheIdentity, suffix: str
) -> Path:
    """Return the deterministic path used in Neuralset/Exca event UIDs."""

    normalized_suffix = suffix.lower()
    if normalized_suffix not in {".mp4", ".avi", ".mkv", ".mov", ".webm"}:
        raise ValueError("unsupported staging suffix")
    return (
        work_dir
        / "content-addressed"
        / identity.key[:2]
        / identity.key
        / f"input{normalized_suffix}"
    )


@contextmanager
def stage_uploaded_video(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> Iterator[Path]:
    """Atomically stage an upload at a stable path, then remove source material.

    The caller must hold the process-wide inference lock. A leftover staging
    file from an interrupted run is replaced by the freshly hashed upload.
    Neuralset's persistent feature cache remains available, but the raw video
    and any sibling audio/transcript scratch files are removed after inference.
    """

    if not SHA256_RE.fullmatch(expected_sha256) or expected_size <= 0:
        raise ValueError("invalid staged upload integrity metadata")
    if not source.is_file() or source.stat().st_size != expected_size:
        raise OSError("uploaded video disappeared or changed before staging")

    staging_directory = destination.parent
    staging_directory.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    try:
        if destination.stat().st_size != expected_size:
            raise OSError("staged video length changed before inference")
        if sha256_file(destination) != expected_sha256:
            raise OSError("staged video failed its SHA-256 check")
        yield destination
    finally:
        # The directory is unique to the provenance-bound cache key, so this
        # also removes full-mode sibling .wav/transcript scratch artifacts.
        shutil.rmtree(staging_directory, ignore_errors=True)
        for parent in (staging_directory.parent, staging_directory.parent.parent):
            try:
                parent.rmdir()
            except OSError:
                break


def _cache_record_path(cache_dir: Path, identity: CacheIdentity) -> Path:
    return (
        cache_dir
        / "completed-results"
        / identity.key[:2]
        / f"{identity.key}.json"
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON root is not an object")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".cache-record-",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_manifest_contract(
    manifest: dict[str, Any],
    *,
    result_id: str,
    identity: CacheIdentity,
    settings: Any,
    frames_path: Path,
) -> None:
    if manifest.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("cached manifest schema is incompatible")
    if manifest.get("resultId") != result_id:
        raise ValueError("cached manifest result ID does not match its directory")
    if manifest.get("predictionType") != "average-subject cortical BOLD response":
        raise ValueError("cached manifest prediction type is invalid")
    if manifest.get("isViralityScore") is not False:
        raise ValueError("cached result is not a cortical BOLD artifact")
    if manifest.get("inferenceMode") != settings.inference_mode:
        raise ValueError("cached inference mode does not match")
    if manifest.get("modalitiesUsed") != list(settings.modalities_used):
        raise ValueError("cached modalities do not match")
    expected_missing = [
        item
        for item in ("video", "audio", "text")
        if item not in settings.modalities_used
    ]
    if manifest.get("missingModalities") != expected_missing:
        raise ValueError("cached missing-modality declaration is invalid")
    if manifest.get("extractors") != {"video": dict(settings.video_extractor)}:
        raise ValueError("cached video extractor provenance does not match")
    if manifest.get("model") != {
        "id": settings.model_id,
        "revision": settings.model_revision,
        "weightsSha256": TRIBE_MODEL_WEIGHTS_SHA256,
        "codeRevision": TRIBE_CODE_REVISION,
        "license": "CC-BY-NC-4.0",
    }:
        raise ValueError("cached TRIBE provenance does not match")
    if manifest.get("cache") != cache_manifest_metadata(identity):
        raise ValueError("cached artifact is not bound to this cache identity")

    surface = manifest.get("surface")
    if not isinstance(surface, dict) or surface != {
        "space": SURFACE_SPACE,
        "mappingId": SURFACE_MAPPING_ID,
        "vertexCount": VERTEX_COUNT,
        "hemispheres": [
            {"id": "left", "offset": 0, "count": VERTICES_PER_HEMISPHERE},
            {
                "id": "right",
                "offset": VERTICES_PER_HEMISPHERE,
                "count": VERTICES_PER_HEMISPHERE,
            },
        ],
    }:
        raise ValueError("cached fsaverage5 mapping contract is invalid")

    frames = manifest.get("frames")
    if not isinstance(frames, dict):
        raise ValueError("cached frames metadata is missing")
    shape = frames.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not isinstance(shape[0], int)
        or isinstance(shape[0], bool)
        or shape[0] < 1
        or shape[1] != VERTEX_COUNT
    ):
        raise ValueError("cached frame shape is invalid")
    frame_count = shape[0]
    expected_length = frame_count * VERTEX_COUNT * 4
    if frames.get("byteLength") != expected_length:
        raise ValueError("cached frame byte length is inconsistent with its shape")
    if frames.get("url") != f"/api/tribe/v1/results/{result_id}/frames.f32":
        raise ValueError("cached frame URL is invalid")
    if (
        frames.get("dtype") != "float32"
        or frames.get("byteOrder") != "little-endian"
        or frames.get("layout") != "time-major"
    ):
        raise ValueError("cached frame encoding contract is invalid")
    if not frames_path.is_file() or frames_path.stat().st_size != expected_length:
        raise ValueError("cached frames file length does not match its manifest")
    expected_frames_sha = frames.get("sha256")
    if not isinstance(expected_frames_sha, str) or not SHA256_RE.fullmatch(
        expected_frames_sha
    ):
        raise ValueError("cached frame SHA-256 is invalid")
    if sha256_file(frames_path) != expected_frames_sha:
        raise ValueError("cached frames file failed its SHA-256 check")

    times = frames.get("timesSeconds")
    durations = frames.get("durationsSeconds")
    if (
        not isinstance(times, list)
        or not isinstance(durations, list)
        or len(times) != frame_count
        or len(durations) != frame_count
        or not all(_is_finite_number(value) for value in times)
        or not all(
            _is_finite_number(value) and float(value) > 0 for value in durations
        )
        or any(
            float(times[index]) <= float(times[index - 1])
            for index in range(1, frame_count)
        )
    ):
        raise ValueError("cached frame timing is invalid")
    offset = frames.get("hemodynamicOffsetSeconds")
    if not _is_finite_number(offset) or not math.isclose(
        float(offset), EXPECTED_HEMODYNAMIC_OFFSET_SECONDS, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("cached hemodynamic offset is invalid")
    scale = frames.get("displayScale")
    domain = scale.get("domain") if isinstance(scale, dict) else None
    if (
        not isinstance(domain, list)
        or len(domain) != 3
        or not all(_is_finite_number(value) for value in domain)
        or not float(domain[0]) < float(domain[1]) < float(domain[2])
        or scale.get("scope") != "all-frames-all-vertices"
    ):
        raise ValueError("cached display scale is invalid")

    thumbnails = manifest.get("thumbnails")
    if not isinstance(thumbnails, dict) or thumbnails.get("source") != "uploaded-video":
        raise ValueError("cached source-video thumbnail metadata is missing")
    if (
        thumbnails.get("captureTimeBasis") != THUMBNAIL_CAPTURE_TIME_BASIS
        or thumbnails.get("selection") != THUMBNAIL_SELECTION
        or thumbnails.get("format") != THUMBNAIL_FORMAT
        or thumbnails.get("maxWidthPixels") != THUMBNAIL_MAX_WIDTH_PIXELS
    ):
        raise ValueError("cached source-video thumbnail provenance is invalid")
    status = thumbnails.get("status")
    items = thumbnails.get("items")
    thumbnail_directory = frames_path.parent / "thumbnails"
    if status == "unavailable":
        if (
            set(thumbnails) != {
                "status",
                "source",
                "captureTimeBasis",
                "selection",
                "format",
                "maxWidthPixels",
                "reason",
                "items",
            }
            or thumbnails.get("reason") != THUMBNAIL_UNAVAILABLE_REASON
            or items != []
        ):
            raise ValueError("cached unavailable-thumbnail declaration is invalid")
        if thumbnail_directory.exists() and any(thumbnail_directory.iterdir()):
            raise ValueError("unavailable thumbnails cannot retain source stills")
        return
    if status == "available":
        expected_keys = {
            "status",
            "source",
            "captureTimeBasis",
            "selection",
            "format",
            "maxWidthPixels",
            "items",
        }
        unavailable_items: list[Any] = []
    elif status == "partial":
        expected_keys = {
            "status",
            "source",
            "captureTimeBasis",
            "selection",
            "format",
            "maxWidthPixels",
            "items",
            "unavailableItems",
        }
        unavailable_items = thumbnails.get("unavailableItems")
        if (
            not isinstance(unavailable_items, list)
            or not unavailable_items
            or not isinstance(items, list)
            or not items
        ):
            raise ValueError("cached partial-thumbnail declaration is invalid")
    else:
        raise ValueError("cached thumbnail status is invalid")
    if set(thumbnails) != expected_keys or not isinstance(items, list):
        raise ValueError("cached thumbnail declaration is invalid")
    if len(items) + len(unavailable_items) != frame_count:
        raise ValueError("cached thumbnail count does not match TRIBE frames")

    available_indices: list[int] = []
    unavailable_indices: list[int] = []
    for item in items:
        index = item.get("index") if isinstance(item, dict) else None
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("cached thumbnail index is invalid")
        available_indices.append(index)
    for item in unavailable_items:
        index = item.get("index") if isinstance(item, dict) else None
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("cached unavailable-thumbnail index is invalid")
        unavailable_indices.append(index)
    if (
        available_indices != sorted(available_indices)
        or unavailable_indices != sorted(unavailable_indices)
        or sorted(available_indices + unavailable_indices) != list(range(frame_count))
    ):
        raise ValueError("cached thumbnail entries do not partition TRIBE frames")
    if not thumbnail_directory.is_dir() or {
        path.name for path in thumbnail_directory.iterdir()
    } != {f"{index:04d}.jpg" for index in available_indices}:
        raise ValueError("cached thumbnail file set is invalid")
    for item in items:
        index = item["index"]
        if (
            set(item) != {
                "index",
                "requestedTimeSeconds",
                "url",
                "byteLength",
                "sha256",
            }
        ):
            raise ValueError("cached thumbnail item is invalid")
        requested_time = item.get("requestedTimeSeconds")
        if (
            not _is_finite_number(requested_time)
            or float(requested_time) < 0
            or not math.isclose(
                float(requested_time), float(times[index]), rel_tol=0.0, abs_tol=1e-9
            )
        ):
            raise ValueError("cached thumbnail timing is not bound to TRIBE timing")
        if item.get("url") != (
            f"/api/tribe/v1/results/{result_id}/thumbnails/{index:04d}.jpg"
        ):
            raise ValueError("cached thumbnail URL is invalid")
        byte_length = item.get("byteLength")
        expected_sha256 = item.get("sha256")
        if (
            not isinstance(byte_length, int)
            or isinstance(byte_length, bool)
            or byte_length <= 4
            or byte_length > MAX_THUMBNAIL_BYTES
            or not isinstance(expected_sha256, str)
            or not SHA256_RE.fullmatch(expected_sha256)
        ):
            raise ValueError("cached thumbnail integrity metadata is invalid")
        thumbnail_path = thumbnail_directory / f"{index:04d}.jpg"
        if (
            not thumbnail_path.is_file()
            or thumbnail_path.stat().st_size != byte_length
            or sha256_file(thumbnail_path) != expected_sha256
        ):
            raise ValueError("cached thumbnail failed its length or SHA-256 check")
    for item in unavailable_items:
        index = item["index"]
        if set(item) != {"index", "requestedTimeSeconds", "reason"}:
            raise ValueError("cached unavailable-thumbnail item is invalid")
        requested_time = item.get("requestedTimeSeconds")
        if (
            not _is_finite_number(requested_time)
            or float(requested_time) < 0
            or not math.isclose(
                float(requested_time), float(times[index]), rel_tol=0.0, abs_tol=1e-9
            )
            or item.get("reason") != THUMBNAIL_OUTSIDE_SOURCE_REASON
        ):
            raise ValueError("cached unavailable-thumbnail timing or reason is invalid")


def validate_completed_result(
    result_dir: Path, identity: CacheIdentity, settings: Any
) -> dict[str, Any]:
    """Load and fully revalidate one persisted manifest plus frames payload."""

    result_id = result_dir.name.lower()
    if not RESULT_ID_RE.fullmatch(result_id) or UUID(hex=result_id).hex != result_id:
        raise ValueError("cached result directory is not a canonical UUID")
    manifest = _read_json_object(result_dir / "manifest.json")
    _validate_manifest_contract(
        manifest,
        result_id=result_id,
        identity=identity,
        settings=settings,
        frames_path=result_dir / "frames.f32",
    )
    return manifest


def load_cached_result(
    cache_dir: Path,
    result_dir: Path,
    identity: CacheIdentity,
    settings: Any,
) -> dict[str, Any] | None:
    """Return a valid completed result, or fail closed to fresh inference."""

    record_path = _cache_record_path(cache_dir, identity)
    try:
        record = _read_json_object(record_path)
        result_id = record.get("resultId")
        if (
            record.get("cacheKey") != identity.key
            or record.get("identity") != identity.payload
        ):
            return None
        if not isinstance(result_id, str) or not RESULT_ID_RE.fullmatch(result_id):
            return None
        return validate_completed_result(result_dir / result_id, identity, settings)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def memoize_completed_result(
    cache_dir: Path,
    result_directory: Path,
    identity: CacheIdentity,
    settings: Any,
) -> dict[str, Any]:
    """Validate first, then atomically publish the completed-result pointer."""

    try:
        manifest = validate_completed_result(result_directory, identity, settings)
        _atomic_write_json(
            _cache_record_path(cache_dir, identity),
            {
                "cacheKey": identity.key,
                "identity": identity.payload,
                "resultId": result_directory.name,
            },
        )
        return manifest
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise CacheValidationError(
            "completed TRIBE result failed cache integrity validation"
        ) from exc
