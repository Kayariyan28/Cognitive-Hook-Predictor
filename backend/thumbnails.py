"""Source-video thumbnail extraction with an auditable timing contract.

Only downscaled JPEG stills are retained. The uploaded video remains governed
by ``stage_uploaded_video`` and is deleted after inference/result persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Sequence

from .errors import ThumbnailExtractionError


THUMBNAIL_FORMAT = "image/jpeg"
THUMBNAIL_MAX_WIDTH_PIXELS = 320
THUMBNAIL_CAPTURE_TIME_BASIS = "tribe-segment-start"
THUMBNAIL_SELECTION = "first decoded source frame at-or-after requested time"
THUMBNAIL_TIMEOUT_SECONDS = 30
MAX_THUMBNAIL_BYTES = 2 * 1024 * 1024
THUMBNAIL_UNAVAILABLE_REASON = "source-frame-extraction-failed"
THUMBNAIL_OUTSIDE_SOURCE_REASON = "outside-source-video"
THUMBNAIL_CONTRACT_VERSION = "source-thumbnails/2"


@dataclass(frozen=True, slots=True)
class SerializedThumbnail:
    index: int
    requested_time_seconds: float
    path: Path
    byte_length: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SerializedUnavailableThumbnail:
    index: int
    requested_time_seconds: float
    reason: str = THUMBNAIL_OUTSIDE_SOURCE_REASON


@dataclass(frozen=True, slots=True)
class SerializedThumbnails:
    items: tuple[SerializedThumbnail, ...]
    unavailable_items: tuple[SerializedUnavailableThumbnail, ...] = ()


def _probe_video_duration_seconds(video_path: Path) -> float | None:
    """Return the selected video stream's declared duration when trustworthy.

    A failed seek alone is not enough to distinguish an out-of-range timestamp
    from a codec/decode failure. ffprobe's selected-stream duration supplies
    that boundary. If it cannot be established, extraction fails closed instead
    of mislabeling a decode problem as an expected end-of-video omission.
    """

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=THUMBNAIL_TIMEOUT_SECONDS,
        )
        if completed.returncode != 0:
            return None
        payload = json.loads(completed.stdout.decode("utf-8"))
        streams = payload.get("streams") if isinstance(payload, dict) else None
        if not isinstance(streams, list) or len(streams) != 1:
            return None
        duration = float(streams[0].get("duration"))
        if not math.isfinite(duration) or duration <= 0:
            return None
        return duration
    except (OSError, ValueError, TypeError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return None


def _validate_jpeg(path: Path) -> tuple[int, str]:
    byte_length = path.stat().st_size
    if byte_length <= 4 or byte_length > MAX_THUMBNAIL_BYTES:
        raise ThumbnailExtractionError("source-video thumbnail has an invalid size")
    digest = hashlib.sha256()
    first = b""
    last = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            if not first:
                first = chunk[:2]
            last = chunk[-2:]
            digest.update(chunk)
    if first != b"\xff\xd8" or last != b"\xff\xd9":
        raise ThumbnailExtractionError("ffmpeg did not produce a complete JPEG thumbnail")
    return byte_length, digest.hexdigest()


def extract_video_thumbnails(
    video_path: Path,
    times_seconds: Sequence[float],
    destination: Path,
) -> SerializedThumbnails:
    """Decode one real source frame at each authoritative TRIBE interval start.

    ``-ss`` is placed after ``-i`` so ffmpeg decodes forward from the input
    rather than returning only a potentially earlier keyframe. Since encoded
    video is discrete, the selected still is the first decoded frame at or
    after the requested segment-start time; the requested time is preserved in
    the manifest instead of claiming frame-exact capture at an impossible PTS.
    """

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ThumbnailExtractionError(
            "ffmpeg is required to extract source-video thumbnails"
        )
    if not video_path.is_file():
        raise ThumbnailExtractionError("source video is unavailable for thumbnails")
    if not times_seconds:
        raise ThumbnailExtractionError("cannot extract thumbnails without TRIBE timing")

    normalized_times: list[float] = []
    previous = -math.inf
    for index, raw_time in enumerate(times_seconds):
        try:
            time_seconds = float(raw_time)
        except (TypeError, ValueError) as exc:
            raise ThumbnailExtractionError(
                f"thumbnail time {index} is not numeric"
            ) from exc
        if not math.isfinite(time_seconds) or time_seconds < 0:
            raise ThumbnailExtractionError(
                f"thumbnail time {index} must be finite and non-negative"
            )
        if time_seconds <= previous:
            raise ThumbnailExtractionError(
                "thumbnail times must be strictly increasing"
            )
        normalized_times.append(time_seconds)
        previous = time_seconds

    video_duration_seconds = _probe_video_duration_seconds(video_path)

    destination.mkdir(parents=True, exist_ok=False)
    serialized: list[SerializedThumbnail] = []
    unavailable: list[SerializedUnavailableThumbnail] = []
    try:
        for index, time_seconds in enumerate(normalized_times):
            final_path = destination / f"{index:04d}.jpg"
            temporary_path = destination / f".{index:04d}.jpg.tmp"
            command = [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-ss",
                format(time_seconds, ".17g"),
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-vf",
                (
                    f"scale={THUMBNAIL_MAX_WIDTH_PIXELS}:-2:"
                    "force_original_aspect_ratio=decrease"
                ),
                "-frames:v",
                "1",
                "-c:v",
                "mjpeg",
                "-q:v",
                "4",
                "-f",
                "image2",
                "-y",
                str(temporary_path),
            ]
            try:
                completed = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=THUMBNAIL_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                raise ThumbnailExtractionError(
                    f"thumbnail extraction timed out at interval {index}"
                ) from exc
            if completed.returncode != 0 or not temporary_path.is_file():
                temporary_path.unlink(missing_ok=True)
                is_outside_source = (
                    bool(serialized)
                    and video_duration_seconds is not None
                    and (
                        time_seconds > video_duration_seconds
                        or math.isclose(
                            time_seconds,
                            video_duration_seconds,
                            rel_tol=0.0,
                            abs_tol=1e-6,
                        )
                    )
                )
                if is_outside_source:
                    unavailable.extend(
                        SerializedUnavailableThumbnail(
                            index=remaining_index,
                            requested_time_seconds=remaining_time,
                        )
                        for remaining_index, remaining_time in enumerate(
                            normalized_times[index:], start=index
                        )
                    )
                    break
                detail = completed.stderr.decode("utf-8", errors="replace").strip()
                detail = detail[-300:] if detail else "no image was decoded"
                raise ThumbnailExtractionError(
                    f"thumbnail extraction failed at interval {index}: {detail}"
                )
            byte_length, sha256 = _validate_jpeg(temporary_path)
            os.replace(temporary_path, final_path)
            serialized.append(
                SerializedThumbnail(
                    index=index,
                    requested_time_seconds=time_seconds,
                    path=final_path,
                    byte_length=byte_length,
                    sha256=sha256,
                )
            )
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise

    return SerializedThumbnails(tuple(serialized), tuple(unavailable))


def thumbnails_manifest(
    result_id: str, thumbnails: SerializedThumbnails
) -> dict[str, object]:
    """Serialize the stable public contract for persisted thumbnail stills."""

    available_indices = [item.index for item in thumbnails.items]
    unavailable_indices = [item.index for item in thumbnails.unavailable_items]
    if not thumbnails.items:
        raise ValueError("at least one decoded thumbnail is required")
    if available_indices != sorted(available_indices) or unavailable_indices != sorted(
        unavailable_indices
    ):
        raise ValueError("thumbnail indices must be ordered")
    if sorted(available_indices + unavailable_indices) != list(
        range(len(available_indices) + len(unavailable_indices))
    ):
        raise ValueError("thumbnail entries must partition the TRIBE frames")
    if any(
        item.reason != THUMBNAIL_OUTSIDE_SOURCE_REASON
        for item in thumbnails.unavailable_items
    ):
        raise ValueError("unsupported unavailable-thumbnail reason")

    manifest: dict[str, object] = {
        "status": "partial" if thumbnails.unavailable_items else "available",
        "source": "uploaded-video",
        "captureTimeBasis": THUMBNAIL_CAPTURE_TIME_BASIS,
        "selection": THUMBNAIL_SELECTION,
        "format": THUMBNAIL_FORMAT,
        "maxWidthPixels": THUMBNAIL_MAX_WIDTH_PIXELS,
        "items": [
            {
                "index": item.index,
                "requestedTimeSeconds": item.requested_time_seconds,
                "url": (
                    f"/api/tribe/v1/results/{result_id}/thumbnails/"
                    f"{item.index:04d}.jpg"
                ),
                "byteLength": item.byte_length,
                "sha256": item.sha256,
            }
            for item in thumbnails.items
        ],
    }
    if thumbnails.unavailable_items:
        manifest["unavailableItems"] = [
            {
                "index": item.index,
                "requestedTimeSeconds": item.requested_time_seconds,
                "reason": item.reason,
            }
            for item in thumbnails.unavailable_items
        ]
    return manifest


def unavailable_thumbnails_manifest() -> dict[str, object]:
    """Return a non-deceptive fallback when no real source still was decoded."""

    return {
        "status": "unavailable",
        "source": "uploaded-video",
        "captureTimeBasis": THUMBNAIL_CAPTURE_TIME_BASIS,
        "selection": THUMBNAIL_SELECTION,
        "format": THUMBNAIL_FORMAT,
        "maxWidthPixels": THUMBNAIL_MAX_WIDTH_PIXELS,
        "reason": THUMBNAIL_UNAVAILABLE_REASON,
        "items": [],
    }
