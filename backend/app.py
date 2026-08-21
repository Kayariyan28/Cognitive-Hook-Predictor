"""FastAPI entry point for strict, real TRIBE v2 video inference."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .artifacts import (
    SCHEMA_VERSION,
    SURFACE_MAPPING_ID,
    build_manifest,
    calculate_display_scale,
    extract_segment_timing,
    prepare_predictions,
    write_frames,
)
from .config import Settings
from .config import TRIBE_MODEL_WEIGHTS_SHA256
from .errors import (
    ConfigurationError,
    InferenceError,
    ModelUnavailableError,
    PredictionValidationError,
    ThumbnailExtractionError,
    UploadValidationError,
)
from .forecast.router import create_forecast_router
from .forecast.jobs import ForecastJobService, JobStoreError
from .forecast.orchestrator import MultimodalEvidenceOrchestrator
from .forecast.registry import ForecastRegistry
from .insight.bundle import BundleUnavailableError
from .insight.config import InsightSettings
from .insight.router import create_insight_router
from .insight.service import InsightService
from .model_runtime import TribeRuntime
from .result_cache import (
    CacheIdentity,
    CacheValidationError,
    build_cache_identity,
    cache_manifest_metadata,
    load_cached_result,
    memoize_completed_result,
    stable_staging_path,
    stage_uploaded_video,
)
from .thumbnails import (
    extract_video_thumbnails,
    thumbnails_manifest,
    unavailable_thumbnails_manifest,
)


LOGGER = logging.getLogger("tribe_backend")
logging.basicConfig(level=os.getenv("TRIBE_LOG_LEVEL", "INFO").upper())

VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
UPLOAD_CHUNK_BYTES = 1024 * 1024

settings = Settings.from_env()
runtime = TribeRuntime(settings)
inference_lock = asyncio.Lock()
forecast_registry = ForecastRegistry.from_env()
forecast_orchestrator = MultimodalEvidenceOrchestrator.from_env(
    forecast_registry=forecast_registry
)
forecast_jobs = ForecastJobService.from_env(orchestrator=forecast_orchestrator)
insight_settings = InsightSettings.from_env()


def load_completed_forecast_result(result_id: str) -> dict[str, Any] | None:
    """Read one published forecast result; unreadable state fails closed."""

    try:
        return forecast_jobs.store.read_result(result_id)
    except JobStoreError as exc:
        raise BundleUnavailableError(
            f"the stored forecast result {result_id} is unreadable"
        ) from exc


insight_service = InsightService(
    insight_settings, forecast_result_loader=load_completed_forecast_result
)


@dataclass(frozen=True, slots=True)
class UploadedVideo:
    size: int
    sha256: str


def _http_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


async def _save_upload(upload: UploadFile, destination: Path) -> UploadedVideo:
    filename = Path(upload.filename or "")
    suffix = filename.suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise UploadValidationError(
            f"video filename must end in one of {sorted(VIDEO_SUFFIXES)}"
        )
    content_type = (upload.content_type or "").lower()
    if content_type and not (
        content_type.startswith("video/") or content_type == "application/octet-stream"
    ):
        raise UploadValidationError("multipart field 'video' must contain a video")

    size = 0
    digest = hashlib.sha256()
    with destination.open("xb") as handle:
        while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
            size += len(chunk)
            if size > settings.max_upload_bytes:
                raise UploadValidationError(
                    f"video exceeds the {settings.max_upload_bytes}-byte upload limit"
                )
            handle.write(chunk)
            digest.update(chunk)
    if size == 0:
        raise UploadValidationError("uploaded video is empty")
    return UploadedVideo(size=size, sha256=digest.hexdigest())


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    payload = json.dumps(manifest, indent=2, separators=(",", ": ")) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _persist_result(
    output: Any, cache_identity: CacheIdentity, source_video_path: Path
) -> dict[str, Any]:
    """Persist, self-validate, then publish one completed cache record."""

    result_directory: Path | None = None
    try:
        predictions = prepare_predictions(output.predictions)
        display_scale = calculate_display_scale(predictions)
        timing = extract_segment_timing(
            output.segments, expected_frames=int(predictions.shape[0])
        )
        result_id = uuid4().hex
        result_directory = settings.result_dir / result_id
        result_directory.mkdir(parents=True, exist_ok=False)
        frames = write_frames(predictions, result_directory / "frames.f32")
        try:
            thumbnails = extract_video_thumbnails(
                source_video_path,
                timing.times_seconds,
                result_directory / "thumbnails",
            )
            thumbnail_metadata = thumbnails_manifest(result_id, thumbnails)
        except ThumbnailExtractionError as exc:
            LOGGER.warning(
                "Source-video thumbnails are unavailable for result %s: %s",
                result_id,
                exc,
            )
            thumbnail_metadata = unavailable_thumbnails_manifest()
        manifest = build_manifest(
            result_id=result_id,
            model_id=settings.model_id,
            model_revision=settings.model_revision,
            model_weights_sha256=TRIBE_MODEL_WEIGHTS_SHA256,
            inference_mode=settings.inference_mode,
            modalities_used=settings.modalities_used,
            video_extractor=settings.video_extractor,
            frames=frames,
            timing=timing,
            hemodynamic_offset_seconds=output.hemodynamic_offset_seconds,
            display_scale=display_scale,
        )
        manifest["cache"] = cache_manifest_metadata(cache_identity)
        manifest["thumbnails"] = thumbnail_metadata
        _write_manifest(result_directory / "manifest.json", manifest)
        return memoize_completed_result(
            settings.cache_dir,
            result_directory,
            cache_identity,
            settings,
        )
    except Exception:
        if result_directory is not None:
            shutil.rmtree(result_directory, ignore_errors=True)
        raise


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.result_dir.mkdir(parents=True, exist_ok=True)
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    if settings.preload_model:
        if not settings.is_model_configured:
            LOGGER.error(
                "TRIBE_PRELOAD_MODEL is enabled but TRIBE_MODEL_REVISION is not an "
                "immutable commit SHA"
            )
        else:
            try:
                await asyncio.to_thread(runtime.load)
            except (ConfigurationError, ModelUnavailableError):
                LOGGER.exception("TRIBE v2 preload failed; requests will return 503")
    yield


app = FastAPI(
    title="TRIBE v2 fsaverage5 inference backend",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Accept", "Content-Type", "Idempotency-Key"],
    expose_headers=["Content-Length", "ETag", "Idempotency-Key", "Location"],
)
app.include_router(
    create_forecast_router(
        tribe_status_provider=runtime.status,
        registry=forecast_registry,
        job_service=forecast_jobs,
    )
)
app.include_router(create_insight_router(service=insight_service))


@app.get("/api/tribe/v1/health")
def health() -> dict[str, Any]:
    """Liveness only. Readiness is reported separately by the status endpoint."""

    return {
        "status": "ok",
        "schemaVersion": SCHEMA_VERSION,
        "surfaceMappingId": SURFACE_MAPPING_ID,
    }


@app.get("/api/tribe/v1/status")
def model_status() -> dict[str, Any]:
    """Report real model state without pretending an unloaded model is ready."""

    return runtime.status()


@app.post("/api/tribe/v1/predict")
async def predict(video: UploadFile = File(...)) -> dict[str, Any]:
    """Upload one video and synchronously return its actual TRIBE manifest."""

    filename = Path(video.filename or "")
    suffix = filename.suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        await video.close()
        raise _http_error(
            400,
            "invalid_video",
            f"video filename must end in one of {sorted(VIDEO_SUFFIXES)}",
        )

    settings.work_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix="tribe-upload-", dir=settings.work_dir
        ) as temporary_directory:
            video_path = Path(temporary_directory) / f"input{suffix}"
            try:
                upload = await _save_upload(video, video_path)
            finally:
                await video.close()

            cache_identity = build_cache_identity(settings, upload.sha256)
            async with inference_lock:
                cached_manifest = await asyncio.to_thread(
                    load_cached_result,
                    settings.cache_dir,
                    settings.result_dir,
                    cache_identity,
                    settings,
                )
                if cached_manifest is not None:
                    LOGGER.info(
                        "Reused validated TRIBE result %s for cache key %s",
                        cached_manifest["resultId"],
                        cache_identity.key,
                    )
                    return cached_manifest

                staging_path = stable_staging_path(
                    settings.work_dir, cache_identity, suffix
                )
                with stage_uploaded_video(
                    video_path,
                    staging_path,
                    expected_sha256=upload.sha256,
                    expected_size=upload.size,
                ) as stable_video_path:
                    output = await asyncio.to_thread(
                        runtime.predict_video, stable_video_path
                    )
                    return await asyncio.to_thread(
                        _persist_result,
                        output,
                        cache_identity,
                        stable_video_path,
                    )
    except UploadValidationError as exc:
        status_code = 413 if "upload limit" in str(exc) else 400
        raise _http_error(status_code, "invalid_video", str(exc)) from exc
    except ConfigurationError as exc:
        raise _http_error(503, "model_unconfigured", str(exc)) from exc
    except ModelUnavailableError as exc:
        LOGGER.exception("Pinned TRIBE v2 model is unavailable")
        raise _http_error(
            503,
            "model_unavailable",
            "The pinned TRIBE v2 model is unavailable; check the status endpoint",
        ) from exc
    except InferenceError as exc:
        LOGGER.exception("Official TRIBE v2 inference failed")
        raise _http_error(
            500,
            "inference_failed",
            "The official TRIBE v2 pipeline failed; no neural result was produced",
        ) from exc
    except PredictionValidationError as exc:
        LOGGER.exception("TRIBE v2 returned an invalid fsaverage5 result")
        raise _http_error(
            502,
            "invalid_model_output",
            "TRIBE v2 output failed strict fsaverage5/timing validation",
        ) from exc
    except (CacheValidationError, ThumbnailExtractionError, OSError) as exc:
        LOGGER.exception("Could not persist TRIBE v2 result")
        raise _http_error(
            500,
            "result_storage_failed",
            "TRIBE v2 ran, but its result artifact could not be stored",
        ) from exc


def _result_file(result_id: UUID, filename: str) -> Path:
    return settings.result_dir / result_id.hex / filename


@app.get("/api/tribe/v1/results/{result_id}/frames.f32", name="get_frames")
def get_frames(result_id: UUID) -> FileResponse:
    path = _result_file(result_id, "frames.f32")
    if not path.is_file():
        raise _http_error(404, "result_not_found", "TRIBE result was not found")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"tribe-{result_id.hex}-frames.f32",
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/api/tribe/v1/results/{result_id}/manifest.json")
def get_manifest(result_id: UUID) -> FileResponse:
    path = _result_file(result_id, "manifest.json")
    if not path.is_file():
        raise _http_error(404, "result_not_found", "TRIBE result was not found")
    return FileResponse(
        path,
        media_type="application/json",
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/api/tribe/v1/results/{result_id}/thumbnails/{index}.jpg")
def get_thumbnail(result_id: UUID, index: int) -> FileResponse:
    if index < 0:
        raise _http_error(404, "thumbnail_not_found", "Thumbnail was not found")
    manifest_path = _result_file(result_id, "manifest.json")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        thumbnails = manifest.get("thumbnails")
        items = thumbnails.get("items") if isinstance(thumbnails, dict) else None
        expected_url = (
            f"/api/tribe/v1/results/{result_id.hex}/thumbnails/{index:04d}.jpg"
        )
        declared_item = next(
            (
                item
                for item in items
                if isinstance(item, dict) and item.get("index") == index
            ),
            None,
        ) if isinstance(items, list) else None
        if (
            not isinstance(thumbnails, dict)
            or thumbnails.get("status") not in {"available", "partial"}
            or declared_item is None
            or declared_item.get("url") != expected_url
        ):
            raise ValueError("thumbnail is not declared by this result")
    except (OSError, ValueError, json.JSONDecodeError, AttributeError):
        raise _http_error(404, "thumbnail_not_found", "Thumbnail was not found")
    path = _result_file(result_id, f"thumbnails/{index:04d}.jpg")
    if not path.is_file():
        raise _http_error(404, "thumbnail_not_found", "Thumbnail was not found")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; sandbox",
        },
    )
