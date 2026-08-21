"""HTTP surface for the descriptive insight lane.

Route conventions follow `/api/forecast/v1`: private/no-store responses, a
`{code, message}` detail for genuine caller errors, and a fail-closed
`unavailable` document — not an HTTP error — when the lane itself cannot
produce something, exactly as an evidence branch reports unavailability inside
an otherwise successful result.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from starlette.background import BackgroundTask

from fastapi import APIRouter, Body, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse

from .config import InsightSettings
from .experiments import ExperimentError, ExperimentRequest, ExperimentTracker
from .outcomes import OutcomeImportError, OutcomeLedger
from .recut import OPERATIONS, RecutAssistant, RecutError
from .variants import VariantComparisonError
from .service import InsightRequest, InsightService
from .store import InsightStoreError


PRIVATE_NO_STORE_HEADERS = {
    "Cache-Control": "private, no-store",
    "Pragma": "no-cache",
    "X-Content-Type-Options": "nosniff",
}

RESULT_ID_LENGTH = 32
MAXIMUM_RECUT_BYTES = 256 * 1024 * 1024


def _http_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
        headers=PRIVATE_NO_STORE_HEADERS,
    )


def _result_id(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise _http_error(400, "invalid_request", f"{field} must be a string")
    normalized = value.strip().lower().replace("-", "")
    if len(normalized) != RESULT_ID_LENGTH or not all(
        character in "0123456789abcdef" for character in normalized
    ):
        raise _http_error(400, "invalid_request", f"{field} must be a result identifier")
    return normalized


def _parse_request(body: Mapping[str, Any]) -> InsightRequest:
    if not isinstance(body, Mapping):
        raise _http_error(400, "invalid_request", "the request body must be a JSON object")
    allowed = {"forecastResultId", "tribeResultId", "tribeDescriptors", "hookOnly"}
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise _http_error(
            400, "invalid_request", f"unsupported request fields: {', '.join(unknown)}"
        )
    if "forecastResultId" not in body:
        raise _http_error(400, "invalid_request", "forecastResultId is required")
    hook_only = body.get("hookOnly", False)
    if not isinstance(hook_only, bool):
        raise _http_error(400, "invalid_request", "hookOnly must be a boolean")
    descriptors = body.get("tribeDescriptors")
    if descriptors is not None and not isinstance(descriptors, Mapping):
        raise _http_error(400, "invalid_request", "tribeDescriptors must be a JSON object")
    tribe_result_id = body.get("tribeResultId")
    return InsightRequest(
        forecast_result_id=_result_id(body["forecastResultId"], "forecastResultId"),
        tribe_result_id=(
            _result_id(tribe_result_id, "tribeResultId") if tribe_result_id is not None else None
        ),
        tribe_descriptors=descriptors,
        hook_only=hook_only,
    )


def create_insight_router(
    *,
    service: InsightService | None = None,
    settings: InsightSettings | None = None,
    forecast_result_loader: Any = None,
    tracker: ExperimentTracker | None = None,
    outcome_ledger: OutcomeLedger | None = None,
    recut_assistant: RecutAssistant | None = None,
) -> APIRouter:
    active_settings = settings or InsightSettings.from_env()
    active_service = service or InsightService(
        active_settings, forecast_result_loader=forecast_result_loader
    )
    active_tracker = tracker or ExperimentTracker(
        active_service.store,
        forecast_result_loader=active_service.forecast_result_loader,
    )

    active_recut = recut_assistant or RecutAssistant()

    router = APIRouter(prefix="/api/insight/v1", tags=["creator-insight"])

    @router.post("/variants")
    def compare_variants(body: dict[str, Any] = Body(...)) -> JSONResponse:
        """Measured signals across several analysed cuts. No model is consulted."""

        if not isinstance(body, Mapping) or set(body) - {"resultIds", "labels"}:
            raise _http_error(
                400, "invalid_request", "the request body accepts resultIds and labels"
            )
        raw_ids = body.get("resultIds")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise _http_error(400, "invalid_request", "resultIds must be a non-empty array")
        result_ids = [_result_id(value, "resultIds") for value in raw_ids]
        raw_labels = body.get("labels")
        labels: dict[str, str] = {}
        if raw_labels is not None:
            if not isinstance(raw_labels, Mapping):
                raise _http_error(400, "invalid_request", "labels must be a JSON object")
            for key, value in raw_labels.items():
                if isinstance(value, str) and value.strip():
                    labels[_result_id(key, "labels")] = value.strip()[:64]
        try:
            comparison = active_service.compare_variants(result_ids, labels)
        except VariantComparisonError as exc:
            return JSONResponse(
                exc.as_dict(), status_code=200, headers=PRIVATE_NO_STORE_HEADERS
            )
        return JSONResponse(comparison, headers=PRIVATE_NO_STORE_HEADERS)

    @router.get("/recut/operations")
    def recut_operations() -> JSONResponse:
        return JSONResponse(
            {
                "schemaVersion": "insight-recut-operations/1",
                "available": active_recut.available(),
                "operations": [
                    {"id": key, "description": value} for key, value in OPERATIONS.items()
                ],
                "limits": (
                    "A recut changes the clip, not any measurement of it. The result is "
                    "returned to you and must be submitted as a new evidence job before "
                    "anything can be said about it."
                ),
            },
            headers=PRIVATE_NO_STORE_HEADERS,
        )

    @router.get("/status")
    def insight_status() -> JSONResponse:
        return JSONResponse(active_service.status(), headers=PRIVATE_NO_STORE_HEADERS)

    @router.post("/generate")
    def generate_insight(body: dict[str, Any] = Body(...)) -> JSONResponse:
        request = _parse_request(body)
        try:
            document, cached = active_service.generate(request)
        except InsightStoreError as exc:
            raise _http_error(
                500, "insight_storage_failed", "Insight state could not be persisted."
            ) from exc
        return JSONResponse(
            document,
            headers={
                **PRIVATE_NO_STORE_HEADERS,
                "X-Insight-Cache": "hit" if cached else "miss",
            },
        )

    @router.post("/recut")
    async def render_recut(
        video: UploadFile = File(...),
        operation: str = Form(...),
        durationSeconds: float = Form(...),
        seconds: float | None = Form(default=None),
        startSec: float | None = Form(default=None),
        endSec: float | None = Form(default=None),
    ) -> Any:
        """Render one mechanical recut and hand the clip straight back.

        Nothing is retained: the render lives in a temporary directory that is
        removed as the response finishes, and the creator submits the result as
        an ordinary new evidence job.
        """

        suffix = Path(video.filename or "clip.mp4").suffix.lower()
        if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
            raise _http_error(400, "invalid_video", "the clip must be a video file")
        workspace = Path(tempfile.mkdtemp(prefix="signalframe-recut-"))
        source = workspace / f"source{suffix}"
        destination = workspace / f"recut{suffix}"
        try:
            size = 0
            with source.open("wb") as handle:
                while chunk := await video.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAXIMUM_RECUT_BYTES:
                        raise _http_error(
                            413, "invalid_video", "the clip exceeds the recut size limit"
                        )
                    handle.write(chunk)
            if size == 0:
                raise _http_error(400, "invalid_video", "the uploaded clip is empty")
            rendered, plan = active_recut.recut(
                source,
                destination,
                operation=operation,
                duration_seconds=durationSeconds,
                seconds=seconds,
                start_seconds=startSec,
                end_seconds=endSec,
            )
        except RecutError as exc:
            shutil.rmtree(workspace, ignore_errors=True)
            return JSONResponse(
                exc.as_dict(), status_code=422, headers=PRIVATE_NO_STORE_HEADERS
            )
        except HTTPException:
            shutil.rmtree(workspace, ignore_errors=True)
            raise
        except Exception:
            shutil.rmtree(workspace, ignore_errors=True)
            raise _http_error(500, "recut_failed", "The recut could not be rendered.")
        finally:
            await video.close()

        return FileResponse(
            rendered,
            media_type="application/octet-stream",
            filename=f"recut-{plan.operation}{suffix}",
            background=BackgroundTask(shutil.rmtree, workspace, ignore_errors=True),
            headers={
                **PRIVATE_NO_STORE_HEADERS,
                "X-Recut-Plan": json.dumps(plan.public_value(), separators=(",", ":")),
            },
        )

    @router.post("/hook-readout")
    def hook_readout(body: dict[str, Any] = Body(...)) -> JSONResponse:
        """Deterministic timeline and checklist. This route never calls a model."""

        request = _parse_request(body)
        readout = active_service.hook_readout(request)
        return JSONResponse(readout, headers=PRIVATE_NO_STORE_HEADERS)

    @router.get("/results/{insight_id}")
    def get_insight(insight_id: str) -> JSONResponse:
        identifier = _result_id(insight_id, "insightId")
        try:
            artifact = active_service.read_artifact(identifier)
        except InsightStoreError as exc:
            raise _http_error(
                500, "insight_storage_failed", "Stored insight state is unreadable."
            ) from exc
        if artifact is None:
            raise _http_error(404, "insight_not_found", "Insight artifact was not found.")
        return JSONResponse(artifact, headers=PRIVATE_NO_STORE_HEADERS)

    @router.get("/results/{insight_id}/evidence")
    def get_insight_evidence(insight_id: str) -> JSONResponse:
        identifier = _result_id(insight_id, "insightId")
        try:
            bundle = active_service.read_bundle(identifier)
        except InsightStoreError as exc:
            raise _http_error(
                500, "insight_storage_failed", "Stored insight state is unreadable."
            ) from exc
        if bundle is None:
            raise _http_error(
                404, "evidence_not_found", "Insight evidence was not found."
            )
        return JSONResponse(bundle, headers=PRIVATE_NO_STORE_HEADERS)

    @router.get("/rejections/{rejection_id}")
    def get_rejection(rejection_id: str) -> JSONResponse:
        identifier = _result_id(rejection_id, "rejectionId")
        try:
            record = active_service.read_rejection(identifier)
        except InsightStoreError as exc:
            raise _http_error(
                500, "insight_storage_failed", "Stored insight state is unreadable."
            ) from exc
        if record is None:
            raise _http_error(404, "rejection_not_found", "Rejection record was not found.")
        return JSONResponse(record, headers=PRIVATE_NO_STORE_HEADERS)

    def experiment_error(exc: ExperimentError) -> HTTPException:
        message = str(exc)
        status_code = 404 if "was not found" in message or "proposes no experiment" in message else 409
        return _http_error(status_code, "experiment_unavailable", message)

    @router.post("/experiments", status_code=201)
    def create_experiment(body: dict[str, Any] = Body(...)) -> JSONResponse:
        if not isinstance(body, Mapping):
            raise _http_error(400, "invalid_request", "the request body must be a JSON object")
        unknown = sorted(set(body) - {"insightId", "experimentId"})
        if unknown:
            raise _http_error(
                400, "invalid_request", f"unsupported request fields: {', '.join(unknown)}"
            )
        experiment_id = body.get("experimentId")
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise _http_error(400, "invalid_request", "experimentId is required")
        request = ExperimentRequest(
            insight_id=_result_id(body.get("insightId"), "insightId"),
            experiment_id=experiment_id.strip(),
        )
        try:
            record = active_tracker.create(request)
        except ExperimentError as exc:
            raise experiment_error(exc) from exc
        except InsightStoreError as exc:
            raise _http_error(
                500, "insight_storage_failed", "Experiment state could not be persisted."
            ) from exc
        return JSONResponse(record, status_code=201, headers=PRIVATE_NO_STORE_HEADERS)

    @router.post("/experiments/{experiment_id}/edited")
    def mark_experiment_edited(experiment_id: str) -> JSONResponse:
        identifier = _result_id(experiment_id, "experimentId")
        try:
            record = active_tracker.mark_edited(identifier)
        except ExperimentError as exc:
            raise experiment_error(exc) from exc
        except InsightStoreError as exc:
            raise _http_error(
                500, "insight_storage_failed", "Experiment state could not be persisted."
            ) from exc
        return JSONResponse(record, headers=PRIVATE_NO_STORE_HEADERS)

    @router.post("/experiments/{experiment_id}/variant")
    def attach_experiment_variant(
        experiment_id: str, body: dict[str, Any] = Body(...)
    ) -> JSONResponse:
        identifier = _result_id(experiment_id, "experimentId")
        if not isinstance(body, Mapping) or set(body) - {"forecastResultId"}:
            raise _http_error(
                400, "invalid_request", "the request body accepts forecastResultId only"
            )
        variant_result_id = _result_id(body.get("forecastResultId"), "forecastResultId")
        try:
            record = active_tracker.attach_variant(identifier, variant_result_id)
        except ExperimentError as exc:
            raise experiment_error(exc) from exc
        except InsightStoreError as exc:
            raise _http_error(
                500, "insight_storage_failed", "Experiment state could not be persisted."
            ) from exc
        return JSONResponse(record, headers=PRIVATE_NO_STORE_HEADERS)

    @router.get("/experiments")
    def list_experiments() -> JSONResponse:
        try:
            records = active_tracker.list_experiments()
        except InsightStoreError as exc:
            raise _http_error(
                500, "insight_storage_failed", "Stored experiment state is unreadable."
            ) from exc
        return JSONResponse(
            {"schemaVersion": "insight-experiment-list/1", "experiments": records},
            headers=PRIVATE_NO_STORE_HEADERS,
        )

    @router.get("/experiments/{experiment_id}")
    def get_experiment(experiment_id: str) -> JSONResponse:
        identifier = _result_id(experiment_id, "experimentId")
        try:
            record = active_tracker.read(identifier)
        except InsightStoreError as exc:
            raise _http_error(
                500, "insight_storage_failed", "Stored experiment state is unreadable."
            ) from exc
        if record is None:
            raise _http_error(404, "experiment_not_found", "Experiment was not found.")
        return JSONResponse(record, headers=PRIVATE_NO_STORE_HEADERS)

    # -- outcomes ----------------------------------------------------------
    #
    # Creator-declared labels for a future calibration head. These routes write
    # to a store nothing in the insight path can read.

    @router.post("/outcomes", status_code=201)
    async def import_outcomes(request: Request) -> JSONResponse:
        if outcome_ledger is None:
            raise _http_error(
                503,
                "outcomes_unavailable",
                "Outcome ingestion is not configured on this service.",
            )
        raw = await request.body()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _http_error(
                400, "invalid_request", "the CSV must be UTF-8 encoded"
            ) from exc
        try:
            document = outcome_ledger.import_csv(text)
        except OutcomeImportError as exc:
            return JSONResponse(
                exc.as_dict(), status_code=422, headers=PRIVATE_NO_STORE_HEADERS
            )
        except InsightStoreError as exc:
            raise _http_error(
                500, "insight_storage_failed", "Outcome state could not be persisted."
            ) from exc
        return JSONResponse(document, status_code=201, headers=PRIVATE_NO_STORE_HEADERS)

    @router.get("/outcomes")
    def list_outcomes() -> JSONResponse:
        if outcome_ledger is None:
            raise _http_error(
                503,
                "outcomes_unavailable",
                "Outcome ingestion is not configured on this service.",
            )
        try:
            sets = outcome_ledger.list_sets()
        except InsightStoreError as exc:
            raise _http_error(
                500, "insight_storage_failed", "Stored outcome state is unreadable."
            ) from exc
        return JSONResponse(
            {
                "schemaVersion": "insight-outcome-list/1",
                "outcomeSets": sets,
                "limits": (
                    "Creator-declared, unverified post-publish numbers. They are training "
                    "labels for a future calibration head and are never read by the "
                    "insight lane."
                ),
            },
            headers=PRIVATE_NO_STORE_HEADERS,
        )

    @router.delete("/outcomes/{outcome_set_id}")
    def delete_outcomes(outcome_set_id: str) -> JSONResponse:
        if outcome_ledger is None:
            raise _http_error(
                503,
                "outcomes_unavailable",
                "Outcome ingestion is not configured on this service.",
            )
        identifier = _result_id(outcome_set_id, "outcomeSetId")
        try:
            deleted = outcome_ledger.delete(identifier)
        except InsightStoreError as exc:
            raise _http_error(
                500, "insight_storage_failed", "Outcome state could not be removed."
            ) from exc
        if not deleted:
            raise _http_error(404, "outcomes_not_found", "Outcome set was not found.")
        return JSONResponse(
            {"outcomeSetId": identifier, "deleted": True},
            headers=PRIVATE_NO_STORE_HEADERS,
        )

    return router
