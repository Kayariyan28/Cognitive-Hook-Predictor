"""HTTP surface for the descriptive insight lane.

Route conventions follow `/api/forecast/v1`: private/no-store responses, a
`{code, message}` detail for genuine caller errors, and a fail-closed
`unavailable` document — not an HTTP error — when the lane itself cannot
produce something, exactly as an evidence branch reports unavailability inside
an otherwise successful result.
"""

from __future__ import annotations

from typing import Any, Mapping

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse

from .config import InsightSettings
from .experiments import ExperimentError, ExperimentRequest, ExperimentTracker
from .service import InsightRequest, InsightService
from .store import InsightStoreError


PRIVATE_NO_STORE_HEADERS = {
    "Cache-Control": "private, no-store",
    "Pragma": "no-cache",
    "X-Content-Type-Options": "nosniff",
}

RESULT_ID_LENGTH = 32


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
) -> APIRouter:
    active_settings = settings or InsightSettings.from_env()
    active_service = service or InsightService(
        active_settings, forecast_result_loader=forecast_result_loader
    )
    active_tracker = tracker or ExperimentTracker(
        active_service.store,
        forecast_result_loader=active_service.forecast_result_loader,
    )

    router = APIRouter(prefix="/api/insight/v1", tags=["creator-insight"])

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

    return router
