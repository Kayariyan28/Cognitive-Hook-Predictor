"""The insight service: assemble evidence, ask a model, refuse or publish.

Every path through `generate` ends in one of two places: a validated artifact
persisted atomically, or an explicit unavailable state carrying a reasonCode.
There is no third outcome, and nothing partially valid is ever stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any, Callable, Mapping
from uuid import uuid4

from .bundle import (
    BUNDLE_SCHEMA_VERSION,
    BundleUnavailableError,
    assemble_evidence_bundle,
    hook_evidence_card,
)
from .config import InsightSettings
from .prompts.hook_doctor_v1 import PROMPT_TEMPLATE_ID, prompt_hash
from .provenance import LIMITS_STATEMENT, build_provenance, cache_key
from .providers import (
    ProviderExecutionError,
    ProviderUnavailableError,
    all_availability,
    build_provider,
)
from .store import InsightStore, InsightStoreError
from .validation import INSIGHT_SCHEMA_VERSION, validate_insight


LOGGER = logging.getLogger("insight_service")

STATUS_SCHEMA_VERSION = "insight-service-status/1"
REJECTION_SCHEMA_VERSION = "insight-rejection/1"
TRIBE_DESCRIPTOR_SCHEMA_VERSION = "tribe-cortical-descriptors/1"

BOUNDARIES = (
    "Every sentence is generated from cited evidence and re-checked against that evidence.",
    "A hypothesis is an untested heuristic, never a finding and never a forecast.",
    "TRIBE values are predicted average-subject cortical BOLD, not audience behavior.",
    "This lane declares behavioralOutcome false; language-model output can never qualify.",
    "A rejected generation is persisted for inspection and is never cached as a success.",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class InsightRequest:
    forecast_result_id: str
    tribe_result_id: str | None = None
    tribe_descriptors: Mapping[str, Any] | None = None
    hook_only: bool = False


class InsightService:
    def __init__(
        self,
        settings: InsightSettings,
        *,
        forecast_result_loader: Callable[[str], Mapping[str, Any] | None],
        store: InsightStore | None = None,
        provider_factory: Callable[[InsightSettings], Any] | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or InsightStore(settings.insight_dir)
        self._load_forecast_result = forecast_result_loader
        self._provider_factory = provider_factory or build_provider

    # -- status ------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        try:
            selected = self._provider_factory(self.settings).availability().public_value()
        except Exception:  # a misconfigured provider is unavailable, not a crash
            selected = {
                "provider": self.settings.provider,
                "configured": False,
                "available": False,
                "reason": "The configured provider could not be constructed.",
                "model": {"id": None, "revision": None},
            }
        return {
            "schemaVersion": STATUS_SCHEMA_VERSION,
            "service": "creator-insight",
            "state": "ready" if selected.get("available") else "provider-unavailable",
            "generationAvailable": bool(selected.get("available")),
            "provider": selected,
            "providers": all_availability(self.settings),
            "promptTemplate": {"id": PROMPT_TEMPLATE_ID, "hash": prompt_hash()},
            "bundleSchemaVersion": BUNDLE_SCHEMA_VERSION,
            "artifactSchemaVersion": INSIGHT_SCHEMA_VERSION,
            "config": self.settings.public_summary(),
            "behavioralOutcome": False,
            "boundaries": list(BOUNDARIES),
        }

    # -- generation --------------------------------------------------------

    def generate(self, request: InsightRequest) -> tuple[dict[str, Any], bool]:
        """Return `(document, cached)`; the document is an artifact or unavailable."""

        try:
            bundle = self._bundle(request)
        except BundleUnavailableError as exc:
            return self._unavailable("bundle_unavailable", str(exc)), False

        try:
            provider = self._provider_factory(self.settings)
            availability = provider.availability()
        except Exception as exc:
            return (
                self._unavailable(
                    "provider_unavailable",
                    f"The configured provider could not be constructed: {type(exc).__name__}",
                ),
                False,
            )
        if not availability.available:
            return self._unavailable("provider_unavailable", availability.reason), False

        key = cache_key(
            input_evidence_hash=bundle["inputEvidenceHash"],
            provider=self.settings.provider,
            model_revision=availability.model_revision or "",
            temperature=self.settings.temperature,
        )
        cached_id = self.store.cache_lookup(key)
        if cached_id is not None:
            cached = self.store.read_artifact(cached_id)
            if cached is not None:
                return cached, True

        try:
            generation = provider.generate(bundle, hook_only=request.hook_only)
        except ProviderUnavailableError as exc:
            return self._unavailable("provider_unavailable", str(exc)), False
        except ProviderExecutionError as exc:
            return self._unavailable("provider_error", str(exc)), False
        except Exception as exc:
            LOGGER.exception("The insight provider raised an unexpected error")
            return (
                self._unavailable("provider_error", f"provider failure: {type(exc).__name__}"),
                False,
            )

        raw_output = generation.raw_output
        if len(raw_output.encode("utf-8")) > self.settings.max_output_bytes:
            return (
                self._persist_rejection(
                    request,
                    bundle,
                    "output_too_large",
                    f"The provider returned more than {self.settings.max_output_bytes} bytes.",
                ),
                False,
            )

        outcome = validate_insight(raw_output, bundle)
        if outcome["status"] != "valid":
            return (
                self._persist_rejection(
                    request, bundle, outcome["reasonCode"], outcome["detail"]
                ),
                False,
            )

        artifact = self._artifact(request, bundle, outcome["artifact"], generation)
        try:
            self.store.publish_artifact(artifact["insightId"], artifact)
            self.store.cache_store(key, artifact["insightId"])
        except InsightStoreError as exc:
            return (
                self._unavailable(
                    "bundle_unavailable",
                    f"The validated artifact could not be persisted: {exc}",
                ),
                False,
            )
        return artifact, False

    def read_artifact(self, insight_id: str) -> dict[str, Any] | None:
        return self.store.read_artifact(insight_id)

    def read_rejection(self, rejection_id: str) -> dict[str, Any] | None:
        return self.store.read_rejection(rejection_id)

    # -- internals ---------------------------------------------------------

    def _bundle(self, request: InsightRequest) -> dict[str, Any]:
        result = self._load_forecast_result(request.forecast_result_id)
        if result is None:
            raise BundleUnavailableError(
                f"forecast result {request.forecast_result_id} was not found or is not complete"
            )
        descriptors = request.tribe_descriptors
        if request.tribe_result_id and descriptors is None:
            raise BundleUnavailableError(
                "a tribeResultId was supplied without its tribe-cortical-descriptors/1 "
                "document; the backend does not derive cortical descriptors from a tensor"
            )
        if descriptors is not None:
            if descriptors.get("schemaVersion") != TRIBE_DESCRIPTOR_SCHEMA_VERSION:
                raise BundleUnavailableError(
                    f"TRIBE descriptors must be a {TRIBE_DESCRIPTOR_SCHEMA_VERSION} document"
                )
            source = descriptors.get("source")
            declared = source.get("resultId") if isinstance(source, Mapping) else None
            if request.tribe_result_id and declared != request.tribe_result_id:
                raise BundleUnavailableError(
                    "the supplied TRIBE descriptors do not belong to the requested tribeResultId"
                )
        bundle = assemble_evidence_bundle(result, tribe_descriptors=descriptors)
        return hook_evidence_card(bundle) if request.hook_only else bundle

    def _artifact(
        self,
        request: InsightRequest,
        bundle: Mapping[str, Any],
        model_fields: Mapping[str, Any],
        generation: Any,
    ) -> dict[str, Any]:
        provenance = build_provenance(
            settings=self.settings,
            generation=generation,
            input_evidence_hash=bundle["inputEvidenceHash"],
            artifact=model_fields,
            hook_only=request.hook_only,
        )
        return {
            "schemaVersion": INSIGHT_SCHEMA_VERSION,
            "insightId": uuid4().hex,
            "generatedAt": _utc_now(),
            "source": {
                "forecastResultId": request.forecast_result_id,
                "tribeResultId": bundle["source"].get("tribeResultId"),
                "window": bundle["source"].get("window"),
            },
            "hookReport": model_fields["hookReport"],
            "phaseCommentary": model_fields["phaseCommentary"],
            "tribeNotes": model_fields["tribeNotes"],
            "behavioralOutcome": False,
            "limits": LIMITS_STATEMENT,
            "provenance": provenance,
        }

    def _unavailable(self, reason_code: str, detail: Any) -> dict[str, Any]:
        return {
            "unavailable": True,
            "reasonCode": reason_code,
            "detail": detail,
            "behavioralOutcome": False,
        }

    def _persist_rejection(
        self,
        request: InsightRequest,
        bundle: Mapping[str, Any],
        reason_code: str,
        detail: Any,
    ) -> dict[str, Any]:
        rejection_id = uuid4().hex
        record = {
            "schemaVersion": REJECTION_SCHEMA_VERSION,
            "rejectionId": rejection_id,
            "createdAt": _utc_now(),
            "reasonCode": reason_code,
            "detail": detail,
            "source": {
                "forecastResultId": request.forecast_result_id,
                "tribeResultId": bundle["source"].get("tribeResultId"),
                "hookOnly": request.hook_only,
            },
            "inputEvidenceHash": bundle["inputEvidenceHash"],
            "provider": self.settings.provider,
            "promptTemplateId": PROMPT_TEMPLATE_ID,
            "promptHash": prompt_hash(),
        }
        try:
            self.store.publish_rejection(rejection_id, record)
        except InsightStoreError:
            LOGGER.exception("Could not persist an insight rejection record")
            return self._unavailable(reason_code, detail)
        response = self._unavailable(reason_code, detail)
        response["rejectionId"] = rejection_id
        return response
