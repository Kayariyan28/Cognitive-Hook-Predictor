"""Late-fused, failure-isolated forecast evidence orchestration.

Feature workers may describe a clip, but they are never treated as behavioral
predictors here.  Target-specific calibrated heads are a separate layer and
therefore remain unavailable until a validated scorer is explicitly installed.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .adapters import HttpBranchAdapter, build_http_adapters
from .calibrated_heads import CalibratedHeadRuntime
from .execution_registry import WorkerRegistry
from .jobs import (
    BEHAVIORAL_HEAD_KEYS,
    EVIDENCE_SCHEMA_VERSION,
    StageCallback,
)
from .registry import ForecastRegistry
from .workers.asr_whisper import AsrWhisperAdapter
from .workers.ast_audio import AstAudioAdapter
from .workers.local_vjepa21 import LocalVJepa21Adapter
from .workers.measured_audio import MeasuredAudioAdapter
from .workers.nanollava import NanoLlavaSemanticAdapter


LOGGER = logging.getLogger("forecast_orchestrator")
MODEL_PROVIDER_KEYS = (
    "vjepa21",
    "videoLlama21Av",
    "audioModel",
    "semanticModel",
    "measuredAudio",
    "asr",
)
FEATURE_PREFIXES = {
    "vjepa21": "vjepa2_1.",
    "videoLlama21Av": "videollama2.",
    "audioModel": "audio.",
    "semanticModel": "nanollava.",
    "measuredAudio": "measured_audio.",
    "asr": "asr.",
}
FORECAST_FEATURE_SOURCES = {
    "videoMetadata",
    "vjepa21",
    "videoLlama21Av",
    "audioModel",
    "semanticModel",
    "measuredAudio",
    "asr",
    "account",
    "trends",
    "competitors",
}


def _unavailable_provider(reason: str, *, configured: bool) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "configured": configured,
        "forecastContribution": False,
        "reason": reason,
    }


def _availability(adapter: Any) -> dict[str, Any] | None:
    provider = getattr(adapter, "availability", None)
    if not callable(provider):
        return None
    value = provider()
    if not isinstance(value, dict):
        raise ValueError("local adapter availability must be an object")
    if not isinstance(value.get("configured"), bool) or not isinstance(
        value.get("executionAvailable"), bool
    ):
        raise ValueError("local adapter availability flags are invalid")
    return value


def _validated_scalar_features(branch: str, output: Any) -> dict[str, float]:
    raw = getattr(output, "features", None)
    if not isinstance(raw, Mapping):
        raise ValueError("forecast adapter output features must be a mapping")
    prefix = FEATURE_PREFIXES[branch]
    normalized: dict[str, float] = {}
    for name, value in raw.items():
        if (
            not isinstance(name, str)
            or not name.startswith(prefix)
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError("forecast adapter returned an invalid scalar feature")
        normalized[name] = float(value)
    return normalized


def _forecast_use_ledgers(
    behavioral_heads: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Derive result-use provenance from each successful head's locked contract.

    Feature workers remain descriptive (`behavioralOutcome: false`), but it is
    false provenance to call them non-contributing when their exact scalar
    features entered a calibrated head.  Only heads that actually returned a
    probability are included; an attempted/withheld score consumed no feature
    into a reported forecast.
    """

    source_contracts: dict[str, list[dict[str, Any]]] = {}
    context_contracts: list[dict[str, Any]] = []
    for metric_id in BEHAVIORAL_HEAD_KEYS:
        head = behavioral_heads.get(metric_id)
        if not isinstance(head, Mapping) or head.get("status") != "available":
            continue
        feature_contract = head.get("featureContract")
        target_contract = head.get("targetContract")
        if (
            not isinstance(feature_contract, Mapping)
            or set(feature_contract) != {"schemaVersion", "sha256", "features"}
            or not isinstance(feature_contract.get("sha256"), str)
            or not isinstance(feature_contract.get("features"), list)
            or not isinstance(target_contract, Mapping)
        ):
            raise ValueError(
                f"available behavioral head {metric_id} omitted its immutable contracts"
            )

        names_by_source: dict[str, list[str]] = {}
        for feature in feature_contract["features"]:
            if (
                not isinstance(feature, Mapping)
                or set(feature) != {"name", "source"}
                or not isinstance(feature.get("name"), str)
                or feature.get("source") not in FORECAST_FEATURE_SOURCES
            ):
                raise ValueError(
                    f"available behavioral head {metric_id} has an invalid feature ledger"
                )
            names_by_source.setdefault(feature["source"], []).append(feature["name"])

        for source, feature_names in names_by_source.items():
            source_contracts.setdefault(source, []).append(
                {
                    "head": metric_id,
                    "featureContractSha256": feature_contract["sha256"],
                    "featureNames": feature_names,
                }
            )
        context_contracts.append(
            {
                "head": metric_id,
                "targetId": target_contract.get("targetId"),
                "platform": target_contract.get("platform"),
                "domain": target_contract.get("domain"),
                "locale": target_contract.get("locale"),
                "evaluationManifestId": target_contract.get(
                    "evaluationManifestId"
                ),
                "evaluationManifestSha256": target_contract.get(
                    "evaluationManifestSha256"
                ),
            }
        )

    source_ledgers = {
        source: {
            "kind": "calibrated-head-feature-input",
            "usedByHeads": [entry["head"] for entry in contracts],
            "featureContracts": contracts,
        }
        for source, contracts in source_contracts.items()
    }
    context_ledger = None
    if context_contracts:
        context_ledger = {
            "kind": "calibrated-head-eligibility-context",
            "usedByHeads": [entry["head"] for entry in context_contracts],
            "targetContracts": context_contracts,
        }
    return source_ledgers, context_ledger


def _mark_forecast_use(
    evidence: dict[str, Any], ledger: Mapping[str, Any] | None
) -> None:
    """Add the expanded contract only when a score actually consumed evidence.

    Keeping an unused record's existing shape preserves the V1 result contract
    while no production head is release-approved.  A future approved score is
    intentionally schema-visible and must be understood by its consumer.
    """

    if ledger is not None:
        evidence["forecastContribution"] = True
        evidence["forecastUse"] = dict(ledger)


class MultimodalEvidenceOrchestrator:
    """Execute configured workers without conflating evidence and outcomes."""

    def __init__(
        self,
        adapters: Mapping[str, Any],
        head_runtime: CalibratedHeadRuntime | None = None,
    ) -> None:
        unknown = set(adapters) - set(MODEL_PROVIDER_KEYS)
        if unknown:
            raise ValueError(f"unsupported forecast adapters: {sorted(unknown)}")
        self.adapters = dict(adapters)
        self.head_runtime = head_runtime or CalibratedHeadRuntime({})

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        forecast_registry: ForecastRegistry | None = None,
    ) -> "MultimodalEvidenceOrchestrator":
        source = os.environ if environ is None else environ
        worker_registry = WorkerRegistry.from_env(source)
        model_registry = forecast_registry or ForecastRegistry.from_env(source)
        adapters: dict[str, Any] = dict(
            build_http_adapters(worker_registry, environ=source)
        )
        local_vjepa21 = LocalVJepa21Adapter.optional_from_env(source)
        if "vjepa21" not in adapters and local_vjepa21 is not None:
            adapters["vjepa21"] = local_vjepa21
        if "audioModel" not in adapters:
            adapters["audioModel"] = AstAudioAdapter.from_env(source)
        # These job-only lanes retain distinct branch identities in status and
        # results. NanoLLaVA is a keyframe fallback, while measured audio never
        # impersonates the optional learned ``audioModel`` branch.
        adapters["semanticModel"] = NanoLlavaSemanticAdapter.from_env(source)
        adapters["measuredAudio"] = MeasuredAudioAdapter.from_env(source)
        adapters["asr"] = AsrWhisperAdapter.from_env(source)
        return cls(adapters, CalibratedHeadRuntime.from_registry(model_registry))

    def run(
        self,
        video_path: Path,
        input_sha256: str,
        media_probe: Mapping[str, Any],
        context: Mapping[str, Any] | None,
        stage_callback: StageCallback,
    ) -> dict[str, Any]:
        duration_seconds = float(media_probe["durationSeconds"])
        safe_context = dict(context or {})
        optional_providers: dict[str, dict[str, Any]] = {}
        forecast_features: dict[str, float] = {
            "metadata.duration_seconds": duration_seconds,
        }

        for branch in MODEL_PROVIDER_KEYS:
            adapter = self.adapters.get(branch)
            if adapter is None:
                optional_providers[branch] = _unavailable_provider(
                    "No immutable executable worker is configured for this branch.",
                    configured=False,
                )
                continue
            # Readiness may verify a large checkpoint and instantiate a local
            # model. Surface that work before it begins so the durable job does
            # not appear frozen at the generic orchestration stage.
            stage_callback(f"checking-{branch.lower()}")
            try:
                availability = _availability(adapter)
            except Exception:
                availability = {
                    "configured": True,
                    "executionAvailable": False,
                    "reason": "The local provider failed its availability validation.",
                }
            if availability is not None and not availability["executionAvailable"]:
                optional_providers[branch] = {
                    **_unavailable_provider(
                        availability.get("reason")
                        or "The local evidence provider is not executable.",
                        configured=availability["configured"],
                    ),
                    "provider": {
                        key: value
                        for key, value in availability.items()
                        if key not in {"configured", "executionAvailable", "reason"}
                    },
                }
                continue
            stage_callback(f"extracting-{branch.lower()}")
            try:
                output = adapter.run(
                    video_path=video_path,
                    input_sha256=input_sha256,
                    duration_seconds=duration_seconds,
                    context=safe_context,
                )
                validated_features = _validated_scalar_features(branch, output)
            except Exception as exc:  # one model must not erase other valid evidence
                LOGGER.warning("Forecast branch %s failed: %s", branch, type(exc).__name__)
                failed_provider = _unavailable_provider(
                    "The configured worker failed readiness, provenance, or extraction validation for this request.",
                    configured=True,
                )
                if availability is not None:
                    failed_provider["provider"] = {
                        key: value
                        for key, value in availability.items()
                        if key not in {"configured", "executionAvailable", "reason"}
                    }
                optional_providers[branch] = failed_provider
                continue
            optional_providers[branch] = {
                "status": "available",
                "configured": True,
                "forecastContribution": False,
                "behavioralOutcome": False,
                "evidenceKind": output.evidence_kind,
                "result": output.public_value(),
            }
            if availability is not None:
                optional_providers[branch]["provider"] = {
                    key: value
                    for key, value in availability.items()
                    if key not in {"configured", "executionAvailable", "reason"}
                }
            forecast_features.update(validated_features)

        for branch in ("account", "trends", "competitors"):
            optional_providers[branch] = _unavailable_provider(
                "No authenticated, versioned provider is configured for this branch.",
                configured=False,
            )

        stage_callback("assembling-evidence")
        behavioral_heads = self.head_runtime.score_all(forecast_features, safe_context)
        source_ledgers, context_ledger = _forecast_use_ledgers(behavioral_heads)
        video_metadata: dict[str, Any] = {
            "status": "available",
            "source": "server-ffprobe",
            "sha256": input_sha256,
            "sizeBytes": media_probe["sizeBytes"],
            "durationSeconds": duration_seconds,
            "contentType": media_probe.get("contentType"),
        }
        creator_context: dict[str, Any] = {
            "status": "provided" if context is not None else "not-provided",
            "value": context,
            "forecastContribution": False,
        }
        _mark_forecast_use(video_metadata, source_ledgers.get("videoMetadata"))
        if context_ledger is not None:
            if context is None:
                raise ValueError(
                    "an available behavioral head omitted required target context"
                )
            _mark_forecast_use(creator_context, context_ledger)
        for source, ledger in source_ledgers.items():
            if source == "videoMetadata":
                continue
            provider = optional_providers[source]
            if provider.get("status") != "available":
                raise ValueError(
                    f"an available behavioral head used unavailable {source} evidence"
                )
            _mark_forecast_use(provider, ledger)
        return {
            "evidence": {
                "schemaVersion": EVIDENCE_SCHEMA_VERSION,
                "videoMetadata": video_metadata,
                "creatorContext": creator_context,
                "optionalProviders": optional_providers,
                "tribe": {
                    "status": "not-invoked",
                    "forecastContribution": False,
                    "reason": "TRIBE remains an independent cortical prediction path.",
                },
            },
            "behavioralHeads": behavioral_heads,
            "boundaries": [
                "Model-worker outputs are descriptive feature evidence, not audience-outcome probabilities.",
                "A failed or missing branch is never replaced or silently reweighted.",
                "Behavioral values require a separately trained, target-specific, production-calibrated head.",
                "Every evidence input used by an available behavioral head is listed by head and immutable feature-contract digest.",
                "TRIBE cortical output does not contribute to this forecast.",
            ],
        }
