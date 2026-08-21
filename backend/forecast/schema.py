"""Public, score-free creator forecast capability contract."""

from __future__ import annotations

import math
from typing import Any

from .registry import BRANCH_KEYS, HEAD_KEYS, ForecastRegistry


CAPABILITIES_SCHEMA_VERSION = "creator-forecast-capabilities/1"
MINIMUM_DURATION_SECONDS = 10.0
MAXIMUM_DURATION_SECONDS = 60.0

BRANCH_DEFINITIONS: dict[str, dict[str, str]] = {
    "vjepa21": {
        "label": "V-JEPA 2.1 visual representation",
        "role": "optional-temporal-visual-representation",
        "evidenceKind": "learned-visual-representation",
        "missingReason": "No hash-verified V-JEPA 2.1 artifact is registered.",
    },
    "videoLlama21Av": {
        "label": "VideoLLaMA 2.1 audio-visual semantics",
        "role": "optional-structured-semantic-evidence",
        "evidenceKind": "learned-audio-visual-semantics",
        "missingReason": "No hash-verified VideoLLaMA 2.1 AV artifact is registered.",
    },
    "audioModel": {
        "label": "Dedicated speech and music model",
        "role": "optional-learned-audio-evidence",
        "evidenceKind": "learned-speech-music-representation",
        "missingReason": "No task-specific, hash-verified audio model is registered.",
    },
    "semanticModel": {
        "label": "NanoLLaVA keyframe semantics",
        "role": "optional-local-keyframe-semantic-evidence",
        "evidenceKind": "learned-keyframe-semantics",
        "missingReason": "The exact pinned NanoLLaVA snapshot or its executable runtime is not configured.",
    },
    "measuredAudio": {
        "label": "Measured audio descriptors",
        "role": "optional-server-measured-audio-evidence",
        "evidenceKind": "measured-audio-descriptors",
        "missingReason": "The server audio decoder is not configured.",
    },
    "asr": {
        "label": "Spoken-word transcript",
        "role": "optional-measured-speech-transcript",
        "evidenceKind": "measured-speech-transcript",
        "missingReason": "No pinned mlx-whisper revision is configured for the transcript branch.",
    },
    "account": {
        "label": "Account-history context",
        "role": "optional-distribution-context",
        "evidenceKind": "platform-account-history",
        "missingReason": "No versioned account-history adapter is registered.",
    },
    "trends": {
        "label": "Current web-trend context",
        "role": "optional-time-sensitive-context",
        "evidenceKind": "time-stamped-external-retrieval",
        "missingReason": "No versioned trend-retrieval provider is registered.",
    },
    "competitors": {
        "label": "Competing-content context",
        "role": "optional-saturation-context",
        "evidenceKind": "time-stamped-competitor-retrieval",
        "missingReason": "No versioned competing-content provider is registered.",
    },
}

TARGET_DEFINITIONS: dict[str, dict[str, str]] = {
    "hookStrength": {
        "label": "Hook strength",
        "requiredDefinition": "A platform-specific observed hook outcome with an explicit eligible-impression denominator.",
    },
    "retention5Second": {
        "label": "5-second retention",
        "requiredDefinition": "Eligible starts still playing at five seconds divided by eligible starts.",
    },
    "completionProbability": {
        "label": "Completion probability",
        "requiredDefinition": "Eligible starts reaching clip end divided by eligible starts, conditioned on clip duration.",
    },
    "rewatchPotential": {
        "label": "Rewatch probability",
        "requiredDefinition": "Eligible unique viewers with a replay divided by eligible unique viewers within a declared horizon.",
    },
    "sharePotential": {
        "label": "Share probability",
        "requiredDefinition": "Eligible viewers who share divided by eligible viewers within a declared horizon.",
    },
    "predictedEngagement": {
        "label": "Predicted engagement",
        "requiredDefinition": "A declared platform outcome and denominator; no universal engagement target is assumed.",
    },
    "viralityPotential": {
        "label": "Virality potential",
        "requiredDefinition": "Exceeding a declared account/platform baseline threshold within a fixed post horizon.",
    },
}


def validate_duration_seconds(value: Any) -> float:
    """Validate the declared range; this is not an authoritative media probe."""

    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError("durationSeconds must be a finite number")
    duration = float(value)
    if not MINIMUM_DURATION_SECONDS <= duration <= MAXIMUM_DURATION_SECONDS:
        raise ValueError(
            f"durationSeconds must be from {MINIMUM_DURATION_SECONDS:g} to "
            f"{MAXIMUM_DURATION_SECONDS:g} seconds inclusive"
        )
    return duration


def _no_substitution(statement: str) -> dict[str, Any]:
    return {
        "isSubstitute": False,
        "substitutesFor": [],
        "equivalentTo": [],
        "statement": statement,
    }


def _browser_provider(
    *,
    label: str,
    role: str,
    evidence_kind: str,
    source_sha256: str | None,
    per_clip_limit: str,
) -> dict[str, Any]:
    source_verified = isinstance(source_sha256, str) and len(source_sha256) == 64
    return {
        "label": label,
        "role": role,
        "state": "implemented-client-runtime" if source_verified else "source-unverified",
        "configured": source_verified,
        "executionScope": "browser-client-runtime",
        "serverExecutionAvailable": False,
        "perClipAvailability": per_clip_limit,
        "forecastContribution": False,
        "evidenceKind": evidence_kind,
        "usesLearnedModel": False,
        "implementation": {
            "id": "signalframe/browser-local-analysis",
            "formulaVersion": "local-content-outlook-v1",
            "sourceSha256": source_sha256,
        },
        "substitution": _no_substitution(
            "These descriptive measurements are not equivalent to a learned visual, semantic, audio, retention, engagement, or virality model."
        ),
        "readinessDiagnostics": [
            "Implemented in src/analysis.js and executed only in a compatible browser.",
            per_clip_limit,
        ],
    }


def _registered_branch(
    key: str,
    registry: ForecastRegistry,
    runtime: Any = None,
) -> dict[str, Any]:
    definition = BRANCH_DEFINITIONS[key]
    pin = registry.branches.get(key)
    common = {
        "label": definition["label"],
        "role": definition["role"],
        "serverExecutionAvailable": False,
        "forecastContribution": False,
        "evidenceKind": definition["evidenceKind"],
        "substitution": _no_substitution(
            "No built-in or browser evidence is silently relabeled as this provider."
        ),
    }
    if isinstance(runtime, dict):
        configured = runtime.get("configured") is True
        execution_available = runtime.get("executionAvailable")
        readiness_deferred = runtime.get("readinessDeferred") is True
        if execution_available is True:
            state = "ready"
            reason = runtime.get("reason") or "The configured server adapter passed its non-destructive readiness check."
        elif readiness_deferred:
            state = "configured-readiness-deferred"
            reason = runtime.get("reason") or "Readiness is verified per job to avoid loading a model or contacting a remote worker from the status route."
        elif configured:
            state = "configured-runtime-unavailable"
            reason = runtime.get("reason") or "The configured server adapter is not currently executable."
        else:
            state = "not-configured"
            reason = runtime.get("reason") or definition["missingReason"]
        raw_diagnostics = runtime.get("readinessDiagnostics")
        diagnostics = (
            [item for item in raw_diagnostics if isinstance(item, str) and item]
            if isinstance(raw_diagnostics, list)
            else []
        )
        result = {
            **common,
            "state": state,
            "configured": configured,
            "executionScope": runtime.get("executionScope", "forecast-evidence-job"),
            "serverExecutionAvailable": execution_available is True,
            "currentlyReady": execution_available if isinstance(execution_available, bool) else None,
            "reason": reason,
            "readinessDiagnostics": diagnostics,
        }
        provider = runtime.get("provider")
        if isinstance(provider, dict) and provider:
            result["runtimeProvider"] = provider
        if pin is not None:
            result["provenance"] = pin.public_provenance()
        return result
    if pin is None:
        return {
            **common,
            "state": "not-configured",
            "configured": False,
            "reason": definition["missingReason"],
            "readinessDiagnostics": [
                "No registry entry passed immutable revision, artifact, and SHA-256 validation."
            ],
        }
    return {
        **common,
        "state": "registered-artifact-runtime-unavailable",
        "configured": True,
        "provenance": pin.public_provenance(),
        "reason": "The adapter artifact is integrity-verified, but no executable evidence-job adapter is configured for this branch.",
        "readinessDiagnostics": [
            "Registry artifact and immutable provenance validated.",
            "Runtime execution is intentionally disabled until separately implemented and tested.",
        ],
    }


def _tribe_branch(runtime_status: Any) -> dict[str, Any]:
    valid_status = isinstance(runtime_status, dict)
    model = runtime_status.get("model") if valid_status else None
    configured = bool(runtime_status.get("configured")) if valid_status else False
    runtime_state = runtime_status.get("state") if valid_status else "status-unavailable"
    if not isinstance(runtime_state, str):
        runtime_state = "status-unavailable"
    if not isinstance(model, dict):
        model = {}
    extractors = model.get("extractors")
    video_extractor = extractors.get("video") if isinstance(extractors, dict) else None
    diagnostics = [
        "The authoritative readiness source is /api/tribe/v1/status.",
        "The pinned V-JEPA2 extractor is part of the cortical pipeline only.",
    ]
    if not valid_status:
        diagnostics.append("The TRIBE runtime status provider returned no valid status object.")
    elif runtime_status.get("lastError"):
        diagnostics.append("The TRIBE runtime reports a last error; inspect its status endpoint and worker logs.")
    return {
        "label": "Pinned V-JEPA2 → TRIBE v2 cortical pipeline",
        "role": "cortical-only",
        "state": runtime_state,
        "configured": configured,
        "executionScope": "separate-tribe-service",
        "predictionRouteAvailable": configured,
        "currentlyReady": runtime_state in {"ready", "predicting"},
        "forecastContribution": False,
        "predictionType": "average-subject cortical BOLD response",
        "statusEndpoint": "/api/tribe/v1/status",
        "provenance": {
            "model": {
                "id": model.get("id"),
                "revision": model.get("revision"),
                "weightsSha256": model.get("weightsSha256"),
                "inferenceMode": model.get("inferenceMode"),
                "modalitiesUsed": model.get("modalitiesUsed"),
            },
            "videoExtractor": video_extractor,
        },
        "substitution": _no_substitution(
            "Neither V-JEPA2 embeddings nor TRIBE cortical output substitute for V-JEPA 2.1 content features or calibrated behavioral outcomes."
        ),
        "reason": "This path can drive only the validated cortical viewer and cortical descriptors; it is not a performance forecast.",
        "readinessDiagnostics": diagnostics,
    }


def _calibration_head(
    key: str,
    registry: ForecastRegistry,
    executable_head_keys: frozenset[str],
) -> dict[str, Any]:
    definition = TARGET_DEFINITIONS[key]
    pin = registry.heads.get(key)
    common = {
        "label": definition["label"],
        "scoringAvailable": False,
        "outputKind": "calibrated-probability",
        "requiredDefinition": definition["requiredDefinition"],
        "substitutionAllowed": False,
    }
    if pin is None:
        return {
            **common,
            "state": "not-configured",
            "configured": False,
            "reason": "No integrity-verified target-specific model and calibration artifact are registered.",
        }
    if key in executable_head_keys:
        return {
            **common,
            "state": "executable-production-calibrated",
            "configured": True,
            "scoringAvailable": True,
            "provenance": pin.public_provenance(),
            "reason": "This target-specific head is executable inside the evidence-job route; values are returned only in a validated completed job result.",
        }
    return {
        **common,
        "state": "registered-calibrator-runtime-unavailable",
        "configured": True,
        "provenance": pin.public_provenance(),
        "reason": "The target and calibration artifacts are integrity-verified, but no executable scorer was loaded into the evidence-job route.",
    }


def build_capabilities(
    registry: ForecastRegistry,
    *,
    tribe_runtime_status: Any = None,
    browser_analysis_source_sha256: str | None = None,
    forecast_job_runtime_status: Any = None,
) -> dict[str, Any]:
    """Build job-route readiness without executing models or inferring scores."""

    job_runtime = (
        forecast_job_runtime_status
        if isinstance(forecast_job_runtime_status, dict)
        else {}
    )
    runtime_branches = (
        job_runtime.get("branches")
        if isinstance(job_runtime.get("branches"), dict)
        else {}
    )
    raw_executable_heads = job_runtime.get("executableHeadKeys")
    executable_head_keys = frozenset(
        key
        for key in (raw_executable_heads if isinstance(raw_executable_heads, (list, tuple, set, frozenset)) else ())
        if key in HEAD_KEYS and key in registry.heads
    )

    branches = {
        "browserLocalMetadata": _browser_provider(
            label="Browser-decoded video metadata",
            role="descriptive-input-metadata",
            evidence_kind="measured-metadata",
            source_sha256=browser_analysis_source_sha256,
            per_clip_limit="Availability requires browser-decodable video metadata.",
        ),
        "browserLocalVisual": _browser_provider(
            label="Browser-local visual measurements",
            role="descriptive-visual-evidence",
            evidence_kind="measured-pixel-and-frame-change-descriptors",
            source_sha256=browser_analysis_source_sha256,
            per_clip_limit="Availability requires browser-decodable frames and readable canvas pixels.",
        ),
        "browserLocalAudio": _browser_provider(
            label="Browser-local waveform measurements",
            role="descriptive-audio-evidence",
            evidence_kind="measured-waveform-energy-and-dynamics",
            source_sha256=browser_analysis_source_sha256,
            per_clip_limit="Audio may be unavailable when the clip has no decodable track or Web Audio decoding fails.",
        ),
        **{
            key: _registered_branch(key, registry, runtime_branches.get(key))
            for key in BRANCH_KEYS
        },
        "semanticModel": _registered_branch(
            "semanticModel", registry, runtime_branches.get("semanticModel")
        ),
        "measuredAudio": _registered_branch(
            "measuredAudio", registry, runtime_branches.get("measuredAudio")
        ),
        "asr": _registered_branch("asr", registry, runtime_branches.get("asr")),
        "tribeCortical": _tribe_branch(tribe_runtime_status),
    }
    calibration_heads = {
        key: _calibration_head(key, registry, executable_head_keys) for key in HEAD_KEYS
    }
    implemented_browser = [
        key
        for key in ("browserLocalMetadata", "browserLocalVisual", "browserLocalAudio")
        if branches[key]["configured"]
    ]
    return {
        "schemaVersion": CAPABILITIES_SCHEMA_VERSION,
        "service": "creator-forecast",
        "state": "evidence-job-service",
        "scoreGenerationAvailable": bool(executable_head_keys),
        "uploadsAccepted": True,
        "jobExecution": {
            "schemaVersion": "creator-forecast-job-runtime/1",
            "state": "ready",
            "submissionRoute": "/api/forecast/v1/jobs",
            "uploadsAccepted": True,
            "authoritativeMediaProbe": True,
            "durableIdempotency": "client-uuid-job-identity",
            "configuredBranchCount": sum(
                branch.get("configured") is True
                for key, branch in branches.items()
                if key not in {"tribeCortical", "browserLocalMetadata", "browserLocalVisual", "browserLocalAudio"}
            ),
            "readyBranchCount": sum(
                branch.get("serverExecutionAvailable") is True
                for key, branch in branches.items()
                if key not in {"tribeCortical", "browserLocalMetadata", "browserLocalVisual", "browserLocalAudio"}
            ),
            "executableCalibrationHeadCount": len(executable_head_keys),
        },
        "supportedInput": {
            "kind": "short-video",
            "durationSeconds": {
                "minimum": MINIMUM_DURATION_SECONDS,
                "maximum": MAXIMUM_DURATION_SECONDS,
                "inclusive": True,
            },
            "durationValidation": "POST /api/forecast/v1/jobs validates duration with the authoritative server-side ffprobe before model execution.",
        },
        "availableNowProfile": {
            "id": "available-now-evidence/1",
            "implementedDescriptiveProviders": implemented_browser,
            "conditionalIndependentProvider": "tribeCortical",
            "behavioralForecastAvailable": bool(executable_head_keys),
            "statement": "The job route reports configured evidence workers truthfully. Behavioral values appear only when an executable target-specific head returns them in a validated completed result.",
        },
        "registry": {
            "schemaVersion": "creator-forecast-registry/1",
            "source": registry.source,
            "configuredOptionalBranchCount": len(registry.branches),
            "configuredCalibrationHeadCount": len(registry.heads),
        },
        "branches": branches,
        "calibrationHeads": calibration_heads,
        "boundaries": [
            "No score is emitted without a target-specific trained and calibrated model.",
            "Input coverage is not forecast confidence.",
            "Missing branches are not silently replaced or weight-renormalized.",
            "A capability response never unlocks a browser-derived behavioral value; only a validated completed job result can do so.",
            "TRIBE v2 and its pinned V-JEPA2 extractor have forecastContribution false.",
        ],
    }
