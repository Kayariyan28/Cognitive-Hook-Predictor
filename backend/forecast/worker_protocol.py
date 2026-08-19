"""Strict protocol shared by creator-forecast model workers.

Workers are evidence producers, not virality or retention predictors.  A
worker may return learned features and structured observations.  Behavioral
values are accepted only from a separately registered target/calibration head.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re
from types import MappingProxyType
from typing import Any, Mapping

from .registry import ArtifactPin, BRANCH_KEYS


WORKER_STATUS_SCHEMA_VERSION = "creator-forecast-worker-status/1"
WORKER_OUTPUT_SCHEMA_VERSION = "creator-forecast-worker-output/1"
WORKER_REQUEST_SCHEMA_VERSION = "creator-forecast-worker-request/1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
FEATURE_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._][a-z0-9]+)*$")

MODEL_BRANCH_KEYS = (
    "vjepa21",
    "videoLlama21Av",
    "audioModel",
    "semanticModel",
)
FEATURE_PREFIXES = {
    "vjepa21": "vjepa2_1.",
    "videoLlama21Av": "videollama2.",
    "audioModel": "audio.",
    "semanticModel": "nanollava.",
}
ALLOWED_EVIDENCE_KINDS = {
    "learned-visual-representation",
    "learned-audio-visual-semantics",
    "learned-speech-music-representation",
    "learned-keyframe-semantics",
    "learned-audioset-label-evidence",
}
FORBIDDEN_FEATURE_FRAGMENTS = (
    "tribe",
    "bold",
    "cortex",
    "brain",
    "parcel",
    "virality",
    "retention",
    "completion",
    "rewatch",
    "share_probability",
)


class WorkerProtocolError(ValueError):
    """A model worker returned an invalid or provenance-mismatched document."""


def _record(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkerProtocolError(f"{label} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any], required: set[str], optional: set[str], label: str
) -> None:
    actual = set(value)
    missing = required - actual
    unknown = actual - required - optional
    if missing or unknown:
        parts: list[str] = []
        if missing:
            parts.append(f"missing {sorted(missing)}")
        if unknown:
            parts.append(f"unknown {sorted(unknown)}")
        raise WorkerProtocolError(f"{label} has invalid fields ({'; '.join(parts)})")


def _text(value: Any, label: str, *, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > maximum
        or any(ord(character) < 32 and character not in "\n\t" for character in value)
    ):
        raise WorkerProtocolError(f"{label} must be a non-empty trimmed string")
    return value


def _sha256(value: Any, label: str) -> str:
    digest = _text(value, label, maximum=64).lower()
    if not SHA256_RE.fullmatch(digest):
        raise WorkerProtocolError(f"{label} must be a SHA-256 digest")
    return digest


def _commit(value: Any, label: str) -> str:
    revision = _text(value, label, maximum=40).lower()
    if not COMMIT_RE.fullmatch(revision):
        raise WorkerProtocolError(f"{label} must be a full immutable commit SHA")
    return revision


def _timestamp(value: Any, label: str) -> str:
    text = _text(value, label, maximum=64)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WorkerProtocolError(f"{label} must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise WorkerProtocolError(f"{label} must include a UTC offset")
    return text


def _finite(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise WorkerProtocolError(f"{label} must be finite")
    return float(value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class WorkerIdentity:
    branch: str
    model_id: str
    model_revision: str
    model_weights_sha256: str
    adapter_id: str
    code_revision: str
    preprocessing_id: str
    preprocessing_sha256: str

    @classmethod
    def from_status(
        cls, value: Any, *, expected_branch: str, expected_pin: ArtifactPin | Any
    ) -> "WorkerIdentity":
        status = _record(value, "worker status")
        _exact_keys(
            status,
            {
                "schemaVersion",
                "service",
                "state",
                "executionAvailable",
                "branch",
                "model",
                "codeRevision",
                "preprocessing",
            },
            {"device", "precision", "lastError"},
            "worker status",
        )
        if status["schemaVersion"] != WORKER_STATUS_SCHEMA_VERSION:
            raise WorkerProtocolError("worker status schemaVersion is unsupported")
        if status["service"] != "creator-forecast-model-worker":
            raise WorkerProtocolError("worker status service is unsupported")
        if status["state"] != "ready" or status["executionAvailable"] is not True:
            raise WorkerProtocolError("worker is not ready for execution")
        if status["branch"] != expected_branch or expected_branch not in MODEL_BRANCH_KEYS:
            raise WorkerProtocolError("worker branch does not match the requested branch")
        model = _record(status["model"], "worker status.model")
        _exact_keys(
            model,
            {"id", "revision", "weightsSha256", "adapterId", "license"},
            set(),
            "worker status.model",
        )
        preprocessing = _record(status["preprocessing"], "worker status.preprocessing")
        _exact_keys(
            preprocessing,
            {"id", "sha256"},
            set(),
            "worker status.preprocessing",
        )
        identity = cls(
            branch=expected_branch,
            model_id=_text(model["id"], "worker status.model.id", maximum=256),
            model_revision=_commit(model["revision"], "worker status.model.revision"),
            model_weights_sha256=_sha256(
                model["weightsSha256"], "worker status.model.weightsSha256"
            ),
            adapter_id=_text(
                model["adapterId"], "worker status.model.adapterId", maximum=128
            ),
            code_revision=_commit(status["codeRevision"], "worker status.codeRevision"),
            preprocessing_id=_text(
                preprocessing["id"], "worker status.preprocessing.id", maximum=256
            ),
            preprocessing_sha256=_sha256(
                preprocessing["sha256"], "worker status.preprocessing.sha256"
            ),
        )
        expected = expected_pin.public_provenance()
        if (
            identity.model_id != expected["id"]
            or identity.model_revision != expected["revision"]
            or identity.model_weights_sha256 != expected["artifactSha256"]
            or identity.adapter_id != expected["adapterId"]
            or model["license"] != expected["license"]
        ):
            raise WorkerProtocolError("worker model provenance does not match the registry")
        expected_code_revision = getattr(expected_pin, "code_revision", None)
        expected_preprocessing_id = getattr(expected_pin, "preprocessing_id", None)
        expected_preprocessing_sha256 = getattr(
            expected_pin, "preprocessing_sha256", None
        )
        if expected_code_revision is not None and (
            identity.code_revision != expected_code_revision
            or identity.preprocessing_id != expected_preprocessing_id
            or identity.preprocessing_sha256 != expected_preprocessing_sha256
        ):
            raise WorkerProtocolError(
                "worker code or preprocessing provenance does not match the execution registry"
            )
        return identity

    def public_value(self) -> dict[str, str]:
        return {
            "branch": self.branch,
            "modelId": self.model_id,
            "modelRevision": self.model_revision,
            "modelWeightsSha256": self.model_weights_sha256,
            "adapterId": self.adapter_id,
            "codeRevision": self.code_revision,
            "preprocessingId": self.preprocessing_id,
            "preprocessingSha256": self.preprocessing_sha256,
        }


@dataclass(frozen=True, slots=True)
class BranchOutput:
    branch: str
    input_sha256: str
    started_at: str
    completed_at: str
    evidence_kind: str
    features: Mapping[str, float]
    observations: tuple[Mapping[str, Any], ...]
    warnings: tuple[str, ...]
    identity: WorkerIdentity

    def public_value(self) -> dict[str, Any]:
        return {
            "schemaVersion": WORKER_OUTPUT_SCHEMA_VERSION,
            "branch": self.branch,
            "inputSha256": self.input_sha256,
            "startedAt": self.started_at,
            "completedAt": self.completed_at,
            "evidenceKind": self.evidence_kind,
            "features": dict(self.features),
            "observations": [dict(item) for item in self.observations],
            "warnings": list(self.warnings),
            "provenance": self.identity.public_value(),
            "behavioralOutcome": False,
        }


def validate_branch_output(
    value: Any,
    *,
    expected_branch: str,
    expected_input_sha256: str,
    identity: WorkerIdentity,
) -> BranchOutput:
    output = _record(value, "worker output")
    _exact_keys(
        output,
        {
            "schemaVersion",
            "branch",
            "inputSha256",
            "startedAt",
            "completedAt",
            "evidenceKind",
            "features",
            "observations",
            "warnings",
            "provenance",
            "behavioralOutcome",
        },
        set(),
        "worker output",
    )
    if output["schemaVersion"] != WORKER_OUTPUT_SCHEMA_VERSION:
        raise WorkerProtocolError("worker output schemaVersion is unsupported")
    if output["branch"] != expected_branch or expected_branch != identity.branch:
        raise WorkerProtocolError("worker output branch does not match the request")
    input_sha256 = _sha256(output["inputSha256"], "worker output.inputSha256")
    if input_sha256 != expected_input_sha256:
        raise WorkerProtocolError("worker output input hash does not match the upload")
    if output["behavioralOutcome"] is not False:
        raise WorkerProtocolError("feature workers cannot claim behavioral outcomes")
    if output["evidenceKind"] not in ALLOWED_EVIDENCE_KINDS:
        raise WorkerProtocolError("worker output evidenceKind is unsupported")
    if output["provenance"] != identity.public_value():
        raise WorkerProtocolError("worker output provenance changed after readiness")

    raw_features = _record(output["features"], "worker output.features")
    if len(raw_features) > 4096:
        raise WorkerProtocolError("worker output has too many scalar features")
    features: dict[str, float] = {}
    for name, raw_value in raw_features.items():
        if not isinstance(name, str) or not FEATURE_NAME_RE.fullmatch(name):
            raise WorkerProtocolError("worker output contains an invalid feature name")
        lowered = name.lower()
        if not lowered.startswith(FEATURE_PREFIXES[expected_branch].lower()):
            raise WorkerProtocolError(
                f"worker output feature {name!r} does not belong to {expected_branch}"
            )
        if any(fragment in lowered for fragment in FORBIDDEN_FEATURE_FRAGMENTS):
            raise WorkerProtocolError(
                f"worker output feature {name!r} crosses the evidence boundary"
            )
        features[name] = _finite(raw_value, f"worker output.features.{name}")

    raw_observations = output["observations"]
    if not isinstance(raw_observations, list) or len(raw_observations) > 256:
        raise WorkerProtocolError("worker output.observations must be a bounded array")
    observations: list[Mapping[str, Any]] = []
    for index, raw_observation in enumerate(raw_observations):
        observation = _record(raw_observation, f"worker output.observations[{index}]")
        _exact_keys(
            observation,
            {"kind", "startTime", "endTime", "text"},
            {"labels"},
            f"worker output.observations[{index}]",
        )
        start = _finite(observation["startTime"], "observation.startTime")
        end = _finite(observation["endTime"], "observation.endTime")
        if start < 0 or end <= start:
            raise WorkerProtocolError("worker observation timing is invalid")
        normalized: dict[str, Any] = {
            "kind": _text(observation["kind"], "observation.kind", maximum=128),
            "startTime": start,
            "endTime": end,
            "text": _text(observation["text"], "observation.text", maximum=4096),
        }
        if "labels" in observation:
            labels = observation["labels"]
            if (
                not isinstance(labels, list)
                or len(labels) > 64
                or any(not isinstance(item, str) or not item.strip() for item in labels)
            ):
                raise WorkerProtocolError("worker observation labels are invalid")
            normalized["labels"] = tuple(item.strip() for item in labels)
        observations.append(MappingProxyType(normalized))

    raw_warnings = output["warnings"]
    if (
        not isinstance(raw_warnings, list)
        or len(raw_warnings) > 64
        or any(not isinstance(item, str) or not item.strip() for item in raw_warnings)
    ):
        raise WorkerProtocolError("worker output.warnings must be a bounded text array")

    return BranchOutput(
        branch=expected_branch,
        input_sha256=input_sha256,
        started_at=_timestamp(output["startedAt"], "worker output.startedAt"),
        completed_at=_timestamp(output["completedAt"], "worker output.completedAt"),
        evidence_kind=output["evidenceKind"],
        features=MappingProxyType(features),
        observations=tuple(observations),
        warnings=tuple(item.strip() for item in raw_warnings),
        identity=identity,
    )


def build_worker_request(
    *,
    branch: str,
    input_sha256: str,
    duration_seconds: float,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    # ``semanticModel`` is intentionally a job-only fallback lane.  It is not
    # added to the frozen capability-v1 branch ledger, where doing so would be
    # mistaken for VideoLLaMA availability by older clients.
    if branch not in MODEL_BRANCH_KEYS or (
        branch != "semanticModel" and branch not in BRANCH_KEYS
    ):
        raise WorkerProtocolError("worker request branch is unsupported")
    duration = _finite(duration_seconds, "worker request.durationSeconds")
    if duration < 10 or duration > 60:
        raise WorkerProtocolError("worker request duration must be from 10 to 60 seconds")
    return {
        "schemaVersion": WORKER_REQUEST_SCHEMA_VERSION,
        "requestId": _sha256(input_sha256, "worker request.inputSha256"),
        "branch": branch,
        "inputSha256": input_sha256,
        "durationSeconds": duration,
        "context": dict(context),
    }
