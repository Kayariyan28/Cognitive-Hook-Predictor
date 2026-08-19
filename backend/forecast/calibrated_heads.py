"""Release-pinned, target-specific probability heads.

Hashing a JSON file proves only that the file did not change after it was
declared. It does not prove that the file implements the outcome named inside
it. A behavioral score is therefore admitted only when the code-owned target
specification, a release-owned deployment approval, and both hash-verified
artifacts all agree exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .jobs import BEHAVIORAL_HEAD_KEYS, unavailable_behavioral_heads
from .registry import CalibratedHeadPin, ForecastRegistry


HEAD_SCHEMA_VERSION = "creator-forecast-linear-head/2"
CALIBRATION_SCHEMA_VERSION = "creator-forecast-probability-calibration/2"
FEATURE_CONTRACT_SCHEMA_VERSION = "creator-forecast-feature-contract/1"
MAX_ARTIFACT_BYTES = 1024 * 1024
FEATURE_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._][a-z0-9]+)*$")
FORBIDDEN_FEATURE = re.compile(
    r"(^|[._-])(tribe|bold|brain|cortex|cortical|parcel)([._-]|$)", re.I
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

FEATURE_SOURCE_PREFIXES = MappingProxyType(
    {
        "videoMetadata": "metadata.",
        "vjepa21": "vjepa2_1.",
        "videoLlama21Av": "videollama2.",
        "audioModel": "audio.",
        "semanticModel": "nanollava.",
        "measuredAudio": "measured_audio.",
        "account": "account.",
        "trends": "trends.",
        "competitors": "competitors.",
    }
)


class CalibratedHeadError(ValueError):
    pass


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise CalibratedHeadError(f"{label} exceeds the runtime JSON limit")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise CalibratedHeadError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise CalibratedHeadError(f"{label} contains non-finite value {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibratedHeadError(f"{label} is not readable strict JSON") from exc
    if not isinstance(value, dict):
        raise CalibratedHeadError(f"{label} must be a JSON object")
    return value


def _exact(record: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(record) != keys:
        raise CalibratedHeadError(f"{label} has an unsupported shape")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibratedHeadError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise CalibratedHeadError(f"{label} must be finite")
    return result


def _sigmoid(value: float) -> float:
    if value >= 0:
        denominator = 1.0 + math.exp(-value)
        return 1.0 / denominator
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise CalibratedHeadError(f"{label} is not an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CalibratedHeadError(f"{label} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class MetricTargetSpec:
    metric_id: str
    target_id: str
    target_definition: str
    denominator_id: str
    denominator_definition: str
    horizon_kind: str
    horizon_value: float | None

    @property
    def target_definition_sha256(self) -> str:
        return _text_sha256(self.target_definition)

    @property
    def denominator_definition_sha256(self) -> str:
        return _text_sha256(self.denominator_definition)

    def validate_pin(self, pin: CalibratedHeadPin) -> None:
        if (
            pin.target_id != self.target_id
            or pin.target_definition != self.target_definition
            or pin.target_definition_sha256 != self.target_definition_sha256
            or pin.denominator_id != self.denominator_id
            or pin.denominator_definition != self.denominator_definition
            or pin.denominator_definition_sha256
            != self.denominator_definition_sha256
            or pin.horizon.kind != self.horizon_kind
            or pin.horizon.value != self.horizon_value
        ):
            raise CalibratedHeadError(
                f"heads.{self.metric_id} does not match the code-owned target specification"
            )


SUPPORTED_TARGET_SPECS = MappingProxyType(
    {
        "hookStrength": MetricTargetSpec(
            "hookStrength",
            "creator-forecast.hook-continuation-at-2s.v1",
            "An eligible play start with continuous playback reaching 2.000 seconds, excluding seeks, loops, invalid traffic, and autoplay starts without a viewable impression.",
            "creator-forecast.eligible-viewable-play-starts.v1",
            "All eligible viewable play starts for clips with at least 2.000 seconds of playable media, excluding invalid traffic.",
            "seconds-after-start",
            2.0,
        ),
        "retention5Second": MetricTargetSpec(
            "retention5Second",
            "creator-forecast.continuous-retention-at-5s.v1",
            "An eligible play start with continuous playback reaching 5.000 seconds, excluding seeks, loops, and invalid traffic.",
            "creator-forecast.eligible-five-second-play-starts.v1",
            "All eligible play starts for clips with at least 5.000 seconds of playable media, excluding invalid traffic.",
            "seconds-after-start",
            5.0,
        ),
        "completionProbability": MetricTargetSpec(
            "completionProbability",
            "creator-forecast.first-pass-completion.v1",
            "An eligible play start reaching the clip end on its first pass, excluding seeks, loops, and invalid traffic.",
            "creator-forecast.eligible-play-starts.v1",
            "All eligible play starts, excluding invalid traffic.",
            "clip-end",
            None,
        ),
        "rewatchPotential": MetricTargetSpec(
            "rewatchPotential",
            "creator-forecast.unique-viewer-replay-within-24h.v1",
            "An eligible unique viewer beginning at least one additional eligible play of the clip within 24 hours of the first eligible play.",
            "creator-forecast.eligible-unique-viewers.v1",
            "All eligible unique viewers with a first eligible play, excluding invalid traffic.",
            "hours-after-post",
            24.0,
        ),
        "sharePotential": MetricTargetSpec(
            "sharePotential",
            "creator-forecast.unique-viewer-share-within-24h.v1",
            "An eligible unique viewer generating at least one platform-recorded share of the clip within 24 hours after posting.",
            "creator-forecast.eligible-unique-viewers.v1",
            "All eligible unique viewers with an eligible play, excluding invalid traffic.",
            "hours-after-post",
            24.0,
        ),
        "predictedEngagement": MetricTargetSpec(
            "predictedEngagement",
            "creator-forecast.unique-viewer-qualified-engagement-within-24h.v1",
            "An eligible unique viewer generating at least one release-defined qualified engagement event within 24 hours after posting.",
            "creator-forecast.eligible-unique-viewers.v1",
            "All eligible unique viewers with an eligible play, excluding invalid traffic.",
            "hours-after-post",
            24.0,
        ),
        "viralityPotential": MetricTargetSpec(
            "viralityPotential",
            "creator-forecast.account-baseline-exceedance-within-7d.v1",
            "A posted clip exceeding the release-defined account and platform distribution baseline within seven days after posting.",
            "creator-forecast.eligible-posted-clips.v1",
            "All eligible posted clips in the declared evaluation population and period.",
            "days-after-post",
            7.0,
        ),
    }
)

if set(SUPPORTED_TARGET_SPECS) != set(BEHAVIORAL_HEAD_KEYS):
    raise RuntimeError("Every behavioral metric must have one code-owned target spec")


@dataclass(frozen=True, slots=True)
class ApprovedTargetContract:
    """A deployment approval that must be reviewed into application code."""

    metric_id: str
    output_kind: str
    target_id: str
    target_definition_sha256: str
    denominator_id: str
    denominator_definition_sha256: str
    horizon_kind: str
    horizon_value: float | None
    head_id: str
    head_revision: str
    head_artifact_sha256: str
    head_license: str
    feature_contract_sha256: str
    platform: str
    domain: str
    locale: str
    population: str
    training_data_cutoff: str
    calibration_revision: str
    calibration_artifact_sha256: str
    calibration_method: str
    evaluation_manifest_id: str
    evaluation_manifest_sha256: str
    evaluated_at: str
    valid_until: str

    def matches(self, pin: CalibratedHeadPin) -> bool:
        return (
            self.metric_id in SUPPORTED_TARGET_SPECS
            and self.output_kind == "calibrated-probability"
            and self.target_id == pin.target_id
            and self.target_definition_sha256
            == pin.target_definition_sha256
            and self.denominator_id == pin.denominator_id
            and self.denominator_definition_sha256
            == pin.denominator_definition_sha256
            and self.horizon_kind == pin.horizon.kind
            and self.horizon_value == pin.horizon.value
            and self.head_id == pin.id
            and self.head_revision == pin.revision
            and self.head_artifact_sha256 == pin.artifact_sha256
            and self.head_license == pin.license
            and self.feature_contract_sha256 == pin.feature_contract_sha256
            and self.platform == pin.platform
            and self.domain == pin.domain
            and self.locale == pin.locale
            and self.population == pin.population
            and self.training_data_cutoff == pin.training_data_cutoff
            and self.calibration_revision == pin.calibration.revision
            and self.calibration_artifact_sha256
            == pin.calibration.artifact_sha256
            and self.calibration_method == pin.calibration.method
            and self.evaluation_manifest_id
            == pin.calibration.evaluation_manifest_id
            and self.evaluation_manifest_sha256
            == pin.calibration.evaluation_manifest_sha256
            and self.evaluated_at == pin.calibration.evaluated_at
            and self.valid_until == pin.calibration.valid_until
        )

    def validate_freshness(self, now: datetime) -> None:
        evaluated_at = _timestamp(self.evaluated_at, "approved evaluatedAt")
        valid_until = _timestamp(self.valid_until, "approved validUntil")
        moment = now.astimezone(timezone.utc)
        if evaluated_at > moment or moment > valid_until:
            raise CalibratedHeadError(
                f"heads.{self.metric_id} evaluation manifest is not currently fresh"
            )


# Non-empty by supported metric, but deliberately contains no deployment until
# exact model and locked evaluation-manifest pins are reviewed into a release.
APPROVED_TARGET_CONTRACTS = MappingProxyType(
    {metric_id: tuple() for metric_id in BEHAVIORAL_HEAD_KEYS}
)


@dataclass(frozen=True, slots=True)
class FeatureBinding:
    name: str
    source: str

    def public_value(self) -> dict[str, str]:
        return {"name": self.name, "source": self.source}


def _feature_contract_payload(
    bindings: Sequence[FeatureBinding],
) -> dict[str, Any]:
    return {
        "schemaVersion": FEATURE_CONTRACT_SCHEMA_VERSION,
        "features": [binding.public_value() for binding in bindings],
    }


def feature_contract_sha256(features: Sequence[Mapping[str, Any]]) -> str:
    """Return the canonical digest used by model and release manifests."""

    bindings: list[FeatureBinding] = []
    for index, feature in enumerate(features):
        if not isinstance(feature, Mapping) or set(feature) != {"name", "source"}:
            raise CalibratedHeadError(
                f"feature contract item {index} has an unsupported shape"
            )
        name = feature["name"]
        source = feature["source"]
        if (
            not isinstance(name, str)
            or not FEATURE_RE.fullmatch(name)
            or FORBIDDEN_FEATURE.search(name)
            or not isinstance(source, str)
            or source not in FEATURE_SOURCE_PREFIXES
            or not name.startswith(FEATURE_SOURCE_PREFIXES[source])
        ):
            raise CalibratedHeadError(
                f"feature contract item {index} is invalid or source-mismatched"
            )
        bindings.append(FeatureBinding(name=name, source=source))
    if not bindings or len(bindings) > 4096:
        raise CalibratedHeadError("feature contract must contain 1 to 4096 features")
    if len({binding.name for binding in bindings}) != len(bindings):
        raise CalibratedHeadError("feature contract contains duplicate names")
    return _canonical_sha256(_feature_contract_payload(bindings))


def _load_feature_contract(
    value: Any, *, metric_id: str, expected_sha256: str
) -> tuple[FeatureBinding, ...]:
    if not isinstance(value, dict):
        raise CalibratedHeadError(f"heads.{metric_id}.featureContract must be an object")
    _exact(
        value,
        {"schemaVersion", "sha256", "features"},
        f"heads.{metric_id}.featureContract",
    )
    features = value["features"]
    if (
        value["schemaVersion"] != FEATURE_CONTRACT_SCHEMA_VERSION
        or not isinstance(value["sha256"], str)
        or not SHA256_RE.fullmatch(value["sha256"])
        or not isinstance(features, list)
    ):
        raise CalibratedHeadError(f"heads.{metric_id}.featureContract is invalid")
    digest = feature_contract_sha256(features)
    if digest != value["sha256"] or digest != expected_sha256:
        raise CalibratedHeadError(
            f"heads.{metric_id}.featureContract digest does not match the release pin"
        )
    return tuple(
        FeatureBinding(name=feature["name"], source=feature["source"])
        for feature in features
    )


def _expected_artifact_contract(pin: CalibratedHeadPin) -> dict[str, Any]:
    return {
        "targetId": pin.target_id,
        "targetDefinitionSha256": pin.target_definition_sha256,
        "denominatorId": pin.denominator_id,
        "denominatorDefinitionSha256": pin.denominator_definition_sha256,
        "horizon": pin.horizon.public_value(),
        "platform": pin.platform,
        "domain": pin.domain,
        "locale": pin.locale,
        "population": pin.population,
        "trainingDataCutoff": pin.training_data_cutoff,
    }


def _verify_artifact_contract(
    record: Mapping[str, Any], pin: CalibratedHeadPin, label: str
) -> None:
    for key, value in _expected_artifact_contract(pin).items():
        if record.get(key) != value:
            raise CalibratedHeadError(f"{label}.{key} does not match the registry pin")


def _select_approved_contract(
    metric_id: str,
    pin: CalibratedHeadPin,
    approved_contracts: Mapping[str, Sequence[ApprovedTargetContract]],
    now: datetime,
) -> tuple[MetricTargetSpec, ApprovedTargetContract]:
    spec = SUPPORTED_TARGET_SPECS[metric_id]
    spec.validate_pin(pin)
    contracts = approved_contracts.get(metric_id, ())
    approved = next(
        (
            item
            for item in contracts
            if item.metric_id == metric_id and item.matches(pin)
        ),
        None,
    )
    if approved is None:
        raise CalibratedHeadError(
            f"heads.{metric_id} has no exact release-approved target contract"
        )
    approved.validate_freshness(now)
    return spec, approved


@dataclass(frozen=True, slots=True)
class ExecutableHead:
    metric_id: str
    feature_bindings: tuple[FeatureBinding, ...]
    coefficients: tuple[float, ...]
    intercept: float
    calibration_slope: float
    calibration_intercept: float
    pin: CalibratedHeadPin
    target_spec: MetricTargetSpec
    approved_contract: ApprovedTargetContract

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(binding.name for binding in self.feature_bindings)

    @classmethod
    def load(
        cls,
        metric_id: str,
        pin: CalibratedHeadPin,
        *,
        target_spec: MetricTargetSpec,
        approved_contract: ApprovedTargetContract,
    ) -> "ExecutableHead":
        model = _strict_json(pin.artifact_path, f"heads.{metric_id}.artifact")
        calibration = _strict_json(
            pin.calibration.artifact_path, f"heads.{metric_id}.calibration"
        )
        contract_fields = set(_expected_artifact_contract(pin))
        _exact(
            model,
            {
                "schemaVersion",
                "metricId",
                "outputKind",
                "link",
                "missingPolicy",
                "featureContract",
                "coefficients",
                "intercept",
                *contract_fields,
            },
            f"heads.{metric_id}.artifact",
        )
        _exact(
            calibration,
            {
                "schemaVersion",
                "metricId",
                "outputKind",
                "featureContractSha256",
                "method",
                "slope",
                "intercept",
                "clip",
                "evaluation",
                *contract_fields,
            },
            f"heads.{metric_id}.calibration",
        )
        if (
            model["schemaVersion"] != HEAD_SCHEMA_VERSION
            or model["metricId"] != metric_id
            or model["outputKind"] != "calibrated-probability"
            or model["link"] != "logit"
            or model["missingPolicy"] != "withhold"
        ):
            raise CalibratedHeadError(f"heads.{metric_id}.artifact contract mismatch")
        _verify_artifact_contract(model, pin, f"heads.{metric_id}.artifact")
        bindings = _load_feature_contract(
            model["featureContract"],
            metric_id=metric_id,
            expected_sha256=pin.feature_contract_sha256,
        )
        coefficients = model["coefficients"]
        if not isinstance(coefficients, list) or len(coefficients) != len(bindings):
            raise CalibratedHeadError(f"heads.{metric_id} coefficients are invalid")

        if (
            calibration["schemaVersion"] != CALIBRATION_SCHEMA_VERSION
            or calibration["metricId"] != metric_id
            or calibration["outputKind"] != "calibrated-probability"
            or calibration["featureContractSha256"]
            != pin.feature_contract_sha256
            or calibration["method"] != pin.calibration.method
            or calibration["clip"] != {"minimum": 0.0, "maximum": 1.0}
        ):
            raise CalibratedHeadError(
                f"heads.{metric_id}.calibration contract mismatch"
            )
        _verify_artifact_contract(
            calibration, pin, f"heads.{metric_id}.calibration"
        )
        evaluation = calibration["evaluation"]
        if not isinstance(evaluation, dict):
            raise CalibratedHeadError(
                f"heads.{metric_id}.calibration.evaluation must be an object"
            )
        _exact(
            evaluation,
            {
                "manifestId",
                "dataset",
                "evaluatedAt",
                "validUntil",
                "sampleCount",
                "brierScore",
                "expectedCalibrationError",
            },
            f"heads.{metric_id}.calibration.evaluation",
        )
        expected_evaluation = {
            "manifestId": pin.calibration.evaluation_manifest_id,
            "dataset": pin.calibration.evaluation_dataset,
            "evaluatedAt": pin.calibration.evaluated_at,
            "validUntil": pin.calibration.valid_until,
            "sampleCount": pin.calibration.sample_count,
            "brierScore": pin.calibration.brier_score,
            "expectedCalibrationError": pin.calibration.expected_calibration_error,
        }
        if evaluation != expected_evaluation:
            raise CalibratedHeadError(
                f"heads.{metric_id}.calibration evaluation manifest mismatch"
            )
        slope = _finite(calibration["slope"], "calibration.slope")
        if slope <= 0:
            raise CalibratedHeadError("calibration.slope must be positive")
        return cls(
            metric_id=metric_id,
            feature_bindings=bindings,
            coefficients=tuple(
                _finite(value, f"coefficients[{index}]")
                for index, value in enumerate(coefficients)
            ),
            intercept=_finite(model["intercept"], "model.intercept"),
            calibration_slope=slope,
            calibration_intercept=_finite(
                calibration["intercept"], "calibration.intercept"
            ),
            pin=pin,
            target_spec=target_spec,
            approved_contract=approved_contract,
        )

    @staticmethod
    def _unavailable(reason: str) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "value": None,
            "validation": "not-calibrated",
            "reason": reason,
        }

    def score(
        self, features: Mapping[str, Any], context: Mapping[str, Any]
    ) -> dict[str, Any]:
        try:
            self.approved_contract.validate_freshness(datetime.now(timezone.utc))
        except CalibratedHeadError:
            return self._unavailable(
                "The release-pinned evaluation manifest is no longer current."
            )
        for key, expected in {
            "platform": self.pin.platform,
            "domain": self.pin.domain,
            "locale": self.pin.locale,
        }.items():
            value = context.get(key)
            if not isinstance(value, str) or value.strip().lower() != expected:
                return self._unavailable(
                    "The selected platform, domain, or locale does not match this calibrated head."
                )
        values: list[float] = []
        for name in self.feature_names:
            value = features.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                return self._unavailable(
                    "One or more required, provenance-bound features are unavailable."
                )
            values.append(float(value))
        raw_logit = self.intercept + sum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, values, strict=True)
        )
        calibrated = _sigmoid(
            self.calibration_slope * raw_logit + self.calibration_intercept
        )
        feature_contract = _feature_contract_payload(self.feature_bindings)
        feature_contract["sha256"] = self.pin.feature_contract_sha256
        return {
            "status": "available",
            "value": calibrated,
            "validation": "production-calibrated",
            "outputKind": "calibrated-probability",
            "targetDefinition": self.pin.target_definition,
            "denominatorDefinition": self.pin.denominator_definition,
            # Keep these established fields alongside the immutable nested
            # contract so existing result consumers have a safe migration
            # path.  The values are duplicated, never independently derived.
            "platform": self.pin.platform,
            "population": self.pin.population,
            "horizon": self.pin.horizon.public_value(),
            "targetContract": {
                "targetId": self.pin.target_id,
                "targetDefinitionSha256": self.pin.target_definition_sha256,
                "denominatorId": self.pin.denominator_id,
                "denominatorDefinitionSha256": self.pin.denominator_definition_sha256,
                "horizon": self.pin.horizon.public_value(),
                "platform": self.pin.platform,
                "domain": self.pin.domain,
                "locale": self.pin.locale,
                "population": self.pin.population,
                "trainingDataCutoff": self.pin.training_data_cutoff,
                "evaluationManifestId": self.pin.calibration.evaluation_manifest_id,
                "evaluationManifestSha256": self.pin.calibration.evaluation_manifest_sha256,
                "evaluatedAt": self.pin.calibration.evaluated_at,
                "validUntil": self.pin.calibration.valid_until,
            },
            "featureContract": feature_contract,
            "model": self.pin.public_provenance(),
        }


class CalibratedHeadRuntime:
    def __init__(self, heads: Mapping[str, ExecutableHead]) -> None:
        self.heads = dict(heads)

    @classmethod
    def from_registry(cls, registry: ForecastRegistry) -> "CalibratedHeadRuntime":
        return cls._from_registry_with_contracts(
            registry,
            APPROVED_TARGET_CONTRACTS,
            now=datetime.now(timezone.utc),
        )

    @classmethod
    def _from_registry_with_contracts(
        cls,
        registry: ForecastRegistry,
        approved_contracts: Mapping[str, Sequence[ApprovedTargetContract]],
        *,
        now: datetime,
    ) -> "CalibratedHeadRuntime":
        if now.tzinfo is None or now.utcoffset() is None:
            raise CalibratedHeadError("target-contract validation time must be aware")
        unknown = set(approved_contracts) - set(BEHAVIORAL_HEAD_KEYS)
        if unknown:
            raise CalibratedHeadError(
                f"approved target contracts contain unknown metrics {sorted(unknown)}"
            )
        heads: dict[str, ExecutableHead] = {}
        for metric_id, pin in registry.heads.items():
            if metric_id not in SUPPORTED_TARGET_SPECS:
                raise CalibratedHeadError(f"unknown calibrated metric {metric_id}")
            spec, approved = _select_approved_contract(
                metric_id, pin, approved_contracts, now
            )
            heads[metric_id] = ExecutableHead.load(
                metric_id,
                pin,
                target_spec=spec,
                approved_contract=approved,
            )
        return cls(heads)

    def score_all(
        self, features: Mapping[str, Any], context: Mapping[str, Any] | None
    ) -> dict[str, dict[str, Any]]:
        result = unavailable_behavioral_heads(
            "No executable target-specific production-calibrated head is configured for this metric."
        )
        safe_context = dict(context or {})
        for metric_id, head in self.heads.items():
            if metric_id not in BEHAVIORAL_HEAD_KEYS:
                raise CalibratedHeadError(f"unknown calibrated metric {metric_id}")
            result[metric_id] = head.score(features, safe_context)
        return result
