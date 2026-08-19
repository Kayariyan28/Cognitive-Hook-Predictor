"""Strict, integrity-checked registry for optional forecast providers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

from ..errors import ConfigurationError


REGISTRY_SCHEMA_VERSION = "creator-forecast-registry/1"
REGISTRY_PATH_ENV = "FORECAST_REGISTRY_PATH"
REGISTRY_JSON_ENV = "FORECAST_REGISTRY_JSON"
MAX_REGISTRY_BYTES = 1024 * 1024
HASH_CHUNK_BYTES = 1024 * 1024

BRANCH_KEYS = (
    "vjepa21",
    "videoLlama21Av",
    "audioModel",
    "account",
    "trends",
    "competitors",
)
HEAD_KEYS = (
    "hookStrength",
    "retention5Second",
    "completionProbability",
    "rewatchPotential",
    "sharePotential",
    "predictedEngagement",
    "viralityPotential",
)

ADAPTER_IDS = {
    "vjepa21": "vjepa21-pytorch",
    "videoLlama21Av": "videollama21-av-pytorch",
    "audioModel": "speech-music-model",
    "account": "account-history-v1",
    "trends": "web-trend-evidence-card-v1",
    "competitors": "competitor-evidence-card-v1",
}

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,255}$")
HORIZON_KINDS = {
    "seconds-after-start",
    "clip-end",
    "hours-after-post",
    "days-after-post",
}


def _configuration_error(message: str) -> ConfigurationError:
    return ConfigurationError(f"Forecast registry: {message}")


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _configuration_error(f"{label} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], required: set[str], label: str
) -> None:
    actual = set(value)
    if actual != required:
        missing = sorted(required - actual)
        unknown = sorted(actual - required)
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise _configuration_error(f"{label} has invalid fields ({'; '.join(details)})")


def _require_text(value: Any, label: str, *, identifier: bool = False) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise _configuration_error(f"{label} must be a non-empty trimmed string")
    if any(ord(character) < 32 for character in value):
        raise _configuration_error(f"{label} contains control characters")
    if identifier and not IDENTIFIER_RE.fullmatch(value):
        raise _configuration_error(f"{label} is not a stable identifier")
    return value


def _require_commit(value: Any, label: str) -> str:
    revision = _require_text(value, label).lower()
    if not COMMIT_RE.fullmatch(revision):
        raise _configuration_error(
            f"{label} must be a full immutable 40-character commit SHA"
        )
    return revision


def _require_sha256(value: Any, label: str) -> str:
    digest = _require_text(value, label).lower()
    if not SHA256_RE.fullmatch(digest):
        raise _configuration_error(f"{label} must be a 64-character SHA-256")
    return digest


def _require_timestamp(value: Any, label: str) -> str:
    timestamp = _require_text(value, label)
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _configuration_error(f"{label} must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _configuration_error(f"{label} must include a UTC offset")
    return timestamp


def _timestamp_value(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_definition_hash(
    definition: str, raw_digest: Any, label: str
) -> str:
    digest = _require_sha256(raw_digest, label)
    if digest != _text_sha256(definition):
        raise _configuration_error(
            f"{label} does not match the exact UTF-8 definition text"
        )
    return digest


def _require_probability_metric(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise _configuration_error(f"{label} must be a finite value from 0 to 1")
    return float(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_artifact_path(
    raw_path: Any,
    expected_sha256: str,
    label: str,
    base_directory: Path,
) -> Path:
    path_text = _require_text(raw_path, f"{label}.artifactPath")
    unresolved = Path(path_text).expanduser()
    path = (
        unresolved if unresolved.is_absolute() else base_directory / unresolved
    ).resolve()
    if not path.is_file():
        raise _configuration_error(f"{label}.artifactPath is not a readable file")
    try:
        actual_sha256 = _sha256_file(path)
    except OSError as exc:
        raise _configuration_error(f"{label}.artifactPath could not be read") from exc
    if actual_sha256 != expected_sha256:
        raise _configuration_error(f"{label} artifact failed its SHA-256 check")
    return path


def _json_without_duplicate_keys(payload: str, label: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _configuration_error(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(payload, object_pairs_hook=object_pairs)
    except ConfigurationError:
        raise
    except json.JSONDecodeError as exc:
        raise _configuration_error(f"{label} is not valid JSON") from exc
    return _require_object(parsed, label)


@dataclass(frozen=True, slots=True)
class ArtifactPin:
    id: str
    revision: str
    artifact_path: Path
    artifact_sha256: str
    license: str
    adapter_id: str

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        branch_key: str,
        label: str,
        base_directory: Path,
    ) -> "ArtifactPin":
        record = _require_object(value, label)
        _require_exact_keys(
            record,
            {
                "id",
                "revision",
                "artifactPath",
                "artifactSha256",
                "license",
                "adapterId",
            },
            label,
        )
        expected_adapter = ADAPTER_IDS[branch_key]
        if record["adapterId"] != expected_adapter:
            raise _configuration_error(
                f"{label}.adapterId must be {expected_adapter}"
            )
        artifact_sha256 = _require_sha256(
            record["artifactSha256"], f"{label}.artifactSha256"
        )
        return cls(
            id=_require_text(record["id"], f"{label}.id", identifier=True),
            revision=_require_commit(record["revision"], f"{label}.revision"),
            artifact_path=_verified_artifact_path(
                record["artifactPath"],
                artifact_sha256,
                label,
                base_directory,
            ),
            artifact_sha256=artifact_sha256,
            license=_require_text(record["license"], f"{label}.license"),
            adapter_id=expected_adapter,
        )

    def public_provenance(self) -> dict[str, str]:
        return {
            "id": self.id,
            "revision": self.revision,
            "artifactSha256": self.artifact_sha256,
            "license": self.license,
            "adapterId": self.adapter_id,
        }


@dataclass(frozen=True, slots=True)
class ForecastHorizon:
    kind: str
    value: float | None

    @classmethod
    def from_mapping(cls, value: Any, *, label: str) -> "ForecastHorizon":
        record = _require_object(value, label)
        _require_exact_keys(record, {"kind", "value"}, label)
        kind = _require_text(record["kind"], f"{label}.kind")
        if kind not in HORIZON_KINDS:
            raise _configuration_error(
                f"{label}.kind must be one of {sorted(HORIZON_KINDS)}"
            )
        raw_value = record["value"]
        if kind == "clip-end":
            if raw_value is not None:
                raise _configuration_error(f"{label}.value must be null for clip-end")
            return cls(kind=kind, value=None)
        if (
            not isinstance(raw_value, (int, float))
            or isinstance(raw_value, bool)
            or not math.isfinite(float(raw_value))
            or float(raw_value) <= 0
        ):
            raise _configuration_error(f"{label}.value must be a positive finite number")
        return cls(kind=kind, value=float(raw_value))

    def public_value(self) -> dict[str, Any]:
        return {"kind": self.kind, "value": self.value}


@dataclass(frozen=True, slots=True)
class CalibrationEvidence:
    revision: str
    artifact_path: Path
    artifact_sha256: str
    method: str
    evaluation_manifest_id: str
    evaluation_manifest_sha256: str
    evaluation_dataset: str
    evaluated_at: str
    valid_until: str
    sample_count: int
    brier_score: float
    expected_calibration_error: float

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        label: str,
        base_directory: Path,
    ) -> "CalibrationEvidence":
        record = _require_object(value, label)
        _require_exact_keys(
            record,
            {
                "revision",
                "artifactPath",
                "artifactSha256",
                "method",
                "evaluationManifestId",
                "evaluationManifestSha256",
                "evaluationDataset",
                "evaluatedAt",
                "validUntil",
                "sampleCount",
                "brierScore",
                "expectedCalibrationError",
            },
            label,
        )
        artifact_sha256 = _require_sha256(
            record["artifactSha256"], f"{label}.artifactSha256"
        )
        sample_count = record["sampleCount"]
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count <= 0
        ):
            raise _configuration_error(f"{label}.sampleCount must be a positive integer")
        evaluated_at = _require_timestamp(
            record["evaluatedAt"], f"{label}.evaluatedAt"
        )
        valid_until = _require_timestamp(
            record["validUntil"], f"{label}.validUntil"
        )
        if _timestamp_value(valid_until) <= _timestamp_value(evaluated_at):
            raise _configuration_error(
                f"{label}.validUntil must be later than evaluatedAt"
            )
        evaluation_manifest_id = _require_text(
            record["evaluationManifestId"],
            f"{label}.evaluationManifestId",
            identifier=True,
        )
        evaluation_dataset = _require_text(
            record["evaluationDataset"], f"{label}.evaluationDataset"
        )
        brier_score = _require_probability_metric(
            record["brierScore"], f"{label}.brierScore"
        )
        expected_calibration_error = _require_probability_metric(
            record["expectedCalibrationError"],
            f"{label}.expectedCalibrationError",
        )
        manifest_digest = _require_sha256(
            record["evaluationManifestSha256"],
            f"{label}.evaluationManifestSha256",
        )
        manifest_payload = {
            "manifestId": evaluation_manifest_id,
            "dataset": evaluation_dataset,
            "evaluatedAt": evaluated_at,
            "validUntil": valid_until,
            "sampleCount": sample_count,
            "brierScore": brier_score,
            "expectedCalibrationError": expected_calibration_error,
        }
        if manifest_digest != _canonical_sha256(manifest_payload):
            raise _configuration_error(
                f"{label}.evaluationManifestSha256 does not match the exact evaluation manifest"
            )
        return cls(
            revision=_require_commit(record["revision"], f"{label}.revision"),
            artifact_path=_verified_artifact_path(
                record["artifactPath"],
                artifact_sha256,
                label,
                base_directory,
            ),
            artifact_sha256=artifact_sha256,
            method=_require_text(record["method"], f"{label}.method"),
            evaluation_manifest_id=evaluation_manifest_id,
            evaluation_manifest_sha256=manifest_digest,
            evaluation_dataset=evaluation_dataset,
            evaluated_at=evaluated_at,
            valid_until=valid_until,
            sample_count=sample_count,
            brier_score=brier_score,
            expected_calibration_error=expected_calibration_error,
        )

    def public_evidence(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "artifactSha256": self.artifact_sha256,
            "evaluationManifestId": self.evaluation_manifest_id,
            "evaluationManifestSha256": self.evaluation_manifest_sha256,
            "method": self.method,
            "evaluationDataset": self.evaluation_dataset,
            "evaluatedAt": self.evaluated_at,
            "validUntil": self.valid_until,
            "sampleCount": self.sample_count,
            "brierScore": self.brier_score,
            "expectedCalibrationError": self.expected_calibration_error,
        }


@dataclass(frozen=True, slots=True)
class CalibratedHeadPin:
    id: str
    revision: str
    artifact_path: Path
    artifact_sha256: str
    license: str
    target_id: str
    target_definition: str
    target_definition_sha256: str
    denominator_id: str
    denominator_definition: str
    denominator_definition_sha256: str
    platform: str
    domain: str
    locale: str
    population: str
    horizon: ForecastHorizon
    feature_contract_sha256: str
    training_data_cutoff: str
    calibration: CalibrationEvidence

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        label: str,
        base_directory: Path,
    ) -> "CalibratedHeadPin":
        record = _require_object(value, label)
        _require_exact_keys(
            record,
            {
                "id",
                "revision",
                "artifactPath",
                "artifactSha256",
                "license",
                "outputKind",
                "targetId",
                "targetDefinition",
                "targetDefinitionSha256",
                "denominatorId",
                "denominatorDefinition",
                "denominatorDefinitionSha256",
                "platform",
                "domain",
                "locale",
                "population",
                "horizon",
                "featureContractSha256",
                "trainingDataCutoff",
                "calibration",
            },
            label,
        )
        if record["outputKind"] != "calibrated-probability":
            raise _configuration_error(
                f"{label}.outputKind must be calibrated-probability"
            )
        artifact_sha256 = _require_sha256(
            record["artifactSha256"], f"{label}.artifactSha256"
        )
        target_definition = _require_text(
            record["targetDefinition"], f"{label}.targetDefinition"
        )
        denominator_definition = _require_text(
            record["denominatorDefinition"], f"{label}.denominatorDefinition"
        )
        training_data_cutoff = _require_timestamp(
            record["trainingDataCutoff"], f"{label}.trainingDataCutoff"
        )
        calibration = CalibrationEvidence.from_mapping(
            record["calibration"],
            label=f"{label}.calibration",
            base_directory=base_directory,
        )
        if _timestamp_value(training_data_cutoff) > _timestamp_value(
            calibration.evaluated_at
        ):
            raise _configuration_error(
                f"{label}.trainingDataCutoff cannot be later than calibration.evaluatedAt"
            )
        return cls(
            id=_require_text(record["id"], f"{label}.id", identifier=True),
            revision=_require_commit(record["revision"], f"{label}.revision"),
            artifact_path=_verified_artifact_path(
                record["artifactPath"], artifact_sha256, label, base_directory
            ),
            artifact_sha256=artifact_sha256,
            license=_require_text(record["license"], f"{label}.license"),
            target_id=_require_text(
                record["targetId"], f"{label}.targetId", identifier=True
            ),
            target_definition=target_definition,
            target_definition_sha256=_require_definition_hash(
                target_definition,
                record["targetDefinitionSha256"],
                f"{label}.targetDefinitionSha256",
            ),
            denominator_id=_require_text(
                record["denominatorId"],
                f"{label}.denominatorId",
                identifier=True,
            ),
            denominator_definition=denominator_definition,
            denominator_definition_sha256=_require_definition_hash(
                denominator_definition,
                record["denominatorDefinitionSha256"],
                f"{label}.denominatorDefinitionSha256",
            ),
            platform=_require_text(
                record["platform"], f"{label}.platform", identifier=True
            ).lower(),
            domain=_require_text(
                record["domain"], f"{label}.domain", identifier=True
            ).lower(),
            locale=_require_text(
                record["locale"], f"{label}.locale", identifier=True
            ).lower(),
            population=_require_text(record["population"], f"{label}.population"),
            horizon=ForecastHorizon.from_mapping(
                record["horizon"], label=f"{label}.horizon"
            ),
            feature_contract_sha256=_require_sha256(
                record["featureContractSha256"],
                f"{label}.featureContractSha256",
            ),
            training_data_cutoff=training_data_cutoff,
            calibration=calibration,
        )

    def public_provenance(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "revision": self.revision,
            "artifactSha256": self.artifact_sha256,
            "license": self.license,
            "outputKind": "calibrated-probability",
            "targetId": self.target_id,
            "targetDefinition": self.target_definition,
            "targetDefinitionSha256": self.target_definition_sha256,
            "denominatorId": self.denominator_id,
            "denominatorDefinition": self.denominator_definition,
            "denominatorDefinitionSha256": self.denominator_definition_sha256,
            "platform": self.platform,
            "domain": self.domain,
            "locale": self.locale,
            "population": self.population,
            "horizon": self.horizon.public_value(),
            "featureContractSha256": self.feature_contract_sha256,
            "trainingDataCutoff": self.training_data_cutoff,
            "calibration": self.calibration.public_evidence(),
        }


@dataclass(frozen=True, slots=True)
class ForecastRegistry:
    branches: Mapping[str, ArtifactPin]
    heads: Mapping[str, CalibratedHeadPin]
    source: str

    @classmethod
    def empty(cls) -> "ForecastRegistry":
        return cls(MappingProxyType({}), MappingProxyType({}), "none")

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        base_directory: Path,
        source: str = "memory",
    ) -> "ForecastRegistry":
        root = _require_object(value, "root")
        _require_exact_keys(root, {"schemaVersion", "branches", "heads"}, "root")
        if root["schemaVersion"] != REGISTRY_SCHEMA_VERSION:
            raise _configuration_error(
                f"schemaVersion must be {REGISTRY_SCHEMA_VERSION}"
            )
        raw_branches = _require_object(root["branches"], "branches")
        raw_heads = _require_object(root["heads"], "heads")
        unknown_branches = sorted(set(raw_branches) - set(BRANCH_KEYS))
        unknown_heads = sorted(set(raw_heads) - set(HEAD_KEYS))
        if unknown_branches:
            raise _configuration_error(f"unknown branch keys {unknown_branches}")
        if unknown_heads:
            raise _configuration_error(f"unknown calibration head keys {unknown_heads}")

        branches = {
            key: ArtifactPin.from_mapping(
                raw_branches[key],
                branch_key=key,
                label=f"branches.{key}",
                base_directory=base_directory,
            )
            for key in BRANCH_KEYS
            if key in raw_branches
        }
        heads = {
            key: CalibratedHeadPin.from_mapping(
                raw_heads[key],
                label=f"heads.{key}",
                base_directory=base_directory,
            )
            for key in HEAD_KEYS
            if key in raw_heads
        }
        return cls(
            MappingProxyType(branches), MappingProxyType(heads), source
        )

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "ForecastRegistry":
        environment = os.environ if environ is None else environ
        inline = str(environment.get(REGISTRY_JSON_ENV, "")).strip()
        path_value = str(environment.get(REGISTRY_PATH_ENV, "")).strip()
        if inline and path_value:
            raise _configuration_error(
                f"set only one of {REGISTRY_JSON_ENV} or {REGISTRY_PATH_ENV}"
            )
        if not inline and not path_value:
            return cls.empty()
        if inline:
            if len(inline.encode("utf-8")) > MAX_REGISTRY_BYTES:
                raise _configuration_error(f"{REGISTRY_JSON_ENV} is too large")
            return cls.from_mapping(
                _json_without_duplicate_keys(inline, REGISTRY_JSON_ENV),
                base_directory=Path.cwd(),
                source="environment-json",
            )

        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise _configuration_error(f"{REGISTRY_PATH_ENV} is not a readable file")
        if path.stat().st_size > MAX_REGISTRY_BYTES:
            raise _configuration_error(f"{REGISTRY_PATH_ENV} is too large")
        try:
            payload = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise _configuration_error(f"{REGISTRY_PATH_ENV} could not be read") from exc
        return cls.from_mapping(
            _json_without_duplicate_keys(payload, REGISTRY_PATH_ENV),
            base_directory=path.parent,
            source="registry-file",
        )
