from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from backend.errors import ConfigurationError
from backend.forecast.calibrated_heads import (
    CALIBRATION_SCHEMA_VERSION,
    FEATURE_CONTRACT_SCHEMA_VERSION,
    HEAD_SCHEMA_VERSION,
    ApprovedTargetContract,
    CalibratedHeadError,
    CalibratedHeadRuntime,
    SUPPORTED_TARGET_SPECS,
    feature_contract_sha256,
)
from backend.forecast.registry import ForecastRegistry


METRIC = "retention5Second"
NOW = datetime(2026, 8, 19, tzinfo=timezone.utc)
EVALUATED_AT = "2026-08-01T00:00:00Z"
VALID_UNTIL = "2026-12-31T23:59:59Z"


def write_json(path: Path, value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def head_fixture(
    directory: Path,
    *,
    feature_name: str = "metadata.duration_seconds",
    feature_source: str = "videoMetadata",
    registry_target_definition: str | None = None,
    artifact_target_definition_sha256: str | None = None,
    evaluation_dataset_in_artifact: str = "locked-account-disjoint-test",
    evaluation_dataset_in_registry: str = "locked-account-disjoint-test",
    registry_evaluation_manifest_sha256: str | None = None,
) -> tuple[ForecastRegistry, dict[str, tuple[ApprovedTargetContract, ...]]]:
    spec = SUPPORTED_TARGET_SPECS[METRIC]
    target_definition = registry_target_definition or spec.target_definition
    target_definition_sha256 = text_sha256(target_definition)
    denominator_definition_sha256 = text_sha256(spec.denominator_definition)
    feature_rows = [{"name": feature_name, "source": feature_source}]
    feature_sha256 = feature_contract_sha256(feature_rows)
    horizon = {"kind": spec.horizon_kind, "value": spec.horizon_value}
    common = {
        "targetId": spec.target_id,
        "targetDefinitionSha256": artifact_target_definition_sha256
        or target_definition_sha256,
        "denominatorId": spec.denominator_id,
        "denominatorDefinitionSha256": denominator_definition_sha256,
        "horizon": horizon,
        "platform": "youtube-shorts",
        "domain": "short-form-video",
        "locale": "en-us",
        "population": "held-out creator accounts in the United States",
        "trainingDataCutoff": "2026-07-01T00:00:00Z",
    }
    evaluation = {
        "manifestId": "test/retention5/evaluation-v1",
        "dataset": evaluation_dataset_in_artifact,
        "evaluatedAt": EVALUATED_AT,
        "validUntil": VALID_UNTIL,
        "sampleCount": 1000,
        "brierScore": 0.12,
        "expectedCalibrationError": 0.03,
    }
    model_path = directory / "head.json"
    calibration_path = directory / "calibration.json"
    model_hash = write_json(
        model_path,
        {
            "schemaVersion": HEAD_SCHEMA_VERSION,
            "metricId": METRIC,
            "outputKind": "calibrated-probability",
            "link": "logit",
            "missingPolicy": "withhold",
            "featureContract": {
                "schemaVersion": FEATURE_CONTRACT_SCHEMA_VERSION,
                "sha256": feature_sha256,
                "features": feature_rows,
            },
            "coefficients": [0.1],
            "intercept": -1.0,
            **common,
        },
    )
    calibration_hash = write_json(
        calibration_path,
        {
            "schemaVersion": CALIBRATION_SCHEMA_VERSION,
            "metricId": METRIC,
            "outputKind": "calibrated-probability",
            "featureContractSha256": feature_sha256,
            "method": "platt-logit",
            "slope": 1.0,
            "intercept": 0.0,
            "clip": {"minimum": 0.0, "maximum": 1.0},
            "evaluation": evaluation,
            **common,
        },
    )
    registry = ForecastRegistry.from_mapping(
        {
            "schemaVersion": "creator-forecast-registry/1",
            "branches": {},
            "heads": {
                METRIC: {
                    "id": "example/retention-head",
                    "revision": "1" * 40,
                    "artifactPath": str(model_path),
                    "artifactSha256": model_hash,
                    "license": "test-only",
                    "outputKind": "calibrated-probability",
                    "targetId": spec.target_id,
                    "targetDefinition": target_definition,
                    "targetDefinitionSha256": target_definition_sha256,
                    "denominatorId": spec.denominator_id,
                    "denominatorDefinition": spec.denominator_definition,
                    "denominatorDefinitionSha256": denominator_definition_sha256,
                    "platform": "youtube-shorts",
                    "domain": "short-form-video",
                    "locale": "en-us",
                    "population": "held-out creator accounts in the United States",
                    "horizon": horizon,
                    "featureContractSha256": feature_sha256,
                    "trainingDataCutoff": "2026-07-01T00:00:00Z",
                    "calibration": {
                        "revision": "2" * 40,
                        "artifactPath": str(calibration_path),
                        "artifactSha256": calibration_hash,
                        "method": "platt-logit",
                        "evaluationManifestId": "test/retention5/evaluation-v1",
                        "evaluationManifestSha256": canonical_sha256(
                            {
                                **evaluation,
                                "dataset": evaluation_dataset_in_registry,
                            }
                        )
                        if registry_evaluation_manifest_sha256 is None
                        else registry_evaluation_manifest_sha256,
                        "evaluationDataset": evaluation_dataset_in_registry,
                        "evaluatedAt": EVALUATED_AT,
                        "validUntil": VALID_UNTIL,
                        "sampleCount": 1000,
                        "brierScore": 0.12,
                        "expectedCalibrationError": 0.03,
                    },
                }
            },
        },
        base_directory=directory,
    )
    pin = registry.heads[METRIC]
    approval = ApprovedTargetContract(
        metric_id=METRIC,
        output_kind="calibrated-probability",
        target_id=spec.target_id,
        target_definition_sha256=spec.target_definition_sha256,
        denominator_id=spec.denominator_id,
        denominator_definition_sha256=spec.denominator_definition_sha256,
        horizon_kind=spec.horizon_kind,
        horizon_value=spec.horizon_value,
        head_id="example/retention-head",
        head_revision="1" * 40,
        head_artifact_sha256=model_hash,
        head_license="test-only",
        feature_contract_sha256=feature_sha256,
        platform="youtube-shorts",
        domain="short-form-video",
        locale="en-us",
        population="held-out creator accounts in the United States",
        training_data_cutoff="2026-07-01T00:00:00Z",
        calibration_revision="2" * 40,
        calibration_artifact_sha256=pin.calibration.artifact_sha256,
        calibration_method="platt-logit",
        evaluation_manifest_id="test/retention5/evaluation-v1",
        evaluation_manifest_sha256=pin.calibration.evaluation_manifest_sha256,
        evaluated_at=EVALUATED_AT,
        valid_until=VALID_UNTIL,
    )
    return registry, {METRIC: (approval,)}


def approved_runtime(
    registry: ForecastRegistry,
    approvals: dict[str, tuple[ApprovedTargetContract, ...]],
    *,
    now: datetime = NOW,
) -> CalibratedHeadRuntime:
    return CalibratedHeadRuntime._from_registry_with_contracts(
        registry, approvals, now=now
    )


class CalibratedHeadTests(unittest.TestCase):
    def test_exact_release_pinned_head_scores_only_under_exact_context(self):
        with tempfile.TemporaryDirectory() as directory:
            registry, approvals = head_fixture(Path(directory))
            runtime = approved_runtime(registry, approvals)
            result = runtime.score_all(
                {"metadata.duration_seconds": 20.0},
                {
                    "platform": "youtube-shorts",
                    "domain": "short-form-video",
                    "locale": "en-us",
                },
            )
        head = result[METRIC]
        self.assertEqual(head["status"], "available")
        self.assertEqual(head["validation"], "production-calibrated")
        self.assertAlmostEqual(head["value"], 0.7310585786300049)
        self.assertEqual(
            head["targetContract"]["targetId"],
            SUPPORTED_TARGET_SPECS[METRIC].target_id,
        )
        self.assertEqual(
            head["featureContract"]["features"],
            [{"name": "metadata.duration_seconds", "source": "videoMetadata"}],
        )
        self.assertEqual(result["sharePotential"]["status"], "unavailable")

    def test_missing_feature_or_context_contract_withholds(self):
        with tempfile.TemporaryDirectory() as directory:
            registry, approvals = head_fixture(Path(directory))
            runtime = approved_runtime(registry, approvals)
            valid_context = {
                "platform": "youtube-shorts",
                "domain": "short-form-video",
                "locale": "en-us",
            }
            missing = runtime.score_all({}, valid_context)
            for key, wrong in (
                ("platform", "tiktok"),
                ("domain", "long-form-video"),
                ("locale", "en-gb"),
            ):
                context = {**valid_context, key: wrong}
                mismatch = runtime.score_all(
                    {"metadata.duration_seconds": 20.0}, context
                )
                self.assertEqual(mismatch[METRIC]["status"], "unavailable")
        self.assertEqual(missing[METRIC]["status"], "unavailable")

    def test_self_declared_registry_has_no_authority_to_unlock_score(self):
        with tempfile.TemporaryDirectory() as directory:
            registry, _approvals = head_fixture(Path(directory))
            with self.assertRaisesRegex(
                CalibratedHeadError, "no exact release-approved target contract"
            ):
                CalibratedHeadRuntime.from_registry(registry)

    def test_forged_target_or_approval_pin_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            forged_definition = (
                "A self-declared outcome that is not the release-owned target."
            )
            registry, approvals = head_fixture(
                Path(directory), registry_target_definition=forged_definition
            )
            with self.assertRaisesRegex(CalibratedHeadError, "code-owned target"):
                approved_runtime(registry, approvals)

        with tempfile.TemporaryDirectory() as directory:
            registry, approvals = head_fixture(Path(directory))
            exact = approvals[METRIC][0]
            forged_approval = replace(
                exact, evaluation_manifest_sha256="f" * 64
            )
            with self.assertRaisesRegex(
                CalibratedHeadError, "no exact release-approved target contract"
            ):
                approved_runtime(registry, {METRIC: (forged_approval,)})

            forged_calibration = replace(
                exact, calibration_artifact_sha256="a" * 64
            )
            with self.assertRaisesRegex(
                CalibratedHeadError, "no exact release-approved target contract"
            ):
                approved_runtime(registry, {METRIC: (forged_calibration,)})

            forged_semantics = replace(
                exact, target_definition_sha256="e" * 64
            )
            with self.assertRaisesRegex(
                CalibratedHeadError, "no exact release-approved target contract"
            ):
                approved_runtime(registry, {METRIC: (forged_semantics,)})

            cross_metric_approval = replace(exact, metric_id="hookStrength")
            with self.assertRaisesRegex(
                CalibratedHeadError, "no exact release-approved target contract"
            ):
                approved_runtime(
                    registry, {METRIC: (cross_metric_approval,)}
                )

    def test_artifact_contract_and_evaluation_manifest_must_match_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ConfigurationError, "evaluationManifestSha256"
            ):
                head_fixture(
                    Path(directory),
                    registry_evaluation_manifest_sha256="0" * 64,
                )

        with tempfile.TemporaryDirectory() as directory:
            spec = SUPPORTED_TARGET_SPECS[METRIC]
            registry, approvals = head_fixture(
                Path(directory),
                artifact_target_definition_sha256="0" * 64,
            )
            self.assertEqual(
                registry.heads[METRIC].target_definition_sha256,
                spec.target_definition_sha256,
            )
            with self.assertRaisesRegex(CalibratedHeadError, "registry pin"):
                approved_runtime(registry, approvals)

        with tempfile.TemporaryDirectory() as directory:
            registry, approvals = head_fixture(
                Path(directory),
                evaluation_dataset_in_artifact="forged-evaluation",
                evaluation_dataset_in_registry="locked-account-disjoint-test",
            )
            with self.assertRaisesRegex(
                CalibratedHeadError, "evaluation manifest mismatch"
            ):
                approved_runtime(registry, approvals)

    def test_expired_evaluation_manifest_cannot_score(self):
        with tempfile.TemporaryDirectory() as directory:
            registry, approvals = head_fixture(Path(directory))
            with self.assertRaisesRegex(CalibratedHeadError, "not currently fresh"):
                approved_runtime(
                    registry,
                    approvals,
                    now=datetime(2027, 1, 1, tzinfo=timezone.utc),
                )

    def test_cortical_or_source_mismatched_feature_contract_is_rejected(self):
        with self.assertRaises(CalibratedHeadError):
            feature_contract_sha256(
                [{"name": "tribe.cortical.mean", "source": "vjepa21"}]
            )
        with self.assertRaises(CalibratedHeadError):
            feature_contract_sha256(
                [{"name": "vjepa2_1.motion_change", "source": "audioModel"}]
            )


if __name__ == "__main__":
    unittest.main()
