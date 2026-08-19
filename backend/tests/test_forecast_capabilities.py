from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import backend.app as app_module
from backend.errors import ConfigurationError
from backend.forecast.registry import ForecastRegistry
from backend.forecast.schema import build_capabilities, validate_duration_seconds


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def tribe_status(state: str = "ready") -> dict:
    return {
        "state": state,
        "configured": True,
        "model": {
            "id": "facebook/tribev2",
            "revision": "a" * 40,
            "weightsSha256": "b" * 64,
            "inferenceMode": "vision-only",
            "modalitiesUsed": ["video"],
            "extractors": {
                "video": {
                    "id": "facebook/vjepa2-vitg-fpc64-256",
                    "revision": "c" * 40,
                    "weightsSha256": "d" * 64,
                    "device": "mps",
                    "precision": "autocast-fp16",
                }
            },
        },
        "lastError": None,
    }


def branch_record(path: Path, payload: bytes, *, adapter_id: str) -> dict:
    path.write_bytes(payload)
    return {
        "id": "example/provider",
        "revision": "1" * 40,
        "artifactPath": str(path),
        "artifactSha256": sha256_bytes(payload),
        "license": "Apache-2.0",
        "adapterId": adapter_id,
    }


def head_record(model_path: Path, calibration_path: Path) -> dict:
    model_payload = b"calibrated-head"
    calibration_payload = b"calibration-evidence"
    target_definition = "Eligible starts still playing at five seconds."
    denominator_definition = "All eligible starts."
    evaluation_manifest = {
        "manifestId": "example.retention-evaluation.v1",
        "dataset": "account-disjoint-temporal-holdout",
        "evaluatedAt": "2026-07-01T00:00:00Z",
        "validUntil": "2027-07-01T00:00:00Z",
        "sampleCount": 1200,
        "brierScore": 0.18,
        "expectedCalibrationError": 0.04,
    }
    evaluation_manifest_sha256 = sha256_bytes(
        json.dumps(
            evaluation_manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    )
    model_path.write_bytes(model_payload)
    calibration_path.write_bytes(calibration_payload)
    return {
        "id": "example/retention-head",
        "revision": "2" * 40,
        "artifactPath": str(model_path),
        "artifactSha256": sha256_bytes(model_payload),
        "license": "Apache-2.0",
        "outputKind": "calibrated-probability",
        "targetId": "example.retention-at-five-seconds.v1",
        "targetDefinition": target_definition,
        "targetDefinitionSha256": sha256_bytes(target_definition.encode()),
        "denominatorId": "example.eligible-starts.v1",
        "denominatorDefinition": denominator_definition,
        "denominatorDefinitionSha256": sha256_bytes(
            denominator_definition.encode()
        ),
        "platform": "example-shorts",
        "domain": "short-form-video",
        "locale": "en-us",
        "population": "Held-out consenting creator accounts.",
        "horizon": {"kind": "seconds-after-start", "value": 5},
        "featureContractSha256": "f" * 64,
        "trainingDataCutoff": "2026-06-01T00:00:00Z",
        "calibration": {
            "revision": "3" * 40,
            "artifactPath": str(calibration_path),
            "artifactSha256": sha256_bytes(calibration_payload),
            "method": "isotonic-held-out",
            "evaluationManifestId": "example.retention-evaluation.v1",
            "evaluationManifestSha256": evaluation_manifest_sha256,
            "evaluationDataset": "account-disjoint-temporal-holdout",
            "evaluatedAt": "2026-07-01T00:00:00Z",
            "validUntil": "2027-07-01T00:00:00Z",
            "sampleCount": 1200,
            "brierScore": 0.18,
            "expectedCalibrationError": 0.04,
        },
    }


class ForecastCapabilitiesTests(unittest.TestCase):
    def test_default_profile_reports_real_current_roles_without_scores(self) -> None:
        manifest = build_capabilities(
            ForecastRegistry.empty(),
            tribe_runtime_status=tribe_status(),
            browser_analysis_source_sha256="e" * 64,
        )

        self.assertEqual(
            manifest["schemaVersion"], "creator-forecast-capabilities/1"
        )
        self.assertEqual(manifest["state"], "evidence-job-service")
        self.assertFalse(manifest["scoreGenerationAvailable"])
        self.assertTrue(manifest["uploadsAccepted"])
        self.assertEqual(manifest["jobExecution"]["state"], "ready")
        self.assertTrue(manifest["jobExecution"]["uploadsAccepted"])
        self.assertEqual(
            manifest["supportedInput"]["durationSeconds"],
            {"minimum": 10.0, "maximum": 60.0, "inclusive": True},
        )
        self.assertEqual(
            manifest["availableNowProfile"]["implementedDescriptiveProviders"],
            ["browserLocalMetadata", "browserLocalVisual", "browserLocalAudio"],
        )

        visual = manifest["branches"]["browserLocalVisual"]
        audio = manifest["branches"]["browserLocalAudio"]
        self.assertEqual(visual["state"], "implemented-client-runtime")
        self.assertEqual(visual["role"], "descriptive-visual-evidence")
        self.assertFalse(visual["forecastContribution"])
        self.assertFalse(visual["substitution"]["isSubstitute"])
        self.assertEqual(audio["role"], "descriptive-audio-evidence")
        self.assertIn("may be unavailable", audio["perClipAvailability"])

        cortical = manifest["branches"]["tribeCortical"]
        self.assertEqual(cortical["role"], "cortical-only")
        self.assertEqual(cortical["state"], "ready")
        self.assertTrue(cortical["currentlyReady"])
        self.assertFalse(cortical["forecastContribution"])
        self.assertEqual(
            cortical["provenance"]["videoExtractor"]["id"],
            "facebook/vjepa2-vitg-fpc64-256",
        )

        for key in (
            "vjepa21",
            "videoLlama21Av",
            "audioModel",
            "account",
            "trends",
            "competitors",
        ):
            with self.subTest(branch=key):
                branch = manifest["branches"][key]
                self.assertEqual(branch["state"], "not-configured")
                self.assertFalse(branch["configured"])
                self.assertFalse(branch["serverExecutionAvailable"])
                self.assertFalse(branch["forecastContribution"])

        for head in manifest["calibrationHeads"].values():
            self.assertEqual(head["state"], "not-configured")
            self.assertFalse(head["configured"])
            self.assertFalse(head["scoringAvailable"])
            self.assertNotIn("value", head)

    def test_tribe_readiness_is_runtime_derived_not_inferred_from_registry(self) -> None:
        loading = build_capabilities(
            ForecastRegistry.empty(),
            tribe_runtime_status=tribe_status("loading"),
            browser_analysis_source_sha256="e" * 64,
        )["branches"]["tribeCortical"]
        self.assertEqual(loading["state"], "loading")
        self.assertFalse(loading["currentlyReady"])

        unavailable = build_capabilities(
            ForecastRegistry.empty(), tribe_runtime_status=None
        )["branches"]["tribeCortical"]
        self.assertEqual(unavailable["state"], "status-unavailable")
        self.assertFalse(unavailable["configured"])
        self.assertFalse(unavailable["currentlyReady"])

    def test_duration_range_is_inclusive_and_rejects_invalid_values(self) -> None:
        self.assertEqual(validate_duration_seconds(10), 10.0)
        self.assertEqual(validate_duration_seconds(60.0), 60.0)
        for value in (9.999, 60.001, True, "30", math.nan, math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_duration_seconds(value)

    def test_registered_artifacts_are_hash_checked_but_not_called_executable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ForecastRegistry.from_mapping(
                {
                    "schemaVersion": "creator-forecast-registry/1",
                    "branches": {
                        "vjepa21": branch_record(
                            root / "vjepa21.bin",
                            b"real-vjepa21-artifact",
                            adapter_id="vjepa21-pytorch",
                        )
                    },
                    "heads": {
                        "retention5Second": head_record(
                            root / "retention.bin", root / "calibration.json"
                        )
                    },
                },
                base_directory=root,
            )
            manifest = build_capabilities(
                registry,
                tribe_runtime_status=tribe_status(),
                browser_analysis_source_sha256="e" * 64,
            )

        branch = manifest["branches"]["vjepa21"]
        self.assertTrue(branch["configured"])
        self.assertEqual(branch["state"], "registered-artifact-runtime-unavailable")
        self.assertFalse(branch["serverExecutionAvailable"])
        self.assertEqual(branch["provenance"]["adapterId"], "vjepa21-pytorch")
        self.assertNotIn("artifactPath", json.dumps(manifest))

        head = manifest["calibrationHeads"]["retention5Second"]
        self.assertTrue(head["configured"])
        self.assertEqual(head["state"], "registered-calibrator-runtime-unavailable")
        self.assertFalse(head["scoringAvailable"])
        self.assertEqual(head["provenance"]["calibration"]["brierScore"], 0.18)
        self.assertNotIn("value", head)

    def test_registry_rejects_mutable_mismatched_and_unknown_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "provider.bin"
            valid = branch_record(
                artifact, b"provider", adapter_id="vjepa21-pytorch"
            )
            base = {
                "schemaVersion": "creator-forecast-registry/1",
                "branches": {"vjepa21": valid},
                "heads": {},
            }

            mutable = json.loads(json.dumps(base))
            mutable["branches"]["vjepa21"]["revision"] = "main"
            with self.assertRaisesRegex(ConfigurationError, "immutable"):
                ForecastRegistry.from_mapping(mutable, base_directory=root)

            bad_hash = json.loads(json.dumps(base))
            bad_hash["branches"]["vjepa21"]["artifactSha256"] = "0" * 64
            with self.assertRaisesRegex(ConfigurationError, "SHA-256 check"):
                ForecastRegistry.from_mapping(bad_hash, base_directory=root)

            wrong_adapter = json.loads(json.dumps(base))
            wrong_adapter["branches"]["vjepa21"]["adapterId"] = "generic"
            with self.assertRaisesRegex(ConfigurationError, "adapterId"):
                ForecastRegistry.from_mapping(wrong_adapter, base_directory=root)

            unknown = json.loads(json.dumps(base))
            unknown["branches"] = {"magicViralityModel": valid}
            with self.assertRaisesRegex(ConfigurationError, "unknown branch"):
                ForecastRegistry.from_mapping(unknown, base_directory=root)

    def test_environment_registry_is_explicit_and_mutually_exclusive(self) -> None:
        empty_payload = json.dumps(
            {
                "schemaVersion": "creator-forecast-registry/1",
                "branches": {},
                "heads": {},
            }
        )
        inline = ForecastRegistry.from_env({"FORECAST_REGISTRY_JSON": empty_payload})
        self.assertEqual(inline.source, "environment-json")
        with self.assertRaisesRegex(ConfigurationError, "only one"):
            ForecastRegistry.from_env(
                {
                    "FORECAST_REGISTRY_JSON": empty_payload,
                    "FORECAST_REGISTRY_PATH": "/tmp/registry.json",
                }
            )

    def test_fastapi_keeps_the_read_only_forecast_status_contract(self) -> None:
        routes = [
            route
            for route in app_module.app.routes
            if getattr(route, "path", "").startswith("/api/forecast/v1")
        ]
        status_routes = [
            route for route in routes if route.path == "/api/forecast/v1/status"
        ]
        self.assertEqual(len(status_routes), 1)
        self.assertEqual(status_routes[0].methods, {"GET"})
        response = status_routes[0].endpoint()
        payload = json.loads(response.body)
        self.assertEqual(payload["schemaVersion"], "creator-forecast-capabilities/1")
        self.assertFalse(payload["scoreGenerationAvailable"])
        self.assertEqual(response.headers["cache-control"], "no-store")


if __name__ == "__main__":
    unittest.main()
