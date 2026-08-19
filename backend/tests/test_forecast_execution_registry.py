from __future__ import annotations

import json
import unittest

from backend.errors import ConfigurationError
from backend.forecast.execution_registry import WorkerRegistry


def registry_document(base_url: str = "https://gpu.example.test/forecast") -> dict:
    return {
        "schemaVersion": "creator-forecast-worker-registry/1",
        "workers": {
            "vjepa21": {
                "baseUrl": base_url,
                "authTokenEnv": "FORECAST_VJEPA21_TOKEN",
                "timeoutSeconds": 900,
                "model": {
                    "id": "facebookresearch/vjepa2.1-vit-base-384",
                    "revision": "1" * 40,
                    "weightsSha256": "2" * 64,
                    "license": "MIT",
                    "adapterId": "vjepa21-pytorch",
                },
                "codeRevision": "3" * 40,
                "preprocessing": {
                    "id": "vjepa2.1-64f-384/1",
                    "sha256": "4" * 64,
                },
            }
        },
    }


class ForecastExecutionRegistryTests(unittest.TestCase):
    def test_remote_worker_is_immutable_and_secret_value_is_not_public(self) -> None:
        registry = WorkerRegistry.from_mapping(registry_document())
        pin = registry.workers["vjepa21"]
        self.assertEqual(pin.timeout_seconds, 900.0)
        self.assertEqual(pin.auth_token({"FORECAST_VJEPA21_TOKEN": "secret"}), "secret")
        self.assertNotIn("FORECAST_VJEPA21_TOKEN", json.dumps(pin.public_runtime()))
        self.assertNotIn("secret", json.dumps(pin.public_runtime()))

    def test_http_is_allowed_only_for_loopback(self) -> None:
        local = WorkerRegistry.from_mapping(
            registry_document("http://127.0.0.1:9101/")
        ).workers["vjepa21"]
        self.assertEqual(local.base_url, "http://127.0.0.1:9101")

        with self.assertRaisesRegex(ConfigurationError, "HTTPS"):
            WorkerRegistry.from_mapping(registry_document("http://gpu.example.test"))

    def test_registry_rejects_mutable_or_mismatched_worker_configuration(self) -> None:
        mutable = registry_document()
        mutable["workers"]["vjepa21"]["codeRevision"] = "main"
        with self.assertRaisesRegex(ConfigurationError, "immutable"):
            WorkerRegistry.from_mapping(mutable)

        wrong = registry_document()
        wrong["workers"]["vjepa21"]["model"]["adapterId"] = "generic"
        with self.assertRaisesRegex(ConfigurationError, "adapterId"):
            WorkerRegistry.from_mapping(wrong)

    def test_empty_environment_is_valid_and_duplicate_json_keys_fail(self) -> None:
        self.assertEqual(len(WorkerRegistry.from_env({}).workers), 0)
        payload = (
            '{"schemaVersion":"creator-forecast-worker-registry/1",'
            '"workers":{},"workers":{}}'
        )
        with self.assertRaisesRegex(ConfigurationError, "duplicate"):
            WorkerRegistry.from_env({"FORECAST_WORKER_REGISTRY_JSON": payload})


if __name__ == "__main__":
    unittest.main()
