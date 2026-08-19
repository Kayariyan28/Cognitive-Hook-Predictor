from __future__ import annotations

from types import SimpleNamespace
import unittest

from backend.forecast.registry import ForecastRegistry
from backend.forecast.router import (
    _adapter_runtime_status,
    _forecast_job_runtime_status,
)
from backend.forecast.schema import build_capabilities


class ReadyLocalAdapter:
    def availability(self):
        return {
            "configured": True,
            "executionAvailable": True,
            "reason": None,
            "role": "measured-test-evidence",
            "usesLearnedModel": False,
            "secret": "must-not-be-public",
        }


class RemoteAdapter:
    def probe(self):  # pragma: no cover - status must never call it
        raise AssertionError("status contacted a remote worker")


class LocalVJepa21Adapter:
    def availability(self):  # pragma: no cover - status must never call it
        raise AssertionError("status loaded the V-JEPA runtime")


class FakePin:
    def public_provenance(self):
        return {"id": "fixture", "artifactSha256": "a" * 64}


class ForecastRuntimeStatusTests(unittest.TestCase):
    def test_expensive_and_remote_readiness_is_deferred_without_execution(self) -> None:
        for adapter in (RemoteAdapter(), LocalVJepa21Adapter()):
            status = _adapter_runtime_status(adapter)
            self.assertTrue(status["configured"])
            self.assertIsNone(status["executionAvailable"])
            self.assertTrue(status["readinessDeferred"])

    def test_job_runtime_reports_only_safe_adapter_state_and_executable_heads(self) -> None:
        service = SimpleNamespace(
            orchestrator=SimpleNamespace(
                adapters={
                    "vjepa21": LocalVJepa21Adapter(),
                    "measuredAudio": ReadyLocalAdapter(),
                },
                head_runtime=SimpleNamespace(heads={"retention5Second": object()}),
            )
        )
        status = _forecast_job_runtime_status(service)
        self.assertEqual(status["executableHeadKeys"], ["retention5Second"])
        self.assertTrue(status["branches"]["measuredAudio"]["executionAvailable"])
        self.assertNotIn("secret", status["branches"]["measuredAudio"]["provider"])

    def test_capability_document_reflects_job_route_but_requires_registry_bound_head(self) -> None:
        runtime = {
            "branches": {
                "measuredAudio": {
                    "configured": True,
                    "executionAvailable": True,
                    "readinessDeferred": False,
                    "reason": None,
                    "readinessDiagnostics": [],
                },
                "vjepa21": {
                    "configured": True,
                    "executionAvailable": None,
                    "readinessDeferred": True,
                    "reason": "Checked per job.",
                    "readinessDiagnostics": [],
                },
            },
            "executableHeadKeys": ["retention5Second"],
        }
        empty = build_capabilities(
            ForecastRegistry.empty(), forecast_job_runtime_status=runtime
        )
        self.assertEqual(empty["state"], "evidence-job-service")
        self.assertTrue(empty["uploadsAccepted"])
        self.assertEqual(empty["branches"]["measuredAudio"]["state"], "ready")
        self.assertEqual(
            empty["branches"]["vjepa21"]["state"],
            "configured-readiness-deferred",
        )
        self.assertFalse(empty["scoreGenerationAvailable"])
        self.assertEqual(empty["jobExecution"]["executableCalibrationHeadCount"], 0)

        registry = SimpleNamespace(
            branches={},
            heads={"retention5Second": FakePin()},
            source="test-fixture",
        )
        executable = build_capabilities(
            registry, forecast_job_runtime_status=runtime
        )
        self.assertTrue(executable["scoreGenerationAvailable"])
        self.assertEqual(
            executable["calibrationHeads"]["retention5Second"]["state"],
            "executable-production-calibrated",
        )
        self.assertTrue(
            executable["calibrationHeads"]["retention5Second"]["scoringAvailable"]
        )


if __name__ == "__main__":
    unittest.main()
