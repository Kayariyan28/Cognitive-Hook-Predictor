from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.insight.bundle import BundleUnavailableError
from backend.insight.config import InsightSettings
from backend.insight.providers.base import (
    Availability,
    GenerationResult,
    ProviderExecutionError,
    ProviderUnavailableError,
)
from backend.insight.router import create_insight_router
from backend.insight.service import InsightRequest, InsightService
from backend.insight.store import InsightStore, InsightStoreError
from backend.tests.insight_support import (
    FORECAST_RESULT_ID,
    TRIBE_RESULT_ID,
    forecast_result,
    tribe_descriptors,
)


FIXTURES = Path(__file__).resolve().parent / "insight_fixtures"
PINNED_REVISION = "b" * 40
MISSING_RESULT_ID = "0" * 32


def valid_output() -> str:
    return (FIXTURES / "valid_full_artifact.json").read_text(encoding="utf-8")


class FakeProvider:
    """A provider with no model, no network, and a scripted set of outcomes."""

    name = "mlx-local"

    def __init__(self, outputs=None, available=True, reason="ready", error=None):
        self._outputs = list(outputs or [valid_output()])
        self._available = available
        self._reason = reason
        self._error = error
        self.calls = 0

    def availability(self):
        return Availability(
            provider="mlx-local",
            configured=True,
            available=self._available,
            reason=self._reason,
            model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
            model_revision=PINNED_REVISION,
        )

    def generate(self, bundle, *, hook_only):
        self.calls += 1
        if self._error is not None:
            raise self._error
        raw = self._outputs.pop(0) if len(self._outputs) > 1 else self._outputs[0]
        return GenerationResult(
            raw_output=raw,
            model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
            model_revision=PINNED_REVISION,
            started_at="2026-02-01T10:00:00Z",
            completed_at="2026-02-01T10:00:02Z",
            elapsed_seconds=2.0,
        )


class ServiceHarness(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="insight-tests-")
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)
        self.results = {FORECAST_RESULT_ID: forecast_result()}

    def settings(self, **overrides) -> InsightSettings:
        environ = {
            "INSIGHT_PROVIDER": "mlx-local",
            "INSIGHT_LOCAL_MODEL_REVISION": PINNED_REVISION,
            "INSIGHT_DIR": str(self.root / "insight"),
        }
        environ.update(overrides)
        return InsightSettings.from_env(environ)

    def request(self, **overrides) -> InsightRequest:
        return InsightRequest(
            FORECAST_RESULT_ID, tribe_descriptors=tribe_descriptors(), **overrides
        )

    def body(self, **overrides) -> dict:
        payload = {
            "forecastResultId": FORECAST_RESULT_ID,
            "tribeDescriptors": tribe_descriptors(),
        }
        payload.update(overrides)
        return payload

    def service(self, provider=None, **overrides) -> tuple[InsightService, FakeProvider]:
        active = provider or FakeProvider()
        settings = self.settings(**overrides)
        service = InsightService(
            settings,
            forecast_result_loader=self.results.get,
            store=InsightStore(settings.insight_dir),
            provider_factory=lambda _: active,
        )
        return service, active


class StatusTests(ServiceHarness):
    def test_status_reports_provider_identity_and_prompt_version(self):
        service, _ = self.service()
        status = service.status()
        self.assertEqual(status["state"], "ready")
        self.assertTrue(status["generationAvailable"])
        self.assertEqual(status["provider"]["model"]["revision"], PINNED_REVISION)
        self.assertEqual(status["promptTemplate"]["id"], "hook-doctor.v3")
        self.assertEqual(len(status["promptTemplate"]["hash"]), 64)
        self.assertFalse(status["behavioralOutcome"])

    def test_status_never_leaks_the_api_key(self):
        service, _ = self.service(
            INSIGHT_PROVIDER="anthropic",
            INSIGHT_CLOUD_ENABLED="true",
            ANTHROPIC_API_KEY="sk-ant-super-secret",
        )
        serialized = json.dumps(service.status())
        self.assertNotIn("sk-ant-super-secret", serialized)
        self.assertIn('"apiKeyPresent": true', json.dumps(service.status(), indent=1))

    def test_unavailable_provider_is_named_in_status(self):
        service, _ = self.service(FakeProvider(available=False, reason="mlx-lm is not installed."))
        status = service.status()
        self.assertEqual(status["state"], "provider-unavailable")
        self.assertIn("mlx-lm", status["provider"]["reason"])


class GenerateTests(ServiceHarness):
    def test_happy_path_publishes_a_validated_artifact(self):
        service, provider = self.service()
        document, cached = service.generate(self.request())
        self.assertFalse(cached)
        self.assertEqual(document["schemaVersion"], "insight/2")
        self.assertFalse(document["behavioralOutcome"])
        self.assertIn("descriptive lane", document["limits"])
        self.assertEqual(document["provenance"]["provider"], "mlx-local")
        self.assertEqual(document["provenance"]["modelRevision"], PINNED_REVISION)
        self.assertEqual(len(document["provenance"]["outputHash"]), 64)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(
            service.read_artifact(document["insightId"])["insightId"], document["insightId"]
        )

    def test_hook_only_generation_records_the_window(self):
        service, _ = self.service()
        document, _ = service.generate(self.request(hook_only=True))
        self.assertTrue(document["provenance"]["hookOnly"])
        self.assertEqual(document["source"]["window"], [0.0, 3.0])

    def test_server_owns_limits_and_behavioral_outcome(self):
        payload = json.loads(valid_output())
        payload["hookReport"]["observations"][0]["text"] = "The opening AudioSet window carries Speech."
        payload["hookReport"]["observations"][0]["citations"] = ["ast:/windows/0"]
        service, _ = self.service(FakeProvider(outputs=[json.dumps(payload)]))
        document, _ = service.generate(self.request())
        self.assertIs(document["behavioralOutcome"], False)
        self.assertTrue(document["limits"])

    def test_exact_repeat_returns_the_cached_artifact(self):
        service, provider = self.service()
        first, first_cached = service.generate(self.request())
        second, second_cached = service.generate(self.request())
        self.assertFalse(first_cached)
        self.assertTrue(second_cached)
        self.assertEqual(first["insightId"], second["insightId"])
        self.assertEqual(provider.calls, 1)

    def test_a_different_window_is_a_different_cache_entry(self):
        service, provider = self.service()
        full, _ = service.generate(self.request())
        hook, cached = service.generate(self.request(hook_only=True))
        self.assertFalse(cached)
        self.assertNotEqual(full["insightId"], hook["insightId"])
        self.assertEqual(provider.calls, 2)

    def test_missing_result_is_bundle_unavailable(self):
        service, provider = self.service()
        document, _ = service.generate(InsightRequest(MISSING_RESULT_ID))
        self.assertTrue(document["unavailable"])
        self.assertEqual(document["reasonCode"], "bundle_unavailable")
        self.assertEqual(provider.calls, 0)

    def test_unreadable_result_is_bundle_unavailable(self):
        settings = self.settings()

        def broken(result_id):
            raise BundleUnavailableError("the stored forecast result is unreadable")

        service = InsightService(
            settings,
            forecast_result_loader=broken,
            store=InsightStore(settings.insight_dir),
            provider_factory=lambda _: FakeProvider(),
        )
        document, _ = service.generate(self.request())
        self.assertEqual(document["reasonCode"], "bundle_unavailable")

    def test_tribe_result_id_without_descriptors_is_refused(self):
        service, _ = self.service()
        document, _ = service.generate(
            InsightRequest(FORECAST_RESULT_ID, tribe_result_id=TRIBE_RESULT_ID)
        )
        self.assertEqual(document["reasonCode"], "bundle_unavailable")
        self.assertIn("tribe-cortical-descriptors/1", document["detail"])

    def test_mismatched_tribe_descriptors_are_refused(self):
        service, _ = self.service()
        document, _ = service.generate(
            InsightRequest(
                FORECAST_RESULT_ID,
                tribe_result_id="f" * 32,
                tribe_descriptors=tribe_descriptors(),
            )
        )
        self.assertEqual(document["reasonCode"], "bundle_unavailable")
        self.assertIn("do not belong", document["detail"])

    def test_provider_unavailable_never_calls_the_model(self):
        provider = FakeProvider(available=False, reason="INSIGHT_CLOUD_ENABLED is false.")
        service, _ = self.service(provider)
        document, _ = service.generate(self.request())
        self.assertEqual(document["reasonCode"], "provider_unavailable")
        self.assertEqual(provider.calls, 0)

    def test_provider_errors_are_reported_as_provider_error(self):
        service, _ = self.service(FakeProvider(error=ProviderExecutionError("no text")))
        document, _ = service.generate(self.request())
        self.assertEqual(document["reasonCode"], "provider_error")

        service, _ = self.service(FakeProvider(error=ProviderUnavailableError("gone")))
        document, _ = service.generate(self.request())
        self.assertEqual(document["reasonCode"], "provider_unavailable")

    def test_oversized_output_is_rejected_and_recorded(self):
        oversized = json.dumps({"hookReport": {"padding": "x" * 5000}})
        service, _ = self.service(
            FakeProvider(outputs=[oversized]), INSIGHT_MAX_OUTPUT_BYTES="512"
        )
        document, _ = service.generate(self.request())
        self.assertEqual(document["reasonCode"], "output_too_large")
        self.assertIn("rejectionId", document)


class RejectionRecordTests(ServiceHarness):
    def rejected(self, fixture: str):
        raw = (FIXTURES / fixture).read_text(encoding="utf-8")
        service, _ = self.service(FakeProvider(outputs=[raw]))
        document, cached = service.generate(self.request())
        return service, document, cached

    def test_every_validator_reason_code_reaches_the_endpoint(self):
        for fixture, reason in (
            ("global_virality_claim.json", "claim_boundary_violation"),
            ("invented_number.json", "numeric_not_in_evidence"),
            ("unknown_top_level_key.json", "unknown_field"),
            ("server_owned_limits.json", "server_owned_field"),
            ("empty_citations.json", "missing_citation"),
            ("absent_lane_citation.json", "citation_unresolvable"),
            ("unknown_lane_citation.json", "citation_malformed"),
            ("effort_enum_invalid.json", "schema_invalid"),
        ):
            with self.subTest(fixture=fixture):
                _, document, _ = self.rejected(fixture)
                self.assertEqual(document["reasonCode"], reason)
                self.assertTrue(document["unavailable"])

    def test_rejections_are_persisted_with_the_offending_sentence(self):
        service, document, _ = self.rejected("global_virality_claim.json")
        record = service.read_rejection(document["rejectionId"])
        self.assertEqual(record["reasonCode"], "claim_boundary_violation")
        self.assertEqual(record["detail"]["term"], "go viral")
        self.assertIn("go viral", record["detail"]["sentence"])
        self.assertEqual(record["promptTemplateId"], "hook-doctor.v3")

    def test_a_rejection_is_never_cached_as_a_success(self):
        raw = (FIXTURES / "global_virality_claim.json").read_text(encoding="utf-8")
        provider = FakeProvider(outputs=[raw])
        service, _ = self.service(provider)
        service.generate(self.request())
        service.generate(self.request())
        self.assertEqual(provider.calls, 2)

    def test_malformed_json_is_rejected_not_repaired(self):
        service, _ = self.service(FakeProvider(outputs=['{"hookReport": {']))
        document, _ = service.generate(self.request())
        self.assertEqual(document["reasonCode"], "output_not_json")


class AtomicPersistenceTests(ServiceHarness):
    def test_a_failed_write_leaves_no_partial_artifact(self):
        store = InsightStore(self.root / "insight")
        store.initialize()

        with self.assertRaises(InsightStoreError):
            store.publish_artifact("a" * 32, {"insightId": "a" * 32, "bad": object()})
        self.assertFalse((store.artifacts_dir / ("a" * 32)).exists())
        self.assertEqual(list(store.artifacts_dir.iterdir()), [])

    def test_published_artifacts_are_complete_and_identity_checked(self):
        store = InsightStore(self.root / "insight")
        store.publish_artifact("b" * 32, {"insightId": "b" * 32, "value": 1})
        self.assertEqual(store.read_artifact("b" * 32)["value"], 1)
        store.publish_artifact("c" * 32, {"insightId": "wrong"})
        with self.assertRaises(InsightStoreError):
            store.read_artifact("c" * 32)

    def test_republishing_the_same_identity_is_refused(self):
        store = InsightStore(self.root / "insight")
        store.publish_artifact("d" * 32, {"insightId": "d" * 32})
        with self.assertRaises(InsightStoreError):
            store.publish_artifact("d" * 32, {"insightId": "d" * 32})

    def test_no_staging_directory_survives_a_publish(self):
        store = InsightStore(self.root / "insight")
        store.publish_artifact("e" * 32, {"insightId": "e" * 32})
        self.assertEqual(
            [path.name for path in store.artifacts_dir.iterdir() if path.name.startswith(".")],
            [],
        )


class EndpointTests(ServiceHarness):
    def client(self, provider=None, **overrides) -> tuple[TestClient, InsightService]:
        service, _ = self.service(provider, **overrides)
        app = FastAPI()
        app.include_router(create_insight_router(service=service))
        return TestClient(app), service

    def test_status_endpoint_is_private_and_no_store(self):
        client, _ = self.client()
        response = client.get("/api/insight/v1/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.json()["service"], "creator-insight")

    def test_generate_endpoint_returns_the_artifact_and_cache_header(self):
        client, _ = self.client()
        body = self.body()
        first = client.post("/api/insight/v1/generate", json=body)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.headers["x-insight-cache"], "miss")
        self.assertEqual(first.headers["cache-control"], "private, no-store")
        second = client.post("/api/insight/v1/generate", json=body)
        self.assertEqual(second.headers["x-insight-cache"], "hit")
        self.assertEqual(first.json()["insightId"], second.json()["insightId"])

    def test_generate_endpoint_reports_unavailable_states(self):
        client, _ = self.client(FakeProvider(available=False, reason="not installed"))
        response = client.post("/api/insight/v1/generate", json=self.body())
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["unavailable"])
        self.assertEqual(payload["reasonCode"], "provider_unavailable")

    def test_invalid_request_bodies_are_caller_errors(self):
        client, _ = self.client()
        for body in (
            {},
            {"forecastResultId": "not-a-result-id"},
            {"forecastResultId": FORECAST_RESULT_ID, "hookOnly": "yes"},
            {"forecastResultId": FORECAST_RESULT_ID, "surprise": 1},
            {"forecastResultId": FORECAST_RESULT_ID, "tribeDescriptors": []},
        ):
            with self.subTest(body=body):
                response = client.post("/api/insight/v1/generate", json=body)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["detail"]["code"], "invalid_request")

    def test_results_endpoint_serves_published_artifacts_only(self):
        client, _ = self.client()
        created = client.post("/api/insight/v1/generate", json=self.body()).json()
        found = client.get(f"/api/insight/v1/results/{created['insightId']}")
        self.assertEqual(found.status_code, 200)
        self.assertEqual(found.json()["insightId"], created["insightId"])
        self.assertEqual(found.headers["cache-control"], "private, no-store")

        missing = client.get(f"/api/insight/v1/results/{MISSING_RESULT_ID}")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["detail"]["code"], "insight_not_found")

    def test_rejection_records_are_retrievable(self):
        raw = (FIXTURES / "global_retention_claim.json").read_text(encoding="utf-8")
        client, _ = self.client(FakeProvider(outputs=[raw]))
        rejected = client.post("/api/insight/v1/generate", json=self.body()).json()
        record = client.get(f"/api/insight/v1/rejections/{rejected['rejectionId']}")
        self.assertEqual(record.status_code, 200)
        self.assertEqual(record.json()["detail"]["term"], "retention")


if __name__ == "__main__":
    unittest.main()
