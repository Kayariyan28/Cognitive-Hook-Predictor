from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.insight.config import InsightSettings
from backend.insight.experiments import (
    COMPARISON_TOLERANCE,
    ExperimentError,
    ExperimentRequest,
    ExperimentTracker,
    classify_match,
    measure_deltas,
    observed_direction,
)
from backend.insight.router import create_insight_router
from backend.insight.service import InsightRequest, InsightService
from backend.insight.store import InsightStore
from backend.tests.insight_support import FORECAST_RESULT_ID, forecast_result, tribe_descriptors
from backend.tests.test_insight_service import FakeProvider


VARIANT_RESULT_ID = "7c2d9f1a4b6e8c0d2f4a6b8c0d2e4f6a"
MISSING_ID = "0" * 32


def variant_result(**overrides):
    """A second analysed clip with the same shape and different measurements."""

    result = forecast_result()
    result["resultId"] = VARIANT_RESULT_ID
    result["jobId"] = VARIANT_RESULT_ID
    audio = result["evidence"]["optionalProviders"]["measuredAudio"]["result"]["features"]
    audio["measured_audio.silent_window_fraction"] = overrides.pop("silent_window_fraction", 0.1125)
    audio["measured_audio.rms"] = overrides.pop("rms", 0.1372)
    return result


class ExperimentHarness(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="experiment-tests-")
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)
        self.results = {
            FORECAST_RESULT_ID: forecast_result(),
            VARIANT_RESULT_ID: variant_result(),
        }
        settings = InsightSettings.from_env(
            {
                "INSIGHT_PROVIDER": "mlx-local",
                "INSIGHT_LOCAL_MODEL_REVISION": "b" * 40,
                "INSIGHT_DIR": str(self.root / "insight"),
            }
        )
        self.store = InsightStore(settings.insight_dir)
        self.service = InsightService(
            settings,
            forecast_result_loader=self.results.get,
            store=self.store,
            provider_factory=lambda _: FakeProvider(),
        )
        self.tracker = ExperimentTracker(
            self.store, forecast_result_loader=self.results.get
        )

    def insight(self):
        artifact, _ = self.service.generate(
            InsightRequest(FORECAST_RESULT_ID, tribe_descriptors=tribe_descriptors())
        )
        self.assertIn("insightId", artifact)
        return artifact

    def proposed(self):
        artifact = self.insight()
        return artifact, self.tracker.create(
            ExperimentRequest(insight_id=artifact["insightId"], experiment_id="x1")
        )


class DirectionRuleTests(unittest.TestCase):
    def test_the_tolerance_matches_the_existing_ab_comparison_rule(self):
        source = Path("src/tribe/variantComparison.js").read_text(encoding="utf-8")
        match = re.search(r"const COMPARISON_TOLERANCE = ([0-9.e-]+);", source)
        self.assertIsNotNone(match, "variantComparison.js no longer declares its tolerance")
        self.assertEqual(float(match.group(1)), COMPARISON_TOLERANCE)

    def test_direction_classification_mirrors_the_ab_lab(self):
        self.assertEqual(observed_direction(0.4, 0.5)[1], "increase")
        self.assertEqual(observed_direction(0.5, 0.4)[1], "decrease")
        self.assertEqual(observed_direction(0.5, 0.5)[1], "unchanged")
        # A difference inside the tolerance is not a change.
        self.assertEqual(observed_direction(1.0, 1.0 + 1e-9)[1], "unchanged")
        self.assertEqual(observed_direction(1.0, 1.0 + 1e-3)[1], "increase")

    def test_match_classification(self):
        self.assertEqual(classify_match("increase", "increase"), "matched")
        self.assertEqual(classify_match("increase", "decrease"), "opposite")
        self.assertEqual(classify_match("increase", "unchanged"), "unmatched")
        self.assertEqual(classify_match("unchanged", "unchanged"), "matched")
        self.assertEqual(classify_match("decrease", "increase"), "opposite")


class DeltaMathTests(ExperimentHarness):
    def bundles(self, **overrides):
        from backend.insight.bundle import assemble_evidence_bundle

        return (
            assemble_evidence_bundle(forecast_result()),
            assemble_evidence_bundle(variant_result(**overrides)),
        )

    def test_deltas_are_measured_against_both_results(self):
        baseline, variant = self.bundles()
        deltas = measure_deltas(
            [{"metricPath": "measured:/audio/descriptors/silent_window_fraction", "direction": "decrease"}],
            baseline,
            variant,
        )
        self.assertEqual(deltas[0]["baselineValue"], 0.1875)
        self.assertEqual(deltas[0]["variantValue"], 0.1125)
        self.assertAlmostEqual(deltas[0]["delta"], -0.075)
        self.assertEqual(deltas[0]["observedDirection"], "decrease")
        self.assertEqual(deltas[0]["match"], "matched")

    def test_an_opposite_movement_is_named_opposite(self):
        baseline, variant = self.bundles(silent_window_fraction=0.4)
        deltas = measure_deltas(
            [{"metricPath": "measured:/audio/descriptors/silent_window_fraction", "direction": "decrease"}],
            baseline,
            variant,
        )
        self.assertEqual(deltas[0]["match"], "opposite")

    def test_an_unmoved_signal_is_not_forced_into_a_direction(self):
        baseline, variant = self.bundles(silent_window_fraction=0.1875)
        deltas = measure_deltas(
            [{"metricPath": "measured:/audio/descriptors/silent_window_fraction", "direction": "decrease"}],
            baseline,
            variant,
        )
        self.assertEqual(deltas[0]["observedDirection"], "unchanged")
        self.assertEqual(deltas[0]["match"], "unmatched")

    def test_a_missing_metric_path_fails_closed_as_unmeasured(self):
        baseline, variant = self.bundles()
        for metric_path in (
            "measured:/audio/descriptors/does_not_exist",
            "tribe:/intervals/0/magnitude",
            "measured:/audio/descriptors",
            "not-a-citation",
        ):
            with self.subTest(metric_path=metric_path):
                deltas = measure_deltas(
                    [{"metricPath": metric_path, "direction": "increase"}], baseline, variant
                )
                self.assertEqual(deltas[0]["match"], "unmeasured")
                self.assertIsNone(deltas[0]["delta"])
                self.assertTrue(deltas[0]["reason"])

    def test_an_invalid_expected_shift_is_refused(self):
        baseline, variant = self.bundles()
        with self.assertRaises(ExperimentError):
            measure_deltas([{"metricPath": 7, "direction": "increase"}], baseline, variant)
        with self.assertRaises(ExperimentError):
            measure_deltas(
                [{"metricPath": "measured:/audio/descriptors/rms", "direction": "up"}],
                baseline,
                variant,
            )


class LifecycleTests(ExperimentHarness):
    def test_an_experiment_is_created_from_an_insight_experiment(self):
        artifact, record = self.proposed()
        self.assertEqual(record["status"], "proposed")
        self.assertEqual(record["sourceInsightId"], artifact["insightId"])
        self.assertEqual(record["sourceExperimentId"], "x1")
        self.assertEqual(record["hypothesisIds"], ["h1"])
        self.assertEqual(record["linkedResultIds"], [FORECAST_RESULT_ID])
        self.assertIsNone(record["measuredDeltas"])
        self.assertIs(record["behavioralOutcome"], False)

    def test_the_full_lifecycle_reaches_measured_deltas(self):
        _, record = self.proposed()
        edited = self.tracker.mark_edited(record["id"])
        self.assertEqual(edited["status"], "edited")
        compared = self.tracker.attach_variant(record["id"], VARIANT_RESULT_ID)
        self.assertEqual(compared["status"], "compared")
        self.assertEqual(compared["linkedResultIds"], [FORECAST_RESULT_ID, VARIANT_RESULT_ID])
        self.assertEqual(compared["measuredDeltas"][0]["match"], "matched")
        self.assertEqual(self.tracker.read(record["id"])["status"], "compared")

    def test_a_variant_may_be_attached_without_the_edited_step(self):
        _, record = self.proposed()
        compared = self.tracker.attach_variant(record["id"], VARIANT_RESULT_ID)
        self.assertEqual(compared["status"], "compared")

    def test_a_compared_experiment_never_returns_to_edited(self):
        _, record = self.proposed()
        self.tracker.attach_variant(record["id"], VARIANT_RESULT_ID)
        with self.assertRaises(ExperimentError):
            self.tracker.mark_edited(record["id"])

    def test_an_unknown_insight_or_experiment_is_refused(self):
        with self.assertRaises(ExperimentError):
            self.tracker.create(ExperimentRequest(insight_id=MISSING_ID, experiment_id="x1"))
        artifact = self.insight()
        with self.assertRaises(ExperimentError):
            self.tracker.create(
                ExperimentRequest(insight_id=artifact["insightId"], experiment_id="x9")
            )

    def test_a_missing_variant_result_fails_closed(self):
        _, record = self.proposed()
        with self.assertRaises(ExperimentError):
            self.tracker.attach_variant(record["id"], MISSING_ID)
        self.assertEqual(self.tracker.read(record["id"])["status"], "proposed")

    def test_the_variant_must_differ_from_the_baseline(self):
        _, record = self.proposed()
        with self.assertRaises(ExperimentError):
            self.tracker.attach_variant(record["id"], FORECAST_RESULT_ID)

    def test_experiments_are_listed_newest_first(self):
        artifact = self.insight()
        first = self.tracker.create(
            ExperimentRequest(insight_id=artifact["insightId"], experiment_id="x1")
        )
        second = copy.deepcopy(first)
        second["id"] = "1" * 32
        second["createdAt"] = "2099-01-01T00:00:00Z"
        self.store.publish_experiment(second["id"], second)
        listed = self.tracker.list_experiments()
        self.assertEqual([item["id"] for item in listed], [second["id"], first["id"]])


class EndpointTests(ExperimentHarness):
    def client(self) -> TestClient:
        app = FastAPI()
        app.include_router(
            create_insight_router(service=self.service, tracker=self.tracker)
        )
        return TestClient(app)

    def test_the_endpoints_drive_the_whole_loop(self):
        client = self.client()
        artifact = self.insight()
        created = client.post(
            "/api/insight/v1/experiments",
            json={"insightId": artifact["insightId"], "experimentId": "x1"},
        )
        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.headers["cache-control"], "private, no-store")
        experiment_id = created.json()["id"]

        edited = client.post(f"/api/insight/v1/experiments/{experiment_id}/edited")
        self.assertEqual(edited.json()["status"], "edited")

        compared = client.post(
            f"/api/insight/v1/experiments/{experiment_id}/variant",
            json={"forecastResultId": VARIANT_RESULT_ID},
        )
        self.assertEqual(compared.status_code, 200)
        self.assertEqual(compared.json()["measuredDeltas"][0]["match"], "matched")

        listed = client.get("/api/insight/v1/experiments")
        self.assertEqual(len(listed.json()["experiments"]), 1)
        fetched = client.get(f"/api/insight/v1/experiments/{experiment_id}")
        self.assertEqual(fetched.json()["status"], "compared")

    def test_caller_errors_use_the_repo_error_shape(self):
        client = self.client()
        artifact = self.insight()
        for body in ({}, {"insightId": artifact["insightId"]}, {"insightId": "nope", "experimentId": "x1"}):
            with self.subTest(body=body):
                response = client.post("/api/insight/v1/experiments", json=body)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["detail"]["code"], "invalid_request")

        missing = client.post(
            "/api/insight/v1/experiments",
            json={"insightId": MISSING_ID, "experimentId": "x1"},
        )
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["detail"]["code"], "experiment_unavailable")

    def test_no_outcome_language_reaches_the_persisted_record(self):
        _, record = self.proposed()
        compared = self.tracker.attach_variant(record["id"], VARIANT_RESULT_ID)
        serialized = json.dumps(compared).lower()
        for term in ("viral", "retention", "views", "engagement", "watch time", "outcome:"):
            self.assertNotIn(term, serialized, term)
        self.assertIn("measured signal", compared["limits"])


if __name__ == "__main__":
    unittest.main()
