"""Roadmap steps 3 and 4: recut the clip, then measure the cuts side by side."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.insight.bundle import assemble_evidence_bundle
from backend.insight.config import InsightSettings
from backend.insight.recut import (
    MAXIMUM_TRIM_SECONDS,
    OPERATIONS,
    RecutAssistant,
    RecutError,
    plan_recut,
)
from backend.insight.router import create_insight_router
from backend.insight.service import InsightService
from backend.insight.store import InsightStore
from backend.insight.variants import (
    MAXIMUM_VARIANTS,
    Variant,
    VariantComparisonError,
    build_variant_comparison,
    variants_from_results,
)
from backend.tests.insight_support import FORECAST_RESULT_ID, forecast_result
from backend.tests.test_insight_service import FakeProvider


SILENCE = "measured:/audio/descriptors/silent_window_fraction"
RMS = "measured:/audio/descriptors/rms"


def cut(index: int, *, silence: float, rms: float = 0.1372) -> dict:
    result = forecast_result()
    identifier = f"{index:032x}"
    result["resultId"] = identifier
    result["jobId"] = identifier
    features = result["evidence"]["optionalProviders"]["measuredAudio"]["result"]["features"]
    features["measured_audio.silent_window_fraction"] = silence
    features["measured_audio.rms"] = rms
    return result


class RecutPlanTests(unittest.TestCase):
    def test_trimming_the_start_keeps_the_remainder(self):
        plan = plan_recut("trim_start", duration_seconds=21.5, seconds=1.5)
        self.assertEqual(plan.startSec, 1.5)
        self.assertEqual(plan.endSec, 21.5)
        self.assertEqual(plan.result_duration, 20.0)

    def test_trimming_the_end_keeps_the_opening(self):
        plan = plan_recut("trim_end", duration_seconds=21.5, seconds=1.5)
        self.assertEqual(plan.startSec, 0.0)
        self.assertEqual(plan.endSec, 20.0)

    def test_keeping_a_window_is_explicit(self):
        plan = plan_recut("keep_window", duration_seconds=45.0, start_seconds=2, end_seconds=32)
        self.assertEqual((plan.startSec, plan.endSec), (2.0, 32.0))
        self.assertEqual(plan.result_duration, 30.0)

    def test_a_recut_that_leaves_an_unanalysable_clip_is_refused(self):
        for kwargs in (
            {"operation": "trim_start", "seconds": 20.0},
            {"operation": "trim_end", "seconds": 15.0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(RecutError) as caught:
                    plan_recut(duration_seconds=21.5, **kwargs)
                self.assertEqual(caught.exception.reason_code, "duration_out_of_range")

    def test_invalid_requests_are_refused_before_any_frame_is_decoded(self):
        for kwargs, reason in (
            ({"operation": "sharpen", "seconds": 1.0}, "unknown_operation"),
            ({"operation": "trim_start", "seconds": 0}, "invalid_request"),
            ({"operation": "trim_start", "seconds": -1}, "invalid_request"),
            ({"operation": "trim_start", "seconds": MAXIMUM_TRIM_SECONDS + 1}, "invalid_request"),
            ({"operation": "trim_start", "seconds": float("nan")}, "invalid_request"),
            ({"operation": "keep_window", "start_seconds": 5, "end_seconds": 5}, "invalid_request"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(RecutError) as caught:
                    plan_recut(duration_seconds=45.0, **kwargs)
                self.assertEqual(caught.exception.reason_code, reason)

    def test_the_plan_states_that_a_recut_measures_nothing(self):
        plan = plan_recut("trim_start", duration_seconds=21.5, seconds=1.0).public_value()
        self.assertIs(plan["behavioralOutcome"], False)
        self.assertIn("changes the clip, not any measurement", plan["limits"])
        self.assertIn(plan["operation"], OPERATIONS)

    def test_the_operation_vocabulary_is_code_owned(self):
        self.assertEqual(sorted(OPERATIONS), ["keep_window", "trim_end", "trim_start"])


class FakeRunner:
    def __init__(self, *, available=True, fail=False):
        self._available = available
        self._fail = fail
        self.rendered: list[tuple[float, float]] = []

    def available(self) -> bool:
        return self._available

    def render(self, source: Path, plan, destination: Path) -> Path:
        if self._fail:
            raise RecutError("recut_failed", "ffmpeg produced no usable clip.")
        self.rendered.append((plan.startSec, plan.endSec))
        destination.write_bytes(b"recut-video")
        return destination


class RecutAssistantTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="recut-")
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)
        self.source = self.root / "clip.mp4"
        self.source.write_bytes(b"video")

    def test_a_render_receives_the_validated_plan(self):
        runner = FakeRunner()
        assistant = RecutAssistant(runner)
        rendered, plan = assistant.recut(
            self.source,
            self.root / "out.mp4",
            operation="trim_start",
            duration_seconds=21.5,
            seconds=1.5,
        )
        self.assertTrue(rendered.is_file())
        self.assertEqual(runner.rendered, [(1.5, 21.5)])
        self.assertEqual(plan.result_duration, 20.0)

    def test_an_unavailable_ffmpeg_never_pretends_to_recut(self):
        assistant = RecutAssistant(FakeRunner(available=False))
        with self.assertRaises(RecutError) as caught:
            assistant.recut(
                self.source,
                self.root / "out.mp4",
                operation="trim_start",
                duration_seconds=21.5,
                seconds=1.0,
            )
        self.assertEqual(caught.exception.reason_code, "recut_unavailable")

    def test_a_failed_render_is_reported_not_swallowed(self):
        assistant = RecutAssistant(FakeRunner(fail=True))
        with self.assertRaises(RecutError) as caught:
            assistant.recut(
                self.source,
                self.root / "out.mp4",
                operation="trim_end",
                duration_seconds=21.5,
                seconds=1.0,
            )
        self.assertEqual(caught.exception.reason_code, "recut_failed")

    def test_a_missing_source_is_refused(self):
        assistant = RecutAssistant(FakeRunner())
        with self.assertRaises(RecutError):
            assistant.recut(
                self.root / "absent.mp4",
                self.root / "out.mp4",
                operation="trim_start",
                duration_seconds=21.5,
                seconds=1.0,
            )


class VariantComparisonTests(unittest.TestCase):
    def variants(self, *silences: float) -> list[Variant]:
        return variants_from_results(
            [cut(index + 1, silence=value) for index, value in enumerate(silences)],
            assemble=assemble_evidence_bundle,
        )

    def test_cuts_are_laid_out_side_by_side(self):
        comparison = build_variant_comparison(self.variants(0.30, 0.12, 0.21))
        self.assertEqual(len(comparison["variants"]), 3)
        silence = next(
            metric for metric in comparison["metrics"] if metric["metricPath"] == SILENCE
        )
        self.assertEqual([entry["value"] for entry in silence["values"]], [0.30, 0.12, 0.21])
        self.assertTrue(silence["differs"])
        self.assertEqual(silence["lowestResultId"], f"{2:032x}")
        self.assertEqual(silence["highestResultId"], f"{1:032x}")

    def test_an_identical_signal_names_no_winner(self):
        comparison = build_variant_comparison(self.variants(0.2, 0.2))
        rms = next(metric for metric in comparison["metrics"] if metric["metricPath"] == RMS)
        self.assertFalse(rms["differs"])
        self.assertIsNone(rms["lowestResultId"])
        self.assertIsNone(rms["highestResultId"])
        self.assertEqual(rms["spread"], 0.0)

    def test_a_metric_missing_from_one_cut_is_skipped_not_imputed(self):
        variants = [
            Variant("a" * 32, "Cut 1", {SILENCE: 0.2, RMS: 0.1}),
            Variant("b" * 32, "Cut 2", {SILENCE: 0.3}),
        ]
        comparison = build_variant_comparison(variants)
        paths = [metric["metricPath"] for metric in comparison["metrics"]]
        self.assertIn(SILENCE, paths)
        self.assertNotIn(RMS, paths)
        skipped = [entry["metricPath"] for entry in comparison["skippedMetrics"]]
        self.assertIn(RMS, skipped)

    def test_the_comparison_refuses_to_rank_a_cut_as_better(self):
        comparison = build_variant_comparison(self.variants(0.3, 0.1))
        self.assertIs(comparison["behavioralOutcome"], False)
        self.assertIn("Higher is not better", comparison["limits"])
        serialized = str(comparison).lower()
        for term in ("winner", "best", "recommend", "viral", "retention", "engagement"):
            self.assertNotIn(term, serialized, term)

    def test_bounds_and_duplicates_are_refused(self):
        with self.assertRaises(VariantComparisonError):
            build_variant_comparison(self.variants(0.2))
        with self.assertRaises(VariantComparisonError):
            build_variant_comparison([Variant("a" * 32, "x", {RMS: 1.0})] * 2)
        with self.assertRaises(VariantComparisonError):
            build_variant_comparison(
                [Variant(f"{i:032x}", "x", {RMS: float(i)}) for i in range(MAXIMUM_VARIANTS + 1)]
            )

    def test_labels_are_carried_through(self):
        variants = variants_from_results(
            [cut(1, silence=0.2), cut(2, silence=0.3)],
            assemble=assemble_evidence_bundle,
            labels={f"{1:032x}": "Original", f"{2:032x}": "Tighter open"},
        )
        comparison = build_variant_comparison(variants)
        self.assertEqual(
            [entry["label"] for entry in comparison["variants"]], ["Original", "Tighter open"]
        )


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="recut-routes-")
        self.addCleanup(self._temporary.cleanup)
        root = Path(self._temporary.name)
        settings = InsightSettings.from_env({"INSIGHT_DIR": str(root / "insight")})
        self.results = {
            FORECAST_RESULT_ID: forecast_result(),
            f"{2:032x}": cut(2, silence=0.12),
        }
        service = InsightService(
            settings,
            forecast_result_loader=self.results.get,
            store=InsightStore(settings.insight_dir),
            provider_factory=lambda _: FakeProvider(),
        )
        app = FastAPI()
        app.include_router(
            create_insight_router(
                service=service, recut_assistant=RecutAssistant(FakeRunner())
            )
        )
        self.client = TestClient(app)

    def test_the_variants_route_compares_analysed_cuts(self):
        response = self.client.post(
            "/api/insight/v1/variants",
            json={"resultIds": [FORECAST_RESULT_ID, f"{2:032x}"], "labels": {f"{2:032x}": "Tighter"}},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["variants"]), 2)
        self.assertEqual(payload["variants"][1]["label"], "Tighter")
        self.assertGreater(payload["differingMetricCount"], 0)

    def test_an_unknown_cut_fails_closed(self):
        response = self.client.post(
            "/api/insight/v1/variants",
            json={"resultIds": [FORECAST_RESULT_ID, "0" * 32]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["unavailable"])
        self.assertEqual(response.json()["reasonCode"], "bundle_unavailable")

    def test_invalid_variant_requests_are_caller_errors(self):
        for body in ({}, {"resultIds": []}, {"resultIds": ["nope"]}, {"resultIds": [FORECAST_RESULT_ID], "extra": 1}):
            with self.subTest(body=body):
                self.assertEqual(
                    self.client.post("/api/insight/v1/variants", json=body).status_code, 400
                )

    def test_the_recut_operations_route_declares_what_it_can_do(self):
        payload = self.client.get("/api/insight/v1/recut/operations").json()
        self.assertTrue(payload["available"])
        self.assertEqual(len(payload["operations"]), 3)
        self.assertIn("submitted as a new evidence job", payload["limits"])

    def test_a_recut_returns_the_clip_and_its_plan(self):
        response = self.client.post(
            "/api/insight/v1/recut",
            files={"video": ("clip.mp4", b"video-bytes", "video/mp4")},
            data={"operation": "trim_start", "durationSeconds": "21.5", "seconds": "1.5"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"recut-video")
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        import json as _json

        plan = _json.loads(response.headers["x-recut-plan"])
        self.assertEqual(plan["operation"], "trim_start")
        self.assertEqual(plan["resultDurationSec"], 20.0)

    def test_a_refused_recut_explains_itself(self):
        response = self.client.post(
            "/api/insight/v1/recut",
            files={"video": ("clip.mp4", b"video-bytes", "video/mp4")},
            data={"operation": "trim_start", "durationSeconds": "21.5", "seconds": "20"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["reasonCode"], "duration_out_of_range")

    def test_a_non_video_upload_is_refused(self):
        response = self.client.post(
            "/api/insight/v1/recut",
            files={"video": ("notes.txt", b"text", "text/plain")},
            data={"operation": "trim_start", "durationSeconds": "21.5", "seconds": "1"},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
