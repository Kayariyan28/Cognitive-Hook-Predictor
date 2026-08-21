"""The hook readout is deterministic and needs no model to be useful."""

from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.insight.bundle import assemble_evidence_bundle, hook_evidence_card
from backend.insight.config import InsightSettings
from backend.insight.hook_readout import (
    FIRST_WORDS_SECONDS,
    OPENING_SILENCE_SECONDS,
    READOUT_SCHEMA_VERSION,
    build_checklist,
    build_hook_readout,
    build_timeline,
    readout_digest,
)
from backend.insight.citations import parse_citation, resolve_citation
from backend.insight.router import create_insight_router
from backend.insight.service import InsightRequest, InsightService
from backend.insight.store import InsightStore
from backend.tests.insight_support import (
    FORECAST_RESULT_ID,
    asr_provider,
    forecast_result,
    ocr_provider,
    tribe_descriptors,
    unavailable_provider,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def full_card(**providers):
    lanes = {"asr": asr_provider(), "ocr": ocr_provider()}
    lanes.update(providers)
    return hook_evidence_card(
        assemble_evidence_bundle(
            forecast_result(optionalProviders=lanes), tribe_descriptors=tribe_descriptors()
        )
    )


def checks_by_id(bundle):
    return {check["id"]: check for check in build_checklist(bundle)}


class TimelineTests(unittest.TestCase):
    def test_every_marker_carries_a_resolvable_citation(self):
        card = full_card()
        markers = build_timeline(card)
        self.assertTrue(markers)
        for marker in markers:
            citation = parse_citation(marker["citation"])
            self.assertIsNotNone(resolve_citation(card, citation))

    def test_markers_are_ordered_and_typed(self):
        markers = build_timeline(full_card())
        starts = [marker["startSec"] for marker in markers]
        self.assertEqual(starts, sorted(starts))
        kinds = {marker["kind"] for marker in markers}
        self.assertIn("spoken-segment", kinds)
        self.assertIn("on-screen-text", kinds)
        self.assertIn("cortical-interval", kinds)
        self.assertIn("audio-peak", kinds)

    def test_a_lane_with_nothing_in_the_window_contributes_no_markers(self):
        card = full_card(
            asr=unavailable_provider("mlx-whisper is not installed."),
            ocr=unavailable_provider("No recognition engine is available."),
        )
        kinds = {marker["kind"] for marker in build_timeline(card)}
        self.assertNotIn("spoken-segment", kinds)
        self.assertNotIn("on-screen-text", kinds)

    def test_the_timeline_never_invents_a_marker(self):
        card = hook_evidence_card(assemble_evidence_bundle(forecast_result()))
        for marker in build_timeline(card):
            self.assertIsNotNone(resolve_citation(card, parse_citation(marker["citation"])))


class ChecklistTests(unittest.TestCase):
    def test_a_quiet_opening_is_flagged_against_a_declared_convention(self):
        check = checks_by_id(full_card())["opening_silence"]
        self.assertEqual(check["status"], "flagged")
        self.assertEqual(check["measured"], 1.44)
        self.assertEqual(check["threshold"], OPENING_SILENCE_SECONDS)
        self.assertEqual(check["thresholdKind"], "declared-convention")

    def test_speech_that_starts_immediately_is_clear(self):
        check = checks_by_id(full_card())["first_words_late"]
        self.assertEqual(check["status"], "clear")
        self.assertEqual(check["measured"], 0.0)
        self.assertEqual(check["threshold"], FIRST_WORDS_SECONDS)

    def test_late_speech_is_flagged(self):
        provider = asr_provider()
        for observation in provider["result"]["observations"]:
            observation["startTime"] += 2.0
            observation["endTime"] += 2.0
        check = checks_by_id(full_card(asr=provider))["first_words_late"]
        self.assertEqual(check["status"], "flagged")
        self.assertEqual(check["measured"], 2.0)

    def test_an_absent_lane_is_unmeasured_never_clear(self):
        card = full_card(
            asr=unavailable_provider("mlx-whisper is not installed."),
            ocr=unavailable_provider("No recognition engine is available."),
            vjepa21=unavailable_provider("No V-JEPA artifact is registered."),
            semanticModel=unavailable_provider("No NanoLLaVA snapshot is configured."),
        )
        checks = checks_by_id(card)
        for check_id in (
            "first_words_late",
            "no_opening_text",
            "low_visual_change",
            "opening_frame_described",
        ):
            with self.subTest(check=check_id):
                self.assertEqual(checks[check_id]["status"], "unmeasured", check_id)

    def test_a_keyframe_with_no_recognized_text_is_flagged(self):
        provider = ocr_provider()
        for observation in provider["result"]["observations"]:
            observation["text"] = "[]"
        check = checks_by_id(full_card(ocr=provider))["no_opening_text"]
        self.assertEqual(check["status"], "flagged")
        self.assertEqual(check["measured"], 0.0)

    def test_every_check_cites_the_evidence_it_used(self):
        card = full_card()
        for check in build_checklist(card):
            if check["status"] == "unmeasured":
                continue
            self.assertTrue(check["citations"], check["id"])
            for citation in check["citations"]:
                resolve_citation(card, parse_citation(citation))


class ReadoutTests(unittest.TestCase):
    def test_the_readout_declares_its_window_and_its_limits(self):
        readout = build_hook_readout(full_card())
        self.assertEqual(readout["schemaVersion"], READOUT_SCHEMA_VERSION)
        self.assertEqual(readout["windowSeconds"], [0.0, 3.0])
        self.assertIs(readout["behavioralOutcome"], False)
        self.assertIn("declared as a convention", readout["limits"])
        self.assertIn("a flag is not a defect", readout["limits"])

    def test_the_readout_is_deterministic(self):
        self.assertEqual(
            readout_digest(build_hook_readout(full_card())),
            readout_digest(build_hook_readout(full_card())),
        )

    def test_counts_match_the_checklist(self):
        readout = build_hook_readout(full_card())
        self.assertEqual(
            readout["flaggedCount"],
            sum(1 for check in readout["checklist"] if check["status"] == "flagged"),
        )

    def test_the_readout_module_never_reaches_a_provider(self):
        source = (REPOSITORY_ROOT / "backend/insight/hook_readout.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        for forbidden in ("providers", "prompts", "judge", "outcomes", "anthropic", "mlx_lm"):
            self.assertNotIn(forbidden, imported)


class ReadoutEndpointTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="readout-")
        self.addCleanup(self._temporary.cleanup)
        settings = InsightSettings.from_env(
            {"INSIGHT_DIR": str(Path(self._temporary.name) / "insight")}
        )

        def explode(_settings):
            raise AssertionError("the readout route must never build a provider")

        service = InsightService(
            settings,
            forecast_result_loader={
                FORECAST_RESULT_ID: forecast_result(
                    optionalProviders={"asr": asr_provider(), "ocr": ocr_provider()}
                )
            }.get,
            store=InsightStore(settings.insight_dir),
            provider_factory=explode,
        )
        app = FastAPI()
        app.include_router(create_insight_router(service=service))
        self.client = TestClient(app)

    def test_the_route_works_with_no_model_installed(self):
        response = self.client.post(
            "/api/insight/v1/hook-readout", json={"forecastResultId": FORECAST_RESULT_ID}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        payload = response.json()
        self.assertEqual(payload["schemaVersion"], READOUT_SCHEMA_VERSION)
        self.assertTrue(payload["timeline"])
        self.assertEqual(len(payload["checklist"]), 5)

    def test_a_missing_result_fails_closed(self):
        response = self.client.post(
            "/api/insight/v1/hook-readout", json={"forecastResultId": "0" * 32}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["unavailable"])
        self.assertEqual(response.json()["reasonCode"], "bundle_unavailable")


if __name__ == "__main__":
    unittest.main()
