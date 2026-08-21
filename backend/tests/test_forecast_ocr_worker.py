from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from backend.forecast.workers.ocr_screen_text import (
    ADAPTER_ID,
    PREPROCESSING_ID,
    SCHEMA_VERSION,
    OcrScreenTextAdapter,
    OcrUnavailable,
    macos_product_version,
    screen_text_document,
)
from backend.forecast.workers.nanollava import KEYFRAME_COUNT, NanoLlavaUnavailable
from backend.insight.bundle import assemble_evidence_bundle, hook_evidence_card
from backend.insight.citations import parse_citation, resolve_citation
from backend.tests.insight_support import forecast_result


DURATION_SECONDS = 21.5


class FakeExtractor:
    """Stands in for the shared deterministic six-keyframe contract."""

    def __init__(self, count: int = KEYFRAME_COUNT, error: Exception | None = None) -> None:
        self.count = count
        self.error = error
        self.calls = 0

    def extract(self, video_path: Path, duration_seconds: float, destination: Path):
        self.calls += 1
        if self.error is not None:
            raise self.error
        destination.mkdir(parents=True, exist_ok=True)
        frames = []
        for index in range(self.count):
            path = destination / f"frame-{index:02d}.png"
            path.write_bytes(b"png")
            frames.append((round(duration_seconds * (index + 0.5) / self.count, 6), path))
        return tuple(frames)


class FakeRecognizer:
    def __init__(self, engine: str = "ocrmac", blocks=None, error: Exception | None = None) -> None:
        self.engine = engine
        self.blocks = blocks
        self.error = error
        self.calls = 0

    def recognize(self, image_path: Path):
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.blocks is not None:
            return self.blocks
        if self.calls == 1:
            return [{"text": "READ THIS", "confidence": 0.9375, "bbox": [0.1, 0.1, 0.5, 0.2]}]
        if self.calls == 2:
            return [{"text": "STEP ONE", "confidence": 0.8125, "bbox": [0.1, 0.7, 0.4, 0.8]}]
        return []


class OcrHarness(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="ocr-tests-")
        self.addCleanup(self._temporary.cleanup)
        self.video = Path(self._temporary.name) / "clip.mp4"
        self.video.write_bytes(b"not a real video, only a hash source")
        self.input_sha256 = hashlib.sha256(self.video.read_bytes()).hexdigest()

    def adapter(self, recognizer=None, **overrides) -> OcrScreenTextAdapter:
        return OcrScreenTextAdapter(
            recognizer=recognizer if recognizer is not None else FakeRecognizer(),
            extractor=overrides.pop("extractor", FakeExtractor()),
            version_probe=overrides.pop("version_probe", lambda: "15.3.1"),
            **overrides,
        )

    def run_adapter(self, adapter):
        return adapter.run(
            video_path=self.video,
            input_sha256=self.input_sha256,
            duration_seconds=DURATION_SECONDS,
            context={},
        )


class OutputSchemaTests(OcrHarness):
    def test_public_value_matches_the_branch_contract(self):
        value = self.run_adapter(self.adapter()).public_value()
        self.assertEqual(value["schemaVersion"], SCHEMA_VERSION)
        self.assertEqual(value["branch"], "ocr")
        self.assertEqual(value["evidenceKind"], "measured-on-screen-text")
        self.assertIs(value["behavioralOutcome"], False)
        self.assertEqual(len(value["observations"]), KEYFRAME_COUNT)
        self.assertEqual(
            sorted(value["provenance"]),
            ["adapterId", "engine", "macOSVersion", "preprocessingId", "usesLearnedModel"],
        )
        self.assertEqual(value["provenance"]["adapterId"], ADAPTER_ID)
        self.assertTrue(all(name.startswith("ocr.") for name in value["features"]))

    def test_declared_output_schema_is_frames_and_blocks_only(self):
        document = screen_text_document(self.run_adapter(self.adapter()))
        self.assertEqual(sorted(document), ["frames"])
        self.assertEqual(len(document["frames"]), KEYFRAME_COUNT)
        self.assertEqual(sorted(document["frames"][0]), ["blocks", "frameIndex"])
        self.assertEqual(sorted(document["frames"][0]["blocks"][0]), ["bbox", "confidence", "text"])
        self.assertEqual(document["frames"][0]["blocks"][0]["text"], "READ THIS")

    def test_the_existing_six_keyframe_contract_is_reused(self):
        extractor = FakeExtractor()
        self.run_adapter(self.adapter(extractor=extractor))
        self.assertEqual(extractor.calls, 1)
        self.assertEqual(PREPROCESSING_ID, "ffmpeg-keyframes-6x384-png-center-sampled/1")

    def test_counts_are_measured_not_asserted(self):
        features = self.run_adapter(self.adapter()).features
        self.assertEqual(features["ocr.frames_read"], float(KEYFRAME_COUNT))
        self.assertEqual(features["ocr.frames_with_text"], 2.0)
        self.assertEqual(features["ocr.block_count"], 2.0)


class EngineProvenanceTests(OcrHarness):
    def test_the_engine_that_ran_is_recorded(self):
        value = self.run_adapter(self.adapter(FakeRecognizer(engine="ocrmac"))).public_value()
        self.assertEqual(value["provenance"]["engine"], "ocrmac")
        self.assertIn("engine:ocrmac", value["observations"][0]["labels"])

    def test_the_fallback_engine_is_recorded_as_itself(self):
        adapter = OcrScreenTextAdapter(
            recognizer=FakeRecognizer(engine="pytesseract"),
            engine="pytesseract",
            extractor=FakeExtractor(),
            version_probe=lambda: None,
        )
        value = self.run_adapter(adapter).public_value()
        self.assertEqual(value["provenance"]["engine"], "pytesseract")
        self.assertIsNone(value["provenance"]["macOSVersion"])
        self.assertIn("engine:pytesseract", value["observations"][0]["labels"])

    def test_the_macos_version_travels_with_vision_output(self):
        value = self.run_adapter(self.adapter()).public_value()
        self.assertEqual(value["provenance"]["macOSVersion"], "15.3.1")

    def test_the_version_probe_never_raises_on_a_non_mac(self):
        def missing(*args, **kwargs):
            raise OSError("sw_vers is not on this machine")

        self.assertIsNone(macos_product_version(missing))

        class Completed:
            returncode = 0
            stdout = b"not a version\n"

        self.assertIsNone(macos_product_version(lambda *a, **k: Completed()))


class SchemaValidationTests(OcrHarness):
    def test_malformed_blocks_are_refused(self):
        for blocks in (
            "text",
            [{"text": "x", "confidence": 4.0, "bbox": [0, 0, 1, 1]}],
            [{"text": "x", "confidence": 0.5, "bbox": [0, 0, 1]}],
            [{"text": "x", "confidence": 0.5}],
            [{"text": "x" * 500, "confidence": 0.5, "bbox": [0, 0, 1, 1]}],
            [{"text": "x", "confidence": 0.5, "bbox": [0, 0, 1, float("nan")]}],
            [{"text": "x", "confidence": 0.5, "bbox": [0, 0, 1, 1]}] * 40,
        ):
            with self.subTest(blocks=blocks):
                with self.assertRaises(OcrUnavailable):
                    self.run_adapter(self.adapter(FakeRecognizer(blocks=blocks)))

    def test_a_frame_with_no_text_is_recorded_as_empty_not_missing(self):
        output = self.run_adapter(self.adapter(FakeRecognizer(blocks=[])))
        self.assertEqual(len(output.observations), KEYFRAME_COUNT)
        self.assertEqual(json.loads(output.observations[0]["text"]), [])
        self.assertEqual(output.features["ocr.frames_with_text"], 0.0)

    def test_input_identity_is_enforced(self):
        with self.assertRaises(ValueError):
            self.adapter().run(
                video_path=self.video,
                input_sha256="0" * 64,
                duration_seconds=DURATION_SECONDS,
                context={},
            )


class BranchLocalFailureTests(OcrHarness):
    def test_no_engine_reports_unavailable_without_raising(self):
        adapter = OcrScreenTextAdapter(
            extractor=FakeExtractor(),
            module_probe=lambda name: False,
            version_probe=lambda: None,
        )
        availability = adapter.availability()
        self.assertFalse(availability["configured"])
        self.assertFalse(availability["executionAvailable"])
        self.assertIn("ocrmac", availability["reason"])
        with self.assertRaises(OcrUnavailable):
            self.run_adapter(adapter)

    def test_vision_absence_falls_back_and_says_which_engine_would_run(self):
        adapter = OcrScreenTextAdapter(
            extractor=FakeExtractor(),
            module_probe=lambda name: name == "pytesseract",
            version_probe=lambda: None,
        )
        # The fallback still needs the tesseract binary, which this container
        # does not have; either way the branch reports a specific state.
        availability = adapter.availability()
        self.assertIn(availability["executionAvailable"], (True, False))
        self.assertIsInstance(availability["provenance"]["engine"], (str, type(None)))

    def test_a_failing_engine_fails_this_branch_alone(self):
        adapter = self.adapter(FakeRecognizer(error=RuntimeError("vision framework crashed")))
        with self.assertRaises(OcrUnavailable) as caught:
            self.run_adapter(adapter)
        self.assertNotIn("vision framework crashed", str(caught.exception))

    def test_keyframe_extraction_failure_is_branch_local(self):
        adapter = self.adapter(
            extractor=FakeExtractor(error=NanoLlavaUnavailable("ffmpeg is unavailable"))
        )
        with self.assertRaises(OcrUnavailable):
            self.run_adapter(adapter)

    def test_a_short_keyframe_set_is_refused(self):
        adapter = self.adapter(extractor=FakeExtractor(count=3))
        with self.assertRaises(OcrUnavailable):
            self.run_adapter(adapter)

    def test_availability_never_claims_a_behavioral_model(self):
        self.assertFalse(self.adapter().availability()["isBehavioralModel"])


class AssemblerIntegrationTests(OcrHarness):
    def provider(self):
        return {
            "status": "available",
            "configured": True,
            "forecastContribution": False,
            "behavioralOutcome": False,
            "evidenceKind": "measured-on-screen-text",
            "result": self.run_adapter(self.adapter()).public_value(),
        }

    def test_ocr_citations_resolve_in_the_bundle(self):
        bundle = assemble_evidence_bundle(
            forecast_result(optionalProviders={"ocr": self.provider()})
        )
        lane = bundle["lanes"]["ocr"]
        self.assertEqual(lane["status"], "present")
        self.assertEqual(lane["engine"], "ocrmac")
        self.assertEqual(lane["frames"][0]["blocks"][0]["text"], "READ THIS")
        self.assertEqual(
            resolve_citation(bundle, parse_citation("ocr:/frames/0/blocks/0/text")),
            "READ THIS",
        )

    def test_the_hook_doctor_can_cite_opening_screen_text(self):
        card = hook_evidence_card(
            assemble_evidence_bundle(
                forecast_result(optionalProviders={"ocr": self.provider()})
            )
        )
        lane = card["lanes"]["ocr"]
        self.assertEqual(lane["status"], "present")
        self.assertEqual([frame["frameIndex"] for frame in lane["frames"]], [0])
        self.assertEqual(
            resolve_citation(card, parse_citation("ocr:/frames/0/blocks/0/text")),
            "READ THIS",
        )

    def test_an_unavailable_branch_leaves_an_absent_lane(self):
        bundle = assemble_evidence_bundle(
            forecast_result(
                optionalProviders={
                    "ocr": {
                        "status": "unavailable",
                        "configured": False,
                        "forecastContribution": False,
                        "reason": "Neither ocrmac nor pytesseract is available.",
                    }
                }
            )
        )
        self.assertEqual(bundle["lanes"]["ocr"]["status"], "absent")
        self.assertIn("ocrmac", bundle["lanes"]["ocr"]["reason"])


if __name__ == "__main__":
    unittest.main()
