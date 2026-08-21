from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from backend.forecast.workers.asr_whisper import (
    ADAPTER_ID,
    PREPROCESSING_ID,
    SCHEMA_VERSION,
    AsrUnavailable,
    AsrWhisperAdapter,
    transcript_document,
)
from backend.insight.bundle import assemble_evidence_bundle
from backend.insight.citations import parse_citation, resolve_citation
from backend.tests.insight_support import forecast_result


PINNED_REVISION = "c" * 40
DURATION_SECONDS = 21.5


class FakeDecoder:
    def __init__(self, seconds: float = DURATION_SECONDS) -> None:
        self.samples = np.zeros(int(seconds * 16_000), dtype=np.float32)
        self.calls = 0

    def decode(self, video_path: Path) -> np.ndarray:
        self.calls += 1
        return self.samples


class FakeTranscriber:
    def __init__(self, payload=None, error: Exception | None = None) -> None:
        self.payload = payload if payload is not None else {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 2.6, "text": " Stop scrolling, this jar changed my kitchen. "},
                {"start": 2.6, "end": 12.5, "text": "You only need three things."},
            ],
        }
        self.error = error
        self.calls = 0

    def transcribe(self, samples):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.payload


class AsrHarness(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="asr-tests-")
        self.addCleanup(self._temporary.cleanup)
        self.video = Path(self._temporary.name) / "clip.mp4"
        self.video.write_bytes(b"not a real video, only a hash source")
        self.input_sha256 = hashlib.sha256(self.video.read_bytes()).hexdigest()

    def adapter(self, transcriber=None, **overrides) -> AsrWhisperAdapter:
        return AsrWhisperAdapter(
            transcriber=transcriber or FakeTranscriber(),
            model_revision=overrides.pop("model_revision", PINNED_REVISION),
            decoder=overrides.pop("decoder", FakeDecoder()),
            **overrides,
        )

    def run_adapter(self, adapter):
        return adapter.run(
            video_path=self.video,
            input_sha256=self.input_sha256,
            duration_seconds=DURATION_SECONDS,
            context={},
        )


class OutputSchemaTests(AsrHarness):
    def test_public_value_matches_the_branch_contract(self):
        value = self.run_adapter(self.adapter()).public_value()
        self.assertEqual(value["schemaVersion"], SCHEMA_VERSION)
        self.assertEqual(value["branch"], "asr")
        self.assertEqual(value["evidenceKind"], "measured-speech-transcript")
        self.assertIs(value["behavioralOutcome"], False)
        self.assertEqual(value["inputSha256"], self.input_sha256)
        self.assertEqual(
            sorted(value["provenance"]),
            ["adapterId", "modelId", "modelRevision", "preprocessingId", "sampleRateHz", "usesLearnedModel"],
        )
        self.assertEqual(value["provenance"]["adapterId"], ADAPTER_ID)
        self.assertEqual(value["provenance"]["preprocessingId"], PREPROCESSING_ID)
        self.assertEqual(value["provenance"]["modelRevision"], PINNED_REVISION)
        self.assertTrue(all(name.startswith("asr.") for name in value["features"]))

    def test_declared_output_schema_is_language_and_segments_only(self):
        document = transcript_document(self.run_adapter(self.adapter()))
        self.assertEqual(sorted(document), ["language", "segments"])
        self.assertEqual(document["language"], "en")
        self.assertEqual(sorted(document["segments"][0]), ["endSec", "startSec", "text"])
        self.assertEqual(document["segments"][0]["startSec"], 0.0)

    def test_no_speaker_identity_or_sentiment_is_ever_recorded(self):
        transcriber = FakeTranscriber(
            {
                "language": "en",
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.6,
                        "text": "Stop scrolling.",
                        "speaker": "SPEAKER_01",
                        "sentiment": "excited",
                        "confidence": 0.91,
                    }
                ],
            }
        )
        value = self.run_adapter(self.adapter(transcriber)).public_value()
        serialized = str(value)
        self.assertNotIn("SPEAKER_01", serialized)
        self.assertNotIn("excited", serialized)
        self.assertIn("no-speaker-identity", value["observations"][0]["labels"])
        self.assertIn("no-sentiment", value["observations"][0]["labels"])

    def test_timings_are_deterministic_and_bounded(self):
        observations = self.run_adapter(self.adapter()).observations
        self.assertEqual([item["startTime"] for item in observations], [0.0, 2.6])
        self.assertEqual([item["endTime"] for item in observations], [2.6, 12.5])
        for item in observations:
            self.assertLess(item["startTime"], item["endTime"])
            self.assertLessEqual(item["endTime"], DURATION_SECONDS + 0.5)

    def test_the_existing_decode_contract_is_reused_once(self):
        decoder = FakeDecoder()
        self.run_adapter(self.adapter(decoder=decoder))
        self.assertEqual(decoder.calls, 1)
        self.assertEqual(PREPROCESSING_ID, "ffmpeg-f32le-mono-16khz/1")


class SchemaValidationTests(AsrHarness):
    def test_malformed_transcripts_are_refused(self):
        for payload in (
            {"language": "en"},
            {"language": "en", "segments": "words"},
            {"language": "not a language tag", "segments": []},
            {"language": "en", "segments": [{"start": 2.0, "end": 1.0, "text": "backwards"}]},
            {"language": "en", "segments": [{"start": 0.0, "end": float("inf"), "text": "x"}]},
            {"language": "en", "segments": [{"start": 0.0, "end": 900.0, "text": "beyond the clip"}]},
            {"language": "en", "segments": [{"start": 0.0, "end": 1.0, "text": "x" * 2000}]},
            {"language": "en", "segments": [{"start": 0.0, "end": 1.0}]},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(AsrUnavailable):
                    self.run_adapter(self.adapter(FakeTranscriber(payload)))

    def test_input_identity_is_enforced(self):
        with self.assertRaises(ValueError):
            self.adapter().run(
                video_path=self.video,
                input_sha256="0" * 64,
                duration_seconds=DURATION_SECONDS,
                context={},
            )
        with self.assertRaises(ValueError):
            self.adapter().run(
                video_path=self.video,
                input_sha256=self.input_sha256,
                duration_seconds=120.0,
                context={},
            )


class BranchLocalFailureTests(AsrHarness):
    def test_a_mutable_revision_is_never_executable(self):
        adapter = self.adapter(model_revision="main")
        availability = adapter.availability()
        self.assertFalse(availability["configured"])
        self.assertFalse(availability["executionAvailable"])
        self.assertIn("40-character commit SHA", availability["reason"])
        with self.assertRaises(AsrUnavailable):
            self.run_adapter(adapter)

    def test_a_missing_runtime_reports_unavailable_without_raising(self):
        adapter = AsrWhisperAdapter(
            model_revision=PINNED_REVISION,
            decoder=FakeDecoder(),
            module_probe=lambda name: False,
        )
        availability = adapter.availability()
        self.assertTrue(availability["configured"])
        self.assertFalse(availability["executionAvailable"])
        self.assertIn("mlx-whisper is not installed", availability["reason"])

    def test_a_failing_model_fails_this_branch_alone(self):
        adapter = self.adapter(FakeTranscriber(error=RuntimeError("weights missing")))
        with self.assertRaises(AsrUnavailable) as caught:
            self.run_adapter(adapter)
        self.assertNotIn("weights missing", str(caught.exception))

    def test_availability_never_claims_a_behavioral_model(self):
        availability = self.adapter().availability()
        self.assertFalse(availability["isBehavioralModel"])
        self.assertTrue(availability["isSpeechTranscript"])


class AssemblerIntegrationTests(AsrHarness):
    def provider(self):
        return {
            "status": "available",
            "configured": True,
            "forecastContribution": False,
            "behavioralOutcome": False,
            "evidenceKind": "measured-speech-transcript",
            "result": self.run_adapter(self.adapter()).public_value(),
        }

    def test_asr_citations_resolve_in_the_bundle(self):
        bundle = assemble_evidence_bundle(
            forecast_result(optionalProviders={"asr": self.provider()})
        )
        lane = bundle["lanes"]["asr"]
        self.assertEqual(lane["status"], "present")
        self.assertEqual(lane["language"], "en")
        self.assertEqual(lane["segments"][0]["text"], "Stop scrolling, this jar changed my kitchen.")
        resolved = resolve_citation(bundle, parse_citation("asr:/segments/0/text"))
        self.assertEqual(resolved, "Stop scrolling, this jar changed my kitchen.")

    def test_the_hook_doctor_can_cite_the_first_spoken_sentence(self):
        from backend.insight.bundle import hook_evidence_card

        card = hook_evidence_card(
            assemble_evidence_bundle(
                forecast_result(optionalProviders={"asr": self.provider()})
            )
        )
        self.assertEqual(card["lanes"]["asr"]["status"], "present")
        self.assertEqual(len(card["lanes"]["asr"]["segments"]), 2)
        resolved = resolve_citation(
            card, parse_citation("asr:/segments/0@window(0.0,2.6)")
        )
        self.assertEqual(resolved["startSec"], 0.0)

    def test_an_unavailable_branch_leaves_an_absent_lane(self):
        bundle = assemble_evidence_bundle(
            forecast_result(
                optionalProviders={
                    "asr": {
                        "status": "unavailable",
                        "configured": False,
                        "forecastContribution": False,
                        "reason": "mlx-whisper is not installed in this backend environment.",
                    }
                }
            )
        )
        self.assertEqual(bundle["lanes"]["asr"]["status"], "absent")
        self.assertIn("mlx-whisper", bundle["lanes"]["asr"]["reason"])


if __name__ == "__main__":
    unittest.main()
