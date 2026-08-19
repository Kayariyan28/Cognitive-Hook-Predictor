from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from backend.forecast.worker_protocol import build_worker_request
from backend.forecast.workers.ast_audio import (
    MODEL_ID as AST_MODEL_ID,
    MODEL_REVISION as AST_MODEL_REVISION,
    MODEL_WEIGHTS_SHA256 as AST_MODEL_WEIGHTS_SHA256,
    AstAudioAdapter,
)
from backend.forecast.workers.measured_audio import (
    MeasuredAudioAdapter,
    SAMPLE_RATE,
)
from backend.forecast.workers.nanollava import (
    MODEL_ID,
    MODEL_REVISION,
    MODEL_WEIGHTS_SHA256,
    NanoLlavaSemanticAdapter,
    deterministic_sample_times,
)


VALID_DESCRIPTION = json.dumps(
    {
        "scene": "A person stands beside a blue table.",
        "action": "The person raises one hand.",
        "visibleText": ["START"],
        "shot": "Medium static shot.",
        "uncertainties": [],
    },
    separators=(",", ":"),
)
VIDEO_BYTES = b"video"
VIDEO_SHA256 = hashlib.sha256(VIDEO_BYTES).hexdigest()


class FakeFrames:
    def extract(self, video_path, duration_seconds, destination):
        del video_path
        values = []
        for index, timestamp in enumerate(deterministic_sample_times(duration_seconds)):
            path = destination / f"fake-{index}.png"
            path.write_bytes(b"png")
            values.append((timestamp, path))
        return tuple(values)


class FakeSemantics:
    def __init__(self, responses=None):
        self.responses = list(responses or [VALID_DESCRIPTION] * 6)
        self.prompts = []

    def describe(self, image_path, prompt):
        self.prompts.append((image_path.name, prompt))
        return self.responses.pop(0)


class FakePcm:
    def __init__(self, samples):
        self.samples = samples

    def decode(self, video_path):
        del video_path
        return self.samples


class FakeAst:
    def classify(self, samples):
        del samples
        return (("Speech", 0.7), ("Music", 0.2), ("Silence", 0.1))


class ForecastLocalWorkerTests(unittest.TestCase):
    def test_nanollava_is_hash_pinned_keyframe_fallback_not_videollama(self):
        self.assertEqual(MODEL_ID, "mlx-community/nanoLLaVA-1.5-8bit")
        self.assertEqual(MODEL_REVISION, "6e2bc13b87ab178668313552b9d69026af7d556f")
        self.assertEqual(
            MODEL_WEIGHTS_SHA256,
            "f0a1d1517ad9e6e810bc6ac99956643c66e4b87a2f82bd5d1b5cb0966e5c5476",
        )
        self.assertEqual(deterministic_sample_times(12), (1.0, 3.0, 5.0, 7.0, 9.0, 11.0))

        backend = FakeSemantics()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.write_bytes(VIDEO_BYTES)
            adapter = NanoLlavaSemanticAdapter(
                root / "snapshot",
                backend=backend,
                frame_extractor=FakeFrames(),
            )
            with mock.patch.object(adapter, "_verify_snapshot"):
                output = adapter.run(
                    video_path=video,
                    input_sha256=VIDEO_SHA256,
                    duration_seconds=12,
                    context={},
                )

        public = output.public_value()
        self.assertEqual(public["branch"], "semanticModel")
        self.assertEqual(public["evidenceKind"], "learned-keyframe-semantics")
        self.assertFalse(public["behavioralOutcome"])
        self.assertEqual(public["features"]["nanollava.keyframes_described"], 6.0)
        self.assertEqual(len(public["observations"]), 6)
        self.assertTrue(all("videollama" not in key.lower() for key in public["features"]))
        self.assertIn("Do not predict attention", backend.prompts[0][1])

    def test_nanollava_omits_non_json_model_text_instead_of_trusting_it(self):
        backend = FakeSemantics(["```json\n{}\n```", *([VALID_DESCRIPTION] * 5)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.write_bytes(VIDEO_BYTES)
            adapter = NanoLlavaSemanticAdapter(
                root / "snapshot",
                backend=backend,
                frame_extractor=FakeFrames(),
            )
            with mock.patch.object(adapter, "_verify_snapshot"):
                output = adapter.run(
                    video_path=video,
                    input_sha256=VIDEO_SHA256,
                    duration_seconds=10,
                    context={},
                )
        self.assertEqual(output.features["nanollava.keyframes_described"], 5.0)
        self.assertTrue(any("strict semantic JSON" in item for item in output.warnings))

    def test_measured_audio_reports_signal_facts_without_semantic_or_behavioral_claims(self):
        seconds = 2
        time = np.arange(seconds * SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
        samples = 0.25 * np.sin(2 * np.pi * 440 * time)
        samples[: SAMPLE_RATE // 2] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "clip.mp4"
            video.write_bytes(VIDEO_BYTES)
            output = MeasuredAudioAdapter(FakePcm(samples)).run(
                video_path=video,
                input_sha256=VIDEO_SHA256,
                duration_seconds=10,
                context={},
            )

        public = output.public_value()
        self.assertEqual(public["branch"], "measuredAudio")
        self.assertFalse(public["behavioralOutcome"])
        self.assertFalse(public["provenance"]["usesLearnedModel"])
        self.assertEqual(public["features"]["measured_audio.present"], 1.0)
        self.assertGreater(public["features"]["measured_audio.spectral_centroid_hz_mean"], 0)
        serialized = json.dumps(public).lower()
        self.assertNotIn("speech_probability", serialized)
        self.assertNotIn("music_probability", serialized)
        self.assertNotIn("engagement_score", serialized)

    def test_measured_audio_cleanly_records_no_decodable_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "clip.mp4"
            video.write_bytes(VIDEO_BYTES)
            output = MeasuredAudioAdapter(FakePcm(np.array([], dtype=np.float64))).run(
                video_path=video,
                input_sha256=VIDEO_SHA256,
                duration_seconds=10,
                context={},
            )
        self.assertEqual(output.features, {
            "measured_audio.present": 0.0,
            "measured_audio.decoded_seconds": 0.0,
            "measured_audio.sample_count": 0.0,
        })

    def test_ast_is_pinned_and_returns_only_descriptive_audioset_windows(self):
        self.assertEqual(AST_MODEL_ID, "MIT/ast-finetuned-audioset-10-10-0.4593")
        self.assertEqual(AST_MODEL_REVISION, "f826b80d28226b62986cc218e5cec390b1096902")
        self.assertEqual(
            AST_MODEL_WEIGHTS_SHA256,
            "ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d",
        )
        samples = np.zeros(12 * SAMPLE_RATE, dtype=np.float64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.write_bytes(VIDEO_BYTES)
            adapter = AstAudioAdapter(
                root / "snapshot", backend=FakeAst(), decoder=FakePcm(samples)
            )
            with mock.patch.object(adapter, "_verify_snapshot"):
                output = adapter.run(
                    video_path=video,
                    input_sha256=VIDEO_SHA256,
                    duration_seconds=12,
                    context={},
                )
        public = output.public_value()
        self.assertEqual(public["branch"], "audioModel")
        self.assertEqual(public["features"]["audio.ast.windows_classified"], 2.0)
        self.assertEqual(len(public["observations"]), 2)
        self.assertFalse(public["behavioralOutcome"])
        self.assertIn("AudioSet", public["warnings"][0])

    def test_semantic_worker_request_is_job_only_and_duration_bound(self):
        request = build_worker_request(
            branch="semanticModel",
            input_sha256="e" * 64,
            duration_seconds=60,
            context={},
        )
        self.assertEqual(request["branch"], "semanticModel")

    def test_local_workers_reject_an_input_hash_mismatch_before_inference(self):
        semantic_backend = FakeSemantics()
        audio_decoder = FakePcm(np.zeros(SAMPLE_RATE, dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.write_bytes(VIDEO_BYTES)
            semantic = NanoLlavaSemanticAdapter(
                root / "snapshot",
                backend=semantic_backend,
                frame_extractor=FakeFrames(),
            )
            with mock.patch.object(semantic, "_verify_snapshot"):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    semantic.run(
                        video_path=video,
                        input_sha256="f" * 64,
                        duration_seconds=10,
                        context={},
                    )
            with self.assertRaisesRegex(ValueError, "does not match"):
                MeasuredAudioAdapter(audio_decoder).run(
                    video_path=video,
                    input_sha256="f" * 64,
                    duration_seconds=10,
                    context={},
                )
            ast = AstAudioAdapter(
                root / "ast", backend=FakeAst(), decoder=audio_decoder
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                ast.run(
                    video_path=video,
                    input_sha256="f" * 64,
                    duration_seconds=10,
                    context={},
                )
        self.assertEqual(semantic_backend.prompts, [])


if __name__ == "__main__":
    unittest.main()
