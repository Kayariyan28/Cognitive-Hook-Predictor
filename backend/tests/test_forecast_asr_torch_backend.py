"""The portable transcript backend must produce the same evidence as the MLX one.

The two runtimes load different artifacts, so the manifest has to say which one
ran; everything downstream of that — the schema, the validator, the citations —
is deliberately shared rather than duplicated per backend.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from backend.forecast.workers.asr_whisper import (
    ADAPTER_ID,
    ADAPTER_IDS,
    BACKEND_ENV,
    DEFAULT_TORCH_MODEL_ID,
    MLX_BACKEND,
    SCHEMA_VERSION,
    TORCH_BACKEND,
    AsrUnavailable,
    AsrWhisperAdapter,
    TransformersWhisperTranscriber,
)


PINNED_REVISION = "d" * 40
DURATION_SECONDS = 14.0


class FakeDecoder:
    def decode(self, video_path: Path) -> np.ndarray:
        return np.zeros(int(DURATION_SECONDS * 16_000), dtype=np.float32)


class NormalisationTests(unittest.TestCase):
    """`_normalised` turns pipeline output into the branch's segment shape."""

    def normalise(self, raw, duration: float = DURATION_SECONDS):
        return TransformersWhisperTranscriber._normalised(raw, duration=duration)

    def test_an_open_final_timestamp_closes_at_the_clip_duration(self) -> None:
        result = self.normalise(
            {"chunks": [{"timestamp": (1.0, None), "text": "trailing words"}]}
        )
        self.assertEqual(result["segments"][0]["end"], DURATION_SECONDS)

    def test_language_is_read_from_the_first_chunk_that_carries_one(self) -> None:
        result = self.normalise(
            {
                "chunks": [
                    {"timestamp": (0.0, 1.0), "text": "one"},
                    {"timestamp": (1.0, 2.0), "text": "two", "language": "en"},
                ]
            }
        )
        self.assertEqual(result["language"], "en")

    def test_an_absent_language_is_reported_as_none_rather_than_guessed(self) -> None:
        result = self.normalise({"chunks": [{"timestamp": (0.0, 1.0), "text": "hi"}]})
        self.assertIsNone(result["language"])

    def test_a_malformed_timestamp_fails_closed(self) -> None:
        for raw in (
            {"chunks": [{"timestamp": (0.0,), "text": "x"}]},
            {"chunks": [{"timestamp": None, "text": "x"}]},
            {"chunks": [{"timestamp": (None, 1.0), "text": "x"}]},
            {"chunks": "not a list"},
            "not a mapping",
        ):
            with self.subTest(raw=raw), self.assertRaises(AsrUnavailable):
                self.normalise(raw)


class TorchBackendAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="asr-torch-tests-")
        self.addCleanup(temporary.cleanup)
        self.video = Path(temporary.name) / "clip.mp4"
        self.video.write_bytes(b"hash source only")
        self.input_sha256 = hashlib.sha256(self.video.read_bytes()).hexdigest()

    def test_from_env_selects_the_upstream_model_for_the_torch_backend(self) -> None:
        adapter = AsrWhisperAdapter.from_env(
            {BACKEND_ENV: "transformers", "INSIGHT_ASR_MODEL_REVISION": PINNED_REVISION}
        )
        self.assertEqual(adapter.backend, TORCH_BACKEND)
        # The MLX repository holds MLX-quantised weights torch cannot read, so
        # the portable backend must not inherit that default.
        self.assertEqual(adapter.model_id, DEFAULT_TORCH_MODEL_ID)

    def test_the_default_backend_is_still_mlx(self) -> None:
        self.assertEqual(AsrWhisperAdapter.from_env({}).backend, MLX_BACKEND)

    def test_an_unknown_backend_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            AsrWhisperAdapter.from_env({BACKEND_ENV: "tpu-whisper"})
        with self.assertRaises(ValueError):
            AsrWhisperAdapter(backend="tpu-whisper")

    def test_a_missing_transformers_reports_that_package_by_name(self) -> None:
        adapter = AsrWhisperAdapter(
            model_revision=PINNED_REVISION,
            backend=TORCH_BACKEND,
            decoder=FakeDecoder(),
            module_probe=lambda name: False,
        )
        state = adapter.availability()
        self.assertFalse(state["executionAvailable"])
        self.assertIn("transformers is not installed", state["reason"])
        self.assertEqual(state["provenance"]["adapterId"], ADAPTER_IDS[TORCH_BACKEND])

    def test_an_uncached_pin_is_reported_before_the_job_runs(self) -> None:
        def refuse(model_id: str, revision: str) -> str:
            raise OSError("not in the local cache")

        adapter = AsrWhisperAdapter(
            model_revision=PINNED_REVISION,
            backend=TORCH_BACKEND,
            decoder=FakeDecoder(),
            module_probe=lambda name: True,
            snapshot_resolver=refuse,
        )
        state = adapter.availability()
        self.assertFalse(state["executionAvailable"])
        self.assertIn("not present in the local snapshot cache", state["reason"])

    def test_a_cached_pin_reports_ready(self) -> None:
        adapter = AsrWhisperAdapter(
            model_revision=PINNED_REVISION,
            backend=TORCH_BACKEND,
            decoder=FakeDecoder(),
            module_probe=lambda name: True,
            snapshot_resolver=lambda model_id, revision: "/snapshot",
        )
        self.assertTrue(adapter.availability()["executionAvailable"])

    def test_the_manifest_names_the_backend_that_actually_ran(self) -> None:
        class FakePipeline:
            def __call__(self, audio, **kwargs):
                return {
                    "chunks": [
                        {"timestamp": (0.0, 3.0), "text": "Stop scrolling.", "language": "en"},
                        {"timestamp": (3.0, None), "text": "Here is why."},
                    ]
                }

        transcriber = TransformersWhisperTranscriber(
            model_id=DEFAULT_TORCH_MODEL_ID,
            revision=PINNED_REVISION,
            snapshot_resolver=lambda model_id, revision: "/snapshot",
            pipeline_factory=lambda snapshot: FakePipeline(),
        )
        adapter = AsrWhisperAdapter(
            transcriber=transcriber,
            model_id=DEFAULT_TORCH_MODEL_ID,
            model_revision=PINNED_REVISION,
            backend=TORCH_BACKEND,
            decoder=FakeDecoder(),
        )
        value = adapter.run(
            video_path=self.video,
            input_sha256=self.input_sha256,
            duration_seconds=DURATION_SECONDS,
            context={},
        ).public_value()

        self.assertEqual(value["schemaVersion"], SCHEMA_VERSION)
        self.assertEqual(value["provenance"]["adapterId"], ADAPTER_IDS[TORCH_BACKEND])
        self.assertNotEqual(value["provenance"]["adapterId"], ADAPTER_ID)
        self.assertEqual(value["behavioralOutcome"], False)
        # The open final chunk was closed against the clip, not left dangling.
        self.assertEqual(value["observations"][-1]["endTime"], DURATION_SECONDS)
        self.assertIn("language:en", value["observations"][0]["labels"])


if __name__ == "__main__":
    unittest.main()
