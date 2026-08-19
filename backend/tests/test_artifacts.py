from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from backend.artifacts import (
    SURFACE_MAPPING_ID,
    VERTEX_COUNT,
    build_manifest,
    calculate_display_scale,
    extract_segment_timing,
    prepare_predictions,
    write_frames,
)
from backend.errors import PredictionValidationError


VIDEO_EXTRACTOR = {
    "id": "facebook/vjepa2-vitg-fpc64-256",
    "revision": "875c192b7b704b87d1e1d99345769632dd5f739a",
    "weightsSha256": "f205e77aa2ade168db6b09d4bc420d156141f64ab964278a9c181a2bdf2a232b",
    "device": "cpu",
    "precision": "fp32",
}
MODEL_WEIGHTS_SHA256 = "9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321"


class PredictionValidationTests(unittest.TestCase):
    def test_prepares_exact_fsaverage5_shape_as_little_endian_float32(self) -> None:
        source = np.arange(2 * VERTEX_COUNT, dtype=np.float64).reshape(
            2, VERTEX_COUNT
        )
        prepared = prepare_predictions(source)

        self.assertEqual(prepared.shape, (2, VERTEX_COUNT))
        self.assertEqual(prepared.dtype, np.dtype("<f4"))
        self.assertTrue(prepared.flags.c_contiguous)
        self.assertEqual(float(prepared[1, 17]), float(source[1, 17]))

    def test_rejects_wrong_vertex_count(self) -> None:
        with self.assertRaisesRegex(PredictionValidationError, "exactly 20484"):
            prepare_predictions(np.zeros((3, VERTEX_COUNT - 1), dtype=np.float32))

    def test_rejects_no_frames(self) -> None:
        with self.assertRaisesRegex(PredictionValidationError, "no prediction frames"):
            prepare_predictions(np.zeros((0, VERTEX_COUNT), dtype=np.float32))

    def test_rejects_non_finite_values(self) -> None:
        predictions = np.zeros((1, VERTEX_COUNT), dtype=np.float32)
        predictions[0, 42] = np.nan
        with self.assertRaisesRegex(PredictionValidationError, "NaN or infinity"):
            prepare_predictions(predictions)

    def test_rejects_complex_values(self) -> None:
        with self.assertRaisesRegex(PredictionValidationError, "Complex"):
            prepare_predictions(np.zeros((1, VERTEX_COUNT), dtype=np.complex64))


class SegmentTimingTests(unittest.TestCase):
    def test_uses_returned_segment_start_and_duration_exactly(self) -> None:
        timing = extract_segment_timing(
            [
                {"start": -5.0, "duration": 1.0},
                {"start": -4.0, "duration": 1.0},
                {"start": -2.5, "duration": 0.5},
            ],
            expected_frames=3,
        )

        self.assertEqual(timing.times_seconds, (-5.0, -4.0, -2.5))
        self.assertEqual(timing.durations_seconds, (1.0, 1.0, 0.5))

    def test_rejects_frame_segment_count_mismatch(self) -> None:
        with self.assertRaisesRegex(PredictionValidationError, "frames but"):
            extract_segment_timing(
                [{"start": 0.0, "duration": 1.0}], expected_frames=2
            )

    def test_rejects_implicit_or_non_monotonic_timing(self) -> None:
        with self.assertRaisesRegex(PredictionValidationError, "explicit"):
            extract_segment_timing([{"duration": 1.0}], expected_frames=1)
        with self.assertRaisesRegex(PredictionValidationError, "strictly increasing"):
            extract_segment_timing(
                [
                    {"start": 1.0, "duration": 1.0},
                    {"start": 1.0, "duration": 1.0},
                ],
                expected_frames=2,
            )


class SerializationAndManifestTests(unittest.TestCase):
    def test_writes_time_major_little_endian_bytes_and_sha256(self) -> None:
        predictions = np.zeros((2, VERTEX_COUNT), dtype=np.float64)
        predictions[0, 0:2] = [1.25, -2.5]
        predictions[1, 0:2] = [3.5, 4.75]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result" / "frames.f32"
            artifact = write_frames(predictions, path)
            payload = path.read_bytes()

        self.assertEqual(payload[:8], struct.pack("<2f", 1.25, -2.5))
        second_frame = VERTEX_COUNT * 4
        self.assertEqual(
            payload[second_frame : second_frame + 8], struct.pack("<2f", 3.5, 4.75)
        )
        self.assertEqual(artifact.shape, (2, VERTEX_COUNT))
        self.assertEqual(artifact.byte_length, 2 * VERTEX_COUNT * 4)
        self.assertEqual(artifact.sha256, hashlib.sha256(payload).hexdigest())

    def test_manifest_locks_mapping_order_timing_and_display_scale(self) -> None:
        predictions = np.linspace(
            -2.0, 3.0, 2 * VERTEX_COUNT, dtype=np.float32
        ).reshape(2, VERTEX_COUNT)
        display_scale = calculate_display_scale(predictions)
        timing = extract_segment_timing(
            [
                {"start": 0.25, "duration": 1.0},
                {"start": 1.25, "duration": 1.0},
            ],
            expected_frames=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            frames = write_frames(predictions, Path(directory) / "frames.f32")
            manifest = build_manifest(
                result_id="a" * 32,
                model_id="facebook/tribev2",
                model_revision="f" * 40,
                model_weights_sha256=MODEL_WEIGHTS_SHA256,
                inference_mode="vision-only",
                modalities_used=("video",),
                video_extractor=VIDEO_EXTRACTOR,
                frames=frames,
                timing=timing,
                hemodynamic_offset_seconds=5.0,
                display_scale=display_scale,
            )

        self.assertFalse(manifest["isViralityScore"])
        self.assertEqual(manifest["inferenceMode"], "vision-only")
        self.assertEqual(manifest["modalitiesUsed"], ["video"])
        self.assertEqual(manifest["missingModalities"], ["audio", "text"])
        self.assertEqual(manifest["surface"]["mappingId"], SURFACE_MAPPING_ID)
        self.assertEqual(
            manifest["surface"]["hemispheres"],
            [
                {"id": "left", "offset": 0, "count": 10242},
                {"id": "right", "offset": 10242, "count": 10242},
            ],
        )
        self.assertEqual(manifest["frames"]["shape"], [2, VERTEX_COUNT])
        self.assertEqual(manifest["frames"]["timesSeconds"], [0.25, 1.25])
        self.assertEqual(manifest["frames"]["hemodynamicOffsetSeconds"], 5.0)
        self.assertEqual(manifest["frames"]["displayScale"], display_scale)
        self.assertEqual(display_scale["scope"], "all-frames-all-vertices")
        self.assertEqual(display_scale["centerMethod"], "zero")
        self.assertLess(display_scale["domain"][0], 0.0)
        self.assertEqual(display_scale["domain"][1], 0.0)
        self.assertGreater(display_scale["domain"][2], 0.0)

    def test_global_display_scale_uses_median_when_zero_is_outside(self) -> None:
        predictions = np.linspace(
            2.0, 8.0, 2 * VERTEX_COUNT, dtype=np.float32
        ).reshape(2, VERTEX_COUNT)
        display_scale = calculate_display_scale(predictions)

        self.assertEqual(display_scale["centerMethod"], "global-median")
        self.assertLess(
            display_scale["domain"][0],
            display_scale["domain"][1],
        )
        self.assertLess(
            display_scale["domain"][1],
            display_scale["domain"][2],
        )

    def test_global_display_scale_rejects_degenerate_predictions(self) -> None:
        with self.assertRaisesRegex(PredictionValidationError, "non-degenerate"):
            calculate_display_scale(
                np.zeros((1, VERTEX_COUNT), dtype=np.float32)
            )

    def test_manifest_rejects_unpublished_hemodynamic_offset(self) -> None:
        predictions = np.linspace(
            -1.0, 1.0, VERTEX_COUNT, dtype=np.float32
        ).reshape(1, VERTEX_COUNT)
        display_scale = calculate_display_scale(predictions)
        timing = extract_segment_timing(
            [{"start": 0.0, "duration": 1.0}], expected_frames=1
        )
        with tempfile.TemporaryDirectory() as directory:
            frames = write_frames(predictions, Path(directory) / "frames.f32")
            with self.assertRaisesRegex(PredictionValidationError, "5-second"):
                build_manifest(
                    result_id="b" * 32,
                    model_id="facebook/tribev2",
                    model_revision="e" * 40,
                    model_weights_sha256=MODEL_WEIGHTS_SHA256,
                    inference_mode="full",
                    modalities_used=("video", "audio", "text"),
                    video_extractor=VIDEO_EXTRACTOR,
                    frames=frames,
                    timing=timing,
                    hemodynamic_offset_seconds=4.0,
                    display_scale=display_scale,
                )

    def test_manifest_accepts_full_mode_and_rejects_inconsistent_modalities(self) -> None:
        predictions = np.linspace(-1.0, 1.0, VERTEX_COUNT, dtype=np.float32).reshape(1, VERTEX_COUNT)
        timing = extract_segment_timing([{"start": 0.0, "duration": 1.0}], expected_frames=1)
        with tempfile.TemporaryDirectory() as directory:
            frames = write_frames(predictions, Path(directory) / "frames.f32")
            full = build_manifest(
                result_id="c" * 32,
                model_id="facebook/tribev2",
                model_revision="d" * 40,
                model_weights_sha256=MODEL_WEIGHTS_SHA256,
                inference_mode="full",
                modalities_used=("video", "audio", "text"),
                video_extractor=VIDEO_EXTRACTOR,
                frames=frames,
                timing=timing,
                hemodynamic_offset_seconds=5.0,
                display_scale=calculate_display_scale(predictions),
            )
            self.assertEqual(full["missingModalities"], [])
            with self.assertRaisesRegex(PredictionValidationError, "modalities"):
                build_manifest(
                    result_id="d" * 32,
                    model_id="facebook/tribev2",
                    model_revision="d" * 40,
                    model_weights_sha256=MODEL_WEIGHTS_SHA256,
                    inference_mode="vision-only",
                    modalities_used=("video", "audio", "text"),
                    video_extractor=VIDEO_EXTRACTOR,
                    frames=frames,
                    timing=timing,
                    hemodynamic_offset_seconds=5.0,
                    display_scale=calculate_display_scale(predictions),
                )

    def test_manifest_requires_exact_video_pin_device_and_precision(self) -> None:
        predictions = np.linspace(
            -1.0, 1.0, VERTEX_COUNT, dtype=np.float32
        ).reshape(1, VERTEX_COUNT)
        timing = extract_segment_timing(
            [{"start": 0.0, "duration": 1.0}], expected_frames=1
        )
        invalid_extractors = (
            {key: value for key, value in VIDEO_EXTRACTOR.items() if key != "precision"},
            {**VIDEO_EXTRACTOR, "id": "untrusted/vjepa"},
            {
                **VIDEO_EXTRACTOR,
                "device": "cuda",
                "precision": "autocast-fp16",
            },
            {**VIDEO_EXTRACTOR, "precision": "raw-fp16"},
        )
        with tempfile.TemporaryDirectory() as directory:
            frames = write_frames(predictions, Path(directory) / "frames.f32")
            for extractor in invalid_extractors:
                with self.subTest(extractor=extractor):
                    with self.assertRaises(PredictionValidationError):
                        build_manifest(
                            result_id="e" * 32,
                            model_id="facebook/tribev2",
                            model_revision="d" * 40,
                            model_weights_sha256=MODEL_WEIGHTS_SHA256,
                            inference_mode="vision-only",
                            modalities_used=("video",),
                            video_extractor=extractor,
                            frames=frames,
                            timing=timing,
                            hemodynamic_offset_seconds=5.0,
                            display_scale=calculate_display_scale(predictions),
                        )


if __name__ == "__main__":
    unittest.main()
