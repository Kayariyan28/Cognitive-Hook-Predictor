"""The hook window is sampled at fixed absolute times, on every clip length.

The six keyframes both visual branches read are sampled proportionally, at
``duration x (i + 0.5) / 6``. The first therefore always lands at one twelfth of
the clip, so the hook window's share of them shrinks as a clip gets longer and
past roughly 36 seconds none of them fall inside the first three seconds at all.
These tests pin the second pass that closes that gap.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from backend.forecast.workers.nanollava import (
    HOOK_FRAME_TIMES,
    HOOK_PREPROCESSING_ID,
    KEYFRAME_COUNT,
    PREPROCESSING_ID,
    PREPROCESSING_SHA256,
    TWO_PASS_PREPROCESSING_ID,
    TWO_PASS_PREPROCESSING_SHA256,
    NanoLlavaSemanticAdapter,
    deterministic_sample_times,
    hook_frame_times,
)


VALID_DESCRIPTION = json.dumps(
    {
        "scene": "A person stands beside a kitchen counter.",
        "action": "The person lifts a jar.",
        "visibleText": ["READ THIS"],
        "shot": "Medium static shot.",
        "uncertainties": [],
    },
    separators=(",", ":"),
)
VIDEO_BYTES = b"video"
VIDEO_SHA256 = hashlib.sha256(VIDEO_BYTES).hexdigest()


class TwoPassFrames:
    """A frame extractor that honours explicit sample times."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, ...]] = []

    def extract(self, video_path, duration_seconds, destination, sample_times=None):
        del video_path
        destination.mkdir(parents=True, exist_ok=True)
        planned = (
            sample_times
            if sample_times is not None
            else deterministic_sample_times(duration_seconds)
        )
        self.calls.append(tuple(planned))
        values = []
        for index, timestamp in enumerate(planned):
            path = destination / f"fake-{index}.png"
            path.write_bytes(b"png")
            values.append((timestamp, path))
        return tuple(values)


class LegacyFrames:
    """An extractor written before the hook pass existed."""

    def extract(self, video_path, duration_seconds, destination):
        del video_path
        destination.mkdir(parents=True, exist_ok=True)
        values = []
        for index, timestamp in enumerate(deterministic_sample_times(duration_seconds)):
            path = destination / f"fake-{index}.png"
            path.write_bytes(b"png")
            values.append((timestamp, path))
        return tuple(values)


class FakeSemantics:
    def __init__(self) -> None:
        self.described = 0

    def describe(self, image_path, prompt):
        del image_path, prompt
        self.described += 1
        return VALID_DESCRIPTION


class SampleTimeTests(unittest.TestCase):
    def test_the_proportional_pass_misses_the_hook_on_a_long_clip(self):
        for duration, expected in ((12.0, 1), (21.5, 1), (36.0, 0), (45.0, 0), (60.0, 0)):
            with self.subTest(duration=duration):
                inside = [
                    moment
                    for moment in deterministic_sample_times(duration)
                    if moment < 3.0
                ]
                self.assertEqual(len(inside), expected)

    def test_the_first_proportional_frame_is_always_a_twelfth_of_the_clip(self):
        for duration in (12.0, 21.5, 45.0, 60.0):
            self.assertAlmostEqual(
                deterministic_sample_times(duration)[0], duration / 12.0, places=5
            )

    def test_hook_times_are_absolute_and_do_not_move_with_duration(self):
        for duration in (12.0, 21.5, 45.0, 60.0):
            self.assertEqual(hook_frame_times(duration), HOOK_FRAME_TIMES)
            self.assertEqual(hook_frame_times(duration)[0], 0.0)

    def test_hook_times_are_clipped_inside_a_very_short_clip(self):
        times = hook_frame_times(2.5)
        self.assertEqual(times[0], 0.0)
        self.assertTrue(all(moment < 2.5 for moment in times))
        self.assertEqual(len(times), len(set(times)))

    def test_the_hook_pass_matches_the_six_frame_runtime_contract(self):
        self.assertEqual(len(HOOK_FRAME_TIMES), KEYFRAME_COUNT)

    def test_an_invalid_duration_is_refused(self):
        for duration in (0, -1, float("nan")):
            with self.subTest(duration=duration):
                with self.assertRaises(ValueError):
                    hook_frame_times(duration)


class NanoLlavaHookPassTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="hookframes-")
        self.addCleanup(self._temporary.cleanup)
        root = Path(self._temporary.name)
        self.snapshot = root / "snapshot"
        self.snapshot.mkdir()
        (self.snapshot / "model.safetensors").write_bytes(b"weights")
        (self.snapshot / "config.json").write_text("{}")
        (self.snapshot / "tokenizer.json").write_text("{}")
        self.video = root / "clip.mp4"
        self.video.write_bytes(VIDEO_BYTES)

    def adapter(self, *, hook_pass=True, extractor=None):
        adapter = NanoLlavaSemanticAdapter(
            self.snapshot,
            backend=FakeSemantics(),
            frame_extractor=extractor or TwoPassFrames(),
            hook_pass=hook_pass,
        )
        # The snapshot digest is pinned; this harness has no real weights.
        adapter._verify_snapshot = lambda: None
        return adapter

    def run_adapter(self, adapter, duration=45.0):
        return adapter.run(
            video_path=self.video,
            input_sha256=VIDEO_SHA256,
            duration_seconds=duration,
            context={},
        )

    def test_the_opening_is_described_from_a_frame_taken_in_the_opening(self):
        """The clip pass covers 0 s with a frame decoded at 3.75 s; the hook pass
        covers it with a frame decoded at 0 s. Both land in the hook card, so the
        fix is not about presence — it is about where the pixels came from."""

        value = self.run_adapter(self.adapter()).public_value()
        hook_items = [
            item for item in value["observations"] if "pass:hook" in item["labels"]
        ]
        self.assertEqual(len(hook_items), KEYFRAME_COUNT)
        self.assertEqual(hook_items[0]["startTime"], 0.0)
        self.assertLessEqual(hook_items[0]["endTime"], 0.5)

        # Without the hook pass, the only description covering the opening is a
        # window that starts at 0 s but was sampled from a frame at duration/12.
        single = self.run_adapter(self.adapter(hook_pass=False)).public_value()
        covering = [item for item in single["observations"] if item["startTime"] < 3.0]
        self.assertEqual(len(covering), 1)
        self.assertEqual(covering[0]["startTime"], 0.0)
        self.assertGreater(covering[0]["endTime"], 3.0)
        self.assertAlmostEqual(deterministic_sample_times(45.0)[0], 3.75, places=3)

    def test_both_passes_are_labelled_and_counted(self):
        extractor = TwoPassFrames()
        value = self.run_adapter(self.adapter(extractor=extractor)).public_value()
        self.assertEqual(len(extractor.calls), 2)
        self.assertEqual(extractor.calls[1], HOOK_FRAME_TIMES)
        self.assertEqual(len(value["observations"]), KEYFRAME_COUNT * 2)
        passes = {
            label
            for item in value["observations"]
            for label in item["labels"]
            if label.startswith("pass:")
        }
        self.assertEqual(passes, {"pass:clip", "pass:hook"})

    def test_provenance_declares_both_contracts_when_both_ran(self):
        value = self.run_adapter(self.adapter()).public_value()
        provenance = value["provenance"]
        self.assertEqual(provenance["preprocessingId"], TWO_PASS_PREPROCESSING_ID)
        self.assertEqual(provenance["preprocessingSha256"], TWO_PASS_PREPROCESSING_SHA256)
        self.assertIn(PREPROCESSING_ID, provenance["preprocessingId"])
        self.assertIn(HOOK_PREPROCESSING_ID, provenance["preprocessingId"])

    def test_provenance_keeps_the_audited_contract_when_only_one_pass_ran(self):
        value = self.run_adapter(self.adapter(hook_pass=False)).public_value()
        provenance = value["provenance"]
        self.assertEqual(provenance["preprocessingId"], PREPROCESSING_ID)
        self.assertEqual(provenance["preprocessingSha256"], PREPROCESSING_SHA256)
        self.assertEqual(len(value["observations"]), KEYFRAME_COUNT)

    def test_an_extractor_without_the_two_pass_contract_degrades_with_a_warning(self):
        value = self.run_adapter(self.adapter(extractor=LegacyFrames())).public_value()
        self.assertEqual(len(value["observations"]), KEYFRAME_COUNT)
        self.assertEqual(value["provenance"]["preprocessingId"], PREPROCESSING_ID)
        self.assertTrue(
            any("hook window" in warning for warning in value["warnings"]),
            value["warnings"],
        )

    def test_the_switch_is_read_from_the_environment(self):
        from backend.forecast.workers.nanollava import HOOK_PASS_ENV

        enabled = NanoLlavaSemanticAdapter.from_env(
            {"FORECAST_NANOLLAVA_SNAPSHOT": str(self.snapshot)}
        )
        self.assertTrue(enabled.hook_pass)
        disabled = NanoLlavaSemanticAdapter.from_env(
            {"FORECAST_NANOLLAVA_SNAPSHOT": str(self.snapshot), HOOK_PASS_ENV: "false"}
        )
        self.assertFalse(disabled.hook_pass)


class HookFrameBundleTests(unittest.TestCase):
    def test_hook_keyframes_reach_the_hook_card(self):
        from backend.insight.bundle import assemble_evidence_bundle, hook_evidence_card
        from backend.tests.insight_support import forecast_result, semantic_provider

        provider = semantic_provider()
        observations = provider["result"]["observations"]
        # A 45-second clip: the proportional frames all sit outside the hook.
        for index, item in enumerate(observations):
            item["startTime"] = 3.75 + index * 7.5
            item["endTime"] = item["startTime"] + 7.5
            item["labels"] = [*item["labels"], "pass:clip"]
        opening = dict(observations[0])
        opening["startTime"] = 0.0
        opening["endTime"] = 0.4
        opening["labels"] = ["model-derived", "single-keyframe", "pass:hook"]
        observations.append(opening)

        bundle = assemble_evidence_bundle(
            forecast_result(optionalProviders={"semanticModel": provider})
        )
        card = hook_evidence_card(bundle)
        lane = card["lanes"]["nanollava"]
        self.assertEqual(lane["status"], "present")
        self.assertEqual(len(lane["keyframes"]), 1)
        self.assertEqual(lane["keyframes"][0]["startSec"], 0.0)


if __name__ == "__main__":
    unittest.main()
