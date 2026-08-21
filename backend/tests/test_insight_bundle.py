from __future__ import annotations

import copy
import json
import unittest

from backend.insight.bundle import (
    BUNDLE_SCHEMA_VERSION,
    HOOK_WINDOW_SECONDS,
    LANE_KEYS,
    BundleUnavailableError,
    assemble_evidence_bundle,
    canonical_json,
    hook_evidence_card,
    input_evidence_hash,
)
from backend.insight.citations import (
    CitationMalformedError,
    CitationUnresolvableError,
    enumerate_pointers,
    parse_citation,
    resolve_citation,
)
from backend.tests.insight_support import (
    FORECAST_RESULT_ID,
    TRIBE_RESULT_ID,
    forecast_result,
    tribe_descriptors,
    unavailable_provider,
)


def full_bundle():
    return assemble_evidence_bundle(
        forecast_result(),
        tribe_descriptors=tribe_descriptors(),
        declared_context={"creatorNote": "second attempt at the opening"},
    )


class BundleAssemblyTests(unittest.TestCase):
    def test_golden_bundle_shape(self):
        bundle = full_bundle()
        self.assertEqual(bundle["schemaVersion"], BUNDLE_SCHEMA_VERSION)
        self.assertEqual(bundle["source"]["forecastResultId"], FORECAST_RESULT_ID)
        self.assertEqual(bundle["source"]["tribeResultId"], TRIBE_RESULT_ID)
        self.assertIsNone(bundle["source"]["window"])
        self.assertEqual(sorted(bundle["lanes"]), sorted(LANE_KEYS))
        for lane in LANE_KEYS:
            if lane in {"asr", "ocr"}:
                continue
            self.assertEqual(bundle["lanes"][lane]["status"], "present", lane)

    def test_measured_lane_copies_values_verbatim_and_strips_prefix(self):
        lane = full_bundle()["lanes"]["measured"]
        self.assertEqual(lane["video"]["durationSeconds"], 21.5)
        self.assertNotIn("sha256", lane["video"])
        self.assertEqual(lane["audio"]["descriptors"]["rms"], 0.1372)
        self.assertEqual(lane["audio"]["descriptors"]["spectral_centroid_hz_mean"], 2418.5)
        self.assertNotIn("measured_audio.rms", lane["audio"]["descriptors"])
        self.assertEqual(
            [peak["startSec"] for peak in lane["audio"]["energyPeaks"]], [1.44, 9.12]
        )

    def test_nanollava_lane_parses_validated_json_text(self):
        lane = full_bundle()["lanes"]["nanollava"]
        self.assertEqual(lane["keyframes"][0]["parsed"]["scene"], "a kitchen counter")
        self.assertEqual(lane["keyframes"][0]["parsed"]["visibleText"], "READ THIS")
        self.assertIsNone(lane["keyframes"][1]["parsed"]["visibleText"])
        self.assertEqual(json.loads(lane["keyframes"][0]["text"])["shot"], "medium")

    def test_ast_lane_carries_labels_with_model_scores(self):
        lane = full_bundle()["lanes"]["ast"]
        self.assertEqual(lane["windows"][0]["labels"][0], {"label": "Speech", "modelScore": 0.8125})
        self.assertEqual(lane["descriptors"]["mean_top_label_score"], 0.6425)

    def test_vjepa_lane_drops_per_dimension_embedding_features(self):
        lane = full_bundle()["lanes"]["vjepa"]
        self.assertIn("temporal_change_peak", lane["descriptors"])
        self.assertFalse(
            [key for key in lane["descriptors"] if key.startswith("embedding_0")]
        )
        self.assertEqual(lane["descriptors"]["embedding_norm_mean"], 12.75)

    def test_tribe_lane_is_descriptors_only_and_bounded(self):
        lane = full_bundle()["lanes"]["tribe"]
        self.assertEqual(lane["rankedBy"], "rms")
        self.assertLessEqual(len(lane["parcels"]), 8)
        self.assertEqual(lane["parcels"][0]["rms"], 0.9)
        self.assertGreaterEqual(lane["parcels"][0]["rms"], lane["parcels"][-1]["rms"])
        self.assertEqual(lane["intervals"][0]["startSec"], 0.0)
        self.assertIsNone(lane["intervals"][0]["continuity"])
        self.assertEqual([phase["id"] for phase in lane["phases"]], ["early", "middle", "late"])
        serialized = canonical_json(lane)
        self.assertNotIn("frames.f32", serialized)
        self.assertNotIn("vertexSlots", serialized)

    def test_context_lane_merges_stored_and_declared_context(self):
        lane = full_bundle()["lanes"]["context"]
        self.assertEqual(lane["declared"]["platform"], "reels")
        self.assertEqual(lane["declared"]["creatorNote"], "second attempt at the opening")

    def test_tribe_descriptors_are_optional(self):
        bundle = assemble_evidence_bundle(forecast_result())
        self.assertEqual(bundle["lanes"]["tribe"]["status"], "absent")
        self.assertIsNone(bundle["source"]["tribeResultId"])

    def test_wrong_schema_versions_fail_closed(self):
        with self.assertRaises(BundleUnavailableError):
            assemble_evidence_bundle(forecast_result(schemaVersion="creator-forecast-result/2"))
        with self.assertRaises(BundleUnavailableError):
            assemble_evidence_bundle(
                forecast_result(),
                tribe_descriptors=tribe_descriptors(schemaVersion="tribe-cortical-descriptors/2"),
            )
        with self.assertRaises(BundleUnavailableError):
            assemble_evidence_bundle({"schemaVersion": "creator-forecast-result/1"})


class AbsentLaneTests(unittest.TestCase):
    def test_absent_lane_is_a_marker_not_a_default(self):
        result = forecast_result(
            optionalProviders={
                "measuredAudio": unavailable_provider("The server audio decoder is not configured."),
                "audioModel": unavailable_provider("No task-specific audio model is registered."),
                "semanticModel": unavailable_provider("The pinned NanoLLaVA snapshot is not configured."),
                "vjepa21": unavailable_provider("No hash-verified V-JEPA 2.1 artifact is registered."),
            }
        )
        bundle = assemble_evidence_bundle(result)
        for lane in ("nanollava", "ast", "vjepa", "asr", "ocr", "tribe"):
            self.assertEqual(bundle["lanes"][lane]["status"], "absent", lane)
            self.assertTrue(bundle["lanes"][lane]["reason"])
            self.assertEqual(set(bundle["lanes"][lane]), {"status", "reason"})
        audio = bundle["lanes"]["measured"]["audio"]
        self.assertEqual(audio["status"], "absent")
        self.assertNotIn("descriptors", audio)
        self.assertIn("audio decoder", audio["reason"])

    def test_absent_lane_reason_repeats_the_branch_reason(self):
        result = forecast_result(
            optionalProviders={"semanticModel": unavailable_provider("Runtime not installed.")}
        )
        bundle = assemble_evidence_bundle(result)
        self.assertEqual(bundle["lanes"]["nanollava"]["reason"], "Runtime not installed.")

    def test_absent_context_when_nothing_declared(self):
        bundle = assemble_evidence_bundle(forecast_result(context=None))
        self.assertEqual(bundle["lanes"]["context"]["status"], "absent")

    def test_missing_video_metadata_makes_the_measured_lane_absent(self):
        result = forecast_result()
        result["evidence"]["videoMetadata"] = {"status": "unavailable"}
        bundle = assemble_evidence_bundle(result)
        self.assertEqual(bundle["lanes"]["measured"]["status"], "absent")


class OptionalLaneTests(unittest.TestCase):
    def test_asr_and_ocr_lanes_populate_when_the_branches_publish(self):
        from backend.tests.insight_support import asr_provider, ocr_provider

        bundle = assemble_evidence_bundle(
            forecast_result(optionalProviders={"asr": asr_provider(), "ocr": ocr_provider()})
        )
        asr = bundle["lanes"]["asr"]
        self.assertEqual(asr["language"], "en")
        self.assertEqual(asr["segments"][0]["text"], "Stop scrolling, this jar changed my kitchen.")
        ocr = bundle["lanes"]["ocr"]
        self.assertEqual(ocr["engine"], "ocrmac")
        self.assertEqual(ocr["frames"][0]["blocks"][0]["text"], "READ THIS")
        self.assertEqual(ocr["frames"][0]["blocks"][0]["bbox"], [0.1, 0.1, 0.5, 0.2])


class PointerResolvabilityTests(unittest.TestCase):
    def test_every_emitted_pointer_resolves_through_the_citation_grammar(self):
        bundle = full_bundle()
        checked = 0
        for lane in LANE_KEYS:
            root = bundle["lanes"][lane]
            if root["status"] != "present":
                continue
            for pointer in enumerate_pointers(root):
                citation = parse_citation(f"{lane}:{pointer}")
                resolve_citation(bundle, citation)
                checked += 1
        self.assertGreater(checked, 100)

    def test_window_assertions_resolve_against_timed_items(self):
        bundle = full_bundle()
        citation = parse_citation("vjepa:/windows/0@window(0.0,2.0)")
        self.assertEqual(resolve_citation(bundle, citation)["startSec"], 0.0)
        with self.assertRaises(CitationUnresolvableError):
            resolve_citation(bundle, parse_citation("vjepa:/windows/0@window(10.0,12.0)"))
        with self.assertRaises(CitationUnresolvableError):
            resolve_citation(bundle, parse_citation("vjepa:/descriptors@window(0.0,2.0)"))

    def test_absent_lane_citations_are_unresolvable(self):
        bundle = assemble_evidence_bundle(forecast_result())
        with self.assertRaises(CitationUnresolvableError):
            resolve_citation(bundle, parse_citation("tribe:/intervals/0"))

    def test_malformed_citations_are_rejected_before_resolution(self):
        for text in (
            "measured/audio/descriptors/rms",
            "unknownlane:/x",
            "measured:audio",
            "measured:/audio@window(3.0,1.0)",
            "measured:/audio@window(a,b)",
            "",
        ):
            with self.subTest(text=text):
                with self.assertRaises(CitationMalformedError):
                    parse_citation(text)

    def test_dangling_pointers_are_unresolvable(self):
        bundle = full_bundle()
        for pointer in ("/audio/descriptors/nope", "/audio/energyPeaks/99", "/video/durationSeconds/x"):
            with self.subTest(pointer=pointer):
                with self.assertRaises(CitationUnresolvableError):
                    resolve_citation(bundle, parse_citation(f"measured:{pointer}"))


class HookEvidenceCardTests(unittest.TestCase):
    def test_hook_card_slices_to_the_window(self):
        card = hook_evidence_card(full_bundle())
        self.assertEqual(card["source"]["window"], [0.0, 3.0])
        self.assertEqual(
            [item["startSec"] for item in card["lanes"]["vjepa"]["windows"]], [0.0, 2.0]
        )
        self.assertEqual(
            [item["startSec"] for item in card["lanes"]["nanollava"]["keyframes"]], [0.0, 2.15]
        )
        self.assertEqual([item["startSec"] for item in card["lanes"]["ast"]["windows"]], [0.0])
        self.assertEqual(
            [item["startSec"] for item in card["lanes"]["tribe"]["intervals"]],
            [0.0, 1.49, 2.98],
        )

    def test_boundary_straddling_and_touching_items(self):
        # 2.0 -> 4.0 straddles the 3.0 boundary and is kept; 19.5 -> 21.5 is not.
        card = hook_evidence_card(full_bundle())
        kept = {(item["startSec"], item["endSec"]) for item in card["lanes"]["vjepa"]["windows"]}
        self.assertIn((2.0, 4.0), kept)
        self.assertNotIn((19.5, 21.5), kept)

        touching = hook_evidence_card(full_bundle(), window=(0.0, 2.0))
        starts = [item["startSec"] for item in touching["lanes"]["vjepa"]["windows"]]
        self.assertEqual(starts, [0.0])

    def test_audio_onset_is_measured_or_explicitly_absent(self):
        card = hook_evidence_card(full_bundle())
        onset = card["lanes"]["measured"]["audio"]["onset"]
        self.assertEqual(onset["firstEnergyPeakSec"], 1.44)
        self.assertEqual(onset["prePeakSilenceSec"], 1.44)

        quiet = hook_evidence_card(full_bundle(), window=(3.0, 6.0))
        quiet_onset = quiet["lanes"]["measured"]["audio"]["onset"]
        self.assertIsNone(quiet_onset["firstEnergyPeakSec"])
        self.assertIsNone(quiet_onset["prePeakSilenceSec"])

    def test_empty_lane_after_slicing_becomes_an_absent_marker(self):
        card = hook_evidence_card(full_bundle(), window=(11.0, 12.0))
        self.assertEqual(card["lanes"]["nanollava"]["status"], "absent")
        self.assertIn("requested window", card["lanes"]["nanollava"]["reason"])

    def test_untimed_lanes_survive_slicing(self):
        card = hook_evidence_card(full_bundle())
        self.assertEqual(card["lanes"]["context"]["declared"]["platform"], "reels")
        self.assertEqual(card["lanes"]["measured"]["video"]["durationSeconds"], 21.5)
        self.assertEqual(len(card["lanes"]["tribe"]["parcels"]), 8)

    def test_card_pointers_still_resolve(self):
        card = hook_evidence_card(full_bundle())
        for lane in LANE_KEYS:
            root = card["lanes"][lane]
            if root["status"] != "present":
                continue
            for pointer in enumerate_pointers(root):
                resolve_citation(card, parse_citation(f"{lane}:{pointer}"))

    def test_invalid_windows_and_bundles_fail_closed(self):
        with self.assertRaises(BundleUnavailableError):
            hook_evidence_card(full_bundle(), window=(3.0, 3.0))
        with self.assertRaises(BundleUnavailableError):
            hook_evidence_card({"schemaVersion": "nope"})


class EvidenceHashTests(unittest.TestCase):
    def test_hash_is_stable_across_dict_ordering(self):
        bundle = full_bundle()
        reordered = json.loads(json.dumps(bundle))
        reordered["lanes"] = dict(reversed(list(reordered["lanes"].items())))
        reordered["source"] = dict(reversed(list(reordered["source"].items())))
        self.assertEqual(input_evidence_hash(bundle), input_evidence_hash(reordered))
        self.assertEqual(bundle["inputEvidenceHash"], input_evidence_hash(reordered))

    def test_hash_changes_when_a_value_changes(self):
        bundle = full_bundle()
        mutated = copy.deepcopy(bundle)
        mutated["lanes"]["measured"]["audio"]["descriptors"]["rms"] = 0.1373
        self.assertNotEqual(input_evidence_hash(bundle), input_evidence_hash(mutated))

    def test_hook_card_hash_differs_from_the_full_bundle(self):
        bundle = full_bundle()
        card = hook_evidence_card(bundle, window=HOOK_WINDOW_SECONDS)
        self.assertNotEqual(bundle["inputEvidenceHash"], card["inputEvidenceHash"])

    def test_assembly_is_deterministic(self):
        self.assertEqual(
            canonical_json(full_bundle()),
            canonical_json(full_bundle()),
        )


if __name__ == "__main__":
    unittest.main()
