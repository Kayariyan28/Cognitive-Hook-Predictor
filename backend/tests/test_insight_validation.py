from __future__ import annotations

import json
from pathlib import Path
import unittest

from backend.insight.bundle import assemble_evidence_bundle, hook_evidence_card
from backend.insight.claim_terms import (
    CLAIM_TERMS_SCHEMA_VERSION,
    GLOBAL_OUTCOME_TERMS,
    as_dict,
    sentence_violations,
    split_sentences,
)
from backend.insight.validation import (
    REASON_CODES,
    round_to_significant,
    validate_insight,
)
from backend.tests.insight_support import (
    asr_provider,
    forecast_result,
    ocr_provider,
    tribe_descriptors,
)


FIXTURES = Path(__file__).resolve().parent / "insight_fixtures"

# Fixtures that cite the optional transcript and on-screen-text lanes need a
# bundle where those branches actually published.
EXTENDED_LANE_FIXTURES = frozenset({"asr_outcome_claim.json", "ocr_outcome_claim.json"})
HOOK_CARD_FIXTURES = frozenset({"valid_hook_only_artifact.json"})

SCHEMA_EXPECTATIONS = {
    "valid_full_artifact.json": None,
    "valid_hook_only_artifact.json": None,
    "unknown_top_level_key.json": "unknown_field",
    "unknown_nested_key.json": "unknown_field",
    "server_owned_behavioral_outcome.json": "server_owned_field",
    "server_owned_limits.json": "server_owned_field",
    "server_owned_provenance.json": "server_owned_field",
    "empty_citations.json": "missing_citation",
    "malformed_citation_no_lane.json": "citation_malformed",
    "malformed_citation_bad_window.json": "citation_malformed",
    "unknown_lane_citation.json": "citation_malformed",
    "absent_lane_citation.json": "citation_unresolvable",
    "dangling_pointer_citation.json": "citation_unresolvable",
    "window_outside_interval.json": "citation_unresolvable",
    "invented_number.json": "numeric_not_in_evidence",
    "over_rounded_number.json": "numeric_not_in_evidence",
    "rounded_number_valid_twin.json": None,
    "window_time_exempt_valid_twin.json": None,
    "hypothesis_missing_label.json": "schema_invalid",
    "effort_enum_invalid.json": "schema_invalid",
    "phase_enum_invalid.json": "schema_invalid",
    "text_too_long.json": "schema_invalid",
    "too_many_experiments.json": "schema_invalid",
    "unknown_hypothesis_reference.json": "schema_invalid",
    "metric_path_not_numeric.json": "schema_invalid",
}

CLAIM_EXPECTATIONS = {
    "global_virality_claim.json": "go viral",
    "global_retention_claim.json": "retention",
    "global_views_claim.json": "views",
    "global_algorithm_claim.json": "algorithm",
    "global_mental_state_claim.json": "attention",
    "experiment_edit_outcome_claim.json": "watch time",
    "tribe_mental_state.json": "emotion",
    "tribe_brain_lights_up.json": "lights up",
    "tribe_reverse_inference.json": "subconscious",
    "asr_outcome_claim.json": "viral",
    "ocr_outcome_claim.json": "hooked",
}

CLAIM_VALID_TWINS = (
    "negated_virality_valid_twin.json",
    "negated_retention_valid_twin.json",
    "tribe_caption_valid_twin.json",
    "untested_heuristic_valid_twin.json",
    "hook_noun_valid_twin.json",
)


def golden_bundle():
    return assemble_evidence_bundle(
        forecast_result(),
        tribe_descriptors=tribe_descriptors(),
        declared_context={"creatorNote": "second attempt at the opening"},
    )


def extended_bundle():
    return assemble_evidence_bundle(
        forecast_result(optionalProviders={"asr": asr_provider(), "ocr": ocr_provider()}),
        tribe_descriptors=tribe_descriptors(),
    )


def bundle_for(name: str):
    if name in EXTENDED_LANE_FIXTURES:
        return extended_bundle()
    if name in HOOK_CARD_FIXTURES:
        return hook_evidence_card(golden_bundle())
    return golden_bundle()


def load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class FixtureCorpusTests(unittest.TestCase):
    def test_every_named_fixture_exists(self):
        # This module owns the schema and claim corpora; the new-lane fixtures
        # are owned by test_insight_claim_ci, which asserts total coverage.
        expected = set(SCHEMA_EXPECTATIONS) | set(CLAIM_EXPECTATIONS) | set(CLAIM_VALID_TWINS)
        present = {path.name for path in FIXTURES.glob("*.json")}
        self.assertEqual(expected - present, set())

    def test_schema_fixtures_reach_their_exact_reason_code(self):
        for name, expected in SCHEMA_EXPECTATIONS.items():
            with self.subTest(fixture=name):
                outcome = validate_insight(load(name), bundle_for(name))
                if expected is None:
                    self.assertEqual(outcome["status"], "valid", outcome)
                else:
                    self.assertEqual(outcome["status"], "rejected", outcome)
                    self.assertEqual(outcome["reasonCode"], expected, outcome)

    def test_claim_fixtures_report_their_exact_term(self):
        for name, term in CLAIM_EXPECTATIONS.items():
            with self.subTest(fixture=name):
                outcome = validate_insight(load(name), bundle_for(name))
                self.assertEqual(outcome["status"], "rejected", outcome)
                self.assertEqual(outcome["reasonCode"], "claim_boundary_violation", outcome)
                detail = outcome["detail"]
                self.assertEqual(detail["term"], term, detail)
                self.assertTrue(detail["sentence"])
                self.assertTrue(detail["itemPath"].startswith("/"))

    def test_claim_valid_twins_pass_unchanged(self):
        for name in CLAIM_VALID_TWINS:
            with self.subTest(fixture=name):
                outcome = validate_insight(load(name), bundle_for(name))
                self.assertEqual(outcome["status"], "valid", outcome)

    def test_every_rejection_uses_the_documented_enum(self):
        for name in SCHEMA_EXPECTATIONS | CLAIM_EXPECTATIONS:
            outcome = validate_insight(load(name), bundle_for(name))
            if outcome["status"] == "rejected":
                self.assertIn(outcome["reasonCode"], REASON_CODES, name)


class AllOrNothingTests(unittest.TestCase):
    def test_valid_artifact_returns_only_model_settable_fields(self):
        outcome = validate_insight(load("valid_full_artifact.json"), golden_bundle())
        self.assertEqual(outcome["status"], "valid")
        self.assertEqual(
            sorted(outcome["artifact"]), ["hookReport", "phaseCommentary", "tribeNotes"]
        )

    def test_malformed_json_is_never_repaired(self):
        for raw in ('{"hookReport": ', "not json at all", "[]", "null"):
            with self.subTest(raw=raw):
                outcome = validate_insight(raw, golden_bundle())
                self.assertEqual(outcome["reasonCode"], "output_not_json")

    def test_one_bad_item_rejects_the_whole_artifact(self):
        payload = json.loads(load("valid_full_artifact.json"))
        payload["hookReport"]["observations"].append(
            {"text": "This clip will go viral.", "citations": ["ast:/windows/0"]}
        )
        outcome = validate_insight(json.dumps(payload), golden_bundle())
        self.assertEqual(outcome["status"], "rejected")
        self.assertNotIn("artifact", outcome)


class NumericCopyTests(unittest.TestCase):
    def test_rounding_is_allowed_down_to_two_significant_figures(self):
        self.assertEqual(round_to_significant(0.1372, 2), 0.14)
        self.assertEqual(round_to_significant(0.1372, 3), 0.137)
        self.assertEqual(round_to_significant(2418.5, 2), 2400.0)

    def test_percent_forms_resolve_against_the_fraction(self):
        payload = json.loads(load("valid_full_artifact.json"))
        payload["hookReport"]["observations"][0] = {
            "text": "Measured silent windows cover 19% of the clip.",
            "citations": ["measured:/audio/descriptors/silent_window_fraction"],
        }
        outcome = validate_insight(json.dumps(payload), golden_bundle())
        self.assertEqual(outcome["status"], "valid", outcome)

    def test_window_bounds_are_exempt_but_other_times_are_not(self):
        payload = json.loads(load("valid_full_artifact.json"))
        payload["hookReport"]["observations"][0] = {
            "text": "Across 0 to 3 seconds the counter stays in frame.",
            "citations": ["nanollava:/keyframes/0/parsed"],
        }
        self.assertEqual(
            validate_insight(json.dumps(payload), golden_bundle())["status"], "valid"
        )
        payload["hookReport"]["observations"][0]["text"] = (
            "Across 0 to 4 seconds the counter stays in frame."
        )
        outcome = validate_insight(json.dumps(payload), golden_bundle())
        self.assertEqual(outcome["reasonCode"], "numeric_not_in_evidence")


class ClaimTermDataTests(unittest.TestCase):
    def test_terms_are_exported_once_for_every_consumer(self):
        exported = as_dict()
        self.assertEqual(exported["schemaVersion"], CLAIM_TERMS_SCHEMA_VERSION)
        self.assertEqual(exported["globalOutcomeTerms"], list(GLOBAL_OUTCOME_TERMS))
        self.assertIn("subconscious", exported["tribeScopedTerms"])

    def test_lint_is_sentence_scoped(self):
        text = "This is not a virality claim. This clip will go viral."
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentence_violations(sentences[0], tribe_scoped=False), [])
        self.assertEqual(sentence_violations(sentences[1], tribe_scoped=False), ["go viral"])

    def test_tribe_terms_apply_only_to_tribe_cited_items(self):
        sentence = "The parcel lights up in the opening."
        self.assertEqual(sentence_violations(sentence, tribe_scoped=False), [])
        self.assertEqual(sentence_violations(sentence, tribe_scoped=True), ["lights up"])

    def test_product_nouns_stay_legal(self):
        self.assertEqual(
            sentence_violations("The hook window opens on a counter.", tribe_scoped=True), []
        )
        self.assertEqual(
            sentence_violations("Viewers get hooked here.", tribe_scoped=False), ["hooked"]
        )


if __name__ == "__main__":
    unittest.main()
