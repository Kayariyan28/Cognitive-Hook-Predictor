"""Phase 9: the enforcement layers, kept honest by CI.

The prompt asks; the validator decides; these tests make sure the deciding
layer never quietly loses a rule, that the frontend reads the same vocabulary
the backend enforces, and that the optional judge stays out of the request
path.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import unittest

from backend.insight.bundle import assemble_evidence_bundle
from backend.insight.claim_terms import export_json
from backend.insight.judge import (
    JUDGE_SYSTEM_PROMPT,
    JudgeUnavailable,
    build_judge_request,
    collect_items,
    judge_artifacts,
    parse_judge_response,
)
from backend.insight.validation import validate_insight
from backend.tests.insight_support import (
    asr_provider,
    forecast_result,
    ocr_provider,
    tribe_descriptors,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "insight_fixtures"
EXPORTED_TERMS = REPOSITORY_ROOT / "src" / "insight" / "claim-terms.json"

# Owned by test_insight_phase10.py, which asserts each one there.
PHASE_TEN_FIXTURES = frozenset(
    {
        "outcomes_lane_citation.json",
        "comparative_rank_valid_twin.json",
        "comparative_verdict_claim.json",
        "rewrite_valid_twin.json",
        "rewrite_outcome_claim.json",
        "rewrite_too_long.json",
        "rewrite_unknown_basis.json",
    }
)

NEW_LANE_EXPECTATIONS = {
    "ocr_reader_outcome_claim.json": ("claim_boundary_violation", "hooked"),
    "asr_retention_claim.json": ("claim_boundary_violation", "retention"),
    "asr_mental_state_claim.json": ("claim_boundary_violation", "attention"),
    "ocr_text_valid_twin.json": (None, None),
    "asr_transcript_valid_twin.json": (None, None),
}


def extended_bundle():
    return assemble_evidence_bundle(
        forecast_result(optionalProviders={"asr": asr_provider(), "ocr": ocr_provider()}),
        tribe_descriptors=tribe_descriptors(),
    )


class FixtureCoverageTests(unittest.TestCase):
    def test_every_fixture_on_disk_is_asserted_by_a_test(self):
        from backend.tests.test_insight_validation import (
            CLAIM_EXPECTATIONS,
            CLAIM_VALID_TWINS,
            SCHEMA_EXPECTATIONS,
        )

        asserted = (
            set(SCHEMA_EXPECTATIONS)
            | set(CLAIM_EXPECTATIONS)
            | set(CLAIM_VALID_TWINS)
            | set(NEW_LANE_EXPECTATIONS)
            | PHASE_TEN_FIXTURES
        )
        on_disk = {path.name for path in FIXTURES.glob("*.json")}
        self.assertEqual(
            on_disk - asserted, set(), "a fixture exists that no test asserts"
        )
        self.assertEqual(
            asserted - on_disk, set(), "a test names a fixture that does not exist"
        )

    def test_the_new_lanes_are_red_teamed_too(self):
        bundle = extended_bundle()
        for name, (reason, term) in NEW_LANE_EXPECTATIONS.items():
            with self.subTest(fixture=name):
                outcome = validate_insight(
                    (FIXTURES / name).read_text(encoding="utf-8"), bundle
                )
                if reason is None:
                    self.assertEqual(outcome["status"], "valid", outcome)
                    continue
                self.assertEqual(outcome["reasonCode"], reason, outcome)
                self.assertEqual(outcome["detail"]["term"], term, outcome)

    def test_an_ocr_note_claiming_readers_get_hooked_rejects_globally(self):
        # It cites ocr:, not tribe:, so only the global rules can catch it.
        outcome = validate_insight(
            (FIXTURES / "ocr_reader_outcome_claim.json").read_text(encoding="utf-8"),
            extended_bundle(),
        )
        self.assertEqual(outcome["reasonCode"], "claim_boundary_violation")
        self.assertEqual(outcome["detail"]["term"], "hooked")
        self.assertNotIn("tribe:", json.dumps(outcome["detail"]))

    def test_the_fixture_corpus_runs_without_any_environment_flag(self):
        # Nothing in the corpus needs a model, a key, or a network.
        self.assertGreaterEqual(len(list(FIXTURES.glob("*.json"))), 40)


class ExportedVocabularyTests(unittest.TestCase):
    def test_the_frontend_json_is_generated_from_the_python_module(self):
        self.assertTrue(
            EXPORTED_TERMS.is_file(),
            "src/insight/claim-terms.json is missing; regenerate it from claim_terms.export_json()",
        )
        self.assertEqual(
            EXPORTED_TERMS.read_text(encoding="utf-8"),
            export_json(),
            "src/insight/claim-terms.json has drifted from backend/insight/claim_terms.py",
        )

    def test_no_second_copy_of_the_vocabulary_exists(self):
        # A hard-coded term list anywhere else is the defect this guards.
        offenders: list[str] = []
        for path in (REPOSITORY_ROOT / "src").rglob("*.js"):
            if path.name == "claim-terms.json":
                continue
            source = path.read_text(encoding="utf-8")
            if '"go viral"' in source and "claim-terms.json" not in source:
                offenders.append(str(path.relative_to(REPOSITORY_ROOT)))
        for path in (REPOSITORY_ROOT / "backend").rglob("*.py"):
            if path.name in {"claim_terms.py", "judge.py"} or "tests" in path.parts:
                continue
            if '"go viral"' in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(REPOSITORY_ROOT)))
        self.assertEqual(offenders, [])


class JudgeTripwireTests(unittest.TestCase):
    def test_the_judge_never_reaches_the_request_path(self):
        for module in ("service.py", "router.py", "validation.py", "bundle.py"):
            source = (REPOSITORY_ROOT / "backend" / "insight" / module).read_text(encoding="utf-8")
            tree = ast.parse(source)
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.add(node.module or "")
            self.assertNotIn("judge", imported, module)
            self.assertNotIn("backend.insight.judge", imported, module)

    def test_the_judge_prompt_restates_the_shared_vocabulary(self):
        for term in ("go viral", "retention", "attention", "subconscious"):
            self.assertIn(term, JUDGE_SYSTEM_PROMPT)
        self.assertIn("predicted average-subject cortical BOLD", JUDGE_SYSTEM_PROMPT)
        self.assertIn('{"violations": []}', JUDGE_SYSTEM_PROMPT)

    def test_items_are_collected_with_their_tribe_scope(self):
        artifact = json.loads((FIXTURES / "valid_full_artifact.json").read_text())
        items = collect_items(artifact)
        paths = {item["itemPath"] for item in items}
        self.assertIn("/hookReport/observations/0/text", paths)
        self.assertIn("/hookReport/experiments/0/edit", paths)
        self.assertIn("/tribeNotes/0/text", paths)
        tribe_note = next(item for item in items if item["itemPath"] == "/tribeNotes/0/text")
        self.assertTrue(tribe_note["citesTribe"])
        observation = next(
            item for item in items if item["itemPath"] == "/hookReport/observations/0/text"
        )
        self.assertFalse(observation["citesTribe"])

    def test_an_unreadable_verdict_is_not_a_clean_verdict(self):
        for raw in ("not json", "[]", '{"ok": true}', '{"violations": "none"}', '{"violations": [1]}'):
            with self.subTest(raw=raw):
                with self.assertRaises(JudgeUnavailable):
                    parse_judge_response(raw)

    def test_a_clean_verdict_parses_to_no_violations(self):
        self.assertEqual(parse_judge_response('{"violations": []}'), [])

    def test_judging_runs_entirely_through_the_injected_generator(self):
        artifact = json.loads((FIXTURES / "valid_full_artifact.json").read_text())
        captured: dict[str, str] = {}

        def generate(system: str, user: str) -> str:
            captured["system"] = system
            captured["user"] = user
            return '{"violations": [{"itemPath": "/tribeNotes/0/text", "quote": "x", "reason": "y"}]}'

        violations = judge_artifacts([artifact], generate=generate)
        self.assertEqual(violations[0]["itemPath"], "/tribeNotes/0/text")
        self.assertIn("auditing generated text", captured["system"])
        self.assertIn("citesTribe", captured["user"])

    def test_an_empty_artifact_set_cannot_be_judged(self):
        with self.assertRaises(JudgeUnavailable):
            build_judge_request([])

    @unittest.skipUnless(
        os.getenv("INSIGHT_JUDGE_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"},
        "The LLM judge tripwire is env-gated and skipped by default.",
    )
    def test_sampled_artifacts_pass_the_configured_provider_judge(self):
        from backend.insight.config import InsightSettings
        from backend.insight.providers import build_provider

        settings = InsightSettings.from_env()
        provider = build_provider(settings)
        availability = provider.availability()
        if not availability.available:
            self.skipTest(f"no provider is configured: {availability.reason}")

        samples = [
            json.loads((FIXTURES / name).read_text(encoding="utf-8"))
            for name in ("valid_full_artifact.json", "valid_hook_only_artifact.json")
        ]
        violations = judge_artifacts(
            samples, generate=lambda system, user: provider.generate_text(system, user).raw_output
        )
        self.assertEqual(violations, [], f"the judge flagged: {violations}")


if __name__ == "__main__":
    unittest.main()
