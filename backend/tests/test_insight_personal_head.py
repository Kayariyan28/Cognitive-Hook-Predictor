"""Roadmap step 7: a calibration candidate is evaluated, and approved by nothing."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

from backend.forecast.calibrated_heads import APPROVED_TARGET_CONTRACTS
from backend.insight.bundle import assemble_evidence_bundle
from backend.insight.personal_head import (
    CANDIDATE_SCHEMA_VERSION,
    MINIMUM_TOTAL_CLIPS,
    CalibrationCandidateError,
    build_candidate_manifest,
    build_training_table,
    evaluate_candidate,
    evaluate_personal_candidate,
)
from backend.tests.insight_support import forecast_result


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def clip(index: int, *, silence: float, rms: float) -> dict:
    result = forecast_result()
    identifier = f"{index:032x}"
    result["resultId"] = identifier
    result["jobId"] = identifier
    features = result["evidence"]["optionalProviders"]["measuredAudio"]["result"]["features"]
    features["measured_audio.silent_window_fraction"] = silence
    features["measured_audio.rms"] = rms
    features["measured_audio.peak"] = 0.5 + silence
    return result


def corpus(count: int):
    """A synthetic creator whose quieter openings correlate with the outcome."""

    results: dict[str, dict] = {}
    records = []
    for index in range(count):
        silence = 0.10 + (index % 10) * 0.03
        rms = 0.20 - (index % 7) * 0.01
        result = clip(index + 1, silence=silence, rms=rms)
        results[result["resultId"]] = result
        records.append(
            {
                "resultId": result["resultId"],
                "platform": "reels",
                "postedAt": f"2026-01-{(index % 28) + 1:02d}T10:00:00Z",
                "measuredAt": f"2026-02-{(index % 28) + 1:02d}T10:00:00Z",
                "metric": "retention_5s",
                "metricKind": "rate",
                "value": round(min(0.95, max(0.05, 0.8 - silence * 2)), 4),
                "denominator": "eligible_starts",
                "note": None,
            }
        )
    outcome_set = {
        "schemaVersion": "creator-outcome-set/1",
        "outcomeSetId": "a" * 32,
        "verified": False,
        "records": records,
    }
    return outcome_set, results


class TrainingTableTests(unittest.TestCase):
    def test_a_thin_history_cannot_be_evaluated(self):
        outcome_set, results = corpus(12)
        with self.assertRaises(CalibrationCandidateError) as caught:
            build_training_table(
                outcome_set, results, metric="retention_5s", assemble=assemble_evidence_bundle
            )
        self.assertEqual(caught.exception.reason_code, "insufficient_data")
        self.assertIn(str(MINIMUM_TOTAL_CLIPS), caught.exception.detail)

    def test_rows_are_ordered_chronologically_never_shuffled(self):
        outcome_set, results = corpus(40)
        rows = build_training_table(
            outcome_set, results, metric="retention_5s", assemble=assemble_evidence_bundle
        )
        self.assertEqual([row.posted_at for row in rows], sorted(row.posted_at for row in rows))

    def test_a_count_metric_needs_a_declared_threshold(self):
        outcome_set, results = corpus(40)
        for record in outcome_set["records"]:
            record["metric"] = "views"
            record["metricKind"] = "count"
            record["value"] = 1000
        with self.assertRaises(CalibrationCandidateError) as caught:
            build_training_table(
                outcome_set, results, metric="views", assemble=assemble_evidence_bundle
            )
        self.assertEqual(caught.exception.reason_code, "undeclared_threshold")

    def test_a_declared_threshold_turns_a_count_into_a_label(self):
        outcome_set, results = corpus(40)
        for index, record in enumerate(outcome_set["records"]):
            record["metric"] = "views"
            record["metricKind"] = "count"
            record["value"] = 500 + index * 100
        rows = build_training_table(
            outcome_set,
            results,
            metric="views",
            assemble=assemble_evidence_bundle,
            threshold=2000,
        )
        self.assertEqual({row.label for row in rows}, {0.0, 1.0})


class EvaluationTests(unittest.TestCase):
    def rows(self, count=44):
        outcome_set, results = corpus(count)
        return build_training_table(
            outcome_set, results, metric="retention_5s", assemble=assemble_evidence_bundle
        )

    def test_the_split_is_chronological_and_reported(self):
        evaluation = evaluate_candidate(self.rows())
        self.assertEqual(evaluation["splitKind"], "chronological-holdout")
        self.assertGreaterEqual(evaluation["testClipCount"], 10)
        self.assertEqual(
            evaluation["trainClipCount"] + evaluation["testClipCount"], len(self.rows())
        )
        self.assertLessEqual(evaluation["trainStartedAt"], evaluation["testStartedAt"])

    def test_every_gate_metric_is_reported(self):
        evaluation = evaluate_candidate(self.rows())
        for key in (
            "brierScore",
            "logLoss",
            "expectedCalibrationError",
            "calibrationSlope",
            "calibrationIntercept",
            "baseRate",
            "baselineBrierScore",
            "beatsBaseRate",
        ):
            self.assertIn(key, evaluation)
        self.assertGreaterEqual(evaluation["brierScore"], 0.0)
        self.assertGreaterEqual(evaluation["expectedCalibrationError"], 0.0)

    def test_a_candidate_is_compared_against_the_creators_own_base_rate(self):
        evaluation = evaluate_candidate(self.rows())
        self.assertIsInstance(evaluation["beatsBaseRate"], bool)
        self.assertGreater(evaluation["baselineBrierScore"], 0.0)

    def test_identical_feature_vectors_across_the_split_are_refused(self):
        outcome_set, results = corpus(40)
        # Collapse every clip onto the same measurements: the test set would
        # then be indistinguishable from the training set.
        for result in results.values():
            features = result["evidence"]["optionalProviders"]["measuredAudio"]["result"]["features"]
            features["measured_audio.silent_window_fraction"] = 0.2
            features["measured_audio.rms"] = 0.15
            features["measured_audio.peak"] = 0.7
        rows = build_training_table(
            outcome_set, results, metric="retention_5s", assemble=assemble_evidence_bundle
        )
        with self.assertRaises(CalibrationCandidateError) as caught:
            evaluate_candidate(rows)
        self.assertEqual(caught.exception.reason_code, "near_duplicate_leakage")

    def test_a_single_valued_outcome_teaches_nothing(self):
        outcome_set, results = corpus(40)
        for record in outcome_set["records"]:
            record["value"] = 0.5
        rows = build_training_table(
            outcome_set, results, metric="retention_5s", assemble=assemble_evidence_bundle
        )
        with self.assertRaises(CalibrationCandidateError) as caught:
            evaluate_candidate(rows)
        self.assertEqual(caught.exception.reason_code, "degenerate_labels")


class ApprovalTests(unittest.TestCase):
    def manifest(self):
        outcome_set, results = corpus(44)
        return evaluate_personal_candidate(
            outcome_set,
            results,
            metric="retention_5s",
            platform="reels",
            assemble=assemble_evidence_bundle,
        )

    def test_the_manifest_approves_nothing(self):
        manifest = self.manifest()
        self.assertEqual(manifest["schemaVersion"], CANDIDATE_SCHEMA_VERSION)
        self.assertIs(manifest["approved"], False)
        self.assertIs(manifest["scoringAvailable"], False)
        self.assertIs(manifest["behavioralOutcome"], False)
        self.assertIn("not a head", manifest["limits"])

    def test_the_manifest_lists_what_is_still_missing(self):
        blockers = " ".join(self.manifest()["blockers"]).lower()
        for phrase in (
            "approved_target_contracts",
            "creator-declared and unverified",
            "prospective shadow validation",
            "drift monitoring",
            "one creator only",
        ):
            self.assertIn(phrase, blockers)

    def test_a_candidate_that_loses_to_the_base_rate_says_so_first(self):
        manifest = build_candidate_manifest(
            {"beatsBaseRate": False, "brierScore": 0.4, "baselineBrierScore": 0.2},
            metric="retention_5s",
            platform="reels",
        )
        self.assertIn("does not beat", manifest["blockers"][0])

    def test_the_manifest_is_digested_for_review(self):
        manifest = self.manifest()
        self.assertEqual(len(manifest["manifestSha256"]), 64)

    def test_the_production_approval_table_is_still_empty(self):
        for metric, contracts in APPROVED_TARGET_CONTRACTS.items():
            self.assertEqual(contracts, (), metric)

    def test_the_harness_cannot_reach_the_approval_table(self):
        source = (REPOSITORY_ROOT / "backend/insight/personal_head.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        self.assertNotIn("backend.forecast.calibrated_heads", imported)
        self.assertNotIn("calibrated_heads", imported)
        # The module names the approval table in prose, to explain that it is
        # empty and stays that way. What matters is that it cannot assign to it.
        assignments = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertNotIn("APPROVED_TARGET_CONTRACTS", assignments)

    def test_no_probability_from_the_harness_reaches_a_creator(self):
        # The manifest is a review document: it carries evaluation metrics, but
        # no per-clip probability anyone could render.
        manifest = self.manifest()
        # Aggregate evaluation metrics only. No per-clip score exists in the
        # document, so there is nothing a UI could render as a forecast.
        self.assertNotIn("predictions", manifest["evaluation"])
        self.assertNotIn("scores", manifest["evaluation"])
        self.assertFalse(
            any(isinstance(value, list) and value and isinstance(value[0], float)
                for value in manifest["evaluation"].values())
        )
        self.assertIs(manifest["scoringAvailable"], False)


if __name__ == "__main__":
    unittest.main()
