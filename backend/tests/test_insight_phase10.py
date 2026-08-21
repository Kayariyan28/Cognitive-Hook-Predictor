"""Phase 10: comparative context, outcome labels, and the wall between them.

The two features in this phase pull in opposite directions. Retrieval compares a
creator's own measurements; outcome ingestion stores their results. These tests
exist mostly to prove the second can never reach the first.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.insight.bundle import (
    BUNDLE_SCHEMA_VERSION,
    assemble_evidence_bundle,
    hook_evidence_card,
)
from backend.insight.citations import (
    CitationMalformedError,
    parse_citation,
    resolve_citation,
)
from backend.insight.comparative import (
    COMPARATIVE_METRIC_PATHS,
    CorpusClip,
    build_comparative,
    corpus_from_results,
    extract_metric_values,
)
from backend.insight.config import InsightSettings
from backend.insight.outcomes import (
    OUTCOME_METRICS,
    OutcomeImportError,
    OutcomeLedger,
    parse_outcome_csv,
)
from backend.insight.router import create_insight_router
from backend.insight.service import InsightRequest, InsightService
from backend.insight.store import InsightStore
from backend.insight.validation import validate_insight
from backend.tests.insight_support import (
    FORECAST_RESULT_ID,
    forecast_result,
    tribe_descriptors,
)
from backend.tests.test_insight_service import FakeProvider


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "insight_fixtures"
PINNED_REVISION = "b" * 40

HEADER = "resultId,platform,postedAt,measuredAt,metric,value,denominator,note"
VALID_ROW = (
    f"{FORECAST_RESULT_ID},reels,2026-02-03T18:00:00Z,2026-02-10T18:00:00Z,views,18432,,"
)


def corpus_result(index: int, *, silence: float) -> dict:
    result = forecast_result()
    identifier = f"{index:032x}"
    result["resultId"] = identifier
    result["jobId"] = identifier
    result["createdAt"] = f"2026-01-{(index % 27) + 1:02d}T10:00:00Z"
    features = result["evidence"]["optionalProviders"]["measuredAudio"]["result"]["features"]
    features["measured_audio.silent_window_fraction"] = silence
    return result


class ComparativeMathTests(unittest.TestCase):
    def clips(self, count: int, *, path: str, start: float, step: float):
        return [
            CorpusClip(
                result_id=f"{index:032x}",
                created_at=f"2026-01-{index + 1:02d}T10:00:00Z",
                values={path: start + step * index},
            )
            for index in range(count)
        ]

    def test_a_small_corpus_produces_no_comparison(self):
        path = "measured:/audio/descriptors/rms"
        outcome = build_comparative({path: 0.1}, self.clips(4, path=path, start=0.2, step=0.1))
        self.assertEqual(outcome["status"], "absent")
        self.assertIn("at least 20", outcome["reason"])
        self.assertIn("has 5", outcome["reason"])

    def test_rank_and_percentile_are_exact(self):
        path = "measured:/audio/descriptors/rms"
        # Corpus values 1.0 … 20.0; subject 0.5 is the lowest of 21.
        corpus = self.clips(20, path=path, start=1.0, step=1.0)
        outcome = build_comparative({path: 0.5}, corpus)
        self.assertEqual(outcome["status"], "present")
        metric = outcome["metrics"][0]
        self.assertEqual(metric["rank"], 1)
        self.assertEqual(metric["outOf"], 21)
        self.assertEqual(metric["percentile"], 0.0)
        self.assertEqual(metric["corpusMinimum"], 0.5)
        self.assertEqual(metric["corpusMaximum"], 20.0)

        highest = build_comparative({path: 99.0}, corpus)["metrics"][0]
        self.assertEqual(highest["rank"], 21)
        self.assertEqual(highest["percentile"], 100.0)

    def test_the_corpus_is_declared_not_implied(self):
        path = "measured:/audio/descriptors/rms"
        outcome = build_comparative({path: 5.0}, self.clips(24, path=path, start=1.0, step=1.0))
        self.assertEqual(outcome["corpus"]["clipCount"], 25)
        self.assertEqual(outcome["corpus"]["kind"], "local-forecast-results")
        self.assertTrue(outcome["corpus"]["oldestCreatedAt"])
        self.assertLessEqual(
            outcome["corpus"]["oldestCreatedAt"], outcome["corpus"]["newestCreatedAt"]
        )

    def test_a_metric_missing_from_the_corpus_is_excluded_not_imputed(self):
        path = "measured:/audio/descriptors/rms"
        corpus = self.clips(24, path=path, start=1.0, step=1.0)
        outcome = build_comparative(
            {path: 5.0, "vjepa:/descriptors/temporal_change_mean": 0.3}, corpus
        )
        paths = [metric["metricPath"] for metric in outcome["metrics"]]
        self.assertIn(path, paths)
        self.assertNotIn("vjepa:/descriptors/temporal_change_mean", paths)

    def test_the_window_bounds_the_corpus(self):
        path = "measured:/audio/descriptors/rms"
        corpus = self.clips(40, path=path, start=1.0, step=1.0)
        outcome = build_comparative({path: 5.0}, corpus, window=24)
        self.assertEqual(outcome["metrics"][0]["outOf"], 25)

    def test_the_metric_list_is_code_owned(self):
        self.assertIn("measured:/audio/descriptors/silent_window_fraction", COMPARATIVE_METRIC_PATHS)
        for path in COMPARATIVE_METRIC_PATHS:
            citation = parse_citation(path)
            self.assertIsNone(citation.window, path)
            self.assertIn(citation.lane, {"measured", "ast", "vjepa"}, path)


class ComparativeBundleTests(unittest.TestCase):
    def test_an_uncomputed_comparison_is_an_explicit_absent_marker(self):
        bundle = assemble_evidence_bundle(forecast_result())
        comparative = bundle["lanes"]["measured"]["comparative"]
        self.assertEqual(comparative["status"], "absent")
        self.assertTrue(comparative["reason"])
        self.assertEqual(bundle["schemaVersion"], BUNDLE_SCHEMA_VERSION)
        self.assertEqual(BUNDLE_SCHEMA_VERSION, "insight-evidence-bundle/2")

    def test_comparative_context_is_citable_and_survives_hook_slicing(self):
        path = "measured:/audio/descriptors/silent_window_fraction"
        corpus = [
            CorpusClip(f"{i:032x}", f"2026-01-{i + 1:02d}T10:00:00Z", {path: 0.30 + i * 0.01})
            for i in range(20)
        ]
        comparative = build_comparative({path: 0.1875}, corpus)
        bundle = assemble_evidence_bundle(forecast_result(), comparative=comparative)
        resolved = resolve_citation(
            bundle, parse_citation("measured:/comparative/metrics/0/rank")
        )
        self.assertEqual(resolved, 1)
        card = hook_evidence_card(bundle)
        self.assertEqual(
            resolve_citation(card, parse_citation("measured:/comparative/corpus/clipCount")),
            21,
        )

    def test_extraction_reads_only_through_the_citation_resolver(self):
        values = extract_metric_values(assemble_evidence_bundle(forecast_result()))
        self.assertEqual(values["measured:/audio/descriptors/rms"], 0.1372)
        self.assertNotIn("measured:/audio/descriptors/does_not_exist", values)

    def test_one_unreadable_prior_result_never_breaks_the_corpus(self):
        results = [corpus_result(index, silence=0.2) for index in range(3)]
        results.append({"resultId": "broken", "schemaVersion": "wrong"})
        clips = corpus_from_results(
            results, exclude_result_id="", assemble=assemble_evidence_bundle
        )
        self.assertEqual(len(clips), 3)
        self.assertEqual([clip.created_at for clip in clips], sorted(
            [clip.created_at for clip in clips], reverse=True
        ))


class ComparativeServiceTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="phase10-")
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)

    def service(self, corpus, **overrides):
        environ = {
            "INSIGHT_PROVIDER": "mlx-local",
            "INSIGHT_LOCAL_MODEL_REVISION": PINNED_REVISION,
            "INSIGHT_DIR": str(self.root / "insight"),
        }
        environ.update(overrides)
        settings = InsightSettings.from_env(environ)
        return InsightService(
            settings,
            forecast_result_loader={FORECAST_RESULT_ID: forecast_result()}.get,
            store=InsightStore(settings.insight_dir),
            provider_factory=lambda _: FakeProvider(),
            forecast_corpus_loader=lambda limit: corpus[:limit],
        )

    def test_a_thin_corpus_yields_an_absent_comparison(self):
        service = self.service([corpus_result(i, silence=0.3) for i in range(3)])
        bundle = service._bundle(InsightRequest(FORECAST_RESULT_ID))
        self.assertEqual(bundle["lanes"]["measured"]["comparative"]["status"], "absent")

    def test_a_full_corpus_produces_a_ranked_comparison(self):
        corpus = [corpus_result(i, silence=0.30 + i * 0.01) for i in range(1, 25)]
        service = self.service(corpus)
        bundle = service._bundle(InsightRequest(FORECAST_RESULT_ID))
        comparative = bundle["lanes"]["measured"]["comparative"]
        self.assertEqual(comparative["status"], "present")
        silence = next(
            metric
            for metric in comparative["metrics"]
            if metric["metricPath"].endswith("silent_window_fraction")
        )
        self.assertEqual(silence["value"], 0.1875)
        self.assertEqual(silence["rank"], 1)

    def test_a_broken_corpus_loader_never_blocks_the_report(self):
        def explode(limit):
            raise RuntimeError("the result directory is unreadable")

        settings = InsightSettings.from_env(
            {"INSIGHT_LOCAL_MODEL_REVISION": PINNED_REVISION, "INSIGHT_DIR": str(self.root / "b")}
        )
        service = InsightService(
            settings,
            forecast_result_loader={FORECAST_RESULT_ID: forecast_result()}.get,
            store=InsightStore(settings.insight_dir),
            provider_factory=lambda _: FakeProvider(),
            forecast_corpus_loader=explode,
        )
        bundle = service._bundle(InsightRequest(FORECAST_RESULT_ID))
        self.assertEqual(bundle["lanes"]["measured"]["comparative"]["status"], "absent")

    def test_generation_still_succeeds_with_comparative_context(self):
        corpus = [corpus_result(i, silence=0.30 + i * 0.01) for i in range(1, 25)]
        service = self.service(corpus)
        artifact, _ = service.generate(
            InsightRequest(FORECAST_RESULT_ID, tribe_descriptors=tribe_descriptors())
        )
        self.assertEqual(artifact["schemaVersion"], "insight/1")
        self.assertEqual(artifact["provenance"]["promptTemplateId"], "hook-doctor.v2")


class OutcomeImportTests(unittest.TestCase):
    def parse(self, csv_text: str, *, known=(FORECAST_RESULT_ID,)):
        return parse_outcome_csv(csv_text, result_exists=lambda value: value in known)

    def test_a_valid_csv_imports_whole(self):
        rate_row = (
            f"{FORECAST_RESULT_ID},reels,2026-02-03T18:00:00Z,2026-02-10T18:00:00Z,"
            "retention_5s,0.41,eligible_starts,second attempt"
        )
        records = self.parse(f"{HEADER}\n{VALID_ROW}\n{rate_row}\n")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].metric, "views")
        self.assertEqual(records[1].value, 0.41)
        self.assertEqual(records[1].denominator, "eligible_starts")

    def test_every_refusal_is_whole_file_and_names_its_row(self):
        cases = {
            "unknown_metric": f"{FORECAST_RESULT_ID},reels,2026-02-03T18:00:00Z,2026-02-10T18:00:00Z,vibes,5,,",
            "invalid_value": f"{FORECAST_RESULT_ID},reels,2026-02-03T18:00:00Z,2026-02-10T18:00:00Z,retention_5s,1.4,eligible_starts,",
            "invalid_timestamp": f"{FORECAST_RESULT_ID},reels,2026-02-11T18:00:00Z,2026-02-10T18:00:00Z,views,5,,",
            "unknown_result": f"{'9' * 32},reels,2026-02-03T18:00:00Z,2026-02-10T18:00:00Z,views,5,,",
            "invalid_result_id": "not-an-id,reels,2026-02-03T18:00:00Z,2026-02-10T18:00:00Z,views,5,,",
            "missing_denominator": f"{FORECAST_RESULT_ID},reels,2026-02-03T18:00:00Z,2026-02-10T18:00:00Z,retention_5s,0.4,,",
            "invalid_value_counts": f"{FORECAST_RESULT_ID},reels,2026-02-03T18:00:00Z,2026-02-10T18:00:00Z,views,12.5,,",
        }
        for expected, row in cases.items():
            with self.subTest(case=expected):
                with self.assertRaises(OutcomeImportError) as caught:
                    self.parse(f"{HEADER}\n{VALID_ROW}\n{row}\n")
                self.assertEqual(caught.exception.row, 3)
                if not expected.endswith("_counts"):
                    self.assertEqual(caught.exception.reason_code, expected)

    def test_a_duplicate_measurement_rejects_the_file(self):
        with self.assertRaises(OutcomeImportError) as caught:
            self.parse(f"{HEADER}\n{VALID_ROW}\n{VALID_ROW}\n")
        self.assertEqual(caught.exception.reason_code, "duplicate_record")

    def test_headers_are_closed(self):
        for header in (
            "resultId,platform,postedAt,measuredAt,metric",
            f"{HEADER},sentiment",
        ):
            with self.subTest(header=header):
                with self.assertRaises(OutcomeImportError) as caught:
                    self.parse(f"{header}\n")
                self.assertEqual(caught.exception.reason_code, "invalid_header")

    def test_an_empty_file_is_refused(self):
        with self.assertRaises(OutcomeImportError):
            self.parse(f"{HEADER}\n")

    def test_the_metric_vocabulary_is_code_owned(self):
        self.assertEqual(OUTCOME_METRICS["retention_5s"], "rate")
        self.assertEqual(OUTCOME_METRICS["views"], "count")
        self.assertNotIn("hook_strength", OUTCOME_METRICS)


class OutcomeLedgerTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="outcomes-")
        self.addCleanup(self._temporary.cleanup)
        self.store = InsightStore(Path(self._temporary.name) / "insight")
        self.ledger = OutcomeLedger(
            self.store, result_exists=lambda value: value == FORECAST_RESULT_ID
        )

    def test_an_imported_set_records_that_it_is_unverified(self):
        document = self.ledger.import_csv(f"{HEADER}\n{VALID_ROW}\n")
        self.assertEqual(document["schemaVersion"], "creator-outcome-set/1")
        self.assertIs(document["verified"], False)
        self.assertIs(document["isTrainingLabelOnly"], True)
        self.assertEqual(document["declaredBy"], "creator")
        self.assertEqual(len(document["sourceFileSha256"]), 64)
        self.assertIn("did not measure or verify", document["limits"])
        self.assertIn("read by the insight lane", document["limits"])

    def test_listing_returns_summaries_not_every_record(self):
        self.ledger.import_csv(f"{HEADER}\n{VALID_ROW}\n")
        listed = self.ledger.list_sets()
        self.assertEqual(len(listed), 1)
        self.assertNotIn("records", listed[0])
        self.assertEqual(listed[0]["recordCount"], 1)

    def test_deletion_is_real_and_immediate(self):
        document = self.ledger.import_csv(f"{HEADER}\n{VALID_ROW}\n")
        identifier = document["outcomeSetId"]
        self.assertTrue(self.ledger.delete(identifier))
        self.assertIsNone(self.ledger.read(identifier))
        self.assertFalse(self.ledger.delete(identifier))
        self.assertEqual(self.ledger.list_sets(), [])


class SeparationTests(unittest.TestCase):
    """The wall between outcome labels and everything the model can see."""

    def reachable(self, entry: str) -> set[str]:
        seen: set[str] = set()
        pending = [entry]
        while pending:
            module_name = pending.pop()
            if module_name in seen:
                continue
            seen.add(module_name)
            path = REPOSITORY_ROOT / Path(module_name.replace(".", "/") + ".py")
            if not path.is_file():
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            package = module_name.rsplit(".", 1)[0]
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    pending.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        parts = package.split(".")
                        base = ".".join(parts[: len(parts) - node.level + 1])
                        pending.append(f"{base}.{node.module}" if node.module else base)
                    elif node.module:
                        pending.append(node.module)
        return seen

    def test_nothing_the_model_sees_can_reach_the_outcome_store(self):
        for module in (
            "backend.insight.bundle",
            "backend.insight.comparative",
            "backend.insight.validation",
            "backend.insight.service",
            "backend.insight.prompts.hook_doctor",
            "backend.insight.providers.anthropic_cloud",
            "backend.insight.providers.mlx_local",
        ):
            with self.subTest(module=module):
                self.assertNotIn("backend.insight.outcomes", self.reachable(module))

    def test_there_is_no_outcomes_lane_to_cite(self):
        from backend.insight.bundle import LANE_KEYS

        self.assertNotIn("outcomes", LANE_KEYS)
        with self.assertRaises(CitationMalformedError):
            parse_citation("outcomes:/records/0/value")

    def test_an_artifact_citing_outcomes_is_refused(self):
        bundle = assemble_evidence_bundle(
            forecast_result(), tribe_descriptors=tribe_descriptors()
        )
        outcome = validate_insight(
            (FIXTURES / "outcomes_lane_citation.json").read_text(encoding="utf-8"), bundle
        )
        self.assertEqual(outcome["reasonCode"], "citation_malformed")

    def test_the_outcome_module_reads_no_evidence(self):
        source = (REPOSITORY_ROOT / "backend/insight/outcomes.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        self.assertEqual(imported & {"bundle", "citations", "comparative", "validation"}, set())


class ComparativeClaimTests(unittest.TestCase):
    def bundle(self):
        path = "measured:/audio/descriptors/silent_window_fraction"
        corpus = [
            CorpusClip(f"{i:032x}", f"2026-01-{i + 1:02d}T10:00:00Z", {path: 0.30 + i * 0.01})
            for i in range(20)
        ]
        return assemble_evidence_bundle(
            forecast_result(),
            tribe_descriptors=tribe_descriptors(),
            comparative=build_comparative({path: 0.1875}, corpus),
        )

    def test_a_rank_may_be_restated_and_cited(self):
        outcome = validate_insight(
            (FIXTURES / "comparative_rank_valid_twin.json").read_text(encoding="utf-8"),
            self.bundle(),
        )
        self.assertEqual(outcome["status"], "valid", outcome)

    def test_a_rank_restated_as_a_verdict_is_still_refused(self):
        outcome = validate_insight(
            (FIXTURES / "comparative_verdict_claim.json").read_text(encoding="utf-8"),
            self.bundle(),
        )
        self.assertEqual(outcome["reasonCode"], "claim_boundary_violation")

    def test_an_invented_rank_fails_the_numeric_copy_rule(self):
        payload = json.loads(
            (FIXTURES / "comparative_rank_valid_twin.json").read_text(encoding="utf-8")
        )
        payload["hookReport"]["observations"][0]["text"] = (
            "Measured silent-window coverage here is 3 of 40 among this machine's analysed clips."
        )
        outcome = validate_insight(json.dumps(payload), self.bundle())
        self.assertEqual(outcome["reasonCode"], "numeric_not_in_evidence")


class OutcomeEndpointTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="outcome-routes-")
        self.addCleanup(self._temporary.cleanup)
        root = Path(self._temporary.name)
        settings = InsightSettings.from_env(
            {"INSIGHT_LOCAL_MODEL_REVISION": PINNED_REVISION, "INSIGHT_DIR": str(root / "insight")}
        )
        self.store = InsightStore(settings.insight_dir)
        service = InsightService(
            settings,
            forecast_result_loader={FORECAST_RESULT_ID: forecast_result()}.get,
            store=self.store,
            provider_factory=lambda _: FakeProvider(),
        )
        ledger = OutcomeLedger(
            self.store, result_exists=lambda value: value == FORECAST_RESULT_ID
        )
        app = FastAPI()
        app.include_router(create_insight_router(service=service, outcome_ledger=ledger))
        self.client = TestClient(app)

    def test_the_import_lifecycle(self):
        created = self.client.post(
            "/api/insight/v1/outcomes",
            content=f"{HEADER}\n{VALID_ROW}\n".encode("utf-8"),
            headers={"Content-Type": "text/csv"},
        )
        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.headers["cache-control"], "private, no-store")
        identifier = created.json()["outcomeSetId"]

        listed = self.client.get("/api/insight/v1/outcomes")
        self.assertEqual(len(listed.json()["outcomeSets"]), 1)
        self.assertIn("never read by the insight lane", listed.json()["limits"])

        removed = self.client.delete(f"/api/insight/v1/outcomes/{identifier}")
        self.assertEqual(removed.status_code, 200)
        self.assertTrue(removed.json()["deleted"])
        self.assertEqual(
            self.client.delete(f"/api/insight/v1/outcomes/{identifier}").status_code, 404
        )

    def test_a_rejected_import_explains_itself_and_stores_nothing(self):
        bad = f"{HEADER}\n{FORECAST_RESULT_ID},reels,2026-02-03T18:00:00Z,2026-02-10T18:00:00Z,vibes,5,,\n"
        response = self.client.post(
            "/api/insight/v1/outcomes",
            content=bad.encode("utf-8"),
            headers={"Content-Type": "text/csv"},
        )
        self.assertEqual(response.status_code, 422)
        payload = response.json()
        self.assertTrue(payload["unavailable"])
        self.assertEqual(payload["reasonCode"], "unknown_metric")
        self.assertEqual(payload["row"], 2)
        self.assertEqual(self.client.get("/api/insight/v1/outcomes").json()["outcomeSets"], [])

    def test_outcomes_are_absent_when_no_ledger_is_configured(self):
        settings = InsightSettings.from_env({"INSIGHT_DIR": str(self.store.root)})
        service = InsightService(
            settings,
            forecast_result_loader=lambda _: None,
            store=self.store,
            provider_factory=lambda _: FakeProvider(),
        )
        app = FastAPI()
        app.include_router(create_insight_router(service=service))
        client = TestClient(app)
        self.assertEqual(client.get("/api/insight/v1/outcomes").status_code, 503)


if __name__ == "__main__":
    unittest.main()
