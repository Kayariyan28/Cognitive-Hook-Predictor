"""Experiments: the loop the insight lane opens, closed.

An experiment records one proposed edit, the signal shift it was proposed to
produce, and — once a variant clip has been analysed — what the signals
actually did. A measured delta is a change in a measured signal. It is not a
result, not a win, and not an outcome.

The direction rule here is the one `src/tribe/variantComparison.js` already
uses for A/B comparison; a test keeps the two in step rather than letting a
second rule drift into existence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re
from typing import Any, Callable, Mapping
from uuid import uuid4

from .bundle import BundleUnavailableError, assemble_evidence_bundle
from .citations import (
    CitationMalformedError,
    CitationUnresolvableError,
    parse_citation,
    resolve_citation,
)


EXPERIMENT_SCHEMA_VERSION = "insight-experiment/1"
EXPERIMENT_STATES = ("proposed", "edited", "compared")

# Mirrors COMPARISON_TOLERANCE in src/tribe/variantComparison.js.
COMPARISON_TOLERANCE = 1e-6

DIRECTIONS = ("increase", "decrease", "unchanged")
MATCH_STATES = ("matched", "opposite", "unmatched", "unmeasured")

ID_RE = re.compile(r"^[0-9a-f]{32}$")


class ExperimentError(ValueError):
    """The experiment request cannot be satisfied; the caller must fail closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def observed_direction(baseline: float, variant: float) -> tuple[float, str]:
    """Classify a measured change with the repo's existing A/B tolerance rule."""

    delta = variant - baseline
    tolerance = COMPARISON_TOLERANCE * max(1.0, abs(baseline), abs(variant))
    if abs(delta) <= tolerance:
        return 0.0, "unchanged"
    return delta, "increase" if delta > 0 else "decrease"


def classify_match(expected: str, observed: str) -> str:
    """Compare an untested expectation against what the signals actually did."""

    if observed == expected:
        return "matched"
    if {expected, observed} == {"increase", "decrease"}:
        return "opposite"
    return "unmatched"


def _numeric_at(bundle: Mapping[str, Any], metric_path: str) -> float | None:
    try:
        citation = parse_citation(metric_path)
    except CitationMalformedError:
        return None
    if citation.window is not None:
        return None
    try:
        value = resolve_citation(bundle, citation)
    except CitationUnresolvableError:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def measure_deltas(
    expected_signal_shift: list[Mapping[str, Any]],
    baseline_bundle: Mapping[str, Any],
    variant_bundle: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Measure each expected metric in both results, or say it was unmeasured."""

    deltas: list[dict[str, Any]] = []
    for shift in expected_signal_shift:
        metric_path = shift.get("metricPath")
        expected = shift.get("direction")
        if not isinstance(metric_path, str) or expected not in DIRECTIONS:
            raise ExperimentError("the experiment declares an invalid expected signal shift")
        baseline = _numeric_at(baseline_bundle, metric_path)
        variant = _numeric_at(variant_bundle, metric_path)
        if baseline is None or variant is None:
            deltas.append(
                {
                    "metricPath": metric_path,
                    "expectedDirection": expected,
                    "baselineValue": baseline,
                    "variantValue": variant,
                    "delta": None,
                    "observedDirection": None,
                    "match": "unmeasured",
                    "reason": (
                        "This metric path does not resolve to a number in "
                        f"{'the baseline result' if baseline is None else 'the variant result'}."
                    ),
                }
            )
            continue
        delta, observed = observed_direction(baseline, variant)
        deltas.append(
            {
                "metricPath": metric_path,
                "expectedDirection": expected,
                "baselineValue": baseline,
                "variantValue": variant,
                "delta": round(delta, 9),
                "observedDirection": observed,
                "match": classify_match(expected, observed),
                "reason": None,
            }
        )
    return deltas


@dataclass(frozen=True, slots=True)
class ExperimentRequest:
    insight_id: str
    experiment_id: str


class ExperimentTracker:
    """Create, advance, and measure experiments proposed by the insight lane."""

    def __init__(
        self,
        store: Any,
        *,
        forecast_result_loader: Callable[[str], Mapping[str, Any] | None],
    ) -> None:
        self.store = store
        self._load_forecast_result = forecast_result_loader

    # -- lifecycle ---------------------------------------------------------

    def create(self, request: ExperimentRequest) -> dict[str, Any]:
        artifact = self.store.read_artifact(request.insight_id)
        if artifact is None:
            raise ExperimentError(f"insight artifact {request.insight_id} was not found")
        hook_report = artifact.get("hookReport")
        experiments = hook_report.get("experiments") if isinstance(hook_report, Mapping) else None
        source = None
        for entry in experiments or ():
            if isinstance(entry, Mapping) and entry.get("id") == request.experiment_id:
                source = entry
                break
        if source is None:
            raise ExperimentError(
                f"insight {request.insight_id} proposes no experiment {request.experiment_id!r}"
            )
        baseline_result_id = (artifact.get("source") or {}).get("forecastResultId")
        if not isinstance(baseline_result_id, str) or not baseline_result_id:
            raise ExperimentError("the insight artifact names no baseline forecast result")

        record = {
            "schemaVersion": EXPERIMENT_SCHEMA_VERSION,
            "id": uuid4().hex,
            "createdAt": _utc_now(),
            "updatedAt": _utc_now(),
            "sourceInsightId": request.insight_id,
            "sourceExperimentId": request.experiment_id,
            "hypothesisIds": [source.get("hypothesisId")],
            "edit": source.get("edit"),
            "effort": source.get("effort"),
            "expectedSignalShift": [dict(shift) for shift in source.get("expectedSignalShift", [])],
            "status": "proposed",
            "linkedResultIds": [baseline_result_id],
            "measuredDeltas": None,
            "behavioralOutcome": False,
            "limits": (
                "A measured delta is a change in a measured signal between two analysed "
                "clips. It is not an audience outcome and it is not a result."
            ),
        }
        self.store.publish_experiment(record["id"], record)
        return record

    def mark_edited(self, experiment_id: str) -> dict[str, Any]:
        record = self._require(experiment_id)
        if record["status"] == "compared":
            raise ExperimentError("a compared experiment cannot return to the edited state")
        record["status"] = "edited"
        record["updatedAt"] = _utc_now()
        self.store.update_experiment(experiment_id, record)
        return record

    def attach_variant(self, experiment_id: str, variant_result_id: str) -> dict[str, Any]:
        record = self._require(experiment_id)
        baseline_result_id = record["linkedResultIds"][0]
        if variant_result_id == baseline_result_id:
            raise ExperimentError("the variant result must differ from the baseline result")

        baseline_bundle = self._bundle(baseline_result_id, "baseline")
        variant_bundle = self._bundle(variant_result_id, "variant")
        record["measuredDeltas"] = measure_deltas(
            record["expectedSignalShift"], baseline_bundle, variant_bundle
        )
        record["linkedResultIds"] = [baseline_result_id, variant_result_id]
        record["status"] = "compared"
        record["updatedAt"] = _utc_now()
        self.store.update_experiment(experiment_id, record)
        return record

    # -- reads -------------------------------------------------------------

    def read(self, experiment_id: str) -> dict[str, Any] | None:
        return self.store.read_experiment(experiment_id)

    def list_experiments(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.list_experiments(limit=limit)

    # -- internals ---------------------------------------------------------

    def _require(self, experiment_id: str) -> dict[str, Any]:
        if not ID_RE.fullmatch(experiment_id):
            raise ExperimentError("an experiment identifier is required")
        record = self.store.read_experiment(experiment_id)
        if record is None:
            raise ExperimentError(f"experiment {experiment_id} was not found")
        return record

    def _bundle(self, result_id: str, label: str) -> dict[str, Any]:
        result = self._load_forecast_result(result_id)
        if result is None:
            raise ExperimentError(f"the {label} forecast result {result_id} was not found")
        try:
            return assemble_evidence_bundle(result)
        except BundleUnavailableError as exc:
            raise ExperimentError(f"the {label} result cannot be compared: {exc}") from exc
