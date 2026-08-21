"""Compare several analysed cuts of the same idea, side by side.

This is the step that turns advice into a decision: a creator records two or
three openings, the pipeline measures all of them, and this module lays the
measurements out together. Nothing here ranks a cut as better. It reports which
cut sits highest and lowest on each measured signal, using the same tolerance
rule the A/B lab already applies, and says plainly that a difference in a
measured signal is not a difference in outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .comparative import COMPARATIVE_METRIC_PATHS, extract_metric_values
from .experiments import COMPARISON_TOLERANCE


VARIANT_COMPARISON_SCHEMA_VERSION = "insight-variant-comparison/1"
MINIMUM_VARIANTS = 2
MAXIMUM_VARIANTS = 6


class VariantComparisonError(ValueError):
    """The comparison is refused; a partial comparison would mislead."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail

    def as_dict(self) -> dict[str, Any]:
        return {"unavailable": True, "reasonCode": self.reason_code, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class Variant:
    result_id: str
    label: str
    values: Mapping[str, float]


def _spread(values: Sequence[float]) -> float:
    return max(values) - min(values)


def _is_flat(values: Sequence[float]) -> bool:
    """One tolerance rule across the project, from src/tribe/variantComparison.js."""

    largest = max(abs(value) for value in values)
    tolerance = COMPARISON_TOLERANCE * max(1.0, largest)
    return _spread(values) <= tolerance


def build_variant_comparison(variants: Sequence[Variant]) -> dict[str, Any]:
    """Lay measured signals out across cuts, naming what is not comparable."""

    if not MINIMUM_VARIANTS <= len(variants) <= MAXIMUM_VARIANTS:
        raise VariantComparisonError(
            "invalid_request",
            f"a comparison needs {MINIMUM_VARIANTS} to {MAXIMUM_VARIANTS} analysed cuts",
        )
    identifiers = [variant.result_id for variant in variants]
    if len(set(identifiers)) != len(identifiers):
        raise VariantComparisonError(
            "invalid_request", "each cut in a comparison must be a different result"
        )

    metrics: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for path in COMPARATIVE_METRIC_PATHS:
        present = [
            (variant, variant.values[path]) for variant in variants if path in variant.values
        ]
        # A metric measured on only some cuts is not comparable across them, and
        # filling the gap would invent the number that decides the comparison.
        if len(present) != len(variants):
            skipped.append(
                {
                    "metricPath": path,
                    "reason": (
                        f"measured on {len(present)} of {len(variants)} cuts, so the cuts "
                        "are not comparable on it"
                    ),
                }
            )
            continue
        values = [value for _, value in present]
        flat = _is_flat(values)
        lowest = min(present, key=lambda entry: entry[1])
        highest = max(present, key=lambda entry: entry[1])
        metrics.append(
            {
                "metricPath": path,
                "values": [
                    {
                        "resultId": variant.result_id,
                        "label": variant.label,
                        "value": value,
                    }
                    for variant, value in present
                ],
                "spread": round(_spread(values), 9),
                "differs": not flat,
                "lowestResultId": None if flat else lowest[0].result_id,
                "highestResultId": None if flat else highest[0].result_id,
            }
        )

    if not metrics:
        raise VariantComparisonError(
            "no_comparable_metric",
            "these cuts share no measured signal, so there is nothing to compare",
        )
    return {
        "schemaVersion": VARIANT_COMPARISON_SCHEMA_VERSION,
        "variants": [
            {"resultId": variant.result_id, "label": variant.label} for variant in variants
        ],
        "metrics": metrics,
        "skippedMetrics": skipped,
        "differingMetricCount": sum(1 for metric in metrics if metric["differs"]),
        "behavioralOutcome": False,
        "limits": (
            "These are measured signal differences between cuts of the same idea. Higher "
            "is not better and lower is not worse: no cut here has been shown to an "
            "audience, and nothing in this comparison predicts what one would do. Use it "
            "to see what an edit actually changed."
        ),
    }


def variants_from_results(
    results: Iterable[Mapping[str, Any]],
    *,
    assemble: Any,
    labels: Mapping[str, str] | None = None,
) -> list[Variant]:
    """Measure each cut through the citation resolver, exactly as the bundle does."""

    declared = labels or {}
    variants: list[Variant] = []
    for index, result in enumerate(results):
        result_id = result.get("resultId")
        if not isinstance(result_id, str) or not result_id:
            raise VariantComparisonError(
                "invalid_request", "every cut must be a published forecast result"
            )
        values = extract_metric_values(assemble(result))
        if not values:
            raise VariantComparisonError(
                "no_comparable_metric",
                f"result {result_id} carries no measured signal to compare",
            )
        variants.append(
            Variant(
                result_id=result_id,
                label=declared.get(result_id) or f"Cut {index + 1}",
                values=values,
            )
        )
    return variants
