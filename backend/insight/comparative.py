"""Comparative context from the creator's own past measurements.

A percentile among a creator's own measured clips is a measurement about
measurements. It is not a prediction and it is not an outcome, which is why it
can live in the ``measured`` lane as ordinary citable evidence.

Two rules keep it that way. The metric list is code-owned, so the model can
never ask for an arbitrary comparison. And a corpus too small to mean anything
produces an explicit absent marker rather than a percentile over four clips.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

from .citations import (
    CitationMalformedError,
    CitationUnresolvableError,
    parse_citation,
    resolve_citation,
)


COMPARATIVE_SCHEMA_VERSION = "insight-comparative/1"

# Code-owned. A metric appears here because a creator can act on it and because
# it is a scalar the pipeline measures the same way on every clip.
COMPARATIVE_METRIC_PATHS: tuple[str, ...] = (
    "measured:/video/durationSeconds",
    "measured:/audio/descriptors/rms",
    "measured:/audio/descriptors/peak",
    "measured:/audio/descriptors/silent_window_fraction",
    "measured:/audio/descriptors/window_dynamic_range_db",
    "measured:/audio/descriptors/spectral_centroid_hz_mean",
    "measured:/audio/descriptors/normalized_spectral_flux_mean",
    "ast:/descriptors/mean_top_label_score",
    "vjepa:/descriptors/temporal_change_mean",
    "vjepa:/descriptors/temporal_change_peak",
    "vjepa:/descriptors/temporal_consistency_mean",
)

# Mirrors MINIMUM_TREND_REFERENCE_SIZE in src/forecast/contract.js: below this,
# a comparison describes noise rather than a creator's habits.
DEFAULT_MINIMUM_CLIPS = 20
DEFAULT_WINDOW = 50


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def extract_metric_values(
    bundle: Mapping[str, Any], paths: Iterable[str] = COMPARATIVE_METRIC_PATHS
) -> dict[str, float]:
    """Read the eligible metrics out of one bundle through the citation resolver."""

    values: dict[str, float] = {}
    for path in paths:
        try:
            citation = parse_citation(path)
            resolved = resolve_citation(bundle, citation)
        except (CitationMalformedError, CitationUnresolvableError):
            continue
        number = _finite(resolved)
        if number is not None:
            values[path] = number
    return values


def _median(sorted_values: Sequence[float]) -> float:
    count = len(sorted_values)
    middle = count // 2
    if count % 2 == 1:
        return sorted_values[middle]
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2.0


@dataclass(frozen=True, slots=True)
class CorpusClip:
    """One prior clip's eligible measurements, with the identity to explain it."""

    result_id: str
    created_at: str
    values: Mapping[str, float]


def absent_comparative(reason: str) -> dict[str, Any]:
    return {"status": "absent", "reason": reason}


def build_comparative(
    subject_values: Mapping[str, float],
    corpus: Sequence[CorpusClip],
    *,
    minimum_clips: int = DEFAULT_MINIMUM_CLIPS,
    window: int = DEFAULT_WINDOW,
) -> dict[str, Any]:
    """Rank this clip's measurements among the creator's own recent clips."""

    if minimum_clips < 2:
        raise ValueError("a comparison needs at least two clips to be meaningful")
    recent = list(corpus)[:window]
    population = len(recent) + 1
    if population < minimum_clips:
        return absent_comparative(
            f"a comparison needs at least {minimum_clips} analysed clips; this "
            f"machine has {population}"
        )

    timestamps = sorted(clip.created_at for clip in recent if clip.created_at)
    metrics: list[dict[str, Any]] = []
    for path in COMPARATIVE_METRIC_PATHS:
        subject = _finite(subject_values.get(path))
        if subject is None:
            continue
        others = [
            value
            for value in (clip.values.get(path) for clip in recent)
            if _finite(value) is not None
        ]
        # A metric missing from any counted clip is excluded, never imputed;
        # `outOf` therefore always states the population it was ranked against.
        if len(others) + 1 < minimum_clips:
            continue
        ranked = sorted([*others, subject])
        out_of = len(ranked)
        rank = sum(1 for value in ranked if value < subject) + 1
        metrics.append(
            {
                "metricPath": path,
                "value": subject,
                "rank": rank,
                "outOf": out_of,
                "percentile": round(100.0 * (rank - 1) / (out_of - 1), 4),
                "corpusMinimum": ranked[0],
                "corpusMedian": _median(ranked),
                "corpusMaximum": ranked[-1],
            }
        )

    if not metrics:
        return absent_comparative(
            "no eligible metric is measured on enough of this machine's analysed clips"
        )
    return {
        "status": "present",
        "schemaVersion": COMPARATIVE_SCHEMA_VERSION,
        "rankedBy": "value-ascending",
        "corpus": {
            "kind": "local-forecast-results",
            "clipCount": population,
            "oldestCreatedAt": timestamps[0] if timestamps else None,
            "newestCreatedAt": timestamps[-1] if timestamps else None,
        },
        "metrics": metrics,
    }


def corpus_from_results(
    results: Iterable[Mapping[str, Any]],
    *,
    exclude_result_id: str,
    assemble: Any,
) -> list[CorpusClip]:
    """Turn completed results into corpus clips, newest first, subject removed."""

    clips: list[CorpusClip] = []
    for result in results:
        result_id = result.get("resultId")
        if not isinstance(result_id, str) or result_id == exclude_result_id:
            continue
        try:
            bundle = assemble(result)
        except Exception:
            # One unreadable prior result never breaks the current job.
            continue
        values = extract_metric_values(bundle)
        if not values:
            continue
        created_at = result.get("createdAt")
        clips.append(
            CorpusClip(
                result_id=result_id,
                created_at=created_at if isinstance(created_at, str) else "",
                values=values,
            )
        )
    clips.sort(key=lambda clip: clip.created_at, reverse=True)
    return clips
