"""An evaluation harness for a single-creator calibration candidate.

This module trains and honestly evaluates a candidate. It **approves nothing**.

The production gate lives in ``backend/forecast/calibrated_heads.py``, where
``APPROVED_TARGET_CONTRACTS`` is empty and an approval is a reviewed code change
rather than a data file. Nothing here writes to that table, and a test asserts
it stays empty. What this produces is an evaluation manifest: the numbers a
reviewer would need before considering an approval, plus an explicit list of
what is still missing.

The scope is deliberately narrow. A general behavioral head over creator
outcomes may never be justifiable on obtainable data. One creator, one platform,
one declared target, evaluated chronologically on their own clips, is a target a
small tool can actually reach — and it is still only a candidate until someone
reads these numbers and decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .comparative import COMPARATIVE_METRIC_PATHS, extract_metric_values


CANDIDATE_SCHEMA_VERSION = "insight-calibration-candidate/1"

# Below this there is nothing to evaluate: a chronological holdout of a handful
# of clips measures noise, and a calibration curve over it is decoration.
MINIMUM_TOTAL_CLIPS = 30
MINIMUM_HOLDOUT_CLIPS = 10
HOLDOUT_FRACTION = 0.3
CALIBRATION_BINS = 10
MAXIMUM_ITERATIONS = 400
LEARNING_RATE = 0.1
L2_PENALTY = 1e-3


class CalibrationCandidateError(ValueError):
    """The candidate cannot be evaluated; nothing is produced."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail

    def as_dict(self) -> dict[str, Any]:
        return {"unavailable": True, "reasonCode": self.reason_code, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class TrainingRow:
    result_id: str
    posted_at: str
    features: Mapping[str, float]
    label: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_training_table(
    outcome_set: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
    *,
    metric: str,
    assemble: Any,
    threshold: float | None = None,
) -> list[TrainingRow]:
    """Join creator-declared outcomes to measured evidence, one row per clip.

    A rate metric is used directly as the label. A count metric needs a declared
    threshold, because "a lot of views" is not a target definition.
    """

    records = outcome_set.get("records")
    if not isinstance(records, Sequence) or not records:
        raise CalibrationCandidateError("empty_outcomes", "the outcome set carries no records")

    rows: list[TrainingRow] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping) or record.get("metric") != metric:
            continue
        result_id = record.get("resultId")
        if not isinstance(result_id, str) or result_id in seen:
            continue
        result = results.get(result_id)
        if result is None:
            continue
        raw_value = record.get("value")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            continue
        value = float(raw_value)
        kind = record.get("metricKind")
        if kind == "rate":
            if not 0.0 <= value <= 1.0:
                continue
            label = value
        else:
            if threshold is None:
                raise CalibrationCandidateError(
                    "undeclared_threshold",
                    f"{metric} is a {kind}; a target needs a declared threshold to become a label",
                )
            label = 1.0 if value >= threshold else 0.0
        features = extract_metric_values(assemble(result))
        if not features:
            continue
        posted_at = record.get("postedAt")
        seen.add(result_id)
        rows.append(
            TrainingRow(
                result_id=result_id,
                posted_at=posted_at if isinstance(posted_at, str) else "",
                features=features,
                label=label,
            )
        )

    if len(rows) < MINIMUM_TOTAL_CLIPS:
        raise CalibrationCandidateError(
            "insufficient_data",
            f"a candidate needs at least {MINIMUM_TOTAL_CLIPS} clips with both an outcome "
            f"and measured evidence; this creator has {len(rows)}",
        )
    # Chronological, never shuffled: a creator's later clips must never train a
    # model that is then evaluated on their earlier ones.
    rows.sort(key=lambda row: row.posted_at)
    return rows


def _matrix(rows: Sequence[TrainingRow], paths: Sequence[str]) -> np.ndarray:
    return np.array(
        [[float(row.features.get(path, math.nan)) for path in paths] for row in rows],
        dtype=np.float64,
    )


def _usable_paths(rows: Sequence[TrainingRow]) -> list[str]:
    return [
        path
        for path in COMPARATIVE_METRIC_PATHS
        if all(path in row.features and math.isfinite(row.features[path]) for row in rows)
    ]


def _fit_logistic(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Plain gradient descent. Small, deterministic, and dependency-free."""

    samples, dimensions = features.shape
    weights = np.zeros(dimensions, dtype=np.float64)
    bias = 0.0
    for _ in range(MAXIMUM_ITERATIONS):
        predictions = 1.0 / (1.0 + np.exp(-(features @ weights + bias)))
        error = predictions - labels
        weights -= LEARNING_RATE * ((features.T @ error) / samples + L2_PENALTY * weights)
        bias -= LEARNING_RATE * float(np.mean(error))
    return weights, bias


def _predict(features: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(features @ weights + bias)))


def _expected_calibration_error(
    predictions: np.ndarray, labels: np.ndarray, bins: int = CALIBRATION_BINS
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (predictions >= lower) & (
            predictions < upper if index < bins - 1 else predictions <= upper
        )
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        total += (count / predictions.size) * abs(
            float(np.mean(predictions[mask])) - float(np.mean(labels[mask]))
        )
    return round(total, 6)


def _calibration_line(predictions: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if float(np.std(predictions)) < 1e-12:
        return 0.0, float(np.mean(labels))
    slope, intercept = np.polyfit(predictions, labels, 1)
    return round(float(slope), 6), round(float(intercept), 6)


def evaluate_candidate(
    rows: Sequence[TrainingRow],
) -> dict[str, Any]:
    """Fit on the earlier clips, score the later ones, and report honestly."""

    paths = _usable_paths(rows)
    if not paths:
        raise CalibrationCandidateError(
            "no_common_features",
            "no measured signal is present on every clip, so there is nothing to fit",
        )
    holdout_size = max(MINIMUM_HOLDOUT_CLIPS, int(round(len(rows) * HOLDOUT_FRACTION)))
    if len(rows) - holdout_size < MINIMUM_HOLDOUT_CLIPS:
        raise CalibrationCandidateError(
            "insufficient_data",
            "there are not enough clips to hold out a chronological test set",
        )
    train_rows, test_rows = rows[:-holdout_size], rows[-holdout_size:]

    train_features = _matrix(train_rows, paths)
    test_features = _matrix(test_rows, paths)
    # Near-duplicate guard: identical feature vectors across the split leak.
    train_keys = {tuple(np.round(row, 9)) for row in train_features}
    leaked = sum(1 for row in test_features if tuple(np.round(row, 9)) in train_keys)
    if leaked:
        raise CalibrationCandidateError(
            "near_duplicate_leakage",
            f"{leaked} test clip(s) have feature vectors identical to a training clip",
        )

    mean = train_features.mean(axis=0)
    deviation = train_features.std(axis=0)
    deviation[deviation < 1e-12] = 1.0
    train_scaled = (train_features - mean) / deviation
    test_scaled = (test_features - mean) / deviation

    train_labels = np.array([row.label for row in train_rows], dtype=np.float64)
    test_labels = np.array([row.label for row in test_rows], dtype=np.float64)
    if float(np.std(train_labels)) < 1e-12:
        raise CalibrationCandidateError(
            "degenerate_labels",
            "every training clip carries the same outcome, so nothing can be learned",
        )

    weights, bias = _fit_logistic(train_scaled, train_labels)
    predictions = np.clip(_predict(test_scaled, weights, bias), 1e-9, 1 - 1e-9)

    brier = float(np.mean((predictions - test_labels) ** 2))
    log_loss = float(
        -np.mean(test_labels * np.log(predictions) + (1 - test_labels) * np.log(1 - predictions))
    )
    slope, intercept = _calibration_line(predictions, test_labels)
    baseline = float(np.mean(train_labels))
    baseline_brier = float(np.mean((baseline - test_labels) ** 2))

    return {
        "featurePaths": list(paths),
        "trainClipCount": len(train_rows),
        "testClipCount": len(test_rows),
        "splitKind": "chronological-holdout",
        "brierScore": round(brier, 6),
        "logLoss": round(log_loss, 6),
        "expectedCalibrationError": _expected_calibration_error(predictions, test_labels),
        "calibrationSlope": slope,
        "calibrationIntercept": intercept,
        "baseRate": round(baseline, 6),
        "baselineBrierScore": round(baseline_brier, 6),
        # The number that matters most: does it beat predicting the base rate?
        "beatsBaseRate": bool(brier < baseline_brier),
        "trainStartedAt": train_rows[0].posted_at,
        "testStartedAt": test_rows[0].posted_at,
    }


def build_candidate_manifest(
    evaluation: Mapping[str, Any],
    *,
    metric: str,
    platform: str,
    creator_scope: str = "single-creator-local",
    threshold: float | None = None,
) -> dict[str, Any]:
    """Wrap an evaluation with what it is, and with what it is still missing."""

    blockers = [
        "No approval exists: APPROVED_TARGET_CONTRACTS in "
        "backend/forecast/calibrated_heads.py is empty, and an approval is a reviewed "
        "code change, not a data file.",
        "The target event, denominator, horizon, locale, and population are not pinned "
        "to an immutable definition digest.",
        "The labels are creator-declared and unverified; SignalFrame did not observe them.",
        "Evaluation is chronological on one creator only, so nothing here generalises "
        "to another account, platform, or audience.",
        "No prospective shadow validation has been run.",
        "No drift monitoring, freshness window, or out-of-distribution gate is defined.",
    ]
    if not evaluation.get("beatsBaseRate"):
        blockers.insert(
            0,
            "This candidate does not beat predicting the creator's own base rate, so it "
            "carries no information a constant would not.",
        )
    payload = {
        "schemaVersion": CANDIDATE_SCHEMA_VERSION,
        "createdAt": _utc_now(),
        "scope": creator_scope,
        "metric": metric,
        "platform": platform,
        "countThreshold": threshold,
        "evaluation": dict(evaluation),
        # The whole point of this document.
        "approved": False,
        "scoringAvailable": False,
        "behavioralOutcome": False,
        "blockers": blockers,
        "limits": (
            "This is an evaluation of a candidate, not a head. Nothing in this document "
            "makes a probability available anywhere in the product, and no value from it "
            "is displayed to a creator. It exists so a reviewer can see whether a "
            "single-creator calibration is worth pursuing at all."
        ),
    }
    payload["manifestSha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "createdAt"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload


def evaluate_personal_candidate(
    outcome_set: Mapping[str, Any],
    results: Mapping[str, Mapping[str, Any]],
    *,
    metric: str,
    platform: str,
    assemble: Any,
    threshold: float | None = None,
) -> dict[str, Any]:
    """The whole harness: join, split chronologically, fit, evaluate, refuse to approve."""

    rows = build_training_table(
        outcome_set, results, metric=metric, assemble=assemble, threshold=threshold
    )
    evaluation = evaluate_candidate(rows)
    return build_candidate_manifest(
        evaluation, metric=metric, platform=platform, threshold=threshold
    )
