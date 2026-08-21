"""Creator-declared post-publish outcomes, stored locally and read by nothing.

These are **labels for a future, separately trained calibration head**. They are
not evidence, not a forecast, and not something any part of the insight lane may
read. That separation is structural rather than disciplinary:

* this module is a leaf — the assembler, validator, prompt template, and service
  do not import it, and a test walks their import graphs to prove it;
* there is no ``outcomes:`` lane, so no citation can name one and
  ``parse_citation`` rejects the attempt before resolution;
* nothing here is ever placed in a bundle, so nothing here can reach a provider.

Every value is self-reported by the creator from a platform's own analytics.
SignalFrame did not measure it and cannot verify it, and the schema records that
as a field rather than a footnote.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import math
import re
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4


OUTCOME_SET_SCHEMA_VERSION = "creator-outcome-set/1"

MAXIMUM_ROWS = 5_000
MAXIMUM_CSV_BYTES = 2 * 1024 * 1024
REQUIRED_COLUMNS = ("resultId", "platform", "postedAt", "measuredAt", "metric", "value")
OPTIONAL_COLUMNS = ("denominator", "note")
MAXIMUM_NOTE_CHARACTERS = 280

# Code-owned. A metric is here because its meaning is unambiguous enough to be a
# training label later; `kind` says how the value must be shaped.
OUTCOME_METRICS: Mapping[str, str] = {
    "views": "count",
    "plays": "count",
    "likes": "count",
    "comments": "count",
    "shares": "count",
    "saves": "count",
    "follows": "count",
    "retention_5s": "rate",
    "completion_rate": "rate",
    "replay_rate": "rate",
    "average_watch_seconds": "duration",
}

PLATFORM_RE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
RESULT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
DENOMINATOR_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class OutcomeImportError(ValueError):
    """The import is refused whole; a partly imported set biases what it trains."""

    def __init__(self, reason_code: str, detail: str, *, row: int | None = None) -> None:
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail
        self.row = row

    def as_dict(self) -> dict[str, Any]:
        return {
            "unavailable": True,
            "reasonCode": self.reason_code,
            "detail": self.detail,
            "row": self.row,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp(value: str, label: str, row: int) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise OutcomeImportError(
            "invalid_timestamp", f"{label} is not an ISO-8601 timestamp", row=row
        ) from exc
    if parsed.tzinfo is None:
        raise OutcomeImportError(
            "invalid_timestamp", f"{label} has no timezone", row=row
        )
    return parsed.astimezone(timezone.utc)


def _value_for(metric: str, raw: str, row: int) -> float:
    try:
        number = float(raw.strip())
    except (TypeError, ValueError) as exc:
        raise OutcomeImportError(
            "invalid_value", f"value {raw!r} is not a number", row=row
        ) from exc
    if not math.isfinite(number):
        raise OutcomeImportError("invalid_value", "value is not finite", row=row)
    kind = OUTCOME_METRICS[metric]
    if kind == "rate" and not 0.0 <= number <= 1.0:
        raise OutcomeImportError(
            "invalid_value", f"{metric} is a rate and must be within 0 to 1", row=row
        )
    if kind in {"count", "duration"} and number < 0:
        raise OutcomeImportError(
            "invalid_value", f"{metric} cannot be negative", row=row
        )
    if kind == "count" and number != int(number):
        raise OutcomeImportError(
            "invalid_value", f"{metric} is a count and must be a whole number", row=row
        )
    return number


@dataclass(frozen=True, slots=True)
class OutcomeRecord:
    result_id: str
    platform: str
    posted_at: str
    measured_at: str
    metric: str
    value: float
    denominator: str | None
    note: str | None

    def public_value(self) -> dict[str, Any]:
        return {
            "resultId": self.result_id,
            "platform": self.platform,
            "postedAt": self.posted_at,
            "measuredAt": self.measured_at,
            "metric": self.metric,
            "metricKind": OUTCOME_METRICS[self.metric],
            "value": self.value,
            "denominator": self.denominator,
            "note": self.note,
        }


def parse_outcome_csv(
    text: str, *, result_exists: Callable[[str], bool]
) -> list[OutcomeRecord]:
    """Validate a whole CSV or refuse it. There is no partial import, by design."""

    encoded = text.encode("utf-8")
    if len(encoded) > MAXIMUM_CSV_BYTES:
        raise OutcomeImportError(
            "file_too_large",
            f"the CSV exceeds the {MAXIMUM_CSV_BYTES}-byte import limit",
        )
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise OutcomeImportError("invalid_header", "the CSV has no header row")
    columns = [name.strip() for name in reader.fieldnames]
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise OutcomeImportError(
            "invalid_header", f"the CSV is missing columns: {', '.join(missing)}"
        )
    unknown = [
        name
        for name in columns
        if name and name not in REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    ]
    if unknown:
        raise OutcomeImportError(
            "invalid_header", f"the CSV has unsupported columns: {', '.join(unknown)}"
        )

    records: list[OutcomeRecord] = []
    seen: set[tuple[str, str, str, str]] = set()
    for index, raw_row in enumerate(reader, start=2):
        if len(records) >= MAXIMUM_ROWS:
            raise OutcomeImportError(
                "too_many_rows", f"the CSV exceeds {MAXIMUM_ROWS} rows", row=index
            )
        row = {key.strip(): (value or "").strip() for key, value in raw_row.items() if key}
        if not any(row.get(name) for name in REQUIRED_COLUMNS):
            continue

        result_id = row.get("resultId", "").lower().replace("-", "")
        if not RESULT_ID_RE.fullmatch(result_id):
            raise OutcomeImportError(
                "invalid_result_id", "resultId is not a result identifier", row=index
            )
        if not result_exists(result_id):
            raise OutcomeImportError(
                "unknown_result",
                f"no published forecast result {result_id} exists on this machine",
                row=index,
            )
        platform = row.get("platform", "").lower()
        if not PLATFORM_RE.fullmatch(platform):
            raise OutcomeImportError(
                "invalid_platform", f"platform {platform!r} is not a plain name", row=index
            )
        metric = row.get("metric", "").lower()
        if metric not in OUTCOME_METRICS:
            raise OutcomeImportError(
                "unknown_metric",
                f"metric {metric!r} is not one of {sorted(OUTCOME_METRICS)}",
                row=index,
            )
        posted_at = _timestamp(row.get("postedAt", ""), "postedAt", index)
        measured_at = _timestamp(row.get("measuredAt", ""), "measuredAt", index)
        if measured_at < posted_at:
            raise OutcomeImportError(
                "invalid_timestamp",
                "measuredAt is earlier than postedAt",
                row=index,
            )
        value = _value_for(metric, row.get("value", ""), index)

        denominator = row.get("denominator") or None
        if denominator is not None and not DENOMINATOR_RE.fullmatch(denominator):
            raise OutcomeImportError(
                "invalid_denominator",
                f"denominator {denominator!r} is not a plain identifier",
                row=index,
            )
        if OUTCOME_METRICS[metric] == "rate" and denominator is None:
            raise OutcomeImportError(
                "missing_denominator",
                f"{metric} is a rate and needs its denominator named",
                row=index,
            )
        note = row.get("note") or None
        if note is not None and len(note) > MAXIMUM_NOTE_CHARACTERS:
            raise OutcomeImportError(
                "invalid_note",
                f"note exceeds {MAXIMUM_NOTE_CHARACTERS} characters",
                row=index,
            )

        identity = (result_id, platform, metric, measured_at.isoformat())
        if identity in seen:
            raise OutcomeImportError(
                "duplicate_record",
                "the CSV repeats a result, platform, metric, and measurement time",
                row=index,
            )
        seen.add(identity)
        records.append(
            OutcomeRecord(
                result_id=result_id,
                platform=platform,
                posted_at=posted_at.isoformat().replace("+00:00", "Z"),
                measured_at=measured_at.isoformat().replace("+00:00", "Z"),
                metric=metric,
                value=value,
                denominator=denominator,
                note=note,
            )
        )

    if not records:
        raise OutcomeImportError("empty_import", "the CSV carries no outcome rows")
    return records


def build_outcome_set(records: Iterable[OutcomeRecord], *, source_text: str) -> dict[str, Any]:
    """Wrap validated records with the provenance a future training run needs."""

    listed = list(records)
    return {
        "schemaVersion": OUTCOME_SET_SCHEMA_VERSION,
        "outcomeSetId": uuid4().hex,
        "importedAt": _utc_now(),
        "source": "manual-csv",
        "sourceFileSha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "declaredBy": "creator",
        # SignalFrame did not observe any of this and cannot verify it.
        "verified": False,
        "isTrainingLabelOnly": True,
        "recordCount": len(listed),
        "records": [record.public_value() for record in listed],
        "limits": (
            "These are creator-declared post-publish numbers, self-reported from a "
            "platform's own analytics. SignalFrame did not measure or verify them. They "
            "are stored as labels for a future, separately trained calibration head that "
            "must still pass the code-owned approval contract; no head is trained or "
            "approved by importing them, and nothing here is read by the insight lane."
        ),
    }


class OutcomeLedger:
    """Import, list, and delete outcome sets. It reads no evidence and no bundle."""

    def __init__(
        self,
        store: Any,
        *,
        result_exists: Callable[[str], bool],
    ) -> None:
        self.store = store
        self._result_exists = result_exists

    def import_csv(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise OutcomeImportError("empty_import", "the CSV is empty")
        records = parse_outcome_csv(text, result_exists=self._result_exists)
        document = build_outcome_set(records, source_text=text)
        self.store.publish_outcome_set(document["outcomeSetId"], document)
        return document

    def read(self, outcome_set_id: str) -> dict[str, Any] | None:
        return self.store.read_outcome_set(outcome_set_id)

    def list_sets(self) -> list[dict[str, Any]]:
        """Summaries only; the ledger never hands out every record by default."""

        return [
            {
                key: document[key]
                for key in (
                    "outcomeSetId",
                    "importedAt",
                    "source",
                    "sourceFileSha256",
                    "verified",
                    "recordCount",
                )
                if key in document
            }
            for document in self.store.list_outcome_sets()
        ]

    def delete(self, outcome_set_id: str) -> bool:
        return self.store.delete_outcome_set(outcome_set_id)
