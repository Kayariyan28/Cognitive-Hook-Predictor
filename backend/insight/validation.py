"""Deterministic acceptance rules for one `insight.v1` artifact.

Nothing here trusts the prompt.  A model's JSON is accepted whole or rejected
whole: there is no partial acceptance, no field dropping, and no repair of
malformed output.  Every rejection names a `reasonCode` from the schema
reference so an operator can read what happened without guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from .bundle import HOOK_WINDOW_SECONDS
from .citations import (
    Citation,
    CitationMalformedError,
    CitationUnresolvableError,
    parse_citation,
    resolve_citation,
)
from .claim_terms import sentence_violations, split_sentences


INSIGHT_SCHEMA_VERSION = "insight/2"

MODEL_SETTABLE_KEYS = ("hookReport", "hookRewrites", "phaseCommentary", "tribeNotes")
SERVER_OWNED_KEYS = frozenset(
    {"schemaVersion", "insightId", "generatedAt", "behavioralOutcome", "limits", "provenance"}
)

REASON_CODES = frozenset(
    {
        "bundle_unavailable",
        "provider_unavailable",
        "provider_error",
        "output_not_json",
        "output_too_large",
        "schema_invalid",
        "unknown_field",
        "server_owned_field",
        "missing_citation",
        "citation_malformed",
        "citation_unresolvable",
        "numeric_not_in_evidence",
        "claim_boundary_violation",
    }
)

EFFORT_VALUES = frozenset({"low", "medium", "high"})
DIRECTION_VALUES = frozenset({"increase", "decrease", "unchanged"})
PHASE_VALUES = frozenset({"early", "middle", "late"})
HYPOTHESIS_LABEL = "untested heuristic"

MAXIMUM_TEXT_CHARACTERS = 400
MAXIMUM_REWRITE_CHARACTERS = 160
REWRITE_BASES = frozenset({"spoken", "on-screen"})
MAXIMUM_CITATIONS_PER_ITEM = 6
ARRAY_BOUNDS = {
    "whatTheHookContains": (1, 8),
    "observations": (0, 8),
    "hypotheses": (0, 6),
    "experiments": (0, 6),
    "expectedSignalShift": (1, 4),
    "phaseCommentary": (0, 3),
    "tribeNotes": (0, 4),
    "hookRewrites": (0, 4),
}

ID_RE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
CONTROL_CHARACTER_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
NUMERAL_RE = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)(%?)")
MAXIMUM_REACHABLE_DEPTH = 6


class InsightRejection(Exception):
    """A fail-closed rejection carrying the reasonCode an operator will read."""

    def __init__(self, reason_code: str, detail: Any) -> None:
        if reason_code not in REASON_CODES:
            raise ValueError(f"unknown insight reasonCode {reason_code!r}")
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.detail = detail

    def as_dict(self) -> dict[str, Any]:
        return {"status": "rejected", "reasonCode": self.reason_code, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class Item:
    """One model-authored unit of text, with the path a rejection will name."""

    path: str
    texts: tuple[tuple[str, str], ...]
    citations: tuple[Citation, ...]
    tribe_scoped: bool
    # A proposed replacement line is a suggestion, not an assertion about the
    # clip, so a numeral inside it is not a claim the evidence must contain.
    # Every claim-boundary rule still applies to it in full.
    numeric_exempt: bool = False


# --------------------------------------------------------------------------
# Structural validation
# --------------------------------------------------------------------------


def _reject(reason_code: str, detail: Any) -> "InsightRejection":
    return InsightRejection(reason_code, detail)


def _require_object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _reject("schema_invalid", f"{path} must be an object")
    return value


def _closed_keys(value: Mapping[str, Any], allowed: Iterable[str], path: str) -> None:
    for key in value:
        if key in SERVER_OWNED_KEYS and path != "":
            raise _reject("server_owned_field", f"{path}/{key} is owned by the server")
        if key not in allowed:
            raise _reject("unknown_field", f"{path}/{key} is not defined by insight.v1")


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise _reject("schema_invalid", f"{path} must be a string")
    stripped = value.strip()
    if not stripped:
        raise _reject("schema_invalid", f"{path} must not be empty")
    if len(stripped) > MAXIMUM_TEXT_CHARACTERS:
        raise _reject(
            "schema_invalid",
            f"{path} exceeds the {MAXIMUM_TEXT_CHARACTERS}-character limit",
        )
    if CONTROL_CHARACTER_RE.search(value):
        raise _reject("schema_invalid", f"{path} contains control characters")
    return stripped


def _bounded_array(value: Any, name: str, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise _reject("schema_invalid", f"{path} must be an array")
    minimum, maximum = ARRAY_BOUNDS[name]
    if not minimum <= len(value) <= maximum:
        raise _reject(
            "schema_invalid",
            f"{path} must hold from {minimum} to {maximum} items, not {len(value)}",
        )
    return value


def _citations(value: Any, path: str) -> tuple[Citation, ...]:
    if not isinstance(value, list):
        raise _reject("schema_invalid", f"{path}/citations must be an array")
    if not value:
        raise _reject("missing_citation", f"{path} carries no citation")
    if len(value) > MAXIMUM_CITATIONS_PER_ITEM:
        raise _reject(
            "schema_invalid",
            f"{path}/citations exceeds {MAXIMUM_CITATIONS_PER_ITEM} citations",
        )
    parsed: list[Citation] = []
    for index, raw in enumerate(value):
        try:
            parsed.append(parse_citation(raw))
        except CitationMalformedError as exc:
            raise _reject("citation_malformed", f"{path}/citations/{index}: {exc}") from exc
    return tuple(parsed)


def _is_tribe_scoped(citations: Iterable[Citation]) -> bool:
    return any(citation.lane == "tribe" for citation in citations)


def _simple_item(value: Any, path: str) -> Item:
    record = _require_object(value, path)
    _closed_keys(record, ("text", "citations"), path)
    if "text" not in record:
        raise _reject("schema_invalid", f"{path}/text is required")
    if "citations" not in record:
        raise _reject("missing_citation", f"{path} carries no citation")
    citations = _citations(record["citations"], path)
    return Item(
        path=path,
        texts=((f"{path}/text", _text(record["text"], f"{path}/text")),),
        citations=citations,
        tribe_scoped=_is_tribe_scoped(citations),
    )


def _hypothesis_item(value: Any, path: str) -> tuple[Item, str]:
    record = _require_object(value, path)
    _closed_keys(record, ("id", "text", "label", "citations"), path)
    for key in ("id", "text", "label", "citations"):
        if key not in record:
            if key == "citations":
                raise _reject("missing_citation", f"{path} carries no citation")
            raise _reject("schema_invalid", f"{path}/{key} is required")
    identifier = record["id"]
    if not isinstance(identifier, str) or not ID_RE.fullmatch(identifier):
        raise _reject("schema_invalid", f"{path}/id must match {ID_RE.pattern}")
    if record["label"] != HYPOTHESIS_LABEL:
        raise _reject("schema_invalid", f"{path}/label must be {HYPOTHESIS_LABEL!r}")
    citations = _citations(record["citations"], path)
    item = Item(
        path=path,
        texts=((f"{path}/text", _text(record["text"], f"{path}/text")),),
        citations=citations,
        tribe_scoped=_is_tribe_scoped(citations),
    )
    return item, identifier


def _expected_signal_shift(value: Any, path: str) -> list[Citation]:
    entries = _bounded_array(value, "expectedSignalShift", path)
    metric_citations: list[Citation] = []
    for index, entry in enumerate(entries):
        entry_path = f"{path}/{index}"
        record = _require_object(entry, entry_path)
        _closed_keys(record, ("metricPath", "direction"), entry_path)
        for key in ("metricPath", "direction"):
            if key not in record:
                raise _reject("schema_invalid", f"{entry_path}/{key} is required")
        if record["direction"] not in DIRECTION_VALUES:
            raise _reject(
                "schema_invalid",
                f"{entry_path}/direction must be one of {sorted(DIRECTION_VALUES)}",
            )
        try:
            citation = parse_citation(record["metricPath"])
        except CitationMalformedError as exc:
            raise _reject("citation_malformed", f"{entry_path}/metricPath: {exc}") from exc
        if citation.window is not None:
            raise _reject(
                "schema_invalid", f"{entry_path}/metricPath must not assert a window"
            )
        metric_citations.append(citation)
    return metric_citations


def _experiment_item(value: Any, path: str) -> tuple[Item, str, str, list[Citation]]:
    record = _require_object(value, path)
    allowed = ("id", "hypothesisId", "edit", "effort", "expectedSignalShift", "citations")
    _closed_keys(record, allowed, path)
    for key in allowed:
        if key not in record:
            if key == "citations":
                raise _reject("missing_citation", f"{path} carries no citation")
            raise _reject("schema_invalid", f"{path}/{key} is required")
    identifier = record["id"]
    hypothesis_id = record["hypothesisId"]
    for key, candidate in (("id", identifier), ("hypothesisId", hypothesis_id)):
        if not isinstance(candidate, str) or not ID_RE.fullmatch(candidate):
            raise _reject("schema_invalid", f"{path}/{key} must match {ID_RE.pattern}")
    if record["effort"] not in EFFORT_VALUES:
        raise _reject("schema_invalid", f"{path}/effort must be one of {sorted(EFFORT_VALUES)}")
    metric_citations = _expected_signal_shift(
        record["expectedSignalShift"], f"{path}/expectedSignalShift"
    )
    citations = _citations(record["citations"], path)
    item = Item(
        path=path,
        texts=((f"{path}/edit", _text(record["edit"], f"{path}/edit")),),
        citations=citations,
        tribe_scoped=_is_tribe_scoped(citations),
    )
    return item, identifier, hypothesis_id, metric_citations


def _rewrite_item(value: Any, path: str) -> Item:
    record = _require_object(value, path)
    _closed_keys(record, ("line", "basis", "citations"), path)
    for key in ("line", "basis", "citations"):
        if key not in record:
            if key == "citations":
                raise _reject("missing_citation", f"{path} carries no citation")
            raise _reject("schema_invalid", f"{path}/{key} is required")
    if record["basis"] not in REWRITE_BASES:
        raise _reject(
            "schema_invalid", f"{path}/basis must be one of {sorted(REWRITE_BASES)}"
        )
    line = record["line"]
    if not isinstance(line, str) or not line.strip():
        raise _reject("schema_invalid", f"{path}/line must be a non-empty string")
    if len(line.strip()) > MAXIMUM_REWRITE_CHARACTERS:
        raise _reject(
            "schema_invalid",
            f"{path}/line exceeds the {MAXIMUM_REWRITE_CHARACTERS}-character limit",
        )
    if CONTROL_CHARACTER_RE.search(line):
        raise _reject("schema_invalid", f"{path}/line contains control characters")
    citations = _citations(record["citations"], path)
    return Item(
        path=path,
        texts=((f"{path}/line", line.strip()),),
        citations=citations,
        tribe_scoped=_is_tribe_scoped(citations),
        numeric_exempt=True,
    )


def _phase_item(value: Any, path: str) -> Item:
    record = _require_object(value, path)
    _closed_keys(record, ("phase", "text", "citations"), path)
    for key in ("phase", "text", "citations"):
        if key not in record:
            if key == "citations":
                raise _reject("missing_citation", f"{path} carries no citation")
            raise _reject("schema_invalid", f"{path}/{key} is required")
    if record["phase"] not in PHASE_VALUES:
        raise _reject("schema_invalid", f"{path}/phase must be one of {sorted(PHASE_VALUES)}")
    citations = _citations(record["citations"], path)
    return Item(
        path=path,
        texts=((f"{path}/text", _text(record["text"], f"{path}/text")),),
        citations=citations,
        tribe_scoped=_is_tribe_scoped(citations),
    )


def _window_seconds(value: Any, expected: tuple[float, float]) -> list[float]:
    path = "/hookReport/windowSeconds"
    if not isinstance(value, list) or len(value) != 2:
        raise _reject("schema_invalid", f"{path} must be a two-element array")
    bounds: list[float] = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)) or not math.isfinite(entry):
            raise _reject("schema_invalid", f"{path} must hold finite numbers")
        bounds.append(float(entry))
    if bounds[0] >= bounds[1]:
        raise _reject("schema_invalid", f"{path} must be increasing")
    if abs(bounds[0] - expected[0]) > 1e-9 or abs(bounds[1] - expected[1]) > 1e-9:
        raise _reject(
            "schema_invalid",
            f"{path} must equal the window this bundle was assembled for: {list(expected)}",
        )
    return bounds


def _hook_report(value: Any, expected_window: tuple[float, float]) -> list[Item]:
    path = "/hookReport"
    record = _require_object(value, path)
    allowed = (
        "windowSeconds",
        "whatTheHookContains",
        "observations",
        "hypotheses",
        "experiments",
    )
    _closed_keys(record, allowed, path)
    for key in allowed:
        if key not in record:
            raise _reject("schema_invalid", f"{path}/{key} is required")
    _window_seconds(record["windowSeconds"], expected_window)

    items: list[Item] = []
    for name in ("whatTheHookContains", "observations"):
        entries = _bounded_array(record[name], name, f"{path}/{name}")
        for index, entry in enumerate(entries):
            items.append(_simple_item(entry, f"{path}/{name}/{index}"))

    hypothesis_ids: list[str] = []
    for index, entry in enumerate(
        _bounded_array(record["hypotheses"], "hypotheses", f"{path}/hypotheses")
    ):
        item, identifier = _hypothesis_item(entry, f"{path}/hypotheses/{index}")
        if identifier in hypothesis_ids:
            raise _reject("schema_invalid", f"{path}/hypotheses/{index}/id is a duplicate")
        hypothesis_ids.append(identifier)
        items.append(item)

    experiment_ids: list[str] = []
    for index, entry in enumerate(
        _bounded_array(record["experiments"], "experiments", f"{path}/experiments")
    ):
        experiment_path = f"{path}/experiments/{index}"
        item, identifier, hypothesis_id, metric_citations = _experiment_item(
            entry, experiment_path
        )
        if identifier in experiment_ids:
            raise _reject("schema_invalid", f"{experiment_path}/id is a duplicate")
        experiment_ids.append(identifier)
        if hypothesis_id not in hypothesis_ids:
            raise _reject(
                "schema_invalid",
                f"{experiment_path}/hypothesisId does not name a hypothesis in this artifact",
            )
        items.append(item)
        items.append(
            Item(
                path=f"{experiment_path}/expectedSignalShift",
                texts=(),
                citations=tuple(metric_citations),
                tribe_scoped=False,
            )
        )
    return items


def _structural_items(payload: Mapping[str, Any], expected_window: tuple[float, float]) -> list[Item]:
    for key in payload:
        if key in SERVER_OWNED_KEYS:
            raise _reject("server_owned_field", f"/{key} is owned by the server")
        if key not in MODEL_SETTABLE_KEYS:
            raise _reject("unknown_field", f"/{key} is not defined by insight.v1")
    if "hookReport" not in payload:
        raise _reject("schema_invalid", "/hookReport is required")

    items = _hook_report(payload["hookReport"], expected_window)

    for index, entry in enumerate(
        _bounded_array(payload.get("hookRewrites", []), "hookRewrites", "/hookRewrites")
    ):
        items.append(_rewrite_item(entry, f"/hookRewrites/{index}"))

    for index, entry in enumerate(
        _bounded_array(payload.get("phaseCommentary", []), "phaseCommentary", "/phaseCommentary")
    ):
        items.append(_phase_item(entry, f"/phaseCommentary/{index}"))

    for index, entry in enumerate(
        _bounded_array(payload.get("tribeNotes", []), "tribeNotes", "/tribeNotes")
    ):
        item = _simple_item(entry, f"/tribeNotes/{index}")
        if not item.tribe_scoped:
            raise _reject(
                "schema_invalid",
                f"/tribeNotes/{index} must carry at least one tribe: citation",
            )
        items.append(item)
    return items


# --------------------------------------------------------------------------
# Citation resolution and the numeric-copy rule
# --------------------------------------------------------------------------


def _reachable_numbers(value: Any, depth: int = MAXIMUM_REACHABLE_DEPTH) -> list[float]:
    if depth < 0:
        return []
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)] if math.isfinite(float(value)) else []
    if isinstance(value, Mapping):
        numbers: list[float] = []
        for entry in value.values():
            numbers.extend(_reachable_numbers(entry, depth - 1))
        return numbers
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        numbers = []
        for entry in value:
            numbers.extend(_reachable_numbers(entry, depth - 1))
        return numbers
    return []


def round_to_significant(value: float, digits: int) -> float:
    """Round to `digits` significant figures; the validator's only tolerance."""

    if value == 0 or not math.isfinite(value):
        return 0.0
    exponent = math.floor(math.log10(abs(value)))
    factor = 10.0 ** (digits - 1 - exponent)
    return round(value * factor) / factor


def _matches_value(numeral: float, value: float) -> bool:
    if math.isclose(numeral, value, rel_tol=1e-12, abs_tol=1e-12):
        return True
    for digits in range(2, 13):
        if math.isclose(
            round_to_significant(value, digits), numeral, rel_tol=1e-9, abs_tol=1e-9
        ):
            return True
    return False


def _numerals(text: str) -> list[tuple[float, bool]]:
    found: list[tuple[float, bool]] = []
    for match in NUMERAL_RE.finditer(text):
        try:
            found.append((float(match.group(1)), match.group(2) == "%"))
        except ValueError:  # pragma: no cover - the pattern only matches decimals
            continue
    return found


def _numeric_copy_violation(
    item: Item, reachable: Sequence[float], window: tuple[float, float]
) -> tuple[str, float] | None:
    for text_path, text in item.texts:
        for numeral, is_percent in _numerals(text):
            if any(math.isclose(numeral, bound, rel_tol=0, abs_tol=1e-9) for bound in window):
                continue
            candidates = list(reachable)
            if is_percent:
                candidates.extend(value * 100.0 for value in reachable)
            if not any(_matches_value(numeral, value) for value in candidates):
                return text_path, numeral
    return None


# --------------------------------------------------------------------------
# Claim-boundary lint
# --------------------------------------------------------------------------


def claim_boundary_violations(items: Iterable[Item]) -> list[dict[str, Any]]:
    """Report every unlimited forbidden claim, in document order."""

    violations: list[dict[str, Any]] = []
    for item in items:
        for text_path, text in item.texts:
            for sentence in split_sentences(text):
                for term in sentence_violations(sentence, tribe_scoped=item.tribe_scoped):
                    violations.append(
                        {
                            "reasonCode": "claim_boundary_violation",
                            "term": term,
                            "sentence": sentence,
                            "itemPath": text_path,
                        }
                    )
    return violations


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def expected_window(bundle: Mapping[str, Any]) -> tuple[float, float]:
    source = bundle.get("source") if isinstance(bundle, Mapping) else None
    window = source.get("window") if isinstance(source, Mapping) else None
    if isinstance(window, (list, tuple)) and len(window) == 2:
        return (float(window[0]), float(window[1]))
    return HOOK_WINDOW_SECONDS


def validate_insight(raw_json: Any, bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Accept one artifact whole, or reject it whole with a reasonCode."""

    try:
        payload = _parse(raw_json)
        window = expected_window(bundle)
        items = _structural_items(payload, window)

        for item in items:
            resolved: list[Any] = []
            for citation in item.citations:
                try:
                    resolved.append(resolve_citation(bundle, citation))
                except CitationUnresolvableError as exc:
                    raise _reject(
                        "citation_unresolvable", f"{item.path}: {exc}"
                    ) from exc
            if item.path.endswith("/expectedSignalShift"):
                for index, value in enumerate(resolved):
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise _reject(
                            "schema_invalid",
                            f"{item.path}/{index}/metricPath must point at a number",
                        )
                continue
            if item.numeric_exempt:
                continue
            reachable = [
                number for value in resolved for number in _reachable_numbers(value)
            ]
            violation = _numeric_copy_violation(item, reachable, window)
            if violation is not None:
                text_path, numeral = violation
                raise _reject(
                    "numeric_not_in_evidence",
                    f"{text_path} states {numeral:g}, which is not a copy of its cited evidence",
                )

        violations = claim_boundary_violations(items)
        if violations:
            raise _reject("claim_boundary_violation", violations[0])
    except InsightRejection as rejection:
        return rejection.as_dict()

    artifact = {
        "hookReport": payload["hookReport"],
        "hookRewrites": payload.get("hookRewrites", []),
        "phaseCommentary": payload.get("phaseCommentary", []),
        "tribeNotes": payload.get("tribeNotes", []),
    }
    return {"status": "valid", "artifact": artifact}


def _parse(raw_json: Any) -> Mapping[str, Any]:
    if isinstance(raw_json, Mapping):
        return raw_json
    if not isinstance(raw_json, str):
        raise _reject("output_not_json", "the model returned no JSON text")
    try:
        payload = json.loads(raw_json)
    except (ValueError, json.JSONDecodeError) as exc:
        raise _reject("output_not_json", f"the model output is not strict JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise _reject("output_not_json", "the model output is not a JSON object")
    return payload
