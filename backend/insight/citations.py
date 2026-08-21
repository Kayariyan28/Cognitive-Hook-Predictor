"""The citation grammar that binds every published sentence to real evidence.

A citation names one lane of the evidence bundle, one RFC 6901 pointer inside
that lane, and optionally the time window the cited item must cover.  Parsing
and resolution are separate because they fail differently: a citation that does
not parse is malformed, and a citation that parses but finds nothing is
unresolvable.  Neither is ever silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

from .bundle import LANE_KEYS


CITATION_RE = re.compile(
    r"^(?P<lane>[a-z][a-z0-9]*):"
    r"(?P<pointer>/[^@]*)"
    r"(?:@window\((?P<start>-?\d+(?:\.\d+)?),\s*(?P<end>-?\d+(?:\.\d+)?)\))?$"
)


class CitationMalformedError(ValueError):
    """The citation text does not parse under the grammar."""


class CitationUnresolvableError(ValueError):
    """The citation parses but names nothing in this bundle."""


@dataclass(frozen=True, slots=True)
class Citation:
    lane: str
    pointer: str
    window: tuple[float, float] | None
    text: str


def parse_citation(text: Any) -> Citation:
    """Parse one citation string, rejecting anything the grammar does not allow."""

    if not isinstance(text, str) or not text:
        raise CitationMalformedError("a citation must be a non-empty string")
    match = CITATION_RE.fullmatch(text)
    if match is None:
        raise CitationMalformedError(f"citation {text!r} does not match the citation grammar")
    lane = match.group("lane")
    if lane not in LANE_KEYS:
        raise CitationMalformedError(f"citation {text!r} names an unknown lane {lane!r}")
    window: tuple[float, float] | None = None
    if match.group("start") is not None:
        start = float(match.group("start"))
        end = float(match.group("end"))
        if not math.isfinite(start) or not math.isfinite(end) or start >= end:
            raise CitationMalformedError(
                f"citation {text!r} declares a window that is not a finite, increasing interval"
            )
        window = (start, end)
    return Citation(lane=lane, pointer=match.group("pointer"), window=window, text=text)


def _unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def resolve_json_pointer(document: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 pointer, raising rather than returning a default."""

    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise CitationMalformedError("a JSON pointer must start with '/'")
    current = document
    for token in pointer.split("/")[1:]:
        key = _unescape(token)
        if isinstance(current, Mapping):
            if key not in current:
                raise CitationUnresolvableError(f"pointer segment {key!r} is not present")
            current = current[key]
            continue
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            if not key.isdigit():
                raise CitationUnresolvableError(
                    f"pointer segment {key!r} is not an array index"
                )
            index = int(key)
            if index >= len(current):
                raise CitationUnresolvableError(f"array index {index} is out of range")
            current = current[index]
            continue
        raise CitationUnresolvableError(
            f"pointer segment {key!r} cannot be resolved against a scalar value"
        )
    return current


def lane_root(bundle: Mapping[str, Any], lane: str) -> Any:
    """Return one present lane, or raise because an absent lane cites nothing."""

    lanes = bundle.get("lanes")
    if not isinstance(lanes, Mapping) or lane not in lanes:
        raise CitationUnresolvableError(f"the bundle carries no {lane!r} lane")
    root = lanes[lane]
    if not isinstance(root, Mapping) or root.get("status") != "present":
        raise CitationUnresolvableError(
            f"the {lane!r} lane is absent from this bundle and cannot be cited"
        )
    return root


def resolve_citation(bundle: Mapping[str, Any], citation: Citation) -> Any:
    """Resolve a parsed citation, including its optional window assertion."""

    value = resolve_json_pointer(lane_root(bundle, citation.lane), citation.pointer)
    if citation.window is None:
        return value
    if not isinstance(value, Mapping):
        raise CitationUnresolvableError(
            f"citation {citation.text!r} asserts a window on a value that has no time bounds"
        )
    start = value.get("startSec")
    end = value.get("endSec")
    if not _is_number(start) or not _is_number(end):
        raise CitationUnresolvableError(
            f"citation {citation.text!r} asserts a window on a value that has no time bounds"
        )
    if not (float(start) < citation.window[1] and float(end) > citation.window[0]):
        raise CitationUnresolvableError(
            f"citation {citation.text!r} asserts a window the cited item does not cover"
        )
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def enumerate_pointers(document: Any, prefix: str = "", *, max_depth: int = 12) -> list[str]:
    """List every RFC 6901 pointer inside a document, for exhaustive tests."""

    if max_depth <= 0:
        return []
    pointers: list[str] = []
    if isinstance(document, Mapping):
        for key, value in document.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            child = f"{prefix}/{escaped}"
            pointers.append(child)
            pointers.extend(enumerate_pointers(value, child, max_depth=max_depth - 1))
    elif isinstance(document, Sequence) and not isinstance(document, (str, bytes)):
        for index, value in enumerate(document):
            child = f"{prefix}/{index}"
            pointers.append(child)
            pointers.extend(enumerate_pointers(value, child, max_depth=max_depth - 1))
    return pointers
