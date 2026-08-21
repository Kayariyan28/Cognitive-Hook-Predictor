"""A deterministic readout of the hook window: a timeline and a checklist.

Nothing here calls a model. Every marker is an item already in the evidence
bundle, and every check is a measured value compared against a **declared
convention** — a threshold this project chose, not a calibrated boundary and not
a prediction. A check that cannot be measured says so rather than passing.

This exists so the app is useful the moment a job completes, whether or not any
language model is installed, and so the insight lane's job is narrowed to
explaining and proposing rather than to being present at all.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Sequence

from .bundle import HOOK_WINDOW_SECONDS
from .citations import (
    CitationMalformedError,
    CitationUnresolvableError,
    parse_citation,
    resolve_citation,
)


READOUT_SCHEMA_VERSION = "insight-hook-readout/1"

# Conventions, not calibrated boundaries. Each one is a round number this
# project picked so a creator gets a consistent yardstick; none of them has been
# evaluated against any audience outcome, and the readout says so.
OPENING_SILENCE_SECONDS = 0.8
FIRST_WORDS_SECONDS = 1.5
LOW_VISUAL_CHANGE = 0.05
STATUSES = ("clear", "flagged", "unmeasured")

MARKER_KINDS = (
    "audio-peak",
    "spoken-segment",
    "on-screen-text",
    "keyframe",
    "visual-window",
    "cortical-interval",
)


def _resolve(bundle: Mapping[str, Any], citation: str) -> Any:
    try:
        return resolve_citation(bundle, parse_citation(citation))
    except (CitationMalformedError, CitationUnresolvableError):
        return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _lane(bundle: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    lanes = bundle.get("lanes")
    if not isinstance(lanes, Mapping):
        return None
    lane = lanes.get(name)
    if not isinstance(lane, Mapping) or lane.get("status") != "present":
        return None
    return lane


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, (list, tuple)) else ()


def _label_value(item: Mapping[str, Any], prefix: str) -> str | None:
    for label in _sequence(item.get("labels")):
        if isinstance(label, str) and label.startswith(prefix):
            return label[len(prefix) :]
    return None


# --------------------------------------------------------------------------
# Timeline
# --------------------------------------------------------------------------


def _marker(
    kind: str, start: Any, end: Any, label: str, citation: str, value: Any = None
) -> dict[str, Any] | None:
    moment = _number(start)
    if moment is None:
        return None
    return {
        "kind": kind,
        "startSec": moment,
        "endSec": _number(end),
        "label": label,
        "citation": citation,
        "value": _number(value),
    }


def build_timeline(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every timed item in the hook window, on one axis, each one citable."""

    markers: list[dict[str, Any]] = []

    measured = _lane(bundle, "measured")
    audio = measured.get("audio") if isinstance(measured, Mapping) else None
    if isinstance(audio, Mapping) and audio.get("status") == "present":
        for index, peak in enumerate(_sequence(audio.get("energyPeaks"))):
            if not isinstance(peak, Mapping):
                continue
            marker = _marker(
                "audio-peak",
                peak.get("startSec"),
                peak.get("endSec"),
                "Measured short-window energy peak",
                f"measured:/audio/energyPeaks/{index}",
            )
            if marker:
                markers.append(marker)

    asr = _lane(bundle, "asr")
    if asr is not None:
        for index, segment in enumerate(_sequence(asr.get("segments"))):
            if not isinstance(segment, Mapping):
                continue
            text = segment.get("text")
            marker = _marker(
                "spoken-segment",
                segment.get("startSec"),
                segment.get("endSec"),
                text if isinstance(text, str) else "Spoken segment",
                f"asr:/segments/{index}",
            )
            if marker:
                markers.append(marker)

    ocr = _lane(bundle, "ocr")
    if ocr is not None:
        for index, frame in enumerate(_sequence(ocr.get("frames"))):
            if not isinstance(frame, Mapping):
                continue
            blocks = [
                block.get("text")
                for block in _sequence(frame.get("blocks"))
                if isinstance(block, Mapping) and isinstance(block.get("text"), str)
            ]
            if not blocks:
                continue
            marker = _marker(
                "on-screen-text",
                frame.get("startSec"),
                frame.get("endSec"),
                " · ".join(blocks),
                f"ocr:/frames/{index}",
            )
            if marker:
                markers.append(marker)

    nanollava = _lane(bundle, "nanollava")
    if nanollava is not None:
        for index, keyframe in enumerate(_sequence(nanollava.get("keyframes"))):
            if not isinstance(keyframe, Mapping):
                continue
            parsed = keyframe.get("parsed")
            scene = parsed.get("scene") if isinstance(parsed, Mapping) else None
            marker = _marker(
                "keyframe",
                keyframe.get("startSec"),
                keyframe.get("endSec"),
                scene if isinstance(scene, str) else "Sampled keyframe",
                f"nanollava:/keyframes/{index}",
            )
            if marker:
                markers.append(marker)

    vjepa = _lane(bundle, "vjepa")
    if vjepa is not None:
        for index, window in enumerate(_sequence(vjepa.get("windows"))):
            if not isinstance(window, Mapping):
                continue
            relative = _label_value(window, "relative-change-")
            marker = _marker(
                "visual-window",
                window.get("startSec"),
                window.get("endSec"),
                f"Visual change: {relative}" if relative else "Decoded visual window",
                f"vjepa:/windows/{index}",
            )
            if marker:
                markers.append(marker)

    tribe = _lane(bundle, "tribe")
    if tribe is not None:
        for index, interval in enumerate(_sequence(tribe.get("intervals"))):
            if not isinstance(interval, Mapping):
                continue
            marker = _marker(
                "cortical-interval",
                interval.get("startSec"),
                interval.get("endSec"),
                "Predicted cortical interval",
                f"tribe:/intervals/{index}",
                value=interval.get("magnitude"),
            )
            if marker:
                markers.append(marker)

    markers.sort(key=lambda item: (item["startSec"], item["kind"]))
    return markers


# --------------------------------------------------------------------------
# Checklist
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Check:
    id: str
    label: str
    status: str
    detail: str
    citations: tuple[str, ...] = ()
    measured: float | None = None
    threshold: float | None = None

    def public_value(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "status": self.status,
            "detail": self.detail,
            "citations": list(self.citations),
            "measured": self.measured,
            "threshold": self.threshold,
            # A declared convention, never a calibrated boundary.
            "thresholdKind": "declared-convention" if self.threshold is not None else None,
        }


def _unmeasured(check_id: str, label: str, detail: str) -> Check:
    return Check(id=check_id, label=label, status="unmeasured", detail=detail)


def _opening_silence(bundle: Mapping[str, Any]) -> Check:
    label = "Silence before the first sound peak"
    onset = _resolve(bundle, "measured:/audio/onset")
    if not isinstance(onset, Mapping):
        return _unmeasured(
            "opening_silence",
            label,
            "This clip has no measured audio onset for the hook window.",
        )
    silence = _number(onset.get("prePeakSilenceSec"))
    if silence is None:
        return _unmeasured(
            "opening_silence",
            label,
            "No measured energy peak falls inside the hook window, so the gap before one cannot be measured.",
        )
    flagged = silence > OPENING_SILENCE_SECONDS
    return Check(
        id="opening_silence",
        label=label,
        status="flagged" if flagged else "clear",
        detail=(
            f"The first measured energy peak arrives {silence:g} s in, past this "
            f"project's {OPENING_SILENCE_SECONDS:g} s convention."
            if flagged
            else f"The first measured energy peak arrives {silence:g} s in."
        ),
        citations=("measured:/audio/onset/prePeakSilenceSec",),
        measured=silence,
        threshold=OPENING_SILENCE_SECONDS,
    )


def _first_words(bundle: Mapping[str, Any]) -> Check:
    label = "First spoken words"
    first = _resolve(bundle, "asr:/segments/0")
    if not isinstance(first, Mapping):
        return _unmeasured(
            "first_words_late",
            label,
            "No transcript segment falls inside the hook window; the transcript branch may be unavailable.",
        )
    start = _number(first.get("startSec"))
    if start is None:
        return _unmeasured("first_words_late", label, "The first segment has no start time.")
    flagged = start > FIRST_WORDS_SECONDS
    return Check(
        id="first_words_late",
        label=label,
        status="flagged" if flagged else "clear",
        detail=(
            f"Speech starts {start:g} s in, past this project's "
            f"{FIRST_WORDS_SECONDS:g} s convention."
            if flagged
            else f"Speech starts {start:g} s in."
        ),
        citations=("asr:/segments/0/startSec",),
        measured=start,
        threshold=FIRST_WORDS_SECONDS,
    )


def _opening_text(bundle: Mapping[str, Any]) -> Check:
    label = "On-screen text at the opening"
    lane = _lane(bundle, "ocr")
    if lane is None:
        return _unmeasured(
            "no_opening_text",
            label,
            "The on-screen-text branch published nothing for the hook window.",
        )
    frames = [frame for frame in _sequence(lane.get("frames")) if isinstance(frame, Mapping)]
    if not frames:
        return _unmeasured("no_opening_text", label, "No keyframe covers the hook window.")
    earliest = min(frames, key=lambda frame: _number(frame.get("startSec")) or 0.0)
    index = frames.index(earliest)
    blocks = [
        block
        for block in _sequence(earliest.get("blocks"))
        if isinstance(block, Mapping) and isinstance(block.get("text"), str)
    ]
    return Check(
        id="no_opening_text",
        label=label,
        status="flagged" if not blocks else "clear",
        detail=(
            "No text was recognized on the earliest keyframe in the hook window."
            if not blocks
            else f"{len(blocks)} text block(s) recognized on the earliest hook keyframe."
        ),
        citations=(f"ocr:/frames/{index}",),
        measured=float(len(blocks)),
    )


def _visual_change(bundle: Mapping[str, Any]) -> Check:
    label = "Visual change across the opening"
    value = _resolve(bundle, "vjepa:/descriptors/temporal_change_mean")
    change = _number(value)
    if change is None:
        return _unmeasured(
            "low_visual_change",
            label,
            "The V-JEPA branch published no temporal-change descriptor.",
        )
    flagged = change < LOW_VISUAL_CHANGE
    return Check(
        id="low_visual_change",
        label=label,
        status="flagged" if flagged else "clear",
        detail=(
            f"Mean decoded visual change is {change:g}, below this project's "
            f"{LOW_VISUAL_CHANGE:g} convention."
            if flagged
            else f"Mean decoded visual change is {change:g}."
        ),
        citations=("vjepa:/descriptors/temporal_change_mean",),
        measured=change,
        threshold=LOW_VISUAL_CHANGE,
    )


def _described_opening(bundle: Mapping[str, Any]) -> Check:
    label = "A described frame from inside the opening"
    lane = _lane(bundle, "nanollava")
    if lane is None:
        return _unmeasured(
            "opening_frame_described",
            label,
            "The keyframe-semantics branch published nothing for the hook window.",
        )
    keyframes = [
        frame for frame in _sequence(lane.get("keyframes")) if isinstance(frame, Mapping)
    ]
    if not keyframes:
        return _unmeasured("opening_frame_described", label, "No keyframe covers the hook window.")
    return Check(
        id="opening_frame_described",
        label=label,
        status="clear",
        detail=f"{len(keyframes)} described keyframe(s) cover the hook window.",
        citations=("nanollava:/keyframes/0",),
        measured=float(len(keyframes)),
    )


CHECKS = (_opening_silence, _first_words, _opening_text, _visual_change, _described_opening)


def build_checklist(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [check(bundle).public_value() for check in CHECKS]


def build_hook_readout(
    bundle: Mapping[str, Any],
    *,
    window: tuple[float, float] = HOOK_WINDOW_SECONDS,
) -> dict[str, Any]:
    """The whole deterministic readout. No model is consulted at any point."""

    source = bundle.get("source") if isinstance(bundle.get("source"), Mapping) else {}
    declared = source.get("window")
    bounds = (
        (float(declared[0]), float(declared[1]))
        if isinstance(declared, (list, tuple)) and len(declared) == 2
        else window
    )
    checklist = build_checklist(bundle)
    return {
        "schemaVersion": READOUT_SCHEMA_VERSION,
        "windowSeconds": [bounds[0], bounds[1]],
        "source": {
            "forecastResultId": source.get("forecastResultId"),
            "tribeResultId": source.get("tribeResultId"),
            "inputEvidenceHash": bundle.get("inputEvidenceHash"),
        },
        "timeline": build_timeline(bundle),
        "checklist": checklist,
        "flaggedCount": sum(1 for check in checklist if check["status"] == "flagged"),
        "unmeasuredCount": sum(1 for check in checklist if check["status"] == "unmeasured"),
        "behavioralOutcome": False,
        "limits": (
            "Every marker is a measurement already in this clip's evidence, and every "
            "check compares one measurement against a threshold this project declared "
            "as a convention. No threshold here has been evaluated against any audience "
            "outcome, a flag is not a defect, and a clear check is not a prediction that "
            "anything will work."
        ),
    }


def readout_digest(readout: Mapping[str, Any]) -> str:
    """Stable digest of a readout, for callers that cache or compare them."""

    import hashlib

    payload = json.dumps(readout, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
