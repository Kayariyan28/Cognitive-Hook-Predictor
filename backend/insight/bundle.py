"""Deterministic assembly of the citable insight evidence bundle.

This module is pure: it takes result documents that were already validated and
published elsewhere, and returns one JSON-safe dictionary.  It performs no I/O,
starts no model, and never invents a value.  A lane with no evidence becomes an
explicit absent marker so a downstream reader can tell "missing" from "zero".
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence


BUNDLE_SCHEMA_VERSION = "insight-evidence-bundle/2"
LANE_KEYS = (
    "measured",
    "nanollava",
    "ast",
    "vjepa",
    "asr",
    "ocr",
    "context",
    "tribe",
)
HOOK_WINDOW_SECONDS = (0.0, 3.0)
MAXIMUM_TRIBE_PARCELS = 8

FORECAST_RESULT_SCHEMA_VERSION = "creator-forecast-result/1"
TRIBE_DESCRIPTOR_SCHEMA_VERSION = "tribe-cortical-descriptors/1"

_EMBEDDING_FEATURE_RE = re.compile(r"^embedding_\d+$")
_MEASURED_AUDIO_PREFIX = "measured_audio."
_AST_PREFIX = "audio.ast."
_VJEPA_PREFIX = "vjepa2_1."
_LANGUAGE_LABEL_PREFIX = "language:"
_ENGINE_LABEL_PREFIX = "engine:"
_FRAME_LABEL_PREFIX = "frame:"


class BundleUnavailableError(ValueError):
    """The upstream evidence cannot support a bundle; the caller must fail closed."""


def absent_lane(reason: str) -> dict[str, Any]:
    """Build the one legal representation of a lane that carries no evidence."""

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("an absent lane must state why it is absent")
    return {"status": "absent", "reason": reason}


def canonical_json(value: Any) -> str:
    """Serialize so that dictionary insertion order cannot change the digest."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def input_evidence_hash(bundle: Mapping[str, Any]) -> str:
    """Hash the canonical bundle with any existing digest field removed."""

    hashable = {key: value for key, value in bundle.items() if key != "inputEvidenceHash"}
    return hashlib.sha256(canonical_json(hashable).encode("utf-8")).hexdigest()


def _finite_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return value


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, (list, tuple)) else ()


def _available_provider(result: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    """Return one provider's validated worker output, or None when unavailable."""

    evidence = _mapping(result.get("evidence"))
    if evidence is None:
        return None
    providers = _mapping(evidence.get("optionalProviders"))
    if providers is None:
        return None
    provider = _mapping(providers.get(key))
    if provider is None or provider.get("status") != "available":
        return None
    return _mapping(provider.get("result"))


def _unavailable_reason(result: Mapping[str, Any], key: str, label: str) -> str:
    evidence = _mapping(result.get("evidence")) or {}
    providers = _mapping(evidence.get("optionalProviders")) or {}
    provider = _mapping(providers.get(key))
    if provider is None:
        return f"the forecast result declares no {label} branch"
    reason = provider.get("reason")
    if isinstance(reason, str) and reason.strip():
        return reason
    return f"the {label} branch did not publish evidence for this result"


def _descriptors(output: Mapping[str, Any], prefix: str, *, drop_embeddings: bool = False) -> dict[str, Any]:
    """Copy scalar features verbatim, stripping only the branch key prefix."""

    features = _mapping(output.get("features")) or {}
    descriptors: dict[str, Any] = {}
    for name, value in features.items():
        if not isinstance(name, str) or not name.startswith(prefix):
            continue
        short = name[len(prefix) :]
        if drop_embeddings and _EMBEDDING_FEATURE_RE.fullmatch(short):
            continue
        number = _finite_number(value)
        if number is None:
            continue
        descriptors[short] = number
    return descriptors


def _observations(output: Mapping[str, Any], kind: str) -> list[Mapping[str, Any]]:
    return [
        item
        for item in _sequence(output.get("observations"))
        if isinstance(item, Mapping) and item.get("kind") == kind
    ]


def _timed_item(index: int, observation: Mapping[str, Any]) -> dict[str, Any] | None:
    start = _finite_number(observation.get("startTime"))
    end = _finite_number(observation.get("endTime"))
    if start is None or end is None:
        return None
    return {"index": index, "startSec": start, "endSec": end}


def _labels(observation: Mapping[str, Any]) -> list[str]:
    return [item for item in _sequence(observation.get("labels")) if isinstance(item, str)]


def _label_value(observation: Mapping[str, Any], prefix: str) -> str | None:
    for label in _labels(observation):
        if label.startswith(prefix):
            return label[len(prefix) :]
    return None


def _parsed_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    try:
        parsed = json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _parsed_array(text: Any) -> list[Any] | None:
    if not isinstance(text, str):
        return None
    try:
        parsed = json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, list) else None


# --------------------------------------------------------------------------
# Lane builders
# --------------------------------------------------------------------------


def _measured_lane(
    result: Mapping[str, Any], comparative: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    evidence = _mapping(result.get("evidence"))
    metadata = _mapping(evidence.get("videoMetadata")) if evidence else None
    if metadata is None or metadata.get("status") != "available":
        return absent_lane(
            "the forecast result published no authoritative server media metadata"
        )
    # The source-video content hash is deliberately excluded: it is a
    # fingerprint of creator media, it is never citable evidence, and the
    # remote provider payload is built from this bundle.
    video = {
        "durationSeconds": _finite_number(metadata.get("durationSeconds")),
        "sizeBytes": _finite_number(metadata.get("sizeBytes")),
        "contentType": metadata.get("contentType")
        if isinstance(metadata.get("contentType"), str)
        else None,
    }
    if video["durationSeconds"] is None:
        return absent_lane("the server media probe published no finite duration")

    output = _available_provider(result, "measuredAudio")
    if output is None:
        audio: dict[str, Any] = {
            "status": "absent",
            "reason": _unavailable_reason(result, "measuredAudio", "measured audio"),
        }
    else:
        peaks: list[dict[str, Any]] = []
        for index, observation in enumerate(
            _observations(output, "measured-short-window-energy-peak")
        ):
            item = _timed_item(index, observation)
            if item is None:
                continue
            item["text"] = observation.get("text") if isinstance(observation.get("text"), str) else ""
            peaks.append(item)
        peaks.sort(key=lambda item: (item["startSec"], item["index"]))
        audio = {
            "status": "present",
            "descriptors": _descriptors(output, _MEASURED_AUDIO_PREFIX),
            "energyPeaks": peaks,
        }
    return {
        "status": "present",
        "video": video,
        "audio": audio,
        # Computed by code from the creator's own past measurements, never
        # asserted by a model. Absent until enough clips exist to rank against.
        "comparative": dict(comparative)
        if isinstance(comparative, Mapping)
        else {
            "status": "absent",
            "reason": "no comparative context was computed for this bundle",
        },
    }


def _nanollava_lane(result: Mapping[str, Any]) -> dict[str, Any]:
    output = _available_provider(result, "semanticModel")
    if output is None:
        return absent_lane(_unavailable_reason(result, "semanticModel", "keyframe semantics"))
    keyframes: list[dict[str, Any]] = []
    for index, observation in enumerate(_observations(output, "nanollava-keyframe-semantics")):
        item = _timed_item(index, observation)
        if item is None:
            continue
        text = observation.get("text") if isinstance(observation.get("text"), str) else ""
        item["text"] = text
        item["parsed"] = _parsed_object(text)
        keyframes.append(item)
    if not keyframes:
        return absent_lane("the keyframe-semantics branch published no timed observation")
    return {
        "status": "present",
        "keyframes": keyframes,
        "warnings": [item for item in _sequence(output.get("warnings")) if isinstance(item, str)],
    }


def _ast_lane(result: Mapping[str, Any]) -> dict[str, Any]:
    output = _available_provider(result, "audioModel")
    if output is None:
        return absent_lane(_unavailable_reason(result, "audioModel", "AudioSet label"))
    windows: list[dict[str, Any]] = []
    for index, observation in enumerate(_observations(output, "ast-audioset-window-labels")):
        item = _timed_item(index, observation)
        if item is None:
            continue
        parsed = _parsed_array(observation.get("text"))
        labels: list[dict[str, Any]] = []
        if parsed is not None:
            for entry in parsed:
                if not isinstance(entry, Mapping) or not isinstance(entry.get("label"), str):
                    continue
                labels.append(
                    {
                        "label": entry["label"],
                        "modelScore": _finite_number(entry.get("modelScore")),
                    }
                )
        if not labels:
            labels = [{"label": name, "modelScore": None} for name in _labels(observation)]
        item["labels"] = labels
        windows.append(item)
    if not windows:
        return absent_lane("the AudioSet label branch published no timed observation")
    return {
        "status": "present",
        "windows": windows,
        "descriptors": _descriptors(output, _AST_PREFIX),
    }


def _vjepa_lane(result: Mapping[str, Any]) -> dict[str, Any]:
    output = _available_provider(result, "vjepa21")
    if output is None:
        return absent_lane(_unavailable_reason(result, "vjepa21", "V-JEPA 2.1 visual"))
    windows: list[dict[str, Any]] = []
    for index, observation in enumerate(_observations(output, "learned-visual-window")):
        item = _timed_item(index, observation)
        if item is None:
            continue
        item["text"] = observation.get("text") if isinstance(observation.get("text"), str) else ""
        item["labels"] = _labels(observation)
        windows.append(item)
    descriptors = _descriptors(output, _VJEPA_PREFIX, drop_embeddings=True)
    if not windows and not descriptors:
        return absent_lane("the V-JEPA 2.1 branch published no citable summary evidence")
    return {"status": "present", "windows": windows, "descriptors": descriptors}


def _asr_lane(result: Mapping[str, Any]) -> dict[str, Any]:
    output = _available_provider(result, "asr")
    if output is None:
        return absent_lane(_unavailable_reason(result, "asr", "speech transcript"))
    segments: list[dict[str, Any]] = []
    language: str | None = None
    for index, observation in enumerate(_observations(output, "asr-transcript-segment")):
        item = _timed_item(index, observation)
        if item is None:
            continue
        if language is None:
            language = _label_value(observation, _LANGUAGE_LABEL_PREFIX)
        item["text"] = observation.get("text") if isinstance(observation.get("text"), str) else ""
        segments.append(item)
    if not segments:
        return absent_lane("the transcript branch published no timed segment")
    return {"status": "present", "language": language, "segments": segments}


def _ocr_lane(result: Mapping[str, Any]) -> dict[str, Any]:
    output = _available_provider(result, "ocr")
    if output is None:
        return absent_lane(_unavailable_reason(result, "ocr", "on-screen text"))
    frames: list[dict[str, Any]] = []
    engine: str | None = None
    for index, observation in enumerate(_observations(output, "ocr-frame-text-blocks")):
        if engine is None:
            engine = _label_value(observation, _ENGINE_LABEL_PREFIX)
        raw_frame_index = _label_value(observation, _FRAME_LABEL_PREFIX)
        try:
            frame_index = int(raw_frame_index) if raw_frame_index is not None else index
        except ValueError:
            frame_index = index
        blocks: list[dict[str, Any]] = []
        for entry in _parsed_array(observation.get("text")) or ():
            if not isinstance(entry, Mapping) or not isinstance(entry.get("text"), str):
                continue
            bbox = [
                value
                for value in (_finite_number(item) for item in _sequence(entry.get("bbox")))
                if value is not None
            ]
            blocks.append(
                {
                    "text": entry["text"],
                    "confidence": _finite_number(entry.get("confidence")),
                    "bbox": bbox if len(bbox) == 4 else None,
                }
            )
        frame: dict[str, Any] = {"frameIndex": frame_index, "blocks": blocks}
        timed = _timed_item(index, observation)
        frame["startSec"] = timed["startSec"] if timed else None
        frame["endSec"] = timed["endSec"] if timed else None
        frames.append(frame)
    if not frames:
        return absent_lane("the on-screen-text branch published no frame observation")
    return {"status": "present", "engine": engine, "frames": frames}


def _context_lane(
    result: Mapping[str, Any], declared_context: Mapping[str, Any] | None
) -> dict[str, Any]:
    evidence = _mapping(result.get("evidence")) or {}
    creator_context = _mapping(evidence.get("creatorContext")) or {}
    stored = _mapping(creator_context.get("value"))
    declared: dict[str, Any] = {}
    if stored is not None:
        declared.update(stored)
    if declared_context is not None:
        if not isinstance(declared_context, Mapping):
            raise BundleUnavailableError("declared context must be a JSON object")
        declared.update(declared_context)
    if not declared:
        return absent_lane("the creator declared no publishing context for this clip")
    return {"status": "present", "declared": declared}


def _tribe_lane(descriptors: Mapping[str, Any] | None) -> dict[str, Any]:
    if descriptors is None:
        return absent_lane("no TRIBE creator-report descriptor document was supplied")
    if descriptors.get("schemaVersion") != TRIBE_DESCRIPTOR_SCHEMA_VERSION:
        raise BundleUnavailableError(
            "TRIBE descriptors must be a "
            f"{TRIBE_DESCRIPTOR_SCHEMA_VERSION} document"
        )
    source = _mapping(descriptors.get("source")) or {}
    model = _mapping(source.get("model")) or {}
    tensor = _mapping(source.get("tensor")) or {}

    intervals: list[dict[str, Any]] = []
    for frame in _sequence(descriptors.get("frames")):
        record = _mapping(frame)
        if record is None:
            continue
        start = _finite_number(record.get("time"))
        duration = _finite_number(record.get("duration"))
        magnitude = _finite_number(record.get("rms"))
        if start is None or duration is None or magnitude is None:
            continue
        intervals.append(
            {
                "index": len(intervals),
                "startSec": start,
                "endSec": start + duration,
                "durationSec": duration,
                "magnitude": magnitude,
                "continuity": _finite_number(record.get("continuity")),
                "changeRate": _finite_number(record.get("changeRate")),
                "spatialDistribution": _finite_number(record.get("spatialDistribution")),
            }
        )
    if not intervals:
        return absent_lane("the TRIBE descriptor document carries no usable interval")

    phases: list[dict[str, Any]] = []
    for phase in _sequence(descriptors.get("phases")):
        record = _mapping(phase)
        if record is None or record.get("id") not in {"early", "middle", "late"}:
            continue
        start = _finite_number(record.get("startTime"))
        end = _finite_number(record.get("endTime"))
        if start is None or end is None:
            continue
        phases.append(
            {
                "id": record["id"],
                "startSec": start,
                "endSec": end,
                "responseMagnitude": _finite_number(record.get("responseMagnitude")),
                "signedMean": _finite_number(record.get("signedMean")),
            }
        )

    parcels: list[dict[str, Any]] = []
    for parcel in _sequence(descriptors.get("parcels")):
        record = _mapping(parcel)
        if record is None:
            continue
        dispersion = _finite_number(record.get("rms"))
        if dispersion is None or not isinstance(record.get("name"), str):
            continue
        parcels.append(
            {
                "key": record.get("key") if isinstance(record.get("key"), str) else None,
                "name": record["name"],
                "hemisphere": record.get("hemisphere")
                if record.get("hemisphere") in {"left", "right"}
                else None,
                "vertexCount": _finite_number(record.get("vertexCount")),
                "rms": dispersion,
                "signedMean": _finite_number(record.get("signedMean")),
            }
        )
    # ``src/tribe/descriptors.js`` already sorts by rms then hemisphere then
    # label index; re-sorting with the same key keeps the bundle stable even if
    # a caller hands over a reordered document.
    parcels.sort(
        key=lambda item: (
            -item["rms"],
            0 if item["hemisphere"] == "left" else 1,
            item["name"],
        )
    )
    return {
        "status": "present",
        "rankedBy": "rms",
        "provenance": {
            "resultId": source.get("resultId") if isinstance(source.get("resultId"), str) else None,
            "predictionType": source.get("predictionType")
            if isinstance(source.get("predictionType"), str)
            else None,
            "modelId": model.get("id") if isinstance(model.get("id"), str) else None,
            "modelRevision": model.get("revision") if isinstance(model.get("revision"), str) else None,
            "inferenceMode": source.get("inferenceMode")
            if isinstance(source.get("inferenceMode"), str)
            else None,
            "tensorSha256": tensor.get("sha256") if isinstance(tensor.get("sha256"), str) else None,
        },
        "intervals": intervals,
        "phases": phases,
        "parcels": parcels[:MAXIMUM_TRIBE_PARCELS],
    }


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------


def assemble_evidence_bundle(
    forecast_result: Mapping[str, Any],
    *,
    tribe_descriptors: Mapping[str, Any] | None = None,
    declared_context: Mapping[str, Any] | None = None,
    comparative: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one citable bundle from completed, already-validated results."""

    if not isinstance(forecast_result, Mapping):
        raise BundleUnavailableError("the forecast result must be a JSON object")
    if forecast_result.get("schemaVersion") != FORECAST_RESULT_SCHEMA_VERSION:
        raise BundleUnavailableError(
            f"the forecast result must be a {FORECAST_RESULT_SCHEMA_VERSION} document"
        )
    result_id = forecast_result.get("resultId")
    if not isinstance(result_id, str) or not result_id:
        raise BundleUnavailableError("the forecast result carries no result identity")
    if _mapping(forecast_result.get("evidence")) is None:
        raise BundleUnavailableError("the forecast result carries no evidence envelope")

    tribe_result_id: str | None = None
    if tribe_descriptors is not None:
        if not isinstance(tribe_descriptors, Mapping):
            raise BundleUnavailableError("TRIBE descriptors must be a JSON object")
        source = _mapping(tribe_descriptors.get("source")) or {}
        candidate = source.get("resultId")
        tribe_result_id = candidate if isinstance(candidate, str) else None

    bundle = {
        "schemaVersion": BUNDLE_SCHEMA_VERSION,
        "source": {
            "forecastResultId": result_id,
            "tribeResultId": tribe_result_id,
            "window": None,
        },
        "lanes": {
            "measured": _measured_lane(forecast_result, comparative),
            "nanollava": _nanollava_lane(forecast_result),
            "ast": _ast_lane(forecast_result),
            "vjepa": _vjepa_lane(forecast_result),
            "asr": _asr_lane(forecast_result),
            "ocr": _ocr_lane(forecast_result),
            "context": _context_lane(forecast_result, declared_context),
            "tribe": _tribe_lane(tribe_descriptors),
        },
    }
    bundle["inputEvidenceHash"] = input_evidence_hash(bundle)
    return bundle


def _overlaps(start: Any, end: Any, window: tuple[float, float]) -> bool:
    """Keep an item that straddles a boundary; drop one that merely touches it."""

    lower = _finite_number(start)
    upper = _finite_number(end)
    if lower is None or upper is None:
        return False
    return float(lower) < window[1] and float(upper) > window[0]


def _sliced(items: Iterable[Mapping[str, Any]], window: tuple[float, float]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in items
        if _overlaps(item.get("startSec"), item.get("endSec"), window)
    ]


def _audio_onset(peaks: Sequence[Mapping[str, Any]], window: tuple[float, float]) -> dict[str, Any]:
    """Report the first measured energy peak in the window, or explicit nulls."""

    if not peaks:
        return {"firstEnergyPeakSec": None, "prePeakSilenceSec": None}
    first = min(float(item["startSec"]) for item in peaks)
    return {
        "firstEnergyPeakSec": first,
        "prePeakSilenceSec": round(first - window[0], 6),
    }


def hook_evidence_card(
    bundle: Mapping[str, Any],
    window: tuple[float, float] = HOOK_WINDOW_SECONDS,
) -> dict[str, Any]:
    """Slice a bundle to the hook window without inventing a single value."""

    if not isinstance(bundle, Mapping) or bundle.get("schemaVersion") != BUNDLE_SCHEMA_VERSION:
        raise BundleUnavailableError(f"a {BUNDLE_SCHEMA_VERSION} bundle is required")
    start, end = (float(window[0]), float(window[1]))
    if not math.isfinite(start) or not math.isfinite(end) or start >= end:
        raise BundleUnavailableError("the hook window must be a finite, increasing interval")
    bounds = (start, end)
    lanes = _mapping(bundle.get("lanes")) or {}
    sliced: dict[str, Any] = {}

    for key in LANE_KEYS:
        lane = _mapping(lanes.get(key))
        if lane is None or lane.get("status") != "present":
            sliced[key] = dict(lane) if lane is not None else absent_lane(
                f"the {key} lane was not assembled"
            )
            continue
        sliced[key] = _slice_lane(key, lane, bounds)

    card = {
        "schemaVersion": BUNDLE_SCHEMA_VERSION,
        "source": {
            **{k: v for k, v in (_mapping(bundle.get("source")) or {}).items()},
            "window": [start, end],
        },
        "lanes": sliced,
    }
    card["inputEvidenceHash"] = input_evidence_hash(card)
    return card


def _empty_window_marker(key: str) -> dict[str, Any]:
    return absent_lane(f"no {key} evidence falls inside the requested window")


def _slice_lane(
    key: str, lane: Mapping[str, Any], window: tuple[float, float]
) -> dict[str, Any]:
    if key == "measured":
        audio = _mapping(lane.get("audio")) or {}
        if audio.get("status") == "present":
            peaks = _sliced(_sequence(audio.get("energyPeaks")), window)
            audio = {
                "status": "present",
                "descriptors": dict(_mapping(audio.get("descriptors")) or {}),
                "energyPeaks": peaks,
                "onset": _audio_onset(peaks, window),
            }
        return {
            "status": "present",
            "video": dict(_mapping(lane.get("video")) or {}),
            "audio": audio,
            "comparative": dict(_mapping(lane.get("comparative")) or {}),
        }

    if key == "nanollava":
        keyframes = _sliced(_sequence(lane.get("keyframes")), window)
        if not keyframes:
            return _empty_window_marker(key)
        return {
            "status": "present",
            "keyframes": keyframes,
            "warnings": list(_sequence(lane.get("warnings"))),
        }

    if key == "ast":
        windows = _sliced(_sequence(lane.get("windows")), window)
        if not windows:
            return _empty_window_marker(key)
        return {
            "status": "present",
            "windows": windows,
            "descriptors": dict(_mapping(lane.get("descriptors")) or {}),
        }

    if key == "vjepa":
        windows = _sliced(_sequence(lane.get("windows")), window)
        if not windows:
            return _empty_window_marker(key)
        return {
            "status": "present",
            "windows": windows,
            "descriptors": dict(_mapping(lane.get("descriptors")) or {}),
        }

    if key == "asr":
        segments = _sliced(_sequence(lane.get("segments")), window)
        if not segments:
            return _empty_window_marker(key)
        return {"status": "present", "language": lane.get("language"), "segments": segments}

    if key == "ocr":
        frames = [
            dict(frame)
            for frame in _sequence(lane.get("frames"))
            if isinstance(frame, Mapping)
            and (
                frame.get("startSec") is None
                or _overlaps(frame.get("startSec"), frame.get("endSec"), window)
            )
        ]
        if not frames:
            return _empty_window_marker(key)
        return {"status": "present", "engine": lane.get("engine"), "frames": frames}

    if key == "tribe":
        intervals = _sliced(_sequence(lane.get("intervals")), window)
        if not intervals:
            return _empty_window_marker(key)
        return {
            "status": "present",
            "rankedBy": lane.get("rankedBy"),
            "provenance": dict(_mapping(lane.get("provenance")) or {}),
            "intervals": intervals,
            "phases": _sliced(_sequence(lane.get("phases")), window),
            "parcels": [dict(item) for item in _sequence(lane.get("parcels")) if isinstance(item, Mapping)],
        }

    # ``context`` and any future untimed lane describe the whole clip.
    return dict(lane)
