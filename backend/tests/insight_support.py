"""Realistic fake result documents shared by the insight-layer tests.

These mirror the exact shapes ``backend/forecast/jobs.py`` publishes and
``src/tribe/descriptors.js`` derives.  They contain no real media, no model
call, and no network access.
"""

from __future__ import annotations

import json
from typing import Any, Mapping


FORECAST_RESULT_ID = "3f1c2b7d9e0a4c5b8f6d1e2a3b4c5d6e"
TRIBE_RESULT_ID = "9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d"


def _provenance(branch: str, model_id: str) -> dict[str, Any]:
    return {
        "branch": branch,
        "modelId": model_id,
        "modelRevision": "1" * 40,
        "modelWeightsSha256": "2" * 64,
        "adapterId": f"{branch}-adapter",
        "codeRevision": "3" * 40,
        "preprocessingId": f"{branch}-preprocessing/1",
        "preprocessingSha256": "4" * 64,
    }


def _worker_output(
    branch: str,
    *,
    model_id: str,
    evidence_kind: str,
    features: Mapping[str, float],
    observations: list[dict[str, Any]],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schemaVersion": "creator-forecast-worker-output/1",
        "branch": branch,
        "inputSha256": "a" * 64,
        "startedAt": "2026-02-01T10:00:00Z",
        "completedAt": "2026-02-01T10:00:04Z",
        "evidenceKind": evidence_kind,
        "features": dict(features),
        "observations": observations,
        "warnings": list(warnings or []),
        "provenance": _provenance(branch, model_id),
        "behavioralOutcome": False,
    }


def measured_audio_provider() -> dict[str, Any]:
    return {
        "status": "available",
        "configured": True,
        "forecastContribution": False,
        "behavioralOutcome": False,
        "evidenceKind": "measured-audio-descriptors",
        "result": _worker_output(
            "measuredAudio",
            model_id="signalframe/measured-audio",
            evidence_kind="measured-audio-descriptors",
            features={
                "measured_audio.present": 1.0,
                "measured_audio.decoded_seconds": 21.5,
                "measured_audio.rms": 0.1372,
                "measured_audio.peak": 0.9125,
                "measured_audio.silent_window_fraction": 0.1875,
                "measured_audio.spectral_centroid_hz_mean": 2418.5,
            },
            observations=[
                {
                    "kind": "measured-short-window-energy-peak",
                    "startTime": 1.44,
                    "endTime": 1.504,
                    "text": "A relative short-window RMS peak was measured here; this is signal energy, not attention or engagement.",
                    "labels": ["measured", "relative-within-clip", "not-semantic"],
                },
                {
                    "kind": "measured-short-window-energy-peak",
                    "startTime": 9.12,
                    "endTime": 9.184,
                    "text": "A relative short-window RMS peak was measured here; this is signal energy, not attention or engagement.",
                    "labels": ["measured", "relative-within-clip", "not-semantic"],
                },
            ],
            warnings=[
                "These are deterministic PCM and spectral measurements, not learned speech, music, emotion, attention, or audience predictions."
            ],
        ),
    }


def semantic_provider() -> dict[str, Any]:
    def keyframe(start: float, end: float, scene: str, action: str, visible_text: Any) -> dict[str, Any]:
        return {
            "kind": "nanollava-keyframe-semantics",
            "startTime": start,
            "endTime": end,
            "text": json.dumps(
                {
                    "action": action,
                    "scene": scene,
                    "shot": "medium",
                    "uncertainties": None,
                    "visibleText": visible_text,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            "labels": ["model-derived", "single-keyframe", "nanollava-fallback"],
        }

    return {
        "status": "available",
        "configured": True,
        "forecastContribution": False,
        "behavioralOutcome": False,
        "evidenceKind": "learned-keyframe-semantics",
        "result": _worker_output(
            "semanticModel",
            model_id="mlx-community/nanoLLaVA-1.5-8bit",
            evidence_kind="learned-keyframe-semantics",
            features={},
            observations=[
                keyframe(0.0, 2.15, "a kitchen counter", "a person holds a jar", "READ THIS"),
                keyframe(2.15, 6.45, "the same counter", "a person pours liquid", None),
                keyframe(17.2, 21.5, "a plated dish", "a person lifts a fork", None),
            ],
        ),
    }


def ast_provider() -> dict[str, Any]:
    def window(index: int, start: float, end: float, labels: list[tuple[str, float]]) -> dict[str, Any]:
        return {
            "kind": "ast-audioset-window-labels",
            "startTime": start,
            "endTime": end,
            "text": json.dumps(
                [{"label": label, "modelScore": score} for label, score in labels],
                separators=(",", ":"),
            ),
            "labels": [label for label, _ in labels],
        }

    return {
        "status": "available",
        "configured": True,
        "forecastContribution": False,
        "behavioralOutcome": False,
        "evidenceKind": "learned-audioset-label-evidence",
        "result": _worker_output(
            "audioModel",
            model_id="MIT/ast-finetuned-audioset-10-10-0.4593",
            evidence_kind="learned-audioset-label-evidence",
            features={
                "audio.ast.windows_classified": 3.0,
                "audio.ast.mean_top_label_score": 0.6425,
            },
            observations=[
                window(0, 0.0, 10.0, [("Speech", 0.8125), ("Music", 0.1075)]),
                window(1, 10.0, 20.0, [("Music", 0.5525), ("Speech", 0.2025)]),
                window(2, 20.0, 21.5, [("Silence", 0.4375)]),
            ],
        ),
    }


def vjepa_provider() -> dict[str, Any]:
    def window(start: float, end: float, relative: str) -> dict[str, Any]:
        return {
            "kind": "learned-visual-window",
            "startTime": start,
            "endTime": end,
            "text": "Visual representation changed more here than in the clip's other decoded windows. This is descriptive V-JEPA 2.1 evidence, not attention or predicted performance.",
            "labels": ["vjepa2.1", "descriptive-only", f"relative-change-{relative}"],
        }

    features = {
        "vjepa2_1.embedding_norm_mean": 12.75,
        "vjepa2_1.temporal_consistency_mean": 0.8825,
        "vjepa2_1.temporal_change_mean": 0.2475,
        "vjepa2_1.temporal_change_peak": 0.6125,
    }
    for dimension in range(4):
        features[f"vjepa2_1.embedding_{dimension:03d}"] = 0.5 + dimension

    return {
        "status": "available",
        "configured": True,
        "forecastContribution": False,
        "behavioralOutcome": False,
        "evidenceKind": "learned-visual-representation",
        "result": _worker_output(
            "vjepa21",
            model_id="facebookresearch/vjepa2.1-vit-base-384",
            evidence_kind="learned-visual-representation",
            features=features,
            observations=[
                window(0.0, 2.0, "higher"),
                window(2.0, 4.0, "typical"),
                window(19.5, 21.5, "lower"),
            ],
        ),
    }


def asr_provider() -> dict[str, Any]:
    def segment(start: float, end: float, text: str) -> dict[str, Any]:
        return {
            "kind": "asr-transcript-segment",
            "startTime": start,
            "endTime": end,
            "text": text,
            "labels": ["asr", "language:en", "no-speaker-identity"],
        }

    return {
        "status": "available",
        "configured": True,
        "forecastContribution": False,
        "behavioralOutcome": False,
        "evidenceKind": "measured-speech-transcript",
        "result": _worker_output(
            "asr",
            model_id="mlx-community/whisper-large-v3-turbo",
            evidence_kind="measured-speech-transcript",
            features={"asr.segment_count": 2.0, "asr.speech_seconds": 12.5},
            observations=[
                segment(0.0, 2.6, "Stop scrolling, this jar changed my kitchen."),
                segment(2.6, 12.5, "You only need three things and about a minute."),
            ],
        ),
    }


def ocr_provider() -> dict[str, Any]:
    def frame(index: int, start: float, end: float, blocks: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "kind": "ocr-frame-text-blocks",
            "startTime": start,
            "endTime": end,
            "text": json.dumps(blocks, separators=(",", ":")),
            "labels": ["ocr", f"frame:{index}", "engine:ocrmac", "not-semantic"],
        }

    return {
        "status": "available",
        "configured": True,
        "forecastContribution": False,
        "behavioralOutcome": False,
        "evidenceKind": "measured-on-screen-text",
        "result": _worker_output(
            "ocr",
            model_id="apple/vision-text-recognition",
            evidence_kind="measured-on-screen-text",
            features={"ocr.frames_with_text": 2.0, "ocr.block_count": 3.0},
            observations=[
                frame(0, 0.0, 2.15, [{"text": "READ THIS", "confidence": 0.9375, "bbox": [0.1, 0.1, 0.5, 0.2]}]),
                frame(1, 2.15, 6.45, [{"text": "STEP ONE", "confidence": 0.8125, "bbox": [0.1, 0.7, 0.4, 0.8]}]),
            ],
        ),
    }


def unavailable_provider(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "configured": False,
        "forecastContribution": False,
        "reason": reason,
    }


def forecast_result(**overrides: Any) -> dict[str, Any]:
    """Build a completed ``creator-forecast-result/1`` document."""

    providers: dict[str, Any] = {
        "vjepa21": vjepa_provider(),
        "videoLlama21Av": unavailable_provider(
            "No immutable executable worker is configured for this branch."
        ),
        "audioModel": ast_provider(),
        "semanticModel": semantic_provider(),
        "measuredAudio": measured_audio_provider(),
        "account": unavailable_provider(
            "No authenticated, versioned provider is configured for this branch."
        ),
        "trends": unavailable_provider(
            "No authenticated, versioned provider is configured for this branch."
        ),
        "competitors": unavailable_provider(
            "No authenticated, versioned provider is configured for this branch."
        ),
    }
    providers.update(overrides.pop("optionalProviders", {}))
    context = overrides.pop(
        "context",
        {
            "schemaVersion": "creator-forecast-context/1",
            "platform": "reels",
            "postingGoal": "saves",
            "audienceRegion": "IN",
        },
    )
    result: dict[str, Any] = {
        "schemaVersion": "creator-forecast-result/1",
        "resultId": FORECAST_RESULT_ID,
        "jobId": FORECAST_RESULT_ID,
        "createdAt": "2026-02-01T10:00:05Z",
        "input": {
            "sha256": "a" * 64,
            "sizeBytes": 4823104,
            "durationSeconds": 21.5,
            "contentType": "video/mp4",
        },
        "evidence": {
            "schemaVersion": "creator-forecast-evidence/1",
            "videoMetadata": {
                "status": "available",
                "source": "server-ffprobe",
                "sha256": "a" * 64,
                "sizeBytes": 4823104,
                "durationSeconds": 21.5,
                "contentType": "video/mp4",
                "forecastContribution": False,
            },
            "creatorContext": {
                "status": "provided" if context is not None else "not-provided",
                "value": context,
                "forecastContribution": False,
            },
            "optionalProviders": providers,
            "tribe": {
                "status": "not-invoked",
                "forecastContribution": False,
                "reason": "TRIBE remains an independent cortical prediction path.",
            },
        },
        "behavioralHeads": {},
        "boundaries": [
            "Model-worker outputs are descriptive feature evidence, not audience-outcome probabilities."
        ],
    }
    result.update(overrides)
    return result


def tribe_descriptors(**overrides: Any) -> dict[str, Any]:
    """Build a ``tribe-cortical-descriptors/1`` creator-report document."""

    frames = []
    for index in range(6):
        time = round(index * 1.49, 6)
        frames.append(
            {
                "index": index,
                "time": time,
                "duration": 1.49,
                "endTime": round(time + 1.49, 6),
                "mean": 0.0125 * (index + 1),
                "rms": round(0.4 + 0.05 * index, 6),
                "spatialStd": 0.31,
                "positiveFraction": 0.52,
                "negativeFraction": 0.48,
                "spatialDistribution": 41.25,
                "changeRms": None if index == 0 else round(0.11 + 0.01 * index, 6),
                "changeRate": None if index == 0 else round(0.074 + 0.005 * index, 6),
                "continuity": None if index == 0 else round(0.82 - 0.02 * index, 6),
                "topParcel": None,
            }
        )
    parcels = [
        {
            "key": f"left-{index}",
            "hemisphere": "left" if index % 2 == 0 else "right",
            "labelIndex": index,
            "name": f"Parcel {index}",
            "vertexCount": 300 + index,
            "signedMean": 0.01 * index,
            "rms": round(0.9 - 0.05 * index, 6),
            "maxPositive": None,
            "maxNegative": None,
            "maxAbsolute": None,
            "peakMagnitude": None,
        }
        for index in range(12)
    ]
    document: dict[str, Any] = {
        "schemaVersion": "tribe-cortical-descriptors/1",
        "source": {
            "descriptorType": "descriptive cortical response",
            "resultId": TRIBE_RESULT_ID,
            "predictionType": "average-subject cortical BOLD response",
            "isViralityScore": False,
            "behavioralPrediction": False,
            "model": {
                "id": "facebook/tribev2",
                "revision": "f" * 40,
                "weightsSha256": "9" * 64,
                "codeRevision": "a" * 40,
                "license": "CC-BY-NC-4.0",
            },
            "extractor": {"id": "facebook/vjepa2-vitg-fpc64-256"},
            "inferenceMode": "vision-only",
            "modalitiesUsed": ["video"],
            "missingModalities": ["audio", "text"],
            "surface": {"space": "fsaverage5", "mappingId": "destrieux/1", "vertexCount": 20484},
            "tensor": {
                "sha256": "b" * 64,
                "frameCount": 6,
                "dtype": "float32",
                "layout": "time-major",
                "hemodynamicOffsetSeconds": 5.0,
            },
            "atlas": {"name": "Destrieux et al. 2010 cortical parcellation"},
            "excludedAtlasLabels": [],
        },
        "units": {"response": "TRIBE-v2 predicted BOLD (training-target z-score units)"},
        "frames": frames,
        "clip": {"startTime": 0.0, "endTime": 8.94, "coveredDuration": 8.94},
        "phases": [
            {"id": "early", "startTime": 0.0, "endTime": 2.98, "coveredDuration": 2.98, "responseMagnitude": 0.425, "signedMean": 0.0175},
            {"id": "middle", "startTime": 2.98, "endTime": 5.96, "coveredDuration": 2.98, "responseMagnitude": 0.5125, "signedMean": 0.0375},
            {"id": "late", "startTime": 5.96, "endTime": 8.94, "coveredDuration": 2.98, "responseMagnitude": 0.6075, "signedMean": 0.0625},
        ],
        "parcels": parcels,
    }
    document.update(overrides)
    return document
