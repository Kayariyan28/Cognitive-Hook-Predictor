"""Pinned Audio Spectrogram Transformer adapter for descriptive sound labels.

AST supplies AudioSet label evidence only. Its label probabilities are not
speech transcripts, music analysis, audience probabilities, or viral scores.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from ..worker_protocol import (
    WORKER_OUTPUT_SCHEMA_VERSION,
    BranchOutput,
    WorkerIdentity,
    validate_branch_output,
)
from .measured_audio import FfmpegPcmDecoder, PcmDecoder, SAMPLE_RATE


MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
MODEL_REVISION = "f826b80d28226b62986cc218e5cec390b1096902"
MODEL_WEIGHTS_SHA256 = (
    "ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d"
)
MODEL_LICENSE = "BSD-3-Clause"
ADAPTER_ID = "ast-audioset-pytorch"
# Protocol v1 names this 40-character field ``codeRevision``. For this local
# non-Git wrapper it is the SHA-1 content identity of the exact imported source.
WORKER_CODE_REVISION = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()
PREPROCESSING_ID = "ffmpeg-f32le-mono-16khz-10s-nonoverlap-autofeatureextractor/1"
PREPROCESSING_SHA256 = (
    "35dbc5752e043687e6d1353079adfd5e7b21562e7b1d0c0350d61bc17037da57"
)
SNAPSHOT_ENV = "FORECAST_AST_SNAPSHOT"
FFMPEG_ENV = "FORECAST_AUDIO_FFMPEG_BINARY"
WINDOW_SECONDS = 10.0
TOP_LABELS = 5


class AstAudioUnavailable(RuntimeError):
    """The pinned AST snapshot/runtime cannot produce valid evidence."""


class AstBackend(Protocol):
    def classify(self, samples: np.ndarray) -> Sequence[tuple[str, float]]:
        ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _default_snapshot() -> Path:
    huggingface_home = Path(
        os.environ.get(
            "HF_HOME",
            str(Path.home() / ".cache" / "huggingface"),
        )
    )
    return (
        huggingface_home
        / "hub"
        / "models--MIT--ast-finetuned-audioset-10-10-0.4593"
        / "snapshots"
        / MODEL_REVISION
    )


class TransformersAstBackend:
    def __init__(self, snapshot: Path) -> None:
        self.snapshot = snapshot
        self._feature_extractor: Any = None
        self._model: Any = None
        self._torch: Any = None

    @staticmethod
    def installed() -> bool:
        return (
            importlib.util.find_spec("transformers") is not None
            and importlib.util.find_spec("torch") is not None
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            self._feature_extractor = AutoFeatureExtractor.from_pretrained(
                str(self.snapshot), local_files_only=True
            )
            self._model = AutoModelForAudioClassification.from_pretrained(
                str(self.snapshot), local_files_only=True
            )
            self._model.eval()
            self._torch = torch
        except Exception as exc:
            raise AstAudioUnavailable(
                "The local Transformers runtime could not load the pinned AST snapshot."
            ) from exc

    def classify(self, samples: np.ndarray) -> tuple[tuple[str, float], ...]:
        self._load()
        try:
            inputs = self._feature_extractor(
                samples,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            )
            with self._torch.inference_mode():
                logits = self._model(**inputs).logits[0]
                # AudioSet is multi-label. Independent sigmoid scores are the
                # honest model output; they are not calibration probabilities.
                probabilities = self._torch.sigmoid(logits)
                values, indices = self._torch.topk(
                    probabilities, k=min(TOP_LABELS, probabilities.shape[-1])
                )
            id_to_label = self._model.config.id2label
            return tuple(
                (
                    str(id_to_label.get(int(index), id_to_label.get(str(int(index)), int(index)))),
                    float(value),
                )
                for value, index in zip(values.cpu().tolist(), indices.cpu().tolist())
            )
        except Exception as exc:
            raise AstAudioUnavailable("AST audio classification failed.") from exc


def _validated_labels(
    values: Sequence[tuple[str, float]],
) -> tuple[tuple[str, float], ...]:
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= TOP_LABELS:
        raise AstAudioUnavailable("AST returned an invalid label list.")
    result: list[tuple[str, float]] = []
    for label, probability in values:
        if (
            not isinstance(label, str)
            or not label.strip()
            or label != label.strip()
            or len(label) > 200
            or any(ord(character) < 32 for character in label)
            or not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not math.isfinite(float(probability))
            or not 0 <= float(probability) <= 1
        ):
            raise AstAudioUnavailable("AST returned an invalid label value.")
        result.append((label, float(probability)))
    return tuple(result)


class AstAudioAdapter:
    branch = "audioModel"

    def __init__(
        self,
        snapshot: Path,
        *,
        backend: AstBackend | None = None,
        decoder: PcmDecoder | None = None,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
    ) -> None:
        self.snapshot = snapshot.resolve()
        self.backend = backend
        self.decoder = decoder or FfmpegPcmDecoder(
            binary=ffmpeg_binary, probe_binary=ffprobe_binary
        )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "AstAudioAdapter":
        source = os.environ if environ is None else environ
        raw_snapshot = str(source.get(SNAPSHOT_ENV, "")).strip()
        snapshot = Path(raw_snapshot).expanduser() if raw_snapshot else _default_snapshot()
        binary = str(source.get(FFMPEG_ENV, "ffmpeg")).strip() or "ffmpeg"
        probe_binary = str(
            source.get("FORECAST_AUDIO_FFPROBE_BINARY", "ffprobe")
        ).strip() or "ffprobe"
        return cls(
            snapshot, ffmpeg_binary=binary, ffprobe_binary=probe_binary
        )

    def availability(self) -> dict[str, Any]:
        snapshot_present = self.snapshot.is_dir() and (
            self.snapshot / "model.safetensors"
        ).is_file()
        runtime_present = self.backend is not None or TransformersAstBackend.installed()
        reason = None
        if not snapshot_present:
            reason = "The exact pinned AST AudioSet snapshot is not present locally."
        elif not runtime_present:
            reason = "The pinned AST snapshot is present, but Torch/Transformers is unavailable."
        return {
            "configured": snapshot_present,
            "executionAvailable": snapshot_present and runtime_present,
            "reason": reason,
            "role": "ast-audioset-descriptive-labels",
            "usesLearnedModel": True,
            "isSpeechTranscript": False,
            "isBehavioralModel": False,
            "provenance": {
                "modelId": MODEL_ID,
                "modelRevision": MODEL_REVISION,
                "modelWeightsSha256": MODEL_WEIGHTS_SHA256,
                "license": MODEL_LICENSE,
                "adapterId": ADAPTER_ID,
                "codeRevision": WORKER_CODE_REVISION,
                "preprocessingId": PREPROCESSING_ID,
                "preprocessingSha256": PREPROCESSING_SHA256,
            },
        }

    def _verify_snapshot(self) -> None:
        required = (
            self.snapshot / "model.safetensors",
            self.snapshot / "config.json",
            self.snapshot / "preprocessor_config.json",
        )
        if any(not path.is_file() for path in required):
            raise AstAudioUnavailable("The exact pinned AST snapshot is incomplete.")
        try:
            digest = _sha256_file(required[0])
        except OSError as exc:
            raise AstAudioUnavailable("The pinned AST weights could not be read.") from exc
        if digest != MODEL_WEIGHTS_SHA256:
            raise AstAudioUnavailable("The cached AST weights failed the pinned SHA-256 check.")

    def run(
        self,
        *,
        video_path: Path,
        input_sha256: str,
        duration_seconds: float,
        context: Mapping[str, Any],
    ) -> BranchOutput:
        del context
        if not video_path.is_file():
            raise AstAudioUnavailable("The input video is unavailable.")
        if _sha256_file(video_path) != input_sha256:
            raise ValueError("AST input hash does not match the uploaded video")
        if not 10 <= float(duration_seconds) <= 60:
            raise ValueError("AST input duration must be from 10 to 60 seconds")
        self._verify_snapshot()
        samples = self.decoder.decode(video_path)
        if not isinstance(samples, np.ndarray) or samples.ndim != 1 or samples.size == 0:
            raise AstAudioUnavailable("The clip has no decodable audio for AST.")
        if not np.isfinite(samples).all():
            raise AstAudioUnavailable("Decoded AST input contains non-finite samples.")

        backend = self.backend or TransformersAstBackend(self.snapshot)
        started_at = _utc_now()
        window_samples = round(WINDOW_SECONDS * SAMPLE_RATE)
        observations: list[dict[str, Any]] = []
        top_confidences: list[float] = []
        for index, start_sample in enumerate(range(0, samples.size, window_samples)):
            window = samples[start_sample : start_sample + window_samples]
            labels = _validated_labels(backend.classify(window.astype(np.float32, copy=False)))
            start_time = start_sample / SAMPLE_RATE
            end_time = min(samples.size / SAMPLE_RATE, (start_sample + window_samples) / SAMPLE_RATE)
            if end_time <= start_time:
                continue
            top_confidences.append(labels[0][1])
            observations.append(
                {
                    "kind": "ast-audioset-window-labels",
                    "startTime": round(start_time, 6),
                    "endTime": round(end_time, 6),
                    "text": json.dumps(
                        [
                            {"label": label, "modelScore": probability}
                            for label, probability in labels
                        ],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "labels": [label for label, _ in labels],
                }
            )
        if not observations:
            raise AstAudioUnavailable("AST returned no valid descriptive windows.")

        identity = WorkerIdentity(
            branch=self.branch,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            model_weights_sha256=MODEL_WEIGHTS_SHA256,
            adapter_id=ADAPTER_ID,
            code_revision=WORKER_CODE_REVISION,
            preprocessing_id=PREPROCESSING_ID,
            preprocessing_sha256=PREPROCESSING_SHA256,
        )
        output = {
            "schemaVersion": WORKER_OUTPUT_SCHEMA_VERSION,
            "branch": self.branch,
            "inputSha256": input_sha256,
            "startedAt": started_at,
            "completedAt": _utc_now(),
            "evidenceKind": "learned-audioset-label-evidence",
            "features": {
                "audio.ast.windows_classified": float(len(observations)),
                "audio.ast.mean_top_label_score": float(np.mean(top_confidences)),
            },
            "observations": observations,
            "warnings": [
                "AST returns uncalibrated AudioSet sound-label scores only; it is not a transcript, behavioral model, retention model, or virality predictor."
            ],
            "provenance": identity.public_value(),
            "behavioralOutcome": False,
        }
        return validate_branch_output(
            output,
            expected_branch=self.branch,
            expected_input_sha256=input_sha256,
            identity=identity,
        )
