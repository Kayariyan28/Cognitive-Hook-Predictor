"""Pinned local V-JEPA 2.1 Base descriptive-feature worker.

This module deliberately does not download code or weights.  It accepts an
official V-JEPA 2.1 checkout and checkpoint supplied by the operator, verifies
both before importing model code, and always constructs the upstream model
with ``pretrained=False``.  That last invariant avoids the upstream commit's
temporary ``localhost:8300`` pretrained URL.

The worker emits representation statistics only.  It has no behavioral head
and cannot produce attention, retention, engagement, or virality values.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence, TextIO

from ..worker_protocol import (
    WORKER_OUTPUT_SCHEMA_VERSION,
    WORKER_REQUEST_SCHEMA_VERSION,
    WORKER_STATUS_SCHEMA_VERSION,
    WorkerIdentity,
    utc_now,
    validate_branch_output,
)


OFFICIAL_CODE_REVISION = "204698b45b3712590f06245fbfba32d3be539812"
OFFICIAL_REPOSITORY_URL = "https://github.com/facebookresearch/vjepa2"
OFFICIAL_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/vjepa2/"
    "vjepa2_1_vitb_dist_vitG_384.pt"
)
BRANCH = "vjepa21"
MODEL_ID = "facebookresearch/vjepa2.1-vit-base-384"
MODEL_REVISION = OFFICIAL_CODE_REVISION
MODEL_LICENSE = "MIT"
ADAPTER_ID = "vjepa21-pytorch"
EVIDENCE_KIND = "learned-visual-representation"

FRAME_COUNT = 64
SAMPLE_FPS = 4.0
WINDOW_SECONDS = FRAME_COUNT / SAMPLE_FPS
CROP_SIZE = 384
# This is the exact integer calculation in the official eval transform.
RESIZE_SHORT_SIDE = int(CROP_SIZE * 256 / 224)
PATCH_SIZE = 16
TUBELET_SIZE = 2
TEMPORAL_TOKEN_COUNT = FRAME_COUNT // TUBELET_SIZE
SPATIAL_TOKEN_COUNT = (CROP_SIZE // PATCH_SIZE) ** 2
EXPECTED_TOKEN_COUNT = TEMPORAL_TOKEN_COUNT * SPATIAL_TOKEN_COUNT
EXPECTED_EMBED_DIM = 768
RGB_FRAME_BYTES = CROP_SIZE * CROP_SIZE * 3
MAX_REQUEST_BYTES = 256 * 1024
HASH_CHUNK_BYTES = 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEVICE_RE = re.compile(r"^(?:cpu|mps|cuda(?::[0-9]+)?)$")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SOURCE_PATH_ENV = "FORECAST_VJEPA21_SOURCE_PATH"
CHECKPOINT_PATH_ENV = "FORECAST_VJEPA21_CHECKPOINT_PATH"
CHECKPOINT_SHA256_ENV = "FORECAST_VJEPA21_CHECKPOINT_SHA256"
DEVICE_ENV = "FORECAST_VJEPA21_DEVICE"
FFMPEG_BINARY_ENV = "FORECAST_VJEPA21_FFMPEG_BINARY"
FFMPEG_TIMEOUT_ENV = "FORECAST_VJEPA21_FFMPEG_TIMEOUT_SECONDS"

# The pinned upstream tree contains paths that differ only by ASCII case.  A
# default case-insensitive macOS checkout materializes the lowercase blob at
# each uppercase path, so Git reports these nine uppercase YAML paths modified.
# They are data/config files and are not imported by this worker.  No other
# dirty path is tolerated, and each worktree file is byte-compared with its
# corresponding tracked lowercase blob before model code is imported.
CASE_COLLISION_BLOBS = {
    "configs/eval_2_1/vitG-384/coin.yaml": "configs/eval_2_1/vitg-384/coin.yaml",
    "configs/eval_2_1/vitG-384/diving48.yaml": "configs/eval_2_1/vitg-384/diving48.yaml",
    "configs/eval_2_1/vitG-384/ek100.yaml": "configs/eval_2_1/vitg-384/ek100.yaml",
    "configs/eval_2_1/vitG-384/in1k.yaml": "configs/eval_2_1/vitg-384/in1k.yaml",
    "configs/eval_2_1/vitG-384/jester.yaml": "configs/eval_2_1/vitg-384/jester.yaml",
    "configs/eval_2_1/vitG-384/k400.yaml": "configs/eval_2_1/vitg-384/k400.yaml",
    "configs/eval_2_1/vitG-384/ssv2.yaml": "configs/eval_2_1/vitg-384/ssv2.yaml",
    "configs/train_2_1/vitG16/cooldown-256px-64f.yaml": "configs/train_2_1/vitg16/cooldown-256px-64f.yaml",
    "configs/train_2_1/vitG16/pretrain-256px-16f.yaml": "configs/train_2_1/vitg16/pretrain-256px-16f.yaml",
}
EXECUTABLE_SOURCE_PATHS = (
    "hubconf.py",
    "src",
    "app/vjepa_2_1",
    "evals/hub",
    "evals/video_classification_frozen",
)


PREPROCESSING_SPEC = {
    "schemaVersion": "vjepa2.1-preprocessing-contract/1",
    "source": {
        "repository": OFFICIAL_REPOSITORY_URL,
        "revision": OFFICIAL_CODE_REVISION,
        "upstreamTransform": "evals.hub.preprocessor.vjepa2_preprocessor",
    },
    "temporal": {
        "framesPerWindow": FRAME_COUNT,
        "framesPerSecond": SAMPLE_FPS,
        "windowSeconds": WINDOW_SECONDS,
        "sampling": "ffmpeg-fps-round-near",
        "placement": "linear-starts-covering-first-and-last-full-window",
        "shortWindowPadding": "repeat-final-decoded-frame",
    },
    "spatial": {
        "cropSize": CROP_SIZE,
        "resizeShortSide": RESIZE_SHORT_SIDE,
        "resizeInterpolation": "bilinear",
        "crop": "center",
        "pixelFormat": "rgb24",
    },
    "normalization": {
        "scale": 255.0,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
    "tensor": {"layout": "BCTHW", "dtype": "float32"},
    "representation": {
        "tokenGrid": [
            TEMPORAL_TOKEN_COUNT,
            CROP_SIZE // PATCH_SIZE,
            CROP_SIZE // PATCH_SIZE,
        ],
        "temporalPooling": "spatial-mean-per-tubelet",
        "windowPooling": "mean-of-temporal-tokens",
        "change": "one-minus-cosine-similarity",
    },
}
PREPROCESSING_ID = "vjepa2.1-64f4fps-384-ffmpeg/1"
PREPROCESSING_SHA256 = hashlib.sha256(
    json.dumps(
        PREPROCESSING_SPEC,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


class VJepa21WorkerError(RuntimeError):
    """A V-JEPA 2.1 worker invariant was not satisfied."""

    def __init__(self, message: str, *, code: str = "vjepa21_worker_error") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class VJepa21Config:
    source_path: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    device: str = "cpu"
    ffmpeg_binary: str = "ffmpeg"
    ffmpeg_timeout_seconds: float = 120.0
    git_binary: str = "git"

    @classmethod
    def from_values(
        cls,
        *,
        source_path: str | os.PathLike[str] | None,
        checkpoint_path: str | os.PathLike[str] | None,
        checkpoint_sha256: str | None,
        device: str = "cpu",
        ffmpeg_binary: str = "ffmpeg",
        ffmpeg_timeout_seconds: float = 120.0,
        git_binary: str = "git",
    ) -> "VJepa21Config":
        if source_path is None or not str(source_path).strip():
            raise VJepa21WorkerError(
                f"{SOURCE_PATH_ENV} or --source-tree is required",
                code="worker_unconfigured",
            )
        if checkpoint_path is None or not str(checkpoint_path).strip():
            raise VJepa21WorkerError(
                f"{CHECKPOINT_PATH_ENV} or --checkpoint is required",
                code="worker_unconfigured",
            )
        digest = str(checkpoint_sha256 or "").strip().lower()
        if not SHA256_RE.fullmatch(digest):
            raise VJepa21WorkerError(
                f"{CHECKPOINT_SHA256_ENV} or --checkpoint-sha256 must be a SHA-256 digest",
                code="worker_unconfigured",
            )
        normalized_device = str(device).strip().lower()
        if not DEVICE_RE.fullmatch(normalized_device):
            raise VJepa21WorkerError(
                "device must be cpu, mps, cuda, or cuda:<index>",
                code="worker_configuration_invalid",
            )
        ffmpeg = _safe_command_name(ffmpeg_binary, "ffmpeg binary")
        git = _safe_command_name(git_binary, "git binary")
        timeout = _finite(ffmpeg_timeout_seconds, "ffmpeg timeout")
        if timeout < 5 or timeout > 600:
            raise VJepa21WorkerError(
                "ffmpeg timeout must be from 5 to 600 seconds",
                code="worker_configuration_invalid",
            )
        return cls(
            source_path=Path(source_path).expanduser().resolve(),
            checkpoint_path=Path(checkpoint_path).expanduser().resolve(),
            checkpoint_sha256=digest,
            device=normalized_device,
            ffmpeg_binary=ffmpeg,
            ffmpeg_timeout_seconds=timeout,
            git_binary=git,
        )

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> "VJepa21Config":
        values = os.environ if environ is None else environ
        raw_timeout = values.get(FFMPEG_TIMEOUT_ENV, "120")
        try:
            timeout = float(raw_timeout)
        except (TypeError, ValueError) as exc:
            raise VJepa21WorkerError(
                f"{FFMPEG_TIMEOUT_ENV} must be numeric",
                code="worker_configuration_invalid",
            ) from exc
        return cls.from_values(
            source_path=values.get(SOURCE_PATH_ENV),
            checkpoint_path=values.get(CHECKPOINT_PATH_ENV),
            checkpoint_sha256=values.get(CHECKPOINT_SHA256_ENV),
            device=values.get(DEVICE_ENV, "cpu"),
            ffmpeg_binary=values.get(FFMPEG_BINARY_ENV, "ffmpeg"),
            ffmpeg_timeout_seconds=timeout,
        )


@dataclass(frozen=True, slots=True)
class WindowPlan:
    index: int
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass(frozen=True, slots=True)
class WindowEvidence:
    plan: WindowPlan
    pooled_embedding: tuple[float, ...]
    embedding_norm: float
    temporal_consistency: float
    temporal_change_mean: float
    temporal_change_peak: float
    padded_frame_count: int = 0


CommandRunner = Callable[..., Any]
WindowExecutor = Callable[[WindowPlan], WindowEvidence]
Clock = Callable[[], str]


def _safe_command_name(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise VJepa21WorkerError(
            f"{label} must be a non-empty command or path",
            code="worker_configuration_invalid",
        )
    return value


def _finite(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise VJepa21WorkerError(
            f"{label} must be finite", code="worker_configuration_invalid"
        )
    return float(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(HASH_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as exc:
        raise VJepa21WorkerError(
            "an artifact or input file could not be read",
            code="worker_file_unreadable",
        ) from exc
    return digest.hexdigest()


def _strict_json_object(payload: str, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise VJepa21WorkerError(
                    f"{label} contains duplicate key {key!r}",
                    code="worker_request_invalid",
                )
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=pairs)
    except VJepa21WorkerError:
        raise
    except json.JSONDecodeError as exc:
        raise VJepa21WorkerError(
            f"{label} is not valid JSON", code="worker_request_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise VJepa21WorkerError(
            f"{label} must be a JSON object", code="worker_request_invalid"
        )
    return value


def validate_worker_request(
    value: Any, *, actual_input_sha256: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VJepa21WorkerError(
            "worker request must be an object", code="worker_request_invalid"
        )
    required = {
        "schemaVersion",
        "requestId",
        "branch",
        "inputSha256",
        "durationSeconds",
        "context",
    }
    if set(value) != required:
        raise VJepa21WorkerError(
            "worker request has missing or unknown fields",
            code="worker_request_invalid",
        )
    if value["schemaVersion"] != WORKER_REQUEST_SCHEMA_VERSION:
        raise VJepa21WorkerError(
            "worker request schemaVersion is unsupported",
            code="worker_request_invalid",
        )
    if value["branch"] != BRANCH:
        raise VJepa21WorkerError(
            "worker request branch must be vjepa21", code="worker_request_invalid"
        )
    input_sha256 = value["inputSha256"]
    request_id = value["requestId"]
    if (
        not isinstance(input_sha256, str)
        or not SHA256_RE.fullmatch(input_sha256)
        or request_id != input_sha256
        or input_sha256 != actual_input_sha256
    ):
        raise VJepa21WorkerError(
            "worker request hash is not bound to the uploaded video",
            code="worker_input_hash_mismatch",
        )
    duration = _finite(value["durationSeconds"], "request duration")
    if duration < 10 or duration > 60:
        raise VJepa21WorkerError(
            "worker request duration must be from 10 to 60 seconds",
            code="worker_request_invalid",
        )
    if not isinstance(value["context"], dict):
        raise VJepa21WorkerError(
            "worker request context must be an object",
            code="worker_request_invalid",
        )
    return {
        "schemaVersion": WORKER_REQUEST_SCHEMA_VERSION,
        "requestId": input_sha256,
        "branch": BRANCH,
        "inputSha256": input_sha256,
        "durationSeconds": duration,
        "context": dict(value["context"]),
    }


def plan_windows(duration_seconds: float) -> tuple[WindowPlan, ...]:
    duration = _finite(duration_seconds, "video duration")
    if duration < 10 or duration > 60:
        raise VJepa21WorkerError(
            "video duration must be from 10 to 60 seconds",
            code="worker_request_invalid",
        )
    count = max(1, int(math.ceil(duration / WINDOW_SECONDS)))
    if count == 1:
        return (WindowPlan(index=0, start_time=0.0, end_time=duration),)
    last_start = duration - WINDOW_SECONDS
    plans = []
    for index in range(count):
        start = round(last_start * index / (count - 1), 9)
        end = round(start + WINDOW_SECONDS, 9)
        plans.append(WindowPlan(index=index, start_time=start, end_time=end))
    return tuple(plans)


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise VJepa21WorkerError(
            "pooled embeddings have incompatible shapes",
            code="worker_model_output_invalid",
        )
    numerator = math.fsum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


def summarize_window_evidence(
    windows: Sequence[WindowEvidence],
) -> tuple[dict[str, float], list[dict[str, Any]], list[str]]:
    if not windows:
        raise VJepa21WorkerError(
            "at least one representation window is required",
            code="worker_model_output_invalid",
        )
    for item in windows:
        if len(item.pooled_embedding) != EXPECTED_EMBED_DIM or any(
            not math.isfinite(value) for value in item.pooled_embedding
        ):
            raise VJepa21WorkerError(
                "pooled V-JEPA 2.1 embedding must contain 768 finite values",
                code="worker_model_output_invalid",
            )
    norms = [_finite(item.embedding_norm, "embedding norm") for item in windows]
    consistencies = [
        _finite(item.temporal_consistency, "temporal consistency")
        for item in windows
    ]
    change_means = [
        _finite(item.temporal_change_mean, "temporal change") for item in windows
    ]
    change_peaks = [
        _finite(item.temporal_change_peak, "temporal change peak")
        for item in windows
    ]
    features = {
        "vjepa2_1.embedding_norm_mean": statistics.fmean(norms),
        "vjepa2_1.embedding_norm_std": statistics.pstdev(norms),
        "vjepa2_1.temporal_consistency_mean": statistics.fmean(consistencies),
        "vjepa2_1.temporal_change_mean": statistics.fmean(change_means),
        "vjepa2_1.temporal_change_peak": max(change_peaks),
    }
    # Preserve the learned representation for a future separately trained and
    # calibrated content head.  This is a mean over deterministic windows, not
    # a behavioral value and not a probability.
    for dimension in range(EXPECTED_EMBED_DIM):
        features[f"vjepa2_1.embedding_{dimension:03d}"] = statistics.fmean(
            item.pooled_embedding[dimension] for item in windows
        )
    warnings: list[str] = []
    if len(windows) > 1:
        interwindow_consistency = [
            _cosine_similarity(previous.pooled_embedding, current.pooled_embedding)
            for previous, current in zip(windows, windows[1:])
        ]
        interwindow_change = [1.0 - value for value in interwindow_consistency]
        features.update(
            {
                "vjepa2_1.interwindow_consistency_mean": statistics.fmean(
                    interwindow_consistency
                ),
                "vjepa2_1.interwindow_change_mean": statistics.fmean(
                    interwindow_change
                ),
                "vjepa2_1.interwindow_change_peak": max(interwindow_change),
            }
        )
    else:
        warnings.append(
            "Only one temporal window was available; inter-window representation change is omitted."
        )

    if any(item.padded_frame_count for item in windows):
        warnings.append(
            "A source interval shorter than 16 seconds was padded by repeating its final decoded frame to preserve the 64-frame model input."
        )
    if any(
        current.plan.start_time < previous.plan.end_time
        for previous, current in zip(windows, windows[1:])
    ):
        warnings.append(
            "Temporal windows overlap so the first and last full 16-second model windows both cover the source clip."
        )

    maximum_change = max(change_means)
    minimum_change = min(change_means)
    has_spread = maximum_change - minimum_change > 1e-12
    observations: list[dict[str, Any]] = []
    for item in windows:
        if not has_spread:
            relative = "typical"
        elif item.temporal_change_mean == maximum_change:
            relative = "higher"
        elif item.temporal_change_mean == minimum_change:
            relative = "lower"
        else:
            relative = "typical"
        descriptions = {
            "higher": "Visual representation changed more here than in the clip's other decoded windows.",
            "lower": "Visual representation changed less here than in the clip's other decoded windows.",
            "typical": "Visual representation change here was similar to the clip's other decoded windows.",
        }
        observations.append(
            {
                "kind": "learned-visual-window",
                "startTime": item.plan.start_time,
                "endTime": item.plan.end_time,
                "text": (
                    f"{descriptions[relative]} This is descriptive V-JEPA 2.1 evidence, not attention or predicted performance."
                ),
                "labels": [
                    "vjepa2.1",
                    "descriptive-only",
                    f"relative-change-{relative}",
                ],
            }
        )

    for name, value in features.items():
        if not math.isfinite(value):
            raise VJepa21WorkerError(
                f"feature {name} is not finite", code="worker_model_output_invalid"
            )
    return features, observations, warnings


class VJepa21Worker:
    """In-process V-JEPA 2.1 feature worker used by a CLI or service wrapper."""

    def __init__(
        self,
        config: VJepa21Config,
        *,
        torch_module: Any | None = None,
        command_runner: CommandRunner = subprocess.run,
        clock: Clock = utc_now,
        window_executor: WindowExecutor | None = None,
    ) -> None:
        self.config = config
        self._torch = torch_module
        self._command_runner = command_runner
        self._clock = clock
        self._injected_window_executor = window_executor
        self._encoder: Any | None = None
        self._identity: WorkerIdentity | None = None
        self._ffmpeg_version: str | None = None
        self._last_error: str | None = None

    @property
    def identity(self) -> WorkerIdentity:
        self._ensure_loaded()
        assert self._identity is not None
        return self._identity

    def _status_document(
        self, *, state: str, execution_available: bool, last_error: str | None
    ) -> dict[str, Any]:
        return {
            "schemaVersion": WORKER_STATUS_SCHEMA_VERSION,
            "service": "creator-forecast-model-worker",
            "state": state,
            "executionAvailable": execution_available,
            "branch": BRANCH,
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "weightsSha256": self.config.checkpoint_sha256,
                "adapterId": ADAPTER_ID,
                "license": MODEL_LICENSE,
            },
            "codeRevision": OFFICIAL_CODE_REVISION,
            "preprocessing": {
                "id": PREPROCESSING_ID,
                "sha256": PREPROCESSING_SHA256,
            },
            "device": self.config.device,
            "precision": "fp32",
            "lastError": last_error,
        }

    def status(self) -> dict[str, Any]:
        try:
            self._ensure_loaded()
        except Exception as exc:  # status must remain inspectable when unready
            self._last_error = _safe_error_text(exc)
            return self._status_document(
                state="error", execution_available=False, last_error=self._last_error
            )
        return self._status_document(
            state="ready", execution_available=True, last_error=None
        )

    def _run_text(
        self, command: list[str], *, label: str, timeout: float = 15.0
    ) -> str:
        try:
            result = self._command_runner(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise VJepa21WorkerError(
                f"{label} could not be executed", code="worker_dependency_unavailable"
            ) from exc
        if result.returncode != 0:
            raise VJepa21WorkerError(
                f"{label} failed", code="worker_dependency_unavailable"
            )
        # Preserve leading status-column spaces from ``git status --porcelain``.
        # Only subprocess line terminators are removed.
        return str(result.stdout).rstrip("\r\n")

    def _run_bytes(
        self, command: list[str], *, label: str, timeout: float = 15.0
    ) -> bytes:
        try:
            result = self._command_runner(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                check=False,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise VJepa21WorkerError(
                f"{label} could not be executed", code="worker_dependency_unavailable"
            ) from exc
        if result.returncode != 0:
            raise VJepa21WorkerError(
                f"{label} failed", code="worker_source_invalid"
            )
        return bytes(result.stdout)

    def _verify_case_collision_blobs(self) -> None:
        source = self.config.source_path
        for worktree_relative, tracked_lowercase in CASE_COLLISION_BLOBS.items():
            worktree_path = source / worktree_relative
            if worktree_path.is_symlink() or not worktree_path.is_file():
                raise VJepa21WorkerError(
                    "a tolerated V-JEPA 2.1 case-collision path is not a regular file",
                    code="worker_source_dirty",
                )
            expected = self._run_bytes(
                [
                    self.config.git_binary,
                    "-C",
                    str(source),
                    "cat-file",
                    "blob",
                    f"HEAD:{tracked_lowercase}",
                ],
                label="git case-collision blob verification",
            )
            try:
                actual = worktree_path.read_bytes()
            except OSError as exc:
                raise VJepa21WorkerError(
                    "a tolerated V-JEPA 2.1 case-collision file could not be read",
                    code="worker_source_dirty",
                ) from exc
            if actual != expected:
                raise VJepa21WorkerError(
                    "a tolerated V-JEPA 2.1 case-collision file does not match its tracked lowercase blob",
                    code="worker_source_dirty",
                )

    def _verify_source_checkout(self) -> None:
        source = self.config.source_path
        if not source.is_dir():
            raise VJepa21WorkerError(
                "V-JEPA 2.1 source path is not a directory",
                code="worker_source_invalid",
            )
        for relative in ("hubconf.py", "src/hub/backbones.py"):
            if not (source / relative).is_file():
                raise VJepa21WorkerError(
                    "V-JEPA 2.1 source checkout is incomplete",
                    code="worker_source_invalid",
                )
        revision = self._run_text(
            [
                self.config.git_binary,
                "-C",
                str(source),
                "rev-parse",
                "--verify",
                "HEAD^{commit}",
            ],
            label="git revision verification",
        ).lower()
        if revision != OFFICIAL_CODE_REVISION:
            raise VJepa21WorkerError(
                f"V-JEPA 2.1 source must be checked out at {OFFICIAL_CODE_REVISION}",
                code="worker_source_revision_mismatch",
            )
        # Explicitly verify every tracked path that can execute during local
        # hub construction and preprocessing.  Known YAML collisions live
        # outside this set.
        self._run_text(
            [
                self.config.git_binary,
                "-C",
                str(source),
                "diff",
                "--quiet",
                "HEAD",
                "--",
                *EXECUTABLE_SOURCE_PATHS,
            ],
            label="git executable source verification",
        )
        dirty = self._run_text(
            [
                self.config.git_binary,
                "-C",
                str(source),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            label="git cleanliness verification",
        )
        if not dirty:
            return
        dirty_lines = dirty.splitlines()
        expected_lines = {
            f" M {relative}" for relative in CASE_COLLISION_BLOBS
        }
        if len(dirty_lines) != len(expected_lines) or set(dirty_lines) != expected_lines:
            raise VJepa21WorkerError(
                "V-JEPA 2.1 source checkout has changes outside the exact case-collision allowlist",
                code="worker_source_dirty",
            )
        self._verify_case_collision_blobs()

    def _verify_checkpoint(self) -> None:
        checkpoint = self.config.checkpoint_path
        if not checkpoint.is_file():
            raise VJepa21WorkerError(
                "V-JEPA 2.1 checkpoint is not a readable file",
                code="worker_checkpoint_missing",
            )
        actual = _sha256_file(checkpoint)
        if actual != self.config.checkpoint_sha256:
            raise VJepa21WorkerError(
                "V-JEPA 2.1 checkpoint failed its configured SHA-256 check",
                code="worker_checkpoint_hash_mismatch",
            )

    def _verify_ffmpeg(self) -> None:
        output = self._run_text(
            [self.config.ffmpeg_binary, "-version"],
            label="ffmpeg version probe",
            timeout=10.0,
        )
        first_line = output.splitlines()[0] if output else ""
        if not first_line.lower().startswith("ffmpeg version "):
            raise VJepa21WorkerError(
                "configured ffmpeg binary did not identify itself",
                code="worker_dependency_unavailable",
            )
        self._ffmpeg_version = first_line

    def _torch_runtime(self) -> Any:
        if self._torch is None:
            try:
                self._torch = importlib.import_module("torch")
            except (ImportError, OSError) as exc:
                raise VJepa21WorkerError(
                    "PyTorch is not installed for the V-JEPA 2.1 worker",
                    code="worker_dependency_unavailable",
                ) from exc
        return self._torch

    def _verify_device(self, torch: Any) -> None:
        requested = self.config.device
        if requested == "cpu":
            return
        if requested == "mps":
            backend = getattr(getattr(torch, "backends", None), "mps", None)
            if (
                backend is None
                or not bool(backend.is_built())
                or not bool(backend.is_available())
            ):
                raise VJepa21WorkerError(
                    "MPS was requested but is not built and available",
                    code="worker_device_unavailable",
                )
            return
        cuda = getattr(torch, "cuda", None)
        if cuda is None or not bool(cuda.is_available()):
            raise VJepa21WorkerError(
                "CUDA was requested but is not available",
                code="worker_device_unavailable",
            )
        if ":" in requested:
            index = int(requested.split(":", 1)[1])
            if index >= int(cuda.device_count()):
                raise VJepa21WorkerError(
                    "requested CUDA device index does not exist",
                    code="worker_device_unavailable",
                )

    @staticmethod
    def _clean_encoder_state(raw_state: Any) -> dict[str, Any]:
        if not isinstance(raw_state, Mapping) or not raw_state:
            raise VJepa21WorkerError(
                "checkpoint ema_encoder is missing or invalid",
                code="worker_checkpoint_invalid",
            )
        cleaned: dict[str, Any] = {}
        for raw_key, value in raw_state.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise VJepa21WorkerError(
                    "checkpoint ema_encoder contains an invalid key",
                    code="worker_checkpoint_invalid",
                )
            key = raw_key.replace("module.", "").replace("backbone.", "")
            if key in cleaned:
                raise VJepa21WorkerError(
                    "checkpoint key cleaning produced a duplicate",
                    code="worker_checkpoint_invalid",
                )
            cleaned[key] = value
        return cleaned

    def _load_model(self, torch: Any) -> Any:
        try:
            loaded = torch.hub.load(
                str(self.config.source_path),
                "vjepa2_1_vit_base_384",
                source="local",
                pretrained=False,
            )
        except Exception as exc:
            raise VJepa21WorkerError(
                "official V-JEPA 2.1 Base construction failed",
                code="worker_model_load_failed",
            ) from exc
        if not isinstance(loaded, (tuple, list)) or len(loaded) != 2:
            raise VJepa21WorkerError(
                "official V-JEPA 2.1 loader returned an unexpected model tuple",
                code="worker_model_load_failed",
            )
        encoder, _predictor = loaded
        try:
            checkpoint = torch.load(
                str(self.config.checkpoint_path),
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise VJepa21WorkerError(
                "V-JEPA 2.1 checkpoint could not be loaded safely",
                code="worker_checkpoint_invalid",
            ) from exc
        if not isinstance(checkpoint, Mapping) or "ema_encoder" not in checkpoint:
            raise VJepa21WorkerError(
                "V-JEPA 2.1 checkpoint does not contain ema_encoder",
                code="worker_checkpoint_invalid",
            )
        cleaned = self._clean_encoder_state(checkpoint["ema_encoder"])
        try:
            encoder.load_state_dict(cleaned, strict=True)
            encoder.eval()
            encoder.to(self.config.device)
        except Exception as exc:
            raise VJepa21WorkerError(
                "V-JEPA 2.1 ema_encoder failed strict loading",
                code="worker_checkpoint_invalid",
            ) from exc
        return encoder

    def _ensure_loaded(self) -> None:
        if self._encoder is not None:
            return
        self._verify_source_checkout()
        self._verify_checkpoint()
        self._verify_ffmpeg()
        torch = self._torch_runtime()
        self._verify_device(torch)
        encoder = self._load_model(torch)
        self._encoder = encoder
        self._identity = WorkerIdentity(
            branch=BRANCH,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            model_weights_sha256=self.config.checkpoint_sha256,
            adapter_id=ADAPTER_ID,
            code_revision=OFFICIAL_CODE_REVISION,
            preprocessing_id=PREPROCESSING_ID,
            preprocessing_sha256=PREPROCESSING_SHA256,
        )
        self._last_error = None

    def _ffmpeg_command(self, video_path: Path, plan: WindowPlan) -> list[str]:
        resize = (
            "scale="
            f"w='if(gte(iw,ih),-2,{RESIZE_SHORT_SIDE})':"
            f"h='if(gte(iw,ih),{RESIZE_SHORT_SIDE},-2)':flags=bilinear"
        )
        crop = (
            f"crop=w={CROP_SIZE}:h={CROP_SIZE}:"
            f"x=(in_w-{CROP_SIZE})/2:y=(in_h-{CROP_SIZE})/2"
        )
        filters = f"fps=fps={SAMPLE_FPS:g}:round=near,{resize},{crop}"
        return [
            self.config.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-threads",
            "1",
            "-i",
            str(video_path),
            "-ss",
            f"{plan.start_time:.9f}",
            "-t",
            f"{plan.duration:.9f}",
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-filter_threads",
            "1",
            "-vf",
            filters,
            "-frames:v",
            str(FRAME_COUNT),
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]

    def _decode_window(self, video_path: Path, plan: WindowPlan) -> tuple[bytes, int]:
        try:
            result = self._command_runner(
                self._ffmpeg_command(video_path, plan),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                check=False,
                timeout=self.config.ffmpeg_timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise VJepa21WorkerError(
                "ffmpeg frame extraction could not be executed",
                code="worker_decode_failed",
            ) from exc
        if result.returncode != 0:
            raise VJepa21WorkerError(
                "ffmpeg could not decode the requested temporal window",
                code="worker_decode_failed",
            )
        payload = bytes(result.stdout)
        if not payload or len(payload) % RGB_FRAME_BYTES:
            raise VJepa21WorkerError(
                "ffmpeg returned an invalid raw RGB frame payload",
                code="worker_decode_failed",
            )
        decoded_frames = len(payload) // RGB_FRAME_BYTES
        if decoded_frames > FRAME_COUNT:
            raise VJepa21WorkerError(
                "ffmpeg returned more than 64 frames",
                code="worker_decode_failed",
            )
        padded = FRAME_COUNT - decoded_frames
        if padded:
            final_frame = payload[-RGB_FRAME_BYTES:]
            payload += final_frame * padded
        if len(payload) != FRAME_COUNT * RGB_FRAME_BYTES:
            raise VJepa21WorkerError(
                "decoded window did not resolve to exactly 64 frames",
                code="worker_decode_failed",
            )
        return payload, padded

    def _preprocess(self, raw_rgb: bytes) -> Any:
        torch = self._torch_runtime()
        if len(raw_rgb) != FRAME_COUNT * RGB_FRAME_BYTES:
            raise VJepa21WorkerError(
                "preprocessor requires exactly 64 RGB frames",
                code="worker_decode_failed",
            )
        # bytearray gives torch a writable buffer and prevents undefined writes
        # against an immutable bytes view.
        frames = torch.frombuffer(bytearray(raw_rgb), dtype=torch.uint8)
        frames = frames.reshape(FRAME_COUNT, CROP_SIZE, CROP_SIZE, 3)
        tensor = frames.permute(3, 0, 1, 2).contiguous()
        tensor = tensor.to(dtype=torch.float32).div_(255.0)
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1, 1)
        tensor = tensor.sub_(mean).div_(std)
        return tensor.unsqueeze(0)

    def _encode_window(self, video_path: Path, plan: WindowPlan) -> WindowEvidence:
        assert self._encoder is not None
        torch = self._torch_runtime()
        raw_rgb, padded = self._decode_window(video_path, plan)
        model_input = self._preprocess(raw_rgb).to(self.config.device)
        try:
            with torch.inference_mode():
                tokens = self._encoder(model_input)
        except Exception as exc:
            raise VJepa21WorkerError(
                "V-JEPA 2.1 encoder inference failed",
                code="worker_inference_failed",
            ) from exc
        shape = tuple(int(value) for value in getattr(tokens, "shape", ()))
        if shape != (1, EXPECTED_TOKEN_COUNT, EXPECTED_EMBED_DIM):
            raise VJepa21WorkerError(
                "V-JEPA 2.1 encoder returned an unexpected token tensor shape",
                code="worker_model_output_invalid",
            )
        temporal = tokens[0].reshape(
            TEMPORAL_TOKEN_COUNT, SPATIAL_TOKEN_COUNT, EXPECTED_EMBED_DIM
        ).mean(dim=1)
        pooled = temporal.mean(dim=0)
        temporal_norms = torch.linalg.vector_norm(temporal, dim=-1).clamp_min(1e-12)
        pairwise = (temporal[:-1] * temporal[1:]).sum(dim=-1) / (
            temporal_norms[:-1] * temporal_norms[1:]
        )
        pairwise = pairwise.clamp(-1.0, 1.0)
        changes = 1.0 - pairwise
        pooled_cpu = pooled.detach().to(device="cpu", dtype=torch.float32)
        embedding = tuple(float(value) for value in pooled_cpu.tolist())
        evidence = WindowEvidence(
            plan=plan,
            pooled_embedding=embedding,
            embedding_norm=float(torch.linalg.vector_norm(pooled).item()),
            temporal_consistency=float(pairwise.mean().item()),
            temporal_change_mean=float(changes.mean().item()),
            temporal_change_peak=float(changes.max().item()),
            padded_frame_count=padded,
        )
        for value in (
            evidence.embedding_norm,
            evidence.temporal_consistency,
            evidence.temporal_change_mean,
            evidence.temporal_change_peak,
            *evidence.pooled_embedding,
        ):
            if not math.isfinite(value):
                raise VJepa21WorkerError(
                    "V-JEPA 2.1 encoder returned a non-finite representation",
                    code="worker_model_output_invalid",
                )
        return evidence

    def extract(
        self,
        *,
        video_path: str | os.PathLike[str],
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        source_video = Path(video_path).expanduser().resolve()
        if not source_video.is_file():
            raise VJepa21WorkerError(
                "uploaded video is not a readable file", code="worker_input_missing"
            )
        actual_sha256 = _sha256_file(source_video)
        normalized_request = validate_worker_request(
            dict(request), actual_input_sha256=actual_sha256
        )
        # Input and request binding is deliberately checked before any model is
        # loaded or GPU work is admitted.
        started_at = self._clock()
        self._ensure_loaded()
        plans = plan_windows(normalized_request["durationSeconds"])
        executor = self._injected_window_executor
        windows = [
            executor(plan) if executor is not None else self._encode_window(source_video, plan)
            for plan in plans
        ]
        for plan, evidence in zip(plans, windows):
            if evidence.plan != plan:
                raise VJepa21WorkerError(
                    "window executor changed the requested interval",
                    code="worker_model_output_invalid",
                )
        features, observations, warnings = summarize_window_evidence(windows)
        completed_at = self._clock()
        identity = self.identity
        document = {
            "schemaVersion": WORKER_OUTPUT_SCHEMA_VERSION,
            "branch": BRANCH,
            "inputSha256": actual_sha256,
            "startedAt": started_at,
            "completedAt": completed_at,
            "evidenceKind": EVIDENCE_KIND,
            "features": features,
            "observations": observations,
            "warnings": warnings,
            "provenance": identity.public_value(),
            "behavioralOutcome": False,
        }
        # Self-validation means the CLI cannot publish a document that the
        # orchestrator's strict protocol would reject.
        validate_branch_output(
            document,
            expected_branch=BRANCH,
            expected_input_sha256=actual_sha256,
            identity=identity,
        )
        # Return the wire document, which intentionally keeps observation
        # labels as JSON arrays.  BranchOutput is an internal immutable view.
        return document


def _safe_error_text(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    if not message:
        message = exc.__class__.__name__
    return message[:1024]


def load_request_file(path: str, *, stdin: TextIO = sys.stdin) -> dict[str, Any]:
    try:
        if path == "-":
            payload = stdin.read(MAX_REQUEST_BYTES + 1)
        else:
            request_path = Path(path).expanduser().resolve()
            with request_path.open("r", encoding="utf-8") as handle:
                payload = handle.read(MAX_REQUEST_BYTES + 1)
    except OSError as exc:
        raise VJepa21WorkerError(
            "worker request file could not be read", code="worker_request_invalid"
        ) from exc
    if len(payload.encode("utf-8")) > MAX_REQUEST_BYTES:
        raise VJepa21WorkerError(
            "worker request exceeds the size limit", code="worker_request_invalid"
        )
    return _strict_json_object(payload, "worker request")


def _write_json(stream: TextIO, value: Mapping[str, Any]) -> None:
    stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")))
    stream.write("\n")
    stream.flush()


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-tree", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-sha256", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument("--ffmpeg-timeout-seconds", type=float, default=None)
    return parser


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backend.forecast.workers.vjepa21",
        description="Pinned V-JEPA 2.1 descriptive-feature worker",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = _common_parser()
    subparsers.add_parser("status", parents=[common])
    extract = subparsers.add_parser("extract", parents=[common])
    extract.add_argument("--video", required=True)
    extract.add_argument("--request", required=True, help="JSON file path or - for stdin")
    return parser


def _config_from_cli(
    args: argparse.Namespace, environ: Mapping[str, str]
) -> VJepa21Config:
    raw_timeout: Any = args.ffmpeg_timeout_seconds
    if raw_timeout is None:
        raw_timeout = environ.get(FFMPEG_TIMEOUT_ENV, "120")
    try:
        timeout = float(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise VJepa21WorkerError(
            "ffmpeg timeout must be numeric", code="worker_configuration_invalid"
        ) from exc
    return VJepa21Config.from_values(
        source_path=args.source_tree or environ.get(SOURCE_PATH_ENV),
        checkpoint_path=args.checkpoint or environ.get(CHECKPOINT_PATH_ENV),
        checkpoint_sha256=args.checkpoint_sha256
        or environ.get(CHECKPOINT_SHA256_ENV),
        device=args.device or environ.get(DEVICE_ENV, "cpu"),
        ffmpeg_binary=args.ffmpeg or environ.get(FFMPEG_BINARY_ENV, "ffmpeg"),
        ffmpeg_timeout_seconds=timeout,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    stdin: TextIO = sys.stdin,
    worker_factory: Callable[[VJepa21Config], VJepa21Worker] = VJepa21Worker,
) -> int:
    args = build_cli_parser().parse_args(argv)
    environment = os.environ if environ is None else environ
    try:
        config = _config_from_cli(args, environment)
        worker = worker_factory(config)
        if args.command == "status":
            document = worker.status()
            _write_json(stdout, document)
            return 0
        request = load_request_file(args.request, stdin=stdin)
        document = worker.extract(video_path=args.video, request=request)
        _write_json(stdout, document)
        return 0
    except VJepa21WorkerError as exc:
        _write_json(
            stderr,
            {
                "error": {
                    "code": exc.code,
                    "message": _safe_error_text(exc),
                }
            },
        )
        return 2
    except Exception as exc:  # never mix logs/errors into protocol stdout
        _write_json(
            stderr,
            {
                "error": {
                    "code": "vjepa21_worker_error",
                    "message": _safe_error_text(exc),
                }
            },
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
