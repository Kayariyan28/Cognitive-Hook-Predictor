"""Lazy, serialized access to the official TRIBE v2 inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
import math
import os
from pathlib import Path
import threading
from typing import Any, Sequence

from .artifacts import EXPECTED_HEMODYNAMIC_OFFSET_SECONDS
from .config import (
    Settings,
    TRIBE_MODEL_WEIGHTS_SHA256,
    VIDEO_MODEL_ID,
    VIDEO_MODEL_REVISION,
    VIDEO_MODEL_WEIGHTS_SHA256,
)
from .errors import InferenceError, ModelUnavailableError, PredictionValidationError


@dataclass(frozen=True, slots=True)
class InferenceOutput:
    predictions: Any
    segments: Sequence[Any]
    hemodynamic_offset_seconds: float


class TribeRuntime:
    """Own one official model instance and run one GPU inference at a time."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._hemodynamic_offset_seconds: float | None = None
        self._state = "not_loaded" if settings.is_model_configured else "unconfigured"
        self._last_error: str | None = None
        self._operation_lock = threading.RLock()
        self._state_lock = threading.Lock()

    def _set_state(self, state: str, error: str | None = None) -> None:
        with self._state_lock:
            self._state = state
            self._last_error = error

    def _require_video_device(self, torch_module: Any) -> None:
        """Fail closed when the explicitly selected accelerator is unavailable."""

        if self.settings.video_device == "cpu":
            return
        if self.settings.video_device == "cuda":
            cuda = getattr(torch_module, "cuda", None)
            is_available = getattr(cuda, "is_available", None)
            if not callable(is_available) or not is_available():
                raise ModelUnavailableError(
                    "TRIBE_VIDEO_DEVICE=cuda was requested, but CUDA is unavailable"
                )
            return

        mps = getattr(getattr(torch_module, "backends", None), "mps", None)
        is_built = getattr(mps, "is_built", None)
        is_available = getattr(mps, "is_available", None)
        if (
            mps is None
            or not callable(is_available)
            or not is_available()
            or (callable(is_built) and not is_built())
        ):
            raise ModelUnavailableError(
                "TRIBE_VIDEO_DEVICE=mps was requested, but the PyTorch MPS "
                "backend is unavailable"
            )

    @staticmethod
    def _install_mps_autocast_patch(
        video_model_class: Any, torch_module: Any
    ) -> None:
        """Wrap upstream V-JEPA calls in MPS FP16 autocast, exactly once.

        Neuralset creates its private video-model wrapper lazily inside the
        cached extraction method, so there is no model instance to configure at
        TRIBE construction time. The narrow class patch checks both the actual
        model device and V-JEPA name before enabling autocast; CPU and unrelated
        video models keep the unmodified upstream method.
        """

        marker = "_tribe_mps_autocast_fp16_original_predict"
        if getattr(video_model_class, marker, None) is not None:
            return
        original_predict = video_model_class.predict

        @functools.wraps(original_predict)
        def predict_with_mps_autocast(instance: Any, *args: Any, **kwargs: Any) -> Any:
            model = getattr(instance, "model", None)
            device = getattr(model, "device", None)
            device_type = getattr(device, "type", str(device).split(":", 1)[0])
            model_name = str(getattr(instance, "model_name", "")).lower()
            if device_type != "mps" or "vjepa2" not in model_name:
                return original_predict(instance, *args, **kwargs)
            with torch_module.autocast(
                device_type="mps", dtype=torch_module.float16
            ):
                return original_predict(instance, *args, **kwargs)

        setattr(video_model_class, marker, original_predict)
        video_model_class.predict = predict_with_mps_autocast

    def _configure_video_execution(
        self,
        model: Any,
        video_snapshot_path: Path,
        *,
        torch_module: Any,
        video_model_class: Any | None = None,
    ) -> None:
        """Apply the validated device policy after Neuralset construction."""

        self._require_video_device(torch_module)
        data = getattr(model, "data", None)
        video_feature = getattr(data, "video_feature", None)
        image = getattr(video_feature, "image", None)
        if image is None:
            raise ModelUnavailableError(
                "The loaded TRIBE model exposes no V-JEPA video extractor"
            )

        # Neuralset's Pydantic field does not currently admit "mps" during
        # construction. Assignment validation is intentionally disabled
        # upstream, so apply the already validated device only after the exact
        # released model configuration has been constructed.
        image.device = self.settings.video_device
        image.model_name = str(video_snapshot_path)

        if self.settings.video_precision == "autocast-fp16":
            autocast = getattr(torch_module, "autocast", None)
            float16 = getattr(torch_module, "float16", None)
            if not callable(autocast) or float16 is None:
                raise ModelUnavailableError(
                    "This PyTorch build does not expose MPS FP16 autocast"
                )
            try:
                with autocast(device_type="mps", dtype=float16):
                    pass
            except Exception as exc:
                raise ModelUnavailableError(
                    "This PyTorch build cannot enter MPS FP16 autocast"
                ) from exc
            if video_model_class is None:
                from neuralset.extractors.video import _HFVideoModel

                video_model_class = _HFVideoModel
            self._install_mps_autocast_patch(video_model_class, torch_module)

    def status(self) -> dict[str, Any]:
        # State uses its own lock so status remains responsive while model loading
        # or a long GPU inference holds the operation lock.
        with self._state_lock:
            return {
                "state": self._state,
                "configured": self.settings.is_model_configured,
                "model": {
                    "id": self.settings.model_id,
                    "revision": self.settings.model_revision or None,
                    "weightsSha256": TRIBE_MODEL_WEIGHTS_SHA256,
                    "device": self.settings.device,
                    "inferenceMode": self.settings.inference_mode,
                    "modalitiesUsed": list(self.settings.modalities_used),
                    "extractors": {"video": dict(self.settings.video_extractor)},
                },
                "lastError": self._last_error,
            }

    @staticmethod
    def _verify_file_sha256(path: Path, expected: str, label: str) -> None:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    digest.update(chunk)
        except OSError as exc:
            raise ModelUnavailableError(f"Could not read the pinned {label} weights") from exc
        if digest.hexdigest() != expected:
            raise ModelUnavailableError(f"Pinned {label} weights failed their SHA-256 check")

    @staticmethod
    def _extract_hemodynamic_offset(model: Any) -> float:
        data = getattr(model, "data", None)
        neuro = getattr(data, "neuro", None)
        raw_offset = getattr(neuro, "offset", None)
        if raw_offset is None:
            raise PredictionValidationError(
                "The loaded TRIBE model exposes no data.neuro.offset configuration"
            )
        try:
            offset = float(raw_offset)
        except (TypeError, ValueError) as exc:
            raise PredictionValidationError(
                "The loaded TRIBE data.neuro.offset is not numeric"
            ) from exc
        if not math.isfinite(offset) or not math.isclose(
            offset,
            EXPECTED_HEMODYNAMIC_OFFSET_SECONDS,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise PredictionValidationError(
                "The loaded TRIBE model must use its published 5-second "
                "hemodynamic offset"
            )
        return offset

    def load(self) -> Any:
        """Load the exact model snapshot; mutable Hugging Face refs are rejected."""

        self.settings.require_immutable_model_revision()
        with self._operation_lock:
            if self._model is not None:
                return self._model

            self._set_state("loading")
            try:
                from huggingface_hub import snapshot_download
                import torch
                from tribev2 import TribeModel

                self._require_video_device(torch)

                feature_cache = self.settings.cache_dir / "features"
                feature_cache.mkdir(parents=True, exist_ok=True)
                token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

                snapshot_path = snapshot_download(
                    repo_id=self.settings.model_id,
                    revision=self.settings.model_revision,
                    allow_patterns=["config.yaml", self.settings.checkpoint_name],
                    cache_dir=(
                        str(self.settings.hub_cache_dir)
                        if self.settings.hub_cache_dir is not None
                        else None
                    ),
                    token=token,
                )
                video_snapshot_path = snapshot_download(
                    repo_id=VIDEO_MODEL_ID,
                    revision=VIDEO_MODEL_REVISION,
                    cache_dir=(
                        str(self.settings.hub_cache_dir)
                        if self.settings.hub_cache_dir is not None
                        else None
                    ),
                    token=token,
                )
                self._verify_file_sha256(
                    Path(snapshot_path) / self.settings.checkpoint_name,
                    TRIBE_MODEL_WEIGHTS_SHA256,
                    "TRIBE v2",
                )
                self._verify_file_sha256(
                    Path(video_snapshot_path) / "model.safetensors",
                    VIDEO_MODEL_WEIGHTS_SHA256,
                    "V-JEPA2",
                )
                config_update = {}
                if self.settings.inference_mode == "vision-only":
                    config_update.update({
                        "data.features_to_use": ["video"],
                        "data.video_feature.image.device": "cpu",
                        "data.video_feature.image.batch_size": 1,
                        "data.video_feature.use_audio": False,
                        "data.num_workers": 0,
                        "data.batch_size": 1,
                    })
                model = TribeModel.from_pretrained(
                    Path(snapshot_path),
                    checkpoint_name=self.settings.checkpoint_name,
                    cache_folder=feature_cache,
                    cluster=None,
                    device=self.settings.device,
                    config_update=config_update,
                )
                # Neuralset validates only Hub-style IDs while constructing the
                # extractor. Once validation has completed, route the first
                # actual prepare() call to the already resolved immutable local
                # snapshot so Transformers cannot follow a mutable Hub ref.
                self._configure_video_execution(
                    model,
                    Path(video_snapshot_path),
                    torch_module=torch,
                )
                offset = self._extract_hemodynamic_offset(model)
            except Exception as exc:
                self._set_state(
                    "error", f"{type(exc).__name__}: {str(exc)[:500]}"
                )
                raise ModelUnavailableError(
                    "The pinned official TRIBE v2 model could not be loaded; "
                    "inspect server logs and /api/tribe/v1/status"
                ) from exc

            self._model = model
            self._hemodynamic_offset_seconds = offset
            self._set_state("ready")
            return model

    def predict_video(self, video_path: Path) -> InferenceOutput:
        """Run the unmodified official upload-to-prediction API."""

        if not video_path.is_file():
            raise InferenceError("Uploaded video disappeared before inference")

        with self._operation_lock:
            model = self.load()
            self._set_state("predicting")
            try:
                if self.settings.inference_mode == "vision-only":
                    import pandas as pd
                    from neuralset.events.utils import standardize_events

                    events = standardize_events(
                        pd.DataFrame(
                            [
                                {
                                    "type": "Video",
                                    "filepath": str(video_path),
                                    "start": 0,
                                    "timeline": "default",
                                    "subject": "default",
                                }
                            ]
                        )
                    )
                else:
                    events = model.get_events_dataframe(video_path=str(video_path))
                predictions, segments = model.predict(events=events, verbose=False)
            except Exception as exc:
                self._set_state(
                    "ready", f"{type(exc).__name__}: {str(exc)[:500]}"
                )
                raise InferenceError(
                    "The official TRIBE v2 event extraction or prediction pipeline failed"
                ) from exc

            if self._hemodynamic_offset_seconds is None:
                raise PredictionValidationError(
                    "TRIBE runtime lost its hemodynamic offset metadata"
                )
            self._set_state("ready")
            return InferenceOutput(
                predictions=predictions,
                segments=segments,
                hemodynamic_offset_seconds=self._hemodynamic_offset_seconds,
            )
