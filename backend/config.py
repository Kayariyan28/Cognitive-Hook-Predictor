"""Environment-backed configuration with immutable model revision enforcement."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

from .errors import ConfigurationError


TRIBE_CODE_REVISION = "af58661791a351a448a489042a28f6c37e1c14b7"
DEFAULT_MODEL_ID = "facebook/tribev2"
DEFAULT_MODEL_REVISION = ""
TRIBE_MODEL_WEIGHTS_SHA256 = "9c79ffff6b642b7b0c71d558c935fb3fa33f2788bfb509feead94fafbba2f321"
VIDEO_MODEL_ID = "facebook/vjepa2-vitg-fpc64-256"
VIDEO_MODEL_REVISION = "875c192b7b704b87d1e1d99345769632dd5f739a"
VIDEO_MODEL_WEIGHTS_SHA256 = "f205e77aa2ade168db6b09d4bc420d156141f64ab964278a9c181a2bdf2a232b"
# Bump this whenever event construction, decoding, temporal sampling, or any
# other input preprocessing changes. It is deliberately part of both the
# result-cache identity and the stable Neuralset/Exca staging path.
PREPROCESSING_VERSION = "tribev2-neuralset-video-events/1"
DEFAULT_MAX_UPLOAD_BYTES = 512 * 1024 * 1024
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
INFERENCE_MODES = {"full", "vision-only"}
VIDEO_DEVICES = {"cpu", "cuda", "mps"}
VIDEO_PRECISIONS = {"fp32", "autocast-fp16"}


def _parse_bool(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean value")


def _parse_positive_int(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return parsed


def _parse_origins(value: str) -> tuple[str, ...]:
    origins = tuple(item.strip().rstrip("/") for item in value.split(",") if item.strip())
    if not origins:
        raise ConfigurationError("TRIBE_CORS_ORIGINS must contain at least one origin")
    if "*" in origins and len(origins) != 1:
        raise ConfigurationError("TRIBE_CORS_ORIGINS cannot mix '*' with named origins")
    return origins


@dataclass(frozen=True, slots=True)
class Settings:
    model_id: str
    model_revision: str
    checkpoint_name: str
    device: str
    inference_mode: str
    cache_dir: Path
    hub_cache_dir: Path | None
    result_dir: Path
    work_dir: Path
    cors_origins: tuple[str, ...]
    cors_allow_credentials: bool
    max_upload_bytes: int
    preload_model: bool
    video_device: str = "cpu"
    video_precision: str = "fp32"

    def __post_init__(self) -> None:
        if self.video_device not in VIDEO_DEVICES:
            raise ConfigurationError(
                f"TRIBE_VIDEO_DEVICE must be one of {sorted(VIDEO_DEVICES)}"
            )
        if self.video_precision not in VIDEO_PRECISIONS:
            raise ConfigurationError(
                f"TRIBE_VIDEO_PRECISION must be one of {sorted(VIDEO_PRECISIONS)}"
            )
        if (
            self.video_precision == "autocast-fp16"
            and self.video_device != "mps"
        ):
            raise ConfigurationError(
                "TRIBE_VIDEO_PRECISION=autocast-fp16 currently requires "
                "TRIBE_VIDEO_DEVICE=mps"
            )

    @property
    def is_model_configured(self) -> bool:
        return bool(COMMIT_RE.fullmatch(self.model_revision))

    @property
    def modalities_used(self) -> tuple[str, ...]:
        return ("video",) if self.inference_mode == "vision-only" else ("video", "audio", "text")

    @property
    def video_extractor(self) -> dict[str, str]:
        return {
            "id": VIDEO_MODEL_ID,
            "revision": VIDEO_MODEL_REVISION,
            "weightsSha256": VIDEO_MODEL_WEIGHTS_SHA256,
            "device": self.video_device,
            "precision": self.video_precision,
        }

    def require_immutable_model_revision(self) -> None:
        if not self.model_revision:
            raise ConfigurationError(
                "TRIBE_MODEL_REVISION is required and must be the full 40-character "
                "Hugging Face commit SHA"
            )
        if not COMMIT_RE.fullmatch(self.model_revision):
            raise ConfigurationError(
                "TRIBE_MODEL_REVISION must be a full 40-character hexadecimal commit SHA; "
                "mutable names such as 'main' are rejected"
            )

    @classmethod
    def from_env(cls) -> "Settings":
        runtime_root = Path(__file__).resolve().parent / ".runtime"
        origins = _parse_origins(
            os.getenv(
                "TRIBE_CORS_ORIGINS",
                "http://localhost:4173,http://127.0.0.1:4173",
            )
        )
        allow_credentials = _parse_bool(
            os.getenv("TRIBE_CORS_ALLOW_CREDENTIALS", "false"),
            name="TRIBE_CORS_ALLOW_CREDENTIALS",
        )
        if origins == ("*",) and allow_credentials:
            raise ConfigurationError(
                "TRIBE_CORS_ALLOW_CREDENTIALS cannot be true when TRIBE_CORS_ORIGINS is '*'"
            )

        model_id = os.getenv("TRIBE_MODEL_ID", DEFAULT_MODEL_ID).strip()
        if not model_id:
            raise ConfigurationError("TRIBE_MODEL_ID cannot be empty")

        checkpoint_name = os.getenv("TRIBE_CHECKPOINT_NAME", "best.ckpt").strip()
        if not checkpoint_name or Path(checkpoint_name).name != checkpoint_name:
            raise ConfigurationError("TRIBE_CHECKPOINT_NAME must be a plain filename")

        device = os.getenv("TRIBE_DEVICE", "cuda").strip()
        if not device:
            raise ConfigurationError("TRIBE_DEVICE cannot be empty")
        inference_mode = os.getenv("TRIBE_INFERENCE_MODE", "full").strip().lower()
        if inference_mode not in INFERENCE_MODES:
            raise ConfigurationError(
                f"TRIBE_INFERENCE_MODE must be one of {sorted(INFERENCE_MODES)}"
            )
        video_device = os.getenv("TRIBE_VIDEO_DEVICE", "cuda").strip().lower()
        video_precision = os.getenv(
            "TRIBE_VIDEO_PRECISION", "fp32"
        ).strip().lower()
        raw_hub_cache = os.getenv("TRIBE_HUB_CACHE_DIR", "").strip()

        return cls(
            model_id=model_id,
            model_revision=os.getenv(
                "TRIBE_MODEL_REVISION", DEFAULT_MODEL_REVISION
            ).strip().lower(),
            checkpoint_name=checkpoint_name,
            device=device,
            inference_mode=inference_mode,
            cache_dir=Path(
                os.getenv("TRIBE_CACHE_DIR", str(runtime_root / "cache"))
            ).expanduser().resolve(),
            hub_cache_dir=(
                Path(raw_hub_cache).expanduser().resolve() if raw_hub_cache else None
            ),
            result_dir=Path(
                os.getenv("TRIBE_RESULT_DIR", str(runtime_root / "results"))
            ).expanduser().resolve(),
            work_dir=Path(
                os.getenv("TRIBE_WORK_DIR", str(runtime_root / "work"))
            ).expanduser().resolve(),
            cors_origins=origins,
            cors_allow_credentials=allow_credentials,
            max_upload_bytes=_parse_positive_int(
                os.getenv("TRIBE_MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES)),
                name="TRIBE_MAX_UPLOAD_BYTES",
            ),
            preload_model=_parse_bool(
                os.getenv("TRIBE_PRELOAD_MODEL", "false"),
                name="TRIBE_PRELOAD_MODEL",
            ),
            video_device=video_device,
            video_precision=video_precision,
        )
