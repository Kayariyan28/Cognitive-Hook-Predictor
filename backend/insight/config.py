"""Environment-backed insight configuration.

The remote provider is opt-in twice over: the operator must both name it and
set ``INSIGHT_CLOUD_ENABLED=true`` with a key present.  Nothing here ever
returns the key to a caller, and no default sends creator-derived evidence off
the machine.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping


INSIGHT_PROVIDERS = ("mlx-local", "anthropic")
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_LOCAL_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_ASR_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_MAX_OUTPUT_TOKENS = 1536
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_OUTPUT_BYTES = 256 * 1024
OCR_ENGINES = ("ocrmac", "pytesseract")

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)

# Temperature is not configurable. A descriptive lane that must copy numerals
# verbatim has no use for sampling entropy.
INSIGHT_TEMPERATURE = 0.0


class InsightConfigurationError(ValueError):
    """The insight layer is not configured for reproducible generation."""


def _parse_bool(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise InsightConfigurationError(f"{name} must be a boolean value")


def _positive_int(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise InsightConfigurationError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise InsightConfigurationError(f"{name} must be greater than zero")
    return parsed


def _positive_float(value: str, *, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise InsightConfigurationError(f"{name} must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise InsightConfigurationError(f"{name} must be a positive finite number")
    return parsed


@dataclass(frozen=True, slots=True)
class InsightSettings:
    provider: str
    cloud_enabled: bool
    anthropic_api_key: str | None
    anthropic_model: str
    local_model: str
    local_model_revision: str
    max_output_tokens: int
    timeout_seconds: float
    max_output_bytes: int
    insight_dir: Path
    asr_model: str
    asr_model_revision: str
    ocr_engine: str

    @property
    def temperature(self) -> float:
        return INSIGHT_TEMPERATURE

    @property
    def local_revision_is_pinned(self) -> bool:
        return bool(COMMIT_RE.fullmatch(self.local_model_revision))

    @property
    def asr_revision_is_pinned(self) -> bool:
        return bool(COMMIT_RE.fullmatch(self.asr_model_revision))

    @property
    def cloud_is_permitted(self) -> bool:
        return self.cloud_enabled and bool(self.anthropic_api_key)

    def public_summary(self) -> dict[str, Any]:
        """A status-safe view. The API key is represented only as a boolean."""

        return {
            "provider": self.provider,
            "cloudEnabled": self.cloud_enabled,
            "apiKeyPresent": bool(self.anthropic_api_key),
            "anthropicModel": self.anthropic_model,
            "localModel": self.local_model,
            "localModelRevision": self.local_model_revision or None,
            "localRevisionPinned": self.local_revision_is_pinned,
            "maxOutputTokens": self.max_output_tokens,
            "timeoutSeconds": self.timeout_seconds,
            "temperature": self.temperature,
        }

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "InsightSettings":
        source = os.environ if environ is None else environ
        provider = source.get("INSIGHT_PROVIDER", "mlx-local").strip().lower()
        if provider not in INSIGHT_PROVIDERS:
            raise InsightConfigurationError(
                f"INSIGHT_PROVIDER must be one of {sorted(INSIGHT_PROVIDERS)}"
            )
        api_key = source.get("ANTHROPIC_API_KEY", "").strip()
        ocr_engine = source.get("INSIGHT_OCR_ENGINE", "ocrmac").strip().lower()
        if ocr_engine not in OCR_ENGINES:
            raise InsightConfigurationError(
                f"INSIGHT_OCR_ENGINE must be one of {sorted(OCR_ENGINES)}"
            )
        runtime_root = Path(__file__).resolve().parents[1] / ".runtime"
        return cls(
            provider=provider,
            cloud_enabled=_parse_bool(
                source.get("INSIGHT_CLOUD_ENABLED", "false"), name="INSIGHT_CLOUD_ENABLED"
            ),
            anthropic_api_key=api_key or None,
            anthropic_model=source.get(
                "INSIGHT_ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL
            ).strip()
            or DEFAULT_ANTHROPIC_MODEL,
            local_model=source.get("INSIGHT_LOCAL_MODEL", DEFAULT_LOCAL_MODEL).strip()
            or DEFAULT_LOCAL_MODEL,
            local_model_revision=source.get("INSIGHT_LOCAL_MODEL_REVISION", "").strip().lower(),
            max_output_tokens=_positive_int(
                source.get("INSIGHT_MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS)),
                name="INSIGHT_MAX_OUTPUT_TOKENS",
            ),
            timeout_seconds=_positive_float(
                source.get("INSIGHT_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)),
                name="INSIGHT_TIMEOUT_SECONDS",
            ),
            max_output_bytes=_positive_int(
                source.get("INSIGHT_MAX_OUTPUT_BYTES", str(DEFAULT_MAX_OUTPUT_BYTES)),
                name="INSIGHT_MAX_OUTPUT_BYTES",
            ),
            insight_dir=Path(
                source.get("INSIGHT_DIR", str(runtime_root / "insight"))
            ).expanduser().resolve(),
            asr_model=source.get("INSIGHT_ASR_MODEL", DEFAULT_ASR_MODEL).strip()
            or DEFAULT_ASR_MODEL,
            asr_model_revision=source.get("INSIGHT_ASR_MODEL_REVISION", "").strip().lower(),
            ocr_engine=ocr_engine,
        )
