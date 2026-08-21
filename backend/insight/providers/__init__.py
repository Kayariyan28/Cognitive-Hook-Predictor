"""Insight providers behind one interface, selected by configuration only."""

from __future__ import annotations

from typing import Any

from ..config import InsightSettings
from .anthropic_cloud import AnthropicProvider
from .base import (
    Availability,
    GenerationResult,
    InsightProvider,
    ProviderExecutionError,
    ProviderUnavailableError,
)
from .mlx_local import MlxLocalProvider, strict_json_object
from .torch_local import TorchLocalProvider


PROVIDER_CLASSES = {
    "mlx-local": MlxLocalProvider,
    "torch-local": TorchLocalProvider,
    "anthropic": AnthropicProvider,
}


def build_provider(settings: InsightSettings, **kwargs: Any) -> InsightProvider:
    """Build exactly the configured provider. There is no fallback chain."""

    return PROVIDER_CLASSES[settings.provider](settings, **kwargs)


def all_availability(settings: InsightSettings) -> list[dict[str, Any]]:
    """Report every provider's readiness so /status can explain a refusal."""

    return [
        PROVIDER_CLASSES[name](settings).availability().public_value()
        for name in sorted(PROVIDER_CLASSES)
    ]


__all__ = [
    "AnthropicProvider",
    "Availability",
    "GenerationResult",
    "InsightProvider",
    "MlxLocalProvider",
    "TorchLocalProvider",
    "PROVIDER_CLASSES",
    "ProviderExecutionError",
    "ProviderUnavailableError",
    "all_availability",
    "build_provider",
    "strict_json_object",
]
