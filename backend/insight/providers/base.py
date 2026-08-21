"""One interface for every insight provider.

Providers fail closed and never chain: if the configured provider is not
available, the answer is `provider_unavailable`.  The service does not quietly
try a different one, because a creator reading generated text needs to know
which model produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, runtime_checkable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ProviderUnavailableError(RuntimeError):
    """The provider is disabled, unconfigured, unverified, or not importable."""


class ProviderExecutionError(RuntimeError):
    """The provider was reachable but produced no usable output."""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    raw_output: str
    model_id: str
    model_revision: str
    started_at: str
    completed_at: str
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class Availability:
    provider: str
    configured: bool
    available: bool
    reason: str
    model_id: str | None = None
    model_revision: str | None = None

    def public_value(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "configured": self.configured,
            "available": self.available,
            "reason": self.reason,
            "model": {"id": self.model_id, "revision": self.model_revision},
        }


@runtime_checkable
class InsightProvider(Protocol):
    name: str

    def availability(self) -> Availability:
        """Cheap, non-destructive readiness. Never downloads or loads weights."""

    def generate(self, bundle: Mapping[str, Any], *, hook_only: bool) -> GenerationResult:
        """Return raw model text plus timing, or raise a provider error."""
