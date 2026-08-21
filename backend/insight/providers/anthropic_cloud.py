"""The opt-in remote provider.

This module deliberately imports nothing that can read an upload directory, a
staged video, a frame, or a tensor.  Its entire payload is the derived JSON
evidence bundle it is handed.  A test enforces that import boundary, because a
comment is not a guarantee.
"""

from __future__ import annotations

import importlib.util
import time
from typing import Any, Callable, Mapping

from ..config import InsightSettings
from ..prompts.hook_doctor_v1 import build_user_message, system_prompt
from .base import (
    Availability,
    GenerationResult,
    ProviderExecutionError,
    ProviderUnavailableError,
    utc_now,
)


PROVIDER_NAME = "anthropic"
MAXIMUM_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 0.5
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def _status_code(error: BaseException) -> int | None:
    code = getattr(error, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(error, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def is_retryable(error: BaseException) -> bool:
    code = _status_code(error)
    return code in RETRYABLE_STATUS_CODES if code is not None else False


class AnthropicProvider:
    """The Anthropic Messages API behind the same interface as the local model."""

    name = PROVIDER_NAME

    def __init__(
        self,
        settings: InsightSettings,
        *,
        client_factory: Callable[[], Any] | None = None,
        module_probe: Callable[[str], bool] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._client_factory = client_factory
        self._module_probe = module_probe or _module_available
        self._sleep = sleep

    # -- readiness ---------------------------------------------------------

    def availability(self) -> Availability:
        configured = bool(self.settings.anthropic_model)
        if self.settings.provider != PROVIDER_NAME:
            return Availability(
                PROVIDER_NAME,
                configured,
                False,
                f"INSIGHT_PROVIDER selects {self.settings.provider!r}, not the remote provider.",
                self.settings.anthropic_model,
            )
        if not self.settings.cloud_enabled:
            return Availability(
                PROVIDER_NAME,
                configured,
                False,
                "INSIGHT_CLOUD_ENABLED is false; derived evidence stays on this machine.",
                self.settings.anthropic_model,
            )
        if not self.settings.anthropic_api_key:
            return Availability(
                PROVIDER_NAME,
                configured,
                False,
                "ANTHROPIC_API_KEY is not set.",
                self.settings.anthropic_model,
            )
        if self._client_factory is None and not self._module_probe("anthropic"):
            return Availability(
                PROVIDER_NAME,
                configured,
                False,
                "The anthropic SDK is not installed in this backend environment.",
                self.settings.anthropic_model,
            )
        return Availability(
            PROVIDER_NAME,
            True,
            True,
            "The operator enabled the remote provider and a key is present.",
            self.settings.anthropic_model,
        )

    # -- generation --------------------------------------------------------

    def generate(self, bundle: Mapping[str, Any], *, hook_only: bool) -> GenerationResult:
        return self.generate_text(
            system_prompt(), build_user_message(bundle, hook_only=hook_only)
        )

    def generate_text(self, system: str, user: str) -> GenerationResult:
        """One text call. The insight lane and the CI judge share this path."""

        state = self.availability()
        if not state.available:
            raise ProviderUnavailableError(state.reason)

        client = self._client()
        started_at = utc_now()
        started = time.monotonic()
        message = self._create_with_retries(client, system, user)
        raw_output = _first_text_block(message)
        if not raw_output.strip():
            raise ProviderExecutionError("the remote provider returned no text")
        return GenerationResult(
            raw_output=raw_output,
            model_id=self.settings.anthropic_model,
            model_revision=self.settings.anthropic_model,
            started_at=started_at,
            completed_at=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
        )

    def _client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()
        from anthropic import Anthropic

        return Anthropic(
            api_key=self.settings.anthropic_api_key, timeout=self.settings.timeout_seconds
        )

    def _create_with_retries(self, client: Any, system: str, user: str) -> Any:
        # The payload is derived JSON and a static template. No media, no path,
        # no identifier the operator did not already publish.
        request = {
            "model": self.settings.anthropic_model,
            "max_tokens": self.settings.max_output_tokens,
            "temperature": self.settings.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        last_error: BaseException | None = None
        for attempt in range(MAXIMUM_ATTEMPTS):
            try:
                return client.messages.create(**request)
            except Exception as exc:
                last_error = exc
                if not is_retryable(exc) or attempt == MAXIMUM_ATTEMPTS - 1:
                    break
                self._sleep(BACKOFF_BASE_SECONDS * (2**attempt))
        raise ProviderExecutionError(
            f"the remote provider failed after {MAXIMUM_ATTEMPTS} attempts: "
            f"{type(last_error).__name__}"
        ) from last_error


def _first_text_block(message: Any) -> str:
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if not isinstance(content, (list, tuple)):
        raise ProviderExecutionError("the remote provider returned no content blocks")
    for block in content:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if isinstance(text, str):
            return text
    raise ProviderExecutionError("the remote provider returned no text block")
