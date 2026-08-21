"""The default provider: a pinned local MLX model, on this machine only.

The model identity is pinned to an immutable revision and that revision is
resolved from the local snapshot cache *before* the model loads.  A missing
snapshot is an unavailable provider, never a silent download of whatever the
hub is serving today.
"""

from __future__ import annotations

import importlib.util
import json
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


PROVIDER_NAME = "mlx-local"


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def _resolve_pinned_snapshot(model_id: str, revision: str) -> str:
    """Resolve the pinned revision from the local cache, without any network."""

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_id, revision=revision, local_files_only=True)


class MlxLocalProvider:
    """mlx-lm behind the provider interface, with every dependency lazy."""

    name = PROVIDER_NAME

    def __init__(
        self,
        settings: InsightSettings,
        *,
        snapshot_resolver: Callable[[str, str], str] | None = None,
        loader: Callable[[str], tuple[Any, Any]] | None = None,
        generator: Callable[..., str] | None = None,
        module_probe: Callable[[str], bool] | None = None,
    ) -> None:
        self.settings = settings
        self._snapshot_resolver = snapshot_resolver or _resolve_pinned_snapshot
        self._loader = loader
        self._generator = generator
        self._module_probe = module_probe or _module_available

    # -- readiness ---------------------------------------------------------

    def availability(self) -> Availability:
        configured = bool(self.settings.local_model)
        if self.settings.provider != PROVIDER_NAME:
            return Availability(
                PROVIDER_NAME,
                configured,
                False,
                f"INSIGHT_PROVIDER selects {self.settings.provider!r}, not the local MLX provider.",
                self.settings.local_model,
                self.settings.local_model_revision or None,
            )
        if not configured:
            return Availability(
                PROVIDER_NAME, False, False, "INSIGHT_LOCAL_MODEL is not set."
            )
        if not self.settings.local_revision_is_pinned:
            return Availability(
                PROVIDER_NAME,
                True,
                False,
                "INSIGHT_LOCAL_MODEL_REVISION must be a full 40-character commit SHA; "
                "mutable names such as 'main' are rejected.",
                self.settings.local_model,
                self.settings.local_model_revision or None,
            )
        if self._loader is None and not self._module_probe("mlx_lm"):
            return Availability(
                PROVIDER_NAME,
                True,
                False,
                "mlx-lm is not installed in this backend environment.",
                self.settings.local_model,
                self.settings.local_model_revision,
            )
        try:
            self._snapshot_resolver(self.settings.local_model, self.settings.local_model_revision)
        except Exception:
            return Availability(
                PROVIDER_NAME,
                True,
                False,
                "The pinned local model revision is not present in the local snapshot cache.",
                self.settings.local_model,
                self.settings.local_model_revision,
            )
        return Availability(
            PROVIDER_NAME,
            True,
            True,
            "The pinned local model revision is verified and mlx-lm is importable.",
            self.settings.local_model,
            self.settings.local_model_revision,
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

        snapshot = self._snapshot_resolver(
            self.settings.local_model, self.settings.local_model_revision
        )
        started_at = utc_now()
        started = time.monotonic()
        try:
            model, tokenizer = self._load(snapshot)
            prompt = self._render_prompt(tokenizer, system, user)
            raw_output = self._run(model, tokenizer, prompt)
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            raise ProviderExecutionError(
                f"the local MLX provider failed during generation: {type(exc).__name__}"
            ) from exc
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise ProviderExecutionError("the local MLX provider returned no text")
        return GenerationResult(
            raw_output=raw_output,
            model_id=self.settings.local_model,
            model_revision=self.settings.local_model_revision,
            started_at=started_at,
            completed_at=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
        )

    def _load(self, snapshot: str) -> tuple[Any, Any]:
        if self._loader is not None:
            return self._loader(snapshot)
        from mlx_lm import load

        return load(snapshot)

    def _render_prompt(self, tokenizer: Any, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        template = getattr(tokenizer, "apply_chat_template", None)
        if callable(template):
            return template(messages, tokenize=False, add_generation_prompt=True)
        return f"{messages[0]['content']}\n\n{messages[1]['content']}"

    def _run(self, model: Any, tokenizer: Any, prompt: str) -> str:
        if self._generator is not None:
            return self._generator(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=self.settings.max_output_tokens,
                temperature=self.settings.temperature,
            )
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        return generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=self.settings.max_output_tokens,
            sampler=make_sampler(temp=self.settings.temperature),
            verbose=False,
        )


def strict_json_object(raw_output: str, *, max_bytes: int) -> dict[str, Any]:
    """Parse one JSON object strictly. There is no bracket repair, by design."""

    if len(raw_output.encode("utf-8")) > max_bytes:
        raise ProviderExecutionError("the provider output exceeds the configured byte limit")
    payload = json.loads(raw_output)
    if not isinstance(payload, dict):
        raise ValueError("the provider output is not a JSON object")
    return payload
