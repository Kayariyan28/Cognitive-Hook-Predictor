"""A pinned local model on torch, for hosts where MLX does not exist.

The same contract as the MLX provider: the revision is pinned to an immutable
commit, resolved from the local snapshot cache *before* the model loads, and a
missing snapshot is an unavailable provider rather than a silent download of
whatever the hub is serving today.

Only the runtime differs. MLX-quantised repositories cannot be read by torch,
so this provider defaults to the upstream repository instead, and the manifest
records which model actually produced the text.
"""

from __future__ import annotations

import importlib.util
import time
from typing import Any, Callable, Mapping

from ..config import InsightSettings
from ..prompts.hook_doctor import build_user_message, system_prompt
from .base import (
    Availability,
    GenerationResult,
    ProviderExecutionError,
    ProviderUnavailableError,
    utc_now,
)


PROVIDER_NAME = "torch-local"


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def _resolve_pinned_snapshot(model_id: str, revision: str) -> str:
    """Resolve the pinned revision from the local cache, without any network."""

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_id, revision=revision, local_files_only=True)


class TorchLocalProvider:
    """transformers behind the provider interface, with every dependency lazy."""

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
        self.runtime: Any = None

    # -- readiness ---------------------------------------------------------

    def availability(self) -> Availability:
        configured = bool(self.settings.local_model)
        if self.settings.provider != PROVIDER_NAME:
            return Availability(
                PROVIDER_NAME,
                configured,
                False,
                f"INSIGHT_PROVIDER selects {self.settings.provider!r}, not the local torch provider.",
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
        if self._loader is None:
            for module, package in (("torch", "torch"), ("transformers", "transformers")):
                if not self._module_probe(module):
                    return Availability(
                        PROVIDER_NAME,
                        True,
                        False,
                        f"{package} is not installed in this backend environment.",
                        self.settings.local_model,
                        self.settings.local_model_revision,
                    )
            from ...forecast.workers.torch_runtime import unavailable_reason

            blocked = unavailable_reason(
                requested_device=self.settings.local_device,
                requested_dtype=self.settings.local_dtype,
            )
            if blocked:
                return Availability(
                    PROVIDER_NAME,
                    True,
                    False,
                    blocked,
                    self.settings.local_model,
                    self.settings.local_model_revision,
                )
        try:
            self._snapshot_resolver(
                self.settings.local_model, self.settings.local_model_revision
            )
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
            "The pinned local model revision is verified and transformers is importable.",
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
                f"the local torch provider failed during generation: {type(exc).__name__}"
            ) from exc
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise ProviderExecutionError("the local torch provider returned no text")
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
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ...forecast.workers.torch_runtime import resolve_runtime, torch_dtype

        runtime = resolve_runtime(
            requested_device=self.settings.local_device,
            requested_dtype=self.settings.local_dtype,
            torch_module=torch,
        )
        self.runtime = runtime
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, torch_dtype=torch_dtype(torch, runtime.dtype)
        ).to(runtime.device)
        model.eval()
        return model, AutoTokenizer.from_pretrained(snapshot)

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
        import torch

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            produced = model.generate(
                **inputs,
                max_new_tokens=self.settings.max_output_tokens,
                # Temperature is fixed at zero for this lane, so decoding is greedy
                # and two identical bundles produce identical text.
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            produced[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
