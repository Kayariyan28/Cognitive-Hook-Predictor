"""The portable provider must fail closed exactly like the MLX one."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

from backend.insight.config import (
    DEFAULT_LOCAL_MODEL,
    DEFAULT_TORCH_LOCAL_MODEL,
    InsightSettings,
)
from backend.insight.providers import PROVIDER_CLASSES, TorchLocalProvider, build_provider
from backend.insight.providers.base import ProviderUnavailableError


PINNED = "e" * 40


def settings(**overrides: str) -> InsightSettings:
    environ = {"INSIGHT_PROVIDER": "torch-local"}
    environ.update(overrides)
    return InsightSettings.from_env(environ)


class TorchProviderReadinessTests(unittest.TestCase):
    def test_selection_builds_the_torch_provider(self) -> None:
        self.assertIsInstance(build_provider(settings()), TorchLocalProvider)
        self.assertIn("torch-local", PROVIDER_CLASSES)

    def test_the_default_model_is_the_upstream_repository(self) -> None:
        # An MLX-quantised default would be a model this provider can never load.
        self.assertEqual(settings().local_model, DEFAULT_TORCH_LOCAL_MODEL)
        self.assertNotEqual(settings().local_model, DEFAULT_LOCAL_MODEL)
        self.assertEqual(
            InsightSettings.from_env({}).local_model, DEFAULT_LOCAL_MODEL
        )

    def test_an_unpinned_revision_is_refused(self) -> None:
        state = TorchLocalProvider(settings(INSIGHT_LOCAL_MODEL_REVISION="main")).availability()
        self.assertFalse(state.available)
        self.assertIn("40-character commit SHA", state.reason)

    def test_a_missing_runtime_names_the_package(self) -> None:
        provider = TorchLocalProvider(
            settings(INSIGHT_LOCAL_MODEL_REVISION=PINNED),
            module_probe=lambda name: False,
        )
        state = provider.availability()
        self.assertFalse(state.available)
        self.assertIn("torch is not installed", state.reason)

    def test_an_absent_snapshot_is_unavailable_not_a_download(self) -> None:
        def refuse(model_id: str, revision: str) -> str:
            raise OSError("not in the local cache")

        provider = TorchLocalProvider(
            settings(INSIGHT_LOCAL_MODEL_REVISION=PINNED),
            module_probe=lambda name: True,
            snapshot_resolver=refuse,
            loader=lambda snapshot: (object(), object()),
        )
        state = provider.availability()
        self.assertFalse(state.available)
        self.assertIn("not present in the local snapshot cache", state.reason)

    def test_providers_never_chain(self) -> None:
        provider = TorchLocalProvider(
            settings(INSIGHT_LOCAL_MODEL_REVISION="main"),
        )
        with self.assertRaises(ProviderUnavailableError):
            provider.generate_text("system", "user")

    def test_a_provider_not_selected_reports_that_rather_than_readiness(self) -> None:
        state = TorchLocalProvider(InsightSettings.from_env({})).availability()
        self.assertFalse(state.available)
        self.assertIn("not the local torch provider", state.reason)


class TorchProviderGenerationTests(unittest.TestCase):
    def provider(self, generator) -> TorchLocalProvider:
        return TorchLocalProvider(
            settings(INSIGHT_LOCAL_MODEL_REVISION=PINNED),
            module_probe=lambda name: True,
            snapshot_resolver=lambda model_id, revision: "/snapshot",
            loader=lambda snapshot: (object(), object()),
            generator=generator,
        )

    def test_generated_text_carries_the_pinned_identity(self) -> None:
        result = self.provider(lambda **kwargs: '{"hookReport": []}').generate_text(
            "system", "user"
        )
        self.assertEqual(result.model_id, DEFAULT_TORCH_LOCAL_MODEL)
        self.assertEqual(result.model_revision, PINNED)
        self.assertEqual(result.raw_output, '{"hookReport": []}')

    def test_empty_output_is_an_execution_error_not_an_empty_artifact(self) -> None:
        from backend.insight.providers.base import ProviderExecutionError

        with self.assertRaises(ProviderExecutionError):
            self.provider(lambda **kwargs: "   ").generate_text("system", "user")

    def test_decoding_is_greedy_so_two_identical_bundles_agree(self) -> None:
        source = Path("backend/insight/providers/torch_local.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        sampling = [
            keyword
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "do_sample"
        ]
        self.assertTrue(sampling, "the generate call must state its sampling mode")
        for keyword in sampling:
            self.assertIs(keyword.value.value, False)


if __name__ == "__main__":
    unittest.main()
