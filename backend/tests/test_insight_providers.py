from __future__ import annotations

import ast
from pathlib import Path
import unittest

from backend.insight.config import (
    InsightConfigurationError,
    InsightSettings,
)
from backend.insight.prompts.hook_doctor import (
    PROMPT_TEMPLATE_ID,
    build_user_message,
    prompt_hash,
    system_prompt,
)
from backend.insight.provenance import (
    LIMITS_STATEMENT,
    build_provenance,
    cache_key,
    output_hash,
)
from backend.insight.providers import (
    AnthropicProvider,
    MlxLocalProvider,
    ProviderExecutionError,
    ProviderUnavailableError,
    all_availability,
    build_provider,
    strict_json_object,
)
from backend.insight.providers.base import GenerationResult
from backend.tests.insight_support import forecast_result, tribe_descriptors
from backend.insight.bundle import assemble_evidence_bundle


PINNED_REVISION = "b" * 40


def settings(**overrides: str) -> InsightSettings:
    environ = {
        "INSIGHT_PROVIDER": "mlx-local",
        "INSIGHT_LOCAL_MODEL": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "INSIGHT_LOCAL_MODEL_REVISION": PINNED_REVISION,
        "INSIGHT_DIR": "/tmp/signalframe-insight-tests",
    }
    environ.update(overrides)
    return InsightSettings.from_env(environ)


def bundle():
    return assemble_evidence_bundle(forecast_result(), tribe_descriptors=tribe_descriptors())


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(f"{item['role']}: {item['content']}" for item in messages)


class ConfigTests(unittest.TestCase):
    def test_defaults_keep_evidence_on_this_machine(self):
        config = InsightSettings.from_env({})
        self.assertEqual(config.provider, "mlx-local")
        self.assertFalse(config.cloud_enabled)
        self.assertFalse(config.cloud_is_permitted)
        self.assertEqual(config.temperature, 0.0)

    def test_public_summary_never_carries_the_key(self):
        config = settings(
            INSIGHT_PROVIDER="anthropic",
            INSIGHT_CLOUD_ENABLED="true",
            ANTHROPIC_API_KEY="sk-ant-secret-value",
        )
        summary = config.public_summary()
        self.assertTrue(summary["apiKeyPresent"])
        self.assertNotIn("sk-ant-secret-value", repr(summary))
        self.assertNotIn("anthropicApiKey", summary)

    def test_invalid_configuration_is_refused(self):
        for environ in (
            {"INSIGHT_PROVIDER": "openai"},
            {"INSIGHT_CLOUD_ENABLED": "maybe"},
            {"INSIGHT_MAX_OUTPUT_TOKENS": "0"},
            {"INSIGHT_TIMEOUT_SECONDS": "-3"},
            {"INSIGHT_OCR_ENGINE": "easyocr"},
        ):
            with self.subTest(environ=environ):
                with self.assertRaises(InsightConfigurationError):
                    InsightSettings.from_env(environ)


class PromptTemplateTests(unittest.TestCase):
    def test_template_hash_is_stable(self):
        self.assertEqual(prompt_hash(), prompt_hash())
        self.assertEqual(len(prompt_hash()), 64)
        self.assertEqual(PROMPT_TEMPLATE_ID, "hook-doctor.v2")

    def test_template_embeds_the_shared_term_lists_and_limits(self):
        rendered = system_prompt()
        for term in ("go viral", "retention", "subconscious", "attention"):
            self.assertIn(term, rendered)
        self.assertIn("predicted average-subject cortical BOLD", rendered)
        self.assertIn("untested heuristic", rendered)
        self.assertIn("two significant figures", rendered)
        for owned in ("behavioralOutcome", "limits", "provenance"):
            self.assertIn(owned, rendered)

    def test_user_message_carries_only_the_bundle(self):
        message = build_user_message(bundle(), hook_only=True)
        self.assertIn("hook window", message)
        self.assertIn("inputEvidenceHash", message)
        self.assertNotIn(".mp4", message)
        self.assertNotIn("/incoming/", message)

    def test_prompt_hash_tracks_the_term_vocabulary(self):
        import backend.insight.prompts.hook_doctor as template

        original = prompt_hash()
        patched = template.TEMPLATE.replace("Hook Doctor", "Hook Doctor v2")
        try:
            template.TEMPLATE = patched
            self.assertNotEqual(prompt_hash(), original)
        finally:
            template.TEMPLATE = template.TEMPLATE.replace("Hook Doctor v2", "Hook Doctor")
        self.assertEqual(prompt_hash(), original)


class MlxLocalProviderTests(unittest.TestCase):
    def test_available_when_pinned_and_resolvable(self):
        provider = MlxLocalProvider(
            settings(),
            snapshot_resolver=lambda model, revision: "/snapshots/pinned",
            loader=lambda snapshot: (object(), FakeTokenizer()),
            generator=lambda **kwargs: '{"hookReport": {}}',
        )
        state = provider.availability()
        self.assertTrue(state.available)
        self.assertEqual(state.model_revision, PINNED_REVISION)

    def test_mutable_revision_is_refused(self):
        provider = MlxLocalProvider(
            settings(INSIGHT_LOCAL_MODEL_REVISION="main"),
            snapshot_resolver=lambda model, revision: "/snapshots/pinned",
            loader=lambda snapshot: (object(), FakeTokenizer()),
        )
        state = provider.availability()
        self.assertFalse(state.available)
        self.assertIn("40-character commit SHA", state.reason)
        with self.assertRaises(ProviderUnavailableError):
            provider.generate(bundle(), hook_only=True)

    def test_unresolvable_snapshot_is_unavailable(self):
        def missing(model, revision):
            raise FileNotFoundError(model)

        provider = MlxLocalProvider(
            settings(), snapshot_resolver=missing, loader=lambda snapshot: (None, None)
        )
        state = provider.availability()
        self.assertFalse(state.available)
        self.assertIn("snapshot cache", state.reason)

    def test_missing_runtime_is_unavailable(self):
        provider = MlxLocalProvider(
            settings(),
            snapshot_resolver=lambda model, revision: "/snapshots/pinned",
            module_probe=lambda name: False,
        )
        self.assertFalse(provider.availability().available)
        self.assertIn("mlx-lm is not installed", provider.availability().reason)

    def test_generation_returns_raw_text_and_timing(self):
        captured: dict[str, object] = {}

        def generator(**kwargs):
            captured.update(kwargs)
            return '{"hookReport": {"windowSeconds": [0.0, 3.0]}}'

        provider = MlxLocalProvider(
            settings(),
            snapshot_resolver=lambda model, revision: "/snapshots/pinned",
            loader=lambda snapshot: ("model", FakeTokenizer()),
            generator=generator,
        )
        result = provider.generate(bundle(), hook_only=True)
        self.assertIn("hookReport", result.raw_output)
        self.assertEqual(result.model_revision, PINNED_REVISION)
        self.assertGreaterEqual(result.elapsed_seconds, 0.0)
        self.assertEqual(captured["temperature"], 0.0)
        self.assertEqual(captured["max_tokens"], 1536)
        self.assertIn("Hook Doctor", captured["prompt"])

    def test_empty_generation_is_a_provider_error(self):
        provider = MlxLocalProvider(
            settings(),
            snapshot_resolver=lambda model, revision: "/snapshots/pinned",
            loader=lambda snapshot: ("model", FakeTokenizer()),
            generator=lambda **kwargs: "   ",
        )
        with self.assertRaises(ProviderExecutionError):
            provider.generate(bundle(), hook_only=False)

    def test_strict_json_parse_has_no_bracket_repair(self):
        with self.assertRaises(ValueError):
            strict_json_object('{"hookReport": {', max_bytes=1024)
        with self.assertRaises(ProviderExecutionError):
            strict_json_object('{"hookReport": {}}', max_bytes=4)
        self.assertEqual(strict_json_object("{}", max_bytes=1024), {})


class FakeMessages:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.requests: list[dict] = []

    def create(self, **request):
        self.requests.append(request)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.messages = FakeMessages(outcomes)


class RetryableError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def text_message(text: str):
    return {"content": [{"type": "text", "text": text}]}


class AnthropicProviderTests(unittest.TestCase):
    def cloud_settings(self, **overrides):
        return settings(
            INSIGHT_PROVIDER="anthropic",
            INSIGHT_CLOUD_ENABLED="true",
            ANTHROPIC_API_KEY="sk-ant-test",
            **overrides,
        )

    def test_cloud_is_disabled_by_default(self):
        provider = AnthropicProvider(settings(INSIGHT_PROVIDER="anthropic"))
        state = provider.availability()
        self.assertFalse(state.available)
        self.assertIn("INSIGHT_CLOUD_ENABLED is false", state.reason)

    def test_missing_key_blocks_the_remote_provider(self):
        provider = AnthropicProvider(
            settings(INSIGHT_PROVIDER="anthropic", INSIGHT_CLOUD_ENABLED="true")
        )
        self.assertIn("ANTHROPIC_API_KEY is not set", provider.availability().reason)

    def test_generation_sends_only_the_bundle(self):
        client = FakeClient([text_message('{"hookReport": {}}')])
        provider = AnthropicProvider(self.cloud_settings(), client_factory=lambda: client)
        result = provider.generate(bundle(), hook_only=False)
        self.assertEqual(result.raw_output, '{"hookReport": {}}')
        request = client.messages.requests[0]
        self.assertEqual(request["temperature"], 0.0)
        self.assertIn("Hook Doctor", request["system"])
        payload = request["messages"][0]["content"]
        self.assertIn("inputEvidenceHash", payload)
        self.assertNotIn("sk-ant-test", payload)

    def test_retries_with_backoff_on_retryable_status_codes(self):
        delays: list[float] = []
        client = FakeClient(
            [RetryableError(429), RetryableError(503), text_message('{"hookReport": {}}')]
        )
        provider = AnthropicProvider(
            self.cloud_settings(), client_factory=lambda: client, sleep=delays.append
        )
        provider.generate(bundle(), hook_only=False)
        self.assertEqual(delays, [0.5, 1.0])
        self.assertEqual(len(client.messages.requests), 3)

    def test_non_retryable_errors_fail_immediately(self):
        delays: list[float] = []
        client = FakeClient([RetryableError(400)])
        provider = AnthropicProvider(
            self.cloud_settings(), client_factory=lambda: client, sleep=delays.append
        )
        with self.assertRaises(ProviderExecutionError):
            provider.generate(bundle(), hook_only=False)
        self.assertEqual(delays, [])

    def test_exhausted_retries_raise_a_provider_error(self):
        client = FakeClient([RetryableError(500), RetryableError(500), RetryableError(500)])
        provider = AnthropicProvider(
            self.cloud_settings(), client_factory=lambda: client, sleep=lambda _: None
        )
        with self.assertRaises(ProviderExecutionError):
            provider.generate(bundle(), hook_only=False)

    def test_cloud_module_never_reaches_a_media_reader(self):
        # Walk the module's own import graph inside backend/ and assert nothing
        # that can open an upload, a keyframe, a tensor, or a job directory is
        # reachable from the code path that talks to a remote service.
        forbidden_modules = {
            "backend.app",
            "backend.artifacts",
            "backend.model_runtime",
            "backend.result_cache",
            "backend.thumbnails",
            "backend.forecast.jobs",
            "backend.forecast.orchestrator",
            "backend.forecast.workers.nanollava",
        }
        seen: set[str] = set()
        pending = ["backend.insight.providers.anthropic_cloud"]
        while pending:
            module_name = pending.pop()
            if module_name in seen:
                continue
            seen.add(module_name)
            path = Path(module_name.replace(".", "/") + ".py")
            if not path.is_file():
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            package = module_name.rsplit(".", 1)[0]
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    pending.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        parts = package.split(".")
                        base = ".".join(parts[: len(parts) - node.level + 1])
                        pending.append(f"{base}.{node.module}" if node.module else base)
                    elif node.module:
                        pending.append(node.module)
        self.assertEqual(seen & forbidden_modules, set())

        # And nothing in the module itself opens a file or shells out.
        tree = ast.parse(
            Path("backend/insight/providers/anthropic_cloud.py").read_text(encoding="utf-8")
        )
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertEqual(called & {"open", "Path", "exec", "eval", "__import__"}, set())


class ProviderSelectionTests(unittest.TestCase):
    def test_configuration_selects_exactly_one_provider(self):
        self.assertIsInstance(build_provider(settings()), MlxLocalProvider)
        self.assertIsInstance(
            build_provider(settings(INSIGHT_PROVIDER="anthropic")), AnthropicProvider
        )

    def test_status_can_explain_every_provider(self):
        entries = all_availability(settings())
        self.assertEqual([entry["provider"] for entry in entries], ["anthropic", "mlx-local"])
        for entry in entries:
            self.assertIn("reason", entry)
            self.assertNotIn("apiKey", repr(entry))


class ProvenanceTests(unittest.TestCase):
    def generation(self):
        return GenerationResult(
            raw_output="{}",
            model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
            model_revision=PINNED_REVISION,
            started_at="2026-02-01T10:00:00Z",
            completed_at="2026-02-01T10:00:03Z",
            elapsed_seconds=3.0,
        )

    def test_manifest_carries_every_documented_field(self):
        artifact = {"hookReport": {"windowSeconds": [0.0, 3.0]}, "phaseCommentary": [], "tribeNotes": []}
        manifest = build_provenance(
            settings=settings(),
            generation=self.generation(),
            input_evidence_hash="c" * 64,
            artifact=artifact,
            hook_only=True,
        )
        self.assertEqual(manifest["schemaVersion"], "insight-provenance/1")
        self.assertEqual(manifest["promptTemplateId"], PROMPT_TEMPLATE_ID)
        self.assertEqual(manifest["promptHash"], prompt_hash())
        self.assertEqual(manifest["outputHash"], output_hash(artifact))
        self.assertFalse(manifest["behavioralOutcome"])
        self.assertEqual(manifest["temperature"], 0.0)

    def test_cache_key_changes_with_every_identity_component(self):
        base = dict(
            input_evidence_hash="c" * 64,
            provider="mlx-local",
            model_revision=PINNED_REVISION,
            temperature=0.0,
        )
        original = cache_key(**base)
        self.assertEqual(original, cache_key(**base))
        self.assertNotEqual(original, cache_key(**{**base, "provider": "anthropic"}))
        self.assertNotEqual(original, cache_key(**{**base, "model_revision": "d" * 40}))
        self.assertNotEqual(original, cache_key(**{**base, "temperature": 0.2}))
        self.assertNotEqual(original, cache_key(**{**base, "input_evidence_hash": "e" * 64}))

    def test_limits_statement_names_the_lane_boundaries(self):
        for phrase in ("descriptive lane", "untested heuristics", "cortical BOLD", "behavioral"):
            self.assertIn(phrase, LIMITS_STATEMENT)


if __name__ == "__main__":
    unittest.main()
