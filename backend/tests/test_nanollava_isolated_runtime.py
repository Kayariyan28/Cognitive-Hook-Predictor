from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock

from backend.forecast.workers import nanollava_stdio
from backend.forecast.workers.nanollava import (
    BATCH_REQUEST_SCHEMA,
    BATCH_RESPONSE_SCHEMA,
    MODEL_REVISION,
    MODEL_WEIGHTS_SHA256,
    PYTHON_ENV,
    SNAPSHOT_ENV,
    STDIO_HELPER_PATH,
    IsolatedMlxVlmBatchBackend,
    NanoLlavaOutputError,
    NanoLlavaSemanticAdapter,
    NanoLlavaUnavailable,
    _strict_json_object,
    deterministic_sample_times,
)


VALID_DESCRIPTION = json.dumps(
    {
        "scene": "A person stands beside a blue table.",
        "action": "The person raises one hand.",
        "visibleText": ["START"],
        "shot": "Medium static shot.",
        "uncertainties": [],
    },
    separators=(",", ":"),
)
ISOLATED_DESCRIPTION = json.dumps(
    {
        "scene": "A person stands beside a blue table.",
        "action": "The person raises one hand.",
        "visibleText": None,
        "shot": "Medium static shot.",
        "uncertainties": None,
    },
    separators=(",", ":"),
)
VIDEO_BYTES = b"video"
VIDEO_SHA256 = hashlib.sha256(VIDEO_BYTES).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _make_frames(root: Path) -> tuple[Path, ...]:
    frames = []
    for index in range(6):
        path = root / f"frame-{index}.png"
        path.write_bytes(f"png-{index}".encode("ascii"))
        frames.append(path)
    return tuple(frames)


class FakeFrames:
    def extract(self, video_path, duration_seconds, destination):
        del video_path
        frames = _make_frames(destination)
        return tuple(zip(deterministic_sample_times(duration_seconds), frames))


def _successful_response(request: dict[str, object]) -> bytes:
    frames = request["frames"]
    assert isinstance(frames, list)
    return json.dumps(
        {
            "schemaVersion": BATCH_RESPONSE_SCHEMA,
            "responses": [
                {
                    "index": index,
                    "frameSha256": frame["sha256"],
                    "text": ISOLATED_DESCRIPTION,
                }
                for index, frame in enumerate(frames)
            ],
        },
        separators=(",", ":"),
    ).encode("utf-8")


class NanoLlavaIsolatedRuntimeTests(unittest.TestCase):
    def test_instruction_placeholder_echo_is_rejected(self):
        echoed = json.dumps(
            {
                "scene": "short literal scene description",
                "action": "visible action or none",
                "visibleText": None,
                "shot": "literal framing/camera description",
                "uncertainties": None,
            }
        )
        with self.assertRaises(NanoLlavaOutputError):
            _strict_json_object(echoed)

    def test_stdio_helper_loads_snapshot_once_then_generates_six_descriptions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = _make_frames(root)
            load_calls: list[str] = []
            generate_calls: list[str] = []

            class Processor:
                tokenizer = object()

            class SchemaProcessor:
                def clone(self):
                    return lambda input_ids, logits: logits

            def load(snapshot, *, strict, trust_remote_code):
                self.assertTrue(strict)
                self.assertFalse(trust_remote_code)
                load_calls.append(snapshot)
                return object(), Processor()

            def generate(model, processor, prompt, *, image, **kwargs):
                del model, processor, prompt, kwargs
                generate_calls.append(image)
                return ISOLATED_DESCRIPTION

            def apply_chat_template(
                processor, config, prompt, *, add_generation_prompt, num_images
            ):
                del processor, config, add_generation_prompt
                self.assertEqual(num_images, 1)
                return f"<image>\n{prompt}"

            def build_json_schema_logits_processor(tokenizer, schema):
                self.assertIs(tokenizer, Processor.tokenizer)
                self.assertFalse(schema["additionalProperties"])
                self.assertEqual(schema["properties"]["visibleText"], {"type": "null"})
                return SchemaProcessor()

            fake_module = types.SimpleNamespace(
                load=load,
                generate=generate,
                apply_chat_template=apply_chat_template,
                structured=types.SimpleNamespace(
                    build_json_schema_logits_processor=(
                        build_json_schema_logits_processor
                    )
                ),
            )
            with (
                mock.patch.object(
                    nanollava_stdio,
                    "_validate_request",
                    return_value=(root, "Describe these frames.", list(frames), 320),
                ),
                mock.patch.dict(
                    "os.environ",
                    {
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "HF_DATASETS_OFFLINE": "1",
                    },
                    clear=False,
                ),
                mock.patch.dict("sys.modules", {"mlx_vlm": fake_module}),
            ):
                response = nanollava_stdio._execute({})

        self.assertEqual(load_calls, [str(root)])
        self.assertEqual(generate_calls, [str(path) for path in frames])
        self.assertEqual(len(response["responses"]), 6)

    def test_configured_python_must_be_absolute_present_and_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "model.safetensors").write_bytes(b"weights")
            non_executable = root / "python-not-executable"
            non_executable.write_text("#!/bin/sh\n", encoding="utf-8")
            non_executable.chmod(0o644)

            cases = {
                "relative": "venv/bin/python",
                "missing": str(root / "missing-python"),
                "non-executable": str(non_executable),
            }
            for label, python_path in cases.items():
                with self.subTest(label=label):
                    adapter = NanoLlavaSemanticAdapter.from_env(
                        {
                            SNAPSHOT_ENV: str(snapshot),
                            PYTHON_ENV: python_path,
                        }
                    )
                    availability = adapter.availability()
                    self.assertFalse(availability["executionAvailable"])
                    self.assertIsInstance(availability["reason"], str)
                    self.assertTrue(availability["reason"])

    def test_isolated_subprocess_nonzero_exit_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = _make_executable(root / "python")
            frames = _make_frames(root)

            def runner(command, **kwargs):
                del command, kwargs
                return subprocess.CompletedProcess(
                    args=[], returncode=9, stdout=b"", stderr=b"runtime failed"
                )

            backend = IsolatedMlxVlmBatchBackend(
                python_executable=python,
                snapshot=root,
                runner=runner,
            )
            with self.assertRaises(NanoLlavaUnavailable):
                backend.describe_many(frames, "Describe these frames.")

    def test_isolated_subprocess_timeout_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = _make_executable(root / "python")
            frames = _make_frames(root)

            def runner(command, **kwargs):
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])

            backend = IsolatedMlxVlmBatchBackend(
                python_executable=python,
                snapshot=root,
                timeout_seconds=10.0,
                runner=runner,
            )
            with self.assertRaises(NanoLlavaUnavailable):
                backend.describe_many(frames, "Describe these frames.")

    def test_isolated_subprocess_malformed_output_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = _make_executable(root / "python")
            frames = _make_frames(root)

            def runner(command, **kwargs):
                del command, kwargs
                return subprocess.CompletedProcess(
                    args=[], returncode=0, stdout=b'{"responses":[]}', stderr=b""
                )

            backend = IsolatedMlxVlmBatchBackend(
                python_executable=python,
                snapshot=root,
                runner=runner,
            )
            with self.assertRaises(NanoLlavaOutputError):
                backend.describe_many(frames, "Describe these frames.")

    def test_one_isolated_process_describes_all_six_extracted_frames(self):
        calls: list[
            tuple[list[str], dict[str, object], dict[str, object], list[str]]
        ] = []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = _make_executable(root / "nanollava-venv-python")
            snapshot = root / "snapshot"
            snapshot.mkdir()
            video = root / "clip.mp4"
            video.write_bytes(VIDEO_BYTES)

            def runner(command, **kwargs):
                request = json.loads(kwargs["input"].decode("utf-8"))
                actual_frame_hashes = [
                    _sha256(Path(frame["path"])) for frame in request["frames"]
                ]
                calls.append(
                    (list(command), dict(kwargs), request, actual_frame_hashes)
                )
                return subprocess.CompletedProcess(
                    args=command,
                    returncode=0,
                    stdout=_successful_response(request),
                    stderr=b"",
                )

            backend = IsolatedMlxVlmBatchBackend(
                python_executable=python,
                snapshot=snapshot,
                timeout_seconds=15.0,
                runner=runner,
            )
            adapter = NanoLlavaSemanticAdapter(
                snapshot,
                backend=backend,
                frame_extractor=FakeFrames(),
            )
            with mock.patch.object(adapter, "_verify_snapshot"):
                output = adapter.run(
                    video_path=video,
                    input_sha256=VIDEO_SHA256,
                    duration_seconds=12,
                    context={},
                )

        self.assertEqual(len(calls), 1)
        command, kwargs, request, actual_frame_hashes = calls[0]
        self.assertEqual(command, [str(python), "-I", str(STDIO_HELPER_PATH)])
        self.assertEqual(request["schemaVersion"], BATCH_REQUEST_SCHEMA)
        self.assertEqual(request["modelRevision"], MODEL_REVISION)
        self.assertEqual(request["modelWeightsSha256"], MODEL_WEIGHTS_SHA256)
        self.assertEqual(len(request["frames"]), 6)
        self.assertEqual(
            [frame["index"] for frame in request["frames"]], list(range(6))
        )
        self.assertEqual(
            [frame["sha256"] for frame in request["frames"]],
            actual_frame_hashes,
        )
        self.assertEqual(kwargs["timeout"], 15.0)
        self.assertFalse(kwargs["check"])
        environment = kwargs["env"]
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")
        self.assertEqual(environment["HF_DATASETS_OFFLINE"], "1")
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("PYTHONHOME", environment)
        self.assertNotIn("MLX_TRUST_REMOTE_CODE", environment)
        self.assertEqual(output.features["nanollava.keyframes_described"], 6.0)
        self.assertEqual(
            output.features["nanollava.field_coverage.visible_text_fraction"],
            0.0,
        )
        self.assertEqual(len(output.observations), 6)


if __name__ == "__main__":
    unittest.main()
