from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from backend.forecast.execution_registry import WorkerPin
from backend.forecast.worker_protocol import WorkerIdentity, validate_branch_output
from backend.forecast.workers.vjepa21 import (
    ADAPTER_ID,
    BRANCH,
    CASE_COLLISION_BLOBS,
    CHECKPOINT_SHA256_ENV,
    CROP_SIZE,
    DEVICE_ENV,
    EXPECTED_TOKEN_COUNT,
    FRAME_COUNT,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    OFFICIAL_CODE_REVISION,
    PREPROCESSING_ID,
    PREPROCESSING_SHA256,
    RGB_FRAME_BYTES,
    SOURCE_PATH_ENV,
    CHECKPOINT_PATH_ENV,
    VJepa21Config,
    VJepa21Worker,
    VJepa21WorkerError,
    WindowEvidence,
    WindowPlan,
    main,
    plan_windows,
    validate_worker_request,
)


CHECKPOINT_BYTES = b"fixture-vjepa-2.1-base-checkpoint"
CHECKPOINT_SHA = hashlib.sha256(CHECKPOINT_BYTES).hexdigest()
INPUT_BYTES = b"fixture-short-video-container"
INPUT_SHA = hashlib.sha256(INPUT_BYTES).hexdigest()


class FakeEncoder:
    def __init__(self) -> None:
        self.loaded_state = None
        self.strict = None
        self.eval_called = False
        self.device = None

    def load_state_dict(self, state, *, strict):
        self.loaded_state = dict(state)
        self.strict = strict

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.device = device
        return self


class FakeTorch:
    def __init__(self, *, cuda_available: bool = False, mps_available: bool = False):
        self.encoder = FakeEncoder()
        self.hub_calls = []
        self.load_calls = []
        self.hub = SimpleNamespace(load=self._hub_load)
        self.cuda = SimpleNamespace(
            is_available=lambda: cuda_available,
            device_count=lambda: 1 if cuda_available else 0,
        )
        self.backends = SimpleNamespace(
            mps=SimpleNamespace(
                is_built=lambda: mps_available,
                is_available=lambda: mps_available,
            )
        )

    def _hub_load(self, *args, **kwargs):
        self.hub_calls.append((args, kwargs))
        return self.encoder, object()

    def load(self, *args, **kwargs):
        self.load_calls.append((args, kwargs))
        return {
            "ema_encoder": {
                "module.backbone.weight": "encoder-weight",
                "module.backbone.bias": "encoder-bias",
            }
        }


def command_runner(
    *,
    revision: str = OFFICIAL_CODE_REVISION,
    dirty: str = "",
    decode_payload: bytes | None = None,
    case_blob_payloads: dict[str, bytes] | None = None,
    executable_source_clean: bool = True,
):
    calls = []

    def run(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        if command[0] == "git" and "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, revision + "\n", "")
        if command[0] == "git" and "diff" in command:
            return subprocess.CompletedProcess(
                command, 0 if executable_source_clean else 1, "", ""
            )
        if command[0] == "git" and "status" in command:
            return subprocess.CompletedProcess(command, 0, dirty, "")
        if command[0] == "git" and "cat-file" in command:
            tracked_path = command[-1].removeprefix("HEAD:")
            payloads = case_blob_payloads or {}
            if tracked_path not in payloads:
                return subprocess.CompletedProcess(command, 1, b"", b"missing")
            return subprocess.CompletedProcess(
                command, 0, payloads[tracked_path], b""
            )
        if command[0] == "ffmpeg" and command[-1] == "-version":
            return subprocess.CompletedProcess(
                command, 0, "ffmpeg version fixture\n", ""
            )
        if command[0] == "ffmpeg" and command[-1] == "pipe:1":
            return subprocess.CompletedProcess(
                command, 0, decode_payload or b"", b""
            )
        raise AssertionError(f"unexpected command: {command}")

    run.calls = calls
    return run


def fixture_paths(root: Path) -> tuple[Path, Path]:
    source = root / "vjepa2"
    (source / "src" / "hub").mkdir(parents=True)
    (source / "hubconf.py").write_text("# fixture\n", encoding="utf-8")
    (source / "src" / "hub" / "backbones.py").write_text(
        "# fixture\n", encoding="utf-8"
    )
    checkpoint = root / "vjepa2_1_vitb_dist_vitG_384.pt"
    checkpoint.write_bytes(CHECKPOINT_BYTES)
    return source, checkpoint


def materialize_case_collisions(
    source: Path,
) -> tuple[str, dict[str, bytes]]:
    payloads: dict[str, bytes] = {}
    for worktree_relative, tracked_lowercase in CASE_COLLISION_BLOBS.items():
        payload = f"tracked lowercase blob: {tracked_lowercase}\n".encode("utf-8")
        worktree_path = source / worktree_relative
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        worktree_path.write_bytes(payload)
        payloads[tracked_lowercase] = payload
    dirty = "\n".join(f" M {path}" for path in CASE_COLLISION_BLOBS)
    return dirty, payloads


def fixture_config(
    source: Path,
    checkpoint: Path,
    *,
    checkpoint_sha256: str = CHECKPOINT_SHA,
    device: str = "cpu",
) -> VJepa21Config:
    return VJepa21Config.from_values(
        source_path=source,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        device=device,
    )


def execution_pin(checkpoint: Path) -> WorkerPin:
    return WorkerPin.from_mapping(
        BRANCH,
        {
            "baseUrl": "http://127.0.0.1:9101",
            "authTokenEnv": None,
            "timeoutSeconds": 900,
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "weightsSha256": CHECKPOINT_SHA,
                "license": MODEL_LICENSE,
                "adapterId": ADAPTER_ID,
            },
            "codeRevision": OFFICIAL_CODE_REVISION,
            "preprocessing": {
                "id": PREPROCESSING_ID,
                "sha256": PREPROCESSING_SHA256,
            },
        },
    )


def worker_request(*, duration: float = 32.0, digest: str = INPUT_SHA) -> dict:
    return {
        "schemaVersion": "creator-forecast-worker-request/1",
        "requestId": digest,
        "branch": BRANCH,
        "inputSha256": digest,
        "durationSeconds": duration,
        "context": {"platform": "youtube-shorts"},
    }


class VJepa21WorkerTests(unittest.TestCase):
    def test_ready_status_pins_checkout_hash_and_manual_strict_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, checkpoint = fixture_paths(Path(directory))
            fake_torch = FakeTorch()
            runner = command_runner()
            worker = VJepa21Worker(
                fixture_config(source, checkpoint),
                torch_module=fake_torch,
                command_runner=runner,
            )

            status = worker.status()

            self.assertEqual(status["state"], "ready")
            self.assertTrue(status["executionAvailable"])
            self.assertEqual(status["device"], "cpu")
            self.assertEqual(status["precision"], "fp32")
            self.assertEqual(status["codeRevision"], OFFICIAL_CODE_REVISION)
            self.assertEqual(status["preprocessing"]["sha256"], PREPROCESSING_SHA256)
            identity = WorkerIdentity.from_status(
                status,
                expected_branch=BRANCH,
                expected_pin=execution_pin(checkpoint),
            )
            self.assertEqual(identity.model_weights_sha256, CHECKPOINT_SHA)

            self.assertEqual(len(fake_torch.hub_calls), 1)
            args, kwargs = fake_torch.hub_calls[0]
            self.assertEqual(args, (str(source.resolve()), "vjepa2_1_vit_base_384"))
            self.assertEqual(kwargs, {"source": "local", "pretrained": False})
            self.assertEqual(
                fake_torch.encoder.loaded_state,
                {"weight": "encoder-weight", "bias": "encoder-bias"},
            )
            self.assertIs(fake_torch.encoder.strict, True)
            self.assertTrue(fake_torch.encoder.eval_called)
            self.assertEqual(fake_torch.encoder.device, "cpu")
            _, load_kwargs = fake_torch.load_calls[0]
            self.assertEqual(load_kwargs["map_location"], "cpu")
            self.assertIs(load_kwargs["weights_only"], True)

    def test_wrong_revision_hash_or_device_fails_without_fallback_or_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, checkpoint = fixture_paths(Path(directory))

            wrong_revision_torch = FakeTorch()
            wrong_revision = VJepa21Worker(
                fixture_config(source, checkpoint),
                torch_module=wrong_revision_torch,
                command_runner=command_runner(revision="f" * 40),
            ).status()
            self.assertEqual(wrong_revision["state"], "error")
            self.assertFalse(wrong_revision["executionAvailable"])
            self.assertIn(OFFICIAL_CODE_REVISION, wrong_revision["lastError"])
            self.assertEqual(wrong_revision_torch.hub_calls, [])

            wrong_hash_torch = FakeTorch()
            wrong_hash = VJepa21Worker(
                fixture_config(source, checkpoint, checkpoint_sha256="0" * 64),
                torch_module=wrong_hash_torch,
                command_runner=command_runner(),
            ).status()
            self.assertEqual(wrong_hash["state"], "error")
            self.assertIn("SHA-256", wrong_hash["lastError"])
            self.assertEqual(wrong_hash_torch.hub_calls, [])

            mps_torch = FakeTorch(mps_available=False)
            unavailable_mps = VJepa21Worker(
                fixture_config(source, checkpoint, device="mps"),
                torch_module=mps_torch,
                command_runner=command_runner(),
            ).status()
            self.assertEqual(unavailable_mps["state"], "error")
            self.assertIn("MPS", unavailable_mps["lastError"])
            self.assertEqual(mps_torch.hub_calls, [])

    def test_exact_macos_case_collisions_are_content_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, checkpoint = fixture_paths(Path(directory))
            dirty, payloads = materialize_case_collisions(source)
            fake_torch = FakeTorch()
            runner = command_runner(
                dirty=dirty,
                case_blob_payloads=payloads,
            )

            status = VJepa21Worker(
                fixture_config(source, checkpoint),
                torch_module=fake_torch,
                command_runner=runner,
            ).status()

            self.assertEqual(status["state"], "ready")
            cat_file_calls = [
                command
                for command, _kwargs in runner.calls
                if command[0] == "git" and "cat-file" in command
            ]
            self.assertEqual(len(cat_file_calls), 9)
            self.assertEqual(len(fake_torch.hub_calls), 1)

    def test_case_collision_tampering_or_any_other_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, checkpoint = fixture_paths(Path(directory))
            dirty, payloads = materialize_case_collisions(source)
            first_tolerated = next(iter(CASE_COLLISION_BLOBS))
            (source / first_tolerated).write_bytes(b"tampered\n")
            tampered_torch = FakeTorch()

            tampered = VJepa21Worker(
                fixture_config(source, checkpoint),
                torch_module=tampered_torch,
                command_runner=command_runner(
                    dirty=dirty,
                    case_blob_payloads=payloads,
                ),
            ).status()

            self.assertEqual(tampered["state"], "error")
            self.assertIn("tracked lowercase blob", tampered["lastError"])
            self.assertEqual(tampered_torch.hub_calls, [])

            unexpected_torch = FakeTorch()
            unexpected = VJepa21Worker(
                fixture_config(source, checkpoint),
                torch_module=unexpected_torch,
                command_runner=command_runner(
                    dirty=dirty + "\n?? src/injected.py",
                    case_blob_payloads=payloads,
                ),
            ).status()
            self.assertEqual(unexpected["state"], "error")
            self.assertIn("outside the exact", unexpected["lastError"])
            self.assertEqual(unexpected_torch.hub_calls, [])

            executable_torch = FakeTorch()
            executable_change = VJepa21Worker(
                fixture_config(source, checkpoint),
                torch_module=executable_torch,
                command_runner=command_runner(
                    dirty=dirty,
                    case_blob_payloads=payloads,
                    executable_source_clean=False,
                ),
            ).status()
            self.assertEqual(executable_change["state"], "error")
            self.assertIn("executable source", executable_change["lastError"])
            self.assertEqual(executable_torch.hub_calls, [])

    def test_request_hash_is_recomputed_before_model_loading(self) -> None:
        class LoadSentinelWorker(VJepa21Worker):
            load_attempted = False

            def _ensure_loaded(self):
                self.load_attempted = True
                raise AssertionError("model must not load")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, checkpoint = fixture_paths(root)
            video = root / "clip.mp4"
            video.write_bytes(INPUT_BYTES)
            worker = LoadSentinelWorker(fixture_config(source, checkpoint))
            forged = worker_request(digest="a" * 64)

            with self.assertRaisesRegex(VJepa21WorkerError, "uploaded video"):
                worker.extract(video_path=video, request=forged)
            self.assertFalse(worker.load_attempted)

    def test_request_validation_is_exact_and_duration_bounded(self) -> None:
        valid = validate_worker_request(
            worker_request(duration=10.0), actual_input_sha256=INPUT_SHA
        )
        self.assertEqual(valid["durationSeconds"], 10.0)
        valid = validate_worker_request(
            worker_request(duration=60.0), actual_input_sha256=INPUT_SHA
        )
        self.assertEqual(valid["durationSeconds"], 60.0)

        for duration in (9.99, 60.01):
            with self.assertRaisesRegex(VJepa21WorkerError, "10 to 60"):
                validate_worker_request(
                    worker_request(duration=duration),
                    actual_input_sha256=INPUT_SHA,
                )
        unknown = worker_request()
        unknown["score"] = 99
        with self.assertRaisesRegex(VJepa21WorkerError, "missing or unknown"):
            validate_worker_request(unknown, actual_input_sha256=INPUT_SHA)

    def test_window_plan_is_deterministic_and_covers_clip_edges(self) -> None:
        short = plan_windows(10)
        self.assertEqual(short, (WindowPlan(0, 0.0, 10.0),))

        exact = plan_windows(32)
        self.assertEqual(
            exact,
            (WindowPlan(0, 0.0, 16.0), WindowPlan(1, 16.0, 32.0)),
        )

        long_clip = plan_windows(60)
        self.assertEqual(len(long_clip), 4)
        self.assertEqual(long_clip[0].start_time, 0.0)
        self.assertEqual(long_clip[-1].end_time, 60.0)
        self.assertTrue(all(item.duration == 16.0 for item in long_clip))

    def test_ffmpeg_decode_uses_exact_filter_and_pads_to_64_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, checkpoint = fixture_paths(Path(directory))
            payload = b"\x01" * (2 * RGB_FRAME_BYTES)
            runner = command_runner(decode_payload=payload)
            worker = VJepa21Worker(
                fixture_config(source, checkpoint), command_runner=runner
            )

            decoded, padded = worker._decode_window(
                Path(directory) / "clip.mp4", WindowPlan(0, 0.0, 10.0)
            )

            self.assertEqual(padded, 62)
            self.assertEqual(len(decoded), FRAME_COUNT * RGB_FRAME_BYTES)
            command = runner.calls[-1][0]
            self.assertNotIn("decord", " ".join(command).lower())
            filters = command[command.index("-vf") + 1]
            self.assertIn("fps=fps=4:round=near", filters)
            self.assertIn("438", filters)
            self.assertIn(f"crop=w={CROP_SIZE}:h={CROP_SIZE}", filters)

    def test_extract_emits_only_protocol_valid_descriptive_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, checkpoint = fixture_paths(root)
            video = root / "clip.mp4"
            video.write_bytes(INPUT_BYTES)
            fake_torch = FakeTorch()
            timestamps = iter(
                ("2026-08-19T01:00:00Z", "2026-08-19T01:00:02Z")
            )

            def execute(plan: WindowPlan) -> WindowEvidence:
                embedding = [0.0] * 768
                embedding[plan.index] = 1.0
                return WindowEvidence(
                    plan=plan,
                    pooled_embedding=tuple(embedding),
                    embedding_norm=float(plan.index + 1),
                    temporal_consistency=0.9 - plan.index * 0.2,
                    temporal_change_mean=0.1 + plan.index * 0.2,
                    temporal_change_peak=0.2 + plan.index * 0.2,
                )

            worker = VJepa21Worker(
                fixture_config(source, checkpoint),
                torch_module=fake_torch,
                command_runner=command_runner(),
                clock=lambda: next(timestamps),
                window_executor=execute,
            )
            output = worker.extract(
                video_path=video, request=worker_request(duration=32)
            )

            self.assertFalse(output["behavioralOutcome"])
            self.assertEqual(output["evidenceKind"], "learned-visual-representation")
            self.assertEqual(output["features"]["vjepa2_1.embedding_norm_mean"], 1.5)
            embedding_features = {
                key: value
                for key, value in output["features"].items()
                if key.startswith("vjepa2_1.embedding_")
                and key not in {
                    "vjepa2_1.embedding_norm_mean",
                    "vjepa2_1.embedding_norm_std",
                }
            }
            self.assertEqual(len(embedding_features), 768)
            self.assertEqual(embedding_features["vjepa2_1.embedding_000"], 0.5)
            self.assertEqual(embedding_features["vjepa2_1.embedding_001"], 0.5)
            self.assertEqual(embedding_features["vjepa2_1.embedding_767"], 0.0)
            self.assertEqual(
                output["features"]["vjepa2_1.interwindow_change_mean"], 1.0
            )
            self.assertEqual(len(output["observations"]), 2)
            self.assertIn("not attention", output["observations"][0]["text"])
            self.assertNotIn("score", " ".join(output["features"]))

            status = worker.status()
            identity = WorkerIdentity.from_status(
                status,
                expected_branch=BRANCH,
                expected_pin=execution_pin(checkpoint),
            )
            validated = validate_branch_output(
                output,
                expected_branch=BRANCH,
                expected_input_sha256=INPUT_SHA,
                identity=identity,
            )
            self.assertFalse(validated.public_value()["behavioralOutcome"])

    def test_cli_writes_one_json_document_and_keeps_errors_off_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, checkpoint = fixture_paths(root)
            stdout = io.StringIO()
            stderr = io.StringIO()

            class FakeCliWorker:
                def __init__(self, config):
                    self.config = config

                def status(self):
                    return {"state": "fixture-ready"}

            exit_code = main(
                [
                    "status",
                    "--source-tree",
                    str(source),
                    "--checkpoint",
                    str(checkpoint),
                    "--checkpoint-sha256",
                    CHECKPOINT_SHA,
                ],
                environ={},
                stdout=stdout,
                stderr=stderr,
                worker_factory=FakeCliWorker,
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue()), {"state": "fixture-ready"})
            self.assertEqual(stdout.getvalue().count("\n"), 1)
            self.assertEqual(stderr.getvalue(), "")

            stdout = io.StringIO()
            stderr = io.StringIO()
            exit_code = main(
                ["status"],
                environ={},
                stdout=stdout,
                stderr=stderr,
                worker_factory=FakeCliWorker,
            )
            self.assertEqual(exit_code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(
                json.loads(stderr.getvalue())["error"]["code"],
                "worker_unconfigured",
            )

    def test_environment_names_and_model_shape_are_frozen(self) -> None:
        self.assertEqual(SOURCE_PATH_ENV, "FORECAST_VJEPA21_SOURCE_PATH")
        self.assertEqual(CHECKPOINT_PATH_ENV, "FORECAST_VJEPA21_CHECKPOINT_PATH")
        self.assertEqual(
            CHECKPOINT_SHA256_ENV, "FORECAST_VJEPA21_CHECKPOINT_SHA256"
        )
        self.assertEqual(DEVICE_ENV, "FORECAST_VJEPA21_DEVICE")
        self.assertEqual(EXPECTED_TOKEN_COUNT, 32 * 24 * 24)
        self.assertEqual(
            PREPROCESSING_SHA256,
            "d505e3350d91fc2b8679ff04b792e94e115dc4f6c825566643ab280c8e9000da",
        )


if __name__ == "__main__":
    unittest.main()
