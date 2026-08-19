from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from backend.errors import ConfigurationError
from backend.forecast.orchestrator import MultimodalEvidenceOrchestrator
from backend.forecast.worker_protocol import (
    WORKER_OUTPUT_SCHEMA_VERSION,
    WORKER_STATUS_SCHEMA_VERSION,
    WorkerIdentity,
    WorkerProtocolError,
)
from backend.forecast.workers.local_vjepa21 import LocalVJepa21Adapter
from backend.forecast.workers.vjepa21 import (
    ADAPTER_ID,
    CHECKPOINT_PATH_ENV,
    CHECKPOINT_SHA256_ENV,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    OFFICIAL_CODE_REVISION,
    PREPROCESSING_ID,
    PREPROCESSING_SHA256,
    SOURCE_PATH_ENV,
    VJepa21Config,
)


CHECKPOINT_SHA256 = "a" * 64
VIDEO_BYTES = b"local-vjepa-adapter-video"
VIDEO_SHA256 = hashlib.sha256(VIDEO_BYTES).hexdigest()


def config(root: Path) -> VJepa21Config:
    return VJepa21Config.from_values(
        source_path=root / "source",
        checkpoint_path=root / "checkpoint.pt",
        checkpoint_sha256=CHECKPOINT_SHA256,
    )


def status_document(*, weights_sha256: str = CHECKPOINT_SHA256) -> dict:
    return {
        "schemaVersion": WORKER_STATUS_SCHEMA_VERSION,
        "service": "creator-forecast-model-worker",
        "state": "ready",
        "executionAvailable": True,
        "branch": "vjepa21",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weightsSha256": weights_sha256,
            "adapterId": ADAPTER_ID,
            "license": MODEL_LICENSE,
        },
        "codeRevision": OFFICIAL_CODE_REVISION,
        "preprocessing": {
            "id": PREPROCESSING_ID,
            "sha256": PREPROCESSING_SHA256,
        },
        "device": "cpu",
        "precision": "fp32",
        "lastError": None,
    }


def identity() -> WorkerIdentity:
    return WorkerIdentity(
        branch="vjepa21",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        model_weights_sha256=CHECKPOINT_SHA256,
        adapter_id=ADAPTER_ID,
        code_revision=OFFICIAL_CODE_REVISION,
        preprocessing_id=PREPROCESSING_ID,
        preprocessing_sha256=PREPROCESSING_SHA256,
    )


class FakeWorker:
    def __init__(self, *, output_hash: str | None = None, ready: bool = True):
        self.output_hash = output_hash
        self.ready = ready
        self.status_calls = 0
        self.extract_calls = []

    def status(self):
        self.status_calls += 1
        value = status_document()
        if not self.ready:
            value.update(
                {
                    "state": "error",
                    "executionAvailable": False,
                    "lastError": "/private/model/path leaked detail",
                }
            )
        return value

    def extract(self, *, video_path, request):
        self.extract_calls.append((Path(video_path), dict(request)))
        digest = self.output_hash or request["inputSha256"]
        return {
            "schemaVersion": WORKER_OUTPUT_SCHEMA_VERSION,
            "branch": "vjepa21",
            "inputSha256": digest,
            "startedAt": "2026-08-19T00:00:00Z",
            "completedAt": "2026-08-19T00:00:01Z",
            "evidenceKind": "learned-visual-representation",
            "features": {"vjepa2_1.embedding_norm_mean": 1.25},
            "observations": [
                {
                    "kind": "learned-visual-window",
                    "startTime": 0.0,
                    "endTime": 10.0,
                    "text": "Descriptive V-JEPA 2.1 representation evidence.",
                    "labels": ["descriptive-only"],
                }
            ],
            "warnings": [],
            "provenance": identity().public_value(),
            "behavioralOutcome": False,
        }


class LocalVJepaAdapterTests(unittest.TestCase):
    def test_absent_required_environment_keeps_local_worker_unavailable(self):
        self.assertIsNone(LocalVJepa21Adapter.optional_from_env({}))
        orchestrator = MultimodalEvidenceOrchestrator.from_env({})
        self.assertNotIn("vjepa21", orchestrator.adapters)

    def test_partial_required_environment_fails_closed_at_startup(self):
        values = {
            SOURCE_PATH_ENV: "/models/vjepa2",
            CHECKPOINT_PATH_ENV: "/models/vjepa21.pt",
            CHECKPOINT_SHA256_ENV: CHECKPOINT_SHA256,
        }
        for missing in values:
            with self.subTest(missing=missing):
                partial = {key: value for key, value in values.items() if key != missing}
                with self.assertRaisesRegex(ConfigurationError, "all three"):
                    LocalVJepa21Adapter.optional_from_env(partial)
                with self.assertRaises(ConfigurationError):
                    MultimodalEvidenceOrchestrator.from_env(partial)

    def test_complete_environment_constructs_injected_worker_without_loading_model(self):
        created = []

        def factory(worker_config):
            created.append(worker_config)
            return FakeWorker()

        values = {
            SOURCE_PATH_ENV: "/models/vjepa2",
            CHECKPOINT_PATH_ENV: "/models/vjepa21.pt",
            CHECKPOINT_SHA256_ENV: CHECKPOINT_SHA256,
        }
        adapter = LocalVJepa21Adapter.optional_from_env(
            values, worker_factory=factory
        )
        self.assertIsNotNone(adapter)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].checkpoint_sha256, CHECKPOINT_SHA256)
        self.assertEqual(created[0].device, "cpu")

    def test_ready_status_is_provenance_validated_and_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = FakeWorker()
            adapter = LocalVJepa21Adapter(config(Path(directory)), worker=worker)
            first = adapter.availability()
            second = adapter.availability()
        self.assertTrue(first["executionAvailable"])
        self.assertEqual(first, second)
        self.assertEqual(worker.status_calls, 1)
        self.assertEqual(
            first["provenance"]["modelWeightsSha256"], CHECKPOINT_SHA256
        )
        self.assertFalse(first["isBehavioralModel"])

    def test_unready_status_is_generic_and_does_not_leak_worker_detail(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = LocalVJepa21Adapter(
                config(Path(directory)), worker=FakeWorker(ready=False)
            )
            availability = adapter.availability()
        self.assertFalse(availability["executionAvailable"])
        self.assertTrue(availability["configured"])
        self.assertNotIn("/private/model/path", availability["reason"])

    def test_run_binds_upload_hash_builds_strict_request_and_returns_branch_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.write_bytes(VIDEO_BYTES)
            worker = FakeWorker()
            adapter = LocalVJepa21Adapter(config(root), worker=worker)
            output = adapter.run(
                video_path=video,
                input_sha256=VIDEO_SHA256,
                duration_seconds=10,
                context={"platform": "youtube-shorts"},
            )

        self.assertEqual(output.input_sha256, VIDEO_SHA256)
        self.assertEqual(output.features["vjepa2_1.embedding_norm_mean"], 1.25)
        self.assertFalse(output.public_value()["behavioralOutcome"])
        self.assertEqual(worker.status_calls, 1)
        self.assertEqual(len(worker.extract_calls), 1)
        request = worker.extract_calls[0][1]
        self.assertEqual(request["schemaVersion"], "creator-forecast-worker-request/1")
        self.assertEqual(request["requestId"], VIDEO_SHA256)
        self.assertEqual(request["inputSha256"], VIDEO_SHA256)

    def test_hash_mismatch_stops_before_readiness_or_extraction_and_forged_output_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.mp4"
            video.write_bytes(VIDEO_BYTES)
            worker = FakeWorker()
            adapter = LocalVJepa21Adapter(config(root), worker=worker)
            with self.assertRaisesRegex(ValueError, "does not match"):
                adapter.run(
                    video_path=video,
                    input_sha256="f" * 64,
                    duration_seconds=10,
                    context={},
                )
            self.assertEqual(worker.status_calls, 0)
            self.assertEqual(worker.extract_calls, [])

            forged = FakeWorker(output_hash="b" * 64)
            forged_adapter = LocalVJepa21Adapter(config(root), worker=forged)
            with self.assertRaisesRegex(WorkerProtocolError, "does not match"):
                forged_adapter.run(
                    video_path=video,
                    input_sha256=VIDEO_SHA256,
                    duration_seconds=10,
                    context={},
                )

    def test_remote_worker_keeps_precedence_over_complete_local_configuration(self):
        remote = object()
        local = object()
        with mock.patch(
            "backend.forecast.orchestrator.build_http_adapters",
            return_value={"vjepa21": remote},
        ), mock.patch(
            "backend.forecast.orchestrator.LocalVJepa21Adapter.optional_from_env",
            return_value=local,
        ):
            orchestrator = MultimodalEvidenceOrchestrator.from_env({})
        self.assertIs(orchestrator.adapters["vjepa21"], remote)


if __name__ == "__main__":
    unittest.main()
