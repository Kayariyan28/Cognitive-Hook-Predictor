from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from backend.forecast.registry import ForecastRegistry
from backend.forecast.worker_protocol import (
    WORKER_OUTPUT_SCHEMA_VERSION,
    WORKER_STATUS_SCHEMA_VERSION,
    WorkerIdentity,
    WorkerProtocolError,
    build_worker_request,
    validate_branch_output,
)


REVISION = "1" * 40
WEIGHTS = b"vjepa-2.1-test-artifact"
WEIGHTS_SHA = hashlib.sha256(WEIGHTS).hexdigest()


def registry_pin(root: Path):
    path = root / "vjepa21.pt"
    path.write_bytes(WEIGHTS)
    return ForecastRegistry.from_mapping(
        {
            "schemaVersion": "creator-forecast-registry/1",
            "branches": {
                "vjepa21": {
                    "id": "facebookresearch/vjepa2.1-vit-base-384",
                    "revision": REVISION,
                    "artifactPath": str(path),
                    "artifactSha256": WEIGHTS_SHA,
                    "license": "MIT",
                    "adapterId": "vjepa21-pytorch",
                }
            },
            "heads": {},
        },
        base_directory=root,
    ).branches["vjepa21"]


def status_document():
    return {
        "schemaVersion": WORKER_STATUS_SCHEMA_VERSION,
        "service": "creator-forecast-model-worker",
        "state": "ready",
        "executionAvailable": True,
        "branch": "vjepa21",
        "model": {
            "id": "facebookresearch/vjepa2.1-vit-base-384",
            "revision": REVISION,
            "weightsSha256": WEIGHTS_SHA,
            "adapterId": "vjepa21-pytorch",
            "license": "MIT",
        },
        "codeRevision": "2" * 40,
        "preprocessing": {"id": "vjepa2.1-64f-384/1", "sha256": "3" * 64},
        "device": "cuda:0",
        "precision": "bf16",
        "lastError": None,
    }


def output_document(identity: WorkerIdentity, input_sha256: str):
    return {
        "schemaVersion": WORKER_OUTPUT_SCHEMA_VERSION,
        "branch": "vjepa21",
        "inputSha256": input_sha256,
        "startedAt": "2026-08-19T01:00:00Z",
        "completedAt": "2026-08-19T01:00:02Z",
        "evidenceKind": "learned-visual-representation",
        "features": {
            "vjepa2_1.motion_change_mean": 0.125,
            "vjepa2_1.temporal_consistency": 0.875,
        },
        "observations": [
            {
                "kind": "representation-change",
                "startTime": 2.0,
                "endTime": 3.0,
                "text": "Largest within-clip representation change.",
                "labels": ["relative-within-clip"],
            }
        ],
        "warnings": [],
        "provenance": identity.public_value(),
        "behavioralOutcome": False,
    }


class ForecastWorkerProtocolTests(unittest.TestCase):
    def test_status_must_exactly_match_registry_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pin = registry_pin(Path(directory))
            identity = WorkerIdentity.from_status(
                status_document(), expected_branch="vjepa21", expected_pin=pin
            )
            self.assertEqual(identity.model_weights_sha256, WEIGHTS_SHA)

            mismatched = status_document()
            mismatched["model"] = {**mismatched["model"], "revision": "4" * 40}
            with self.assertRaisesRegex(WorkerProtocolError, "provenance"):
                WorkerIdentity.from_status(
                    mismatched, expected_branch="vjepa21", expected_pin=pin
                )

    def test_feature_output_is_finite_hash_bound_and_non_behavioral(self) -> None:
        input_sha256 = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            pin = registry_pin(Path(directory))
            identity = WorkerIdentity.from_status(
                status_document(), expected_branch="vjepa21", expected_pin=pin
            )
            output = validate_branch_output(
                output_document(identity, input_sha256),
                expected_branch="vjepa21",
                expected_input_sha256=input_sha256,
                identity=identity,
            )
            self.assertEqual(output.features["vjepa2_1.temporal_consistency"], 0.875)
            self.assertFalse(output.public_value()["behavioralOutcome"])

            forged = output_document(identity, input_sha256)
            forged["features"] = {"vjepa2_1.virality": 99}
            with self.assertRaisesRegex(WorkerProtocolError, "evidence boundary"):
                validate_branch_output(
                    forged,
                    expected_branch="vjepa21",
                    expected_input_sha256=input_sha256,
                    identity=identity,
                )

            behavioral = output_document(identity, input_sha256)
            behavioral["behavioralOutcome"] = True
            with self.assertRaisesRegex(WorkerProtocolError, "behavioral"):
                validate_branch_output(
                    behavioral,
                    expected_branch="vjepa21",
                    expected_input_sha256=input_sha256,
                    identity=identity,
                )

    def test_worker_request_enforces_real_upload_hash_and_duration(self) -> None:
        request = build_worker_request(
            branch="vjepa21",
            input_sha256="a" * 64,
            duration_seconds=10,
            context={"platform": "youtube-shorts"},
        )
        self.assertEqual(request["durationSeconds"], 10.0)
        self.assertEqual(request["inputSha256"], "a" * 64)
        with self.assertRaisesRegex(WorkerProtocolError, "10 to 60"):
            build_worker_request(
                branch="vjepa21",
                input_sha256="a" * 64,
                duration_seconds=9.99,
                context={},
            )


if __name__ == "__main__":
    unittest.main()
