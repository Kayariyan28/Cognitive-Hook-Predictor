from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import httpx

from backend.forecast.adapters import HttpBranchAdapter, WorkerExecutionError
from backend.forecast.execution_registry import WorkerRegistry
from backend.forecast.worker_protocol import WorkerProtocolError


INPUT_SHA = "a" * 64
WEIGHTS_SHA = "2" * 64
CODE_REVISION = "3" * 40
PREPROCESSING_SHA = "4" * 64


def worker_pin():
    return WorkerRegistry.from_mapping(
        {
            "schemaVersion": "creator-forecast-worker-registry/1",
            "workers": {
                "vjepa21": {
                    "baseUrl": "https://worker.example.test",
                    "authTokenEnv": None,
                    "timeoutSeconds": 30,
                    "model": {
                        "id": "facebookresearch/vjepa2.1-vit-base-384",
                        "revision": "1" * 40,
                        "weightsSha256": WEIGHTS_SHA,
                        "license": "MIT",
                        "adapterId": "vjepa21-pytorch",
                    },
                    "codeRevision": CODE_REVISION,
                    "preprocessing": {
                        "id": "vjepa2.1-64f-384/1",
                        "sha256": PREPROCESSING_SHA,
                    },
                }
            },
        }
    ).workers["vjepa21"]


def status():
    return {
        "schemaVersion": "creator-forecast-worker-status/1",
        "service": "creator-forecast-model-worker",
        "state": "ready",
        "executionAvailable": True,
        "branch": "vjepa21",
        "model": {
            "id": "facebookresearch/vjepa2.1-vit-base-384",
            "revision": "1" * 40,
            "weightsSha256": WEIGHTS_SHA,
            "adapterId": "vjepa21-pytorch",
            "license": "MIT",
        },
        "codeRevision": CODE_REVISION,
        "preprocessing": {
            "id": "vjepa2.1-64f-384/1",
            "sha256": PREPROCESSING_SHA,
        },
        "device": "cuda:0",
        "precision": "bf16",
        "lastError": None,
    }


class ForecastAdapterTests(unittest.TestCase):
    def test_http_adapter_probes_and_validates_real_upload_bound_output(self) -> None:
        seen: list[tuple[str, str, bytes]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = request.read()
            seen.append((request.method, request.url.path, body))
            if request.url.path == "/v1/status":
                return httpx.Response(200, json=status())
            self.assertIn(b'filename="clip.mp4"', body)
            self.assertIn(INPUT_SHA.encode(), body)
            identity = {
                "branch": "vjepa21",
                "modelId": "facebookresearch/vjepa2.1-vit-base-384",
                "modelRevision": "1" * 40,
                "modelWeightsSha256": WEIGHTS_SHA,
                "adapterId": "vjepa21-pytorch",
                "codeRevision": CODE_REVISION,
                "preprocessingId": "vjepa2.1-64f-384/1",
                "preprocessingSha256": PREPROCESSING_SHA,
            }
            return httpx.Response(
                200,
                json={
                    "schemaVersion": "creator-forecast-worker-output/1",
                    "branch": "vjepa21",
                    "inputSha256": INPUT_SHA,
                    "startedAt": "2026-08-19T01:00:00Z",
                    "completedAt": "2026-08-19T01:00:01Z",
                    "evidenceKind": "learned-visual-representation",
                    "features": {"vjepa2_1.temporal_consistency": 0.7},
                    "observations": [],
                    "warnings": [],
                    "provenance": identity,
                    "behavioralOutcome": False,
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "clip.mp4"
            video.write_bytes(b"real-upload-bytes")
            adapter = HttpBranchAdapter(
                pin=worker_pin(), token=None, transport=httpx.MockTransport(handler)
            )
            output = adapter.run(
                video_path=video,
                input_sha256=INPUT_SHA,
                duration_seconds=30,
                context={"platform": "youtube-shorts"},
            )
        self.assertEqual(output.features["vjepa2_1.temporal_consistency"], 0.7)
        self.assertEqual([item[:2] for item in seen], [
            ("GET", "/v1/status"),
            ("POST", "/v1/extract"),
        ])

    def test_redirects_and_provenance_mismatches_fail_closed(self) -> None:
        redirect = httpx.MockTransport(
            lambda _request: httpx.Response(
                307, headers={"location": "https://other.example.test/v1/status"}
            )
        )
        with self.assertRaisesRegex(WorkerExecutionError, "HTTP 307"):
            HttpBranchAdapter(worker_pin(), None, redirect).probe()

        mismatch = status()
        mismatch["preprocessing"] = {
            "id": "unreviewed-preprocessing",
            "sha256": PREPROCESSING_SHA,
        }
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(200, json=mismatch)
        )
        with self.assertRaisesRegex(WorkerProtocolError, "preprocessing provenance"):
            HttpBranchAdapter(worker_pin(), None, transport).probe()

    def test_non_json_or_oversized_worker_responses_are_rejected(self) -> None:
        html = httpx.MockTransport(
            lambda _request: httpx.Response(
                200, content=b"<html></html>", headers={"content-type": "text/html"}
            )
        )
        with self.assertRaisesRegex(WorkerExecutionError, "did not return JSON"):
            HttpBranchAdapter(worker_pin(), None, html).probe()


if __name__ == "__main__":
    unittest.main()
