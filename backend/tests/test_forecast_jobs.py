from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.forecast.jobs import (
    BEHAVIORAL_HEAD_KEYS,
    DefaultEvidenceOrchestrator,
    FFprobeMediaProbe,
    ForecastJobExecutionError,
    ForecastJobService,
    ForecastJobStore,
    UploadedVideoEvidence,
    unavailable_behavioral_heads,
)
from backend.forecast.registry import ForecastRegistry
from backend.forecast.router import create_forecast_router


class StubProbe:
    def __init__(self, duration: float = 24.5) -> None:
        self.duration = duration
        self.paths: list[Path] = []

    def probe(self, path: Path) -> float:
        self.paths.append(path)
        if not path.is_file():
            raise AssertionError("probe did not receive the durable uploaded file")
        return self.duration


def stub_output(marker: str = "deterministic") -> dict:
    return {
        "evidence": {
            "schemaVersion": "test-evidence/1",
            "marker": marker,
        },
        "behavioralHeads": unavailable_behavioral_heads(
            "Test orchestrator has no calibrated behavioral heads."
        ),
        "boundaries": ["Deterministic test evidence is not a behavioral forecast."],
    }


class AsyncStubOrchestrator:
    def __init__(self, marker: str = "async-stub") -> None:
        self.marker = marker
        self.calls: list[dict] = []

    async def __call__(
        self, video_path, input_sha256, media_probe, context, stage_callback
    ) -> dict:
        await stage_callback("stub-evidence")
        self.calls.append(
            {
                "pathExists": video_path.is_file(),
                "sha256": input_sha256,
                "mediaProbe": dict(media_probe),
                "context": context,
            }
        )
        return stub_output(self.marker)


class BlockingOrchestrator:
    def __init__(self) -> None:
        self.release = threading.Event()

    async def __call__(
        self, video_path, input_sha256, media_probe, context, stage_callback
    ) -> dict:
        del video_path, input_sha256, media_probe, context
        await stage_callback("waiting-for-stub")
        while not self.release.is_set():
            await asyncio.sleep(0.01)
        return stub_output("released")


def make_app(service: ForecastJobService) -> FastAPI:
    app = FastAPI()
    app.include_router(
        create_forecast_router(
            tribe_status_provider=lambda: {"state": "not-loaded", "configured": False},
            registry=ForecastRegistry.empty(),
            job_service=service,
        )
    )
    return app


def submit_job(
    client: TestClient,
    payload: bytes = b"video-bytes",
    context=None,
    *,
    idempotency_key: str | None = None,
):
    data = {} if context is None else {"context": context}
    headers = (
        {"Idempotency-Key": idempotency_key}
        if idempotency_key is not None
        else None
    )
    return client.post(
        "/api/forecast/v1/jobs",
        files={"video": ("clip.mp4", payload, "video/mp4")},
        data=data,
        headers=headers,
    )


def wait_for_state(
    client: TestClient,
    job_id: str,
    target: set[str],
    timeout: float = 3.0,
) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/forecast/v1/jobs/{job_id}")
        if response.status_code != 200:
            raise AssertionError(response.text)
        last = response.json()
        if last["state"] in target:
            return last
        time.sleep(0.01)
    raise AssertionError(f"job did not reach {target}; last={last}")


class ForecastJobRouteTests(unittest.TestCase):
    def service(
        self,
        directory: str,
        *,
        orchestrator=None,
        probe=None,
        max_upload_bytes: int = 1024 * 1024,
        max_context_bytes: int = 4096,
        max_result_bytes: int = 1024 * 1024,
        max_active_jobs: int = 2,
        job_timeout_seconds: float = 2.0,
    ) -> ForecastJobService:
        return ForecastJobService(
            store=ForecastJobStore(Path(directory)),
            orchestrator=orchestrator or DefaultEvidenceOrchestrator(),
            media_probe=probe or StubProbe(),
            max_upload_bytes=max_upload_bytes,
            max_context_bytes=max_context_bytes,
            max_result_bytes=max_result_bytes,
            max_active_jobs=max_active_jobs,
            max_workers=1,
            heartbeat_seconds=0.02,
            job_timeout_seconds=job_timeout_seconds,
        )

    def test_environment_bounds_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "cannot exceed"):
                ForecastJobService.from_env(
                    environ={
                        "FORECAST_JOB_DIR": directory,
                        "FORECAST_MAX_ACTIVE_JOBS": "1",
                        "FORECAST_MAX_WORKERS": "2",
                    }
                )
            with self.assertRaisesRegex(ValueError, "non-empty"):
                ForecastJobService.from_env(environ={"FORECAST_JOB_DIR": ""})
            with self.assertRaisesRegex(ValueError, "greater than zero"):
                ForecastJobService.from_env(
                    environ={
                        "FORECAST_JOB_DIR": directory,
                        "FORECAST_MAX_RESULT_BYTES": "0",
                    }
                )

    def test_default_job_publishes_private_evidence_without_invented_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(directory)
            with TestClient(make_app(service)) as client:
                response = submit_job(
                    client,
                    b"definitely-streamed",
                    json.dumps({"platform": "example-shorts", "topic": "travel"}),
                )
                self.assertEqual(response.status_code, 202)
                self.assertEqual(response.headers["cache-control"], "private, no-store")
                self.assertEqual(response.headers["pragma"], "no-cache")
                job = response.json()
                self.assertEqual(job["state"], "queued")
                self.assertNotIn("progress", job)
                self.assertEqual(response.headers["location"], job["links"]["self"])

                complete = wait_for_state(client, job["jobId"], {"complete"})
                self.assertEqual(complete["stage"], "complete")
                self.assertEqual(complete["input"]["durationSeconds"], 24.5)
                self.assertGreaterEqual(complete["elapsedSeconds"], 0)
                self.assertIsInstance(complete["heartbeatAt"], str)

                result_response = client.get(complete["links"]["result"])
                self.assertEqual(result_response.status_code, 200)
                self.assertEqual(
                    result_response.headers["cache-control"], "private, no-store"
                )
                result = result_response.json()
                self.assertEqual(result["schemaVersion"], "creator-forecast-result/1")
                self.assertEqual(
                    result["input"]["sha256"],
                    hashlib.sha256(b"definitely-streamed").hexdigest(),
                )
                self.assertEqual(result["evidence"]["videoMetadata"]["source"], "server-ffprobe")
                self.assertEqual(
                    result["evidence"]["creatorContext"]["value"]["topic"], "travel"
                )
                self.assertEqual(set(result["behavioralHeads"]), set(BEHAVIORAL_HEAD_KEYS))
                for head in result["behavioralHeads"].values():
                    self.assertEqual(head["status"], "unavailable")
                    self.assertIsNone(head["value"])
                    self.assertEqual(head["validation"], "not-calibrated")

            root = Path(directory)
            self.assertFalse(any((root / "jobs" / job["jobId"]).glob("input.*")))
            self.assertTrue((root / "results" / job["jobId"] / "result.json").is_file())
            self.assertFalse(any((root / "results").glob(".publishing-*")))

    def test_injected_async_orchestrator_receives_the_five_argument_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = AsyncStubOrchestrator()
            probe = StubProbe(31.25)
            service = self.service(directory, orchestrator=orchestrator, probe=probe)
            with TestClient(make_app(service)) as client:
                response = submit_job(client, b"five-argument-input", '{"genre":"tutorial"}')
                job = wait_for_state(client, response.json()["jobId"], {"complete"})
                result = client.get(job["links"]["result"]).json()

            self.assertEqual(result["evidence"]["marker"], "async-stub")
            self.assertEqual(len(orchestrator.calls), 1)
            call = orchestrator.calls[0]
            self.assertTrue(call["pathExists"])
            self.assertEqual(
                call["sha256"], hashlib.sha256(b"five-argument-input").hexdigest()
            )
            self.assertEqual(call["mediaProbe"]["durationSeconds"], 31.25)
            self.assertEqual(call["context"], {"genre": "tutorial"})

    def test_idempotency_key_replays_one_durable_job_without_duplicate_execution(self) -> None:
        idempotency_key = "12345678-1234-4234-9234-123456789abc"
        expected_job_id = "12345678123442349234123456789abc"
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = AsyncStubOrchestrator()
            service = self.service(directory, orchestrator=orchestrator)
            with TestClient(make_app(service)) as client:
                first_response = submit_job(
                    client,
                    b"idempotent-input",
                    '{"genre":"tutorial"}',
                    idempotency_key=idempotency_key,
                )
                self.assertEqual(first_response.status_code, 202)
                self.assertEqual(first_response.headers["idempotency-key"], expected_job_id)
                first = first_response.json()
                self.assertEqual(first["jobId"], expected_job_id)
                wait_for_state(client, expected_job_id, {"complete"})

                replay_response = submit_job(
                    client,
                    b"a-replay-body-is-not-a-second-job",
                    '{"genre":"different"}',
                    idempotency_key=idempotency_key,
                )
                replay = replay_response.json()
                self.assertEqual(replay_response.status_code, 202)
                self.assertEqual(replay["jobId"], expected_job_id)
                self.assertEqual(replay["state"], "complete")
                self.assertEqual(len(orchestrator.calls), 1)

    def test_invalid_idempotency_key_fails_before_job_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(directory)
            with TestClient(make_app(service)) as client:
                response = submit_job(
                    client,
                    idempotency_key="not-a-uuid",
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(
                    response.json()["detail"]["code"],
                    "invalid_idempotency_key",
                )
            self.assertEqual(list((Path(directory) / "jobs").iterdir()), [])

    def test_sync_callable_orchestrator_can_report_a_stage_from_its_worker_thread(self) -> None:
        calls = []

        def synchronous(video_path, input_sha256, media_probe, context, stage_callback):
            stage_callback("sync-model-stage")
            calls.append(
                (video_path.is_file(), input_sha256, dict(media_probe), context)
            )
            return stub_output("sync-stub")

        with tempfile.TemporaryDirectory() as directory:
            service = self.service(directory, orchestrator=synchronous)
            with TestClient(make_app(service)) as client:
                response = submit_job(client, b"sync-input")
                job = wait_for_state(client, response.json()["jobId"], {"complete"})
                result = client.get(job["links"]["result"]).json()
        self.assertEqual(result["evidence"]["marker"], "sync-stub")
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][0])

    def test_queue_capacity_and_not_ready_result_are_explicit_and_private(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = BlockingOrchestrator()
            service = self.service(
                directory, orchestrator=orchestrator, max_active_jobs=1
            )
            with TestClient(make_app(service)) as client:
                first = submit_job(client, b"first")
                job_id = first.json()["jobId"]
                running = wait_for_state(client, job_id, {"running"})
                self.assertIn(running["stage"], {"orchestrating-evidence", "waiting-for-stub"})

                pending = client.get(f"/api/forecast/v1/results/{job_id}")
                self.assertEqual(pending.status_code, 409)
                self.assertEqual(pending.json()["detail"]["code"], "result_not_ready")
                self.assertEqual(pending.headers["cache-control"], "private, no-store")

                second = submit_job(client, b"second")
                self.assertEqual(second.status_code, 429)
                self.assertEqual(
                    second.json()["detail"]["code"], "forecast_capacity_reached"
                )
                orchestrator.release.set()
                wait_for_state(client, job_id, {"complete"})

    def test_queued_jobs_receive_durable_heartbeats_without_fake_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = BlockingOrchestrator()
            service = self.service(
                directory, orchestrator=orchestrator, max_active_jobs=2
            )
            with TestClient(make_app(service)) as client:
                first = submit_job(client, b"worker-slot")
                wait_for_state(client, first.json()["jobId"], {"running"})
                second = submit_job(client, b"queued-slot").json()
                self.assertEqual(second["state"], "queued")
                initial_heartbeat = second["heartbeatAt"]
                time.sleep(0.05)
                queued = client.get(second["links"]["self"]).json()
                self.assertEqual(queued["state"], "queued")
                self.assertNotEqual(queued["heartbeatAt"], initial_heartbeat)
                self.assertNotIn("progress", queued)
                orchestrator.release.set()
                wait_for_state(client, first.json()["jobId"], {"complete"})
                wait_for_state(client, second["jobId"], {"complete"})

    def test_upload_and_context_limits_fail_before_a_job_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(
                directory, max_upload_bytes=4, max_context_bytes=12
            )
            with TestClient(make_app(service)) as client:
                oversized_video = submit_job(client, b"12345")
                self.assertEqual(oversized_video.status_code, 413)
                self.assertEqual(
                    oversized_video.json()["detail"]["code"], "video_too_large"
                )
                oversized_context = submit_job(client, b"1234", '{"long":"value"}')
                self.assertEqual(oversized_context.status_code, 413)
                self.assertEqual(
                    oversized_context.json()["detail"]["code"], "context_too_large"
                )
            root = Path(directory)
            self.assertEqual(list((root / "jobs").iterdir()), [])
            self.assertEqual(list((root / "incoming").iterdir()), [])

    def test_context_video_type_and_duplicate_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(directory)
            with TestClient(make_app(service)) as client:
                duplicate = submit_job(client, context='{"topic":"a","topic":"b"}')
                self.assertEqual(duplicate.status_code, 400)
                self.assertEqual(duplicate.json()["detail"]["code"], "invalid_context")

                array_context = submit_job(client, context="[]")
                self.assertEqual(array_context.status_code, 400)

                bad_suffix = client.post(
                    "/api/forecast/v1/jobs",
                    files={"video": ("clip.txt", b"video", "video/mp4")},
                )
                self.assertEqual(bad_suffix.status_code, 400)

                bad_media_type = client.post(
                    "/api/forecast/v1/jobs",
                    files={"video": ("clip.mp4", b"video", "text/plain")},
                )
                self.assertEqual(bad_media_type.status_code, 400)

    def test_authoritative_duration_failure_is_durable_and_has_no_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(directory, probe=StubProbe(9.999))
            with TestClient(make_app(service)) as client:
                response = submit_job(client)
                failed = wait_for_state(client, response.json()["jobId"], {"failed"})
                self.assertEqual(failed["error"]["code"], "duration_out_of_range")
                self.assertIsNone(failed["resultId"])
                result = client.get(f"/api/forecast/v1/results/{failed['jobId']}")
                self.assertEqual(result.status_code, 409)
                self.assertEqual(result.json()["detail"]["code"], "job_failed")

    def test_authoritative_duration_boundaries_are_inclusive(self) -> None:
        for duration in (10.0, 60.0):
            with self.subTest(duration=duration), tempfile.TemporaryDirectory() as directory:
                service = self.service(directory, probe=StubProbe(duration))
                with TestClient(make_app(service)) as client:
                    response = submit_job(client)
                    complete = wait_for_state(
                        client, response.json()["jobId"], {"complete"}
                    )
                    self.assertEqual(complete["input"]["durationSeconds"], duration)

    def test_malformed_probe_duration_fails_as_media_metadata_not_orchestration(self) -> None:
        for duration in ("not-a-number", float("nan")):
            with self.subTest(duration=duration), tempfile.TemporaryDirectory() as directory:
                service = self.service(directory, probe=StubProbe(duration))
                with TestClient(make_app(service)) as client:
                    response = submit_job(client)
                    failed = wait_for_state(
                        client, response.json()["jobId"], {"failed"}
                    )
                    self.assertEqual(
                        failed["error"]["code"], "invalid_media_metadata"
                    )

    def test_invalid_orchestrator_output_fails_closed(self) -> None:
        async def invalid(video_path, input_sha256, media_probe, context, stage_callback):
            del video_path, input_sha256, media_probe, context, stage_callback
            return {"evidence": {"inventedScore": 99}}

        with tempfile.TemporaryDirectory() as directory:
            service = self.service(directory, orchestrator=invalid)
            with TestClient(make_app(service)) as client:
                response = submit_job(client)
                failed = wait_for_state(client, response.json()["jobId"], {"failed"})
                self.assertEqual(
                    failed["error"]["code"], "invalid_orchestrator_output"
                )
                self.assertFalse((Path(directory) / "results" / failed["jobId"]).exists())

    def test_unexpected_orchestrator_exception_persists_generic_failure(self) -> None:
        async def crashes(video_path, input_sha256, media_probe, context, stage_callback):
            del video_path, input_sha256, media_probe, context, stage_callback
            raise RuntimeError("private executor detail")

        with tempfile.TemporaryDirectory() as directory:
            service = self.service(directory, orchestrator=crashes)
            with TestClient(make_app(service)) as client:
                response = submit_job(client)
                failed = wait_for_state(client, response.json()["jobId"], {"failed"})
                self.assertEqual(failed["error"]["code"], "orchestrator_failed")
                self.assertEqual(
                    failed["error"]["message"],
                    "The forecast evidence orchestrator failed; no result was published.",
                )
                self.assertNotIn("private executor detail", json.dumps(failed))
                self.assertFalse((Path(directory) / "results" / failed["jobId"]).exists())

    def test_orchestrator_runtime_is_bounded_by_a_real_timeout(self) -> None:
        orchestrator = BlockingOrchestrator()
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(
                directory,
                orchestrator=orchestrator,
                job_timeout_seconds=0.04,
            )
            with TestClient(make_app(service)) as client:
                response = submit_job(client)
                failed = wait_for_state(client, response.json()["jobId"], {"failed"})
                self.assertEqual(failed["error"]["code"], "orchestrator_timeout")
                self.assertIsNone(failed["links"]["result"])

    def test_timed_out_sync_runner_holds_capacity_and_cannot_rewrite_terminal_stage(self) -> None:
        def slow_sync(video_path, input_sha256, media_probe, context, stage_callback):
            del video_path, input_sha256, media_probe, context
            stage_callback("sync-started")
            time.sleep(0.12)
            stage_callback("sync-finished-late")
            return stub_output("too-late")

        with tempfile.TemporaryDirectory() as directory:
            service = self.service(
                directory,
                orchestrator=slow_sync,
                max_active_jobs=1,
                job_timeout_seconds=0.03,
            )
            with TestClient(make_app(service)) as client:
                first = submit_job(client)
                failed = wait_for_state(client, first.json()["jobId"], {"failed"})
                self.assertEqual(failed["error"]["code"], "orchestrator_timeout")
                stored = service.store.read_job(failed["jobId"])
                self.assertIsNotNone(stored)
                input_path = service.store.input_path(stored)
                self.assertTrue(input_path.is_file())

                capacity = submit_job(client, b"must-wait")
                self.assertEqual(capacity.status_code, 429)
                time.sleep(0.14)
                still_failed = client.get(failed["links"]["self"]).json()
                self.assertEqual(still_failed["state"], "failed")
                self.assertEqual(still_failed["stage"], "failed")
                self.assertFalse(input_path.exists())

    def test_orchestrator_result_bytes_are_bounded_before_atomic_publish(self) -> None:
        async def oversized(video_path, input_sha256, media_probe, context, stage_callback):
            del video_path, input_sha256, media_probe, context, stage_callback
            output = stub_output()
            output["evidence"]["large"] = "x" * 2048
            return output

        with tempfile.TemporaryDirectory() as directory:
            service = self.service(
                directory,
                orchestrator=oversized,
                max_result_bytes=512,
            )
            with TestClient(make_app(service)) as client:
                response = submit_job(client)
                failed = wait_for_state(client, response.json()["jobId"], {"failed"})
                self.assertEqual(
                    failed["error"]["code"], "orchestrator_result_too_large"
                )
                self.assertFalse((Path(directory) / "results" / failed["jobId"]).exists())

    def test_startup_marks_persisted_active_jobs_interrupted_and_discards_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ForecastJobStore(Path(directory))
            incoming = store.incoming_path(".mp4")
            incoming.write_bytes(b"interrupted")
            job_id = "a" * 32
            record = store.create_job(
                job_id=job_id,
                incoming_path=incoming,
                upload=UploadedVideoEvidence(
                    size_bytes=11,
                    sha256=hashlib.sha256(b"interrupted").hexdigest(),
                    suffix=".mp4",
                    content_type="video/mp4",
                ),
                context=None,
            )
            self.assertTrue(store.input_path(record).is_file())

            service = self.service(directory)
            with TestClient(make_app(service)) as client:
                response = client.get(f"/api/forecast/v1/jobs/{job_id}")
                self.assertEqual(response.status_code, 200)
                recovered = response.json()
                self.assertEqual(recovered["state"], "failed")
                self.assertEqual(recovered["stage"], "interrupted")
                self.assertEqual(recovered["error"]["code"], "worker_interrupted")
            self.assertFalse(store.input_path(record).exists())

    def test_startup_repairs_a_job_when_its_atomic_result_was_already_published(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ForecastJobStore(Path(directory))
            incoming = store.incoming_path(".mp4")
            incoming.write_bytes(b"published-before-restart")
            job_id = "b" * 32
            record = store.create_job(
                job_id=job_id,
                incoming_path=incoming,
                upload=UploadedVideoEvidence(
                    size_bytes=24,
                    sha256=hashlib.sha256(b"published-before-restart").hexdigest(),
                    suffix=".mp4",
                    content_type="video/mp4",
                ),
                context=None,
            )
            store.publish_result(
                job_id,
                {
                    "schemaVersion": "creator-forecast-result/1",
                    "resultId": job_id,
                    "jobId": job_id,
                    "createdAt": "2026-08-19T00:00:00Z",
                    "input": {},
                    "evidence": {},
                    "behavioralHeads": {},
                    "boundaries": [],
                },
            )
            abandoned_staging = store.jobs_dir / f".creating-{job_id}-stale"
            abandoned_staging.mkdir()
            (abandoned_staging / "input.mp4").write_bytes(b"partial")

            service = self.service(directory)
            with TestClient(make_app(service)) as client:
                recovered = client.get(f"/api/forecast/v1/jobs/{job_id}").json()
                self.assertEqual(recovered["state"], "complete")
                self.assertEqual(recovered["resultId"], job_id)
                self.assertEqual(
                    client.get(recovered["links"]["result"]).status_code, 200
                )
            self.assertFalse(store.input_path(record).exists())
            self.assertFalse(abandoned_staging.exists())


class FFprobeMediaProbeTests(unittest.TestCase):
    def test_probe_uses_ffprobe_json_and_requires_a_video_stream(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [{"codec_type": "video", "duration": "21.0"}],
                    "format": {"duration": "20.75"},
                }
            ),
            stderr="",
        )
        with patch("backend.forecast.jobs.subprocess.run", return_value=completed) as run:
            duration = FFprobeMediaProbe("ffprobe-custom", 7).probe(Path("clip.mp4"))
        self.assertEqual(duration, 20.75)
        command = run.call_args.args[0]
        self.assertEqual(command[0], "ffprobe-custom")
        self.assertIn("format=duration:stream=codec_type,duration", command)
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs["timeout"], 7.0)

    def test_probe_rejects_missing_video_invalid_json_and_missing_binary(self) -> None:
        no_video = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='{"streams":[{"codec_type":"audio"}]}', stderr=""
        )
        with patch("backend.forecast.jobs.subprocess.run", return_value=no_video):
            with self.assertRaisesRegex(ForecastJobExecutionError, "video stream"):
                FFprobeMediaProbe().probe(Path("clip.mp4"))

        malformed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not-json", stderr=""
        )
        with patch("backend.forecast.jobs.subprocess.run", return_value=malformed):
            with self.assertRaisesRegex(ForecastJobExecutionError, "malformed"):
                FFprobeMediaProbe().probe(Path("clip.mp4"))

        with patch("backend.forecast.jobs.subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(ForecastJobExecutionError, "unavailable"):
                FFprobeMediaProbe().probe(Path("clip.mp4"))


if __name__ == "__main__":
    unittest.main()
