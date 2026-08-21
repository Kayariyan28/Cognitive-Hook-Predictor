"""Bounded, durable creator-forecast evidence jobs.

This module deliberately has no dependency on the TRIBE runtime.  The default
orchestrator publishes authoritative upload/media evidence and explicit
unavailability for every behavioral head; it never derives a score from media
metadata or creator-supplied context.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Awaitable, Callable, Mapping, Protocol
from uuid import UUID, uuid4

from fastapi import UploadFile

from .schema import MAXIMUM_DURATION_SECONDS, MINIMUM_DURATION_SECONDS


LOGGER = logging.getLogger("forecast_jobs")

JOB_SCHEMA_VERSION = "creator-forecast-job/1"
RESULT_SCHEMA_VERSION = "creator-forecast-result/1"
EVIDENCE_SCHEMA_VERSION = "creator-forecast-evidence/1"
ACTIVE_STATES = frozenset({"queued", "probing", "running"})
TERMINAL_STATES = frozenset({"complete", "failed"})
ALL_STATES = ACTIVE_STATES | TERMINAL_STATES
STATE_TRANSITIONS = {
    "queued": frozenset({"probing", "failed"}),
    "probing": frozenset({"running", "failed"}),
    "running": frozenset({"complete", "failed"}),
    "complete": frozenset(),
    "failed": frozenset(),
}
VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mkv", ".mov", ".webm"})
STAGE_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
IDEMPOTENCY_KEY_RE = re.compile(
    r"^(?:[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
    re.IGNORECASE,
)
UPLOAD_CHUNK_BYTES = 1024 * 1024
DEFAULT_MAX_UPLOAD_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_CONTEXT_BYTES = 32 * 1024
DEFAULT_MAX_RESULT_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_ACTIVE_JOBS = 4
DEFAULT_MAX_WORKERS = 1
DEFAULT_PROBE_TIMEOUT_SECONDS = 30.0
DEFAULT_HEARTBEAT_SECONDS = 5.0
DEFAULT_JOB_TIMEOUT_SECONDS = 600.0
OPTIONAL_PROVIDER_KEYS = (
    "vjepa21",
    "videoLlama21Av",
    "audioModel",
    "account",
    "trends",
    "competitors",
)
BEHAVIORAL_HEAD_KEYS = (
    "hookStrength",
    "retention5Second",
    "completionProbability",
    "rewatchPotential",
    "sharePotential",
    "predictedEngagement",
    "viralityPotential",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp is not a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp has no timezone")
    return parsed


def _elapsed_seconds(record: Mapping[str, Any]) -> float:
    try:
        start = _parse_timestamp(record["createdAt"])
        end_value = record.get("completedAt") or _utc_now()
        end = _parse_timestamp(end_value)
    except (KeyError, TypeError, ValueError):
        return 0.0
    return round(max(0.0, (end - start).total_seconds()), 3)


def _positive_int_from_env(
    environ: Mapping[str, str], name: str, default: int
) -> int:
    raw = environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _positive_float_from_env(
    environ: Mapping[str, str], name: str, default: float
) -> float:
    raw = environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def normalize_idempotency_key(value: str | None) -> str | None:
    """Normalize one client-generated UUID without accepting arbitrary job IDs."""

    if value is None:
        return None
    raw = value.strip().lower()
    if not IDEMPOTENCY_KEY_RE.fullmatch(raw):
        raise ForecastJobAPIError(
            400,
            "invalid_idempotency_key",
            "Idempotency-Key must be a UUID.",
        )
    try:
        return UUID(raw).hex
    except ValueError as exc:
        raise ForecastJobAPIError(
            400,
            "invalid_idempotency_key",
            "Idempotency-Key must be a UUID.",
        ) from exc


class ForecastJobAPIError(Exception):
    """A safe, caller-visible job admission or lookup error."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class ForecastJobExecutionError(Exception):
    """A safe, persisted worker failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class JobStoreError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class UploadedVideoEvidence:
    size_bytes: int
    sha256: str
    suffix: str
    content_type: str | None


StageCallback = Callable[[str], Any]


class ForecastOrchestrator(Protocol):
    def __call__(
        self,
        video_path: Path,
        input_sha256: str,
        media_probe: Mapping[str, Any],
        context: Mapping[str, Any] | None,
        stage_callback: StageCallback,
    ) -> Mapping[str, Any] | Awaitable[Mapping[str, Any]]:
        ...


def unavailable_behavioral_heads(reason: str) -> dict[str, dict[str, Any]]:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("unavailable behavioral-head reason must be non-empty")
    return {
        key: {
            "status": "unavailable",
            "value": None,
            "validation": "not-calibrated",
            "reason": reason,
        }
        for key in BEHAVIORAL_HEAD_KEYS
    }


class DefaultEvidenceOrchestrator:
    """Available-now server evidence with no invented behavioral output."""

    async def __call__(
        self,
        video_path: Path,
        input_sha256: str,
        media_probe: Mapping[str, Any],
        context: Mapping[str, Any] | None,
        stage_callback: StageCallback,
    ) -> Mapping[str, Any]:
        del video_path
        await stage_callback("assembling-evidence")
        reason = (
            "No executable, target-specific production-calibrated behavioral "
            "head is configured for this service."
        )
        optional_providers = {
            key: {
                "status": "unavailable",
                "forecastContribution": False,
                "reason": "No executable provider adapter is configured for this job layer.",
            }
            for key in OPTIONAL_PROVIDER_KEYS
        }
        return {
            "evidence": {
                "schemaVersion": EVIDENCE_SCHEMA_VERSION,
                "videoMetadata": {
                    "status": "available",
                    "source": "server-ffprobe",
                    "sha256": input_sha256,
                    "sizeBytes": media_probe["sizeBytes"],
                    "durationSeconds": media_probe["durationSeconds"],
                    "contentType": media_probe.get("contentType"),
                },
                "creatorContext": {
                    "status": "provided" if context is not None else "not-provided",
                    "value": context,
                    "forecastContribution": False,
                },
                "optionalProviders": optional_providers,
                "tribe": {
                    "status": "not-invoked",
                    "forecastContribution": False,
                    "reason": "Forecast jobs do not call or alter the separate TRIBE cortical pipeline.",
                },
            },
            "behavioralHeads": unavailable_behavioral_heads(reason),
            "boundaries": [
                "Media metadata and creator context are evidence, not audience-outcome probabilities.",
                "No behavioral value is emitted without an executable production-calibrated head.",
                "TRIBE cortical output is not used by this forecast job.",
            ],
        }


class FFprobeMediaProbe:
    """Authoritative duration probe using ffprobe without a shell."""

    def __init__(self, binary: str = "ffprobe", timeout_seconds: float = 30.0) -> None:
        if not isinstance(binary, str) or not binary.strip():
            raise ValueError("FORECAST_FFPROBE_BINARY must be non-empty")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("ffprobe timeout must be positive and finite")
        self.binary = binary.strip()
        self.timeout_seconds = float(timeout_seconds)

    def probe(self, video_path: Path) -> float:
        command = [
            self.binary,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_entries",
            "format=duration:stream=codec_type,duration",
            str(video_path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise ForecastJobExecutionError(
                "ffprobe_unavailable",
                "The configured ffprobe executable is unavailable.",
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ForecastJobExecutionError(
                "probe_timeout",
                "The authoritative media probe exceeded its time limit.",
            ) from exc
        except OSError as exc:
            raise ForecastJobExecutionError(
                "probe_failed",
                "The authoritative media probe could not be started.",
            ) from exc

        if completed.returncode != 0:
            raise ForecastJobExecutionError(
                "invalid_media",
                "ffprobe could not decode authoritative metadata from the upload.",
            )
        try:
            metadata = json.loads(completed.stdout)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ForecastJobExecutionError(
                "invalid_media_metadata",
                "ffprobe returned malformed media metadata.",
            ) from exc
        if not isinstance(metadata, dict):
            raise ForecastJobExecutionError(
                "invalid_media_metadata",
                "ffprobe returned malformed media metadata.",
            )

        streams = metadata.get("streams")
        if not isinstance(streams, list) or not any(
            isinstance(stream, dict) and stream.get("codec_type") == "video"
            for stream in streams
        ):
            raise ForecastJobExecutionError(
                "video_stream_missing",
                "The uploaded media has no decodable video stream.",
            )

        candidates: list[Any] = []
        format_record = metadata.get("format")
        if isinstance(format_record, dict):
            candidates.append(format_record.get("duration"))
        candidates.extend(
            stream.get("duration")
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "video"
        )
        for raw_duration in candidates:
            try:
                duration = float(raw_duration)
            except (TypeError, ValueError):
                continue
            if math.isfinite(duration) and duration > 0:
                return duration
        raise ForecastJobExecutionError(
            "duration_unavailable",
            "ffprobe could not determine a finite positive video duration.",
        )


class ForecastJobStore:
    """Filesystem-backed state with atomic JSON and result publication."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.jobs_dir = self.root / "jobs"
        self.results_dir = self.root / "results"
        self.incoming_dir = self.root / "incoming"

    def initialize(self) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.incoming_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    def _write_json_atomic(self, path: Path, payload: Mapping[str, Any]) -> None:
        try:
            encoded = (
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise JobStoreError("job payload is not strict JSON") from exc
        temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            self._fsync_directory(path.parent)
        except OSError as exc:
            raise JobStoreError("could not persist forecast job state") from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise JobStoreError("stored forecast JSON is unreadable") from exc
        if not isinstance(payload, dict):
            raise JobStoreError("stored forecast JSON is not an object")
        return payload

    def incoming_path(self, suffix: str) -> Path:
        self.initialize()
        return self.incoming_dir / f"upload-{uuid4().hex}{suffix}"

    def create_job(
        self,
        *,
        job_id: str,
        incoming_path: Path,
        upload: UploadedVideoEvidence,
        context: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        job_directory = self.jobs_dir / job_id
        staging_directory = self.jobs_dir / f".creating-{job_id}-{uuid4().hex}"
        try:
            staging_directory.mkdir(parents=False, exist_ok=False)
            input_name = f"input{upload.suffix}"
            input_path = staging_directory / input_name
            os.replace(incoming_path, input_path)
            now = _utc_now()
            record: dict[str, Any] = {
                "schemaVersion": JOB_SCHEMA_VERSION,
                "jobId": job_id,
                "state": "queued",
                "stage": "queued",
                "createdAt": now,
                "updatedAt": now,
                "heartbeatAt": now,
                "startedAt": None,
                "completedAt": None,
                "resultId": None,
                "error": None,
                "inputFile": input_name,
                "input": {
                    "sha256": upload.sha256,
                    "sizeBytes": upload.size_bytes,
                    "contentType": upload.content_type,
                    "durationSeconds": None,
                },
                "context": context,
            }
            self._write_json_atomic(staging_directory / "job.json", record)
            os.replace(staging_directory, job_directory)
            self._fsync_directory(self.jobs_dir)
            return record
        except Exception:
            shutil.rmtree(staging_directory, ignore_errors=True)
            # The destination may belong to a concurrent process admitting the
            # same idempotency identity. Never delete an already-published job.
            raise

    def read_job(self, job_id: str) -> dict[str, Any] | None:
        record = self._read_json(self.jobs_dir / job_id / "job.json")
        if record is None:
            return None
        if record.get("jobId") != job_id or record.get("state") not in ALL_STATES:
            raise JobStoreError("stored forecast job has an invalid identity or state")
        return record

    def write_job(self, record: Mapping[str, Any]) -> None:
        job_id = record.get("jobId")
        if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise JobStoreError("forecast job id is invalid")
        self._write_json_atomic(self.jobs_dir / job_id / "job.json", record)

    def input_path(self, record: Mapping[str, Any]) -> Path:
        job_id = record.get("jobId")
        input_name = record.get("inputFile")
        if (
            not isinstance(job_id, str)
            or not re.fullmatch(r"[0-9a-f]{32}", job_id)
            or not isinstance(input_name, str)
            or Path(input_name).name != input_name
        ):
            raise JobStoreError("stored forecast input identity is invalid")
        return self.jobs_dir / job_id / input_name

    def discard_input(self, record: Mapping[str, Any]) -> None:
        try:
            self.input_path(record).unlink(missing_ok=True)
        except (JobStoreError, OSError):
            LOGGER.warning("Could not discard forecast upload for job %s", record.get("jobId"))

    def active_count(self) -> int:
        self.initialize()
        active = 0
        for directory in self.jobs_dir.iterdir():
            if not directory.is_dir() or not re.fullmatch(r"[0-9a-f]{32}", directory.name):
                continue
            try:
                record = self.read_job(directory.name)
            except JobStoreError:
                continue
            if record is not None and record.get("state") in ACTIVE_STATES:
                active += 1
        return active

    def recover_interrupted(self) -> int:
        self.initialize()
        recovered = 0
        for staging in self.jobs_dir.glob(".creating-*"):
            shutil.rmtree(staging, ignore_errors=True)
        for directory in self.jobs_dir.iterdir():
            if not directory.is_dir() or not re.fullmatch(r"[0-9a-f]{32}", directory.name):
                continue
            try:
                record = self.read_job(directory.name)
            except JobStoreError:
                for abandoned_input in directory.glob("input.*"):
                    try:
                        abandoned_input.unlink()
                    except OSError:
                        pass
                continue
            if record is None:
                shutil.rmtree(directory, ignore_errors=True)
                continue
            if record.get("state") not in ACTIVE_STATES:
                continue
            now = _utc_now()
            try:
                published = self.read_result(directory.name)
            except JobStoreError:
                published = None
            if published is not None:
                completed_at = published.get("createdAt")
                if not isinstance(completed_at, str):
                    completed_at = now
                record.update(
                    {
                        "state": "complete",
                        "stage": "complete",
                        "updatedAt": now,
                        "heartbeatAt": now,
                        "completedAt": completed_at,
                        "resultId": directory.name,
                        "error": None,
                    }
                )
                self.write_job(record)
                self.discard_input(record)
                recovered += 1
                continue
            record.update(
                {
                    "state": "failed",
                    "stage": "interrupted",
                    "updatedAt": now,
                    "heartbeatAt": now,
                    "completedAt": now,
                    "error": {
                        "code": "worker_interrupted",
                        "message": "The forecast worker stopped before publishing a complete result.",
                    },
                }
            )
            self.write_job(record)
            self.discard_input(record)
            recovered += 1
        for temporary in self.incoming_dir.glob("upload-*"):
            try:
                temporary.unlink()
            except OSError:
                pass
        return recovered

    def publish_result(self, job_id: str, payload: Mapping[str, Any]) -> str:
        result_id = job_id
        destination = self.results_dir / result_id
        if destination.exists():
            raise JobStoreError("forecast result already exists")
        staging = self.results_dir / f".publishing-{result_id}-{uuid4().hex}"
        try:
            staging.mkdir(parents=False, exist_ok=False)
            self._write_json_atomic(staging / "result.json", payload)
            os.replace(staging, destination)
            self._fsync_directory(self.results_dir)
            return result_id
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def list_result_ids(self) -> list[str]:
        """Read-only listing of published results; used for comparative context."""

        if not self.results_dir.is_dir():
            return []
        return sorted(
            entry.name
            for entry in self.results_dir.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        )

    def read_result(self, result_id: str) -> dict[str, Any] | None:
        payload = self._read_json(self.results_dir / result_id / "result.json")
        if payload is None:
            return None
        if payload.get("resultId") != result_id or payload.get("jobId") != result_id:
            raise JobStoreError("stored forecast result has an invalid identity")
        return payload


def _strict_json_copy(
    value: Any, label: str, *, max_bytes: int | None = None
) -> Any:
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
        if max_bytes is not None and len(encoded.encode("utf-8")) > max_bytes:
            raise ForecastJobExecutionError(
                "orchestrator_result_too_large",
                f"{label} exceeds the configured {max_bytes}-byte result limit.",
            )
        return json.loads(encoded)
    except ForecastJobExecutionError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ForecastJobExecutionError(
            "invalid_orchestrator_output", f"{label} is not strict JSON."
        ) from exc


def _validate_orchestrator_output(
    output: Any, *, max_result_bytes: int
) -> dict[str, Any]:
    copied = _strict_json_copy(
        output, "orchestrator result", max_bytes=max_result_bytes
    )
    if not isinstance(copied, dict):
        raise ForecastJobExecutionError(
            "invalid_orchestrator_output",
            "The forecast orchestrator must return a serializable result object.",
        )
    required = {"evidence", "behavioralHeads", "boundaries"}
    if set(copied) != required:
        raise ForecastJobExecutionError(
            "invalid_orchestrator_output",
            "The orchestrator result must contain evidence, behavioralHeads, and boundaries exactly.",
        )
    evidence = copied["evidence"]
    heads = copied["behavioralHeads"]
    boundaries = copied["boundaries"]
    if not isinstance(evidence, dict):
        raise ForecastJobExecutionError(
            "invalid_orchestrator_output", "Orchestrator evidence must be an object."
        )
    if not isinstance(heads, dict) or set(heads) != set(BEHAVIORAL_HEAD_KEYS):
        raise ForecastJobExecutionError(
            "invalid_orchestrator_output",
            "The orchestrator must return every registered behavioral head exactly once.",
        )
    for key, head in heads.items():
        if not isinstance(head, dict):
            raise ForecastJobExecutionError(
                "invalid_orchestrator_output", f"Behavioral head {key} must be an object."
            )
        status = head.get("status")
        value = head.get("value")
        validation = head.get("validation")
        if status == "unavailable":
            if value is not None or validation != "not-calibrated" or not isinstance(head.get("reason"), str) or not head["reason"].strip():
                raise ForecastJobExecutionError(
                    "invalid_orchestrator_output",
                    f"Unavailable behavioral head {key} has an invalid contract.",
                )
        elif status == "available":
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                or validation != "production-calibrated"
            ):
                raise ForecastJobExecutionError(
                    "invalid_orchestrator_output",
                    f"Available behavioral head {key} lacks a calibrated probability.",
                )
        else:
            raise ForecastJobExecutionError(
                "invalid_orchestrator_output",
                f"Behavioral head {key} has an unsupported status.",
            )
    if (
        not isinstance(boundaries, list)
        or not boundaries
        or any(not isinstance(item, str) or not item.strip() for item in boundaries)
    ):
        raise ForecastJobExecutionError(
            "invalid_orchestrator_output",
            "Result boundaries must contain non-empty statements.",
        )
    return {
        "evidence": evidence,
        "behavioralHeads": heads,
        "boundaries": boundaries,
    }


def parse_context_json(raw_context: str | None, max_bytes: int) -> dict[str, Any] | None:
    if raw_context is None or raw_context == "":
        return None
    if len(raw_context.encode("utf-8")) > max_bytes:
        raise ForecastJobAPIError(
            413,
            "context_too_large",
            f"context exceeds the {max_bytes}-byte limit",
        )

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate context key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            raw_context,
            parse_constant=reject_constant,
            object_pairs_hook=object_pairs,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ForecastJobAPIError(
            400, "invalid_context", "context must be strict JSON without duplicate keys"
        ) from exc
    if not isinstance(parsed, dict):
        raise ForecastJobAPIError(
            400, "invalid_context", "context must be a JSON object"
        )
    return parsed


def public_job_record(record: Mapping[str, Any]) -> dict[str, Any]:
    job_id = record.get("jobId")
    state = record.get("state")
    result_id = record.get("resultId")
    payload: dict[str, Any] = {
        "schemaVersion": JOB_SCHEMA_VERSION,
        "jobId": job_id,
        "state": state,
        "stage": record.get("stage"),
        "createdAt": record.get("createdAt"),
        "updatedAt": record.get("updatedAt"),
        "heartbeatAt": record.get("heartbeatAt"),
        "startedAt": record.get("startedAt"),
        "completedAt": record.get("completedAt"),
        "elapsedSeconds": _elapsed_seconds(record),
        "input": record.get("input"),
        "contextAccepted": record.get("context") is not None,
        "resultId": result_id,
        "error": record.get("error"),
        "links": {
            "self": f"/api/forecast/v1/jobs/{job_id}",
            "result": (
                f"/api/forecast/v1/results/{result_id}"
                if state == "complete" and isinstance(result_id, str)
                else None
            ),
        },
    }
    return payload


class _StageCallbackBridge:
    """One stage callback usable by async runners and synchronous worker threads."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        callback: Callable[[str], Awaitable[None]],
    ) -> None:
        self._loop = loop
        self._callback = callback

    def __call__(self, stage: str) -> Any:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self._loop:
            return asyncio.create_task(self._callback(stage))
        future = asyncio.run_coroutine_threadsafe(self._callback(stage), self._loop)
        future.result(timeout=30.0)
        return None


class ForecastJobService:
    """Admission, execution and lifecycle management for forecast jobs."""

    def __init__(
        self,
        *,
        store: ForecastJobStore,
        orchestrator: ForecastOrchestrator | None = None,
        media_probe: FFprobeMediaProbe | Any | None = None,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
        max_context_bytes: int = DEFAULT_MAX_CONTEXT_BYTES,
        max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES,
        max_active_jobs: int = DEFAULT_MAX_ACTIVE_JOBS,
        max_workers: int = DEFAULT_MAX_WORKERS,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
        job_timeout_seconds: float = DEFAULT_JOB_TIMEOUT_SECONDS,
    ) -> None:
        for label, value in (
            ("max_upload_bytes", max_upload_bytes),
            ("max_context_bytes", max_context_bytes),
            ("max_result_bytes", max_result_bytes),
            ("max_active_jobs", max_active_jobs),
            ("max_workers", max_workers),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if max_workers > max_active_jobs:
            raise ValueError("max_workers cannot exceed max_active_jobs")
        if not math.isfinite(heartbeat_seconds) or heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive and finite")
        if not math.isfinite(job_timeout_seconds) or job_timeout_seconds <= 0:
            raise ValueError("job_timeout_seconds must be positive and finite")
        self.store = store
        self.orchestrator = orchestrator or DefaultEvidenceOrchestrator()
        self.media_probe = media_probe or FFprobeMediaProbe()
        self.max_upload_bytes = max_upload_bytes
        self.max_context_bytes = max_context_bytes
        self.max_result_bytes = max_result_bytes
        self.max_active_jobs = max_active_jobs
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.job_timeout_seconds = float(job_timeout_seconds)
        self._semaphore = asyncio.Semaphore(max_workers)
        self._admission_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._startup_lock = asyncio.Lock()
        self._started = False
        self._stopping = False
        self._admitted_uploads = 0
        self._tasks: set[asyncio.Task[None]] = set()
        self._lingering_sync_runners: set[asyncio.Future[Any]] = set()
        # A durable job uses the normalized idempotency UUID as its identity.
        # The in-process lock closes the only race before that job directory is
        # atomically published. After a restart, the durable job record itself
        # remains the idempotency lookup.
        self._idempotency_locks: dict[str, asyncio.Lock] = {}
        self._deferred_input_cleanup: set[str] = set()

    @classmethod
    def from_env(
        cls,
        *,
        orchestrator: ForecastOrchestrator | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "ForecastJobService":
        source = os.environ if environ is None else environ
        default_root = Path(__file__).resolve().parents[1] / ".runtime" / "forecast"
        raw_root = source.get("FORECAST_JOB_DIR", str(default_root)).strip()
        if not raw_root:
            raise ValueError("FORECAST_JOB_DIR must be non-empty")
        root = Path(raw_root).expanduser().resolve()
        probe_timeout = _positive_float_from_env(
            source, "FORECAST_PROBE_TIMEOUT_SECONDS", DEFAULT_PROBE_TIMEOUT_SECONDS
        )
        return cls(
            store=ForecastJobStore(root),
            orchestrator=orchestrator,
            media_probe=FFprobeMediaProbe(
                source.get("FORECAST_FFPROBE_BINARY", "ffprobe"), probe_timeout
            ),
            max_upload_bytes=_positive_int_from_env(
                source, "FORECAST_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES
            ),
            max_context_bytes=_positive_int_from_env(
                source, "FORECAST_MAX_CONTEXT_BYTES", DEFAULT_MAX_CONTEXT_BYTES
            ),
            max_result_bytes=_positive_int_from_env(
                source, "FORECAST_MAX_RESULT_BYTES", DEFAULT_MAX_RESULT_BYTES
            ),
            max_active_jobs=_positive_int_from_env(
                source, "FORECAST_MAX_ACTIVE_JOBS", DEFAULT_MAX_ACTIVE_JOBS
            ),
            max_workers=_positive_int_from_env(
                source, "FORECAST_MAX_WORKERS", DEFAULT_MAX_WORKERS
            ),
            heartbeat_seconds=_positive_float_from_env(
                source, "FORECAST_HEARTBEAT_SECONDS", DEFAULT_HEARTBEAT_SECONDS
            ),
            job_timeout_seconds=_positive_float_from_env(
                source, "FORECAST_JOB_TIMEOUT_SECONDS", DEFAULT_JOB_TIMEOUT_SECONDS
            ),
        )

    async def startup(self) -> None:
        async with self._startup_lock:
            if self._started:
                return
            recovered = await asyncio.to_thread(self.store.recover_interrupted)
            if recovered:
                LOGGER.warning("Recovered %d interrupted forecast job record(s)", recovered)
            self._started = True
            self._stopping = False

    async def shutdown(self) -> None:
        self._stopping = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        async with self._startup_lock:
            self._started = False

    async def _stream_upload(
        self, upload: UploadFile, suffix: str
    ) -> tuple[Path, UploadedVideoEvidence]:
        destination = self.store.incoming_path(suffix)
        size = 0
        digest = hashlib.sha256()
        try:
            with destination.open("xb") as handle:
                while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
                    size += len(chunk)
                    if size > self.max_upload_bytes:
                        raise ForecastJobAPIError(
                            413,
                            "video_too_large",
                            f"video exceeds the {self.max_upload_bytes}-byte upload limit",
                        )
                    handle.write(chunk)
                    digest.update(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if size == 0:
                raise ForecastJobAPIError(400, "invalid_video", "uploaded video is empty")
            return destination, UploadedVideoEvidence(
                size_bytes=size,
                sha256=digest.hexdigest(),
                suffix=suffix,
                content_type=(upload.content_type or None),
            )
        except Exception:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    async def submit(
        self,
        upload: UploadFile,
        raw_context: str | None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        await self.startup()
        try:
            normalized_key = normalize_idempotency_key(idempotency_key)
        except ForecastJobAPIError:
            await upload.close()
            raise
        job_id = normalized_key or uuid4().hex
        if normalized_key is not None:
            lock = self._idempotency_locks.setdefault(normalized_key, asyncio.Lock())
            async with lock:
                return await self._submit_once(
                    upload,
                    raw_context,
                    job_id=job_id,
                    allow_existing=True,
                )
        return await self._submit_once(
            upload,
            raw_context,
            job_id=job_id,
            allow_existing=False,
        )

    async def _submit_once(
        self,
        upload: UploadFile,
        raw_context: str | None,
        *,
        job_id: str,
        allow_existing: bool,
    ) -> dict[str, Any]:
        if allow_existing:
            try:
                existing = await asyncio.to_thread(self.store.read_job, job_id)
            except JobStoreError as exc:
                await upload.close()
                raise ForecastJobAPIError(
                    500,
                    "job_storage_failed",
                    "Stored forecast job state is unreadable.",
                ) from exc
            if existing is not None:
                await upload.close()
                return public_job_record(existing)
        try:
            context = parse_context_json(raw_context, self.max_context_bytes)
        except ForecastJobAPIError:
            await upload.close()
            raise
        filename = Path(upload.filename or "")
        suffix = filename.suffix.lower()
        content_type = (upload.content_type or "").lower()
        if suffix not in VIDEO_SUFFIXES:
            await upload.close()
            raise ForecastJobAPIError(
                400,
                "invalid_video",
                f"video filename must end in one of {sorted(VIDEO_SUFFIXES)}",
            )
        if content_type and not (
            content_type.startswith("video/") or content_type == "application/octet-stream"
        ):
            await upload.close()
            raise ForecastJobAPIError(
                400, "invalid_video", "multipart field 'video' must contain a video"
            )
        async with self._admission_lock:
            active = await asyncio.to_thread(self.store.active_count)
            occupied = active + self._admitted_uploads + len(self._lingering_sync_runners)
            if self._stopping or occupied >= self.max_active_jobs:
                await upload.close()
                raise ForecastJobAPIError(
                    429,
                    "forecast_capacity_reached",
                    "The bounded forecast queue is full; retry after an active job finishes.",
                )
            self._admitted_uploads += 1
        incoming: Path | None = None
        try:
            incoming, uploaded = await self._stream_upload(upload, suffix)
            record = await asyncio.to_thread(
                self.store.create_job,
                job_id=job_id,
                incoming_path=incoming,
                upload=uploaded,
                context=context,
            )
            incoming = None
            task = asyncio.create_task(self._run_job(job_id), name=f"forecast-job-{job_id}")
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return public_job_record(record)
        except ForecastJobAPIError:
            raise
        except (JobStoreError, OSError) as exc:
            if allow_existing:
                try:
                    existing = await asyncio.to_thread(self.store.read_job, job_id)
                except JobStoreError:
                    existing = None
                if existing is not None:
                    return public_job_record(existing)
            raise ForecastJobAPIError(
                500, "job_storage_failed", "The forecast upload could not be stored."
            ) from exc
        finally:
            await upload.close()
            if incoming is not None:
                try:
                    incoming.unlink(missing_ok=True)
                except OSError:
                    pass
            async with self._admission_lock:
                self._admitted_uploads -= 1

    async def _mutate_job(self, job_id: str, **updates: Any) -> dict[str, Any]:
        async with self._state_lock:
            record = await asyncio.to_thread(self.store.read_job, job_id)
            if record is None:
                raise JobStoreError("forecast job disappeared")
            current_state = record.get("state")
            next_state = updates.get("state", current_state)
            if (
                next_state != current_state
                and next_state not in STATE_TRANSITIONS.get(current_state, frozenset())
            ):
                raise JobStoreError(
                    f"invalid forecast job transition {current_state!r} -> {next_state!r}"
                )
            now = _utc_now()
            record.update(updates)
            record["updatedAt"] = now
            record["heartbeatAt"] = now
            await asyncio.to_thread(self.store.write_job, record)
            return record

    async def _heartbeat(self, job_id: str, stage: str | None = None) -> None:
        if stage is not None and not STAGE_RE.fullmatch(stage):
            raise ForecastJobExecutionError(
                "invalid_stage", "The orchestrator supplied an invalid stage identifier."
            )
        async with self._state_lock:
            record = await asyncio.to_thread(self.store.read_job, job_id)
            if record is None:
                raise JobStoreError("forecast job disappeared")
            if record.get("state") not in ACTIVE_STATES:
                return
            now = _utc_now()
            if stage is not None:
                record["stage"] = stage
            record["updatedAt"] = now
            record["heartbeatAt"] = now
            await asyncio.to_thread(self.store.write_job, record)

    async def _heartbeat_loop(self, job_id: str) -> None:
        try:
            while True:
                await asyncio.sleep(self.heartbeat_seconds)
                await self._heartbeat(job_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("Forecast heartbeat failed for job %s", job_id)

    async def _invoke_orchestrator(
        self,
        *,
        job_id: str,
        video_path: Path,
        input_sha256: str,
        media_probe: Mapping[str, Any],
        context: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        runner = getattr(self.orchestrator, "run", None)
        if runner is None:
            runner = self.orchestrator
        if not callable(runner):
            raise ForecastJobExecutionError(
                "invalid_orchestrator",
                "The configured forecast orchestrator is not callable.",
            )
        stage_callback = _StageCallbackBridge(
            asyncio.get_running_loop(), lambda stage: self._heartbeat(job_id, stage)
        )
        arguments = (
            video_path,
            input_sha256,
            media_probe,
            context,
            stage_callback,
        )
        asynchronous = inspect.iscoroutinefunction(runner) or inspect.iscoroutinefunction(
            getattr(runner, "__call__", None)
        )
        try:
            if asynchronous:
                output = await asyncio.wait_for(
                    runner(*arguments), timeout=self.job_timeout_seconds
                )
            else:
                loop = asyncio.get_running_loop()
                future = loop.run_in_executor(None, lambda: runner(*arguments))
                try:
                    output = await asyncio.wait_for(
                        asyncio.shield(future), timeout=self.job_timeout_seconds
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    self._defer_lingering_runner(job_id, future)
                    raise
                if inspect.isawaitable(output):
                    output = await asyncio.wait_for(
                        output, timeout=self.job_timeout_seconds
                    )
        except asyncio.TimeoutError as exc:
            raise ForecastJobExecutionError(
                "orchestrator_timeout",
                "The forecast evidence orchestrator exceeded its configured time limit.",
            ) from exc
        if not isinstance(output, Mapping):
            raise ForecastJobExecutionError(
                "invalid_orchestrator_output",
                "The forecast orchestrator must return a serializable result object.",
            )
        return output

    def _defer_lingering_runner(
        self, job_id: str, future: asyncio.Future[Any]
    ) -> None:
        """Keep the upload alive until a non-cancellable sync runner exits.

        Python cannot safely terminate a thread that is inside a native model
        forward. The job is marked failed/interrupted and its capacity remains
        occupied, but deleting the upload at that point would race the still
        running worker. Cleanup therefore moves to the future's completion
        callback. The late runner can never publish or mutate terminal state.
        """

        self._lingering_sync_runners.add(future)
        self._deferred_input_cleanup.add(job_id)

        def finish(done: asyncio.Future[Any]) -> None:
            self._lingering_sync_runners.discard(done)
            self._deferred_input_cleanup.discard(job_id)
            try:
                record = self.store.read_job(job_id)
                if record is not None:
                    self.store.discard_input(record)
            except Exception:
                LOGGER.exception(
                    "Could not clean up upload after lingering forecast runner %s",
                    job_id,
                )

        future.add_done_callback(finish)

    async def _fail_job(self, job_id: str, code: str, message: str) -> None:
        try:
            await self._mutate_job(
                job_id,
                state="failed",
                stage="failed" if code != "worker_interrupted" else "interrupted",
                completedAt=_utc_now(),
                error={"code": code, "message": message},
            )
        except Exception:
            LOGGER.exception("Could not persist failed forecast state for %s", job_id)

    async def _run_job(self, job_id: str) -> None:
        heartbeat_task: asyncio.Task[None] | None = asyncio.create_task(
            self._heartbeat_loop(job_id), name=f"forecast-heartbeat-{job_id}"
        )
        record: dict[str, Any] | None = None
        try:
            async with self._semaphore:
                record = await self._mutate_job(
                    job_id,
                    state="probing",
                    stage="probing-media",
                    startedAt=_utc_now(),
                )
                video_path = self.store.input_path(record)
                raw_duration = await asyncio.to_thread(self.media_probe.probe, video_path)
                try:
                    duration = float(raw_duration)
                except (TypeError, ValueError) as exc:
                    raise ForecastJobExecutionError(
                        "invalid_media_metadata",
                        "The authoritative media probe returned a non-numeric duration.",
                    ) from exc
                if not math.isfinite(duration):
                    raise ForecastJobExecutionError(
                        "invalid_media_metadata",
                        "The authoritative media probe returned a non-finite duration.",
                    )
                if not MINIMUM_DURATION_SECONDS <= duration <= MAXIMUM_DURATION_SECONDS:
                    raise ForecastJobExecutionError(
                        "duration_out_of_range",
                        f"Authoritative ffprobe duration must be {MINIMUM_DURATION_SECONDS:g}–{MAXIMUM_DURATION_SECONDS:g} seconds inclusive.",
                    )
                record["input"]["durationSeconds"] = duration
                record = await self._mutate_job(
                    job_id,
                    state="running",
                    stage="orchestrating-evidence",
                    input=record["input"],
                )
                media_probe = {
                    "source": "server-ffprobe",
                    "durationSeconds": duration,
                    "sizeBytes": record["input"]["sizeBytes"],
                    "contentType": record["input"].get("contentType"),
                }
                raw_output = await self._invoke_orchestrator(
                    job_id=job_id,
                    video_path=video_path,
                    input_sha256=record["input"]["sha256"],
                    media_probe=media_probe,
                    context=record.get("context"),
                )
                output = _validate_orchestrator_output(
                    raw_output, max_result_bytes=self.max_result_bytes
                )
                completed_at = _utc_now()
                result_payload = {
                    "schemaVersion": RESULT_SCHEMA_VERSION,
                    "resultId": job_id,
                    "jobId": job_id,
                    "createdAt": completed_at,
                    "input": {
                        "sha256": record["input"]["sha256"],
                        "sizeBytes": record["input"]["sizeBytes"],
                        "durationSeconds": duration,
                        "contentType": record["input"].get("contentType"),
                    },
                    "evidence": output["evidence"],
                    "behavioralHeads": output["behavioralHeads"],
                    "boundaries": output["boundaries"],
                }
                result_id = await asyncio.to_thread(
                    self.store.publish_result, job_id, result_payload
                )
                await self._mutate_job(
                    job_id,
                    state="complete",
                    stage="complete",
                    completedAt=completed_at,
                    resultId=result_id,
                    error=None,
                )
        except asyncio.CancelledError:
            await self._fail_job(
                job_id,
                "worker_interrupted",
                "The forecast worker stopped before publishing a complete result.",
            )
            raise
        except ForecastJobExecutionError as exc:
            await self._fail_job(job_id, exc.code, exc.message)
        except JobStoreError:
            LOGGER.exception("Forecast job storage failed for %s", job_id)
            await self._fail_job(
                job_id,
                "job_storage_failed",
                "The forecast job could not atomically persist its state or result.",
            )
        except Exception:
            LOGGER.exception("Forecast orchestrator failed for %s", job_id)
            await self._fail_job(
                job_id,
                "orchestrator_failed",
                "The forecast evidence orchestrator failed; no result was published.",
            )
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                await asyncio.gather(heartbeat_task, return_exceptions=True)
            if record is None:
                try:
                    record = await asyncio.to_thread(self.store.read_job, job_id)
                except JobStoreError:
                    record = None
            if record is not None and job_id not in self._deferred_input_cleanup:
                await asyncio.to_thread(self.store.discard_input, record)

    async def get_job(self, job_id: UUID | str) -> dict[str, Any]:
        await self.startup()
        normalized = job_id.hex if isinstance(job_id, UUID) else str(job_id)
        try:
            record = await asyncio.to_thread(self.store.read_job, normalized)
        except JobStoreError as exc:
            raise ForecastJobAPIError(
                500, "job_storage_failed", "Stored forecast job state is unreadable."
            ) from exc
        if record is None:
            raise ForecastJobAPIError(404, "job_not_found", "Forecast job was not found.")
        return public_job_record(record)

    async def get_result(self, result_id: UUID | str) -> dict[str, Any]:
        await self.startup()
        normalized = result_id.hex if isinstance(result_id, UUID) else str(result_id)
        try:
            result = await asyncio.to_thread(self.store.read_result, normalized)
            if result is not None:
                return result
            job = await asyncio.to_thread(self.store.read_job, normalized)
        except JobStoreError as exc:
            raise ForecastJobAPIError(
                500, "result_storage_failed", "Stored forecast result is unreadable."
            ) from exc
        if job is None:
            raise ForecastJobAPIError(
                404, "result_not_found", "Forecast result was not found."
            )
        if job.get("state") == "failed":
            raise ForecastJobAPIError(
                409, "job_failed", "The forecast job failed and produced no result."
            )
        raise ForecastJobAPIError(
            409, "result_not_ready", "The forecast result has not been published yet."
        )
