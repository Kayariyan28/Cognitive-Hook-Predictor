"""Network adapters for separately deployed open-source model workers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import httpx

from .execution_registry import WorkerPin, WorkerRegistry
from .worker_protocol import (
    BranchOutput,
    WorkerIdentity,
    WorkerProtocolError,
    build_worker_request,
    validate_branch_output,
)


MAX_JSON_RESPONSE_BYTES = 4 * 1024 * 1024


class WorkerExecutionError(RuntimeError):
    """An installed worker could not be reached or execute a request safely."""


def _response_json(response: httpx.Response, label: str) -> dict[str, Any]:
    if len(response.content) > MAX_JSON_RESPONSE_BYTES:
        raise WorkerExecutionError(f"{label} response exceeds the JSON size limit")
    if response.status_code < 200 or response.status_code >= 300:
        raise WorkerExecutionError(f"{label} returned HTTP {response.status_code}")
    content_type = response.headers.get("content-type", "").lower()
    if "application/json" not in content_type:
        raise WorkerExecutionError(f"{label} did not return JSON")
    try:
        value = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise WorkerExecutionError(f"{label} returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise WorkerExecutionError(f"{label} returned a non-object JSON document")
    return value


@dataclass(slots=True)
class HttpBranchAdapter:
    pin: WorkerPin
    token: str | None
    transport: httpx.BaseTransport | None = None

    @property
    def branch(self) -> str:
        return self.pin.branch

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "X-Forecast-Protocol": "creator-forecast-worker/1",
        }
        if self.token is not None:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _client(self, *, timeout: float) -> httpx.Client:
        return httpx.Client(
            timeout=httpx.Timeout(timeout),
            follow_redirects=False,
            transport=self.transport,
            headers=self._headers(),
        )

    def probe(self) -> WorkerIdentity:
        try:
            with self._client(timeout=min(self.pin.timeout_seconds, 10.0)) as client:
                response = client.get(f"{self.pin.base_url}/v1/status")
            return WorkerIdentity.from_status(
                _response_json(response, f"{self.branch} status"),
                expected_branch=self.branch,
                expected_pin=self.pin,
            )
        except WorkerProtocolError:
            raise
        except (httpx.HTTPError, OSError) as exc:
            raise WorkerExecutionError(
                f"{self.branch} worker readiness request failed"
            ) from exc

    def run(
        self,
        *,
        video_path: Path,
        input_sha256: str,
        duration_seconds: float,
        context: Mapping[str, Any],
    ) -> BranchOutput:
        if not video_path.is_file():
            raise WorkerExecutionError("forecast worker input video is unavailable")
        identity = self.probe()
        request = build_worker_request(
            branch=self.branch,
            input_sha256=input_sha256,
            duration_seconds=duration_seconds,
            context=context,
        )
        try:
            with video_path.open("rb") as handle:
                files = {
                    "video": (
                        video_path.name,
                        handle,
                        "application/octet-stream",
                    )
                }
                data = {
                    "request": json.dumps(
                        request, sort_keys=True, separators=(",", ":")
                    )
                }
                with self._client(timeout=self.pin.timeout_seconds) as client:
                    response = client.post(
                        f"{self.pin.base_url}/v1/extract", files=files, data=data
                    )
            output = _response_json(response, f"{self.branch} extraction")
            return validate_branch_output(
                output,
                expected_branch=self.branch,
                expected_input_sha256=input_sha256,
                identity=identity,
            )
        except WorkerProtocolError:
            raise
        except (httpx.HTTPError, OSError) as exc:
            raise WorkerExecutionError(f"{self.branch} worker execution failed") from exc


def build_http_adapters(
    registry: WorkerRegistry,
    *,
    environ: Mapping[str, str],
    transports: Mapping[str, httpx.BaseTransport] | None = None,
) -> Mapping[str, HttpBranchAdapter]:
    transport_map = transports or {}
    return MappingProxyType(
        {
            branch: HttpBranchAdapter(
                pin=pin,
                token=pin.auth_token(environ),
                transport=transport_map.get(branch),
            )
            for branch, pin in registry.workers.items()
        }
    )
