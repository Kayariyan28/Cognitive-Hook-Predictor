"""Immutable endpoint registry for executable forecast model workers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import re
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from ..errors import ConfigurationError
from .registry import ADAPTER_IDS


WORKER_REGISTRY_SCHEMA_VERSION = "creator-forecast-worker-registry/1"
WORKER_REGISTRY_JSON_ENV = "FORECAST_WORKER_REGISTRY_JSON"
MODEL_BRANCH_KEYS = ("vjepa21", "videoLlama21Av", "audioModel")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,255}$")
MAX_REGISTRY_BYTES = 256 * 1024


def _fail(message: str) -> ConfigurationError:
    return ConfigurationError(f"Forecast worker registry: {message}")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _fail(f"{label} must be an object")
    return value


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    actual = set(value)
    if actual != keys:
        raise _fail(
            f"{label} has invalid fields (missing {sorted(keys - actual)}; "
            f"unknown {sorted(actual - keys)})"
        )


def _text(value: Any, label: str, *, identifier: bool = False) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise _fail(f"{label} must be a non-empty trimmed string")
    if any(ord(character) < 32 for character in value):
        raise _fail(f"{label} contains control characters")
    if identifier and not IDENTIFIER_RE.fullmatch(value):
        raise _fail(f"{label} is not a stable identifier")
    return value


def _commit(value: Any, label: str) -> str:
    revision = _text(value, label).lower()
    if not COMMIT_RE.fullmatch(revision):
        raise _fail(f"{label} must be a full immutable commit SHA")
    return revision


def _sha256(value: Any, label: str) -> str:
    digest = _text(value, label).lower()
    if not SHA256_RE.fullmatch(digest):
        raise _fail(f"{label} must be a SHA-256 digest")
    return digest


def _endpoint(value: Any, label: str) -> str:
    raw = _text(value, label)
    parsed = urlsplit(raw)
    host = (parsed.hostname or "").lower()
    is_loopback = host in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme not in ({"https"} if not is_loopback else {"http", "https"}):
        raise _fail(f"{label} must use HTTPS (HTTP is allowed only on loopback)")
    if not parsed.netloc or parsed.username or parsed.password:
        raise _fail(f"{label} must be an absolute URL without embedded credentials")
    if parsed.query or parsed.fragment:
        raise _fail(f"{label} cannot contain a query or fragment")
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _json_without_duplicates(payload: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise _fail(f"configuration contains duplicate key {key!r}")
            value[key] = item
        return value

    try:
        parsed = json.loads(payload, object_pairs_hook=pairs)
    except ConfigurationError:
        raise
    except json.JSONDecodeError as exc:
        raise _fail("configuration is not valid JSON") from exc
    return _object(parsed, "root")


@dataclass(frozen=True, slots=True)
class WorkerPin:
    branch: str
    base_url: str
    auth_token_env: str | None
    timeout_seconds: float
    id: str
    revision: str
    weights_sha256: str
    license: str
    adapter_id: str
    code_revision: str
    preprocessing_id: str
    preprocessing_sha256: str

    @classmethod
    def from_mapping(cls, branch: str, value: Any) -> "WorkerPin":
        if branch not in MODEL_BRANCH_KEYS:
            raise _fail(f"unknown worker branch {branch!r}")
        record = _object(value, f"workers.{branch}")
        _exact(
            record,
            {
                "baseUrl",
                "authTokenEnv",
                "timeoutSeconds",
                "model",
                "codeRevision",
                "preprocessing",
            },
            f"workers.{branch}",
        )
        model = _object(record["model"], f"workers.{branch}.model")
        _exact(
            model,
            {"id", "revision", "weightsSha256", "license", "adapterId"},
            f"workers.{branch}.model",
        )
        expected_adapter = ADAPTER_IDS[branch]
        if model["adapterId"] != expected_adapter:
            raise _fail(f"workers.{branch}.model.adapterId must be {expected_adapter}")
        preprocessing = _object(
            record["preprocessing"], f"workers.{branch}.preprocessing"
        )
        _exact(
            preprocessing,
            {"id", "sha256"},
            f"workers.{branch}.preprocessing",
        )
        raw_token_env = record["authTokenEnv"]
        if raw_token_env is not None and (
            not isinstance(raw_token_env, str)
            or not ENV_NAME_RE.fullmatch(raw_token_env)
        ):
            raise _fail(f"workers.{branch}.authTokenEnv must be null or an env name")
        timeout = record["timeoutSeconds"]
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(float(timeout))
            or not 5 <= float(timeout) <= 7200
        ):
            raise _fail(f"workers.{branch}.timeoutSeconds must be from 5 to 7200")
        return cls(
            branch=branch,
            base_url=_endpoint(record["baseUrl"], f"workers.{branch}.baseUrl"),
            auth_token_env=raw_token_env,
            timeout_seconds=float(timeout),
            id=_text(model["id"], f"workers.{branch}.model.id", identifier=True),
            revision=_commit(
                model["revision"], f"workers.{branch}.model.revision"
            ),
            weights_sha256=_sha256(
                model["weightsSha256"], f"workers.{branch}.model.weightsSha256"
            ),
            license=_text(model["license"], f"workers.{branch}.model.license"),
            adapter_id=expected_adapter,
            code_revision=_commit(
                record["codeRevision"], f"workers.{branch}.codeRevision"
            ),
            preprocessing_id=_text(
                preprocessing["id"],
                f"workers.{branch}.preprocessing.id",
                identifier=True,
            ),
            preprocessing_sha256=_sha256(
                preprocessing["sha256"],
                f"workers.{branch}.preprocessing.sha256",
            ),
        )

    def public_provenance(self) -> dict[str, str]:
        # artifactSha256 is retained as a compatibility alias for ArtifactPin.
        return {
            "id": self.id,
            "revision": self.revision,
            "artifactSha256": self.weights_sha256,
            "weightsSha256": self.weights_sha256,
            "license": self.license,
            "adapterId": self.adapter_id,
        }

    def public_runtime(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "baseUrl": self.base_url,
            "timeoutSeconds": self.timeout_seconds,
            "authConfigured": self.auth_token_env is not None,
            "model": {
                "id": self.id,
                "revision": self.revision,
                "weightsSha256": self.weights_sha256,
                "license": self.license,
                "adapterId": self.adapter_id,
            },
            "codeRevision": self.code_revision,
            "preprocessing": {
                "id": self.preprocessing_id,
                "sha256": self.preprocessing_sha256,
            },
        }

    def auth_token(self, environ: Mapping[str, str]) -> str | None:
        if self.auth_token_env is None:
            return None
        token = str(environ.get(self.auth_token_env, "")).strip()
        if not token:
            raise _fail(
                f"workers.{self.branch} requires non-empty {self.auth_token_env}"
            )
        return token


@dataclass(frozen=True, slots=True)
class WorkerRegistry:
    workers: Mapping[str, WorkerPin]
    source: str

    @classmethod
    def empty(cls) -> "WorkerRegistry":
        return cls(MappingProxyType({}), "none")

    @classmethod
    def from_mapping(cls, value: Any, *, source: str = "memory") -> "WorkerRegistry":
        root = _object(value, "root")
        _exact(root, {"schemaVersion", "workers"}, "root")
        if root["schemaVersion"] != WORKER_REGISTRY_SCHEMA_VERSION:
            raise _fail(f"schemaVersion must be {WORKER_REGISTRY_SCHEMA_VERSION}")
        raw_workers = _object(root["workers"], "workers")
        unknown = set(raw_workers) - set(MODEL_BRANCH_KEYS)
        if unknown:
            raise _fail(f"unknown worker branches {sorted(unknown)}")
        workers = {
            key: WorkerPin.from_mapping(key, raw_workers[key])
            for key in MODEL_BRANCH_KEYS
            if key in raw_workers
        }
        return cls(MappingProxyType(workers), source)

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "WorkerRegistry":
        environment = os.environ if environ is None else environ
        payload = str(environment.get(WORKER_REGISTRY_JSON_ENV, "")).strip()
        if not payload:
            return cls.empty()
        if len(payload.encode("utf-8")) > MAX_REGISTRY_BYTES:
            raise _fail(f"{WORKER_REGISTRY_JSON_ENV} is too large")
        return cls.from_mapping(
            _json_without_duplicates(payload), source="environment-json"
        )
