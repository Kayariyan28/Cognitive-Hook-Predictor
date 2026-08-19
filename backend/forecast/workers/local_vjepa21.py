"""In-process adapter for the strict local V-JEPA 2.1 worker."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from ...errors import ConfigurationError
from ..worker_protocol import (
    BranchOutput,
    WorkerIdentity,
    build_worker_request,
    validate_branch_output,
)
from .vjepa21 import (
    ADAPTER_ID,
    BRANCH,
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
    VJepa21Worker,
)


REQUIRED_ENV_NAMES = (
    SOURCE_PATH_ENV,
    CHECKPOINT_PATH_ENV,
    CHECKPOINT_SHA256_ENV,
)
HASH_CHUNK_BYTES = 1024 * 1024


class VJepaWorkerLike(Protocol):
    def status(self) -> dict[str, Any]:
        ...

    def extract(
        self, *, video_path: str | os.PathLike[str], request: Mapping[str, Any]
    ) -> dict[str, Any]:
        ...


WorkerFactory = Callable[[VJepa21Config], VJepaWorkerLike]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _LocalVJepaPin:
    checkpoint_sha256: str
    code_revision: str = OFFICIAL_CODE_REVISION
    preprocessing_id: str = PREPROCESSING_ID
    preprocessing_sha256: str = PREPROCESSING_SHA256

    def public_provenance(self) -> dict[str, str]:
        return {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "artifactSha256": self.checkpoint_sha256,
            "weightsSha256": self.checkpoint_sha256,
            "license": MODEL_LICENSE,
            "adapterId": ADAPTER_ID,
        }


class LocalVJepa21Adapter:
    """Expose ``VJepa21Worker`` through the orchestrator adapter contract."""

    branch = BRANCH

    def __init__(
        self,
        config: VJepa21Config,
        *,
        worker: VJepaWorkerLike | None = None,
        worker_factory: WorkerFactory = VJepa21Worker,
    ) -> None:
        self.config = config
        self.worker = worker if worker is not None else worker_factory(config)
        self._pin = _LocalVJepaPin(config.checkpoint_sha256)
        self._identity: WorkerIdentity | None = None

    @classmethod
    def optional_from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        worker_factory: WorkerFactory = VJepa21Worker,
    ) -> "LocalVJepa21Adapter | None":
        source = os.environ if environ is None else environ
        present = {
            name: bool(str(source.get(name, "")).strip())
            for name in REQUIRED_ENV_NAMES
        }
        if not any(present.values()):
            return None
        if not all(present.values()):
            missing = sorted(name for name, configured in present.items() if not configured)
            raise ConfigurationError(
                "Local V-JEPA 2.1 requires all three artifact variables together; "
                f"missing {missing}."
            )
        try:
            config = VJepa21Config.from_environment(source)
        except Exception as exc:
            raise ConfigurationError(
                "Local V-JEPA 2.1 configuration failed strict validation."
            ) from exc
        return cls(config, worker_factory=worker_factory)

    def _ready_identity(self) -> WorkerIdentity:
        if self._identity is not None:
            return self._identity
        try:
            status = self.worker.status()
            identity = WorkerIdentity.from_status(
                status,
                expected_branch=BRANCH,
                expected_pin=self._pin,
            )
        except Exception as exc:
            # The original worker detail can contain local paths or dependency
            # internals, so only the adapter's stable error is surfaced.
            raise RuntimeError(
                "The local V-JEPA 2.1 worker failed readiness or provenance validation."
            ) from exc
        self._identity = identity
        return identity

    def availability(self) -> dict[str, Any]:
        try:
            identity = self._ready_identity()
        except Exception:
            return {
                "configured": True,
                "executionAvailable": False,
                "reason": (
                    "The configured local V-JEPA 2.1 worker is not ready or its "
                    "runtime provenance did not match the pinned artifacts."
                ),
                "role": "local-vjepa2.1-descriptive-representation",
                "usesLearnedModel": True,
                "isBehavioralModel": False,
                "provenance": {
                    **self._pin.public_provenance(),
                    "codeRevision": OFFICIAL_CODE_REVISION,
                    "preprocessingId": PREPROCESSING_ID,
                    "preprocessingSha256": PREPROCESSING_SHA256,
                },
            }
        return {
            "configured": True,
            "executionAvailable": True,
            "reason": None,
            "role": "local-vjepa2.1-descriptive-representation",
            "usesLearnedModel": True,
            "isBehavioralModel": False,
            "provenance": {
                **identity.public_value(),
                "license": MODEL_LICENSE,
            },
        }

    def run(
        self,
        *,
        video_path: Path,
        input_sha256: str,
        duration_seconds: float,
        context: Mapping[str, Any],
    ) -> BranchOutput:
        if not video_path.is_file():
            raise RuntimeError("The local V-JEPA 2.1 input video is unavailable.")
        try:
            actual_sha256 = _sha256_file(video_path)
        except OSError as exc:
            raise RuntimeError(
                "The local V-JEPA 2.1 input video could not be read."
            ) from exc
        if actual_sha256 != input_sha256:
            raise ValueError(
                "Local V-JEPA 2.1 input hash does not match the uploaded video."
            )
        identity = self._ready_identity()
        request = build_worker_request(
            branch=BRANCH,
            input_sha256=input_sha256,
            duration_seconds=duration_seconds,
            context=context,
        )
        output = self.worker.extract(video_path=video_path, request=request)
        return validate_branch_output(
            output,
            expected_branch=BRANCH,
            expected_input_sha256=input_sha256,
            identity=identity,
        )
