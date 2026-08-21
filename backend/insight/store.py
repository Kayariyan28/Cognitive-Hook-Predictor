"""Atomic persistence for insight artifacts, rejections, and cache entries.

Publication follows the same rule as the rest of the repository: stage, fsync,
rename.  A reader either sees a complete artifact or sees nothing.  Rejections
are persisted so an operator can read the offending sentence, but they are
never recorded as cache hits.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
from uuid import uuid4


class InsightStoreError(RuntimeError):
    """Insight state could not be read or persisted safely."""


class InsightStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.artifacts_dir = self.root / "artifacts"
        self.rejections_dir = self.root / "rejections"
        self.cache_dir = self.root / "cache"
        self.experiments_dir = self.root / "experiments"

    def initialize(self) -> None:
        for directory in (
            self.artifacts_dir,
            self.rejections_dir,
            self.cache_dir,
            self.experiments_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # -- primitives --------------------------------------------------------

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
            raise InsightStoreError("insight payload is not strict JSON") from exc
        temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            self._fsync_directory(path.parent)
        except OSError as exc:
            raise InsightStoreError("could not persist insight state") from exc
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
            raise InsightStoreError("stored insight JSON is unreadable") from exc
        if not isinstance(payload, dict):
            raise InsightStoreError("stored insight JSON is not an object")
        return payload

    def _publish_directory(
        self, parent: Path, name: str, documents: Mapping[str, Mapping[str, Any]]
    ) -> str:
        """Publish every document of one record together, or publish none of them."""

        self.initialize()
        destination = parent / name
        if destination.exists():
            raise InsightStoreError(f"a record already exists for {name}")
        staging = parent / f".publishing-{name}-{uuid4().hex}"
        try:
            staging.mkdir(parents=False, exist_ok=False)
            for filename, payload in documents.items():
                self._write_json_atomic(staging / filename, payload)
            os.replace(staging, destination)
            self._fsync_directory(parent)
            return name
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    # -- artifacts ---------------------------------------------------------

    def publish_artifact(
        self,
        insight_id: str,
        payload: Mapping[str, Any],
        bundle: Mapping[str, Any] | None = None,
    ) -> str:
        """Publish the artifact and the exact evidence it was generated from."""

        documents: dict[str, Mapping[str, Any]] = {"artifact.json": payload}
        if bundle is not None:
            documents["bundle.json"] = bundle
        return self._publish_directory(self.artifacts_dir, insight_id, documents)

    def read_bundle(self, insight_id: str) -> dict[str, Any] | None:
        """Return the evidence bundle a published artifact cites."""

        return self._read_json(self.artifacts_dir / insight_id / "bundle.json")

    def read_artifact(self, insight_id: str) -> dict[str, Any] | None:
        payload = self._read_json(self.artifacts_dir / insight_id / "artifact.json")
        if payload is None:
            return None
        if payload.get("insightId") != insight_id:
            raise InsightStoreError("stored insight artifact has an invalid identity")
        return payload

    # -- rejections --------------------------------------------------------

    def publish_rejection(self, rejection_id: str, payload: Mapping[str, Any]) -> str:
        return self._publish_directory(
            self.rejections_dir, rejection_id, {"rejection.json": payload}
        )

    def read_rejection(self, rejection_id: str) -> dict[str, Any] | None:
        return self._read_json(self.rejections_dir / rejection_id / "rejection.json")

    # -- experiments -------------------------------------------------------

    def publish_experiment(self, experiment_id: str, payload: Mapping[str, Any]) -> str:
        return self._publish_directory(
            self.experiments_dir, experiment_id, {"experiment.json": payload}
        )

    def update_experiment(self, experiment_id: str, payload: Mapping[str, Any]) -> None:
        """Advance one existing experiment; the file swap itself stays atomic."""

        directory = self.experiments_dir / experiment_id
        if not directory.is_dir():
            raise InsightStoreError(f"experiment {experiment_id} does not exist")
        self._write_json_atomic(directory / "experiment.json", payload)

    def read_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        payload = self._read_json(self.experiments_dir / experiment_id / "experiment.json")
        if payload is None:
            return None
        if payload.get("id") != experiment_id:
            raise InsightStoreError("stored experiment has an invalid identity")
        return payload

    def list_experiments(self, *, limit: int = 100) -> list[dict[str, Any]]:
        if not self.experiments_dir.is_dir():
            return []
        records: list[dict[str, Any]] = []
        for directory in sorted(self.experiments_dir.iterdir()):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            payload = self._read_json(directory / "experiment.json")
            if payload is not None:
                records.append(payload)
        records.sort(key=lambda item: str(item.get("createdAt", "")), reverse=True)
        return records[:limit]

    # -- cache -------------------------------------------------------------

    def cache_lookup(self, key: str) -> str | None:
        """Return the artifact id an exact repeat already produced, if any."""

        payload = self._read_json(self.cache_dir / f"{key}.json")
        if payload is None:
            return None
        insight_id = payload.get("insightId")
        return insight_id if isinstance(insight_id, str) and insight_id else None

    def cache_store(self, key: str, insight_id: str) -> None:
        self.initialize()
        self._write_json_atomic(self.cache_dir / f"{key}.json", {"insightId": insight_id})
