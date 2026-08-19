from __future__ import annotations

import asyncio
import hashlib
import json
import io
import hashlib
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from uuid import UUID

import numpy as np

import backend.app as app_module
from backend.artifacts import VERTEX_COUNT
from backend.config import Settings
from backend.model_runtime import InferenceOutput
from backend.thumbnails import SerializedThumbnail, SerializedThumbnails


def make_settings(root: Path) -> Settings:
    return Settings(
        model_id="facebook/tribev2",
        model_revision="f894e783020944dcd96e5568550afe2aa9743f9f",
        checkpoint_name="best.ckpt",
        device="mps",
        inference_mode="vision-only",
        cache_dir=root / "cache",
        hub_cache_dir=None,
        result_dir=root / "results",
        work_dir=root / "work",
        cors_origins=("http://localhost:4173",),
        cors_allow_credentials=False,
        max_upload_bytes=4 * 1024 * 1024,
        preload_model=False,
    )


class MemoryUpload:
    def __init__(self, payload: bytes, filename: str = "clip.mp4") -> None:
        self.filename = filename
        self.content_type = "video/mp4"
        self._stream = io.BytesIO(payload)
        self.read_sizes: list[int] = []
        self.closed = False

    async def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return self._stream.read(size)

    async def close(self) -> None:
        self.closed = True


class CountingRuntime:
    def __init__(self) -> None:
        self.calls = 0
        self.paths: list[Path] = []
        self._counter_lock = threading.Lock()

    def predict_video(self, path: Path) -> InferenceOutput:
        with self._counter_lock:
            self.calls += 1
            self.paths.append(path)
        self.assert_staged_path(path)
        time.sleep(0.05)
        predictions = np.linspace(
            -1.0, 1.0, VERTEX_COUNT, dtype=np.float32
        ).reshape(1, VERTEX_COUNT)
        return InferenceOutput(
            predictions=predictions,
            segments=[{"start": 0.0, "duration": 1.0}],
            hemodynamic_offset_seconds=5.0,
        )

    @staticmethod
    def assert_staged_path(path: Path) -> None:
        if path.parent.parent.parent.name != "content-addressed":
            raise AssertionError(f"runtime received unstable upload path: {path}")


class UploadAndConcurrencyTests(unittest.TestCase):
    def test_upload_hash_is_computed_while_streaming_to_disk(self) -> None:
        payload = b"a" * (2 * app_module.UPLOAD_CHUNK_BYTES + 17)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = make_settings(root)
            destination = root / "upload.mp4"
            upload = MemoryUpload(payload)
            with patch.object(app_module, "settings", settings):
                receipt = asyncio.run(app_module._save_upload(upload, destination))

        self.assertEqual(receipt.size, len(payload))
        self.assertEqual(receipt.sha256, hashlib.sha256(payload).hexdigest())
        self.assertGreaterEqual(len(upload.read_sizes), 4)
        self.assertTrue(
            all(size == app_module.UPLOAD_CHUNK_BYTES for size in upload.read_sizes)
        )

    def test_thumbnail_endpoint_serves_only_manifest_declared_private_media(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            result_id = UUID(hex="d" * 32)
            result_directory = settings.result_dir / result_id.hex
            thumbnail_directory = result_directory / "thumbnails"
            thumbnail_directory.mkdir(parents=True)
            (thumbnail_directory / "0000.jpg").write_bytes(
                b"\xff\xd8source-frame\xff\xd9"
            )
            manifest = {
                "thumbnails": {
                    "status": "available",
                    "items": [
                        {
                            "index": 0,
                            "url": (
                                f"/api/tribe/v1/results/{result_id.hex}/"
                                "thumbnails/0000.jpg"
                            ),
                        }
                    ],
                }
            }
            (result_directory / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with patch.object(app_module, "settings", settings):
                response = app_module.get_thumbnail(result_id, 0)
                self.assertEqual(response.headers["cache-control"], "private, no-store")
                with self.assertRaisesRegex(Exception, "Thumbnail was not found"):
                    app_module.get_thumbnail(result_id, 1)

                manifest["thumbnails"] = {
                    "status": "partial",
                    "items": [
                        {
                            "index": 0,
                            "url": (
                                f"/api/tribe/v1/results/{result_id.hex}/"
                                "thumbnails/0000.jpg"
                            ),
                        }
                    ],
                    "unavailableItems": [
                        {
                            "index": 1,
                            "requestedTimeSeconds": 1.0,
                            "reason": "outside-source-video",
                        }
                    ],
                }
                (result_directory / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                response = app_module.get_thumbnail(result_id, 0)
                self.assertEqual(response.headers["cache-control"], "private, no-store")
                with self.assertRaisesRegex(Exception, "Thumbnail was not found"):
                    app_module.get_thumbnail(result_id, 1)

                manifest["thumbnails"] = {"status": "unavailable", "items": []}
                (result_directory / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                with self.assertRaisesRegex(Exception, "Thumbnail was not found"):
                    app_module.get_thumbnail(result_id, 0)

    def test_concurrent_identical_uploads_run_official_inference_once(self) -> None:
        payload = b"content-addressed test video bytes"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = make_settings(root)
            runtime = CountingRuntime()

            async def run_both():
                lock = asyncio.Lock()

                def fake_thumbnails(video_path, times_seconds, destination):
                    destination.mkdir()
                    items = []
                    for index, requested_time in enumerate(times_seconds):
                        path = destination / f"{index:04d}.jpg"
                        payload = b"\xff\xd8source-frame\xff\xd9"
                        path.write_bytes(payload)
                        items.append(
                            SerializedThumbnail(
                                index=index,
                                requested_time_seconds=float(requested_time),
                                path=path,
                                byte_length=len(payload),
                                sha256=hashlib.sha256(payload).hexdigest(),
                            )
                        )
                    return SerializedThumbnails(tuple(items))

                with (
                    patch.object(app_module, "settings", settings),
                    patch.object(app_module, "runtime", runtime),
                    patch.object(app_module, "inference_lock", lock),
                    patch.object(
                        app_module,
                        "extract_video_thumbnails",
                        fake_thumbnails,
                    ),
                ):
                    return await asyncio.gather(
                        app_module.predict(MemoryUpload(payload)),
                        app_module.predict(MemoryUpload(payload)),
                    )

            first, second = asyncio.run(run_both())

            self.assertEqual(runtime.calls, 1)
            self.assertEqual(first, second)
            self.assertEqual(first["resultId"], second["resultId"])
            self.assertIn("cache", first)
            self.assertEqual(len(list(settings.result_dir.iterdir())), 1)
            self.assertFalse((settings.work_dir / "content-addressed").exists())


if __name__ == "__main__":
    unittest.main()
