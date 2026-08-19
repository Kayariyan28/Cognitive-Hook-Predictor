from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from backend.artifacts import (
    VERTEX_COUNT,
    build_manifest,
    calculate_display_scale,
    extract_segment_timing,
    write_frames,
)
from backend.config import (
    PREPROCESSING_VERSION,
    TRIBE_CODE_REVISION,
    TRIBE_MODEL_WEIGHTS_SHA256,
    Settings,
)
from backend.result_cache import (
    CACHE_IDENTITY_VERSION,
    build_cache_identity,
    cache_manifest_metadata,
    load_cached_result,
    memoize_completed_result,
    stable_staging_path,
    stage_uploaded_video,
)
from backend.thumbnails import (
    SerializedThumbnail,
    SerializedThumbnails,
    SerializedUnavailableThumbnail,
    thumbnails_manifest,
    unavailable_thumbnails_manifest,
)


def make_settings(root: Path, *, mode: str = "vision-only") -> Settings:
    return Settings(
        model_id="facebook/tribev2",
        model_revision="f894e783020944dcd96e5568550afe2aa9743f9f",
        checkpoint_name="best.ckpt",
        device="mps",
        inference_mode=mode,
        cache_dir=root / "cache",
        hub_cache_dir=None,
        result_dir=root / "results",
        work_dir=root / "work",
        cors_origins=("http://localhost:4173",),
        cors_allow_credentials=False,
        max_upload_bytes=1024 * 1024,
        preload_model=False,
    )


def create_completed_result(
    settings: Settings, input_sha256: str, *, result_id: str = "a" * 32
):
    identity = build_cache_identity(settings, input_sha256)
    result_directory = settings.result_dir / result_id
    result_directory.mkdir(parents=True)
    predictions = np.linspace(
        -1.0, 1.0, 2 * VERTEX_COUNT, dtype=np.float32
    ).reshape(2, VERTEX_COUNT)
    frames = write_frames(predictions, result_directory / "frames.f32")
    timing = extract_segment_timing(
        [
            {"start": 0.0, "duration": 1.0},
            {"start": 1.0, "duration": 1.0},
        ],
        expected_frames=2,
    )
    manifest = build_manifest(
        result_id=result_id,
        model_id=settings.model_id,
        model_revision=settings.model_revision,
        model_weights_sha256=TRIBE_MODEL_WEIGHTS_SHA256,
        inference_mode=settings.inference_mode,
        modalities_used=settings.modalities_used,
        video_extractor=settings.video_extractor,
        frames=frames,
        timing=timing,
        hemodynamic_offset_seconds=5.0,
        display_scale=calculate_display_scale(predictions),
    )
    manifest["cache"] = cache_manifest_metadata(identity)
    thumbnail_items = []
    thumbnail_directory = result_directory / "thumbnails"
    thumbnail_directory.mkdir()
    for index, requested_time in enumerate(timing.times_seconds):
        path = thumbnail_directory / f"{index:04d}.jpg"
        payload = b"\xff\xd8" + bytes([index]) + b"source-frame" + b"\xff\xd9"
        path.write_bytes(payload)
        thumbnail_items.append(
            SerializedThumbnail(
                index=index,
                requested_time_seconds=requested_time,
                path=path,
                byte_length=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    manifest["thumbnails"] = thumbnails_manifest(
        result_id, SerializedThumbnails(tuple(thumbnail_items))
    )
    (result_directory / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    memoize_completed_result(
        settings.cache_dir, result_directory, identity, settings
    )
    return identity, result_directory, manifest


class CacheIdentityTests(unittest.TestCase):
    def test_identity_contains_every_required_pin_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            digest = "1" * 64
            first = build_cache_identity(settings, digest)
            second = build_cache_identity(settings, digest)

        self.assertEqual(first, second)
        self.assertEqual(first.payload["identityVersion"], CACHE_IDENTITY_VERSION)
        self.assertEqual(first.payload["inputSha256"], digest)
        self.assertEqual(first.payload["inferenceMode"], "vision-only")
        self.assertEqual(
            first.payload["preprocessingVersion"], PREPROCESSING_VERSION
        )
        self.assertEqual(
            first.payload["tribe"]["weightsSha256"],
            TRIBE_MODEL_WEIGHTS_SHA256,
        )
        self.assertEqual(
            first.payload["tribe"]["codeRevision"], TRIBE_CODE_REVISION
        )
        self.assertEqual(
            first.payload["videoExtractor"], settings.video_extractor
        )
        self.assertEqual(len(first.key), 64)

    def test_identity_invalidates_on_input_mode_revision_or_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            baseline = build_cache_identity(settings, "2" * 64)
            changed = {
                build_cache_identity(settings, "3" * 64).key,
                build_cache_identity(replace(settings, inference_mode="full"), "2" * 64).key,
                build_cache_identity(
                    replace(settings, model_revision="e" * 40), "2" * 64
                ).key,
                build_cache_identity(
                    settings, "2" * 64, preprocessing_version="next-preprocessing"
                ).key,
                build_cache_identity(
                    replace(settings, video_device="mps"), "2" * 64
                ).key,
                build_cache_identity(
                    replace(
                        settings,
                        video_device="mps",
                        video_precision="autocast-fp16",
                    ),
                    "2" * 64,
                ).key,
            }

        self.assertEqual(len(changed), 6)
        self.assertNotIn(baseline.key, changed)


class StableStagingTests(unittest.TestCase):
    def test_path_is_stable_for_exca_and_raw_scratch_is_removed(self) -> None:
        payload = b"same uploaded bytes"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = make_settings(root)
            identity = build_cache_identity(settings, digest)
            first = stable_staging_path(settings.work_dir, identity, ".MP4")
            second = stable_staging_path(settings.work_dir, identity, ".mp4")
            self.assertEqual(first, second)
            self.assertIn(identity.key, str(first))

            source = root / "temporary-upload"
            source.write_bytes(payload)
            with stage_uploaded_video(
                source,
                first,
                expected_sha256=digest,
                expected_size=len(payload),
            ) as staged:
                self.assertEqual(staged, first)
                self.assertEqual(staged.read_bytes(), payload)
                self.assertFalse(source.exists())
                staged.with_suffix(".wav").write_bytes(b"scratch audio")

            self.assertFalse(first.parent.exists())


class CompletedResultCacheTests(unittest.TestCase):
    def test_reuses_only_a_fully_validated_completed_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            identity, _, expected = create_completed_result(settings, "4" * 64)

            loaded = load_cached_result(
                settings.cache_dir, settings.result_dir, identity, settings
            )

        self.assertEqual(loaded, expected)

    def test_rejects_same_length_frames_with_wrong_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            identity, result_directory, _ = create_completed_result(
                settings, "5" * 64
            )
            frames = result_directory / "frames.f32"
            with frames.open("r+b") as handle:
                original = handle.read(1)
                handle.seek(0)
                handle.write(bytes([original[0] ^ 0xFF]))

            loaded = load_cached_result(
                settings.cache_dir, settings.result_dir, identity, settings
            )

        self.assertIsNone(loaded)

    def test_rejects_thumbnail_timing_or_source_bytes_that_do_not_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            identity, result_directory, manifest = create_completed_result(
                settings, "9" * 64
            )
            thumbnail = result_directory / "thumbnails" / "0000.jpg"
            payload = thumbnail.read_bytes()
            thumbnail.write_bytes(payload[:-3] + b"bad")
            self.assertIsNone(
                load_cached_result(
                    settings.cache_dir, settings.result_dir, identity, settings
                )
            )

            identity, result_directory, manifest = create_completed_result(
                settings, "a" * 64, result_id="c" * 32
            )
            manifest["thumbnails"]["items"][0]["requestedTimeSeconds"] = 0.25
            (result_directory / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertIsNone(
                load_cached_result(
                    settings.cache_dir, settings.result_dir, identity, settings
                )
            )

    def test_accepts_explicit_thumbnail_unavailability_without_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            identity, result_directory, manifest = create_completed_result(
                settings, "b" * 64
            )
            manifest["thumbnails"] = unavailable_thumbnails_manifest()
            (result_directory / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            for path in (result_directory / "thumbnails").iterdir():
                path.unlink()
            (result_directory / "thumbnails").rmdir()

            loaded = load_cached_result(
                settings.cache_dir, settings.result_dir, identity, settings
            )

        self.assertEqual(loaded, manifest)

    def test_accepts_partial_thumbnails_as_an_exact_frame_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            identity, result_directory, manifest = create_completed_result(
                settings, "d" * 64
            )
            first_item = manifest["thumbnails"]["items"][0]
            first_path = result_directory / "thumbnails" / "0000.jpg"
            manifest["thumbnails"] = thumbnails_manifest(
                result_directory.name,
                SerializedThumbnails(
                    (
                        SerializedThumbnail(
                            index=0,
                            requested_time_seconds=0.0,
                            path=first_path,
                            byte_length=first_item["byteLength"],
                            sha256=first_item["sha256"],
                        ),
                    ),
                    (
                        SerializedUnavailableThumbnail(
                            index=1,
                            requested_time_seconds=1.0,
                        ),
                    ),
                ),
            )
            (result_directory / "thumbnails" / "0001.jpg").unlink()
            (result_directory / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            loaded = load_cached_result(
                settings.cache_dir, settings.result_dir, identity, settings
            )

        self.assertEqual(loaded, manifest)

    def test_rejects_partial_thumbnail_partition_or_reason_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            identity, result_directory, manifest = create_completed_result(
                settings, "e" * 64
            )
            first_item = manifest["thumbnails"]["items"][0]
            first_path = result_directory / "thumbnails" / "0000.jpg"
            manifest["thumbnails"] = thumbnails_manifest(
                result_directory.name,
                SerializedThumbnails(
                    (
                        SerializedThumbnail(
                            index=0,
                            requested_time_seconds=0.0,
                            path=first_path,
                            byte_length=first_item["byteLength"],
                            sha256=first_item["sha256"],
                        ),
                    ),
                    (SerializedUnavailableThumbnail(1, 1.0),),
                ),
            )
            (result_directory / "thumbnails" / "0001.jpg").unlink()
            manifest["thumbnails"]["unavailableItems"][0]["index"] = 0
            (result_directory / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertIsNone(
                load_cached_result(
                    settings.cache_dir, settings.result_dir, identity, settings
                )
            )

            identity, result_directory, manifest = create_completed_result(
                settings, "f" * 64, result_id="d" * 32
            )
            first_item = manifest["thumbnails"]["items"][0]
            first_path = result_directory / "thumbnails" / "0000.jpg"
            manifest["thumbnails"] = thumbnails_manifest(
                result_directory.name,
                SerializedThumbnails(
                    (
                        SerializedThumbnail(
                            index=0,
                            requested_time_seconds=0.0,
                            path=first_path,
                            byte_length=first_item["byteLength"],
                            sha256=first_item["sha256"],
                        ),
                    ),
                    (SerializedUnavailableThumbnail(1, 1.0),),
                ),
            )
            (result_directory / "thumbnails" / "0001.jpg").unlink()
            manifest["thumbnails"]["unavailableItems"][0]["reason"] = (
                "source-frame-extraction-failed"
            )
            (result_directory / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertIsNone(
                load_cached_result(
                    settings.cache_dir, settings.result_dir, identity, settings
                )
            )

    def test_legacy_thumbnail_contract_cache_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            identity, result_directory, manifest = create_completed_result(
                settings, "0" * 64
            )
            manifest["cache"].pop("thumbnailContractVersion")
            manifest["thumbnails"] = unavailable_thumbnails_manifest()
            (result_directory / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            for path in (result_directory / "thumbnails").iterdir():
                path.unlink()
            (result_directory / "thumbnails").rmdir()

            loaded = load_cached_result(
                settings.cache_dir, settings.result_dir, identity, settings
            )

        self.assertIsNone(loaded)

    def test_rejects_truncated_frames_and_manifest_provenance_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            identity, result_directory, manifest = create_completed_result(
                settings, "6" * 64
            )
            frames = result_directory / "frames.f32"
            frames.write_bytes(frames.read_bytes()[:-4])
            self.assertIsNone(
                load_cached_result(
                    settings.cache_dir, settings.result_dir, identity, settings
                )
            )

            # Restore a fresh artifact, then prove manifest provenance is also
            # checked instead of trusting the cache pointer alone.
            identity, result_directory, manifest = create_completed_result(
                settings, "7" * 64, result_id="b" * 32
            )
            manifest["model"]["revision"] = "0" * 40
            (result_directory / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertIsNone(
                load_cached_result(
                    settings.cache_dir, settings.result_dir, identity, settings
                )
            )

    def test_missing_artifact_is_never_treated_as_a_hit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(Path(directory))
            identity = build_cache_identity(settings, "8" * 64)
            loaded = load_cached_result(
                settings.cache_dir, settings.result_dir, identity, settings
            )

        self.assertIsNone(loaded)


if __name__ == "__main__":
    unittest.main()
