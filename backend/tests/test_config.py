from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.config import Settings
from backend.errors import ConfigurationError


def settings_with_revision(revision: str) -> Settings:
    root = Path(tempfile.gettempdir()) / "tribe-backend-test"
    return Settings(
        model_id="facebook/tribev2",
        model_revision=revision,
        checkpoint_name="best.ckpt",
        device="cuda",
        inference_mode="full",
        cache_dir=root / "cache",
        hub_cache_dir=None,
        result_dir=root / "results",
        work_dir=root / "work",
        cors_origins=("http://localhost:4173",),
        cors_allow_credentials=False,
        max_upload_bytes=1024,
        preload_model=False,
    )


class ImmutableRevisionTests(unittest.TestCase):
    def test_accepts_only_full_commit_sha(self) -> None:
        settings_with_revision("a" * 40).require_immutable_model_revision()

        for revision in ("", "main", "a" * 39, "release-v1"):
            with self.subTest(revision=revision):
                with self.assertRaises(ConfigurationError):
                    settings_with_revision(revision).require_immutable_model_revision()

    def test_mode_exposes_only_the_enabled_model_modalities(self) -> None:
        full = settings_with_revision("a" * 40)
        self.assertEqual(full.modalities_used, ("video", "audio", "text"))
        vision = Settings(
            **{
                field: getattr(full, field)
                for field in full.__dataclass_fields__
                if field != "inference_mode"
            },
            inference_mode="vision-only",
        )
        self.assertEqual(vision.modalities_used, ("video",))

    def test_rejects_unknown_inference_mode_from_environment(self) -> None:
        with patch.dict(
            "os.environ", {"TRIBE_INFERENCE_MODE": "proxy"}, clear=False
        ):
            with self.assertRaisesRegex(ConfigurationError, "TRIBE_INFERENCE_MODE"):
                Settings.from_env()

    def test_video_execution_policy_is_explicit_provenance(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "TRIBE_VIDEO_DEVICE": "MPS",
                "TRIBE_VIDEO_PRECISION": "AUTOCAST-FP16",
            },
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.video_device, "mps")
        self.assertEqual(settings.video_precision, "autocast-fp16")
        self.assertEqual(
            settings.video_extractor,
            {
                "id": "facebook/vjepa2-vitg-fpc64-256",
                "revision": "875c192b7b704b87d1e1d99345769632dd5f739a",
                "weightsSha256": (
                    "f205e77aa2ade168db6b09d4bc420d156141f64ab964278a9c181a2bdf2a232b"
                ),
                "device": "mps",
                "precision": "autocast-fp16",
            },
        )

    def test_rejects_unsupported_or_unsafe_video_execution_policy(self) -> None:
        invalid_environments = (
            {"TRIBE_VIDEO_DEVICE": "metal"},
            {"TRIBE_VIDEO_PRECISION": "fp16"},
            {
                "TRIBE_VIDEO_DEVICE": "cpu",
                "TRIBE_VIDEO_PRECISION": "autocast-fp16",
            },
            {
                "TRIBE_VIDEO_DEVICE": "cuda",
                "TRIBE_VIDEO_PRECISION": "autocast-fp16",
            },
        )
        for environment in invalid_environments:
            with self.subTest(environment=environment):
                with patch.dict("os.environ", environment, clear=True):
                    with self.assertRaises(ConfigurationError):
                        Settings.from_env()


if __name__ == "__main__":
    unittest.main()
