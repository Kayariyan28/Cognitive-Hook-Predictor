from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from backend.config import Settings
from backend.errors import ModelUnavailableError
from backend.model_runtime import TribeRuntime


def make_settings(
    root: Path,
    *,
    video_device: str = "cpu",
    video_precision: str = "fp32",
) -> Settings:
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
        max_upload_bytes=1024,
        preload_model=False,
        video_device=video_device,
        video_precision=video_precision,
    )


class FakeTorch:
    float16 = "fake-float16"

    def __init__(self, *, mps_available: bool, mps_built: bool = True) -> None:
        self.backends = SimpleNamespace(
            mps=SimpleNamespace(
                is_available=lambda: mps_available,
                is_built=lambda: mps_built,
            )
        )
        self.cuda = SimpleNamespace(is_available=lambda: False)
        self.autocast_calls: list[dict[str, object]] = []

    @contextmanager
    def autocast(self, **kwargs):
        self.autocast_calls.append(kwargs)
        yield


class VideoExecutionTests(unittest.TestCase):
    def test_mps_selection_fails_closed_when_backend_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = TribeRuntime(
                make_settings(Path(directory), video_device="mps")
            )
            with self.assertRaisesRegex(ModelUnavailableError, "MPS.*unavailable"):
                runtime._require_video_device(FakeTorch(mps_available=False))
            with self.assertRaisesRegex(ModelUnavailableError, "MPS.*unavailable"):
                runtime._require_video_device(
                    FakeTorch(mps_available=True, mps_built=False)
                )

    def test_post_construction_assignment_and_vjepa_only_autocast(self) -> None:
        class FakeVideoModel:
            def __init__(self, device: str, model_name: str) -> None:
                self.model = SimpleNamespace(device=SimpleNamespace(type=device))
                self.model_name = model_name

            def predict(self, value: str) -> str:
                return f"upstream:{value}"

        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(
                Path(directory),
                video_device="mps",
                video_precision="autocast-fp16",
            )
            runtime = TribeRuntime(settings)
            image = SimpleNamespace(device="cpu", model_name="mutable/hub-ref")
            model = SimpleNamespace(
                data=SimpleNamespace(
                    video_feature=SimpleNamespace(image=image)
                )
            )
            torch_module = FakeTorch(mps_available=True)
            snapshot = Path("/immutable/vjepa-snapshot")

            runtime._configure_video_execution(
                model,
                snapshot,
                torch_module=torch_module,
                video_model_class=FakeVideoModel,
            )

            self.assertEqual(image.device, "mps")
            self.assertEqual(image.model_name, str(snapshot))
            self.assertEqual(len(torch_module.autocast_calls), 1)
            mps_vjepa = FakeVideoModel("mps", "facebook/vjepa2-vitg-fpc64-256")
            self.assertEqual(mps_vjepa.predict("clip"), "upstream:clip")
            self.assertEqual(len(torch_module.autocast_calls), 2)
            self.assertEqual(
                torch_module.autocast_calls[-1],
                {"device_type": "mps", "dtype": "fake-float16"},
            )

            cpu_vjepa = FakeVideoModel("cpu", "facebook/vjepa2-vitg-fpc64-256")
            other_mps = FakeVideoModel("mps", "MCG-NJU/videomae-base")
            self.assertEqual(cpu_vjepa.predict("cpu"), "upstream:cpu")
            self.assertEqual(other_mps.predict("other"), "upstream:other")
            self.assertEqual(len(torch_module.autocast_calls), 2)

    def test_status_exposes_policy_used_by_result_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = make_settings(
                Path(directory),
                video_device="mps",
                video_precision="autocast-fp16",
            )
            status = TribeRuntime(settings).status()

        self.assertEqual(
            status["model"]["extractors"]["video"],
            settings.video_extractor,
        )


if __name__ == "__main__":
    unittest.main()
