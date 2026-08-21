"""Device selection must refuse clearly rather than downgrade quietly."""

from __future__ import annotations

import unittest

from backend.forecast.workers import torch_runtime as tr


class FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class FakeMps:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class FakeBackends:
    def __init__(self, mps: bool) -> None:
        self.mps = FakeMps(mps)


class FakeTorch:
    """Only the attributes the resolver actually reads."""

    float32 = "torch.float32"
    float16 = "torch.float16"
    bfloat16 = "torch.bfloat16"

    def __init__(self, *, cuda: bool = False, mps: bool = False) -> None:
        self.cuda = FakeCuda(cuda)
        self.backends = FakeBackends(mps)


class DeviceResolutionTests(unittest.TestCase):
    def test_auto_prefers_cuda_and_records_the_choice(self) -> None:
        runtime = tr.resolve_runtime(torch_module=FakeTorch(cuda=True, mps=True))
        self.assertEqual(runtime.device, "cuda")
        self.assertEqual(runtime.dtype, "float16")
        self.assertIn("cuda", runtime.selection)
        self.assertEqual(runtime.provenance()["deviceSelection"], runtime.selection)

    def test_auto_falls_back_to_mps_then_cpu(self) -> None:
        self.assertEqual(
            tr.resolve_runtime(torch_module=FakeTorch(mps=True)).device, "mps"
        )
        cpu = tr.resolve_runtime(torch_module=FakeTorch())
        self.assertEqual(cpu.device, "cpu")
        # CPU must not land on float16: it is slower or unimplemented there.
        self.assertEqual(cpu.dtype, "float32")

    def test_an_explicit_request_for_an_absent_device_is_refused(self) -> None:
        with self.assertRaises(tr.TorchRuntimeUnavailable) as caught:
            tr.resolve_runtime(requested_device="cuda", torch_module=FakeTorch())
        message = str(caught.exception)
        self.assertIn("cuda", message)
        self.assertIn("cpu", message)
        # The refusal must say it refused, not imply a silent substitution.
        self.assertIn("refused", message)

    def test_an_explicit_available_device_is_honoured(self) -> None:
        runtime = tr.resolve_runtime(
            requested_device="cpu", torch_module=FakeTorch(cuda=True)
        )
        self.assertEqual(runtime.device, "cpu")
        self.assertIn("operator requested", runtime.selection)

    def test_tpu_is_refused_by_name_rather_than_silently_ignored(self) -> None:
        with self.assertRaises(tr.TorchRuntimeUnavailable) as caught:
            tr.resolve_runtime(requested_device="tpu", torch_module=FakeTorch())
        self.assertIn("TPU", str(caught.exception))

    def test_float16_on_cpu_is_refused(self) -> None:
        with self.assertRaises(tr.TorchRuntimeUnavailable):
            tr.resolve_runtime(
                requested_device="cpu",
                requested_dtype="float16",
                torch_module=FakeTorch(),
            )

    def test_an_absent_torch_reports_a_reason_rather_than_raising_on_import(self) -> None:
        def explode() -> object:
            raise ImportError("no torch here")

        with self.assertRaises(tr.TorchRuntimeUnavailable) as caught:
            tr.resolve_runtime(importer=explode)
        self.assertIn("torch is not installed", str(caught.exception))

    def test_unavailable_reason_is_none_when_a_device_resolves(self) -> None:
        self.assertIsNone(tr.unavailable_reason(torch_module=FakeTorch(cuda=True)))
        self.assertIn(
            "does not offer",
            tr.unavailable_reason(requested_device="cuda", torch_module=FakeTorch()),
        )

    def test_dtype_mapping_never_guesses_by_getattr(self) -> None:
        self.assertEqual(tr.torch_dtype(FakeTorch(), "bfloat16"), "torch.bfloat16")
        with self.assertRaises(tr.TorchRuntimeUnavailable):
            tr.torch_dtype(FakeTorch(), "float8")

    def test_env_reading_defaults_to_auto(self) -> None:
        self.assertEqual(tr.runtime_request_from_env({}), ("auto", "auto"))
        self.assertEqual(
            tr.runtime_request_from_env({tr.DEVICE_ENV: " CUDA ", tr.DTYPE_ENV: "bfloat16"}),
            ("cuda", "bfloat16"),
        )


if __name__ == "__main__":
    unittest.main()
