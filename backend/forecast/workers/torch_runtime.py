"""Device selection for the portable (torch) backends.

Apple silicon stays the primary target and the MLX backends stay the default.
This module exists so the same lanes can also run where MLX cannot — an NVIDIA
Colab runtime, a Linux box, a plain CPU — without any lane growing a
CUDA-only code path.

Two rules shape everything here:

* **An explicit device request is honoured or refused, never downgraded.**
  Asking for ``cuda`` on a machine with no CUDA is a configuration error the
  operator needs to see, not something to paper over with a CPU run that takes
  forty times as long and reports nothing unusual.
* **``auto`` resolves in a documented order and records what it picked.** The
  choice lands in the branch's provenance, so a result carries the device that
  produced it rather than leaving a reader to guess.

Nothing here imports ``torch`` at module scope. An absent runtime makes one
branch unavailable with a reason; it never breaks module import or another lane.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from typing import Any, Callable, Mapping


DEVICE_ENV = "SIGNALFRAME_TORCH_DEVICE"
DTYPE_ENV = "SIGNALFRAME_TORCH_DTYPE"

REQUESTABLE_DEVICES = ("auto", "cuda", "mps", "cpu")
# The order `auto` tries. CUDA first because that is the accelerator on the
# Linux hosts this path was added for; MPS next so an Apple machine that opted
# into the portable backend still gets its GPU; CPU last and always available.
AUTO_ORDER = ("cuda", "mps", "cpu")
REQUESTABLE_DTYPES = ("auto", "float32", "float16", "bfloat16")


class TorchRuntimeUnavailable(RuntimeError):
    """No usable torch device for the requested configuration."""


def module_available(name: str) -> bool:
    """Cheap probe. Imports nothing and never touches the network."""

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def torch_installed() -> bool:
    return module_available("torch")


@dataclass(frozen=True, slots=True)
class TorchRuntime:
    """The device and dtype a portable backend actually ran on."""

    requested_device: str
    device: str
    dtype: str
    selection: str

    def provenance(self) -> dict[str, Any]:
        return {
            "requestedDevice": self.requested_device,
            "device": self.device,
            "dtype": self.dtype,
            "deviceSelection": self.selection,
        }


def _probe_cuda(torch: Any) -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive
        return False


def _probe_mps(torch: Any) -> bool:
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def available_devices(torch: Any) -> tuple[str, ...]:
    """Every device this process could actually place a model on."""

    found = ["cpu"]
    if _probe_mps(torch):
        found.insert(0, "mps")
    if _probe_cuda(torch):
        found.insert(0, "cuda")
    return tuple(found)


def default_dtype_for(device: str) -> str:
    """The dtype each device runs well in when the operator did not choose.

    CUDA gets float16 because these are inference-only lanes on hardware that
    is markedly faster in half precision. MPS gets float16 for the same reason.
    CPU stays float32: half precision on CPU is usually slower, not faster, and
    on some builds is not implemented at all.
    """

    return "float32" if device == "cpu" else "float16"


def resolve_runtime(
    *,
    requested_device: str = "auto",
    requested_dtype: str = "auto",
    torch_module: Any | None = None,
    importer: Callable[[], Any] | None = None,
) -> TorchRuntime:
    """Pick a device, or refuse with a reason an operator can act on."""

    device_request = (requested_device or "auto").strip().lower()
    if device_request not in REQUESTABLE_DEVICES:
        raise TorchRuntimeUnavailable(
            f"{DEVICE_ENV} must be one of {list(REQUESTABLE_DEVICES)}; "
            f"{requested_device!r} is not a device this build selects. "
            "TPU/XLA is not supported by these lanes."
        )
    dtype_request = (requested_dtype or "auto").strip().lower()
    if dtype_request not in REQUESTABLE_DTYPES:
        raise TorchRuntimeUnavailable(
            f"{DTYPE_ENV} must be one of {list(REQUESTABLE_DTYPES)}"
        )

    torch = torch_module
    if torch is None:
        try:
            torch = (importer or _import_torch)()
        except Exception as exc:
            raise TorchRuntimeUnavailable(
                "torch is not installed in this backend environment."
            ) from exc

    present = available_devices(torch)
    if device_request == "auto":
        device = next(name for name in AUTO_ORDER if name in present)
        selection = (
            f"auto selected {device}; devices present: {', '.join(present)}"
        )
    else:
        if device_request not in present:
            raise TorchRuntimeUnavailable(
                f"{DEVICE_ENV} requested {device_request!r}, which this machine does "
                f"not offer. Devices present: {', '.join(present)}. The request is "
                "refused rather than quietly run somewhere else."
            )
        device = device_request
        selection = f"operator requested {device}"

    dtype = default_dtype_for(device) if dtype_request == "auto" else dtype_request
    if device == "cpu" and dtype in {"float16"}:
        raise TorchRuntimeUnavailable(
            "float16 on CPU is not a supported combination for these lanes; "
            "use float32 or bfloat16."
        )
    return TorchRuntime(
        requested_device=device_request, device=device, dtype=dtype, selection=selection
    )


def _import_torch() -> Any:
    import torch

    return torch


def torch_dtype(torch: Any, dtype: str) -> Any:
    """Map our dtype name onto the torch object, with no getattr guesswork."""

    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in table:
        raise TorchRuntimeUnavailable(f"unsupported dtype {dtype!r}")
    return table[dtype]


def runtime_request_from_env(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Read the requested device and dtype. Validation happens on resolve."""

    source = os.environ if environ is None else environ
    device = str(source.get(DEVICE_ENV, "auto")).strip().lower() or "auto"
    dtype = str(source.get(DTYPE_ENV, "auto")).strip().lower() or "auto"
    return device, dtype


def unavailable_reason(
    *,
    requested_device: str = "auto",
    requested_dtype: str = "auto",
    torch_module: Any | None = None,
) -> str | None:
    """``None`` when a portable backend could run here, else why it cannot."""

    if torch_module is None and not torch_installed():
        return "torch is not installed in this backend environment."
    try:
        resolve_runtime(
            requested_device=requested_device,
            requested_dtype=requested_dtype,
            torch_module=torch_module,
        )
    except TorchRuntimeUnavailable as exc:
        return str(exc)
    return None
