"""Executable, provenance-pinned creator-forecast evidence workers.

Exports are loaded lazily so ``python -m ...workers.vjepa21`` does not import
the target module once through this package before executing it as ``__main__``.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "OFFICIAL_CODE_REVISION",
    "PREPROCESSING_ID",
    "PREPROCESSING_SHA256",
    "VJepa21Config",
    "VJepa21Worker",
    "VJepa21WorkerError",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    return getattr(import_module(".vjepa21", __name__), name)
