"""Mechanical recuts, so an experiment can be measured without leaving the app.

An experiment proposes an edit. Until now the creator had to leave, make that
edit in another application, and come back — which is where experiment tracking
quietly dies. This module performs the small mechanical edits a hook experiment
usually needs, deterministically, with the same ffmpeg contract the rest of the
pipeline uses.

Two boundaries hold. The **operation is chosen by a person, never by the model**:
a language model proposes an edit in words, and a creator picks one of the
code-owned operations below. And the recut is a transient render — the clip goes
back to the creator, who submits it as an ordinary new job, so nothing here
retains creator media or bypasses job admission.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping


RECUT_SCHEMA_VERSION = "insight-recut/1"

# Code-owned. Each one is a mechanical edit with an unambiguous definition, not
# a creative judgement, and each is reversible by re-uploading the original.
OPERATIONS: Mapping[str, str] = {
    "trim_start": "Remove the first N seconds.",
    "trim_end": "Remove the last N seconds.",
    "keep_window": "Keep only the window from start to end.",
}

MINIMUM_RESULT_SECONDS = 10.0
MAXIMUM_RESULT_SECONDS = 60.0
MAXIMUM_TRIM_SECONDS = 30.0


class RecutError(ValueError):
    """The requested recut is refused; nothing was rendered."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail

    def as_dict(self) -> dict[str, Any]:
        return {"unavailable": True, "reasonCode": self.reason_code, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class RecutPlan:
    operation: str
    startSec: float
    endSec: float
    sourceDurationSec: float

    @property
    def result_duration(self) -> float:
        return round(self.endSec - self.startSec, 6)

    def public_value(self) -> dict[str, Any]:
        return {
            "schemaVersion": RECUT_SCHEMA_VERSION,
            "operation": self.operation,
            "operationDescription": OPERATIONS[self.operation],
            "startSec": self.startSec,
            "endSec": self.endSec,
            "sourceDurationSec": self.sourceDurationSec,
            "resultDurationSec": self.result_duration,
            "behavioralOutcome": False,
            "limits": (
                "A recut changes the clip, not any measurement of it. The result must be "
                "analysed as a new clip before anything can be said about it, and a "
                "measured difference between the two is a signal change, not an outcome."
            ),
        }


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecutError("invalid_request", f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise RecutError("invalid_request", f"{label} must be finite")
    return number


def plan_recut(
    operation: str,
    *,
    duration_seconds: float,
    seconds: Any = None,
    start_seconds: Any = None,
    end_seconds: Any = None,
) -> RecutPlan:
    """Validate a recut before any frame is decoded. Pure, and fails closed."""

    if operation not in OPERATIONS:
        raise RecutError(
            "unknown_operation", f"operation must be one of {sorted(OPERATIONS)}"
        )
    duration = _finite(duration_seconds, "durationSeconds")
    if duration <= 0:
        raise RecutError("invalid_request", "the source duration must be positive")

    if operation == "keep_window":
        start = _finite(start_seconds, "startSec")
        end = _finite(end_seconds, "endSec")
    else:
        amount = _finite(seconds, "seconds")
        if amount <= 0:
            raise RecutError("invalid_request", "seconds must be greater than zero")
        if amount > MAXIMUM_TRIM_SECONDS:
            raise RecutError(
                "invalid_request",
                f"a single trim is limited to {MAXIMUM_TRIM_SECONDS:g} seconds",
            )
        start = amount if operation == "trim_start" else 0.0
        end = duration if operation == "trim_start" else duration - amount

    start = round(max(0.0, start), 6)
    end = round(min(duration, end), 6)
    if end <= start:
        raise RecutError("invalid_request", "the recut would leave no clip at all")
    result = round(end - start, 6)
    if result < MINIMUM_RESULT_SECONDS or result > MAXIMUM_RESULT_SECONDS:
        raise RecutError(
            "duration_out_of_range",
            f"the recut would be {result:g} s; a clip must be "
            f"{MINIMUM_RESULT_SECONDS:g}–{MAXIMUM_RESULT_SECONDS:g} seconds to be analysed",
        )
    return RecutPlan(
        operation=operation, startSec=start, endSec=end, sourceDurationSec=duration
    )


@dataclass(slots=True)
class FfmpegRecutRunner:
    """Re-encode the selected window deterministically, with no shell."""

    binary: str = "ffmpeg"
    timeout_seconds: float = 120.0

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def render(self, source: Path, plan: RecutPlan, destination: Path) -> Path:
        command = [
            self.binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-ss",
            f"{plan.startSec:.6f}",
            "-to",
            f"{plan.endSec:.6f}",
            "-i",
            str(source),
            "-map",
            "0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-y",
            str(destination),
        ]
        try:
            completed = subprocess.run(
                command, check=False, capture_output=True, timeout=self.timeout_seconds
            )
        except FileNotFoundError as exc:
            raise RecutError(
                "recut_unavailable", "The configured ffmpeg executable is unavailable."
            ) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise RecutError("recut_failed", "The recut could not be rendered.") from exc
        if completed.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
            raise RecutError("recut_failed", "ffmpeg produced no usable clip.")
        return destination


class RecutAssistant:
    """Plan and render one recut. It stores nothing and measures nothing."""

    def __init__(
        self,
        runner: Any | None = None,
        *,
        availability_probe: Callable[[], bool] | None = None,
    ) -> None:
        self.runner = runner or FfmpegRecutRunner()
        self._availability_probe = availability_probe

    def available(self) -> bool:
        if self._availability_probe is not None:
            return bool(self._availability_probe())
        probe = getattr(self.runner, "available", None)
        return bool(probe()) if callable(probe) else True

    def recut(
        self,
        source: Path,
        destination: Path,
        *,
        operation: str,
        duration_seconds: float,
        seconds: Any = None,
        start_seconds: Any = None,
        end_seconds: Any = None,
    ) -> tuple[Path, RecutPlan]:
        if not self.available():
            raise RecutError(
                "recut_unavailable",
                "No ffmpeg executable is configured, so recuts cannot be rendered.",
            )
        if not source.is_file():
            raise RecutError("invalid_request", "the source clip is unavailable")
        plan = plan_recut(
            operation,
            duration_seconds=duration_seconds,
            seconds=seconds,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        return self.runner.render(source, plan, destination), plan
