"""Console progress reporting for long-running training jobs."""

from __future__ import annotations

from time import perf_counter
from typing import Callable


class TrainingProgressReporter:
    """Print periodic, flushed progress lines with elapsed time and ETA."""

    def __init__(
        self,
        *,
        total_steps: int,
        progress_interval: int,
        start_step: int = 0,
        clock: Callable[[], float] = perf_counter,
    ):
        if total_steps <= 0:
            raise ValueError("total_steps must be greater than 0")
        if progress_interval <= 0:
            raise ValueError("progress_interval must be greater than 0")
        if start_step < 0 or start_step >= total_steps:
            raise ValueError("start_step must be in [0, total_steps)")

        self.total_steps = total_steps
        self.progress_interval = progress_interval
        self.start_step = start_step
        self._clock = clock
        self._started_at = clock()

    def write_start(self, *, label: str, device: str) -> None:
        """Report the start of a job before its first optimization step."""
        print(
            f"{label} started | steps={self.start_step}/{self.total_steps} "
            f"| device={device} | progress_interval={self.progress_interval}",
            flush=True,
        )

    def should_report(self, step: int, *, force: bool = False) -> bool:
        """Return whether this completed step should be shown to the user."""
        return (
            force
            or step == self.start_step + 1
            or step % self.progress_interval == 0
            or step == self.total_steps
        )

    def write_progress(
        self,
        *,
        step: int,
        train_loss: float,
        validation_loss: float | None = None,
        learning_rate: float | None = None,
        checkpoint_written: bool = False,
        best_validation: bool = False,
    ) -> None:
        """Print one completed-step summary with throughput-derived ETA."""
        if step <= self.start_step or step > self.total_steps:
            raise ValueError("step must be after start_step and at most total_steps")

        elapsed_seconds = self._clock() - self._started_at
        completed_steps = step - self.start_step
        steps_per_second = completed_steps / elapsed_seconds if elapsed_seconds else 0.0
        remaining_steps = self.total_steps - step
        eta_seconds = remaining_steps / steps_per_second if steps_per_second else 0.0
        fields = [
            f"step={step}/{self.total_steps}",
            f"progress={100 * step / self.total_steps:.1f}%",
            f"train_loss={train_loss:.4f}",
            f"elapsed={format_duration(elapsed_seconds)}",
            f"eta={format_duration(eta_seconds)}",
        ]
        if validation_loss is not None:
            fields.append(f"validation_loss={validation_loss:.4f}")
        if learning_rate is not None:
            fields.append(f"learning_rate={learning_rate:.2e}")
        if checkpoint_written:
            fields.append("checkpoint=saved")
        if best_validation:
            fields.append("best_validation=updated")

        print("progress | " + " | ".join(fields), flush=True)


def format_duration(seconds: float) -> str:
    """Format a non-negative duration compactly for terminal progress output."""
    rounded_seconds = max(0, round(seconds))
    hours, remainder = divmod(rounded_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{remaining_seconds:02d}s"
    if minutes:
        return f"{minutes}m{remaining_seconds:02d}s"
    return f"{remaining_seconds}s"
