#!/usr/bin/env python3
"""Run the locked five-update Minerva 7B full-weight H100 calibration."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_full_weight_calibration import (
    Minerva7BFullWeightCalibrationConfig,
    calibrate_minerva_7b_full_weight,
)


def main() -> None:
    config = Minerva7BFullWeightCalibrationConfig()
    started_at = time.monotonic()

    def progress(message: str) -> None:
        print(
            f"minerva-full-calibration | {message} | "
            f"elapsed={_format_duration(time.monotonic() - started_at)}",
            flush=True,
        )

    print(
        "minerva-full-calibration | start model=Minerva-7B-instruct-v1.0 "
        "weights=all_trainable dtype=bf16 optimizer=PagedAdamW8bit "
        "updates=5 context=512 microbatch=1 estimated_runtime=5m-20m_cached",
        flush=True,
    )
    report = calibrate_minerva_7b_full_weight(
        repo_root=ROOT,
        config=config,
        progress=progress,
    )
    print(
        "minerva-full-calibration | complete status={status} fit={fit} "
        "peak_reserved={peak:.1f}MiB headroom={headroom:.1f}MiB "
        "checkpoint_retained=false output={output}".format(
            status=report["status"],
            fit=report["full_weight_training_fit_decision"],
            peak=report["peak_reserved_mib"],
            headroom=report["minimum_free_after_optimizer_mib"],
            output=ROOT / config.output_path,
        ),
        flush=True,
    )


def _format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


if __name__ == "__main__":
    main()
