#!/usr/bin/env python3
"""Run the exact short calibration for historical Minerva 7B LoRA."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_historical_calibration import (
    calibrate_historical_lora,
)


def main() -> None:
    started_at = time.monotonic()

    def progress(message: str) -> None:
        print(
            f"historical-calibration | {message} | "
            f"elapsed={time.monotonic() - started_at:.1f}s",
            flush=True,
        )

    print(
        "historical-calibration | start warmup_updates=2 timed_updates=10 "
        "estimated_runtime=3m-15m",
        flush=True,
    )
    report = calibrate_historical_lora(
        repo_root=ROOT,
        output_path=(
            ROOT / "data/local/minerva_7b_staged/historical_calibration.json"
        ),
        progress=progress,
    )
    print(
        "historical-calibration | complete status={status} throughput={speed:.1f} "
        "tokens/s peak_reserved={memory:.1f}MiB free_after={free:.1f}MiB".format(
            status=report["status"],
            speed=report["tokens_per_second"],
            memory=report["peak_reserved_mib"],
            free=report["free_memory_after_mib"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
