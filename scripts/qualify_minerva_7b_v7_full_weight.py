#!/usr/bin/env python3
"""Run the frozen dual-H100 2,048-context qualification after approval."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_v7_trainer import (
    qualify_minerva_7b_v7_full_weight,
)


def main() -> None:
    rank = int(os.environ.get("RANK", "0"))
    started = time.monotonic()

    def progress(message: str) -> None:
        if rank == 0:
            print(
                f"minerva-v7-qualification | {message} | "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    if rank == 0:
        print(
            "minerva-v7-qualification | start job=bounded-dual-h100-qualification "
            "device=2xh100-80gb context=2048 candidates=12 warmup_updates=3 "
            "timed_updates=20 progress_interval=1",
            flush=True,
        )
    qualify_minerva_7b_v7_full_weight(repo_root=ROOT, progress=progress)


if __name__ == "__main__":
    main()
