#!/usr/bin/env python3
"""Run the frozen three-stage Minerva V7 curriculum after separate approval."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_v7_execution import V7ExecutionConfig
from sonnet_training.minerva_7b_v7_trainer import train_minerva_7b_v7_full_weight


def main() -> None:
    rank = int(os.environ.get("RANK", "0"))
    started = time.monotonic()

    def progress(message: str) -> None:
        if rank == 0:
            print(
                f"minerva-v7-training | {message} | "
                f"job_elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    if rank == 0:
        print(
            "minerva-v7-training | start job=minerva-v7-full-weight device=2xh100-80gb "
            "total_steps=2960 progress_interval=stage-specific context=2048 "
            "tokens_per_update=32768",
            flush=True,
        )
    config = V7ExecutionConfig(
        repo_root=ROOT,
        execution_path=ROOT / "configs/minerva_7b_v7_execution.json",
        encoded_dir=ROOT / "data/local/minerva_7b_v7/encoded",
        window_index_dir=ROOT / "data/local/minerva_7b_v7/window_indexes",
        modern_encoded_dir=ROOT / "data/local/minerva_7b_full_weight/encoded",
        modern_index_path=ROOT
        / "data/local/minerva_7b_v7/modern_preservation_validation_v1.jsonl",
    )
    result = train_minerva_7b_v7_full_weight(
        repo_root=ROOT,
        execution_config=config,
        resume_from_checkpoint=(
            Path(os.environ["V7_RESUME_CHECKPOINT"])
            if os.environ.get("V7_RESUME_CHECKPOINT")
            else None
        ),
        progress=progress,
    )
    if rank == 0:
        print(
            f"minerva-v7-training | complete status={result['status']} "
            f"elapsed={time.monotonic() - started:.1f}s output={result['telemetry_path']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
