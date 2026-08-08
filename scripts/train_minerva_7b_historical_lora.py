#!/usr/bin/env python3
"""Run the long historical-adaptation stage for Minerva 7B."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_historical_lora import (
    train_minerva_7b_historical_lora,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/minerva_7b_historical_fp16_lora_001"),
    )
    parser.add_argument("--resume-from-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.monotonic()

    def progress(message: str) -> None:
        print(
            f"minerva-historical | {message} | elapsed={time.monotonic() - started_at:.1f}s",
            flush=True,
        )

    print(
        "minerva-historical | start base=Minerva-7B-instruct-v1.0 "
        "weights=fp16 adapter=rank8-attention max_historical_passes=2",
        flush=True,
    )
    result = train_minerva_7b_historical_lora(
        repo_root=ROOT,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        progress=progress,
    )
    print(
        "minerva-historical | complete steps={steps}/{planned} reason={reason} "
        "qualified={qualified} best={best}".format(
            steps=result["completed_steps"],
            planned=result["planned_updates"],
            reason=result["stop_reason"],
            qualified=result["qualified_checkpoint"],
            best=result["best_checkpoint_path"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
