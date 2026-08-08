#!/usr/bin/env python3
"""Run Minerva 7B Stage B specialization on the corrected V6 sonnets."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_sonnet_lora import train_minerva_7b_sonnet_lora


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/minerva_7b_v6_sonnet_fp16_lora_001"),
    )
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()
    started_at = time.monotonic()

    def progress(message: str) -> None:
        print(
            f"minerva-sonnet | {message} | elapsed={time.monotonic() - started_at:.1f}s",
            flush=True,
        )

    print(
        "minerva-sonnet | start parent=historical-step-4000 corpus=V6 "
        "weights=fp16 adapter=rank8-attention",
        flush=True,
    )
    result = train_minerva_7b_sonnet_lora(
        repo_root=ROOT,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        progress=progress,
    )
    print(
        "minerva-sonnet | complete epochs={epochs} steps={steps}/{planned} "
        "reason={reason} qualifying_candidates={count}".format(
            epochs=result["completed_epoch"],
            steps=result["completed_step"],
            planned=result["planned_updates"],
            reason=result["stop_reason"],
            count=len(result["top_qualifying_candidates"]),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
