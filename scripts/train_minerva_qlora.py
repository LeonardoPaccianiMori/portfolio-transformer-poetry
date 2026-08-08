#!/usr/bin/env python3
"""Run the single fixed Minerva 3B QLoRA V5 sonnet comparison experiment."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_qlora_finetuning import (
    MinervaQLoRAFineTuningConfig,
    train_minerva_qlora_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        MinervaQLoRAFineTuningConfig(),
        resume_from_checkpoint=args.resume_from_checkpoint,
        device=args.device,
    )
    result = train_minerva_qlora_run(
        repo_root=ROOT,
        output_dir=ROOT / args.output_dir,
        config=config,
        progress=lambda message: print(message, flush=True),
    )
    best_row = result["best_validation_row"]
    print(f"minerva-train | wrote config: {result['config_path']}", flush=True)
    print(f"minerva-train | wrote log: {result['history_path']}", flush=True)
    print(
        f"minerva-train | wrote best adapter: {result['best_checkpoint_path']}",
        flush=True,
    )
    print(
        f"minerva-train | wrote final adapter: {result['final_checkpoint_path']}",
        flush=True,
    )
    print(
        "minerva-train | complete "
        f"epochs={result['completed_epoch']} steps={result['completed_step']} "
        f"reason={result['stop_reason']} best_validation={best_row}",
        flush=True,
    )


if __name__ == "__main__":
    main()
