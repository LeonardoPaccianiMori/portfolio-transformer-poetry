#!/usr/bin/env python3
"""Run one fixed GPU stage of the PAISA-to-historical rescue curriculum."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.paisa_historical_rescue import build_rescue_stage_config
from sonnet_training.paisa_historical_rescue import build_rescue_training_plan
from sonnet_training.paisa_historical_rescue import write_rescue_training_plan
from sonnet_training.pretraining_run import train_pretraining_run


_DEFAULT_OUTPUT_DIRS = {
    "modern_italian_pretraining": Path(
        "runs/paisa_historical_rescue_v1_modern_italian_001"
    ),
    "historical_italian_annealing": Path(
        "runs/paisa_historical_rescue_v1_historical_italian_001"
    ),
}
_DEFAULT_HISTORICAL_PARENT = Path(
    "runs/paisa_historical_rescue_v1_modern_italian_001/best_validation.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=sorted(_DEFAULT_OUTPUT_DIRS),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument(
        "--historical-parent-checkpoint-path",
        type=Path,
        default=_DEFAULT_HISTORICAL_PARENT,
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_rescue_training_plan(ROOT)
    write_rescue_training_plan(ROOT)
    output_dir = args.output_dir or _DEFAULT_OUTPUT_DIRS[args.stage]
    config = build_rescue_stage_config(
        plan=plan,
        stage_id=args.stage,
        device=args.device,
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint) if args.resume_from_checkpoint else ""
        ),
        historical_parent_checkpoint_path=(
            str(args.historical_parent_checkpoint_path)
            if args.stage == "historical_italian_annealing"
            else ""
        ),
    )
    config = replace(config, run_label=args.stage)
    print(
        "rescue-train | stage={stage} updates={updates:,} device={device} "
        "tokens_per_update=4,096".format(
            stage=args.stage,
            updates=config.train_steps,
            device=args.device,
        ),
        flush=True,
    )
    result = train_pretraining_run(
        repo_root=ROOT,
        output_dir=ROOT / output_dir,
        config=config,
    )
    print(f"rescue-train | wrote config: {result['config_path']}", flush=True)
    print(f"rescue-train | wrote log: {result['log_path']}", flush=True)
    print(f"rescue-train | wrote best checkpoint: {result['best_checkpoint_path']}", flush=True)
    print(f"rescue-train | wrote final checkpoint: {result['checkpoint_path']}", flush=True)


if __name__ == "__main__":
    main()
