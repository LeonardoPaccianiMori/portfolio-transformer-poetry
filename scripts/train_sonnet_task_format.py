#!/usr/bin/env python3
"""Run resumable opening-line sonnet continuation post-training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.task_format_run import (
    TaskFormatRunConfig,
    train_task_format_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="expanded_with_petrarch")
    parser.add_argument(
        "--manifest-path",
        default="data/metadata/sonnets_expanded_v5_manifest.csv",
    )
    parser.add_argument(
        "--parent-checkpoint-path",
        default=(
            "runs/sonnet_control_historical_v2_xxl_v5_stable_eval_20k_001/"
            "best_validation.pt"
        ),
    )
    parser.add_argument(
        "--parent-tokenizer-path",
        default=(
            "runs/sonnet_control_historical_v2_xxl_v5_stable_eval_20k_001/"
            "tokenizer.json"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--train-steps", type=int, default=12_000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--min-validation-improvement", type=float, default=0.01)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--adamw-foreach",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--learning-rate-schedule",
        choices=["constant", "warmup_cosine"],
        default="constant",
    )
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume-from-checkpoint", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TaskFormatRunConfig(
        dataset=args.dataset,
        manifest_path=args.manifest_path,
        parent_checkpoint_path=args.parent_checkpoint_path,
        parent_tokenizer_path=args.parent_tokenizer_path,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        context_length=args.context_length,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        early_stopping_patience=args.early_stopping_patience,
        min_validation_improvement=args.min_validation_improvement,
        checkpoint_interval=args.checkpoint_interval,
        progress_interval=args.progress_interval,
        learning_rate=args.learning_rate,
        adamw_foreach=args.adamw_foreach,
        learning_rate_schedule=args.learning_rate_schedule,
        warmup_steps=args.warmup_steps,
        min_learning_rate=args.min_learning_rate,
        max_gradient_norm=args.max_gradient_norm,
        seed=args.seed,
        device=args.device,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    result = train_task_format_run(ROOT, args.output_dir, config)
    final_row = result["history"][-1]

    print(f"wrote config: {result['config_path']}")
    print(f"wrote log: {result['log_path']}")
    print(f"wrote tokenizer: {result['tokenizer_path']}")
    print(f"wrote best checkpoint: {result['best_checkpoint_path']}")
    print(f"wrote resume checkpoint: {result['resume_checkpoint_path']}")
    print(f"wrote final checkpoint: {result['checkpoint_path']}")
    print(
        "run completion: "
        f"steps={result['completed_steps']}, reason={result['stop_reason']}"
    )
    print(
        "final losses: "
        f"train={final_row['train_loss']:.4f}, "
        f"validation={final_row['validation_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
