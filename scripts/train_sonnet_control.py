#!/usr/bin/env python3
"""Train one controlled sonnet arm from random, pretrained, or converted weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.sonnet_control_run import (
    SonnetControlRunConfig,
    train_sonnet_control_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initialization",
        choices=["pretrained", "random", "layernorm_to_rmsnorm"],
        required=True,
    )
    parser.add_argument("--dataset", default="expanded_with_petrarch")
    parser.add_argument(
        "--model-architecture-path",
        default="runs/finetuning_larger_20k_001/selected_checkpoint.json",
    )
    parser.add_argument(
        "--pretraining-tokenizer-path",
        default="runs/pretraining_larger_200k_001/tokenizer.json",
    )
    parser.add_argument(
        "--pretraining-checkpoint-path",
        default="runs/pretraining_larger_200k_001/model.pt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--train-steps", type=int, default=20_000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument(
        "--learning-rate-schedule",
        choices=["constant", "warmup_cosine"],
        default="constant",
    )
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--max-gradient-norm", type=float)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--prompt", default="Amor")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SonnetControlRunConfig(
        initialization=args.initialization,
        dataset=args.dataset,
        model_architecture_path=args.model_architecture_path,
        pretraining_tokenizer_path=args.pretraining_tokenizer_path,
        pretraining_checkpoint_path=args.pretraining_checkpoint_path,
        batch_size=args.batch_size,
        context_length=args.context_length,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        checkpoint_interval=args.checkpoint_interval,
        progress_interval=args.progress_interval,
        learning_rate=args.learning_rate,
        learning_rate_schedule=args.learning_rate_schedule,
        warmup_steps=args.warmup_steps,
        min_learning_rate=args.min_learning_rate,
        max_gradient_norm=args.max_gradient_norm,
        seed=args.seed,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    result = train_sonnet_control_run(ROOT, args.output_dir, config)
    final_row = result["history"][-1]

    print(f"wrote config: {result['config_path']}")
    print(f"wrote log: {result['log_path']}")
    print(f"wrote tokenizer: {result['tokenizer_path']}")
    print(f"wrote sample: {result['sample_path']}")
    print(f"wrote best checkpoint: {result['best_checkpoint_path']}")
    print(f"wrote final checkpoint: {result['checkpoint_path']}")
    print(
        "final losses: "
        f"train={final_row['train_loss']:.4f}, "
        f"validation={final_row['validation_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
