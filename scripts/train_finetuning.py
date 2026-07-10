#!/usr/bin/env python3
"""Fine-tune a broader-pretrained transformer on classical Italian sonnets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.finetuning_run import FineTuningRunConfig, train_finetuning_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="expanded_with_petrarch")
    parser.add_argument(
        "--pretraining-checkpoint-path",
        default="runs/pretraining_larger_200k_001/model.pt",
    )
    parser.add_argument(
        "--pretraining-tokenizer-path",
        default="runs/pretraining_larger_200k_001/tokenizer.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "finetuning_larger_20k_001",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--train-steps", type=int, default=20_000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument(
        "--reset-optimizer-state",
        action="store_true",
        help="Use fresh AdamW moments instead of restoring the parent optimizer state.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--prompt", default="Amor")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FineTuningRunConfig(
        dataset=args.dataset,
        pretraining_checkpoint_path=args.pretraining_checkpoint_path,
        pretraining_tokenizer_path=args.pretraining_tokenizer_path,
        batch_size=args.batch_size,
        context_length=args.context_length,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        checkpoint_interval=args.checkpoint_interval,
        learning_rate=args.learning_rate,
        restore_optimizer_state=not args.reset_optimizer_state,
        seed=args.seed,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    result = train_finetuning_run(
        repo_root=ROOT,
        output_dir=args.output_dir,
        config=config,
    )
    final_row = result["history"][-1]

    print(f"wrote config: {result['config_path']}")
    print(f"wrote log: {result['log_path']}")
    print(f"wrote tokenizer: {result['tokenizer_path']}")
    print(f"wrote sample: {result['sample_path']}")
    print(f"wrote checkpoint: {result['checkpoint_path']}")
    print(
        "final losses: "
        f"train={final_row['train_loss']:.4f}, "
        f"validation={final_row['validation_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
