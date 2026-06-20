#!/usr/bin/env python3
"""Train the bigram baseline on the sonnet corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.bigram_run import BigramTrainingConfig, train_bigram_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="expanded_with_petrarch")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "bigram")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--prompt", default="Amor")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BigramTrainingConfig(
        dataset=args.dataset,
        batch_size=args.batch_size,
        context_length=args.context_length,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        seed=args.seed,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )

    result = train_bigram_run(
        repo_root=ROOT,
        output_dir=args.output_dir,
        config=config,
    )

    history = result["history"]
    final_row = history[-1]

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
