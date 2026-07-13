#!/usr/bin/env python3
"""Train the causal transformer on the sonnet corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.transformer_run import (
    TransformerTrainingConfig,
    train_transformer_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="expanded_with_petrarch")
    parser.add_argument(
        "--tokenizer-type",
        choices=["character", "bpe"],
        default="character",
    )
    parser.add_argument(
        "--bpe-tokenizer-path",
        default="data/metadata/bpe_tokenizer.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "transformer")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--prompt", default="Amor")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--feed-forward-dim", type=int, default=128)
    parser.add_argument("--max-context-length", type=int, default=128)
    parser.add_argument(
        "--normalization-type",
        choices=["layer_norm", "rms_norm"],
        default="layer_norm",
    )
    parser.add_argument("--normalization-eps", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TransformerTrainingConfig(
        dataset=args.dataset,
        tokenizer_type=args.tokenizer_type,
        bpe_tokenizer_path=args.bpe_tokenizer_path,
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
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        feed_forward_dim=args.feed_forward_dim,
        max_context_length=args.max_context_length,
        normalization_type=args.normalization_type,
        normalization_eps=args.normalization_eps,
    )

    result = train_transformer_run(
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
