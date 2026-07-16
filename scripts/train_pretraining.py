#!/usr/bin/env python3
"""Run a broader Italian transformer pretraining sanity job."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.pretraining_run import PretrainingRunConfig
from sonnet_training.pretraining_run import train_pretraining_run


CORPUS_DIR = "data/local/pretraining/expanded_italian_1200_1800_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-tokens-path",
        default=f"{CORPUS_DIR}/encoded/bpe_8000_train.pt",
    )
    parser.add_argument(
        "--validation-tokens-path",
        default=f"{CORPUS_DIR}/encoded/bpe_8000_validation.pt",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=f"{CORPUS_DIR}/tokenizers/bpe_8000.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "pretraining_sanity_001",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-batches", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--learning-rate-schedule",
        choices=["constant", "warmup_cosine"],
        default="constant",
    )
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--prompt", default="Nel ")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--feed-forward-dim", type=int, default=1024)
    parser.add_argument("--max-context-length", type=int, default=512)
    parser.add_argument(
        "--normalization-type",
        choices=["layer_norm", "rms_norm"],
        default="layer_norm",
    )
    parser.add_argument("--normalization-eps", type=float, default=1e-5)
    parser.add_argument(
        "--position-encoding-type",
        choices=["learned_absolute", "rope"],
        default="learned_absolute",
    )
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument(
        "--feed-forward-type",
        choices=["relu", "swiglu"],
        default="relu",
    )
    parser.add_argument("--tie-token-embeddings", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--resume-from-checkpoint", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PretrainingRunConfig(
        train_tokens_path=args.train_tokens_path,
        validation_tokens_path=args.validation_tokens_path,
        tokenizer_path=args.tokenizer_path,
        batch_size=args.batch_size,
        context_length=args.context_length,
        train_steps=args.train_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        learning_rate_schedule=args.learning_rate_schedule,
        warmup_steps=args.warmup_steps,
        min_learning_rate=args.min_learning_rate,
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
        position_encoding_type=args.position_encoding_type,
        rope_theta=args.rope_theta,
        feed_forward_type=args.feed_forward_type,
        tie_token_embeddings=args.tie_token_embeddings,
        checkpoint_interval=args.checkpoint_interval,
        progress_interval=args.progress_interval,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    result = train_pretraining_run(
        repo_root=ROOT,
        output_dir=args.output_dir,
        config=config,
    )
    final_row = result["history"][-1]

    print(f"wrote config: {result['config_path']}")
    print(f"wrote log: {result['log_path']}")
    print(f"wrote tokenizer: {result['tokenizer_path']}")
    print(f"wrote sample: {result['sample_path']}")
    print(f"wrote best checkpoint: {result['best_checkpoint_path']}")
    print(f"wrote checkpoint: {result['checkpoint_path']}")
    print(
        "final losses: "
        f"train={final_row['train_loss']:.4f}, "
        f"validation={final_row['validation_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
