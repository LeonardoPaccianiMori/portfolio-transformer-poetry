#!/usr/bin/env python3
"""Train the plan-only LoRA from corpus traces and augmented plans."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.plan_following_sft import (
    split_examples,
    train_plan_following_sft,
)
from sonnet_training.reasoning_trace_sft import build_plan_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plans",
        type=Path,
        default=ROOT / "artifacts/local/plan_generator/data/plans_v1.jsonl",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/plan_generator_sft.json",
    )
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=Path("/workspace/merged_sft"),
    )
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/local/plan_generator/training",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    config = json.loads(args.config.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_dir or args.base_model_dir), local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    cards = []
    with args.plans.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cards.append(json.loads(line))
    if args.limit is not None:
        cards = cards[: args.limit]
    examples, skipped = build_plan_examples(
        cards,
        tokenizer,
        max_sequence_tokens=int(config["data"]["max_sequence_tokens"]),
    )
    train, validation = split_examples(
        examples,
        validation_fraction=float(config["data"]["validation_fraction"]),
        seed=int(config["training"]["seed"]),
    )
    print(
        "plan-generator-train | start "
        f"examples={len(train)} validation={len(validation)} skipped={skipped} "
        f"batch={config['training']['batch_size']} "
        f"accumulation={config['training']['gradient_accumulation_steps']}",
        flush=True,
    )
    report = train_plan_following_sft(
        base_model_dir=args.base_model_dir,
        tokenizer_dir=args.tokenizer_dir,
        examples=train,
        validation_examples=validation,
        output_dir=args.output_dir,
        config=config,
        progress=lambda message: print(f"plan-generator-train | {message}", flush=True),
    )
    print(
        "plan-generator-train | complete "
        f"steps={report['steps']}/{report['planned_steps']} "
        f"cost_usd={report['estimated_cost_usd']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
