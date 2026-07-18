#!/usr/bin/env python3
"""Generate fixed prompts from selected pretraining checkpoints and neighbors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.checkpoint_neighborhood import (
    generate_checkpoint_neighborhoods,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "configs" / "pretraining_checkpoint_neighborhoods.json",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=ROOT / "configs" / "evaluation_prompts.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "generations" / "pretraining_neighborhoods",
    )
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def main() -> None:
    args = parse_args()
    metadata = generate_checkpoint_neighborhoods(
        repo_root=ROOT,
        plan_path=args.plan,
        prompts_path=args.prompts,
        output_root=args.output_root,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        device=resolve_device(args.device),
        temperature=args.temperature,
        top_k=args.top_k,
        progress=lambda message: print(f"neighborhood | {message}", flush=True),
    )
    print(f"wrote output root: {args.output_root}", flush=True)
    print(f"generated runs: {len(metadata['runs'])}", flush=True)


if __name__ == "__main__":
    main()
