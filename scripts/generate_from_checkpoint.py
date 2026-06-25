#!/usr/bin/env python3
"""Generate fixed-prompt samples from a saved transformer checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.generation import generate_for_prompts, load_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=ROOT / "configs" / "evaluation_prompts.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--stop-text", default=None)
    parser.add_argument("--target-lines", type=int, default=None)
    parser.add_argument(
        "--suppress-stop-text-until-target-lines",
        action="store_true",
        help=(
            "forbid a one-token stop text until the target line count is reached"
        ),
    )
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return torch.device(device)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    prompts = load_prompts(args.prompts)
    metadata = generate_for_prompts(
        run_dir=args.run_dir,
        prompts=prompts,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        stop_text=args.stop_text,
        target_lines=args.target_lines,
        suppress_stop_text_until_target_lines=(
            args.suppress_stop_text_until_target_lines
        ),
    )

    print(f"wrote output directory: {args.output_dir}")
    print(f"generated files: {len(metadata['generated_files'])}")


if __name__ == "__main__":
    main()
