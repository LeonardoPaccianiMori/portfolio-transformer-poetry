#!/usr/bin/env python3
"""Generate the fixed Minerva 3B base-versus-QLoRA held-out comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.minerva_generation import (
    MINERVA_MAX_NEW_TOKENS,
    generate_fixed_minerva_comparison,
)
from sonnet_evaluation.task_generation import (
    load_task_format_prompts,
    validate_task_format_acceptance_configuration,
    validate_task_format_prompts_against_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-checkpoint",
        type=Path,
        default=Path("runs/minerva_3b_qlora_v5_001/best_adapter.pt"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/generations/minerva_3b_v5_fixed_comparison"),
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=ROOT / "configs" / "task_format_acceptance_prompts.json",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=ROOT / "data" / "metadata" / "sonnets_expanded_v5_manifest.csv",
    )
    parser.add_argument("--dataset", default="expanded_with_petrarch")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MINERVA_MAX_NEW_TOKENS,
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    prompts = load_task_format_prompts(args.prompts)
    validate_task_format_acceptance_configuration(
        prompts=prompts,
        seeds=[1337, 1338],
        temperature=0.8,
        top_k=50,
        continuation_line_target=13,
    )
    validate_task_format_prompts_against_manifest(
        prompts=prompts,
        manifest_path=args.manifest_path,
        repo_root=ROOT,
        dataset=args.dataset,
        split=args.split,
    )
    print(
        "minerva-generation | start "
        f"device={device} variants=2 prompts={len(prompts)} seeds=2 outputs=40",
        flush=True,
    )
    metadata = generate_fixed_minerva_comparison(
        repo_root=ROOT,
        adapter_checkpoint_path=args.adapter_checkpoint,
        output_root=ROOT / args.output_root,
        prompts=prompts,
        prompt_config_path=args.prompts,
        max_new_tokens=args.max_new_tokens,
        device=device,
        cache_dir=ROOT / "data" / "local" / "minerva_qlora" / "huggingface",
        progress=lambda message: print(
            f"minerva-generation | {message}",
            flush=True,
        ),
    )
    print(
        "minerva-generation | complete "
        f"base_outputs={metadata['base_output_count']} "
        f"qlora_outputs={metadata['qlora_output_count']} "
        f"output_root={args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
