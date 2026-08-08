#!/usr/bin/env python3
"""Generate the frozen validation-only Minerva prompting and adapter audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.minerva_sanity_audit import (
    MINERVA_SANITY_CONDITION_IDS,
    MINERVA_SANITY_PROMPT_COUNT,
    MINERVA_SANITY_SEEDS,
    generate_minerva_sanity_audit,
    validate_minerva_sanity_prompts,
)
from sonnet_evaluation.task_generation import (
    load_task_format_prompts,
    validate_task_format_prompts_against_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--best-adapter-checkpoint",
        type=Path,
        default=Path("runs/minerva_3b_qlora_v5_001/best_adapter.pt"),
    )
    parser.add_argument(
        "--final-adapter-checkpoint",
        type=Path,
        default=Path("runs/minerva_3b_qlora_v5_001/final_adapter.pt"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/generations/minerva_3b_validation_sanity_v1"),
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=ROOT / "configs" / "minerva_3b_validation_sanity_prompts.json",
    )
    parser.add_argument(
        "--final-test-prompts",
        type=Path,
        default=ROOT / "configs" / "task_format_acceptance_prompts.json",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=ROOT / "data" / "metadata" / "sonnets_expanded_v5_manifest.csv",
    )
    parser.add_argument("--dataset", default="expanded_with_petrarch")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    prompts = load_task_format_prompts(args.prompts)
    final_test_prompts = load_task_format_prompts(args.final_test_prompts)
    validate_minerva_sanity_prompts(prompts, final_test_prompts)
    validate_task_format_prompts_against_manifest(
        prompts=prompts,
        manifest_path=args.manifest_path,
        repo_root=ROOT,
        dataset=args.dataset,
        split="validation",
    )
    total_outputs = (
        len(MINERVA_SANITY_CONDITION_IDS)
        * MINERVA_SANITY_PROMPT_COUNT
        * len(MINERVA_SANITY_SEEDS)
    )
    print(
        "minerva-sanity | start "
        f"device={device} conditions={len(MINERVA_SANITY_CONDITION_IDS)} "
        f"prompts={len(prompts)} seeds={len(MINERVA_SANITY_SEEDS)} "
        f"outputs={total_outputs}",
        flush=True,
    )
    metadata = generate_minerva_sanity_audit(
        repo_root=ROOT,
        best_checkpoint_path=args.best_adapter_checkpoint,
        final_checkpoint_path=args.final_adapter_checkpoint,
        output_root=ROOT / args.output_root,
        prompts=prompts,
        prompt_config_path=args.prompts,
        device=device,
        cache_dir=ROOT / "data" / "local" / "minerva_qlora" / "huggingface",
        progress=lambda message: print(f"minerva-sanity | {message}", flush=True),
    )
    print(
        "minerva-sanity | complete "
        f"conditions={len(metadata['conditions'])} outputs={total_outputs} "
        f"output_root={args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
