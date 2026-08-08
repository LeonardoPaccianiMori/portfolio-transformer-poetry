#!/usr/bin/env python3
"""Generate the three strongest Stage B adapters on frozen validation prompts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.minerva_7b_sonnet_candidates import (
    CANDIDATE_PROMPT_COUNT,
    CANDIDATE_SEEDS,
    generate_minerva_7b_sonnet_candidates,
)
from sonnet_evaluation.minerva_sanity_audit import validate_minerva_sanity_prompts
from sonnet_evaluation.task_generation import (
    load_task_format_prompts,
    validate_task_format_prompts_against_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/minerva_7b_v6_sonnet_fp16_lora_001"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/generations/minerva_7b_v6_candidates_001"),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    prompt_path = ROOT / "configs/minerva_3b_validation_sanity_prompts.json"
    final_prompt_path = ROOT / "configs/task_format_acceptance_prompts.json"
    manifest_path = ROOT / "data/metadata/sonnets_expanded_v6_manifest.csv"
    prompts = load_task_format_prompts(prompt_path)
    final_prompts = load_task_format_prompts(final_prompt_path)
    validate_minerva_sanity_prompts(prompts, final_prompts)
    validate_task_format_prompts_against_manifest(
        prompts=prompts,
        manifest_path=manifest_path,
        repo_root=ROOT,
        dataset="expanded_with_petrarch",
        split="validation",
    )
    print(
        "minerva-candidates | start "
        f"device={args.device} candidates=3 prompts={CANDIDATE_PROMPT_COUNT} "
        f"seeds={len(CANDIDATE_SEEDS)} outputs=24",
        flush=True,
    )
    metadata = generate_minerva_7b_sonnet_candidates(
        repo_root=ROOT,
        run_dir=args.run_dir,
        output_root=args.output_root,
        prompts=prompts,
        prompt_config_path=prompt_path,
        device=torch.device(args.device),
        cache_dir=ROOT / "data/local/minerva_qlora/huggingface",
        progress=lambda message: print(
            f"minerva-candidates | {message}", flush=True
        ),
    )
    print(
        "minerva-candidates | complete "
        f"conditions={len(metadata['conditions'])} "
        f"output_root={args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
