#!/usr/bin/env python3
"""Evaluate one frozen validation-selected Minerva 7B adapter on V6 final test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.minerva_7b_sonnet_final import (
    FINAL_PROMPT_COUNT,
    FINAL_SEEDS,
    evaluate_minerva_7b_sonnet_final,
)
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
        "--selection-path",
        type=Path,
        default=Path("configs/minerva_7b_v6_selected_adapter.json"),
    )
    parser.add_argument(
        "--candidate-summary-path",
        type=Path,
        default=Path(
            "outputs/generations/minerva_7b_v6_candidates_001/candidate_summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/generations/minerva_7b_v6_final_001"),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    prompt_path = ROOT / "configs/task_format_acceptance_prompts.json"
    manifest_path = ROOT / "data/metadata/sonnets_expanded_v6_manifest.csv"
    prompts = load_task_format_prompts(prompt_path)
    validate_task_format_prompts_against_manifest(
        prompts=prompts,
        manifest_path=manifest_path,
        repo_root=ROOT,
        dataset="expanded_with_petrarch",
        split="test",
    )
    print(
        "minerva-final | start "
        f"device={args.device} prompts={FINAL_PROMPT_COUNT} "
        f"seeds={len(FINAL_SEEDS)} outputs=20",
        flush=True,
    )
    result = evaluate_minerva_7b_sonnet_final(
        repo_root=ROOT,
        run_dir=args.run_dir,
        selection_path=args.selection_path,
        candidate_summary_path=args.candidate_summary_path,
        manifest_path=manifest_path,
        output_dir=args.output_dir,
        prompts=prompts,
        prompt_config_path=prompt_path,
        device=torch.device(args.device),
        cache_dir=ROOT / "data/local/minerva_qlora/huggingface",
        progress=lambda message: print(f"minerva-final | {message}", flush=True),
    )
    print(
        "minerva-final | complete "
        f"epoch={result['selected_epoch']} test_loss={result['final_test_loss']:.4f} "
        f"controlled={result['controlled_sonnet_count']}/20",
        flush=True,
    )


if __name__ == "__main__":
    main()
