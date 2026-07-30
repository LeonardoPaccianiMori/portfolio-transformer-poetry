#!/usr/bin/env python3
"""Generate the fixed opening-line task-format sonnet acceptance set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.task_generation import (
    TASK_CONTINUATION_LINE_TARGET,
    generate_task_format_for_prompts,
    load_task_format_prompts,
    validate_task_format_acceptance_configuration,
    validate_task_format_prompts_against_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Model-only checkpoint to generate from; defaults to --run-dir/model.pt.",
    )
    parser.add_argument(
        "--model-config-path",
        type=Path,
        default=None,
        help="Run configuration containing the model architecture.",
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
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337, 1338])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--continuation-line-target",
        type=int,
        default=TASK_CONTINUATION_LINE_TARGET,
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    prompts = load_task_format_prompts(args.prompts)
    validate_task_format_acceptance_configuration(
        prompts=prompts,
        seeds=args.seeds,
        temperature=args.temperature,
        top_k=args.top_k,
        continuation_line_target=args.continuation_line_target,
    )
    validate_task_format_prompts_against_manifest(
        prompts=prompts,
        manifest_path=args.manifest_path,
        repo_root=ROOT,
        dataset=args.dataset,
        split=args.split,
    )

    expected_output_count = len(prompts) * len(args.seeds)
    print(
        "task-generation | "
        f"start device={device} prompts={len(prompts)} seeds={len(args.seeds)} "
        f"outputs={expected_output_count}",
        flush=True,
    )
    metadata = generate_task_format_for_prompts(
        run_dir=args.run_dir,
        prompts=prompts,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        seeds=args.seeds,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        continuation_line_target=args.continuation_line_target,
        checkpoint_path=args.checkpoint_path,
        model_config_path=args.model_config_path,
        prompt_config_path=args.prompts,
        progress=lambda message: print(f"task-generation | {message}", flush=True),
    )
    print(
        "task-generation | complete "
        f"output_dir={args.output_dir} generated_files={len(metadata['generated_files'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
