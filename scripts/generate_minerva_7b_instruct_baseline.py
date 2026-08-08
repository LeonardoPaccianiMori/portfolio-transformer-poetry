#!/usr/bin/env python3
"""Generate a frozen Minerva 7B Instruct validation baseline."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.minerva_7b_instruct import (
    MINERVA_7B_INSTRUCT_MODEL_ID,
    generate_minerva_7b_instruct_baseline,
)
from sonnet_evaluation.minerva_sanity_audit import validate_minerva_sanity_prompts
from sonnet_evaluation.task_generation import (
    load_task_format_prompts,
    validate_task_format_prompts_against_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
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
    parser.add_argument(
        "--load-mode",
        choices=("nf4", "fp16"),
        default="nf4",
        help="Load the frozen base in 4-bit NF4 or unquantized FP16.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    device = torch.device(args.device)
    output_root = args.output_root or Path(
        "outputs/generations/minerva_7b_instruct_validation_v1"
        if args.load_mode == "nf4"
        else "outputs/generations/minerva_7b_instruct_fp16_validation_v1"
    )
    started_at = time.monotonic()

    def progress(message: str) -> None:
        elapsed = round(time.monotonic() - started_at)
        print(f"minerva-7b-baseline | {message} | elapsed={elapsed}s", flush=True)

    print(
        "minerva-7b-baseline | start model={model} device={device} "
        "load_mode={load_mode} prompts={prompts} outputs={outputs}".format(
            model=MINERVA_7B_INSTRUCT_MODEL_ID,
            device=device,
            load_mode=args.load_mode,
            prompts=len(prompts),
            outputs=len(prompts),
        ),
        flush=True,
    )
    metadata = generate_minerva_7b_instruct_baseline(
        output_root=ROOT / output_root,
        prompts=prompts,
        prompt_config_path=args.prompts,
        device=device,
        cache_dir=ROOT / "data" / "local" / "minerva_qlora" / "huggingface",
        load_mode=args.load_mode,
        progress=progress,
    )
    print(
        "minerva-7b-baseline | complete outputs={outputs} "
        "peak_reserved={memory:.1f}MiB output_root={root}".format(
            outputs=metadata["output_count"],
            memory=metadata["peak_reserved_mib"],
            root=output_root,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
