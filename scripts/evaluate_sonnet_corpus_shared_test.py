#!/usr/bin/env python3
"""Evaluate selected V4 and V5 sonnet models on their shared held-out poems."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.corpus_scaling_test import (
    EvaluationArm,
    evaluate_shared_sonnet_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v4-run-dir",
        type=Path,
        default=Path(
            "runs/sonnet_control_quality_swiglu_larger_v4_stable_eval_20k_001"
        ),
    )
    parser.add_argument(
        "--v4-selection-path",
        type=Path,
        default=Path(
            "runs/sonnet_control_quality_swiglu_larger_v4_stable_eval_20k_001/"
            "selected_checkpoint.json"
        ),
    )
    parser.add_argument(
        "--v4-manifest-path",
        type=Path,
        default=Path("data/metadata/sonnets_expanded_v4_manifest.csv"),
    )
    parser.add_argument(
        "--v5-run-dir",
        type=Path,
        default=Path(
            "runs/sonnet_control_quality_swiglu_larger_v5_stable_eval_20k_001"
        ),
    )
    parser.add_argument(
        "--v5-selection-path",
        type=Path,
        default=Path(
            "runs/sonnet_control_quality_swiglu_larger_v5_stable_eval_20k_001/"
            "selected_checkpoint.json"
        ),
    )
    parser.add_argument(
        "--v5-manifest-path",
        type=Path,
        default=Path("data/metadata/sonnets_expanded_v5_manifest.csv"),
    )
    parser.add_argument("--dataset", default="expanded_with_petrarch")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--per-poem-output",
        type=Path,
        default=ROOT / "reports" / "sonnet_corpus_scaling_shared_test_per_poem.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "reports" / "sonnet_corpus_scaling_shared_test.md",
    )
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(
        "shared-test | start "
        f"device={device} context_length={args.context_length}",
        flush=True,
    )
    result = evaluate_shared_sonnet_test(
        repo_root=ROOT,
        left_arm=EvaluationArm(
            label="v4",
            run_dir=args.v4_run_dir,
            selection_path=args.v4_selection_path,
            manifest_path=args.v4_manifest_path,
        ),
        right_arm=EvaluationArm(
            label="v5",
            run_dir=args.v5_run_dir,
            selection_path=args.v5_selection_path,
            manifest_path=args.v5_manifest_path,
        ),
        dataset=args.dataset,
        context_length=args.context_length,
        device=device,
        per_poem_output_path=args.per_poem_output,
        report_output_path=args.report_output,
        progress=lambda message: print(f"shared-test | {message}", flush=True),
    )
    print(
        "shared-test | complete "
        f"poems={result['shared_test_poem_count']} "
        f"v4_loss={result['arms']['v4']['loss']:.4f} "
        f"v5_loss={result['arms']['v5']['loss']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
