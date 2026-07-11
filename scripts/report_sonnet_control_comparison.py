#!/usr/bin/env python3
"""Write a public report comparing two controlled sonnet-training runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.control_comparison_report import write_control_comparison_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-run-dir", type=Path, required=True)
    parser.add_argument("--pretrained-generation-dir", type=Path, required=True)
    parser.add_argument("--random-run-dir", type=Path, required=True)
    parser.add_argument("--random-generation-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "metadata" / "poems_manifest.csv",
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-name", default="Initialization")
    parser.add_argument("--left-label", default="pretrained")
    parser.add_argument("--right-label", default="random")
    parser.add_argument(
        "--intended-difference",
        default="broader-pretrained weights versus random weights",
    )
    parser.add_argument(
        "--allowed-difference-field",
        action="append",
        default=[],
        help="Config field allowed to differ between arms; may be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = write_control_comparison_report(
        pretrained_run_dir=args.pretrained_run_dir,
        pretrained_generation_dir=args.pretrained_generation_dir,
        random_run_dir=args.random_run_dir,
        random_generation_dir=args.random_generation_dir,
        manifest_path=args.manifest,
        repo_root=args.repo_root,
        output_path=args.output,
        experiment_name=args.experiment_name,
        left_label=args.left_label,
        right_label=args.right_label,
        intended_difference=args.intended_difference,
        allowed_difference_fields=set(args.allowed_difference_field),
    )
    print(f"wrote report: {args.output}")
    print(
        "best validation losses: "
        f"pretrained={summaries['pretrained']['best_row']['validation_loss']:.4f}, "
        f"random={summaries['random']['best_row']['validation_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
