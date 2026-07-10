#!/usr/bin/env python3
"""Write a public Markdown report for one local broader-corpus pretraining run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.pretraining_report import write_pretraining_markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Ignored local pretraining-run directory to summarize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Public Markdown report path to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_pretraining_markdown_report(
        run_dir=args.run_dir,
        output_path=args.output,
    )

    print(f"wrote report: {args.output}")
    print(
        "final losses: "
        f"train={summary['final_train_loss']:.4f}, "
        f"validation={summary['final_validation_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
