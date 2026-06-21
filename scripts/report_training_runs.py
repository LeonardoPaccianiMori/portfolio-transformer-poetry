#!/usr/bin/env python3
"""Write a Markdown summary for selected training runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.training_report import write_training_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        type=Path,
        required=True,
        help="Training run directories to include.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "training_runs.md",
    )
    parser.add_argument("--sample-preview-characters", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = write_training_report(
        run_dirs=args.runs,
        output_path=args.output,
        sample_preview_characters=args.sample_preview_characters,
    )

    print(f"wrote report: {args.output}")
    print(f"included runs: {len(rows)}")


if __name__ == "__main__":
    main()
