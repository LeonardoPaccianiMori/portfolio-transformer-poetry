#!/usr/bin/env python3
"""Write a public Markdown report for one local sonnet fine-tuning run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.finetuning_report import write_finetuning_markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--selection-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_finetuning_markdown_report(
        run_dir=args.run_dir,
        selection_path=args.selection_path,
        output_path=args.output,
    )
    selection = summary["selection"]
    print(f"wrote report: {args.output}")
    print(f"selected checkpoint step: {selection['selected_checkpoint_step']}")


if __name__ == "__main__":
    main()
