#!/usr/bin/env python3
"""Score generated text files with basic automatic metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.metrics import write_generation_metrics_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "generation_metrics.md",
    )
    parser.add_argument("--ngram-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = write_generation_metrics_report(
        generation_dir=args.generation_dir,
        output_path=args.output,
        ngram_size=args.ngram_size,
    )

    print(f"wrote report: {args.output}")
    print(f"scored files: {len(rows)}")


if __name__ == "__main__":
    main()
