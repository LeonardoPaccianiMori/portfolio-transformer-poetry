#!/usr/bin/env python3
"""Create a human qualitative review template for generated samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.qualitative import write_qualitative_review_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "qualitative_review.md",
    )
    parser.add_argument(
        "--review-context",
        choices=["sonnet", "pretraining_prose"],
        default="sonnet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reviews = write_qualitative_review_report(
        generation_dir=args.generation_dir,
        output_path=args.output,
        review_context=args.review_context,
    )

    print(f"wrote report: {args.output}")
    print(f"review sections: {len(reviews)}")


if __name__ == "__main__":
    main()
