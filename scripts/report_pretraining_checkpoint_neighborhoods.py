#!/usr/bin/env python3
"""Write automatic diagnostics for pretraining checkpoint neighborhoods."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.checkpoint_neighborhood_report import (
    write_checkpoint_neighborhood_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "generations"
            / "pretraining_neighborhoods"
            / "metadata.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "pretraining_checkpoint_neighborhoods.md",
    )
    parser.add_argument("--ngram-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("neighborhood-report | scoring checkpoint batches", flush=True)
    summaries = write_checkpoint_neighborhood_report(
        metadata_path=args.metadata,
        output_path=args.output,
        ngram_size=args.ngram_size,
    )
    print(f"neighborhood-report | wrote report: {args.output}", flush=True)
    print(
        f"neighborhood-report | summarized checkpoint batches: {len(summaries)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
