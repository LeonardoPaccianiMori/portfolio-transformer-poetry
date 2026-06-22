#!/usr/bin/env python3
"""Check generated text for near-copying of training poems."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.memorization import (
    DEFAULT_NGRAM_SIZE,
    write_memorization_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "metadata" / "poems_manifest.csv",
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--dataset", default="expanded_with_petrarch")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "memorization_checks.md",
    )
    parser.add_argument("--ngram-size", type=int, default=DEFAULT_NGRAM_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = write_memorization_report(
        generation_dir=args.generation_dir,
        manifest_path=args.manifest,
        repo_root=args.repo_root,
        dataset=args.dataset,
        split=args.split,
        output_path=args.output,
        ngram_size=args.ngram_size,
    )

    print(f"wrote report: {args.output}")
    print(f"scored files: {len(rows)}")


if __name__ == "__main__":
    main()
