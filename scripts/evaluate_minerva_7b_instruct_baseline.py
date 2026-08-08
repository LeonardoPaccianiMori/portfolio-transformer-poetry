#!/usr/bin/env python3
"""Write automatic and human-review artifacts for the Minerva 7B baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.minerva_7b_instruct import (
    write_minerva_7b_instruct_baseline_scaffolds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-dir",
        type=Path,
        default=Path("outputs/generations/minerva_7b_instruct_validation_v1/instruct"),
    )
    parser.add_argument(
        "--automatic-report",
        type=Path,
        default=Path("reports/minerva_7b_instruct_validation_automatic.md"),
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("reports/minerva_7b_instruct_validation_review.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("minerva-7b-eval | scoring eight validation outputs", flush=True)
    rows = write_minerva_7b_instruct_baseline_scaffolds(
        generation_dir=ROOT / args.generation_dir,
        automatic_report_path=ROOT / args.automatic_report,
        review_path=ROOT / args.review,
    )
    print(
        "minerva-7b-eval | complete outputs={outputs} automatic_report={report} "
        "review={review}".format(
            outputs=len(rows),
            report=args.automatic_report,
            review=args.review,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

