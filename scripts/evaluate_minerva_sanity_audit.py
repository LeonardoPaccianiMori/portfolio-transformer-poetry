#!/usr/bin/env python3
"""Create automatic and blinded-review artifacts for the Minerva sanity audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.minerva_sanity_audit import (
    write_minerva_sanity_audit_scaffolds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/generations/minerva_3b_validation_sanity_v1"),
    )
    parser.add_argument(
        "--automatic-report",
        type=Path,
        default=Path("reports/minerva_3b_validation_sanity_automatic.md"),
    )
    parser.add_argument(
        "--blinded-review",
        type=Path,
        default=Path("outputs/reports/minerva_3b_validation_sanity_blinded_review.md"),
    )
    parser.add_argument(
        "--blind-mapping",
        type=Path,
        default=Path(
            "outputs/generations/minerva_3b_validation_sanity_v1/"
            "blind_mapping.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("minerva-sanity-eval | scoring seven conditions", flush=True)
    rows, mapping = write_minerva_sanity_audit_scaffolds(
        output_root=ROOT / args.output_root,
        automatic_report_path=ROOT / args.automatic_report,
        blinded_review_path=ROOT / args.blinded_review,
        blind_mapping_path=ROOT / args.blind_mapping,
    )
    print(
        "minerva-sanity-eval | complete "
        f"conditions={len(rows)} blinded_outputs={len(mapping)} "
        f"automatic_report={args.automatic_report} "
        f"blinded_review={args.blinded_review}",
        flush=True,
    )


if __name__ == "__main__":
    main()
