#!/usr/bin/env python3
"""Build blinded and automatic reports for Minerva 7B recovery outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.minerva_7b_recovery_report import (
    build_recovery_review_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/generations/minerva_7b_quality_recovery_v1"),
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=Path(
            "outputs/generations/minerva_7b_quality_recovery_v1/"
            "blind_mapping.json"
        ),
    )
    parser.add_argument(
        "--review-path",
        type=Path,
        default=Path("outputs/reports/minerva_7b_quality_recovery_blinded_review.md"),
    )
    parser.add_argument(
        "--automatic-report-path",
        type=Path,
        default=Path("reports/minerva_7b_quality_recovery_automatic.md"),
    )
    parser.add_argument(
        "--result-report-path",
        type=Path,
        default=Path("reports/minerva_7b_quality_recovery_result.md"),
    )
    args = parser.parse_args()
    print(
        "recovery-report | start outputs=84 estimated_runtime=under_10s",
        flush=True,
    )
    result = build_recovery_review_artifacts(
        output_root=_resolve(args.output_dir),
        mapping_path=_resolve(args.mapping_path),
        review_path=_resolve(args.review_path),
        automatic_report_path=_resolve(args.automatic_report_path),
        result_report_path=_resolve(args.result_report_path),
    )
    print(
        "recovery-report | complete "
        f"outputs={result['output_count']} "
        f"review={'complete' if result['review_complete'] else 'pending'}",
        flush=True,
    )
    if not result["review_complete"]:
        print(
            "recovery-report | next fill every TODO in "
            f"{_resolve(args.review_path)} and rerun this command",
            flush=True,
        )
    else:
        print(
            f"recovery-report | wrote result: {_resolve(args.result_report_path)}",
            flush=True,
        )


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
