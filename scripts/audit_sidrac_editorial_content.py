#!/usr/bin/env python3
"""Audit editorial material in the local Project Gutenberg Sidrac source."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.sidrac_editorial_audit import audit_sidrac_editorial_content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-path",
        type=Path,
        default=ROOT
        / "data/local/pretraining/expanded_italian_1200_1800_v1/processed/sources/pg_sidrac_44549.txt",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=ROOT
        / "data/local/pretraining/expanded_italian_1200_1800_v1/audits/pg_sidrac_44549_editorial_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"sidrac-audit | reading source: {args.source_path}", flush=True)
    report = audit_sidrac_editorial_content(
        source_path=args.source_path,
        report_path=args.report_path,
    )
    print(
        "sidrac-audit | complete "
        f"candidate_lines={report.primary_text_start_line}-{report.primary_text_end_line} "
        f"candidate_characters={report.candidate_primary_character_count} "
        f"inline_markers={report.candidate_inline_note_marker_count} "
        f"note_lines={report.candidate_note_line_count}",
        flush=True,
    )
    print(f"sidrac-audit | wrote report: {args.report_path}", flush=True)


if __name__ == "__main__":
    main()
