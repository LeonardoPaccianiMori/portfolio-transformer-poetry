#!/usr/bin/env python3
"""Audit source and author concentration in the local pretraining corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.pretraining_balance import (
    PretrainingBalanceConfig,
    audit_pretraining_corpus_balance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=ROOT / "data/metadata/broader_prose_sources_manifest.csv",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data/local/pretraining/expanded_italian_1200_1800_v1/processed",
    )
    parser.add_argument(
        "--json-report-path",
        type=Path,
        default=ROOT / "data/local/pretraining/expanded_italian_1200_1800_v1/balance_report.json",
    )
    parser.add_argument(
        "--markdown-report-path",
        type=Path,
        default=ROOT / "data/local/pretraining/expanded_italian_1200_1800_v1/balance_report.md",
    )
    parser.add_argument("--max-source-share", type=float, default=0.15)
    parser.add_argument("--max-author-share", type=float, default=0.20)
    parser.add_argument("--quiet", action="store_true", help="Hide progress output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_pretraining_corpus_balance(
        PretrainingBalanceConfig(
            manifest_path=args.manifest_path,
            processed_dir=args.processed_dir,
            json_report_path=args.json_report_path,
            markdown_report_path=args.markdown_report_path,
            max_source_share=args.max_source_share,
            max_author_share=args.max_author_share,
        ),
        progress=None if args.quiet else lambda message: print(f"balance | {message}", flush=True),
    )
    print(
        "balance | complete "
        f"sources={report.selected_source_count} "
        f"source_character_violations={len(report.source_character_cap_violations)} "
        f"author_character_violations={len(report.author_character_cap_violations)}",
        flush=True,
    )
    print(f"balance | wrote JSON report: {args.json_report_path}", flush=True)
    print(f"balance | wrote Markdown report: {args.markdown_report_path}", flush=True)


if __name__ == "__main__":
    main()
