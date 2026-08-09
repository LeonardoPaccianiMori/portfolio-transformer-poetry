#!/usr/bin/env python3
"""Resolve all Biblioteca Italiana record and poem review queues."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.bibit_review_resolution import (
    BibItReviewResolutionConfig,
    resolve_bibit_review_queues,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record-audit-csv",
        type=Path,
        default=ROOT / "data/metadata/bibit_tei_audit_records.csv",
    )
    parser.add_argument(
        "--sonnet-audit-csv",
        type=Path,
        default=ROOT / "data/metadata/bibit_sonnet_candidates_audit.csv",
    )
    parser.add_argument(
        "--tei-cache-dir",
        type=Path,
        default=ROOT / "data/local/bibit/tei",
    )
    parser.add_argument(
        "--record-decisions",
        type=Path,
        default=ROOT / "data/metadata/bibit_record_activation_decisions.csv",
    )
    parser.add_argument(
        "--sonnet-decisions",
        type=Path,
        default=ROOT / "data/metadata/bibit_sonnet_activation_decisions.csv",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=ROOT / "reports/bibit_review_resolution.json",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=ROOT / "reports/bibit_review_resolution.md",
    )
    parser.add_argument("--progress-interval", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "bibit-resolution | start device=cpu "
        f"progress_interval={args.progress_interval} estimated_runtime=1m-5m_cached",
        flush=True,
    )

    def progress(message: str) -> None:
        print(f"bibit-resolution | {message}", flush=True)

    report = resolve_bibit_review_queues(
        BibItReviewResolutionConfig(
            repo_root=ROOT,
            record_audit_csv_path=args.record_audit_csv,
            sonnet_audit_csv_path=args.sonnet_audit_csv,
            tei_cache_dir=args.tei_cache_dir,
            record_decision_csv_path=args.record_decisions,
            sonnet_decision_csv_path=args.sonnet_decisions,
            json_report_path=args.json_report,
            markdown_report_path=args.markdown_report,
            progress_interval=args.progress_interval,
        ),
        progress=progress,
    )
    print(
        "bibit-resolution | complete "
        f"records={report['record_count']:,} "
        f"sonnet_candidates={report['sonnet_candidate_count']:,} "
        f"unresolved={report['unresolved_record_count'] + report['unresolved_sonnet_count']}",
        flush=True,
    )
    print(f"bibit-resolution | wrote report: {args.markdown_report}", flush=True)
    print(f"bibit-resolution | wrote record decisions: {args.record_decisions}", flush=True)
    print(f"bibit-resolution | wrote poem decisions: {args.sonnet_decisions}", flush=True)


if __name__ == "__main__":
    main()
