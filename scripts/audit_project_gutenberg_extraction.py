#!/usr/bin/env python3
"""Audit Gutenberg extraction and sonnet boundaries without building corpus text."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.gutenberg_extraction_audit import (
    GutenbergExtractionAuditConfig,
    audit_gutenberg_extraction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-probe", type=Path, default=ROOT / "data/metadata/project_gutenberg_fulltext_probe_v1.csv")
    parser.add_argument("--pass1b-probe", type=Path, default=ROOT / "data/metadata/project_gutenberg_fulltext_probe_pass_1b_v1.csv")
    parser.add_argument("--final-resolution", type=Path, default=ROOT / "data/metadata/project_gutenberg_metadata_final_resolution_v1.csv")
    parser.add_argument("--prior-cache-dir", type=Path, default=ROOT / "data/local/gutenberg/fulltext_gate_v1")
    parser.add_argument("--pass1b-cache-dir", type=Path, default=ROOT / "data/local/gutenberg/metadata_review_v1")
    parser.add_argument("--bibit-record-manifest", type=Path, default=ROOT / "data/processed/bibit_resolved_v1/records_manifest.csv")
    parser.add_argument("--broader-sources-manifest", type=Path, default=ROOT / "data/metadata/broader_prose_sources_manifest.csv")
    parser.add_argument("--sonnet-manifest", type=Path, default=ROOT / "data/metadata/sonnets_expanded_v6_manifest.csv")
    parser.add_argument("--bibit-sonnet-manifest", type=Path, default=ROOT / "data/processed/bibit_resolved_v1/sonnets_manifest.csv")
    parser.add_argument("--source-csv", type=Path, default=ROOT / "data/metadata/project_gutenberg_extraction_decisions_v1.csv")
    parser.add_argument("--segment-csv", type=Path, default=ROOT / "data/metadata/project_gutenberg_segment_decisions_v1.csv")
    parser.add_argument("--sonnet-csv", type=Path, default=ROOT / "data/metadata/project_gutenberg_sonnet_candidates_v1.csv")
    parser.add_argument("--review-csv", type=Path, default=ROOT / "data/metadata/project_gutenberg_sonnet_review_v1.csv")
    parser.add_argument("--json-report", type=Path, default=ROOT / "reports/project_gutenberg_extraction_audit_v1.json")
    parser.add_argument("--markdown-report", type=Path, default=ROOT / "reports/project_gutenberg_extraction_audit_v1.md")
    parser.add_argument("--progress-interval", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "gutenberg-extraction-audit | start device=cpu records=587 "
        f"progress_interval={args.progress_interval} estimated_runtime=1m-3m_cached",
        flush=True,
    )

    def progress(message: str) -> None:
        print(f"gutenberg-extraction-audit | {message}", flush=True)

    report = audit_gutenberg_extraction(
        GutenbergExtractionAuditConfig(
            repo_root=ROOT,
            prior_probe_csv_path=args.prior_probe,
            pass1b_probe_csv_path=args.pass1b_probe,
            final_resolution_csv_path=args.final_resolution,
            prior_cache_dir=args.prior_cache_dir,
            pass1b_cache_dir=args.pass1b_cache_dir,
            bibit_record_manifest_path=args.bibit_record_manifest,
            broader_sources_manifest_path=args.broader_sources_manifest,
            sonnet_manifest_path=args.sonnet_manifest,
            bibit_sonnet_manifest_path=args.bibit_sonnet_manifest,
            source_csv_path=args.source_csv,
            segment_csv_path=args.segment_csv,
            sonnet_csv_path=args.sonnet_csv,
            review_csv_path=args.review_csv,
            json_report_path=args.json_report,
            markdown_report_path=args.markdown_report,
            progress_interval=args.progress_interval,
        ),
        progress=progress,
    )
    print(
        "gutenberg-extraction-audit | complete "
        f"sources={report['source_count']:,} sonnets={report['sonnet_candidate_count']:,} "
        f"unresolved_reviews={report['unresolved_sonnet_review_count']:,}",
        flush=True,
    )
    print(f"gutenberg-extraction-audit | wrote report: {args.markdown_report}", flush=True)


if __name__ == "__main__":
    main()
