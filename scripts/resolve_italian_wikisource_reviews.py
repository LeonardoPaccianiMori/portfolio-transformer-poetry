#!/usr/bin/env python3
"""Resolve checkpoint-4C Wikisource review, rights, and canonical decisions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.wikisource_page_extraction import DUMP_FILENAME
from sonnet_corpus.wikisource_review_resolution import (
    WikisourceReviewResolutionConfig,
    run_wikisource_review_resolution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump", type=Path,
        default=ROOT / "data/local/wikisource/archive_inventory_v1" / DUMP_FILENAME,
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=ROOT / "data/local/wikisource/review_resolution_v1",
    )
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> WikisourceReviewResolutionConfig:
    metadata = ROOT / "data/metadata"
    gutenberg = ROOT / "data/processed/project_gutenberg_resolved_v1"
    return WikisourceReviewResolutionConfig(
        repo_root=ROOT,
        dump_path=args.dump,
        extraction_path=metadata / "italian_wikisource_page_extraction_v1.csv",
        boundaries_path=metadata / "italian_wikisource_page_boundaries_v1.csv",
        review_path=metadata / "italian_wikisource_extraction_review_v1.csv",
        inventory_path=metadata / "italian_wikisource_archive_inventory_v1.csv",
        candidate_resolution_path=metadata / "italian_wikisource_candidate_resolution_v1.csv",
        scan_links_path=metadata / "italian_wikisource_source_scan_links_v1.csv",
        siteinfo_rights_path=ROOT / "data/local/wikisource/archive_inventory_v1/siteinfo_rights_v1.json",
        local_cache_dir=args.cache_dir,
        bibit_record_manifest_path=ROOT / "data/processed/bibit_resolved_v1/records_manifest.csv",
        broader_sources_manifest_path=metadata / "broader_prose_sources_manifest.csv",
        gutenberg_previous_probe_path=metadata / "project_gutenberg_fulltext_probe_v1.csv",
        gutenberg_previous_cache_dir=ROOT / "data/local/gutenberg/fulltext_gate_v1",
        gutenberg_pass_1b_probe_path=metadata / "project_gutenberg_fulltext_probe_pass_1b_v1.csv",
        gutenberg_pass_1b_cache_dir=ROOT / "data/local/gutenberg/metadata_review_v1",
        gutenberg_resolved_record_manifest_path=gutenberg / "records_manifest.csv",
        protected_sonnet_manifest_path=metadata / "sonnets_expanded_v6_manifest.csv",
        bibit_sonnet_manifest_path=ROOT / "data/processed/bibit_resolved_v1/sonnets_manifest.csv",
        gutenberg_resolved_sonnet_manifest_path=gutenberg / "sonnets_manifest.csv",
        root_decisions_path=metadata / "italian_wikisource_root_decisions_v1.csv",
        segment_decisions_path=metadata / "italian_wikisource_segment_decisions_v1.csv",
        sonnet_decisions_path=metadata / "italian_wikisource_sonnet_decisions_v1.csv",
        scan_rights_path=metadata / "italian_wikisource_source_rights_v1.csv",
        review_resolution_path=metadata / "italian_wikisource_review_resolution_v1.csv",
        json_report_path=ROOT / "reports/italian_wikisource_review_resolution_v1.json",
        markdown_report_path=ROOT / "reports/italian_wikisource_review_resolution_v1.md",
        progress_interval=args.progress_interval,
        request_timeout=args.request_timeout,
    )


def main() -> None:
    args = parse_args()
    print(
        "wikisource-review-resolution | start device=cpu roots=4,641 reviews=2,095 "
        f"progress_interval={args.progress_interval} activation=false "
        "estimated_runtime=5m-30m_first_run_or_1m-10m_cached",
        flush=True,
    )
    report = run_wikisource_review_resolution(
        config_from_args(args),
        progress=lambda message: print(f"wikisource-review-resolution | {message}", flush=True),
    )
    print(
        "wikisource-review-resolution | complete "
        f"eligible_roots={report['eligible_root_count']:,} "
        f"eligible_sonnets={report['eligible_sonnet_count']:,} activated=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
