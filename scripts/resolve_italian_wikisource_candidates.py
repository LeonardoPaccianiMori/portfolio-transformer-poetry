#!/usr/bin/env python3
"""Resolve checkpoint-4A Wikisource candidates against pinned scan metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.wikisource_candidate_resolution import (
    DUMP_BASE_URL,
    DUMP_DATE,
    WikisourceCandidateResolutionConfig,
    build_wikisource_candidate_resolution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/local/wikisource/archive_inventory_v1",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=ROOT / "data/metadata/italian_wikisource_archive_inventory_v1.csv",
    )
    parser.add_argument(
        "--hierarchy",
        type=Path,
        default=ROOT / "data/metadata/italian_wikisource_page_hierarchy_v1.csv",
    )
    parser.add_argument(
        "--resolution",
        type=Path,
        default=ROOT / "data/metadata/italian_wikisource_candidate_resolution_v1.csv",
    )
    parser.add_argument(
        "--scan-links",
        type=Path,
        default=ROOT / "data/metadata/italian_wikisource_source_scan_links_v1.csv",
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=ROOT / "data/metadata/italian_wikisource_candidate_review_v1.csv",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=ROOT / "reports/italian_wikisource_candidate_resolution_v1.json",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=ROOT / "reports/italian_wikisource_candidate_resolution_v1.md",
    )
    parser.add_argument("--progress-interval", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "wikisource-candidate-resolution | start device=cpu "
        f"dump={DUMP_DATE} metadata_only=true candidates=6863 "
        f"progress_interval={args.progress_interval} estimated_runtime=1m-4m",
        flush=True,
    )

    def progress(message: str) -> None:
        print(f"wikisource-candidate-resolution | {message}", flush=True)

    report = build_wikisource_candidate_resolution(
        WikisourceCandidateResolutionConfig(
            repo_root=ROOT,
            cache_dir=args.cache_dir,
            inventory_path=args.inventory,
            hierarchy_path=args.hierarchy,
            resolution_path=args.resolution,
            scan_links_path=args.scan_links,
            review_path=args.review,
            json_report_path=args.json_report,
            markdown_report_path=args.markdown_report,
            dump_date=DUMP_DATE,
            dump_base_url=DUMP_BASE_URL,
            progress_interval=args.progress_interval,
        ),
        progress=progress,
    )
    print(
        "wikisource-candidate-resolution | complete "
        f"candidates={report['candidate_count']:,} "
        f"scan_linked={report['direct_scan_linked_candidate_count']:,} "
        f"eligible_audit_queue={report['eligible_page_level_audit_count']:,} "
        f"review_units={report['review_unit_count']:,}",
        flush=True,
    )
    print(f"wikisource-candidate-resolution | wrote: {args.resolution}", flush=True)
    print(f"wikisource-candidate-resolution | wrote: {args.json_report}", flush=True)


if __name__ == "__main__":
    main()
