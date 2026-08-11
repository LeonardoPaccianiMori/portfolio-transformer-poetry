#!/usr/bin/env python3
"""Build the metadata-first Italian Wikisource archive inventory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.wikisource_archive_inventory import (
    DUMP_BASE_URL,
    DUMP_DATE,
    WikisourceArchiveInventoryConfig,
    build_wikisource_archive_inventory,
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
        "--page-hierarchy",
        type=Path,
        default=ROOT / "data/metadata/italian_wikisource_page_hierarchy_v1.csv",
    )
    parser.add_argument(
        "--composition-gate",
        type=Path,
        default=ROOT / "data/metadata/italian_wikisource_composition_gate_v1.csv",
    )
    parser.add_argument(
        "--inspection-sample",
        type=Path,
        default=ROOT / "data/metadata/italian_wikisource_inspection_sample_v1.csv",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=ROOT / "reports/italian_wikisource_archive_inventory_v1.json",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=ROOT / "reports/italian_wikisource_archive_inventory_v1.md",
    )
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--api-retries", type=int, default=6)
    parser.add_argument("--progress-interval", type=int, default=25_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "wikisource-archive-inventory | start device=cpu "
        f"dump={DUMP_DATE} metadata_only=true sample_size={args.sample_size} "
        f"progress_interval={args.progress_interval} estimated_runtime=2m-8m",
        flush=True,
    )

    def progress(message: str) -> None:
        print(f"wikisource-archive-inventory | {message}", flush=True)

    report = build_wikisource_archive_inventory(
        WikisourceArchiveInventoryConfig(
            repo_root=ROOT,
            cache_dir=args.cache_dir,
            inventory_path=args.inventory,
            page_hierarchy_path=args.page_hierarchy,
            composition_gate_path=args.composition_gate,
            inspection_sample_path=args.inspection_sample,
            json_report_path=args.json_report,
            markdown_report_path=args.markdown_report,
            broader_manifest_path=(
                ROOT / "data/metadata/broader_prose_sources_manifest.csv"
            ),
            poems_manifest_path=ROOT / "data/metadata/poems_manifest.csv",
            snapshot_dir=ROOT / "data/metadata/wikisource_snapshots",
            dump_date=DUMP_DATE,
            dump_base_url=DUMP_BASE_URL,
            sample_size=args.sample_size,
            request_delay=args.request_delay,
            api_retries=args.api_retries,
            progress_interval=args.progress_interval,
        ),
        progress=progress,
    )
    inspection = report["bounded_inspection"]
    print(
        "wikisource-archive-inventory | complete "
        f"pages={report['main_namespace_page_count']:,} "
        f"work_roots={report['work_root_count']:,} "
        f"candidates={report['candidate_work_root_count']:,} "
        f"sample_passes={inspection['primary_text_signal_pass_count']:,}/"
        f"{inspection['sample_size']:,}",
        flush=True,
    )
    print(f"wikisource-archive-inventory | wrote: {args.inventory}", flush=True)
    print(f"wikisource-archive-inventory | wrote: {args.json_report}", flush=True)


if __name__ == "__main__":
    main()
