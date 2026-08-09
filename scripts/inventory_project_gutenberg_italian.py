#!/usr/bin/env python3
"""Enumerate and classify the complete Italian Project Gutenberg catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.gutenberg_catalog_inventory import (
    GutenbergCatalogInventoryConfig,
    inventory_italian_gutenberg_catalog,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=ROOT / "data/metadata/project_gutenberg_italian_catalog_v1.json",
    )
    parser.add_argument(
        "--inventory-csv",
        type=Path,
        default=ROOT / "data/metadata/project_gutenberg_italian_inventory_v1.csv",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=ROOT / "reports/project_gutenberg_italian_inventory_v1.json",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=ROOT / "reports/project_gutenberg_italian_inventory_v1.md",
    )
    parser.add_argument("--request-delay", type=float, default=0.25)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "gutenberg-inventory | start device=cpu scope=all_italian_records "
        f"request_delay={args.request_delay:.2f}s "
        "estimated_runtime=20s-3m network_dependent",
        flush=True,
    )

    def progress(message: str) -> None:
        print(f"gutenberg-inventory | {message}", flush=True)

    report = inventory_italian_gutenberg_catalog(
        GutenbergCatalogInventoryConfig(
            repo_root=ROOT,
            snapshot_path=args.snapshot_path,
            inventory_csv_path=args.inventory_csv,
            json_report_path=args.json_report,
            markdown_report_path=args.markdown_report,
            bibit_record_manifest_path=(
                ROOT / "data/processed/bibit_resolved_v1/records_manifest.csv"
            ),
            broader_sources_manifest_path=(
                ROOT / "data/metadata/broader_prose_sources_manifest.csv"
            ),
            sonnet_manifest_path=(
                ROOT / "data/metadata/sonnets_expanded_v6_manifest.csv"
            ),
            request_delay_seconds=args.request_delay,
            request_timeout_seconds=args.request_timeout,
        ),
        progress=progress,
    )
    print(
        "gutenberg-inventory | complete "
        f"records={report['record_count']:,} pages={report['page_count']:,} "
        f"plain_text={report['records_with_plain_text_url']:,}",
        flush=True,
    )
    print(f"gutenberg-inventory | wrote inventory: {args.inventory_csv}", flush=True)
    print(f"gutenberg-inventory | wrote report: {args.markdown_report}", flush=True)


if __name__ == "__main__":
    main()
