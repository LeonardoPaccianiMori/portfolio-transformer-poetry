#!/usr/bin/env python3
"""Inventory the complete Liber Liber book catalog without acquiring full text."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.liber_liber_archive_inventory import (
    LiberLiberArchiveInventoryConfig,
    build_liber_liber_archive_inventory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "data/local/liber_liber/archive_inventory_v1/wordpress_pages.json",
    )
    parser.add_argument("--request-delay", type=float, default=0.10)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--per-page", type=int, default=100)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> LiberLiberArchiveInventoryConfig:
    metadata = ROOT / "data/metadata"
    return LiberLiberArchiveInventoryConfig(
        repo_root=ROOT,
        local_cache_path=args.cache,
        inventory_path=metadata / "liber_liber_archive_inventory_v1.csv",
        rights_path=metadata / "liber_liber_source_rights_v1.csv",
        composition_gate_path=metadata / "liber_liber_composition_gate_v1.csv",
        json_report_path=ROOT / "reports/liber_liber_archive_inventory_v1.json",
        markdown_report_path=ROOT / "reports/liber_liber_archive_inventory_v1.md",
        broader_sources_manifest_path=metadata / "broader_prose_sources_manifest.csv",
        prior_probe_report_path=metadata / "broader_prose_liber_liber_probe_report.json",
        bibit_build_report_path=ROOT / "data/processed/bibit_resolved_v1/build_report.json",
        gutenberg_build_report_path=ROOT / "data/processed/project_gutenberg_resolved_v1/build_report.json",
        wikisource_build_report_path=ROOT / "data/processed/italian_wikisource_resolved_v1/build_report.json",
        request_delay_seconds=args.request_delay,
        request_timeout_seconds=args.request_timeout,
        per_page=args.per_page,
    )


def main() -> None:
    args = parse_args()
    print(
        "liber-liber-inventory | start device=cpu scope=all_public_wordpress_pages "
        f"progress_interval=1_page request_delay={args.request_delay:.2f}s "
        "estimated_runtime=20s-5m_network_or_under_10s_cached activation=false",
        flush=True,
    )
    report = build_liber_liber_archive_inventory(
        config_from_args(args),
        progress=lambda message: print(f"liber-liber-inventory | {message}", flush=True),
    )
    print(
        "liber-liber-inventory | complete "
        f"wordpress_pages={report['wordpress_page_count']:,} "
        f"book_works={report['book_work_count']:,} "
        f"eligible_probes={report['eligible_fulltext_probe_count']:,} activated=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
