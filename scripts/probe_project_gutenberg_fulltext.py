#!/usr/bin/env python3
"""Probe all metadata-eligible Italian Project Gutenberg full texts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.gutenberg_fulltext_probe import (
    GutenbergFullTextProbeConfig,
    run_gutenberg_fulltext_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/local/gutenberg/fulltext_gate_v1",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "data/metadata/project_gutenberg_fulltext_probe_v1.csv",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=ROOT / "reports/project_gutenberg_fulltext_probe_v1.json",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=ROOT / "reports/project_gutenberg_fulltext_probe_v1.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cached_count = len(list(args.cache_dir.glob("pg*.txt"))) if args.cache_dir.is_dir() else 0
    estimated_runtime = (
        "2m-5m_cached" if cached_count >= 400 else "15m-90m_network_and_cpu_dependent"
    )
    print(
        "gutenberg-probe | start device=cpu candidates=416 "
        f"cached_texts={cached_count:,} request_delay={args.request_delay:.1f}s "
        f"cache_reusable=true estimated_runtime={estimated_runtime}",
        flush=True,
    )

    def progress(message: str) -> None:
        print(f"gutenberg-probe | {message}", flush=True)

    report = run_gutenberg_fulltext_probe(
        GutenbergFullTextProbeConfig(
            repo_root=ROOT,
            inventory_csv_path=(
                ROOT / "data/metadata/project_gutenberg_italian_inventory_v1.csv"
            ),
            cache_dir=args.cache_dir,
            output_csv_path=args.output_csv,
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
        "gutenberg-probe | complete "
        f"records={report['candidate_count']:,} "
        f"characters={report['cleaned_character_count']:,} "
        f"cross_duplicates={len(report['cross_corpus_duplicate_pairs']):,}",
        flush=True,
    )
    print(f"gutenberg-probe | wrote manifest: {args.output_csv}", flush=True)
    print(f"gutenberg-probe | wrote report: {args.markdown_report}", flush=True)


if __name__ == "__main__":
    main()
