#!/usr/bin/env python3
"""Run the stratified Project Gutenberg full-text value gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.gutenberg_fulltext_gate import (
    GutenbergFullTextGateConfig,
    run_gutenberg_fulltext_gate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-csv",
        type=Path,
        default=ROOT / "data/metadata/project_gutenberg_italian_inventory_v1.csv",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/local/gutenberg/fulltext_gate_v1",
    )
    parser.add_argument(
        "--sample-csv",
        type=Path,
        default=ROOT / "data/metadata/project_gutenberg_fulltext_gate_sample_v1.csv",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=ROOT / "reports/project_gutenberg_fulltext_gate_v1.json",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=ROOT / "reports/project_gutenberg_fulltext_gate_v1.md",
    )
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--sample-seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "gutenberg-gate | start device=cpu scope=stratified_fulltext_sample "
        f"request_delay={args.request_delay:.1f}s sample_seed={args.sample_seed} "
        "estimated_runtime=1m-8m network_dependent cache_reusable=true",
        flush=True,
    )

    def progress(message: str) -> None:
        print(f"gutenberg-gate | {message}", flush=True)

    report = run_gutenberg_fulltext_gate(
        GutenbergFullTextGateConfig(
            repo_root=ROOT,
            inventory_csv_path=args.inventory_csv,
            cache_dir=args.cache_dir,
            sample_csv_path=args.sample_csv,
            json_report_path=args.json_report,
            markdown_report_path=args.markdown_report,
            bibit_record_manifest_path=(
                ROOT / "data/processed/bibit_resolved_v1/records_manifest.csv"
            ),
            request_delay_seconds=args.request_delay,
            request_timeout_seconds=args.request_timeout,
            sample_seed=args.sample_seed,
        ),
        progress=progress,
    )
    print(
        "gutenberg-gate | complete "
        f"candidates={report['eligible_probe_candidate_count']:,} "
        f"samples={report['sample_count']:,} "
        f"projected_characters={report['projected_total_cleaned_characters']:,}",
        flush=True,
    )
    print(f"gutenberg-gate | wrote sample: {args.sample_csv}", flush=True)
    print(f"gutenberg-gate | wrote report: {args.markdown_report}", flush=True)


if __name__ == "__main__":
    main()
