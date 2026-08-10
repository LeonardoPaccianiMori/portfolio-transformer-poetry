#!/usr/bin/env python3
"""Resolve frozen Gutenberg holds with SBN/ICCU and Wikidata evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.gutenberg_authoritative_review import (
    FROZEN_MANUAL_COUNT,
    GutenbergAuthoritativeReviewConfig,
    run_gutenberg_authoritative_review,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-delay", type=float, default=0.25)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--fetch-attempts", type=int, default=3)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/local/gutenberg/authoritative_review_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cached_count = (
        len(list(args.cache_dir.rglob("*.json"))) if args.cache_dir.is_dir() else 0
    )
    estimate = "under_3m_cached" if cached_count >= 700 else "30m-120m_network_dependent"
    print(
        "gutenberg-authoritative | start device=cpu "
        f"records={FROZEN_MANUAL_COUNT:,} cached_responses={cached_count:,} "
        f"request_delay={args.request_delay:.2f}s retries={args.fetch_attempts} "
        f"estimated_runtime={estimate}",
        flush=True,
    )

    def progress(message: str) -> None:
        print(f"gutenberg-authoritative | {message}", flush=True)

    report = run_gutenberg_authoritative_review(
        GutenbergAuthoritativeReviewConfig(
            repo_root=ROOT,
            pass_1a_csv_path=(
                ROOT / "data/metadata/project_gutenberg_metadata_resolution_v1.csv"
            ),
            manual_csv_path=(
                ROOT / "data/metadata/project_gutenberg_metadata_manual_review_v1.csv"
            ),
            cache_dir=args.cache_dir,
            final_csv_path=(
                ROOT
                / "data/metadata/project_gutenberg_metadata_final_resolution_v1.csv"
            ),
            exclusion_csv_path=(
                ROOT
                / "data/metadata/project_gutenberg_metadata_final_exclusions_v1.csv"
            ),
            json_report_path=(
                ROOT
                / "reports/project_gutenberg_authoritative_resolution_v1.json"
            ),
            markdown_report_path=(
                ROOT
                / "reports/project_gutenberg_authoritative_resolution_v1.md"
            ),
            request_delay_seconds=args.request_delay,
            request_timeout_seconds=args.request_timeout,
            fetch_attempts=args.fetch_attempts,
        ),
        progress=progress,
    )
    print(
        "gutenberg-authoritative | complete "
        f"pass_1b={report['pass_1b_record_count']:,} "
        f"final={report['final_record_count']:,} "
        f"nonstandard_or_excluded={report['explicit_unresolved_or_exclusion_count']:,}",
        flush=True,
    )
    print(
        "gutenberg-authoritative | wrote report: "
        f"{report['outputs']['markdown_report_path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
