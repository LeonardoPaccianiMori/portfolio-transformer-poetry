#!/usr/bin/env python3
"""Acquire evidence and triage the frozen Gutenberg metadata-review queue."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.gutenberg_metadata_review import (
    FROZEN_QUEUE_COUNT,
    GutenbergMetadataReviewConfig,
    run_gutenberg_metadata_review,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--fetch-attempts", type=int, default=3)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/local/gutenberg/metadata_review_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cached_count = (
        len(list(args.cache_dir.glob("pg*.txt"))) if args.cache_dir.is_dir() else 0
    )
    estimate = (
        "under_2m_cached"
        if cached_count >= 650
        else "20m-90m_network_dependent"
    )
    print(
        "gutenberg-metadata-review | start device=cpu "
        f"records={FROZEN_QUEUE_COUNT:,} cached={cached_count:,} "
        f"request_delay={args.request_delay:.1f}s retries={args.fetch_attempts} "
        f"estimated_runtime={estimate}",
        flush=True,
    )

    def progress(message: str) -> None:
        print(f"gutenberg-metadata-review | {message}", flush=True)

    report = run_gutenberg_metadata_review(
        GutenbergMetadataReviewConfig(
            repo_root=ROOT,
            queue_csv_path=(
                ROOT / "data/metadata/project_gutenberg_metadata_review_queue_v1.csv"
            ),
            cache_dir=args.cache_dir,
            output_csv_path=(
                ROOT / "data/metadata/project_gutenberg_metadata_resolution_v1.csv"
            ),
            manual_review_csv_path=(
                ROOT
                / "data/metadata/project_gutenberg_metadata_manual_review_v1.csv"
            ),
            json_report_path=(
                ROOT / "reports/project_gutenberg_metadata_resolution_v1.json"
            ),
            markdown_report_path=(
                ROOT / "reports/project_gutenberg_metadata_resolution_v1.md"
            ),
            request_delay_seconds=args.request_delay,
            request_timeout_seconds=args.request_timeout,
            fetch_attempts=args.fetch_attempts,
        ),
        progress=progress,
    )
    print(
        "gutenberg-metadata-review | complete "
        f"records={report['record_count']:,} "
        f"automatic={report['automatic_resolved_count']:,} "
        f"manual={report['manual_review_count']:,}",
        flush=True,
    )
    print(
        "gutenberg-metadata-review | wrote report: "
        f"{report['outputs']['markdown_report_path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
