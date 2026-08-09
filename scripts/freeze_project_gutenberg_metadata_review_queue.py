#!/usr/bin/env python3
"""Freeze the unresolved Project Gutenberg metadata-review queue."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.gutenberg_metadata_review_queue import (
    GutenbergMetadataReviewQueueConfig,
    freeze_gutenberg_metadata_review_queue,
)


def main() -> None:
    print(
        "gutenberg-review-queue | start device=cpu inventory_records=1112 "
        "estimated_runtime=under_5s",
        flush=True,
    )
    report = freeze_gutenberg_metadata_review_queue(
        GutenbergMetadataReviewQueueConfig(
            repo_root=ROOT,
            inventory_csv_path=(
                ROOT / "data/metadata/project_gutenberg_italian_inventory_v1.csv"
            ),
            queue_csv_path=(
                ROOT / "data/metadata/project_gutenberg_metadata_review_queue_v1.csv"
            ),
            json_report_path=(
                ROOT / "reports/project_gutenberg_metadata_review_queue_v1.json"
            ),
            markdown_report_path=(
                ROOT / "reports/project_gutenberg_metadata_review_queue_v1.md"
            ),
        )
    )
    print(
        "gutenberg-review-queue | complete "
        f"reviews={report['review_record_count']:,} "
        f"accounted={sum(report['inventory_accounting'].values()):,}",
        flush=True,
    )
    print(
        "gutenberg-review-queue | wrote queue: "
        f"{report['outputs']['queue_csv_path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
