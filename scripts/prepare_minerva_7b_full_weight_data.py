#!/usr/bin/env python3
"""Stream and tokenize the complete mixed corpus for Minerva full-weight work."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_full_weight_data import (
    Minerva7BFullWeightDataConfig,
    prepare_minerva_7b_full_weight_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-documents-per-split-run",
        type=int,
        default=None,
        help="Testing/resume control; omit to finish every split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        Minerva7BFullWeightDataConfig(),
        max_documents_per_split_run=args.max_documents_per_split_run,
    )
    started_at = time.monotonic()

    def progress(message: str) -> None:
        print(
            f"minerva-full-data | {message} | "
            f"elapsed={_format_duration(time.monotonic() - started_at)}",
            flush=True,
        )

    print(
        "minerva-full-data | start splits=4 context=512 "
        "shard_target_tokens=8,388,608 estimated_runtime=20m-3h "
        "resumable=true",
        flush=True,
    )
    report = prepare_minerva_7b_full_weight_data(
        repo_root=ROOT,
        config=config,
        progress=progress,
    )
    if report["status"] != "complete":
        print(
            "minerva-full-data | paused at a completed-document checkpoint; "
            "rerun the same command to resume",
            flush=True,
        )
        return
    print(
        "minerva-full-data | complete documents={documents:,} tokens={tokens:,} "
        "shards={shards:,} report={report}".format(
            documents=report["totals"]["documents"],
            tokens=report["totals"]["tokens"],
            shards=report["totals"]["shards"],
            report=ROOT / config.public_report_path,
        ),
        flush=True,
    )


def _format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


if __name__ == "__main__":
    main()
