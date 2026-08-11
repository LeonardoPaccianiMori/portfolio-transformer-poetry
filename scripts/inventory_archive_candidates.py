#!/usr/bin/env python3
"""Inventory checkpoint-6B archive metadata without acquiring corpus text."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import monotonic


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.archive_candidate_inventory import (  # noqa: E402
    ArchiveCandidateInventoryConfig,
    build_archive_candidate_inventory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/local/archive_candidate_inventory_v1",
    )
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--request-delay", type=float, default=0.25)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--ia-rows-per-page", type=int, default=1_000)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ArchiveCandidateInventoryConfig:
    metadata = ROOT / "data/metadata"
    return ArchiveCandidateInventoryConfig(
        repo_root=ROOT,
        cache_dir=args.cache_dir,
        inventory_path=metadata / "corpus_archive_candidate_inventory_v1.csv",
        summary_path=metadata / "corpus_archive_inventory_summary_v1.csv",
        json_report_path=ROOT / "reports/corpus_archive_candidate_inventory_v1.json",
        markdown_report_path=ROOT / "reports/corpus_archive_candidate_inventory_v1.md",
        request_timeout_seconds=args.request_timeout,
        request_delay_seconds=args.request_delay,
        max_attempts=args.max_attempts,
        ia_rows_per_page=args.ia_rows_per_page,
    )


def main() -> int:
    args = parse_args()
    started = monotonic()
    print(
        "archive-candidate-inventory | start device=cpu archives=6 "
        "progress_interval=1_page activation=false corpus_text=false "
        f"request_delay={args.request_delay:.2f}s "
        "estimated_runtime=4-15m_network_or_under_1m_cached",
        flush=True,
    )
    report = build_archive_candidate_inventory(
        config_from_args(args),
        progress=lambda message: print(
            f"archive-candidate-inventory | {message}", flush=True,
        ),
    )
    print(
        "archive-candidate-inventory | complete "
        f"rows={report['normalized_record_count']:,} "
        f"candidates={report['candidate_count']:,} "
        f"conditioned={report['conditioned_count']:,} activated=0 "
        f"elapsed={monotonic() - started:.1f}s "
        "artifacts=data/metadata/corpus_archive_candidate_inventory_v1.csv,"
        "data/metadata/corpus_archive_inventory_summary_v1.csv,"
        "reports/corpus_archive_candidate_inventory_v1.json",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
