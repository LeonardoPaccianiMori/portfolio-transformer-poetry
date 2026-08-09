#!/usr/bin/env python3
"""Build resolved Biblioteca Italiana records and sonnets into bounded shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.bibit_processed_build import (
    BibItProcessedBuildConfig,
    build_bibit_processed_corpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record-decisions",
        type=Path,
        default=ROOT / "data/metadata/bibit_record_activation_decisions.csv",
    )
    parser.add_argument(
        "--sonnet-decisions",
        type=Path,
        default=ROOT / "data/metadata/bibit_sonnet_activation_decisions.csv",
    )
    parser.add_argument(
        "--tei-cache-dir",
        type=Path,
        default=ROOT / "data/local/bibit/tei",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/processed/bibit_resolved_v1",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=ROOT / "reports/bibit_resolved_v1_build.md",
    )
    parser.add_argument("--max-shard-mib", type=int, default=48)
    parser.add_argument("--progress-interval", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "bibit-build | start device=cpu corpus=bibit_resolved_v1 "
        f"max_shard_mib={args.max_shard_mib} "
        f"progress_interval={args.progress_interval} "
        "estimated_runtime=2m-15m_cached",
        flush=True,
    )

    def progress(message: str) -> None:
        print(f"bibit-build | {message}", flush=True)

    report = build_bibit_processed_corpus(
        BibItProcessedBuildConfig(
            repo_root=ROOT,
            record_decisions_path=args.record_decisions,
            sonnet_decisions_path=args.sonnet_decisions,
            tei_cache_dir=args.tei_cache_dir,
            output_dir=args.output_dir,
            markdown_report_path=args.markdown_report,
            max_shard_bytes=args.max_shard_mib * 1024 * 1024,
            progress_interval=args.progress_interval,
        ),
        progress=progress,
    )
    shard_count = sum(len(shards) for shards in report["shards"].values())
    print(
        "bibit-build | complete "
        f"records={report['record_count']:,} sonnets={report['sonnet_count']:,} "
        f"shards={shard_count:,}",
        flush=True,
    )
    print(f"bibit-build | wrote corpus: {args.output_dir}", flush=True)
    print(f"bibit-build | wrote report: {args.markdown_report}", flush=True)


if __name__ == "__main__":
    main()
