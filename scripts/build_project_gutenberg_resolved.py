#!/usr/bin/env python3
"""Build the frozen Project Gutenberg records into bounded resolved shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.gutenberg_resolved_build import (
    GutenbergResolvedBuildConfig,
    build_gutenberg_resolved_corpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-decisions",
        type=Path,
        default=ROOT / "data/metadata/project_gutenberg_extraction_decisions_v1.csv",
    )
    parser.add_argument(
        "--segment-decisions",
        type=Path,
        default=ROOT / "data/metadata/project_gutenberg_segment_decisions_v1.csv",
    )
    parser.add_argument(
        "--sonnet-decisions",
        type=Path,
        default=ROOT / "data/metadata/project_gutenberg_sonnet_candidates_v1.csv",
    )
    parser.add_argument(
        "--sonnet-review",
        type=Path,
        default=ROOT / "data/metadata/project_gutenberg_sonnet_review_v1.csv",
    )
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=ROOT / "reports/project_gutenberg_extraction_audit_v1.json",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=ROOT / "data/metadata/project_gutenberg_italian_inventory_v1.csv",
    )
    parser.add_argument(
        "--bibit-record-manifest",
        type=Path,
        default=ROOT / "data/processed/bibit_resolved_v1/records_manifest.csv",
    )
    parser.add_argument(
        "--broader-sources-manifest",
        type=Path,
        default=ROOT / "data/metadata/broader_prose_sources_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/processed/project_gutenberg_resolved_v1",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=ROOT / "reports/project_gutenberg_resolved_v1_build.md",
    )
    parser.add_argument("--max-shard-mib", type=int, default=64)
    parser.add_argument("--progress-interval", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "gutenberg-resolved-build | start device=cpu "
        "corpus=project_gutenberg_resolved_v1 records=587 "
        f"max_shard_mib={args.max_shard_mib} "
        f"progress_interval={args.progress_interval} "
        "estimated_runtime=3m-12m_cached",
        flush=True,
    )

    def progress(message: str) -> None:
        print(f"gutenberg-resolved-build | {message}", flush=True)

    report = build_gutenberg_resolved_corpus(
        GutenbergResolvedBuildConfig(
            repo_root=ROOT,
            source_decisions_path=args.source_decisions,
            segment_decisions_path=args.segment_decisions,
            sonnet_decisions_path=args.sonnet_decisions,
            sonnet_review_path=args.sonnet_review,
            audit_report_path=args.audit_report,
            inventory_path=args.inventory,
            bibit_record_manifest_path=args.bibit_record_manifest,
            broader_sources_manifest_path=args.broader_sources_manifest,
            output_dir=args.output_dir,
            markdown_report_path=args.markdown_report,
            max_shard_bytes=args.max_shard_mib * 1024 * 1024,
            progress_interval=args.progress_interval,
        ),
        progress=progress,
    )
    shard_count = sum(len(shards) for shards in report["shards"].values())
    print(
        "gutenberg-resolved-build | complete "
        f"sources={report['materialized_source_count']:,} "
        f"sonnets={report['materialized_sonnet_count']:,} "
        f"shards={shard_count:,}",
        flush=True,
    )
    print(f"gutenberg-resolved-build | wrote corpus: {args.output_dir}", flush=True)
    print(
        f"gutenberg-resolved-build | wrote report: {args.markdown_report}",
        flush=True,
    )


if __name__ == "__main__":
    main()
