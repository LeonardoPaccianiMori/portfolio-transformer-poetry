#!/usr/bin/env python3
"""Build deterministic inactive Italian Wikisource role-specific shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.wikisource_resolved_build import (
    WikisourceResolvedBuildConfig,
    build_wikisource_resolved_corpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/processed/italian_wikisource_resolved_v1",
    )
    parser.add_argument("--max-shard-mib", type=int, default=64)
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> WikisourceResolvedBuildConfig:
    metadata = ROOT / "data/metadata"
    gutenberg = ROOT / "data/processed/project_gutenberg_resolved_v1"
    return WikisourceResolvedBuildConfig(
        repo_root=ROOT,
        root_decisions_path=metadata / "italian_wikisource_root_decisions_v1.csv",
        segment_decisions_path=metadata / "italian_wikisource_segment_decisions_v1.csv",
        sonnet_decisions_path=metadata / "italian_wikisource_sonnet_decisions_v1.csv",
        scan_rights_path=metadata / "italian_wikisource_source_rights_v1.csv",
        review_report_path=ROOT / "reports/italian_wikisource_review_resolution_v1.json",
        output_dir=args.output_dir,
        markdown_report_path=ROOT / "reports/italian_wikisource_resolved_v1_build.md",
        bibit_record_manifest_path=ROOT / "data/processed/bibit_resolved_v1/records_manifest.csv",
        broader_sources_manifest_path=metadata / "broader_prose_sources_manifest.csv",
        gutenberg_previous_probe_path=metadata / "project_gutenberg_fulltext_probe_v1.csv",
        gutenberg_previous_cache_dir=ROOT / "data/local/gutenberg/fulltext_gate_v1",
        gutenberg_pass_1b_probe_path=metadata / "project_gutenberg_fulltext_probe_pass_1b_v1.csv",
        gutenberg_pass_1b_cache_dir=ROOT / "data/local/gutenberg/metadata_review_v1",
        gutenberg_resolved_record_manifest_path=gutenberg / "records_manifest.csv",
        protected_sonnet_manifest_path=metadata / "sonnets_expanded_v6_manifest.csv",
        max_shard_bytes=args.max_shard_mib * 1024 * 1024,
        progress_interval=args.progress_interval,
    )


def main() -> None:
    args = parse_args()
    print(
        "wikisource-resolved-build | start device=cpu activation=false "
        f"max_shard_mib={args.max_shard_mib} progress_interval={args.progress_interval} "
        "estimated_runtime=2m-20m",
        flush=True,
    )
    report = build_wikisource_resolved_corpus(
        config_from_args(args),
        progress=lambda message: print(f"wikisource-resolved-build | {message}", flush=True),
    )
    print(
        "wikisource-resolved-build | complete "
        f"records={report['materialized_record_count']:,} "
        f"sonnets={report['materialized_sonnet_count']:,} "
        f"shards={report['shard_count']:,} activated=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
