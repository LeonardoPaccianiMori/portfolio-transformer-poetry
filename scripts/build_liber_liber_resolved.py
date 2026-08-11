#!/usr/bin/env python3
"""Resolve and build the inactive checkpoint-5C Liber Liber corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.liber_liber_resolved_build import (
    LiberLiberResolvedBuildConfig,
    build_liber_liber_resolved_corpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-shard-mib", type=int, default=64)
    parser.add_argument("--progress-interval", type=int, default=25)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> LiberLiberResolvedBuildConfig:
    metadata = ROOT / "data/metadata"
    processed = ROOT / "data/processed"
    reports = ROOT / "reports"
    return LiberLiberResolvedBuildConfig(
        repo_root=ROOT,
        probe_path=metadata / "liber_liber_fulltext_probe_v1.csv",
        probe_review_path=metadata / "liber_liber_fulltext_probe_review_v1.csv",
        probe_report_path=reports / "liber_liber_fulltext_probe_v1.json",
        archive_inventory_path=metadata / "liber_liber_archive_inventory_v1.csv",
        source_rights_path=metadata / "liber_liber_source_rights_v1.csv",
        source_decisions_path=metadata / "liber_liber_extraction_decisions_v1.csv",
        segment_decisions_path=metadata / "liber_liber_segment_decisions_v1.csv",
        sonnet_decisions_path=metadata / "liber_liber_sonnet_candidates_v1.csv",
        sonnet_review_path=metadata / "liber_liber_sonnet_review_v1.csv",
        canonical_editions_path=metadata / "liber_liber_canonical_editions_v1.csv",
        output_dir=processed / "liber_liber_resolved_v1",
        json_report_path=reports / "liber_liber_resolved_v1_build.json",
        markdown_report_path=reports / "liber_liber_resolved_v1_build.md",
        attribution_notice_path=metadata / "liber_liber_resolved_v1_attribution.md",
        bibit_record_manifest_path=processed / "bibit_resolved_v1/records_manifest.csv",
        bibit_sonnet_manifest_path=processed / "bibit_resolved_v1/sonnets_manifest.csv",
        gutenberg_previous_probe_path=metadata / "project_gutenberg_fulltext_probe_v1.csv",
        gutenberg_previous_cache_dir=ROOT / "data/local/gutenberg/fulltext_gate_v1",
        gutenberg_pass_1b_probe_path=metadata / "project_gutenberg_fulltext_probe_pass_1b_v1.csv",
        gutenberg_pass_1b_cache_dir=ROOT / "data/local/gutenberg/metadata_review_v1",
        gutenberg_resolved_record_manifest_path=processed / "project_gutenberg_resolved_v1/records_manifest.csv",
        gutenberg_resolved_sonnet_manifest_path=processed / "project_gutenberg_resolved_v1/sonnets_manifest.csv",
        wikisource_resolved_record_manifest_path=processed / "italian_wikisource_resolved_v1/records_manifest.csv",
        wikisource_resolved_sonnet_manifest_path=processed / "italian_wikisource_resolved_v1/sonnets_manifest.csv",
        broader_sources_manifest_path=metadata / "broader_prose_sources_manifest.csv",
        protected_v6_sonnet_manifest_path=metadata / "sonnets_expanded_v6_manifest.csv",
        cache_dir=ROOT / "data/local/liber_liber/fulltext_probe_v1",
        max_shard_bytes=args.max_shard_mib * 1024 * 1024,
        progress_interval=args.progress_interval,
    )


def main() -> None:
    args = parse_args()
    print(
        "liber-liber-resolved-build | start device=cpu total_records=129 "
        f"max_shard_mib={args.max_shard_mib} progress_interval={args.progress_interval} "
        "estimated_runtime=1m-8m_cached activation=false",
        flush=True,
    )
    report = build_liber_liber_resolved_corpus(
        config_from_args(args),
        progress=lambda message: print(f"liber-liber-resolved-build | {message}", flush=True),
    )
    print(
        "liber-liber-resolved-build | complete "
        f"records={report['materialized_record_count']:,} "
        f"sonnets={report['materialized_sonnet_count']:,} "
        f"characters={report['materialized_broader_character_count']:,} "
        f"shards={report['shard_count']:,} activated=0",
        flush=True,
    )
    print(
        f"liber-liber-resolved-build | wrote corpus: {config_from_args(args).output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
