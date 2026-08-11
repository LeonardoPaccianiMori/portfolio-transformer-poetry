#!/usr/bin/env python3
"""Probe the bounded checkpoint-5A Liber Liber full-text queue."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.liber_liber_archive_probe import (
    LiberLiberArchiveProbeConfig,
    run_liber_liber_archive_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-delay", type=float, default=0.25)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument(
        "--allow-unresolved-review",
        action="store_true",
        help="Write the bounded anomaly queue before its manual decisions are complete.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> LiberLiberArchiveProbeConfig:
    metadata = ROOT / "data/metadata"
    processed = ROOT / "data/processed"
    return LiberLiberArchiveProbeConfig(
        repo_root=ROOT,
        inventory_path=metadata / "liber_liber_archive_inventory_v1.csv",
        cache_dir=ROOT / "data/local/liber_liber/fulltext_probe_v1",
        output_csv_path=metadata / "liber_liber_fulltext_probe_v1.csv",
        review_csv_path=metadata / "liber_liber_fulltext_probe_review_v1.csv",
        json_report_path=ROOT / "reports/liber_liber_fulltext_probe_v1.json",
        markdown_report_path=ROOT / "reports/liber_liber_fulltext_probe_v1.md",
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
        request_delay_seconds=args.request_delay,
        request_timeout_seconds=args.request_timeout,
        require_review_resolutions=not args.allow_unresolved_review,
    )


def main() -> None:
    args = parse_args()
    print(
        "liber-liber-fulltext-probe | start device=cpu total_records=129 "
        "progress_interval=1_record activation=false",
        flush=True,
    )
    report = run_liber_liber_archive_probe(
        config_from_args(args),
        progress=lambda message: print(f"liber-liber-fulltext-probe | {message}", flush=True),
    )
    print(
        "liber-liber-fulltext-probe | complete "
        f"records={report['candidate_count']:,} "
        f"characters={report['cleaned_character_count']:,} "
        f"anomalies={report['manual_review_anomaly_count']:,} "
        f"unresolved={report['manual_review_unresolved_count']:,} activated=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
