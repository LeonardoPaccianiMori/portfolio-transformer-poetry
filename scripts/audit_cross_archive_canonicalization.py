#!/usr/bin/env python3
"""Freeze checkpoint-7A cross-archive overlap and canonical decisions."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.cross_archive_canonicalization import (
    CrossArchiveCanonicalizationConfig,
    run_cross_archive_canonicalization,
)


def build_config() -> CrossArchiveCanonicalizationConfig:
    metadata = ROOT / "data/metadata"
    processed = ROOT / "data/processed"
    return CrossArchiveCanonicalizationConfig(
        repo_root=ROOT,
        existing_historical_reports=(
            processed / "expanded_italian_1200_1800_v1/expanded_italian_1200_1800_v1_build_report.json",
            processed / "pretraining_historical_wikisource_v1/pretraining_historical_wikisource_v1_build_report.json",
        ),
        v6_manifest_path=metadata / "sonnets_expanded_v6_manifest.csv",
        bibit_record_manifest_path=processed / "bibit_resolved_v1/records_manifest.csv",
        bibit_sonnet_manifest_path=processed / "bibit_resolved_v1/sonnets_manifest.csv",
        gutenberg_record_manifest_path=processed / "project_gutenberg_resolved_v1/records_manifest.csv",
        gutenberg_sonnet_manifest_path=processed / "project_gutenberg_resolved_v1/sonnets_manifest.csv",
        wikisource_record_manifest_path=processed / "italian_wikisource_resolved_v1/records_manifest.csv",
        wikisource_sonnet_manifest_path=processed / "italian_wikisource_resolved_v1/sonnets_manifest.csv",
        liber_liber_record_manifest_path=processed / "liber_liber_resolved_v1/records_manifest.csv",
        liber_liber_sonnet_manifest_path=processed / "liber_liber_resolved_v1/sonnets_manifest.csv",
        ilc_ota_unit_path=metadata / "ilc_ota_text_units_v1.csv",
        unit_index_path=metadata / "cross_archive_canonical_units_v1.csv",
        overlap_path=metadata / "cross_archive_overlap_v1.csv",
        decision_path=metadata / "cross_archive_canonical_decisions_v1.csv",
        review_path=metadata / "cross_archive_canonical_review_v1.csv",
        json_report_path=ROOT / "reports/cross_archive_canonicalization_v1.json",
        markdown_report_path=ROOT / "reports/cross_archive_canonicalization_v1.md",
    )


def main() -> None:
    print(
        "cross-archive-canonicalization | start device=cpu "
        "total_inputs=audited_text_only progress_interval=100_broader_or_1000_sonnets "
        "activation=false estimated_runtime=10-40m",
        flush=True,
    )
    report = run_cross_archive_canonicalization(
        build_config(),
        progress=lambda message: print(
            f"cross-archive-canonicalization | {message}", flush=True,
        ),
    )
    print(
        "cross-archive-canonicalization | complete "
        f"units={report['unit_count']} overlaps={report['overlap_pair_count']} "
        f"reviews={report['review_row_count']} activated=0 v7=0 gpu=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
