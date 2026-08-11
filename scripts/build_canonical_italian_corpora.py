#!/usr/bin/env python3
"""Build checkpoint-7B inactive, manifest-backed canonical Italian corpora."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.canonical_corpus_build import (
    CanonicalCorpusBuildConfig,
    run_canonical_corpus_build,
)


def build_config() -> CanonicalCorpusBuildConfig:
    metadata = ROOT / "data/metadata"
    processed = ROOT / "data/processed"
    output = processed / "canonical_italian_corpora_v1"
    return CanonicalCorpusBuildConfig(
        repo_root=ROOT,
        unit_path=metadata / "cross_archive_canonical_units_v1.csv",
        decision_path=metadata / "cross_archive_canonical_decisions_v1.csv",
        review_path=metadata / "cross_archive_canonical_review_v1.csv",
        overlap_path=metadata / "cross_archive_overlap_v1.csv",
        ilc_ota_unit_path=metadata / "ilc_ota_text_units_v1.csv",
        ilc_ota_inventory_path=metadata / "ilc_ota_source_inventory_v1.csv",
        v6_manifest_path=metadata / "sonnets_expanded_v6_manifest.csv",
        bibit_period_path=metadata / "bibit_tei_audit_records.csv",
        gutenberg_period_path=metadata / "project_gutenberg_extraction_decisions_v1.csv",
        gutenberg_attribution_path=processed / "project_gutenberg_resolved_v1/attribution_manifest.csv",
        wikisource_period_path=metadata / "italian_wikisource_root_decisions_v1.csv",
        wikisource_sonnet_path=processed / "italian_wikisource_resolved_v1/sonnets_manifest.csv",
        wikisource_attribution_path=processed / "italian_wikisource_resolved_v1/attribution_manifest.csv",
        liber_liber_period_path=metadata / "liber_liber_extraction_decisions_v1.csv",
        liber_liber_sonnet_path=processed / "liber_liber_resolved_v1/sonnets_manifest.csv",
        liber_liber_attribution_path=processed / "liber_liber_resolved_v1/attribution_manifest.csv",
        existing_historical_reports=(
            processed / "expanded_italian_1200_1800_v1/expanded_italian_1200_1800_v1_build_report.json",
            processed / "pretraining_historical_wikisource_v1/pretraining_historical_wikisource_v1_build_report.json",
        ),
        segment_range_path=metadata / "cross_archive_segment_ranges_v1.csv",
        sonnet_routing_path=metadata / "cross_archive_sonnet_routing_v1.csv",
        output_dir=output,
        json_report_path=ROOT / "reports/canonical_italian_corpora_v1.json",
        markdown_report_path=ROOT / "reports/canonical_italian_corpora_v1.md",
    )


def main() -> None:
    print(
        "canonical-corpus-build | start device=cpu total_units=27311 "
        "progress_interval=250 activation=false estimated_runtime=1-3m",
        flush=True,
    )
    report = run_canonical_corpus_build(
        build_config(),
        progress=lambda message: print(f"canonical-corpus-build | {message}", flush=True),
    )
    print(
        "canonical-corpus-build | complete "
        f"records={report['training_record_count']} sonnets={report['training_sonnet_count']} "
        f"logical_characters={report['logical_character_count']} "
        f"delta_bytes={report['delta_shard_byte_count']} activated=0 v7=0 gpu=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
