#!/usr/bin/env python3
"""Audit all compatible checkpoint-6D ILC-CNR and Oxford text candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.ilc_ota_source_audit import ILCOTAAuditConfig, run_ilc_ota_source_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-delay", type=float, default=0.25)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ILCOTAAuditConfig:
    metadata = ROOT / "data/metadata"
    processed = ROOT / "data/processed"
    return ILCOTAAuditConfig(
        repo_root=ROOT,
        discovery_cache_dir=ROOT / "data/local/archive_discovery_v1",
        cache_dir=ROOT / "data/local/ilc_ota_source_audit_v1",
        inventory_path=metadata / "ilc_ota_source_inventory_v1.csv",
        file_path=metadata / "ilc_ota_source_files_v1.csv",
        unit_path=metadata / "ilc_ota_text_units_v1.csv",
        overlap_path=metadata / "ilc_ota_overlap_v1.csv",
        review_path=metadata / "ilc_ota_source_review_v1.csv",
        decision_path=metadata / "ilc_ota_source_decisions_v1.csv",
        json_report_path=ROOT / "reports/ilc_ota_source_audit_v1.json",
        markdown_report_path=ROOT / "reports/ilc_ota_source_audit_v1.md",
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
        liber_liber_resolved_record_manifest_path=processed / "liber_liber_resolved_v1/records_manifest.csv",
        liber_liber_resolved_sonnet_manifest_path=processed / "liber_liber_resolved_v1/sonnets_manifest.csv",
        broader_sources_manifest_path=metadata / "broader_prose_sources_manifest.csv",
        protected_v6_sonnet_manifest_path=metadata / "sonnets_expanded_v6_manifest.csv",
        request_delay_seconds=args.request_delay,
        request_timeout_seconds=args.request_timeout,
    )


def main() -> None:
    args = parse_args()
    print(
        "ilc-ota-source-audit | start device=cpu total_inventory_records=46 "
        "progress_interval=1_item activation=false estimated_runtime=20-90m_first_run",
        flush=True,
    )
    report = run_ilc_ota_source_audit(
        config_from_args(args),
        progress=lambda message: print(f"ilc-ota-source-audit | {message}", flush=True),
    )
    print(
        "ilc-ota-source-audit | complete "
        f"inventory={report['inventory_record_count']} "
        f"units={report['extracted_unit_count']} "
        f"eligible_units={report['eligible_unit_count']} "
        f"candidate_characters={report['candidate_character_count']} "
        f"eligible_characters={report['eligible_character_count']} activated=0",
        flush=True,
    )


if __name__ == "__main__":
    main()
