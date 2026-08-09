#!/usr/bin/env python3
"""Run the metadata-first Biblioteca Italiana historical-corpus composition gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.bibit_composition_audit import (
    BibItCompositionAuditConfig,
    audit_bibit_composition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-snapshot",
        type=Path,
        default=ROOT / "data/metadata/bibit_catalog_origins_through_ottocento_v1.json",
    )
    parser.add_argument(
        "--decision-csv",
        type=Path,
        default=ROOT / "data/metadata/bibit_historical_composition_decisions.csv",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=ROOT / "reports/bibit_historical_composition_audit.json",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=ROOT / "reports/bibit_historical_composition_audit.md",
    )
    parser.add_argument("--sample-per-stratum", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--rendered-batch-size", type=int, default=100)
    parser.add_argument("--bridge-share-recommendation", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "bibit-audit | start scope=origins_through_ottocento "
        f"sample_per_stratum={args.sample_per_stratum} "
        "estimated_runtime=2m-10m network_dependent",
        flush=True,
    )

    def progress(message: str) -> None:
        print(f"bibit-audit | {message}", flush=True)

    report = audit_bibit_composition(
        BibItCompositionAuditConfig(
            repo_root=ROOT,
            catalog_snapshot_path=args.catalog_snapshot,
            decision_csv_path=args.decision_csv,
            json_report_path=args.json_report,
            markdown_report_path=args.markdown_report,
            sample_per_stratum=args.sample_per_stratum,
            request_timeout_seconds=args.request_timeout,
            rendered_batch_size=args.rendered_batch_size,
            bridge_share_recommendation=args.bridge_share_recommendation,
        ),
        progress=progress,
    )
    print(
        "bibit-audit | complete "
        f"records={report['catalog_record_count']:,} "
        f"historical_token_range={report['estimated_historical_active_min_tokens']:,}-"
        f"{report['estimated_historical_active_max_tokens']:,} "
        f"status={report['corpus_activation_status']}",
        flush=True,
    )
    print(f"bibit-audit | wrote report: {args.markdown_report}", flush=True)
    print(f"bibit-audit | wrote decisions: {args.decision_csv}", flush=True)


if __name__ == "__main__":
    main()
