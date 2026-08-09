#!/usr/bin/env python3
"""Audit every canonical Biblioteca Italiana TEI across the three corpus roles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.bibit_role_audit import (
    BibItRoleAuditConfig,
    audit_bibit_tei_roles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decision-csv",
        type=Path,
        default=ROOT / "data/metadata/bibit_historical_composition_decisions.csv",
    )
    parser.add_argument(
        "--sonnet-manifest",
        type=Path,
        default=ROOT / "data/metadata/sonnets_expanded_v6_manifest.csv",
    )
    parser.add_argument(
        "--tei-cache-dir",
        type=Path,
        default=ROOT / "data/local/bibit/tei",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=ROOT / "data/local/bibit/tei_audit_checkpoint.json",
    )
    parser.add_argument(
        "--record-csv",
        type=Path,
        default=ROOT / "data/metadata/bibit_tei_audit_records.csv",
    )
    parser.add_argument(
        "--sonnet-csv",
        type=Path,
        default=ROOT / "data/metadata/bibit_sonnet_candidates_audit.csv",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=ROOT / "reports/bibit_tei_role_audit.json",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=ROOT / "reports/bibit_tei_role_audit.md",
    )
    parser.add_argument("--request-delay", type=float, default=0.25)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--min-training-characters", type=int, default=200)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Audit only the first N selected records; zero audits the full catalog.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scope = f"first_{args.limit}" if args.limit else "all_canonical_records"
    print(
        "bibit-tei-audit | start "
        f"scope={scope} device=cpu progress_interval={args.progress_interval} "
        "estimated_runtime=15m-90m_full_catalog network_dependent resumable_cache=true",
        flush=True,
    )

    def progress(message: str) -> None:
        print(f"bibit-tei-audit | {message}", flush=True)

    report = audit_bibit_tei_roles(
        BibItRoleAuditConfig(
            repo_root=ROOT,
            decision_csv_path=args.decision_csv,
            sonnet_manifest_path=args.sonnet_manifest,
            tei_cache_dir=args.tei_cache_dir,
            checkpoint_path=args.checkpoint_path,
            record_csv_path=args.record_csv,
            sonnet_csv_path=args.sonnet_csv,
            json_report_path=args.json_report,
            markdown_report_path=args.markdown_report,
            request_delay_seconds=args.request_delay,
            request_timeout_seconds=args.request_timeout,
            max_retries=args.max_retries,
            progress_interval=args.progress_interval,
            checkpoint_interval=args.checkpoint_interval,
            min_training_characters=args.min_training_characters,
            limit=args.limit,
        ),
        progress=progress,
    )
    print(
        "bibit-tei-audit | complete "
        f"records={report['record_count']:,} "
        f"sonnet_candidates={report['sonnet_candidate_count']:,} "
        f"characters={report['routed_training_character_count']:,} "
        f"status={report['corpus_activation_status']}",
        flush=True,
    )
    print(f"bibit-tei-audit | wrote report: {args.markdown_report}", flush=True)
    print(f"bibit-tei-audit | wrote record decisions: {args.record_csv}", flush=True)
    print(f"bibit-tei-audit | wrote sonnet decisions: {args.sonnet_csv}", flush=True)


if __name__ == "__main__":
    main()
