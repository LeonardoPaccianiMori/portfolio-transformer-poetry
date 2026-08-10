#!/usr/bin/env python3
"""Probe the frozen 167-record pass-1B Project Gutenberg queue."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.gutenberg_fulltext_probe import (
    GutenbergFullTextProbeConfig,
    run_gutenberg_fulltext_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/local/gutenberg/metadata_review_v1",
    )
    parser.add_argument(
        "--prior-cache-dir",
        type=Path,
        default=ROOT / "data/local/gutenberg/fulltext_gate_v1",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=(
            ROOT
            / "data/metadata/project_gutenberg_fulltext_probe_pass_1b_v1.csv"
        ),
    )
    parser.add_argument(
        "--review-csv",
        type=Path,
        default=(
            ROOT
            / "data/metadata/project_gutenberg_fulltext_probe_pass_1b_review_v1.csv"
        ),
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=ROOT / "reports/project_gutenberg_fulltext_probe_pass_1b_v1.json",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=ROOT / "reports/project_gutenberg_fulltext_probe_pass_1b_v1.md",
    )
    parser.add_argument(
        "--allow-unresolved-reviews",
        action="store_true",
        help="Write the bounded anomaly ledger before its manual decisions are complete.",
    )
    return parser.parse_args()


def _selected_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            row["ebook_id"]
            for row in csv.DictReader(handle)
            if row["resolution_pass"] == "pass_1b"
            and row["final_activation_class"] == "eligible_probe"
        ]


def main() -> None:
    args = parse_args()
    resolution_path = (
        ROOT / "data/metadata/project_gutenberg_metadata_final_resolution_v1.csv"
    )
    selected_ids = _selected_ids(resolution_path)
    cached_count = sum(
        (args.cache_dir / f"pg{ebook_id}.txt").is_file()
        for ebook_id in selected_ids
    )
    estimated_runtime = (
        "2m-6m_cached"
        if len(selected_ids) == 167 and cached_count == len(selected_ids)
        else "5m-45m_cache_or_network_dependent"
    )
    print(
        "gutenberg-pass-1b-probe | start device=cpu candidates=167 "
        f"cached_texts={cached_count:,} prior_pool=416 "
        f"request_delay={args.request_delay:.1f}s cache_reusable=true "
        f"estimated_runtime={estimated_runtime}",
        flush=True,
    )

    def progress(message: str) -> None:
        print(f"gutenberg-pass-1b-probe | {message}", flush=True)

    report = run_gutenberg_fulltext_probe(
        GutenbergFullTextProbeConfig(
            repo_root=ROOT,
            inventory_csv_path=(
                ROOT / "data/metadata/project_gutenberg_italian_inventory_v1.csv"
            ),
            cache_dir=args.cache_dir,
            output_csv_path=args.output_csv,
            json_report_path=args.json_report,
            markdown_report_path=args.markdown_report,
            bibit_record_manifest_path=(
                ROOT / "data/processed/bibit_resolved_v1/records_manifest.csv"
            ),
            broader_sources_manifest_path=(
                ROOT / "data/metadata/broader_prose_sources_manifest.csv"
            ),
            sonnet_manifest_path=(
                ROOT / "data/metadata/sonnets_expanded_v6_manifest.csv"
            ),
            authoritative_resolution_csv_path=resolution_path,
            required_resolution_pass="pass_1b",
            required_activation_class="eligible_probe",
            expected_candidate_count=167,
            conditioned_activation_class="conditioned_probe",
            expected_conditioned_count=4,
            prior_gutenberg_probe_csv_path=(
                ROOT / "data/metadata/project_gutenberg_fulltext_probe_v1.csv"
            ),
            prior_gutenberg_cache_dir=args.prior_cache_dir,
            expected_prior_gutenberg_count=416,
            review_decisions_csv_path=args.review_csv,
            require_review_resolutions=not args.allow_unresolved_reviews,
            probe_version="project_gutenberg_fulltext_probe_pass_1b_v1",
            request_delay_seconds=args.request_delay,
            request_timeout_seconds=args.request_timeout,
        ),
        progress=progress,
    )
    print(
        "gutenberg-pass-1b-probe | complete "
        f"records={report['candidate_count']:,} "
        f"characters={report['cleaned_character_count']:,} "
        f"prior_overlaps={len(report['prior_gutenberg_duplicate_pairs']):,} "
        f"cross_duplicates={len(report['cross_corpus_duplicate_pairs']):,} "
        f"anomalies={report['manual_review_anomaly_count']:,} "
        f"unresolved={report['manual_review_unresolved_count']:,}",
        flush=True,
    )
    print(f"gutenberg-pass-1b-probe | wrote manifest: {args.output_csv}", flush=True)
    print(f"gutenberg-pass-1b-probe | wrote review ledger: {args.review_csv}", flush=True)
    print(f"gutenberg-pass-1b-probe | wrote report: {args.markdown_report}", flush=True)


if __name__ == "__main__":
    main()
