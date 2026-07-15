#!/usr/bin/env python3
"""Build the local broader Italian pretraining corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.pretraining_build import (
    PretrainingBuildConfig,
    build_pretraining_corpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the local broader Italian pretraining corpus.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=ROOT / "data/metadata/broader_prose_sources_manifest.csv",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT / "data/local/pretraining/processed",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=ROOT / "data/local/pretraining/build_report.json",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=ROOT / "data/interim/pretraining_build",
    )
    parser.add_argument("--corpus-version", default="broader_prose_v1")
    parser.add_argument("--request-delay-seconds", type=float, default=1.0)
    parser.add_argument(
        "--wikisource-snapshot-dir",
        type=Path,
        default=ROOT / "data/metadata/wikisource_snapshots",
    )
    parser.add_argument("--wikisource-request-delay-seconds", type=float, default=6.0)
    parser.add_argument("--min-character-count", type=int, default=200)
    parser.add_argument("--quiet", action="store_true", help="Hide progress output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PretrainingBuildConfig(
        manifest_path=args.manifest_path,
        processed_dir=args.processed_dir,
        report_path=args.report_path,
        temp_dir=args.temp_dir,
        corpus_version=args.corpus_version,
        request_delay_seconds=args.request_delay_seconds,
        wikisource_snapshot_dir=args.wikisource_snapshot_dir,
        wikisource_request_delay_seconds=args.wikisource_request_delay_seconds,
        min_character_count=args.min_character_count,
    )
    report = build_pretraining_corpus(
        config,
        progress=None if args.quiet else lambda message: print(f"build | {message}", flush=True),
    )
    print(
        "Built "
        f"{report.selected_rows} sources, "
        f"{report.total_cleaned_characters} characters, "
        f"{report.total_cleaned_words} whitespace-delimited units."
    )
    print(f"Processed corpus: {report.processed_dir}")
    print(f"Build report: {args.report_path}")


if __name__ == "__main__":
    main()
