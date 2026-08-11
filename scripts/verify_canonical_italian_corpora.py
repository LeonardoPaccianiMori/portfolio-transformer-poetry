#!/usr/bin/env python3
"""Exhaustively verify and freeze the checkpoint-7B logical corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.canonical_corpus_reader import (
    CanonicalCorpusReader,
    write_acceptance_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify every canonical logical-corpus storage slice."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=ROOT / "data/processed/canonical_italian_corpora_v1",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=ROOT / "reports/canonical_italian_corpus_acceptance_v1.json",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=ROOT / "reports/canonical_italian_corpus_acceptance_v1.md",
    )
    parser.add_argument("--progress-interval", type=int, default=100)
    args = parser.parse_args()
    if args.progress_interval <= 0:
        parser.error("--progress-interval must be positive")
    return args


def _duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> None:
    args = parse_args()
    reader = CanonicalCorpusReader(ROOT, args.corpus_dir)
    total_files = len({unit.storage_path for unit in reader.units})
    started = monotonic()
    print(
        "canonical-corpus-acceptance | start device=cpu "
        f"total_files={total_files} total_slices={len(reader.units)} "
        f"progress_interval={args.progress_interval} activation=false "
        "estimated_runtime=1-3m",
        flush=True,
    )

    def progress(completed: int, total: int, path: str) -> None:
        if completed != 1 and completed % args.progress_interval != 0 and completed != total:
            return
        elapsed = monotonic() - started
        remaining = elapsed * (total - completed) / completed
        print(
            "canonical-corpus-acceptance | "
            f"files={completed}/{total} percentage={100 * completed / total:.1f}% "
            f"elapsed={_duration(elapsed)} eta={_duration(remaining)} path={path}",
            flush=True,
        )

    report = reader.verify(progress=progress)
    write_acceptance_reports(report, args.json_report, args.markdown_report)
    print(
        "canonical-corpus-acceptance | complete "
        f"status={report['acceptance_status']} files={report['physical_file_count']} "
        f"slices={report['stored_unit_count']} "
        f"training_characters={report['training_logical_character_count']} "
        f"elapsed={_duration(monotonic() - started)} activated=0 v7=0 tokens=0 gpu=0 "
        f"json_report={args.json_report} markdown_report={args.markdown_report}",
        flush=True,
    )


if __name__ == "__main__":
    main()
