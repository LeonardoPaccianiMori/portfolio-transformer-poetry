#!/usr/bin/env python3
"""Audit all remaining approved Wikisource sonnet candidates without activation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.sonnet_audit_batch import run_remaining_sonnet_source_audits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "data/metadata/sonnet_expansion_sources_manifest.csv",
    )
    parser.add_argument(
        "--active-poems-manifest",
        type=Path,
        default=ROOT / "data/metadata/sonnets_expanded_v3_manifest.csv",
    )
    parser.add_argument(
        "--reports-directory",
        type=Path,
        default=ROOT / "data/local/sonnet_audits",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "data/local/sonnet_audits/remaining_sources_summary.json",
    )
    parser.add_argument("--request-delay", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def progress(message: str) -> None:
        print(f"sonnet-audit | {message}", flush=True)

    summary = run_remaining_sonnet_source_audits(
        source_manifest_path=args.source_manifest,
        active_poems_manifest_path=args.active_poems_manifest,
        repo_root=ROOT,
        reports_directory=args.reports_directory,
        summary_path=args.summary,
        request_delay=args.request_delay,
        progress=progress,
    )
    completed = sum(result["status"] == "ok" for result in summary["results"])
    print(
        f"sonnet-audit | complete sources={completed}/{len(summary['results'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
