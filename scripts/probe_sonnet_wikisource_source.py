#!/usr/bin/env python3
"""Probe one Italian Wikisource sonnet collection without activating it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.sonnet_wikisource_probe import probe_sonnet_wikisource_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "data/metadata/sonnet_expansion_sources_manifest.csv",
        help="Public sonnet expansion source manifest CSV.",
    )
    parser.add_argument(
        "--active-poems-manifest",
        type=Path,
        default=ROOT / "data/metadata/sonnets_expanded_v3_manifest.csv",
        help="Active processed-poem manifest used for duplicate checks.",
    )
    parser.add_argument(
        "--source-id",
        default="ws_foscolo_sonetti",
        help="One Italian-Wikisource source approved for audit.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Local JSON inspection report path; defaults from --source-id.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=6.0,
        help="Minimum seconds between MediaWiki API requests (default: 6).",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide progress output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.report or (
        ROOT / "data/local/sonnet_audits" / f"{args.source_id}_probe.json"
    )

    def progress(message: str) -> None:
        if not args.quiet:
            print(f"sonnet-probe | {message}", flush=True)

    report = probe_sonnet_wikisource_source(
        source_manifest_path=args.source_manifest,
        active_poems_manifest_path=args.active_poems_manifest,
        repo_root=ROOT,
        source_id=args.source_id,
        report_path=report_path,
        request_delay=args.request_delay,
        progress=progress,
    )
    if not args.quiet:
        counts = report["candidate_status_counts"]
        print(
            "sonnet-probe | complete "
            f"pages={report['page_count']} "
            f"eligible_14_lines={counts.get('eligible_14_lines', 0)} "
            f"duplicates={counts.get('exact_duplicate_active_corpus', 0)} "
            f"activation_status={report['activation_status']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
