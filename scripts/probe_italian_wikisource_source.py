#!/usr/bin/env python3
"""Probe one audited Italian Wikisource work without activating it for training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.italian_wikisource_probe import probe_italian_wikisource_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/metadata/broader_prose_sources_manifest.csv",
        help="Broader prose source manifest CSV.",
    )
    parser.add_argument(
        "--max-samples-per-marker",
        type=int,
        default=10,
        help="Maximum retained contexts for each editorial-marker type.",
    )
    parser.add_argument(
        "--source-id",
        default="ws_vico_scienza_nuova",
        help="One audit-only Italian Wikisource source ID.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data/local/pretraining/wikisource/ws_vico_scienza_nuova_probe.json",
        help="Local JSON inspection report path.",
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

    def progress(message: str) -> None:
        if not args.quiet:
            print(f"probe | {message}", flush=True)

    report = probe_italian_wikisource_source(
        manifest_path=args.manifest,
        source_id=args.source_id,
        report_path=args.report,
        request_delay=args.request_delay,
        max_samples_per_marker=args.max_samples_per_marker,
        progress=progress,
    )
    result = report["result"]
    if not args.quiet:
        print(
            "probe | complete "
            f"status={result['status']} pages={result['page_count']} "
            f"characters={result['cleaned_character_count']} "
            "activation_status=audit_then_include",
            flush=True,
        )
    if result["status"] != "ok":
        print(f"probe | failed error={result['error']}", file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
