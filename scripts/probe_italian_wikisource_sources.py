#!/usr/bin/env python3
"""Audit the approved historical Italian Wikisource candidate batch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.italian_wikisource_probe import probe_italian_wikisource_sources


DEFAULT_SOURCE_IDS = [
    "ws_giannone_istoria_civile_vol1",
    "ws_giannone_istoria_civile_vol2",
    "ws_giannone_istoria_civile_vol3",
    "ws_giannone_istoria_civile_vol4",
    "ws_giannone_istoria_civile_vol5",
    "ws_sarpi_istoria_concilio",
    "ws_beccaria_delitti_pene",
    "ws_verri_storia_milano",
    "ws_verri_osservazioni_tortura",
    "ws_verri_meditazioni_economia",
    "ws_verri_discorso_piacere",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/metadata/broader_prose_sources_manifest.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data/local/pretraining/wikisource/historical_core_batch_probe.json",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        dest="source_ids",
        help="Repeat to override the default historical candidate batch.",
    )
    parser.add_argument("--request-delay", type=float, default=6.0)
    parser.add_argument("--max-samples-per-marker", type=int, default=10)
    parser.add_argument("--quiet", action="store_true", help="Hide progress output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_ids = args.source_ids or DEFAULT_SOURCE_IDS

    def progress(message: str) -> None:
        if not args.quiet:
            print(f"historical-audit | {message}", flush=True)

    print(
        "historical-audit | start "
        f"sources={len(source_ids)} request_delay={args.request_delay:g}s",
        flush=True,
    )
    report = probe_italian_wikisource_sources(
        manifest_path=args.manifest,
        source_ids=source_ids,
        report_path=args.report,
        request_delay=args.request_delay,
        max_samples_per_marker=args.max_samples_per_marker,
        progress=progress,
    )
    print(f"historical-audit | wrote report: {args.report}", flush=True)
    print(
        "historical-audit | complete "
        f"ok={report['successful_sources']} errors={report['error_sources']} "
        f"characters={report['total_cleaned_characters']}",
        flush=True,
    )
    if report["error_sources"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
