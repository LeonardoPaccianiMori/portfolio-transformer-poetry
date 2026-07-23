#!/usr/bin/env python3
"""Probe Liber Liber sources for the broader prose corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.liber_liber_probe import probe_liber_liber_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/metadata/broader_prose_sources_manifest.csv",
        help="Broader prose source manifest CSV.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data/metadata/broader_prose_liber_liber_probe_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--attribution",
        type=Path,
        default=ROOT / "data/metadata/broader_prose_attribution.md",
        help="Output Creative Commons attribution Markdown path.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.0,
        help="Minimum seconds between Liber Liber works.",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        default=None,
        help=(
            "Probe only this active prose source ID. Repeat the option to select "
            "multiple sources."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Hide summary output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def progress(message: str) -> None:
        if not args.quiet:
            print(f"probe | {message}", flush=True)

    report = probe_liber_liber_sources(
        manifest_path=args.manifest,
        report_path=args.report,
        attribution_path=args.attribution,
        request_delay=args.request_delay,
        source_ids=set(args.source_id) if args.source_id is not None else None,
        progress=progress,
    )

    if not args.quiet:
        print(
            "probe | complete "
            f"sources={report['selected_rows']} "
            f"successful={report['successful_rows']} "
            f"cleaned_characters={report['total_cleaned_characters']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
