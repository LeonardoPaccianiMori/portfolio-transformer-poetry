#!/usr/bin/env python3
"""Probe Project Gutenberg sources for the broader prose corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.pretraining_probe import probe_gutenberg_sources


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
        default=ROOT / "data/metadata/broader_prose_probe_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "data/metadata/bpe_tokenizer.json",
        help="Optional BPE tokenizer JSON used for token counts.",
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Skip BPE token counts even if a tokenizer file exists.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.0,
        help="Minimum seconds between Project Gutenberg requests.",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide summary output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_path = None if args.no_tokenizer else args.tokenizer
    if tokenizer_path is not None and not tokenizer_path.is_file():
        tokenizer_path = None

    report = probe_gutenberg_sources(
        manifest_path=args.manifest,
        report_path=args.report,
        tokenizer_path=tokenizer_path,
        request_delay=args.request_delay,
    )

    if not args.quiet:
        token_summary = report["total_bpe_tokens"]
        token_text = "not counted" if token_summary is None else str(token_summary)
        print(
            f"probed {report['selected_rows']} Gutenberg rows; "
            f"cleaned characters: {report['total_cleaned_characters']}; "
            f"BPE tokens: {token_text}"
        )


if __name__ == "__main__":
    main()
