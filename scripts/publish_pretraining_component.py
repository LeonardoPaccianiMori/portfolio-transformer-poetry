#!/usr/bin/env python3
"""Publish a reviewed local pretraining component into public processed data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.pretraining_mixture import publish_pretraining_component


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-processed-dir", type=Path, required=True)
    parser.add_argument("--source-report-path", type=Path, required=True)
    parser.add_argument("--target-processed-dir", type=Path, required=True)
    parser.add_argument("--target-report-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = publish_pretraining_component(
        source_processed_dir=args.source_processed_dir,
        source_report_path=args.source_report_path,
        target_processed_dir=args.target_processed_dir,
        target_report_path=args.target_report_path,
    )
    print(
        "publish | complete "
        f"sources={len(report['sources'])} "
        f"processed_dir={args.target_processed_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
