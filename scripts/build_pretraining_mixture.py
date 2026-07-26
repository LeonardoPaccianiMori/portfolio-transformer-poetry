#!/usr/bin/env python3
"""Build the approved combined historical Italian pretraining corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.pretraining_mixture import (
    build_pretraining_mixture,
    load_pretraining_mixture_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=ROOT / "data/metadata/pretraining_historical_italian_v2_mixture.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pretraining_mixture_config(args.config_path)
    print(
        "mixture | start "
        f"corpus_version={config.corpus_version} components={len(config.components)}",
        flush=True,
    )
    report = build_pretraining_mixture(config)
    print(
        "mixture | complete "
        f"sources={report['source_count']} "
        f"cleaned_characters={report['total_cleaned_characters']} "
        f"cleaned_words={report['total_cleaned_words']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
