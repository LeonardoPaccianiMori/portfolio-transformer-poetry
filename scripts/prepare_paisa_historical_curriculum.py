#!/usr/bin/env python3
"""Prepare local PAISÀ-to-historical rescue inputs and train-only BPE sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.pretraining_curriculum import (
    load_paisa_historical_curriculum_config,
    prepare_paisa_historical_curriculum,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=ROOT / "configs/paisa_historical_rescue_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_paisa_historical_curriculum_config(args.config_path)
    print(
        "paisa-curriculum | start "
        f"curriculum_id={config.curriculum_id} "
        f"historical_validation_fraction={config.historical_source_validation_fraction:.2%}",
        flush=True,
    )
    report = prepare_paisa_historical_curriculum(
        config,
        progress=lambda message: print(f"paisa-curriculum | {message}", flush=True),
    )
    tokenizer = report["tokenizer"]
    print(
        "paisa-curriculum | complete "
        f"paisa_sample_characters={tokenizer['paisa_sample']['selected_characters']} "
        f"historical_sample_characters={tokenizer['historical_sample']['selected_characters']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
