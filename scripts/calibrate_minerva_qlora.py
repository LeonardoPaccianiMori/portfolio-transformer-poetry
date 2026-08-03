#!/usr/bin/env python3
"""Run the locked one-batch 4-bit QLoRA calibration for Minerva 3B."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_qlora import (
    MinervaQLoRACalibrationConfig,
    calibrate_minerva_qlora,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / "local" / "minerva_qlora" / "huggingface",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=ROOT / "data" / "local" / "minerva_qlora" / "calibration.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MinervaQLoRACalibrationConfig()
    print(
        "minerva-calibration | start model={model} context_length={context} "
        "batch_size={batch} lora_rank={rank}".format(
            model=config.model_id,
            context=config.context_length,
            batch=config.batch_size,
            rank=config.lora_rank,
        ),
        flush=True,
    )
    report = calibrate_minerva_qlora(
        config=config,
        cache_dir=args.cache_dir,
        output_path=args.output_path,
        progress=lambda message: print(f"minerva-calibration | {message}", flush=True),
    )
    print(
        "minerva-calibration | complete loss={loss:.4f} "
        "peak_allocated={memory:.1f} MiB output={output}".format(
            loss=report["loss"],
            memory=report["peak_allocated_mib"],
            output=args.output_path,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
