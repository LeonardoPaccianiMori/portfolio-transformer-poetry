#!/usr/bin/env python3
"""Calibrate unquantized FP16 Minerva 7B LoRA training on a remote GPU."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_fp16_lora import (
    Minerva7BFP16LoRACalibrationConfig,
    calibrate_minerva_7b_fp16_lora,
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
        default=ROOT
        / "data"
        / "local"
        / "minerva_qlora"
        / "7b_fp16_lora_calibration.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Minerva7BFP16LoRACalibrationConfig()
    started_at = time.monotonic()

    def progress(message: str) -> None:
        elapsed = round(time.monotonic() - started_at)
        print(
            f"minerva-7b-fp16-calibration | {message} | elapsed={elapsed}s",
            flush=True,
        )

    print(
        "minerva-7b-fp16-calibration | start model={model} context={context} "
        "batch={batch} rank={rank} estimated_runtime=2m-15m_cached".format(
            model=config.model_id,
            context=config.context_length,
            batch=config.batch_size,
            rank=config.lora_rank,
        ),
        flush=True,
    )
    report = calibrate_minerva_7b_fp16_lora(
        config=config,
        cache_dir=args.cache_dir,
        output_path=args.output_path,
        progress=progress,
    )
    print(
        "minerva-7b-fp16-calibration | complete status={status} fit={fit} "
        "peak_reserved={peak:.1f}MiB free_after={free:.1f}MiB output={output}".format(
            status=report["status"],
            fit=report["remote_training_fit_decision"],
            peak=report["peak_reserved_mib"],
            free=report["free_memory_after_mib"],
            output=args.output_path,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
