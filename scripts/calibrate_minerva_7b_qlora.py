#!/usr/bin/env python3
"""Run the one permitted 7B Instruct QLoRA training-memory calibration."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_qlora import (
    Minerva7BQLoRACalibrationConfig,
    calibrate_minerva_7b_qlora,
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
        default=ROOT / "data" / "local" / "minerva_qlora" / "7b_calibration.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Minerva7BQLoRACalibrationConfig()
    started_at = time.monotonic()

    def progress(message: str) -> None:
        elapsed = round(time.monotonic() - started_at)
        eta_upper = max(0, 20 * 60 - elapsed)
        print(
            f"minerva-7b-calibration | {message} | elapsed={elapsed}s "
            f"eta_upper={eta_upper}s",
            flush=True,
        )

    print(
        "minerva-7b-calibration | start model={model} context={context} "
        "batch={batch} rank={rank} estimated_runtime=5m-20m_cached".format(
            model=config.model_id,
            context=config.context_length,
            batch=config.batch_size,
            rank=config.lora_rank,
        ),
        flush=True,
    )
    report = calibrate_minerva_7b_qlora(
        config=config,
        cache_dir=args.cache_dir,
        output_path=args.output_path,
        progress=progress,
    )
    print(
        "minerva-7b-calibration | complete status={status} fit={fit} "
        "peak_reserved={peak:.1f}MiB free_after={free:.1f}MiB output={output}".format(
            status=report["status"],
            fit=report["local_training_fit_decision"],
            peak=report["peak_reserved_mib"],
            free=report["free_memory_after_mib"],
            output=args.output_path,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
