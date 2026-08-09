#!/usr/bin/env python3
"""Generate the frozen Minerva 7B parent-decoding confirmation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.minerva_7b_parent_confirmation import (
    CONFIRMATION_CONDITIONS,
    CONFIRMATION_OUTPUT_COUNT,
    generate_parent_decoding_confirmation,
)
from sonnet_training.progress import format_duration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/minerva_7b_parent_decoding_confirmation.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/generations/minerva_7b_parent_decoding_confirmation_v1"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/local/minerva_qlora/huggingface"),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Minerva 7B parent confirmation requires CUDA")

    started_at = perf_counter()
    print(
        "parent-confirmation | start "
        f"device={device} conditions={len(CONFIRMATION_CONDITIONS)} "
        f"outputs={CONFIRMATION_OUTPUT_COUNT} "
        "estimated_runtime=25m-75m_cached final_test=false training=false",
        flush=True,
    )
    summary = generate_parent_decoding_confirmation(
        repo_root=ROOT,
        config_path=args.config_path,
        output_root=args.output_dir,
        device=device,
        cache_dir=args.cache_dir,
        progress=lambda message: print(
            f"parent-confirmation | {message}", flush=True
        ),
    )
    print(
        "parent-confirmation | complete "
        f"conditions={summary['condition_count']} "
        f"outputs={summary['output_count']} "
        f"peak_reserved={summary['peak_cuda_reserved_mib']:.1f}MiB "
        f"elapsed={format_duration(perf_counter() - started_at)} "
        f"summary={_resolve(args.output_dir) / 'confirmation_summary.json'}",
        flush=True,
    )


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
