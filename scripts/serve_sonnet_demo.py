#!/usr/bin/env python3
"""Serve the local Minerva 7B sonnet-generation demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_demo.server import (
    StaticDemoGenerator,
    load_selected_sonnet_generator,
    serve_demo,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/minerva_7b_v6_sonnet_fp16_lora_001"),
    )
    parser.add_argument(
        "--selection-path",
        type=Path,
        default=Path("configs/minerva_7b_v6_selected_adapter.json"),
    )
    parser.add_argument(
        "--candidate-summary-path",
        type=Path,
        default=Path(
            "outputs/generations/minerva_7b_v6_candidates_001/candidate_summary.json"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/local/minerva_qlora/huggingface"),
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Serve the interface without allocating model weights.",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65_535:
        raise ValueError("--port must be between 1 and 65535")

    if args.static_only:
        print("demo | static-only mode; model generation is unavailable", flush=True)
        generator = StaticDemoGenerator()
    else:
        device = torch.device(args.device)
        print(
            "demo | loading selected Minerva 7B adapter "
            f"device={device} estimated_runtime=2m-10m_cached",
            flush=True,
        )
        generator = load_selected_sonnet_generator(
            repo_root=ROOT,
            run_dir=args.run_dir,
            selection_path=args.selection_path,
            candidate_summary_path=args.candidate_summary_path,
            cache_dir=args.cache_dir,
            device=device,
            progress=lambda message: print(f"demo | {message}", flush=True),
        )
    serve_demo(
        host=args.host,
        port=args.port,
        static_root=ROOT / "demo",
        generator=generator,
    )


if __name__ == "__main__":
    main()
