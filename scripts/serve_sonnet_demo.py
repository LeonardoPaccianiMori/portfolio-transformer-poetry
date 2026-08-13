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
    load_v7_dpo_sonnet_generator,
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
        "--v7-state-audit", type=Path,
        default=Path("artifacts/local/minerva_7b_v7_analysis/state_audit.archive.json"),
    )
    parser.add_argument(
        "--v7-adapter", type=Path,
        default=Path("artifacts/local/minerva_7b_v7_dpo/training/best_adapter.pt"),
    )
    parser.add_argument(
        "--v7-selection", type=Path,
        default=Path("artifacts/local/minerva_7b_v7_dpo/final_selection.frozen.json"),
    )
    parser.add_argument(
        "--legacy-v6", action="store_true",
        help="Load the prior V6 epoch-4 LoRA demo instead of the final V7 DPO system.",
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
            "demo | loading selected Minerva 7B system "
            f"device={device} estimated_runtime=2m-10m_cached",
            flush=True,
        )
        progress = lambda message: print(f"demo | {message}", flush=True)
        if args.legacy_v6:
            generator = load_selected_sonnet_generator(
                repo_root=ROOT, run_dir=args.run_dir,
                selection_path=args.selection_path,
                candidate_summary_path=args.candidate_summary_path,
                cache_dir=args.cache_dir, device=device, progress=progress,
            )
        else:
            generator = load_v7_dpo_sonnet_generator(
                repo_root=ROOT, state_audit_path=args.v7_state_audit,
                adapter_path=args.v7_adapter, selection_path=args.v7_selection,
                device=device, progress=progress,
            )
    serve_demo(
        host=args.host,
        port=args.port,
        static_root=ROOT / "demo",
        generator=generator,
    )


if __name__ == "__main__":
    main()
