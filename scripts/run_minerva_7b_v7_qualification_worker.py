#!/usr/bin/env python3
"""Run one isolated distributed worker for Minerva V7 qualification."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_v7_gpu_qualification import (
    qualification_paths,
    run_candidate_worker,
    run_proof_resume_worker,
    run_proof_save_worker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("candidate", "proof-save", "proof-resume")
    )
    parser.add_argument("--candidate-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = qualification_paths(ROOT)
    qualification = ROOT / "configs/minerva_7b_v7_hardware_qualification.json"
    rank = int(os.environ.get("RANK", "0"))
    started = time.monotonic()

    def progress(message: str) -> None:
        if rank == 0:
            print(
                f"minerva-v7-qualification-worker | {message} | "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    if rank == 0:
        print(
            "minerva-v7-qualification-worker | start "
            f"mode={args.mode} candidate={args.candidate_id} device=2xrtx-a6000 "
            "context=2048 progress_interval=5",
            flush=True,
        )
    if args.mode == "candidate":
        result = run_candidate_worker(
            repo_root=ROOT,
            qualification_path=qualification,
            candidate_id=args.candidate_id,
            output_path=paths["candidates"] / f"{args.candidate_id}.json",
            progress=progress,
        )
    elif args.mode == "proof-save":
        result = run_proof_save_worker(
            repo_root=ROOT,
            qualification_path=qualification,
            candidate_id=args.candidate_id,
            output_path=paths["proof_save"],
            checkpoint_path=paths["checkpoint"],
            progress=progress,
        )
    else:
        result = run_proof_resume_worker(
            repo_root=ROOT,
            qualification_path=qualification,
            candidate_id=args.candidate_id,
            save_report_path=paths["proof_save"],
            output_path=paths["proof_resume"],
            checkpoint_path=paths["checkpoint"],
            progress=progress,
        )
    if rank == 0:
        print(
            "minerva-v7-qualification-worker | complete "
            f"mode={args.mode} status={result['status']} "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
