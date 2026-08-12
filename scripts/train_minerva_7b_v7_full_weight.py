#!/usr/bin/env python3
"""Preflight, inspect, or user-launch one Minerva V7 single-H100 stage."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_v7_execution import (
    V7ExecutionConfig,
    build_execution_context,
)
from sonnet_training.minerva_7b_v7_launch import (
    load_single_h100_launch_config,
    stage_launch,
    stage_status,
    validate_stage_boundary,
)
from sonnet_training.minerva_7b_v7_trainer import train_minerva_7b_v7_full_weight


LAUNCH_PATH = ROOT / "configs/minerva_7b_v7_single_h100_launch.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exactly one user-controlled Minerva V7 stage."
    )
    parser.add_argument(
        "--stage",
        choices=(
            "stage_1_historical_general",
            "stage_2_non_sonnet_poetry",
            "stage_3_sonnets",
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="Verify configuration, data lineage, stage order, and disk only; no model/GPU.",
    )
    mode.add_argument(
        "--status", action="store_true", help="Print stage/boundary/resume status only."
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Verified atomic resume directory for this same stage.",
    )
    return parser.parse_args()


def execution_config() -> V7ExecutionConfig:
    return V7ExecutionConfig(
        repo_root=ROOT,
        execution_path=ROOT / "configs/minerva_7b_v7_execution.json",
        encoded_dir=ROOT / "data/local/minerva_7b_v7/encoded",
        window_index_dir=ROOT / "data/local/minerva_7b_v7/window_indexes",
        modern_encoded_dir=ROOT / "data/local/minerva_7b_full_weight/encoded",
        modern_index_path=ROOT
        / "data/local/minerva_7b_v7/modern_preservation_validation_v1.jsonl",
    )


def preflight(stage_id: str, launch: dict) -> dict:
    """Verify all CPU-visible launch prerequisites without importing model weights."""

    context = build_execution_context(execution_config())
    selected = stage_launch(launch, stage_id)
    run_dir = ROOT / context["execution"]["local_paths"]["run_dir"]
    boundary = run_dir / "stage_boundaries" / f"{stage_id}_selected"
    marker = run_dir / "stage_runs" / f"{stage_id}.json"
    if boundary.exists():
        raise RuntimeError(f"stage is already complete: {stage_id}")
    if marker.exists():
        raise RuntimeError("stage already started; use --status and --resume-from")
    required_boundary = selected["required_boundary"]
    preceding_boundary = None
    if required_boundary is not None:
        stage_ids = [row["stage_id"] for row in launch["stage_launches"]]
        previous_stage_id = stage_ids[stage_ids.index(stage_id) - 1]
        preceding_boundary = ROOT / required_boundary
        validate_stage_boundary(
            path=preceding_boundary,
            expected_stage_id=previous_stage_id,
            launch=launch,
        )
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    remaining_stages = len(launch["stage_launches"]) - [
        row["stage_id"] for row in launch["stage_launches"]
    ].index(stage_id)
    estimated_remaining_evidence_gib = remaining_stages * 2 * 14.8 + 56.5
    if free_gib < estimated_remaining_evidence_gib:
        raise RuntimeError(
            "insufficient free disk for retained snapshots plus two resume generations"
        )
    return {
        "status": "passed_no_model_or_training_started",
        "stage_id": stage_id,
        "execution_config_sha256": launch["lineage"]["execution_config_sha256"],
        "protocol_sha256": launch["lineage"]["protocol_sha256"],
        "qualification_report_sha256": launch["lineage"][
            "qualification_report_sha256"
        ],
        "candidate_id": launch["qualified_runtime"]["candidate_id"],
        "free_disk_gib": free_gib,
        "estimated_remaining_evidence_gib": estimated_remaining_evidence_gib,
        "preceding_boundary": str(preceding_boundary) if preceding_boundary else None,
        "v7_test_accessed": False,
        "training_started": False,
    }


def main() -> None:
    args = parse_args()
    launch = load_single_h100_launch_config(LAUNCH_PATH, ROOT)
    if args.status:
        print(json.dumps(stage_status(repo_root=ROOT, launch=launch), indent=2), flush=True)
        return
    if args.stage is None:
        raise SystemExit("--stage is required unless --status is used")
    if args.preflight:
        print(json.dumps(preflight(args.stage, launch), indent=2), flush=True)
        return

    rank = int(os.environ.get("RANK", "0"))
    if "LOCAL_RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("training must be launched with torchrun --nproc_per_node=1")
    runtime = launch["qualified_runtime"]
    os.environ["V7_LOCAL_MICROBATCH_SIZE"] = str(runtime["local_microbatch_size"])
    os.environ["V7_GRADIENT_ACCUMULATION_STEPS"] = str(
        runtime["gradient_accumulation_steps"]
    )
    os.environ["V7_GRADIENT_CHECKPOINTING"] = str(
        runtime["gradient_checkpointing"]
    ).lower()
    os.environ["V7_EXECUTION_MODE"] = str(runtime["execution_mode"])
    os.environ["V7_HOURLY_RATE_USD"] = str(launch["cost"]["hourly_rate_usd"])
    started = time.monotonic()
    selected = stage_launch(launch, args.stage)

    def progress(message: str) -> None:
        if rank == 0:
            print(
                f"minerva-v7-training | {message} | "
                f"job_elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    if rank == 0:
        print(
            "minerva-v7-training | start "
            f"job={args.stage} device=1xh100-80gb "
            f"total_steps={selected['optimizer_updates']} "
            "progress_interval=stage-specific context=2048 tokens_per_update=32768 "
            f"projected_all_in_hours={selected['projected_all_in_hours']:.2f} "
            f"projected_cost_usd={selected['projected_all_in_cost_usd']:.2f}",
            flush=True,
        )
    launch_sha256 = launch["_launch_config_sha256"]
    result = train_minerva_7b_v7_full_weight(
        repo_root=ROOT,
        execution_config=execution_config(),
        launch=launch,
        launch_config_sha256=launch_sha256,
        requested_stage_id=args.stage,
        resume_from_checkpoint=(
            args.resume_from.resolve() if args.resume_from is not None else None
        ),
        progress=progress,
    )
    if rank == 0:
        print(
            f"minerva-v7-training | complete status={result['status']} "
            f"stage={result['stage_id']} elapsed={time.monotonic() - started:.1f}s "
            f"boundary={result['boundary_path']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
