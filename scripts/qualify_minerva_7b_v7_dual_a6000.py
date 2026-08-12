#!/usr/bin/env python3
"""Orchestrate a bounded Minerva V7 hardware qualification."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_v7_gpu_qualification import (
    qualification_paths,
    remove_temporary_proof_checkpoint,
)
from sonnet_training.minerva_7b_v7_qualification import (
    build_qualification_candidates,
    load_hardware_qualification,
    project_stage_costs,
    qualification_gate_reasons,
    select_preliminary_candidate,
)


def main() -> None:
    started = time.monotonic()
    config_path = ROOT / os.environ.get(
        "V7_QUALIFICATION_CONFIG",
        "configs/minerva_7b_v7_hardware_qualification.json",
    )
    config = load_hardware_qualification(config_path, ROOT)
    paths = qualification_paths(
        ROOT, config.get("artifact_directory", "qualification_v2")
    )
    paths["candidates"].mkdir(parents=True, exist_ok=True)
    candidates = build_qualification_candidates(config)
    progress_label = f"minerva-v7-{config['primary_profile']['profile_id']}"
    print(
        "minerva-v7-hardware-qualification | start "
        f"job={config['primary_profile']['profile_id']} "
        f"device_count={config['primary_profile']['world_size']} "
        f"total_steps={len(candidates) + 2} candidates={len(candidates)} "
        "warmup_updates=3 timed_updates=20 progress_interval=1",
        flush=True,
    )
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        output_path = paths["candidates"] / f"{candidate.candidate_id}.json"
        if output_path.is_file():
            row = json.loads(output_path.read_text(encoding="utf-8"))
            print(
                f"{progress_label} | resume "
                f"candidate={index}/{len(candidates)} id={candidate.candidate_id} "
                f"status={row['status']}",
                flush=True,
            )
        else:
            print(
                f"{progress_label} | candidate "
                f"step={index}/{len(candidates) + 2} id={candidate.candidate_id}",
                flush=True,
            )
            completed = _run_worker(
                "candidate", candidate.candidate_id, config_path, config
            )
            if completed.returncode == 0 and output_path.is_file():
                row = json.loads(output_path.read_text(encoding="utf-8"))
            else:
                row = {
                    "candidate": candidate.__dict__,
                    "status": "worker_failed",
                    "preliminary_gate_reasons": ["candidate_worker_failed"],
                    "return_code": completed.returncode,
                    "mean_tokens_per_second": 0.0,
                    "projection": None,
                    "quality_checkpoint_retained": False,
                }
                _write_json(output_path, row)
        rows.append(row)
    selected = select_preliminary_candidate(rows)
    if selected is None:
        report = _final_report(
            config=config,
            rows=rows,
            selected=None,
            proof=None,
            status=_failure_status(config, "preliminary_gates"),
        )
        _write_json(paths["final_report"], report)
        print(
            f"{progress_label} | complete status={{status}} "
            "elapsed={elapsed:.1f}s long_training_started=false output={output}".format(
                status=report["status"],
                elapsed=time.monotonic() - started,
                output=paths["final_report"],
            ),
            flush=True,
        )
        return
    candidate_id = str(selected["candidate"]["candidate_id"])
    print(
        f"{progress_label} | proof step={{step}}/{{total}} "
        "phase=validation-and-atomic-save candidate={candidate}".format(
            step=len(candidates) + 1,
            total=len(candidates) + 2,
            candidate=candidate_id,
        ),
        flush=True,
    )
    save = _run_worker("proof-save", candidate_id, config_path, config)
    if save.returncode != 0 or not paths["proof_save"].is_file():
        proof = {
            "validation_transition_passed": False,
            "atomic_checkpoint_passed": False,
            "fresh_process_resume_passed": False,
            "worker_failure": "proof_save",
        }
    else:
        print(
            f"{progress_label} | proof step={{step}}/{{total}} "
            "phase=fresh-process-resume candidate={candidate}".format(
                step=len(candidates) + 2,
                total=len(candidates) + 2,
                candidate=candidate_id,
            ),
            flush=True,
        )
        resume = _run_worker("proof-resume", candidate_id, config_path, config)
        if resume.returncode == 0 and paths["proof_resume"].is_file():
            proof = json.loads(paths["proof_resume"].read_text(encoding="utf-8"))
            if proof["fresh_process_resume_passed"] and paths["checkpoint"].is_dir():
                remove_temporary_proof_checkpoint(paths["checkpoint"])
        else:
            proof = {
                **json.loads(paths["proof_save"].read_text(encoding="utf-8")),
                "fresh_process_resume_passed": False,
                "worker_failure": "proof_resume",
            }
    final_reasons = qualification_gate_reasons(
        preliminary_reasons=selected["preliminary_gate_reasons"],
        validation_transition_passed=bool(proof["validation_transition_passed"]),
        atomic_checkpoint_passed=bool(proof["atomic_checkpoint_passed"]),
        fresh_process_resume_passed=bool(proof["fresh_process_resume_passed"]),
    )
    status = (
        "passed_qualification_long_training_still_unauthorized"
        if not final_reasons
        else _failure_status(config, "required_proofs")
    )
    report = _final_report(
        config=config,
        rows=rows,
        selected={**selected, "final_gate_reasons": list(final_reasons)},
        proof=proof,
        status=status,
    )
    _write_json(paths["final_report"], report)
    print(
        f"{progress_label} | complete status={{status}} "
        "elapsed={elapsed:.1f}s long_training_started=false output={output}".format(
            status=status,
            elapsed=time.monotonic() - started,
            output=paths["final_report"],
        ),
        flush=True,
    )


def _run_worker(
    mode: str,
    candidate_id: str,
    config_path: Path,
    config: dict[str, Any],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={config['primary_profile']['world_size']}",
            "scripts/run_minerva_7b_v7_qualification_worker.py",
            mode,
            "--candidate-id",
            candidate_id,
            "--qualification-config",
            config_path.relative_to(ROOT).as_posix(),
        ],
        cwd=ROOT,
        text=True,
        check=False,
    )


def _failure_status(config: dict[str, Any], phase: str) -> str:
    action = str(
        config["primary_profile"].get(
            "single_h100_failure_action",
            config["primary_profile"].get(
                "replicated_ddp_failure_action", "do_not_launch_long_run"
            ),
        )
    )
    return f"failed_{phase}_{action}"


def _final_report(
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    selected: dict[str, Any] | None,
    proof: dict[str, Any] | None,
    status: str,
) -> dict[str, Any]:
    stage_projection = None
    if selected is not None and float(selected.get("mean_tokens_per_second", 0.0)) > 0:
        protocol = json.loads(
            (ROOT / config["scientific_protocol"]["path"]).read_text(encoding="utf-8")
        )
        stage_projection = list(
            project_stage_costs(
                config=config,
                stages=protocol["stages"],
                measured_tokens_per_second=float(selected["mean_tokens_per_second"]),
            )
        )
    return {
        "qualification_version": config["qualification_version"],
        "status": status,
        "profile_id": config["primary_profile"]["profile_id"],
        "scientific_protocol": config["scientific_protocol"],
        "observed_machine_preflight": config["observed_machine_preflight"],
        "candidates": rows,
        "selected_candidate": selected,
        "projected_stages": stage_projection,
        "proof": proof,
        "failure_action": str(
            config["primary_profile"].get(
                "single_h100_failure_action",
                config["primary_profile"].get(
                    "replicated_ddp_failure_action", "do_not_launch_long_run"
                ),
            )
        ),
        "fallback_profile": config.get("fallback_profile"),
        "cost": config["cost"],
        "authorization": {
            "qualification_authorized": True,
            "long_training_authorized": False,
            "instance_lifecycle_action_authorized": False,
            "v7_test_access_authorized": False,
            "cache_deletion_authorized": False,
        },
        "quality_checkpoint_retained": False,
        "v7_test_accessed": False,
        "long_training_started": False,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
