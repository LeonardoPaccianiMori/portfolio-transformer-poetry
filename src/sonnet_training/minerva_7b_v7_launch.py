"""Fail-closed single-H100 launch and stage-lineage contracts for Minerva V7."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sonnet_training.minerva_7b_v7_execution import verify_checkpoint_directory


LAUNCH_VERSION = "minerva_7b_v7_single_h100_stage_launch_v1"


def load_single_h100_launch_config(path: Path, repo_root: Path) -> dict[str, Any]:
    """Load the launch contract and verify every immutable qualification input."""

    launch = json.loads(path.read_text(encoding="utf-8"))
    if launch.get("launch_version") != LAUNCH_VERSION:
        raise ValueError("unexpected Minerva V7 single-H100 launch version")
    lineage = _mapping(launch, "lineage")
    for prefix in ("execution_config", "protocol", "qualification_report"):
        source = repo_root / str(lineage[f"{prefix}_path"])
        if not source.is_file() or _sha256(source) != lineage[f"{prefix}_sha256"]:
            raise ValueError(f"single-H100 launch lineage mismatch: {prefix}")

    execution = json.loads(
        (repo_root / str(lineage["execution_config_path"])).read_text(encoding="utf-8")
    )
    protocol = json.loads(
        (repo_root / str(lineage["protocol_path"])).read_text(encoding="utf-8")
    )
    qualification = json.loads(
        (repo_root / str(lineage["qualification_report_path"])).read_text(
            encoding="utf-8"
        )
    )
    runtime = _mapping(launch, "qualified_runtime")
    selected = _mapping(qualification, "selected_candidate")
    candidate_fields = (
        "candidate_id",
        "local_microbatch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "execution_mode",
    )
    if any(runtime[key] != selected[key] for key in candidate_fields):
        raise ValueError("launch runtime differs from the passed H100 qualification")
    if qualification.get("result") != "passed_long_training_still_unauthorized":
        raise ValueError("single-H100 qualification did not pass")
    proofs = _mapping(qualification, "state_transition_proofs")
    if not proofs or not all(bool(value) for value in proofs.values()):
        raise ValueError("single-H100 qualification proof set is incomplete")
    if qualification.get("training_started") or qualification.get("v7_test_accessed"):
        raise ValueError("qualification evidence crossed a prohibited boundary")

    global_windows = int(protocol["data"]["global_windows_per_update"])
    if int(runtime["world_size"]) != 1 or int(runtime["gpu_count"]) != 1:
        raise ValueError("launch contract must use exactly one H100")
    if (
        int(runtime["local_microbatch_size"])
        * int(runtime["gradient_accumulation_steps"])
        * int(runtime["world_size"])
        != global_windows
    ):
        raise ValueError("launch runtime does not preserve the scientific batch")
    authorization = _mapping(launch, "authorization")
    if not authorization.get("long_training_authorized"):
        raise PermissionError("single-H100 long training is not authorized")
    if authorization.get("launch_owner") != "user" or authorization.get(
        "assistant_may_launch_training"
    ):
        raise PermissionError("the launch contract does not preserve user ownership")
    if any(
        authorization.get(key)
        for key in (
            "v7_test_access_authorized",
            "instance_lifecycle_action_authorized",
            "cache_deletion_authorized",
        )
    ):
        raise PermissionError("launch contract authorizes a prohibited side effect")

    protocol_stages = {row["stage_id"]: row for row in protocol["stages"]}
    launch_stages = {row["stage_id"]: row for row in launch["stage_launches"]}
    if tuple(launch_stages) != tuple(protocol_stages):
        raise ValueError("launch stages differ from the frozen protocol")
    for stage_id, row in launch_stages.items():
        frozen = protocol_stages[stage_id]
        if int(row["optimizer_updates"]) != int(frozen["optimizer_updates"]):
            raise ValueError(f"launch update count differs for {stage_id}")
        if int(row["target_tokens"]) != int(frozen["target_tokens"]):
            raise ValueError(f"launch token count differs for {stage_id}")
    launch["_launch_config_sha256"] = _sha256(path)
    return launch


def stage_launch(launch: Mapping[str, Any], stage_id: str) -> Mapping[str, Any]:
    """Return one explicitly approved stage or reject an unknown scope."""

    for row in launch["stage_launches"]:
        if row["stage_id"] == stage_id:
            return row
    raise ValueError(f"stage is outside the single-H100 launch contract: {stage_id}")


def validate_stage_boundary(
    *,
    path: Path,
    expected_stage_id: str,
    launch: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify that a next-stage parent is the selected prior-stage endpoint."""

    manifest = verify_checkpoint_directory(path)
    metadata = _mapping(manifest, "metadata")
    if metadata.get("artifact_type") != "model_only_analysis_snapshot":
        raise ValueError("stage boundary is not a model-only analysis snapshot")
    if metadata.get("snapshot_role") != "validation_selected_endpoint":
        raise ValueError("stage boundary is not a validation-selected endpoint")
    if metadata.get("stage_id") != expected_stage_id:
        raise ValueError("stage boundary belongs to the wrong preceding stage")
    if metadata.get("protocol_sha256") != launch["lineage"]["protocol_sha256"]:
        raise ValueError("stage boundary protocol lineage differs")
    required = (
        "selected_metrics",
        "parent_baseline_metrics",
        "validation_history",
        "launch_config_sha256",
        "preceding_model_identity_sha256",
        "source_candidate_manifest_sha256",
    )
    if any(key not in metadata for key in required):
        raise ValueError("stage boundary continuation metadata is incomplete")
    if metadata["launch_config_sha256"] != launch["_launch_config_sha256"]:
        raise ValueError("stage boundary launch lineage differs")
    return manifest


def stage_status(*, repo_root: Path, launch: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize markers, selected boundaries, and latest resume checkpoints."""

    execution = json.loads(
        (repo_root / launch["lineage"]["execution_config_path"]).read_text(
            encoding="utf-8"
        )
    )
    run_dir = repo_root / execution["local_paths"]["run_dir"]
    resume_rows = []
    for path in sorted((run_dir / "resume").glob("resume_*")):
        try:
            manifest = verify_checkpoint_directory(path)
        except (OSError, ValueError):
            continue
        resume_rows.append(
            {
                "path": str(path),
                "stage_id": manifest["metadata"].get("stage_id"),
                "stage_update": manifest["metadata"].get("stage_update"),
                "global_update": manifest["metadata"].get("global_update"),
            }
        )
    stages = []
    for row in launch["stage_launches"]:
        stage_id = row["stage_id"]
        marker = run_dir / "stage_runs" / f"{stage_id}.json"
        boundary = run_dir / "stage_boundaries" / f"{stage_id}_selected"
        stages.append(
            {
                "stage_id": stage_id,
                "started": marker.is_file(),
                "complete": boundary.is_dir(),
                "boundary": str(boundary) if boundary.is_dir() else None,
                "latest_resume": next(
                    (
                        item
                        for item in reversed(resume_rows)
                        if item["stage_id"] == stage_id
                    ),
                    None,
                ),
            }
        )
    return {"launch_version": launch["launch_version"], "run_dir": str(run_dir), "stages": stages}


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"launch artifact is missing mapping: {key}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
