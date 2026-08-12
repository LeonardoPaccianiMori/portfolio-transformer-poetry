"""Profile-driven, fail-closed qualification policy for Minerva V7 hardware."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


QUALIFICATION_VERSIONS = {
    "minerva_7b_v7_hardware_qualification_v2",
    "minerva_7b_v7_single_h100_qualification_v1",
}


@dataclass(frozen=True)
class QualificationCandidate:
    """One runtime choice preserving the frozen global scientific batch."""

    candidate_id: str
    local_microbatch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    execution_mode: str
    global_windows_per_update: int
    global_target_tokens_per_update: int


def load_hardware_qualification(path: Path, repo_root: Path) -> dict[str, Any]:
    """Load the approved machine profile and verify its frozen scientific inputs."""

    config = _read_json(path)
    if config.get("qualification_version") not in QUALIFICATION_VERSIONS:
        raise ValueError("unexpected Minerva V7 hardware qualification version")
    scientific = _mapping(config, "scientific_protocol")
    for path_key, hash_key in (
        ("path", "sha256"),
        ("execution_path", "execution_sha256"),
    ):
        artifact = _resolve(repo_root, str(scientific[path_key]))
        if _sha256(artifact) != scientific[hash_key]:
            raise ValueError(f"qualification scientific lineage mismatch: {path_key}")
    protocol = _read_json(_resolve(repo_root, str(scientific["path"])))
    execution = _read_json(_resolve(repo_root, str(scientific["execution_path"])))
    data = _mapping(protocol, "data")
    expected = {
        "context_length": int(data["context_length"]),
        "global_windows_per_update": int(data["global_windows_per_update"]),
        "global_target_tokens_per_update": int(
            data["global_target_tokens_per_update"]
        ),
        "window_content_identity_sha256": str(
            protocol["lineage"]["window_index_content_identity_sha256"]
        ),
        "training_target_tokens": sum(
            int(stage["target_tokens"]) for stage in protocol["stages"]
        ),
    }
    for key, value in expected.items():
        if scientific.get(key) != value:
            raise ValueError(f"qualification changed frozen scientific field: {key}")
    if execution["lineage"]["window_content_identity_sha256"] != expected[
        "window_content_identity_sha256"
    ]:
        raise ValueError("qualification execution/window lineage differs")
    _validate_contract(config)
    return config


def build_qualification_candidates(
    config: Mapping[str, Any],
) -> tuple[QualificationCandidate, ...]:
    """Enumerate the profile's bounded matrix with an invariant global batch."""

    profile = _mapping(config, "primary_profile")
    scientific = _mapping(config, "scientific_protocol")
    world_size = int(profile["world_size"])
    global_windows = int(scientific["global_windows_per_update"])
    target_tokens = int(scientific["global_target_tokens_per_update"])
    rows = []
    for microbatch in profile["permitted_local_microbatch_sizes"]:
        denominator = world_size * int(microbatch)
        if global_windows % denominator:
            raise ValueError("qualification candidate cannot preserve global batch")
        accumulation = global_windows // denominator
        for checkpointing in profile["gradient_checkpointing_modes"]:
            for execution_mode in profile["execution_modes"]:
                rows.append(
                    QualificationCandidate(
                        candidate_id=(
                            f"{profile['candidate_id_prefix']}_context2048_"
                            f"micro{microbatch}_accum{accumulation}_"
                            f"gc_{'on' if checkpointing else 'off'}_{execution_mode}"
                        ),
                        local_microbatch_size=int(microbatch),
                        gradient_accumulation_steps=accumulation,
                        gradient_checkpointing=bool(checkpointing),
                        execution_mode=str(execution_mode),
                        global_windows_per_update=global_windows,
                        global_target_tokens_per_update=target_tokens,
                    )
                )
    return tuple(rows)


def candidate_by_id(
    config: Mapping[str, Any], candidate_id: str
) -> QualificationCandidate:
    matches = [
        row for row in build_qualification_candidates(config)
        if row.candidate_id == candidate_id
    ]
    if len(matches) != 1:
        raise KeyError(f"unknown qualification candidate: {candidate_id}")
    return matches[0]


def project_candidate_cost(
    *, config: Mapping[str, Any], measured_tokens_per_second: float
) -> dict[str, float | bool]:
    """Project the complete run using the unchanged contingency and launch gate."""

    if measured_tokens_per_second <= 0 or not math.isfinite(measured_tokens_per_second):
        raise ValueError("measured throughput must be finite and positive")
    scientific = _mapping(config, "scientific_protocol")
    cost = _mapping(config, "cost")
    update_hours = (
        int(scientific["training_target_tokens"])
        / measured_tokens_per_second
        / 3600
    )
    all_in_hours = update_hours * float(cost["projection_overhead_multiplier"])
    projected = all_in_hours * float(cost["hourly_rate_usd"])
    launch_limit = float(cost["maximum_projected_all_in_cost_to_launch_usd"])
    return {
        "update_only_hours": update_hours,
        "projected_all_in_hours": all_in_hours,
        "projected_all_in_cost_usd": projected,
        "launch_limit_usd": launch_limit,
        "passes_launch_gate": projected <= launch_limit,
    }


def project_stage_costs(
    *,
    config: Mapping[str, Any],
    stages: Sequence[Mapping[str, Any]],
    measured_tokens_per_second: float,
) -> tuple[dict[str, float | int | str], ...]:
    """Allocate the measured update time and contingency across frozen stages."""

    if measured_tokens_per_second <= 0 or not math.isfinite(measured_tokens_per_second):
        raise ValueError("measured throughput must be finite and positive")
    cost = _mapping(config, "cost")
    overhead = float(cost["projection_overhead_multiplier"])
    hourly_rate = float(cost["hourly_rate_usd"])
    rows = []
    for stage in stages:
        target_tokens = int(stage["target_tokens"])
        update_hours = target_tokens / measured_tokens_per_second / 3600
        all_in_hours = update_hours * overhead
        rows.append(
            {
                "stage_id": str(stage["stage_id"]),
                "target_tokens": target_tokens,
                "optimizer_updates": int(stage["optimizer_updates"]),
                "update_only_hours": update_hours,
                "projected_all_in_hours": all_in_hours,
                "projected_all_in_cost_usd": all_in_hours * hourly_rate,
            }
        )
    return tuple(rows)


def preliminary_gate_reasons(
    *,
    config: Mapping[str, Any],
    rank_metrics: Sequence[Mapping[str, Any]],
    hardware: Mapping[str, Any],
    communication: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return explicit failure reasons for one bounded performance candidate."""

    profile = _mapping(config, "primary_profile")
    reasons = []
    if len(rank_metrics) != int(profile["world_size"]):
        reasons.append("rank_metrics_incomplete")
    for row in rank_metrics:
        for metric in ("mean_loss", "mean_gradient_norm", "tokens_per_second"):
            if not math.isfinite(float(row.get(metric, math.nan))):
                reasons.append("nonfinite_training_numerics")
                break
        if float(row.get("reserved_headroom_mib", -math.inf)) < float(
            profile["minimum_peak_reserved_headroom_mib"]
        ):
            reasons.append("insufficient_peak_reserved_headroom")
        if float(row.get("reserved_memory_growth_mib", math.inf)) > float(
            profile["maximum_reserved_memory_growth_mib"]
        ):
            reasons.append("progressive_reserved_memory_growth")
    if not bool(hardware.get("profile_passed")):
        reasons.append("hardware_profile_failed")
    if bool(profile.get("communication_measurement_required", True)) and float(
        communication.get("algorithmic_gigabytes_per_second", 0.0)
    ) < float(profile["minimum_measured_nccl_algorithmic_gigabytes_per_second"]):
        reasons.append("nccl_bandwidth_below_profile_floor")
    if not bool(projection.get("passes_launch_gate")):
        reasons.append("projected_cost_exceeds_launch_gate")
    return tuple(dict.fromkeys(reasons))


def qualification_gate_reasons(
    *,
    preliminary_reasons: Sequence[str],
    validation_transition_passed: bool,
    atomic_checkpoint_passed: bool,
    fresh_process_resume_passed: bool,
) -> tuple[str, ...]:
    """Add the required state-transition proofs to preliminary gate results."""

    reasons = list(preliminary_reasons)
    if not validation_transition_passed:
        reasons.append("validation_transition_failed")
    if not atomic_checkpoint_passed:
        reasons.append("atomic_checkpoint_failed")
    if not fresh_process_resume_passed:
        reasons.append("fresh_process_resume_failed")
    return tuple(dict.fromkeys(reasons))


def select_preliminary_candidate(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Select the fastest gate-passing candidate; never select failed rows."""

    passing = [row for row in rows if not row.get("preliminary_gate_reasons")]
    if not passing:
        return None
    return max(
        passing,
        key=lambda row: (
            float(row["mean_tokens_per_second"]),
            -float(row["projection"]["projected_all_in_cost_usd"]),
        ),
    )


def _validate_contract(config: Mapping[str, Any]) -> None:
    profile = _mapping(config, "primary_profile")
    profile_id = profile.get("profile_id")
    if profile_id not in {"dual_rtx_a6000_ddp", "single_h100_sxm"}:
        raise ValueError("unknown qualification profile")
    expected_world_size = 2 if profile_id == "dual_rtx_a6000_ddp" else 1
    if int(profile["world_size"]) != expected_world_size:
        raise ValueError("qualification world size differs from its profile")
    memory_floor = 48000 if expected_world_size == 2 else 76800
    if int(profile["minimum_memory_mib_per_gpu"]) < memory_floor:
        raise ValueError("qualification memory floor is too low")
    if int(profile["minimum_peak_reserved_headroom_mib"]) != 8192:
        raise ValueError("qualification memory margin changed")
    if int(profile["minimum_free_host_scratch_gib"]) < 300:
        raise ValueError("qualification scratch floor is too low")
    expected_microbatches = [1, 2] if expected_world_size == 2 else [1, 2, 4]
    if profile["permitted_local_microbatch_sizes"] != expected_microbatches:
        raise ValueError("qualification candidate microbatches changed")
    expected_candidates = 8 if expected_world_size == 2 else 12
    if len(build_qualification_candidates(config)) != expected_candidates:
        raise ValueError("qualification candidate count changed")
    cost = _mapping(config, "cost")
    if float(cost["maximum_projected_all_in_cost_to_launch_usd"]) != 48.0:
        raise ValueError("qualification launch gate changed")
    if float(cost["absolute_spend_ceiling_usd"]) != 60.0:
        raise ValueError("qualification spend ceiling changed")
    authorization = _mapping(config, "authorization")
    if not bool(authorization["gpu_qualification_authorized"]):
        raise ValueError("approved checkpoint must authorize bounded qualification")
    if any(
        bool(authorization[key])
        for key in (
            "long_training_authorized",
            "instance_lifecycle_action_authorized",
            "v7_test_access_authorized",
            "cache_deletion_authorized",
        )
    ):
        raise ValueError("qualification may not authorize training or lifecycle work")


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"missing qualification mapping: {key}")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
