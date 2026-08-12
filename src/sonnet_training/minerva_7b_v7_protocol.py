"""Validate the frozen Minerva V7 full-weight protocol and preservation index."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "minerva_7b_v7_full_weight_protocol_v1"
Progress = Callable[[str], None]


@dataclass(frozen=True)
class HardwareCandidate:
    """One 2,048-context execution candidate with an invariant global batch."""

    candidate_id: str
    local_microbatch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    execution_mode: str
    global_windows_per_update: int
    global_target_tokens_per_update: int


@dataclass(frozen=True)
class FullWeightProtocolConfig:
    """Pin protocol inputs, one local preservation index, and public evidence."""

    repo_root: Path
    protocol_path: Path
    modern_encoded_dir: Path
    preservation_index_path: Path
    json_report_path: Path
    markdown_report_path: Path


def load_full_weight_protocol(path: Path, repo_root: Path) -> dict[str, Any]:
    """Load the approved policy and verify every committed lineage hash."""

    protocol = _read_json(path)
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unexpected Minerva V7 full-weight protocol version")
    lineage = _mapping(protocol, "lineage")
    for path_key, sha_key in (
        ("stage_windows_report_path", "stage_windows_report_sha256"),
        ("sampling_policy_path", "sampling_policy_sha256"),
        ("preservation_prompts_path", "preservation_prompts_sha256"),
    ):
        artifact = _resolve(repo_root, str(lineage[path_key]))
        if _sha256(artifact) != lineage[sha_key]:
            raise ValueError(f"full-weight protocol lineage mismatch: {path_key}")
    preservation = _mapping(protocol, "preservation")
    modern_report = _resolve(
        repo_root, str(preservation["modern_data_report_path"])
    )
    if _sha256(modern_report) != preservation["modern_data_report_sha256"]:
        raise ValueError("modern preservation report lineage mismatch")
    _validate_protocol_contract(protocol, repo_root)
    return protocol


def build_hardware_candidates(
    protocol: Mapping[str, Any],
) -> tuple[HardwareCandidate, ...]:
    """Enumerate bounded hardware choices without changing the scientific batch."""

    hardware = _mapping(protocol, "hardware_qualification")
    data = _mapping(protocol, "data")
    world_size = int(hardware["world_size"])
    global_windows = int(data["global_windows_per_update"])
    context = int(data["target_tokens_per_window"])
    candidates = []
    for microbatch in hardware["permitted_local_microbatch_sizes"]:
        denominator = world_size * int(microbatch)
        if global_windows % denominator:
            raise ValueError("hardware microbatch cannot preserve the global window batch")
        accumulation = global_windows // denominator
        for checkpointing in hardware["gradient_checkpointing_modes"]:
            for execution_mode in hardware["execution_modes"]:
                mode = "gc_on" if checkpointing else "gc_off"
                execution = str(execution_mode)
                candidates.append(
                    HardwareCandidate(
                        candidate_id=(
                            f"context2048_micro{microbatch}_accum{accumulation}_"
                            f"{mode}_{execution}"
                        ),
                        local_microbatch_size=int(microbatch),
                        gradient_accumulation_steps=accumulation,
                        gradient_checkpointing=bool(checkpointing),
                        execution_mode=execution,
                        global_windows_per_update=global_windows,
                        global_target_tokens_per_update=global_windows * context,
                    )
                )
    return tuple(candidates)


def learning_rate_at_update(stage: Mapping[str, Any], update: int) -> float:
    """Return the approved linear-warmup/cosine-decay rate for one stage."""

    updates = int(stage["optimizer_updates"])
    warmup = int(stage["warmup_updates"])
    peak = float(stage["peak_learning_rate"])
    minimum = float(stage["minimum_learning_rate"])
    if update <= 0 or update > updates:
        raise ValueError("update is outside the stage")
    if update <= warmup:
        return peak * update / warmup
    progress = (update - warmup) / (updates - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum + cosine * (peak - minimum)


def candidate_passes_gates(
    *,
    stage: Mapping[str, Any],
    metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
    preservation: Mapping[str, Any],
) -> bool:
    """Apply improvement, retention, modern, and instruction gates exactly."""

    primary = str(stage["primary_validation_metric"])
    if float(metrics[primary]) > (
        float(baseline_metrics[primary])
        - float(stage["minimum_primary_loss_improvement"])
    ):
        return False
    retention = stage.get("retention_metric")
    if retention is not None:
        retention_key = str(retention)
        if float(metrics[retention_key]) > (
            float(baseline_metrics[retention_key])
            * float(stage["maximum_retention_loss_ratio"])
        ):
            return False
    return (
        float(metrics["modern_validation_loss"])
        <= float(baseline_metrics["modern_validation_loss"])
        * float(preservation["maximum_modern_loss_ratio"])
        and float(metrics["instruction_validation_loss"])
        <= float(baseline_metrics["instruction_validation_loss"])
        * float(preservation["maximum_instruction_loss_ratio"])
    )


def select_stage_checkpoint(
    *,
    stage: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    baseline_metrics: Mapping[str, float],
    preservation: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Select the lowest primary loss among candidates passing every gate."""

    qualifying = [
        row
        for row in history
        if candidate_passes_gates(
            stage=stage,
            metrics=row,
            baseline_metrics=baseline_metrics,
            preservation=preservation,
        )
    ]
    if not qualifying:
        return None
    primary = str(stage["primary_validation_metric"])
    return min(qualifying, key=lambda row: (float(row[primary]), int(row["update"])))


def abort_reasons(
    *,
    protocol: Mapping[str, Any],
    recent_updates: Sequence[Mapping[str, Any]],
    consecutive_preservation_failures: int,
    projected_or_spent_cost_usd: float,
    invariant_failure: str | None = None,
) -> tuple[str, ...]:
    """Return every frozen immediate or sustained abort reason currently met."""

    rules = _mapping(protocol, "abort_rules")
    cost = _mapping(protocol, "cost")
    reasons = []
    if invariant_failure:
        reasons.append(invariant_failure)
    if projected_or_spent_cost_usd >= float(cost["all_in_spend_ceiling"]):
        reasons.append("spend_ceiling_reached")
    if any(
        not math.isfinite(float(row[metric]))
        for row in recent_updates
        for metric in ("loss", "gradient_norm")
    ):
        reasons.append("nonfinite_training_numerics")
    required = int(rules["gradient_limit_consecutive_updates"])
    if len(recent_updates) >= required and all(
        float(row["gradient_norm"]) > float(rules["preclip_gradient_norm_limit"])
        for row in recent_updates[-required:]
    ):
        reasons.append("sustained_gradient_norm_limit")
    spike_required = int(rules["training_loss_spike_consecutive_updates"])
    reference_count = int(rules["training_loss_reference_updates"])
    if len(recent_updates) >= reference_count + spike_required:
        reference_rows = recent_updates[-(reference_count + spike_required):-spike_required]
        reference = sum(float(row["loss"]) for row in reference_rows) / len(reference_rows)
        if all(
            float(row["loss"]) > reference * float(rules["training_loss_spike_ratio"])
            for row in recent_updates[-spike_required:]
        ):
            reasons.append("sustained_training_loss_spike")
    if consecutive_preservation_failures >= int(
        rules["preservation_fail_consecutive_evaluations"]
    ):
        reasons.append("repeated_preservation_gate_failure")
    return tuple(reasons)


def project_all_in_cost(
    *,
    training_tokens: int,
    measured_tokens_per_second: float,
    hourly_rate_usd: float,
    protocol: Mapping[str, Any],
) -> dict[str, float | bool]:
    """Project runtime/cost and apply the 80%-of-ceiling launch gate."""

    if training_tokens <= 0 or measured_tokens_per_second <= 0 or hourly_rate_usd <= 0:
        raise ValueError("cost projection inputs must be positive")
    cost = _mapping(protocol, "cost")
    update_hours = training_tokens / measured_tokens_per_second / 3600
    all_in_hours = update_hours * float(cost["projection_overhead_multiplier"])
    projected_cost = all_in_hours * hourly_rate_usd
    launch_limit = float(cost["maximum_projected_all_in_cost_to_launch"])
    return {
        "update_only_hours": update_hours,
        "projected_all_in_hours": all_in_hours,
        "projected_all_in_cost_usd": projected_cost,
        "launch_limit_usd": launch_limit,
        "passes_launch_gate": projected_cost <= launch_limit,
    }


def build_modern_preservation_index(
    *,
    protocol: Mapping[str, Any],
    repo_root: Path,
    modern_encoded_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze evenly spaced PAISÀ validation spans without publishing token IDs."""

    preservation = _mapping(protocol, "preservation")
    report = _read_json(
        _resolve(repo_root, str(preservation["modern_data_report_path"]))
    )
    splits = report.get("splits")
    if not isinstance(splits, list):
        raise ValueError("modern data report is missing splits")
    split = next(
        (row for row in splits if row.get("split_id") == preservation["modern_split_id"]),
        None,
    )
    if not isinstance(split, Mapping):
        raise ValueError("modern preservation split is absent")
    if int(split["tokens"]) != int(preservation["modern_split_tokens"]):
        raise ValueError("modern preservation token count changed")
    shards = split.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("modern preservation split has no shards")
    loaded_shards = []
    for row in shards:
        shard_index = int(row["shard_index"])
        path = modern_encoded_dir / f"{split['split_id']}-{shard_index:05d}.int32.bin"
        if path.stat().st_size != int(row["bytes"]) or _sha256(path) != row["sha256"]:
            raise ValueError("modern preservation shard verification failed")
        loaded_shards.append(
            {
                "shard_index": shard_index,
                "global_token_start": int(row["global_token_start"]),
                "global_token_end": int(row["global_token_end"]),
                "token_count": int(row["token_count"]),
                "sha256": str(row["sha256"]),
            }
        )
    data = _mapping(protocol, "data")
    stride = int(data["target_tokens_per_window"])
    source_span = int(data["source_span_tokens"])
    candidate_count = (int(split["tokens"]) - 1) // stride
    selected_count = int(preservation["modern_window_count"])
    if selected_count > candidate_count:
        raise ValueError("modern preservation requests too many windows")
    selected_indices = _evenly_spaced_indices(candidate_count, selected_count)
    rows = []
    for output_index, candidate_index in enumerate(selected_indices):
        start = candidate_index * stride
        rows.append(
            {
                "index": output_index,
                "candidate_index": candidate_index,
                "global_source_start": start,
                "source_span_tokens": source_span,
                "target_tokens": stride,
                "source_slices": _shard_slices(loaded_shards, start, start + source_span),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return {
        "split_id": str(split["split_id"]),
        "split_tokens": int(split["tokens"]),
        "candidate_windows": candidate_count,
        "selected_windows": len(rows),
        "target_tokens": len(rows) * stride,
        "first_candidate_index": selected_indices[0],
        "last_candidate_index": selected_indices[-1],
        "index_sha256": _sha256(output_path),
        "index_bytes": output_path.stat().st_size,
        "index_public": False,
    }


def prepare_full_weight_protocol(
    config: FullWeightProtocolConfig,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Verify 8E, materialize its local preservation index, and publish evidence."""

    _report(progress, "validating frozen protocol and committed lineage")
    protocol = load_full_weight_protocol(config.protocol_path, config.repo_root)
    _report(progress, "hash-verifying PAISÀ validation shard")
    modern = build_modern_preservation_index(
        protocol=protocol,
        repo_root=config.repo_root,
        modern_encoded_dir=config.modern_encoded_dir,
        output_path=config.preservation_index_path,
    )
    _report(progress, "validating 2,048-context hardware candidate matrix")
    candidates = build_hardware_candidates(protocol)
    stages = protocol["stages"]
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "build_date": protocol["build_date"],
        "status": "frozen_verified_gpu_unauthorized",
        "protocol_sha256": _sha256(config.protocol_path),
        "lineage": protocol["lineage"],
        "training": {
            "context_length": protocol["data"]["context_length"],
            "global_windows_per_update": protocol["data"]["global_windows_per_update"],
            "global_target_tokens_per_update": protocol["data"]["global_target_tokens_per_update"],
            "total_windows": sum(int(row["windows"]) for row in stages),
            "total_target_tokens": sum(int(row["target_tokens"]) for row in stages),
            "total_optimizer_updates": sum(int(row["optimizer_updates"]) for row in stages),
            "stages": stages,
        },
        "validation": protocol["validation"],
        "preservation": {**protocol["preservation"], "local_index": modern},
        "checkpointing": protocol["checkpointing"],
        "analysis_snapshots": protocol["analysis_snapshots"],
        "activation_analysis_preservation": protocol[
            "activation_analysis_preservation"
        ],
        "abort_rules": protocol["abort_rules"],
        "hardware_qualification": {
            **protocol["hardware_qualification"],
            "candidate_count": len(candidates),
            "candidate_ids": [row.candidate_id for row in candidates],
        },
        "cost": protocol["cost"],
        "authorization": protocol["authorization"],
        "verification": {
            "all_lineage_hashes_match": True,
            "stage_budgets_match_checkpoint_8d": True,
            "every_stage_divides_into_complete_optimizer_updates": True,
            "held_out_modern_index_hash_verified": True,
            "instruction_prompt_count_verified": True,
            "activation_probe_contract_required_before_long_run": True,
            "v7_test_reserved_for_final_evaluation": True,
            "conditioned_material_included": False,
            "protected_v6_training_included": False,
            "gpu_work_started": False,
            "gpu_rental_started": False,
            "cache_deleted": False,
        },
    }
    _write_json(config.json_report_path, report)
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(
        render_protocol_markdown(report), encoding="utf-8"
    )
    _report(progress, f"published protocol evidence: {config.json_report_path}")
    return report


def render_protocol_markdown(report: Mapping[str, Any]) -> str:
    """Render the public 8E protocol without local token or checkpoint contents."""

    training = _mapping(report, "training")
    lines = [
        "# Minerva 7B V7 Full-Weight Training Protocol",
        "",
        "Checkpoint 8E freezes the scientific training, validation, preservation,",
        "checkpoint/resume, abort, hardware-qualification, and cost contracts. It",
        "does not authorize GPU rental, benchmarking, or the long run.",
        "",
        "## Frozen stages",
        "",
        "| Stage | Windows | Target tokens | Updates | Peak LR | Minimum LR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage in training["stages"]:
        lines.append(
            f"| {stage['stage_id']} | {int(stage['windows']):,} | "
            f"{int(stage['target_tokens']):,} | {int(stage['optimizer_updates']):,} | "
            f"{float(stage['peak_learning_rate']):.1e} | "
            f"{float(stage['minimum_learning_rate']):.1e} |"
        )
    lines.extend(
        [
            "",
            f"Every update consumes {int(training['global_windows_per_update'])} "
            f"complete context-2,048 windows / "
            f"{int(training['global_target_tokens_per_update']):,} target tokens. "
            "Optimizer and scheduler state reset at each stage; model weights do not.",
            "",
            "## Promotion and preservation",
            "",
            "A stage promotes only its lowest primary validation-loss checkpoint that",
            "also passes its minimum-improvement rule, any earlier-domain retention",
            "gate, the held-out PAISÀ modern-loss gate (maximum 1.05x the untouched",
            "parent), and the 12-prompt instruction-loss gate (maximum 1.10x). Stage 3",
            "uses V7 validation for selection; the 106-window V7 test remains unopened",
            "until the final stage-3 checkpoint has been selected.",
            "",
            "The modern gate uses 128 deterministic held-out PAISÀ windows. Its local",
            f"index hash is `{report['preservation']['local_index']['index_sha256']}`; "
            "the index and token shard are not public repository data.",
            "",
            "## Checkpoints and later change analysis",
            "",
            "Resume checkpoints are atomic and include model, optimizer, scheduler, RNG,",
            "sampler position, counters, histories, hashes, software, and topology. Two",
            "resume generations are retained, and a fresh-process exact-resume proof is",
            "required before the long run.",
            "",
            "For a later study of what each adaptation stage changed, retain model-only",
            "BF16 snapshots at each stage midpoint and validation-selected endpoint. The",
            "untouched parent is referenced by its pinned published revision. Six new",
            "snapshots project to about 88.8 GB before filesystem/compression overhead.",
            "",
            "Activation changes will be measured post hoc in evaluation mode on a frozen",
            "held-out probe suite. Exact token IDs, positions, tokenizer/model hashes, and",
            "extraction settings must be frozen before training. The planned comparisons",
            "include layerwise CKA, cosine/norm shifts, effective rank, domain probes,",
            "attention summaries, and next-token distribution shifts. Raw tensors stay",
            "local; training batches are not continuously archived.",
            "",
            "## Hardware and cost boundary",
            "",
            f"Qualification requires two matching H100 80 GB SXM GPUs with NVLink, "
            f"native BF16, at least 100 GB/s measured NCCL all-reduce bandwidth, and "
            f"at least 8 GiB peak-reserved headroom. The bounded matrix contains "
            f"{int(report['hardware_qualification']['candidate_count'])} candidates; "
            "all preserve the same 16-window global batch.",
            "",
            "The all-in ceiling is $60. A long-run launch requires a measured all-in",
            "projection no greater than $48, leaving 20% contingency. Changing either",
            "number requires explicit user approval. Instance lifecycle actions also",
            "remain user-controlled.",
            "",
            "## Authorization boundary",
            "",
            "- Protocol design approved: `true`.",
            "- GPU benchmark authorized: `false`.",
            "- GPU rental authorized: `false`.",
            "- Long training authorized: `false`.",
            "- Cache deletion authorized: `false`.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_protocol_contract(protocol: Mapping[str, Any], repo_root: Path) -> None:
    data = _mapping(protocol, "data")
    if (
        int(data["context_length"]) != 2048
        or int(data["source_span_tokens"]) != 2049
        or int(data["target_tokens_per_window"]) != 2048
        or int(data["global_windows_per_update"]) != 16
        or int(data["global_target_tokens_per_update"]) != 32768
    ):
        raise ValueError("full-weight data contract changed")
    windows_report = _read_json(
        _resolve(repo_root, str(protocol["lineage"]["stage_windows_report_path"]))
    )
    expected_stages = {row["stage_id"]: row for row in windows_report["training"]["stages"]}
    stages = protocol.get("stages")
    if not isinstance(stages, list) or len(stages) != 3:
        raise ValueError("full-weight protocol must contain exactly three stages")
    for stage in stages:
        expected = expected_stages.get(stage["stage_id"])
        if expected is None:
            raise ValueError("full-weight protocol has an unknown stage")
        if int(stage["windows"]) != int(expected["windows"]):
            raise ValueError("full-weight stage window count differs from checkpoint 8D")
        if int(stage["target_tokens"]) != int(expected["target_tokens"]):
            raise ValueError("full-weight stage token budget differs from checkpoint 8D")
        if int(stage["windows"]) % int(data["global_windows_per_update"]):
            raise ValueError("full-weight stage would require a partial optimizer update")
        if int(stage["optimizer_updates"]) != int(stage["windows"]) // 16:
            raise ValueError("full-weight stage optimizer-update count is inconsistent")
        if not 0 < int(stage["warmup_updates"]) < int(stage["optimizer_updates"]):
            raise ValueError("full-weight warmup is outside the stage")
        if int(stage["progress_interval_updates"]) <= 0:
            raise ValueError("full-weight progress interval must be positive")
        if int(stage["early_stopping_eligible_after_update"]) != math.ceil(
            int(stage["optimizer_updates"]) / 2
        ):
            raise ValueError("early stopping cannot begin before the stage midpoint")
        if not (
            0 < float(stage["minimum_learning_rate"])
            < float(stage["peak_learning_rate"])
        ):
            raise ValueError("full-weight learning-rate bounds are invalid")
    prompts = json.loads(
        _resolve(repo_root, str(protocol["lineage"]["preservation_prompts_path"])).read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(prompts, list) or len(prompts) != int(
        protocol["preservation"]["instruction_prompt_count"]
    ):
        raise ValueError("instruction preservation prompt count changed")
    snapshots = _mapping(protocol, "analysis_snapshots")
    midpoint_updates = _mapping(snapshots, "midpoint_updates")
    for stage in stages:
        expected_midpoint = math.ceil(int(stage["optimizer_updates"]) / 2)
        if int(midpoint_updates[stage["stage_id"]]) != expected_midpoint:
            raise ValueError("analysis midpoint is not the stage midpoint")
    checkpointing = _mapping(protocol, "checkpointing")
    if int(checkpointing["minimum_free_host_scratch_gib"]) < 300:
        raise ValueError("host scratch is too small for atomic checkpoint retention")
    if int(checkpointing["minimum_free_local_retrieval_gib"]) < 200:
        raise ValueError("local retrieval storage is too small for analysis snapshots")
    activation = _mapping(protocol, "activation_analysis_preservation")
    if activation.get("required_before_long_run") is not True:
        raise ValueError("activation probe contract must be frozen before training")
    domains = activation.get("probe_domains")
    if not isinstance(domains, list) or set(domains) != {
        "modern_instruction",
        "historical_general",
        "historical_non_sonnet_poetry",
        "standard_sonnet",
    }:
        raise ValueError("activation probe domains changed")
    if not (
        8 <= int(activation["minimum_probes_per_domain"])
        <= int(activation["maximum_probes_per_domain"])
        <= 16
    ):
        raise ValueError("activation probe bounds are invalid")
    cost = _mapping(protocol, "cost")
    if float(cost["all_in_spend_ceiling"]) != 60.0:
        raise ValueError("full-weight spend ceiling changed")
    if float(cost["maximum_projected_all_in_cost_to_launch"]) != 48.0:
        raise ValueError("full-weight launch projection gate changed")
    authorization = _mapping(protocol, "authorization")
    if any(
        authorization[key]
        for key in ("gpu_benchmark_authorized", "gpu_rental_authorized", "long_training_authorized", "cache_deletion_authorized")
    ):
        raise ValueError("checkpoint 8E cannot authorize GPU or destructive work")


def _evenly_spaced_indices(candidate_count: int, selected_count: int) -> tuple[int, ...]:
    if candidate_count <= 0 or selected_count <= 0 or selected_count > candidate_count:
        raise ValueError("invalid evenly spaced index request")
    if selected_count == 1:
        return (0,)
    values = tuple(
        round(index * (candidate_count - 1) / (selected_count - 1))
        for index in range(selected_count)
    )
    if len(set(values)) != selected_count:
        raise ValueError("evenly spaced selection produced duplicate indexes")
    return values


def _shard_slices(
    shards: Sequence[Mapping[str, Any]], start: int, end: int
) -> list[dict[str, int]]:
    slices = []
    for shard in shards:
        shard_start = int(shard["global_token_start"])
        shard_end = int(shard["global_token_end"])
        if shard_end <= start:
            continue
        if shard_start >= end:
            break
        overlap_start = max(start, shard_start)
        overlap_end = min(end, shard_end)
        slices.append(
            {
                "shard_index": int(shard["shard_index"]),
                "token_offset": overlap_start - shard_start,
                "token_count": overlap_end - overlap_start,
            }
        )
    if sum(row["token_count"] for row in slices) != end - start:
        raise ValueError("preservation span crosses missing shard coverage")
    return slices


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"expected object at {key}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _report(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)
