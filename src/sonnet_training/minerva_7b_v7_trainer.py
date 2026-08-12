"""Full-weight Minerva V7 trainer primitives and guarded GPU entry points."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from sonnet_training.minerva_7b_model_audit import audit_full_weight_model
from sonnet_training.minerva_7b_v7_execution import (
    FrozenWindowReader,
    Int32ShardStore,
    V7ExecutionConfig,
    atomic_install_checkpoint_writer,
    build_execution_context,
    optimizer_state_inventory,
    make_update_telemetry,
    rotate_resume_checkpoints,
    summarize_named_tensors,
    verify_checkpoint_directory,
)
from sonnet_training.minerva_7b_v7_protocol import (
    abort_reasons,
    build_hardware_candidates,
    candidate_passes_gates,
    learning_rate_at_update,
    select_stage_checkpoint,
)
from sonnet_training.minerva_7b_v7_launch import validate_stage_boundary


TRAINER_VERSION = "minerva_7b_v7_full_weight_trainer_v1"
QUALIFICATION_VERSION = "minerva_7b_v7_dual_h100_qualification_v1"


@dataclass(frozen=True)
class RuntimeCandidate:
    """One hardware-only execution choice preserving the scientific batch."""

    local_microbatch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    execution_mode: str


def shifted_causal_loss(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Score all 2,048 frozen targets without asking Transformers to shift labels."""

    if logits.ndim != 3 or target_ids.ndim != 2:
        raise ValueError("logits and targets must have shapes BxTxV and BxT")
    if logits.shape[:2] != target_ids.shape:
        raise ValueError("logit and target sequence shapes differ")
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
        reduction="mean",
    )


def evaluate_window_rows(
    *,
    model: torch.nn.Module,
    reader: FrozenWindowReader,
    index_id: str,
    device: torch.device,
) -> dict[str, Any]:
    """Compute token-weighted loss over one fixed held-out index."""

    model.eval()
    total_loss = 0.0
    total_targets = 0
    with torch.no_grad():
        for row in reader.rows(index_id):
            source = reader.source_tokens(row)
            inputs = torch.tensor(source[:-1], dtype=torch.long, device=device).unsqueeze(0)
            targets = torch.tensor(source[1:], dtype=torch.long, device=device).unsqueeze(0)
            loss = shifted_causal_loss(model(input_ids=inputs, use_cache=False).logits, targets)
            target_count = int(row["target_tokens"])
            total_loss += float(loss.item()) * target_count
            total_targets += target_count
    if total_targets <= 0:
        raise ValueError("validation index contains no target tokens")
    mean = total_loss / total_targets
    if not math.isfinite(mean):
        raise FloatingPointError("V7 validation produced a non-finite loss")
    return {"index_id": index_id, "target_tokens": total_targets, "loss": mean}


def compose_validation_metrics(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """Create the exact gate metrics from per-pool token-weighted losses."""

    required = (
        "validation_historical_general",
        "validation_historical_non_sonnet_poetry",
        "validation_nineteenth_century_bridge",
        "sonnets_validation",
    )
    if any(key not in rows for key in required):
        raise ValueError("validation rows omit one or more frozen pools")

    def weighted(keys: Sequence[str]) -> float:
        targets = sum(int(rows[key]["target_tokens"]) for key in keys)
        return sum(
            float(rows[key]["loss"]) * int(rows[key]["target_tokens"])
            for key in keys
        ) / targets

    return {
        "historical_general_bridge_token_weighted_loss": weighted(
            ("validation_historical_general", "validation_nineteenth_century_bridge")
        ),
        "stage_1_historical_general_bridge_token_weighted_loss": weighted(
            ("validation_historical_general", "validation_nineteenth_century_bridge")
        ),
        "historical_non_sonnet_poetry_loss": float(
            rows["validation_historical_non_sonnet_poetry"]["loss"]
        ),
        "v7_sonnet_validation_loss": float(rows["sonnets_validation"]["loss"]),
        "all_broader_validation_token_weighted_loss": weighted(required[:3]),
    }


def evaluate_instruction_response_loss(
    *, model: torch.nn.Module, tokenizer: Any, prompts: Sequence[Mapping[str, str]], device: torch.device
) -> dict[str, Any]:
    """Measure assistant-response loss while masking prompt/header labels."""

    model.eval()
    rows = []
    total_loss = 0.0
    total_targets = 0
    with torch.no_grad():
        for prompt in prompts:
            prefix = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
            complete = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt["prompt"]},
                    {"role": "assistant", "content": prompt["response"]},
                ],
                tokenize=True,
                add_generation_prompt=False,
            )
            if complete[: len(prefix)] != prefix:
                raise ValueError("instruction template prefix changed")
            inputs = torch.tensor(complete[:-1], dtype=torch.long, device=device).unsqueeze(0)
            targets = torch.tensor(complete[1:], dtype=torch.long, device=device).unsqueeze(0)
            logits = model(input_ids=inputs, use_cache=False).logits
            per_token = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            ).reshape_as(targets)
            first_response_target = max(0, len(prefix) - 1)
            response_losses = per_token[:, first_response_target:]
            count = response_losses.numel()
            mean = float(response_losses.mean().item())
            rows.append({"id": prompt["id"], "target_tokens": count, "loss": mean})
            total_loss += mean * count
            total_targets += count
    return {
        "rows": rows,
        "target_tokens": total_targets,
        "instruction_validation_loss": total_loss / total_targets,
    }


def evaluate_all_gates(
    *,
    model: torch.nn.Module,
    reader: FrozenWindowReader,
    modern_reader: FrozenWindowReader,
    tokenizer: Any,
    prompts: Sequence[Mapping[str, str]],
    device: torch.device,
) -> dict[str, Any]:
    """Measure every broader, sonnet, modern, and instruction preservation gate."""

    pool_ids = (
        "validation_historical_general",
        "validation_historical_non_sonnet_poetry",
        "validation_nineteenth_century_bridge",
        "sonnets_validation",
    )
    pool_rows = {
        pool_id: evaluate_window_rows(
            model=model, reader=reader, index_id=pool_id, device=device
        )
        for pool_id in pool_ids
    }
    metrics = compose_validation_metrics(pool_rows)
    modern = evaluate_window_rows(
        model=model,
        reader=modern_reader,
        index_id="modern_preservation_validation_v1",
        device=device,
    )
    instruction = evaluate_instruction_response_loss(
        model=model, tokenizer=tokenizer, prompts=prompts, device=device
    )
    metrics["modern_validation_loss"] = float(modern["loss"])
    metrics["instruction_validation_loss"] = float(
        instruction["instruction_validation_loss"]
    )
    return {
        "pool_rows": pool_rows,
        "modern": modern,
        "instruction": instruction,
        "metrics": metrics,
    }


def checkpoint_metadata(
    *,
    protocol: Mapping[str, Any],
    stage_id: str,
    stage_update: int,
    global_update: int,
    next_stage_window_index: int,
    next_window_identity_sha256: str,
    next_learning_rate: float,
    world_size: int,
    git_commit: str,
    package_versions: Mapping[str, str],
    hardware_topology: Mapping[str, Any],
    validation_history: Sequence[Mapping[str, Any]],
    protocol_sha256: str | None = None,
    parent_baseline_metrics: Mapping[str, float] | None = None,
    stage_start_metrics: Mapping[str, float] | None = None,
    recent_updates: Sequence[Mapping[str, float]] = (),
    preservation_failures: int = 0,
    non_improving_evaluations: int = 0,
    best_qualifying_primary: float | None = None,
) -> dict[str, Any]:
    """Build the complete immutable resume manifest metadata."""

    return {
        "trainer_version": TRAINER_VERSION,
        "stage_id": stage_id,
        "stage_update": stage_update,
        "global_update": global_update,
        "next_stage_window_index": next_stage_window_index,
        "next_window_identity_sha256": next_window_identity_sha256,
        "next_learning_rate": next_learning_rate,
        "protocol_sha256": protocol_sha256 or _sha256_json(protocol),
        "encoded_content_identity_sha256": protocol["lineage"][
            "encoded_content_identity_sha256"
        ],
        "window_content_identity_sha256": protocol["lineage"][
            "window_index_content_identity_sha256"
        ],
        "world_size": world_size,
        "git_commit": git_commit,
        "package_versions": dict(package_versions),
        "hardware_topology": dict(hardware_topology),
        "validation_history": list(validation_history),
        "parent_baseline_metrics": dict(parent_baseline_metrics or {}),
        "stage_start_metrics": dict(stage_start_metrics or {}),
        "recent_updates": list(recent_updates),
        "preservation_failures": preservation_failures,
        "non_improving_evaluations": non_improving_evaluations,
        "best_qualifying_primary": best_qualifying_primary,
    }


def save_atomic_resume_checkpoint(
    *,
    destination: Path,
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    sampler_state: Mapping[str, Any],
    rank_rng_states: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Save model/optimizer/scheduler/RNG/sampler state in one atomic directory."""

    def populate(directory: Path) -> None:
        _base_model(model).save_pretrained(
            directory / "model", safe_serialization=True, max_shard_size="5GB"
        )
        tokenizer.save_pretrained(directory / "model")
        torch.save(optimizer.state_dict(), directory / "optimizer.pt")
        torch.save(dict(scheduler_state), directory / "scheduler.pt")
        rng = {
            "per_rank": list(rank_rng_states)
            if rank_rng_states is not None
            else [capture_local_rng_state()],
        }
        torch.save(rng, directory / "rng.pt")
        (directory / "sampler.json").write_text(
            json.dumps(dict(sampler_state), sort_keys=True) + "\n", encoding="utf-8"
        )

    return atomic_install_checkpoint_writer(
        destination=destination, populate=populate, metadata=metadata
    )


def restore_atomic_resume_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rank: int = 0,
) -> dict[str, Any]:
    """Verify all files before restoring weights, optimizer, scheduler, and RNG."""

    manifest = verify_checkpoint_directory(path)
    from safetensors.torch import load_file

    model_state: dict[str, torch.Tensor] = {}
    for weight_path in sorted((path / "model").glob("*.safetensors")):
        model_state.update(load_file(weight_path, device="cpu"))
    missing, unexpected = _base_model(model).load_state_dict(model_state, strict=True)
    if missing or unexpected:
        raise ValueError("resume model state is incomplete")
    optimizer.load_state_dict(
        torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
    )
    scheduler_state = torch.load(
        path / "scheduler.pt", map_location="cpu", weights_only=True
    )
    rng = torch.load(path / "rng.pt", map_location="cpu", weights_only=False)
    per_rank = rng["per_rank"]
    if rank < 0 or rank >= len(per_rank):
        raise ValueError("resume RNG state is missing this rank")
    restore_local_rng_state(per_rank[rank])
    return {
        "manifest": manifest,
        "scheduler_state": scheduler_state,
        "sampler_state": json.loads((path / "sampler.json").read_text()),
    }


def capture_local_rng_state() -> dict[str, Any]:
    """Capture the independent Python, CPU, and current-device CUDA RNG streams."""

    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_current_device": (
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        ),
    }


def restore_local_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda_current_device"] is not None:
        torch.cuda.set_rng_state(state["torch_cuda_current_device"])


def save_model_only_analysis_snapshot(
    *, destination: Path, model: torch.nn.Module, tokenizer: Any, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Save one BF16 model-only state plus hashes; never duplicate optimizer state."""

    def populate(directory: Path) -> None:
        _base_model(model).save_pretrained(
            directory / "model", safe_serialization=True, max_shard_size="5GB"
        )
        tokenizer.save_pretrained(directory / "model")

    return atomic_install_checkpoint_writer(
        destination=destination,
        populate=populate,
        metadata={"artifact_type": "model_only_analysis_snapshot", **dict(metadata)},
    )


def restore_model_only_analysis_snapshot(
    *, path: Path, model: torch.nn.Module
) -> dict[str, Any]:
    """Hash-verify and load a validation-selected model-only stage state."""

    manifest = verify_checkpoint_directory(path)
    from safetensors.torch import load_file

    state: dict[str, torch.Tensor] = {}
    for weight_path in sorted((path / "model").glob("*.safetensors")):
        state.update(load_file(weight_path, device="cpu"))
    _base_model(model).load_state_dict(state, strict=True)
    return manifest


def write_sparse_analysis_summary(
    *,
    path: Path,
    stage_id: str,
    update: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    allocator_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist compact parameter/gradient/optimizer evidence at sparse points."""

    row = {
        "stage_id": stage_id,
        "update": update,
        "parameters": summarize_named_tensors(_base_model(model).named_parameters()),
        "gradients": summarize_named_tensors(
            (name, parameter.grad)
            for name, parameter in _base_model(model).named_parameters()
        ),
        "optimizer": optimizer_state_inventory(optimizer.state_dict()),
        "allocator": dict(allocator_summary),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def record_stage_evaluation(
    *,
    stage: Mapping[str, Any],
    stage_update: int,
    evaluation: Mapping[str, Any],
    baseline_metrics: Mapping[str, float],
    preservation: Mapping[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply all frozen promotion gates and update best-candidate history."""

    row = {"update": stage_update, **dict(evaluation["metrics"])}
    row["passes_all_gates"] = candidate_passes_gates(
        stage=stage,
        metrics=row,
        baseline_metrics=baseline_metrics,
        preservation=preservation,
    )
    history.append(row)
    selected = select_stage_checkpoint(
        stage=stage,
        history=history,
        baseline_metrics=baseline_metrics,
        preservation=preservation,
    )
    row["is_current_selected_candidate"] = selected is row
    return row


def stage_global_update(protocol: Mapping[str, Any], stage_id: str, update: int) -> int:
    total = 0
    for stage in protocol["stages"]:
        if stage["stage_id"] == stage_id:
            if update <= 0 or update > int(stage["optimizer_updates"]):
                raise ValueError("stage update is outside the frozen protocol")
            return total + update
        total += int(stage["optimizer_updates"])
    raise KeyError(f"unknown stage: {stage_id}")


def should_evaluate(stage: Mapping[str, Any], update: int) -> bool:
    return update == int(stage["optimizer_updates"]) or (
        update % int(stage["evaluation_interval_updates"]) == 0
    )


def should_save_resume(stage: Mapping[str, Any], update: int) -> bool:
    return update == int(stage["optimizer_updates"]) or (
        update % int(stage["resume_interval_updates"]) == 0
    )


def should_save_analysis_snapshot(
    *, stage: Mapping[str, Any], update: int, midpoint_update: int
) -> bool:
    return update in (midpoint_update, int(stage["optimizer_updates"]))


def runtime_candidate_from_environment(
    *, world_size: int = 2, global_windows_per_update: int = 16
) -> RuntimeCandidate:
    """Accept only a candidate already selected by the bounded qualification."""

    try:
        microbatch = int(os.environ["V7_LOCAL_MICROBATCH_SIZE"])
        accumulation = int(os.environ["V7_GRADIENT_ACCUMULATION_STEPS"])
    except KeyError as error:
        raise RuntimeError("V7 qualified runtime environment is incomplete") from error
    checkpointing = os.environ.get("V7_GRADIENT_CHECKPOINTING", "false").lower()
    if checkpointing not in ("true", "false"):
        raise ValueError("V7_GRADIENT_CHECKPOINTING must be true or false")
    mode = os.environ.get("V7_EXECUTION_MODE", "eager")
    candidate = RuntimeCandidate(
        local_microbatch_size=microbatch,
        gradient_accumulation_steps=accumulation,
        gradient_checkpointing=checkpointing == "true",
        execution_mode=mode,
    )
    if (
        microbatch not in (1, 2, 4)
        or microbatch * accumulation * world_size != global_windows_per_update
    ):
        raise ValueError("runtime candidate does not preserve 16 global windows")
    if mode not in ("eager", "torch_compile_default"):
        raise ValueError("runtime execution mode is outside the frozen matrix")
    return candidate


def qualification_runtime_candidates(
    protocol: Mapping[str, Any],
) -> tuple[RuntimeCandidate, ...]:
    """Translate the exact 12 approved hardware candidates into runtime values."""

    return tuple(
        RuntimeCandidate(
            local_microbatch_size=row.local_microbatch_size,
            gradient_accumulation_steps=row.gradient_accumulation_steps,
            gradient_checkpointing=row.gradient_checkpointing,
            execution_mode=row.execution_mode,
        )
        for row in build_hardware_candidates(protocol)
    )


def qualification_result_passes(
    *,
    rank_metrics: Sequence[Mapping[str, float]],
    measured_nccl_gbps: float,
    projected_cost_usd: float,
    validation_transition_passes: bool,
    resume_proof_passes: bool,
    protocol: Mapping[str, Any],
) -> bool:
    """Apply every frozen numerical, memory, communication, and proof gate."""

    hardware = protocol["hardware_qualification"]
    return (
        len(rank_metrics) == int(hardware["world_size"])
        and all(
            math.isfinite(float(row[metric]))
            for row in rank_metrics
            for metric in ("mean_loss", "mean_gradient_norm")
        )
        and min(float(row["reserved_headroom_mib"]) for row in rank_metrics)
        >= float(hardware["minimum_peak_reserved_headroom_mib"])
        and measured_nccl_gbps
        >= float(hardware["minimum_nccl_all_reduce_gigabytes_per_second"])
        and projected_cost_usd
        <= float(protocol["cost"]["maximum_projected_all_in_cost_to_launch"])
        and validation_transition_passes
        and resume_proof_passes
    )


def run_one_distributed_update(
    *,
    ddp_model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    parameters: Sequence[torch.nn.Parameter],
    input_microbatches: Sequence[torch.Tensor],
    target_microbatches: Sequence[torch.Tensor],
    max_gradient_norm: float,
) -> tuple[float, float]:
    """Run one exact accumulation boundary and return loss and pre-clip norm."""

    if not input_microbatches or len(input_microbatches) != len(target_microbatches):
        raise ValueError("input/target microbatch sequences must be non-empty and equal")
    ddp_model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    count = len(input_microbatches)
    for index, (inputs, targets) in enumerate(
        zip(input_microbatches, target_microbatches, strict=True)
    ):
        sync = ddp_model.no_sync() if index < count - 1 else nullcontext()
        with sync:
            outputs = ddp_model(input_ids=inputs, use_cache=False)
            loss = shifted_causal_loss(outputs.logits, targets)
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value):
                raise FloatingPointError("V7 training produced a non-finite loss")
            (loss / count).backward()
            losses.append(loss_value)
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_gradient_norm)
    gradient_value = float(gradient_norm.detach().item())
    if not math.isfinite(gradient_value):
        raise FloatingPointError("V7 training produced a non-finite gradient norm")
    optimizer.step()
    return sum(losses) / len(losses), gradient_value


def configure_stage_optimizer(
    *,
    model: torch.nn.Module,
    stage: Mapping[str, Any],
    protocol: Mapping[str, Any],
    bitsandbytes_module: Any,
) -> torch.optim.Optimizer:
    """Reset PagedAdamW8bit at each stage while retaining model weights."""

    return bitsandbytes_module.optim.PagedAdamW8bit(
        list(model.parameters()),
        lr=float(stage["peak_learning_rate"]),
        weight_decay=float(protocol["optimizer"]["weight_decay"]),
    )


def build_modern_preservation_reader(
    *, repo_root: Path, execution_config: V7ExecutionConfig
) -> tuple[Int32ShardStore, FrozenWindowReader]:
    """Adapt the private PAISÀ index to the common verified reader contract."""

    modern_report = json.loads(
        (repo_root / "reports/minerva_7b_full_weight_data_report.json").read_text()
    )
    split = next(
        row for row in modern_report["splits"] if row["split_id"] == "paisa_validation"
    )
    encoded_report = {
        "pools": [
            {
                "pool_id": "paisa_validation",
                "shards": split["shards"],
            }
        ]
    }
    store = Int32ShardStore(
        encoded_dir=execution_config.modern_encoded_dir,
        encoded_report=encoded_report,
        required_pools=["paisa_validation"],
    )
    index_path = execution_config.modern_index_path
    manifest = {
        "files": [
            {
                "path": index_path.name,
                "pool_id": "paisa_validation",
                "rows": 128,
                "bytes": index_path.stat().st_size,
                "sha256": _sha256_path(index_path),
            }
        ]
    }
    reader = FrozenWindowReader(
        index_root=index_path.parent,
        encoded_store=store,
        window_manifest=manifest,
    )
    return store, reader


def apply_stage_learning_rate(
    optimizer: torch.optim.Optimizer, stage: Mapping[str, Any], update: int
) -> float:
    learning_rate = learning_rate_at_update(stage, update)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def validate_long_run_authorization(execution: Mapping[str, Any]) -> None:
    if not bool(execution["authorization"].get("long_training_authorized")):
        raise PermissionError(
            "checkpoint 8F deliberately leaves long Minerva V7 training unauthorized"
        )


def validate_single_h100_runtime(
    *, launch: Mapping[str, Any], world_size: int, visible_gpu_count: int
) -> RuntimeCandidate:
    """Match the process and environment to the exact qualified H100 candidate."""

    runtime = launch["qualified_runtime"]
    if world_size != int(runtime["world_size"]) or visible_gpu_count != int(
        runtime["gpu_count"]
    ):
        raise RuntimeError("Minerva V7 stage training requires exactly one visible H100")
    candidate = runtime_candidate_from_environment(
        world_size=world_size,
        global_windows_per_update=int(runtime["global_windows_per_update"]),
    )
    expected = RuntimeCandidate(
        local_microbatch_size=int(runtime["local_microbatch_size"]),
        gradient_accumulation_steps=int(runtime["gradient_accumulation_steps"]),
        gradient_checkpointing=bool(runtime["gradient_checkpointing"]),
        execution_mode=str(runtime["execution_mode"]),
    )
    if candidate != expected:
        raise RuntimeError("runtime environment differs from the qualified candidate")
    return candidate


def train_minerva_7b_v7_full_weight(
    *,
    repo_root: Path,
    execution_config: V7ExecutionConfig,
    launch: Mapping[str, Any],
    launch_config_sha256: str,
    requested_stage_id: str,
    resume_from_checkpoint: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """Run exactly one user-selected stage under the qualified H100 contract."""

    context = build_execution_context(execution_config)
    execution = context["execution"]
    if not torch.cuda.is_available() or not dist.is_available():
        raise RuntimeError("Minerva V7 full-weight training requires distributed CUDA")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    candidate = validate_single_h100_runtime(
        launch=launch,
        world_size=world_size,
        visible_gpu_count=torch.cuda.device_count(),
    )
    requested_stage = next(
        (
            row
            for row in context["protocol"]["stages"]
            if row["stage_id"] == requested_stage_id
        ),
        None,
    )
    if requested_stage is None:
        raise ValueError("requested stage is outside the frozen protocol")
    launch_stage = next(
        (row for row in launch["stage_launches"] if row["stage_id"] == requested_stage_id),
        None,
    )
    if launch_stage is None:
        raise ValueError("requested stage is outside the launch authorization")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    try:
        _report(progress, f"stage={requested_stage_id} phase=loading_dependencies")
        dependencies = _load_dependencies()
        protocol = context["protocol"]
        torch.manual_seed(int(protocol["optimizer"]["seed"]))
        random.seed(int(protocol["optimizer"]["seed"]))
        _report(progress, f"stage={requested_stage_id} phase=loading_parent_model")
        model = dependencies["AutoModelForCausalLM"].from_pretrained(
            protocol["model"]["model_id"],
            revision=protocol["model"]["revision"],
            cache_dir=repo_root / "data/local/minerva_qlora/huggingface",
            dtype=torch.bfloat16,
            device_map={"": local_rank},
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        model.config.use_cache = False
        if candidate.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            model.gradient_checkpointing_disable()
        audit = audit_full_weight_model(model)
        if not (
            audit["all_weights_trainable"]
            and audit["adapter_free"]
            and audit["quantization_free"]
            and audit["parameter_dtype_counts"]
            == {"bfloat16": audit["total_parameter_count"]}
        ):
            raise ValueError("V7 model failed the full-weight BF16 audit")
        if candidate.execution_mode == "torch_compile_default":
            _report(progress, f"stage={requested_stage_id} phase=compiling_model")
            model = torch.compile(model, mode="default", fullgraph=False, dynamic=False)
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            bucket_cap_mb=25,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
        )
        parameters = list(model.parameters())
        recent_updates: list[dict[str, float]] = []
        validation_history: list[dict[str, Any]] = []
        preservation_failures = 0
        resume_manifest = (
            verify_checkpoint_directory(resume_from_checkpoint)
            if resume_from_checkpoint is not None
            else None
        )
        if resume_manifest is not None:
            resume_metadata = resume_manifest["metadata"]
            if resume_metadata.get("stage_id") != requested_stage_id:
                raise ValueError("resume checkpoint belongs to a different stage")
            if resume_metadata.get("launch_config_sha256") != launch_config_sha256:
                raise ValueError("resume checkpoint launch lineage differs")
        hourly_rate = float(os.environ.get("V7_HOURLY_RATE_USD", "0"))
        if hourly_rate <= 0:
            raise RuntimeError("V7_HOURLY_RATE_USD must record the positive instance rate")
        run_started = time.monotonic()
        run_dir = repo_root / execution["local_paths"]["run_dir"]
        telemetry_path = run_dir / "telemetry.jsonl"
        evaluation_path = run_dir / "evaluations.jsonl"
        sparse_path = run_dir / "sparse_layerwise_summaries.jsonl"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_marker = run_dir / "stage_runs" / f"{requested_stage_id}.json"
        boundary_path = run_dir / "stage_boundaries" / f"{requested_stage_id}_selected"
        if boundary_path.exists():
            raise RuntimeError(f"stage is already complete: {requested_stage_id}")
        if run_marker.exists() and resume_manifest is None:
            raise RuntimeError("stage was already started; use its latest verified resume")
        tokenizer = dependencies["AutoTokenizer"].from_pretrained(
            protocol["model"]["model_id"],
            revision=protocol["model"]["revision"],
            cache_dir=repo_root / "data/local/minerva_qlora/huggingface",
            local_files_only=True,
        )
        prompts = json.loads(
            (repo_root / protocol["lineage"]["preservation_prompts_path"]).read_text()
        )
        modern_store, modern_reader = build_modern_preservation_reader(
            repo_root=repo_root, execution_config=execution_config
        )
        with Int32ShardStore(
            encoded_dir=execution_config.encoded_dir,
            encoded_report=context["encoded_report"],
        ) as store:
            reader = FrozenWindowReader(
                index_root=execution_config.window_index_dir,
                encoded_store=store,
                window_manifest=context["window_manifest"],
            )
            if resume_manifest is None:
                required_boundary = launch_stage["required_boundary"]
                if required_boundary is None:
                    if requested_stage_id != "stage_1_historical_general":
                        raise ValueError("only stage 1 may start from the untouched parent")
                    _report(
                        progress,
                        f"stage={requested_stage_id} phase=parent_baseline_validation",
                    )
                    baseline_evaluation = evaluate_all_gates(
                        model=model,
                        reader=reader,
                        modern_reader=modern_reader,
                        tokenizer=tokenizer,
                        prompts=prompts,
                        device=device,
                    )
                    parent_baseline_metrics = dict(baseline_evaluation["metrics"])
                    _report(
                        progress,
                        "stage={stage} phase=parent_baseline_complete "
                        "historical_general_bridge_loss={historical:.4f} "
                        "modern_loss={modern:.4f} instruction_loss={instruction:.4f}".format(
                            stage=requested_stage_id,
                            historical=parent_baseline_metrics[
                                "historical_general_bridge_token_weighted_loss"
                            ],
                            modern=parent_baseline_metrics["modern_validation_loss"],
                            instruction=parent_baseline_metrics[
                                "instruction_validation_loss"
                            ],
                        ),
                    )
                    stage_start_metrics = dict(parent_baseline_metrics)
                    validation_history = []
                    preceding_model_identity_sha256 = hashlib.sha256(
                        (
                            context["protocol"]["model"]["model_id"]
                            + "@"
                            + context["protocol"]["model"]["revision"]
                        ).encode("utf-8")
                    ).hexdigest()
                else:
                    previous_stage_id = str(
                        context["protocol"]["stages"][
                            [row["stage_id"] for row in context["protocol"]["stages"]].index(
                                requested_stage_id
                            )
                            - 1
                        ]["stage_id"]
                    )
                    previous_path = repo_root / str(required_boundary)
                    previous_manifest = validate_stage_boundary(
                        path=previous_path,
                        expected_stage_id=previous_stage_id,
                        launch=launch,
                    )
                    restore_model_only_analysis_snapshot(path=previous_path, model=model)
                    previous_metadata = previous_manifest["metadata"]
                    parent_baseline_metrics = dict(
                        previous_metadata["parent_baseline_metrics"]
                    )
                    stage_start_metrics = dict(previous_metadata["selected_metrics"])
                    validation_history = list(previous_metadata["validation_history"])
                    preceding_model_identity_sha256 = _sha256_path(
                        previous_path / "manifest.json"
                    )
            else:
                resume_metadata = resume_manifest["metadata"]
                parent_baseline_metrics = dict(
                    resume_metadata["parent_baseline_metrics"]
                )
                stage_start_metrics = dict(resume_metadata["stage_start_metrics"])
                validation_history = list(resume_metadata["validation_history"])
                recent_updates = list(resume_metadata["recent_updates"])
                preservation_failures = int(
                    resume_metadata["preservation_failures"]
                )
                preceding_model_identity_sha256 = str(
                    resume_metadata["preceding_model_identity_sha256"]
                )
            if rank == 0 and not run_marker.exists():
                run_marker.parent.mkdir(parents=True, exist_ok=True)
                run_marker.write_text(
                    json.dumps(
                        {
                            "stage_id": requested_stage_id,
                            "launch_config_sha256": launch_config_sha256,
                            "status": "started",
                            "resume_from_checkpoint": (
                                str(resume_from_checkpoint)
                                if resume_from_checkpoint is not None
                                else None
                            ),
                            "preceding_model_identity_sha256": preceding_model_identity_sha256,
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            dist.barrier()
            for stage in protocol["stages"]:
                if stage["stage_id"] != requested_stage_id:
                    continue
                optimizer = configure_stage_optimizer(
                    model=model,
                    stage=stage,
                    protocol=protocol,
                    bitsandbytes_module=dependencies["bitsandbytes"],
                )
                stage_id = str(stage["stage_id"])
                if resume_manifest is not None:
                    resumed_stage = str(resume_manifest["metadata"]["stage_id"])
                    stage_ids = [str(row["stage_id"]) for row in protocol["stages"]]
                    if stage_ids.index(stage_id) < stage_ids.index(resumed_stage):
                        continue
                    if stage_id == resumed_stage:
                        restored = restore_atomic_resume_checkpoint(
                            path=resume_from_checkpoint,
                            model=model,
                            optimizer=optimizer,
                            rank=rank,
                        )
                        if restored["sampler_state"][
                            "next_stage_window_index"
                        ] != resume_manifest["metadata"]["next_stage_window_index"]:
                            raise ValueError("resume sampler position mismatch")
                        start_update = int(resume_manifest["metadata"]["stage_update"]) + 1
                    else:
                        start_update = 1
                else:
                    start_update = 1
                stage_history: list[dict[str, Any]] = [
                    {key: value for key, value in row.items() if key != "stage_id"}
                    for row in validation_history
                    if row.get("stage_id") == stage_id
                ]
                selected_snapshot_paths: dict[int, Path] = {}
                snapshot_root = run_dir / "analysis_snapshots"
                for existing in snapshot_root.glob(
                    f"{stage_id}_selected_candidate_update_*"
                ):
                    verify_checkpoint_directory(existing)
                    selected_snapshot_paths[int(existing.name.rsplit("_", 1)[1])] = (
                        existing
                    )
                non_improving_evaluations = (
                    int(resume_manifest["metadata"]["non_improving_evaluations"])
                    if resume_manifest is not None
                    else 0
                )
                stored_best = (
                    resume_manifest["metadata"]["best_qualifying_primary"]
                    if resume_manifest is not None
                    else None
                )
                best_qualifying_primary = (
                    float(stored_best) if stored_best is not None else math.inf
                )
                gate_baseline = {
                    **stage_start_metrics,
                    "modern_validation_loss": parent_baseline_metrics[
                        "modern_validation_loss"
                    ],
                    "instruction_validation_loss": parent_baseline_metrics[
                        "instruction_validation_loss"
                    ],
                }
                for update in range(start_update, int(stage["optimizer_updates"]) + 1):
                    batch = reader.optimizer_batch(
                        stage_id=stage_id,
                        update=update,
                        global_windows_per_update=16,
                    )
                    local_inputs = batch.input_ids[rank::world_size]
                    local_targets = batch.target_ids[rank::world_size]
                    input_batches = _tensor_microbatches(
                        local_inputs, candidate.local_microbatch_size, device
                    )
                    target_batches = _tensor_microbatches(
                        local_targets, candidate.local_microbatch_size, device
                    )
                    learning_rate = apply_stage_learning_rate(optimizer, stage, update)
                    update_started = time.monotonic()
                    loss, gradient_norm = run_one_distributed_update(
                        ddp_model=ddp_model,
                        optimizer=optimizer,
                        parameters=parameters,
                        input_microbatches=input_batches,
                        target_microbatches=target_batches,
                        max_gradient_norm=float(protocol["optimizer"]["max_gradient_norm"]),
                    )
                    torch.cuda.synchronize(device)
                    global_numerics = torch.tensor(
                        [loss, gradient_norm], dtype=torch.float64, device=device
                    )
                    dist.all_reduce(global_numerics, op=dist.ReduceOp.SUM)
                    loss, gradient_norm = (
                        float(value) / world_size for value in global_numerics.tolist()
                    )
                    duration = time.monotonic() - update_started
                    recent_updates.append({"loss": loss, "gradient_norm": gradient_norm})
                    reasons = abort_reasons(
                        protocol=protocol,
                        recent_updates=recent_updates,
                        consecutive_preservation_failures=preservation_failures,
                        projected_or_spent_cost_usd=(
                            (time.monotonic() - run_started) / 3600 * hourly_rate
                        ),
                    )
                    if reasons:
                        raise RuntimeError("V7 abort rule: " + ", ".join(reasons))
                    local_memory = {
                        "rank": rank,
                        "allocated_mib": torch.cuda.memory_allocated(device) / 1024**2,
                        "reserved_mib": torch.cuda.memory_reserved(device) / 1024**2,
                    }
                    rank_memory: list[dict[str, float] | None] = [None] * world_size
                    dist.all_gather_object(rank_memory, local_memory)
                    if rank == 0:
                        elapsed = time.monotonic() - run_started
                        global_update = stage_global_update(protocol, stage_id, update)
                        remaining = (int(stage["optimizer_updates"]) - update) * duration
                        row = make_update_telemetry(
                            stage_id=stage_id,
                            stage_update=update,
                            global_update=global_update,
                            batch=batch,
                            loss=loss,
                            gradient_norm=gradient_norm,
                            learning_rate=learning_rate,
                            tokens_per_second=32768 / duration,
                            elapsed_seconds=elapsed,
                            eta_seconds=remaining,
                            rank_memory=[row for row in rank_memory if row is not None],
                            cumulative_cost_usd=elapsed / 3600 * hourly_rate,
                        )
                        with telemetry_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                        if update == 1 or update % int(stage["progress_interval_updates"]) == 0:
                            _report(
                                progress,
                                f"stage={stage_id} update={update}/{stage['optimizer_updates']} "
                                f"progress={100 * update / int(stage['optimizer_updates']):.1f}% "
                                f"loss={loss:.4f} lr={learning_rate:.2e} "
                                f"elapsed={elapsed:.0f}s eta={remaining:.0f}s",
                            )
                    midpoint_update = int(
                        protocol["analysis_snapshots"]["midpoint_updates"][stage_id]
                    )
                    if update == midpoint_update:
                        if rank == 0:
                            write_sparse_analysis_summary(
                                path=sparse_path,
                                stage_id=stage_id,
                                update=update,
                                model=model,
                                optimizer=optimizer,
                                allocator_summary={
                                    "peak_allocated_mib": torch.cuda.max_memory_allocated(
                                        device
                                    )
                                    / 1024**2,
                                    "peak_reserved_mib": torch.cuda.max_memory_reserved(
                                        device
                                    )
                                    / 1024**2,
                                },
                            )
                            save_model_only_analysis_snapshot(
                                destination=run_dir
                                / "analysis_snapshots"
                                / f"{stage_id}_update_{update:06d}",
                                model=model,
                                tokenizer=tokenizer,
                                metadata={
                                    "stage_id": stage_id,
                                    "update": update,
                                    "snapshot_role": "midpoint",
                                    "protocol_sha256": execution["lineage"]["protocol_sha256"],
                                    "launch_config_sha256": launch_config_sha256,
                                    "preceding_model_identity_sha256": preceding_model_identity_sha256,
                                    "encoded_content_identity_sha256": protocol["lineage"]["encoded_content_identity_sha256"],
                                    "window_content_identity_sha256": protocol["lineage"]["window_index_content_identity_sha256"],
                                    "git_commit": os.environ.get("V7_GIT_COMMIT", "unrecorded"),
                                    "package_versions": {
                                        "torch": torch.__version__,
                                        "bitsandbytes": dependencies["bitsandbytes"].__version__,
                                    },
                                    "hardware_topology": {
                                        "platform": platform.platform(),
                                        "gpu_name": torch.cuda.get_device_name(device),
                                        "world_size": world_size,
                                    },
                                },
                            )
                            _report(
                                progress,
                                f"stage={stage_id} update={update} snapshot=midpoint_saved",
                            )
                        dist.barrier()
                    if should_evaluate(stage, update):
                        evaluation = evaluate_all_gates(
                            model=model,
                            reader=reader,
                            modern_reader=modern_reader,
                            tokenizer=tokenizer,
                            prompts=prompts,
                            device=device,
                        )
                        evaluation_row = record_stage_evaluation(
                            stage=stage,
                            stage_update=update,
                            evaluation=evaluation,
                            baseline_metrics=gate_baseline,
                            preservation=protocol["preservation"],
                            history=stage_history,
                        )
                        validation_history.append(
                            {"stage_id": stage_id, **evaluation_row}
                        )
                        preservation_passes = (
                            evaluation_row["modern_validation_loss"]
                            <= parent_baseline_metrics["modern_validation_loss"]
                            * float(protocol["preservation"]["maximum_modern_loss_ratio"])
                            and evaluation_row["instruction_validation_loss"]
                            <= parent_baseline_metrics["instruction_validation_loss"]
                            * float(
                                protocol["preservation"][
                                    "maximum_instruction_loss_ratio"
                                ]
                            )
                        )
                        preservation_failures = 0 if preservation_passes else preservation_failures + 1
                        _report(
                            progress,
                            "stage={stage} update={update} validation "
                            "primary={primary:.4f} sonnet={sonnet:.4f} modern={modern:.4f} "
                            "instruction={instruction:.4f} passes_all_gates={passes}".format(
                                stage=stage_id,
                                update=update,
                                primary=float(evaluation_row[str(stage["primary_validation_metric"])]),
                                sonnet=float(evaluation_row["v7_sonnet_validation_loss"]),
                                modern=float(evaluation_row["modern_validation_loss"]),
                                instruction=float(evaluation_row["instruction_validation_loss"]),
                                passes=evaluation_row["passes_all_gates"],
                            ),
                        )
                        primary = str(stage["primary_validation_metric"])
                        if (
                            evaluation_row["passes_all_gates"]
                            and float(evaluation_row[primary]) < best_qualifying_primary
                        ):
                            best_qualifying_primary = float(evaluation_row[primary])
                            non_improving_evaluations = 0
                        elif update >= int(stage["early_stopping_eligible_after_update"]):
                            non_improving_evaluations += 1
                        reasons = abort_reasons(
                            protocol=protocol,
                            recent_updates=recent_updates,
                            consecutive_preservation_failures=preservation_failures,
                            projected_or_spent_cost_usd=(
                                (time.monotonic() - run_started) / 3600 * hourly_rate
                            ),
                        )
                        if reasons:
                            raise RuntimeError("V7 abort rule: " + ", ".join(reasons))
                        if (
                            rank == 0
                            and evaluation_row["is_current_selected_candidate"]
                        ):
                            if update != midpoint_update:
                                write_sparse_analysis_summary(
                                    path=sparse_path,
                                    stage_id=stage_id,
                                    update=update,
                                    model=model,
                                    optimizer=optimizer,
                                    allocator_summary={
                                        "peak_allocated_mib": torch.cuda.max_memory_allocated(
                                            device
                                        )
                                        / 1024**2,
                                        "peak_reserved_mib": torch.cuda.max_memory_reserved(
                                            device
                                        )
                                        / 1024**2,
                                    },
                                )
                            snapshot_path = (
                                run_dir
                                / "analysis_snapshots"
                                / f"{stage_id}_selected_candidate_update_{update:06d}"
                            )
                            save_model_only_analysis_snapshot(
                                destination=snapshot_path,
                                model=model,
                                tokenizer=tokenizer,
                                metadata={
                                    "stage_id": stage_id,
                                    "update": update,
                                    "snapshot_role": "validation_selected_candidate",
                                    "metrics": evaluation_row,
                                    "protocol_sha256": execution["lineage"]["protocol_sha256"],
                                    "launch_config_sha256": launch_config_sha256,
                                    "preceding_model_identity_sha256": preceding_model_identity_sha256,
                                    "parent_baseline_metrics": parent_baseline_metrics,
                                    "selected_metrics": evaluation_row,
                                    "validation_history": validation_history,
                                    "encoded_content_identity_sha256": protocol["lineage"]["encoded_content_identity_sha256"],
                                    "window_content_identity_sha256": protocol["lineage"]["window_index_content_identity_sha256"],
                                    "git_commit": os.environ.get("V7_GIT_COMMIT", "unrecorded"),
                                    "package_versions": {
                                        "torch": torch.__version__,
                                        "bitsandbytes": dependencies["bitsandbytes"].__version__,
                                    },
                                    "hardware_topology": {
                                        "platform": platform.platform(),
                                        "gpu_name": torch.cuda.get_device_name(device),
                                        "world_size": world_size,
                                    },
                                },
                            )
                            for old_update, old_path in tuple(
                                selected_snapshot_paths.items()
                            ):
                                if old_update != update:
                                    verify_checkpoint_directory(old_path)
                                    shutil.rmtree(old_path)
                                    del selected_snapshot_paths[old_update]
                            selected_snapshot_paths[update] = snapshot_path
                            _report(
                                progress,
                                f"stage={stage_id} update={update} "
                                f"selected_candidate_snapshot={snapshot_path}",
                            )
                        dist.barrier()
                        if rank == 0:
                            with evaluation_path.open("a", encoding="utf-8") as handle:
                                handle.write(
                                    json.dumps(
                                        {
                                            "stage_id": stage_id,
                                            **evaluation,
                                            **evaluation_row,
                                        },
                                        sort_keys=True,
                                    )
                                    + "\n"
                                )
                        if non_improving_evaluations >= int(
                            protocol["abort_rules"][
                                "no_qualifying_improvement_patience_evaluations"
                            ]
                        ):
                            break
                    if should_save_resume(stage, update) and update < int(
                        stage["optimizer_updates"]
                    ):
                        local_rng = capture_local_rng_state()
                        gathered_rng: list[dict[str, Any] | None] = [None] * world_size
                        dist.all_gather_object(gathered_rng, local_rng)
                        next_update = min(update + 1, int(stage["optimizer_updates"]))
                        next_batch = reader.optimizer_batch(
                            stage_id=stage_id,
                            update=next_update,
                            global_windows_per_update=16,
                        )
                        global_update = stage_global_update(protocol, stage_id, update)
                        metadata = checkpoint_metadata(
                            protocol=protocol,
                            stage_id=stage_id,
                            stage_update=update,
                            global_update=global_update,
                            next_stage_window_index=next_batch.first_window_index,
                            next_window_identity_sha256=next_batch.identity_sha256,
                            next_learning_rate=learning_rate_at_update(stage, next_update),
                            world_size=world_size,
                            git_commit=os.environ.get("V7_GIT_COMMIT", "unrecorded"),
                            package_versions={
                                "torch": torch.__version__,
                                "bitsandbytes": dependencies["bitsandbytes"].__version__,
                            },
                            hardware_topology={"platform": platform.platform()},
                            validation_history=validation_history,
                            protocol_sha256=execution["lineage"]["protocol_sha256"],
                            parent_baseline_metrics=parent_baseline_metrics,
                            stage_start_metrics=stage_start_metrics,
                            recent_updates=recent_updates[-23:],
                            preservation_failures=preservation_failures,
                            non_improving_evaluations=non_improving_evaluations,
                            best_qualifying_primary=(
                                best_qualifying_primary
                                if math.isfinite(best_qualifying_primary)
                                else None
                            ),
                        )
                        metadata["launch_config_sha256"] = launch_config_sha256
                        metadata["preceding_model_identity_sha256"] = (
                            preceding_model_identity_sha256
                        )
                        if rank == 0:
                            resume_root = run_dir / "resume"
                            destination = resume_root / f"resume_{global_update:06d}"
                            save_atomic_resume_checkpoint(
                                destination=destination,
                                model=model,
                                tokenizer=tokenizer,
                                optimizer=optimizer,
                                scheduler_state={
                                    "stage_id": stage_id,
                                    "stage_update": update,
                                    "learning_rate": learning_rate,
                                },
                                metadata=metadata,
                                sampler_state={
                                    "next_stage_window_index": next_batch.first_window_index,
                                    "next_window_identity_sha256": next_batch.identity_sha256,
                                },
                                rank_rng_states=[
                                    row for row in gathered_rng if row is not None
                                ],
                            )
                            rotate_resume_checkpoints(resume_root, retain=2)
                            _report(
                                progress,
                                f"stage={stage_id} update={update} "
                                f"resume_checkpoint={destination}",
                            )
                        dist.barrier()
                selected = select_stage_checkpoint(
                    stage=stage,
                    history=stage_history,
                    baseline_metrics=gate_baseline,
                    preservation=protocol["preservation"],
                )
                if selected is None:
                    raise RuntimeError(
                        f"stage {stage_id} has no checkpoint passing all promotion gates"
                    )
                selected_update = int(selected["update"])
                selected_path_text = (
                    str(selected_snapshot_paths[selected_update]) if rank == 0 else None
                )
                selected_path_payload = [selected_path_text]
                dist.broadcast_object_list(selected_path_payload, src=0)
                selected_path = Path(str(selected_path_payload[0]))
                if rank == 0:
                    restore_model_only_analysis_snapshot(path=selected_path, model=model)
                    boundary_path.parent.mkdir(parents=True, exist_ok=True)
                    selected_manifest = json.loads(
                        (selected_path / "manifest.json").read_text(encoding="utf-8")
                    )
                    source_candidate_manifest_sha256 = _sha256_path(
                        selected_path / "manifest.json"
                    )
                    selected_manifest["metadata"]["snapshot_role"] = (
                        "validation_selected_endpoint"
                    )
                    selected_manifest["metadata"]["selected_metrics"] = selected
                    selected_manifest["metadata"]["validation_history"] = (
                        validation_history
                    )
                    selected_manifest["metadata"][
                        "source_candidate_manifest_sha256"
                    ] = source_candidate_manifest_sha256
                    (selected_path / "manifest.json").write_text(
                        json.dumps(selected_manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    selected_path.rename(boundary_path)
                    verify_checkpoint_directory(boundary_path)
                    run_marker.write_text(
                        json.dumps(
                            {
                                "stage_id": stage_id,
                                "launch_config_sha256": launch_config_sha256,
                                "status": "complete",
                                "selected_update": selected_update,
                                "boundary": str(boundary_path),
                                "boundary_manifest_sha256": _sha256_path(
                                    boundary_path / "manifest.json"
                                ),
                            },
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    _report(
                        progress,
                        f"stage={stage_id} complete selected_update={selected_update} "
                        f"boundary={boundary_path}",
                    )
                else:
                    restore_model_only_analysis_snapshot(path=selected_path, model=model)
                dist.barrier()
                stage_start_metrics = dict(selected)
                resume_manifest = None
        modern_store.close()
        if rank == 0:
            return {
                "trainer_version": TRAINER_VERSION,
                "status": "stage_complete",
                "stage_id": requested_stage_id,
                "boundary_path": str(boundary_path),
                "model_audit": audit,
                "telemetry_path": str(telemetry_path),
            }
        return None
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def qualify_minerva_7b_v7_full_weight(
    *,
    repo_root: Path,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run all 12 bounded candidates after a separately committed authorization."""

    execution = json.loads(
        (repo_root / "configs/minerva_7b_v7_execution.json").read_text(encoding="utf-8")
    )
    if not execution["authorization"]["gpu_qualification_authorized"]:
        raise PermissionError(
            "checkpoint 8F freezes the qualification command but does not authorize it"
        )
    protocol = json.loads(
        (repo_root / execution["lineage"]["protocol_path"]).read_text(encoding="utf-8")
    )
    if not torch.cuda.is_available() or not dist.is_available():
        raise RuntimeError("V7 qualification requires distributed CUDA")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 2 or torch.cuda.device_count() != 2:
        raise RuntimeError("V7 qualification requires exactly two visible GPUs")
    hourly_rate = float(os.environ.get("V7_HOURLY_RATE_USD", "0"))
    if hourly_rate <= 0:
        raise RuntimeError("V7_HOURLY_RATE_USD must be positive")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    try:
        dependencies = _load_dependencies()
        _qualify_hardware(protocol=protocol, device=device)
        context = build_execution_context(
            V7ExecutionConfig(
                repo_root=repo_root,
                execution_path=repo_root / "configs/minerva_7b_v7_execution.json",
                encoded_dir=repo_root / execution["local_paths"]["encoded_dir"],
                window_index_dir=repo_root
                / execution["local_paths"]["window_index_dir"],
                modern_encoded_dir=repo_root
                / execution["local_paths"]["modern_encoded_dir"],
                modern_index_path=repo_root
                / execution["local_paths"]["modern_index_path"],
            )
        )
        with Int32ShardStore(
            encoded_dir=repo_root / execution["local_paths"]["encoded_dir"],
            encoded_report=context["encoded_report"],
        ) as store:
            reader = FrozenWindowReader(
                index_root=repo_root / execution["local_paths"]["window_index_dir"],
                encoded_store=store,
                window_manifest=context["window_manifest"],
            )
            batch = reader.optimizer_batch(
                stage_id="stage_1_historical_general",
                update=1,
                global_windows_per_update=16,
            )
            rows = []
            for candidate in qualification_runtime_candidates(protocol):
                torch.manual_seed(int(protocol["optimizer"]["seed"]))
                model = dependencies["AutoModelForCausalLM"].from_pretrained(
                    protocol["model"]["model_id"],
                    revision=protocol["model"]["revision"],
                    cache_dir=repo_root / "data/local/minerva_qlora/huggingface",
                    dtype=torch.bfloat16,
                    device_map={"": local_rank},
                    low_cpu_mem_usage=True,
                    attn_implementation="sdpa",
                )
                model.config.use_cache = False
                if candidate.gradient_checkpointing:
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                else:
                    model.gradient_checkpointing_disable()
                if candidate.execution_mode == "torch_compile_default":
                    model = torch.compile(
                        model, mode="default", fullgraph=False, dynamic=False
                    )
                parameters = list(model.parameters())
                optimizer = dependencies["bitsandbytes"].optim.PagedAdamW8bit(
                    parameters,
                    lr=float(protocol["stages"][0]["peak_learning_rate"]),
                    weight_decay=float(protocol["optimizer"]["weight_decay"]),
                )
                ddp = DistributedDataParallel(
                    model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    broadcast_buffers=False,
                    bucket_cap_mb=25,
                    gradient_as_bucket_view=True,
                    find_unused_parameters=False,
                )
                local_inputs = batch.input_ids[rank::world_size]
                local_targets = batch.target_ids[rank::world_size]
                inputs = _tensor_microbatches(
                    local_inputs, candidate.local_microbatch_size, device
                )
                targets = _tensor_microbatches(
                    local_targets, candidate.local_microbatch_size, device
                )
                losses = []
                gradients = []
                torch.cuda.reset_peak_memory_stats(device)
                started = time.monotonic()
                total_updates = 23
                for update in range(1, total_updates + 1):
                    loss, gradient = run_one_distributed_update(
                        ddp_model=ddp,
                        optimizer=optimizer,
                        parameters=parameters,
                        input_microbatches=inputs,
                        target_microbatches=targets,
                        max_gradient_norm=1.0,
                    )
                    if update > 3:
                        losses.append(loss)
                        gradients.append(gradient)
                    if rank == 0 and (update == 1 or update % 5 == 0 or update == 23):
                        _report(
                            progress,
                            f"candidate={candidate} update={update}/23 loss={loss:.4f}",
                        )
                torch.cuda.synchronize(device)
                timed_seconds = time.monotonic() - started
                timed_throughput = 20 * 32768 / timed_seconds
                total_memory = torch.cuda.get_device_properties(device).total_memory
                peak_reserved = torch.cuda.max_memory_reserved(device)
                headroom_mib = (total_memory - peak_reserved) / 1024**2
                local_row = {
                    "rank": rank,
                    "mean_loss": sum(losses) / len(losses),
                    "mean_gradient_norm": sum(gradients) / len(gradients),
                    "tokens_per_second": timed_throughput,
                    "peak_reserved_mib": peak_reserved / 1024**2,
                    "reserved_headroom_mib": headroom_mib,
                }
                gathered: list[dict[str, Any] | None] = [None] * world_size
                dist.all_gather_object(gathered, local_row)
                if rank == 0:
                    minimum_headroom = min(
                        float(row["reserved_headroom_mib"])
                        for row in gathered
                        if row is not None
                    )
                    mean_throughput = sum(
                        float(row["tokens_per_second"])
                        for row in gathered
                        if row is not None
                    ) / world_size
                    projected_hours = (
                        96_993_280
                        / mean_throughput
                        / 3600
                        * float(protocol["cost"]["projection_overhead_multiplier"])
                    )
                    rows.append(
                        {
                            **candidate.__dict__,
                            "rank_metrics": [row for row in gathered if row is not None],
                            "mean_tokens_per_second": mean_throughput,
                            "minimum_reserved_headroom_mib": minimum_headroom,
                            "projected_all_in_hours": projected_hours,
                            "projected_all_in_cost_usd": projected_hours * hourly_rate,
                            "passes": (
                                minimum_headroom >= 8192
                                and projected_hours * hourly_rate <= 48.0
                            ),
                        }
                    )
                del ddp, optimizer, model, parameters
                torch.cuda.empty_cache()
                dist.barrier()
        if rank != 0:
            return {}
        performance_candidate = max(
            (row for row in rows if row["passes"]),
            key=lambda row: float(row["mean_tokens_per_second"]),
            default=None,
        )
        report = {
            "qualification_version": QUALIFICATION_VERSION,
            "status": (
                "measurements_complete_proofs_pending"
                if performance_candidate is not None
                else "failed_performance_gates"
            ),
            "candidates": rows,
            "performance_candidate": performance_candidate,
            "selected_candidate": None,
            "validation_transition_required": True,
            "validation_transition_passed": False,
            "fresh_process_resume_proof_required": True,
            "fresh_process_resume_proof_passed": False,
            "quality_checkpoint_retained": False,
            "long_training_automatically_authorized": False,
        }
        output = repo_root / "data/local/minerva_7b_v7/qualification_v1.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _tensor_microbatches(
    rows: Sequence[Sequence[int]], microbatch: int, device: torch.device
) -> tuple[torch.Tensor, ...]:
    if len(rows) % microbatch:
        raise ValueError("rank rows do not divide by local microbatch")
    return tuple(
        torch.tensor(rows[start : start + microbatch], dtype=torch.long, device=device)
        for start in range(0, len(rows), microbatch)
    )


def _qualify_hardware(*, protocol: Mapping[str, Any], device: torch.device) -> None:
    hardware = protocol["hardware_qualification"]
    properties = torch.cuda.get_device_properties(device)
    if (
        "h100 80gb hbm3" not in properties.name.lower()
        or properties.total_memory / 1024**2
        < int(hardware["minimum_memory_mib_per_gpu"])
        or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("GPU does not satisfy the frozen H100 BF16 profile")
    if not torch.cuda.can_device_access_peer(0, 1):
        raise RuntimeError("CUDA peer access is unavailable")
    payload = torch.ones(256 * 1024**2, dtype=torch.bfloat16, device=device)
    for _ in range(3):
        dist.all_reduce(payload)
    torch.cuda.synchronize(device)
    dist.barrier()
    started = time.monotonic()
    for _ in range(10):
        dist.all_reduce(payload)
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    bandwidth = payload.numel() * payload.element_size() / (elapsed / 10) / 1e9
    measured = torch.tensor(bandwidth, dtype=torch.float64, device=device)
    dist.all_reduce(measured, op=dist.ReduceOp.MIN)
    if float(measured.item()) < float(
        hardware["minimum_nccl_all_reduce_gigabytes_per_second"]
    ):
        raise RuntimeError("measured NCCL all-reduce bandwidth failed the frozen gate")
    del payload, measured
    torch.cuda.empty_cache()


def _sha256_json(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the serializable model underneath torch.compile, when present."""

    return getattr(model, "_orig_mod", model)


def _load_dependencies() -> dict[str, Any]:
    try:
        import bitsandbytes
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Minerva V7 dependencies are missing") from error
    return {
        "bitsandbytes": bitsandbytes,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
