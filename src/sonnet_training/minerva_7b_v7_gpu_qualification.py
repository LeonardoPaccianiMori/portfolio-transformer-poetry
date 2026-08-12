"""GPU measurements and state-transition proofs for Minerva V7 qualification."""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from sonnet_training.minerva_7b_model_audit import audit_full_weight_model
from sonnet_training.minerva_7b_v7_execution import (
    FrozenWindowReader,
    Int32ShardStore,
    V7ExecutionConfig,
    build_execution_context,
    fresh_process_resume_contract,
    verify_checkpoint_directory,
)
from sonnet_training.minerva_7b_v7_protocol import learning_rate_at_update
from sonnet_training.minerva_7b_v7_qualification import (
    QualificationCandidate,
    candidate_by_id,
    load_hardware_qualification,
    preliminary_gate_reasons,
    project_candidate_cost,
)
from sonnet_training.minerva_7b_v7_trainer import (
    capture_local_rng_state,
    checkpoint_metadata,
    restore_atomic_resume_checkpoint,
    run_one_distributed_update,
    save_atomic_resume_checkpoint,
    shifted_causal_loss,
)


Progress = Callable[[str], None]


def qualification_paths(
    repo_root: Path, artifact_directory: str = "qualification_v2"
) -> dict[str, Path]:
    root = repo_root / "data/local/minerva_7b_v7" / artifact_directory
    return {
        "root": root,
        "candidates": root / "candidates",
        "proof": root / "proof",
        "checkpoint": root / "proof/resume_checkpoint",
        "proof_save": root / "proof/save.json",
        "proof_resume": root / "proof/resume.json",
        "final_report": root / "qualification.json",
    }


def run_candidate_worker(
    *,
    repo_root: Path,
    qualification_path: Path,
    candidate_id: str,
    output_path: Path,
    progress: Progress | None = None,
) -> dict[str, Any] | None:
    """Measure one candidate in its own torchrun process group."""

    config = load_hardware_qualification(qualification_path, repo_root)
    candidate = candidate_by_id(config, candidate_id)
    rank, local_rank, world_size, device = _initialize_distributed(config)
    try:
        hardware = _measure_hardware(config=config, device=device)
        communication = _measure_communication(config=config, device=device)
        dependencies = _load_dependencies()
        context, reader, store = _open_training_reader(repo_root, config)
        try:
            batch = reader.optimizer_batch(
                stage_id="stage_1_historical_general",
                update=1,
                global_windows_per_update=candidate.global_windows_per_update,
            )
            model = _load_model(
                repo_root=repo_root,
                protocol=context["protocol"],
                dependencies=dependencies,
                local_rank=local_rank,
                candidate=candidate,
            )
            model_audit = audit_full_weight_model(_base_model(model))
            _require_full_weight_audit(model_audit)
            parameters = list(model.parameters())
            optimizer = dependencies["bitsandbytes"].optim.PagedAdamW8bit(
                parameters,
                lr=float(context["protocol"]["stages"][0]["peak_learning_rate"]),
                weight_decay=float(context["protocol"]["optimizer"]["weight_decay"]),
            )
            ddp = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                bucket_cap_mb=int(config["primary_profile"]["ddp_bucket_cap_mib"]),
                gradient_as_bucket_view=True,
                find_unused_parameters=False,
            )
            inputs, targets = _candidate_microbatches(
                batch=batch,
                rank=rank,
                world_size=world_size,
                candidate=candidate,
                device=device,
            )
            measurement = _measure_updates(
                ddp_model=ddp,
                optimizer=optimizer,
                parameters=parameters,
                inputs=inputs,
                targets=targets,
                config=config,
                candidate=candidate,
                device=device,
                rank=rank,
                progress=progress,
            )
            local_metrics = measurement["local_metrics"]
            gathered: list[dict[str, Any] | None] = [None] * world_size
            dist.all_gather_object(gathered, local_metrics)
            rank_metrics = [row for row in gathered if row is not None]
            if rank != 0:
                return None
            mean_throughput = float(measurement["global_tokens_per_second"])
            projection = project_candidate_cost(
                config=config, measured_tokens_per_second=mean_throughput
            )
            reasons = preliminary_gate_reasons(
                config=config,
                rank_metrics=rank_metrics,
                hardware=hardware,
                communication=communication,
                projection=projection,
            )
            report = {
                "qualification_version": config["qualification_version"],
                "candidate": candidate.__dict__,
                "status": "passed_preliminary_gates" if not reasons else "failed",
                "hardware": hardware,
                "communication": communication,
                "model_audit": model_audit,
                "rank_metrics": rank_metrics,
                "mean_tokens_per_second": mean_throughput,
                "projection": projection,
                "preliminary_gate_reasons": list(reasons),
                "validation_transition_run": False,
                "checkpoint_or_resume_proof_run": False,
                "quality_checkpoint_retained": False,
            }
            _write_json(output_path, report)
            return report
        finally:
            store.close()
    finally:
        _destroy_distributed()


def run_proof_save_worker(
    *,
    repo_root: Path,
    qualification_path: Path,
    candidate_id: str,
    output_path: Path,
    checkpoint_path: Path,
    progress: Progress | None = None,
) -> dict[str, Any] | None:
    """Prove validation transition and atomically save a one-update resume state."""

    config = load_hardware_qualification(qualification_path, repo_root)
    candidate = candidate_by_id(config, candidate_id)
    rank, local_rank, world_size, device = _initialize_distributed(config)
    try:
        dependencies = _load_dependencies()
        context, reader, store = _open_training_reader(repo_root, config)
        try:
            protocol = context["protocol"]
            model = _load_model(
                repo_root=repo_root,
                protocol=protocol,
                dependencies=dependencies,
                local_rank=local_rank,
                candidate=candidate,
            )
            _require_full_weight_audit(audit_full_weight_model(_base_model(model)))
            parameters = list(model.parameters())
            stage = protocol["stages"][0]
            optimizer = dependencies["bitsandbytes"].optim.PagedAdamW8bit(
                parameters,
                lr=learning_rate_at_update(stage, 1),
                weight_decay=float(protocol["optimizer"]["weight_decay"]),
            )
            ddp = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                bucket_cap_mb=int(config["primary_profile"]["ddp_bucket_cap_mib"]),
                gradient_as_bucket_view=True,
                find_unused_parameters=False,
            )
            validation_row = reader.rows("validation_historical_general")[0]
            validation_before = _evaluate_one_row(
                model=ddp, reader=reader, row=validation_row, device=device
            )
            batch = reader.optimizer_batch(
                stage_id="stage_1_historical_general",
                update=1,
                global_windows_per_update=candidate.global_windows_per_update,
            )
            inputs, targets = _candidate_microbatches(
                batch=batch,
                rank=rank,
                world_size=world_size,
                candidate=candidate,
                device=device,
            )
            loss, gradient_norm = run_one_distributed_update(
                ddp_model=ddp,
                optimizer=optimizer,
                parameters=parameters,
                input_microbatches=inputs,
                target_microbatches=targets,
                max_gradient_norm=float(protocol["optimizer"]["max_gradient_norm"]),
            )
            validation_after = _evaluate_one_row(
                model=ddp, reader=reader, row=validation_row, device=device
            )
            finite = all(
                math.isfinite(value)
                for value in (
                    validation_before,
                    loss,
                    gradient_norm,
                    validation_after,
                )
            )
            local_rng = capture_local_rng_state()
            gathered_rng: list[dict[str, Any] | None] = [None] * world_size
            dist.all_gather_object(gathered_rng, local_rng)
            next_batch = reader.optimizer_batch(
                stage_id="stage_1_historical_general",
                update=2,
                global_windows_per_update=candidate.global_windows_per_update,
            )
            metadata = checkpoint_metadata(
                protocol=protocol,
                stage_id="stage_1_historical_general",
                stage_update=1,
                global_update=1,
                next_stage_window_index=next_batch.first_window_index,
                next_window_identity_sha256=next_batch.identity_sha256,
                next_learning_rate=learning_rate_at_update(stage, 2),
                world_size=world_size,
                git_commit=os.environ.get("V7_GIT_COMMIT", "uncommitted-8g"),
                package_versions={
                    "torch": torch.__version__,
                    "bitsandbytes": dependencies["bitsandbytes"].__version__,
                    "transformers": dependencies["transformers_version"],
                },
                hardware_topology={
                    "platform": platform.platform(),
                    "devices": [
                        torch.cuda.get_device_name(index)
                        for index in range(world_size)
                    ],
                },
                validation_history=[
                    {
                        "proof": "held_out_validation_transition",
                        "loss_before": validation_before,
                        "loss_after": validation_after,
                    }
                ],
                protocol_sha256=config["scientific_protocol"]["sha256"],
            )
            if rank == 0:
                tokenizer = dependencies["AutoTokenizer"].from_pretrained(
                    protocol["model"]["model_id"],
                    revision=protocol["model"]["revision"],
                    cache_dir=repo_root / "data/local/minerva_qlora/huggingface",
                    local_files_only=True,
                )
                manifest = save_atomic_resume_checkpoint(
                    destination=checkpoint_path,
                    model=_base_model(model),
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler_state={
                        "stage_id": "stage_1_historical_general",
                        "stage_update": 1,
                        "learning_rate": learning_rate_at_update(stage, 1),
                    },
                    metadata=metadata,
                    sampler_state={
                        "next_stage_window_index": next_batch.first_window_index,
                        "next_window_identity_sha256": next_batch.identity_sha256,
                    },
                    rank_rng_states=[row for row in gathered_rng if row is not None],
                )
                installed = verify_checkpoint_directory(checkpoint_path)
                report = {
                    "candidate_id": candidate_id,
                    "status": "passed" if finite else "failed",
                    "validation_transition_passed": finite,
                    "validation_loss_before": validation_before,
                    "training_loss": loss,
                    "gradient_norm": gradient_norm,
                    "validation_loss_after": validation_after,
                    "atomic_checkpoint_passed": installed == manifest,
                    "checkpoint_manifest": manifest,
                    "fresh_process_resume_passed": False,
                }
                _write_json(output_path, report)
                _report(progress, "validation transition and atomic save complete")
            dist.barrier()
            return report if rank == 0 else None
        finally:
            store.close()
    finally:
        _destroy_distributed()


def run_proof_resume_worker(
    *,
    repo_root: Path,
    qualification_path: Path,
    candidate_id: str,
    save_report_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    progress: Progress | None = None,
) -> dict[str, Any] | None:
    """Reload in a fresh process, verify exact state, and complete the next update."""

    config = load_hardware_qualification(qualification_path, repo_root)
    candidate = candidate_by_id(config, candidate_id)
    rank, local_rank, world_size, device = _initialize_distributed(config)
    try:
        dependencies = _load_dependencies()
        context, reader, store = _open_training_reader(repo_root, config)
        try:
            protocol = context["protocol"]
            stage = protocol["stages"][0]
            model = _load_model(
                repo_root=repo_root,
                protocol=protocol,
                dependencies=dependencies,
                local_rank=local_rank,
                candidate=candidate,
            )
            parameters = list(model.parameters())
            optimizer = dependencies["bitsandbytes"].optim.PagedAdamW8bit(
                parameters,
                lr=learning_rate_at_update(stage, 1),
                weight_decay=float(protocol["optimizer"]["weight_decay"]),
            )
            restored = restore_atomic_resume_checkpoint(
                path=checkpoint_path, model=_base_model(model), optimizer=optimizer, rank=rank
            )
            manifest = restored["manifest"]
            saved_rng = torch.load(
                checkpoint_path / "rng.pt", map_location="cpu", weights_only=False
            )["per_rank"][rank]
            restored_rng = capture_local_rng_state()
            rng_passed = _rng_states_equal(saved_rng, restored_rng)
            next_batch = reader.optimizer_batch(
                stage_id="stage_1_historical_general",
                update=2,
                global_windows_per_update=candidate.global_windows_per_update,
            )
            expected = {
                "stage_id": "stage_1_historical_general",
                "stage_update": 1,
                "global_update": 1,
                "next_stage_window_index": next_batch.first_window_index,
                "next_window_identity_sha256": next_batch.identity_sha256,
                "next_learning_rate": learning_rate_at_update(stage, 2),
                "protocol_sha256": config["scientific_protocol"]["sha256"],
                "encoded_content_identity_sha256": protocol["lineage"][
                    "encoded_content_identity_sha256"
                ],
                "window_content_identity_sha256": protocol["lineage"][
                    "window_index_content_identity_sha256"
                ],
                "world_size": world_size,
            }
            contract = fresh_process_resume_contract(
                manifest=manifest, expected=expected
            )
            sampler_passed = (
                restored["sampler_state"]["next_stage_window_index"]
                == next_batch.first_window_index
                and restored["sampler_state"]["next_window_identity_sha256"]
                == next_batch.identity_sha256
                and manifest["metadata"]["next_learning_rate"]
                == learning_rate_at_update(stage, 2)
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at_update(stage, 2)
            ddp = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                bucket_cap_mb=int(config["primary_profile"]["ddp_bucket_cap_mib"]),
                gradient_as_bucket_view=True,
                find_unused_parameters=False,
            )
            inputs, targets = _candidate_microbatches(
                batch=next_batch,
                rank=rank,
                world_size=world_size,
                candidate=candidate,
                device=device,
            )
            loss, gradient_norm = run_one_distributed_update(
                ddp_model=ddp,
                optimizer=optimizer,
                parameters=parameters,
                input_microbatches=inputs,
                target_microbatches=targets,
                max_gradient_norm=float(protocol["optimizer"]["max_gradient_norm"]),
            )
            finite = math.isfinite(loss) and math.isfinite(gradient_norm)
            local_passed = bool(
                contract["passes"] and sampler_passed and rng_passed and finite
            )
            passed_tensor = torch.tensor(int(local_passed), dtype=torch.int32, device=device)
            dist.all_reduce(passed_tensor, op=dist.ReduceOp.MIN)
            if rank != 0:
                return None
            save_report = json.loads(save_report_path.read_text(encoding="utf-8"))
            report = {
                "candidate_id": candidate_id,
                "status": "passed" if int(passed_tensor.item()) == 1 else "failed",
                "validation_transition_passed": bool(
                    save_report["validation_transition_passed"]
                ),
                "atomic_checkpoint_passed": bool(
                    save_report["atomic_checkpoint_passed"]
                ),
                "fresh_process_resume_passed": int(passed_tensor.item()) == 1,
                "resume_contract": contract,
                "sampler_and_learning_rate_passed": sampler_passed,
                "per_rank_rng_restore_passed": rng_passed,
                "finite_next_update_passed": finite,
                "next_update_loss": loss,
                "next_update_gradient_norm": gradient_norm,
                "quality_checkpoint_retained": False,
            }
            _write_json(output_path, report)
            _report(progress, "fresh-process reload and finite next update complete")
            return report
        finally:
            store.close()
    finally:
        _destroy_distributed()


def remove_temporary_proof_checkpoint(path: Path) -> None:
    """Remove only the verified bounded proof checkpoint after its resume succeeds."""

    verify_checkpoint_directory(path)
    shutil.rmtree(path)


def _initialize_distributed(
    config: Mapping[str, Any],
) -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available() or not dist.is_available():
        raise RuntimeError("Minerva V7 qualification requires distributed CUDA")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    expected = int(config["primary_profile"]["world_size"])
    if world_size != expected or torch.cuda.device_count() != expected:
        raise RuntimeError(f"qualification requires exactly {expected} visible GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    return rank, local_rank, world_size, device


def _destroy_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _measure_hardware(
    *, config: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    profile = config["primary_profile"]
    properties = torch.cuda.get_device_properties(device)
    free_bytes = shutil.disk_usage(Path.cwd()).free
    local = {
        "rank": dist.get_rank(),
        "gpu_name": properties.name,
        "total_memory_mib": properties.total_memory / 1024**2,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "native_bfloat16": torch.cuda.is_bf16_supported(),
        "free_host_scratch_gib": free_bytes / 1024**3,
    }
    gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    devices = [row for row in gathered if row is not None]
    peer_required = bool(profile.get("cuda_peer_access_required", False))
    peer = (
        bool(
            torch.cuda.can_device_access_peer(0, 1)
            and torch.cuda.can_device_access_peer(1, 0)
        )
        if len(devices) == 2
        else None
    )
    passed = (
        len(devices) == int(profile["world_size"])
        and all(
            str(profile["expected_gpu_name_substring"]).lower()
            in str(row["gpu_name"]).lower()
            and float(row["total_memory_mib"])
            >= float(profile["minimum_memory_mib_per_gpu"])
            and bool(row["native_bfloat16"])
            and float(row["free_host_scratch_gib"])
            >= float(profile["minimum_free_host_scratch_gib"])
            for row in devices
        )
        and (not peer_required or peer is True)
    )
    return {
        "devices": devices,
        "cuda_peer_access_bidirectional": peer,
        "nccl_version": list(torch.cuda.nccl.version()),
        "profile_passed": passed,
    }


def _measure_communication(
    *, config: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    profile = config["primary_profile"]
    if not bool(profile.get("communication_measurement_required", True)):
        return {
            "status": "not_applicable_single_gpu",
            "payload_mib": 0,
            "warmup_iterations": 0,
            "timed_iterations": 0,
            "mean_milliseconds": 0.0,
            "algorithmic_gigabytes_per_second": 0.0,
        }
    element_count = int(profile["communication_payload_mib"]) * 1024**2 // 2
    payload = torch.ones(element_count, dtype=torch.bfloat16, device=device)
    for _ in range(int(profile["communication_warmup_iterations"])):
        dist.all_reduce(payload)
    torch.cuda.synchronize(device)
    dist.barrier()
    started = time.monotonic()
    for _ in range(int(profile["communication_timed_iterations"])):
        dist.all_reduce(payload)
    torch.cuda.synchronize(device)
    dist.barrier()
    elapsed = torch.tensor(
        time.monotonic() - started, dtype=torch.float64, device=device
    )
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    mean_seconds = float(elapsed.item()) / int(
        profile["communication_timed_iterations"]
    )
    payload_bytes = payload.numel() * payload.element_size()
    result = {
        "payload_mib": int(profile["communication_payload_mib"]),
        "warmup_iterations": int(profile["communication_warmup_iterations"]),
        "timed_iterations": int(profile["communication_timed_iterations"]),
        "mean_milliseconds": mean_seconds * 1000,
        "algorithmic_gigabytes_per_second": payload_bytes / mean_seconds / 1e9,
    }
    del payload, elapsed
    torch.cuda.empty_cache()
    return result


def _open_training_reader(
    repo_root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], FrozenWindowReader, Int32ShardStore]:
    execution_path = repo_root / config["scientific_protocol"]["execution_path"]
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution_config = V7ExecutionConfig(
        repo_root=repo_root,
        execution_path=execution_path,
        encoded_dir=repo_root / execution["local_paths"]["encoded_dir"],
        window_index_dir=repo_root / execution["local_paths"]["window_index_dir"],
        modern_encoded_dir=repo_root / execution["local_paths"]["modern_encoded_dir"],
        modern_index_path=repo_root / execution["local_paths"]["modern_index_path"],
    )
    context = build_execution_context(execution_config)
    window_manifest, required_pools = _qualification_window_scope(
        context["window_manifest"]
    )
    store = Int32ShardStore(
        encoded_dir=execution_config.encoded_dir,
        encoded_report=context["encoded_report"],
        required_pools=required_pools,
    )
    reader = FrozenWindowReader(
        index_root=execution_config.window_index_dir,
        encoded_store=store,
        window_manifest=window_manifest,
    )
    return context, reader, store


def _qualification_window_scope(
    window_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Select only frozen training/validation indexes and their encoded pools."""

    files = window_manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("window manifest is missing files")
    scoped_files = [
        row
        for row in files
        if str(row["path"]).startswith(("training/", "validation/"))
    ]
    expected_paths: set[str] = set()
    required_pools: set[str] = set()
    training = window_manifest.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("window manifest is missing training scope")
    for stage in training.get("stages", []):
        expected_paths.add(str(stage["index"]["path"]))
        for component in stage.get("components", []):
            required_pools.update(str(value) for value in component["pool_windows"])
    evaluation = window_manifest.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("window manifest is missing evaluation scope")
    validation = evaluation.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("window manifest is missing validation scope")
    for pool in validation.get("pools", []):
        expected_paths.add(str(pool["index"]["path"]))
        required_pools.add(str(pool["pool_id"]))
    scoped_paths = {str(row["path"]) for row in scoped_files}
    if not required_pools or scoped_paths != expected_paths:
        raise ValueError("training/validation qualification scope is incomplete")
    if "sonnets_test" in required_pools:
        raise ValueError("V7 test pool may not enter hardware qualification")
    return {**window_manifest, "files": scoped_files}, tuple(sorted(required_pools))


def _load_model(
    *,
    repo_root: Path,
    protocol: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    local_rank: int,
    candidate: QualificationCandidate,
) -> torch.nn.Module:
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
        model = torch.compile(model, mode="default", fullgraph=False, dynamic=False)
    return model


def _candidate_microbatches(
    *,
    batch: Any,
    rank: int,
    world_size: int,
    candidate: QualificationCandidate,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    local_inputs = batch.input_ids[rank::world_size]
    local_targets = batch.target_ids[rank::world_size]
    return (
        _tensor_microbatches(local_inputs, candidate.local_microbatch_size, device),
        _tensor_microbatches(local_targets, candidate.local_microbatch_size, device),
    )


def _tensor_microbatches(
    rows: Sequence[Sequence[int]], microbatch: int, device: torch.device
) -> tuple[torch.Tensor, ...]:
    if len(rows) % microbatch:
        raise ValueError("rank rows do not divide by local microbatch")
    return tuple(
        torch.tensor(rows[start : start + microbatch], dtype=torch.long, device=device)
        for start in range(0, len(rows), microbatch)
    )


def _measure_updates(
    *,
    ddp_model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    parameters: Sequence[torch.nn.Parameter],
    inputs: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    config: Mapping[str, Any],
    candidate: QualificationCandidate,
    device: torch.device,
    rank: int,
    progress: Progress | None,
) -> dict[str, Any]:
    profile = config["primary_profile"]
    warmup = int(profile["minimum_warmup_updates"])
    timed = int(profile["minimum_timed_updates"])
    losses = []
    gradients = []
    reserved_samples = []
    torch.cuda.reset_peak_memory_stats(device)
    timed_started = None
    for update in range(1, warmup + timed + 1):
        loss, gradient = run_one_distributed_update(
            ddp_model=ddp_model,
            optimizer=optimizer,
            parameters=parameters,
            input_microbatches=inputs,
            target_microbatches=targets,
            max_gradient_norm=1.0,
        )
        torch.cuda.synchronize(device)
        if update == warmup:
            dist.barrier()
            timed_started = time.monotonic()
        elif update > warmup:
            losses.append(loss)
            gradients.append(gradient)
            reserved_samples.append(torch.cuda.memory_reserved(device) / 1024**2)
        if rank == 0 and (update == 1 or update % 5 == 0 or update == warmup + timed):
            _report(
                progress,
                f"candidate={candidate.candidate_id} update={update}/{warmup + timed} "
                f"loss={loss:.4f}",
            )
    if timed_started is None:
        raise RuntimeError("qualification timed phase never started")
    dist.barrier()
    elapsed = torch.tensor(
        time.monotonic() - timed_started, dtype=torch.float64, device=device
    )
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    global_throughput = (
        timed * int(candidate.global_target_tokens_per_update) / float(elapsed.item())
    )
    properties = torch.cuda.get_device_properties(device)
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    midpoint = max(1, len(reserved_samples) // 2)
    early_peak = max(reserved_samples[:midpoint])
    late_peak = max(reserved_samples[midpoint:])
    growth = max(0.0, late_peak - early_peak)
    return {
        "global_tokens_per_second": global_throughput,
        "local_metrics": {
            "rank": rank,
            "mean_loss": sum(losses) / len(losses),
            "mean_gradient_norm": sum(gradients) / len(gradients),
            "tokens_per_second": global_throughput,
            "peak_reserved_mib": peak_reserved,
            "reserved_headroom_mib": properties.total_memory / 1024**2 - peak_reserved,
            "reserved_memory_growth_mib": growth,
            "timed_updates": timed,
        },
    }


def _evaluate_one_row(
    *,
    model: torch.nn.Module,
    reader: FrozenWindowReader,
    row: Mapping[str, Any],
    device: torch.device,
) -> float:
    model.eval()
    source = reader.source_tokens(row)
    inputs = torch.tensor(source[:-1], dtype=torch.long, device=device).unsqueeze(0)
    targets = torch.tensor(source[1:], dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        loss = shifted_causal_loss(
            model(input_ids=inputs, use_cache=False).logits, targets
        )
    value = float(loss.item())
    if not math.isfinite(value):
        raise FloatingPointError("qualification validation loss is non-finite")
    return value


def _require_full_weight_audit(audit: Mapping[str, Any]) -> None:
    if not (
        audit["all_weights_trainable"]
        and audit["adapter_free"]
        and audit["quantization_free"]
        and audit["parameter_dtype_counts"]
        == {"bfloat16": audit["total_parameter_count"]}
    ):
        raise ValueError("qualification model failed the full-weight BF16 audit")


def _rng_states_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left["python"] != right["python"]:
        return False
    if not torch.equal(left["torch_cpu"], right["torch_cpu"]):
        return False
    left_cuda = left["torch_cuda_current_device"]
    right_cuda = right["torch_cuda_current_device"]
    if left_cuda is None or right_cuda is None:
        return left_cuda is None and right_cuda is None
    return bool(torch.equal(left_cuda.cpu(), right_cuda.cpu()))


def _base_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def _load_dependencies() -> dict[str, Any]:
    try:
        import bitsandbytes
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Minerva V7 qualification dependencies are missing") from error
    return {
        "bitsandbytes": bitsandbytes,
        "transformers_version": transformers.__version__,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _report(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)
