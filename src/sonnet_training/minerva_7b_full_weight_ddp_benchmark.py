"""Two-GPU DDP throughput benchmark for full-weight Minerva 7B."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from sonnet_training.cuda_compat import (
    cuda_device_name,
    cuda_device_properties,
    cuda_memory_info,
    max_cuda_memory_allocated,
    max_cuda_memory_reserved,
    prepare_cuda_memory_measurement,
    synchronize_cuda,
)
from sonnet_training.minerva_7b_full_weight_calibration import (
    MINIMUM_POST_OPTIMIZER_HEADROOM_MIB,
    audit_full_weight_model,
)
from sonnet_training.minerva_7b_full_weight_data import (
    load_full_weight_calibration_windows,
)
from sonnet_training.minerva_7b_full_weight_benchmark import (
    FULL_TRAINING_TOKEN_COUNT,
)
from sonnet_training.minerva_7b_qlora import (
    MINERVA_7B_INSTRUCT_MODEL_ID,
    MINERVA_7B_INSTRUCT_REVISION,
)


BENCHMARK_VERSION = "minerva_7b_full_weight_dual_rtx_pro_ddp_v1"
EXPECTED_WORLD_SIZE = 2
MINIMUM_GPU_MEMORY_MIB = 90 * 1024


@dataclass(frozen=True)
class Minerva7BFullWeightDdpBenchmarkConfig:
    """Freeze the bounded dual-RTX comparison approved after the H100 run."""

    model_id: str = MINERVA_7B_INSTRUCT_MODEL_ID
    revision: str = MINERVA_7B_INSTRUCT_REVISION
    cache_dir: str = "data/local/minerva_qlora/huggingface"
    calibration_windows_path: str = (
        "data/local/minerva_7b_full_weight/encoded/calibration_windows.pt"
    )
    output_path: str = (
        "data/local/minerva_7b_full_weight/full_weight_dual_rtx_pro_benchmark.json"
    )
    context_length: int = 512
    world_size: int = EXPECTED_WORLD_SIZE
    global_sequence_counts: tuple[int, ...] = (8, 16)
    bucket_cap_mib: tuple[int, ...] = (25, 100, 250)
    warmup_updates: int = 1
    timed_updates: int = 5
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    max_gradient_norm: float = 1.0
    minimum_total_memory_mib: int = MINIMUM_GPU_MEMORY_MIB
    minimum_headroom_mib: int = MINIMUM_POST_OPTIMIZER_HEADROOM_MIB
    hourly_rate_usd: float = 2.162
    full_training_token_count: int = FULL_TRAINING_TOKEN_COUNT
    projected_overhead_multiplier: float = 1.15
    communication_payload_mib: int = 512
    communication_warmup_iterations: int = 3
    communication_timed_iterations: int = 10
    seed: int = 1337


@dataclass(frozen=True)
class DdpThroughputCandidate:
    candidate_id: str
    global_sequences_per_update: int
    local_microbatch_size: int
    gradient_accumulation_steps: int
    bucket_cap_mib: int
    tokens_per_update: int


def validate_full_weight_ddp_benchmark_config(
    config: Minerva7BFullWeightDdpBenchmarkConfig,
) -> None:
    if config != Minerva7BFullWeightDdpBenchmarkConfig():
        raise ValueError("Minerva 7B DDP benchmark configuration is locked")


def build_ddp_throughput_candidates(
    config: Minerva7BFullWeightDdpBenchmarkConfig,
) -> tuple[DdpThroughputCandidate, ...]:
    """Compare equal-work DDP bucket sizes at two approved global batches."""
    candidates = []
    for global_sequences in config.global_sequence_counts:
        if global_sequences % config.world_size:
            raise ValueError("global sequence count must divide across DDP ranks")
        local_microbatch = global_sequences // config.world_size
        for bucket_cap_mib in config.bucket_cap_mib:
            candidates.append(DdpThroughputCandidate(
                candidate_id=(
                    f"global{global_sequences * config.context_length}_"
                    f"micro{local_microbatch}_bucket{bucket_cap_mib}"
                ),
                global_sequences_per_update=global_sequences,
                local_microbatch_size=local_microbatch,
                gradient_accumulation_steps=1,
                bucket_cap_mib=bucket_cap_mib,
                tokens_per_update=global_sequences * config.context_length,
            ))
    return tuple(candidates)


def project_distributed_full_run(
    *,
    tokens_per_second: float,
    config: Minerva7BFullWeightDdpBenchmarkConfig,
) -> dict[str, float]:
    if not math.isfinite(tokens_per_second) or tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be finite and positive")
    update_only_hours = config.full_training_token_count / tokens_per_second / 3600
    projected_hours = update_only_hours * config.projected_overhead_multiplier
    return {
        "update_only_hours": update_only_hours,
        "projected_hours_with_overhead": projected_hours,
        "projected_cost_usd": projected_hours * config.hourly_rate_usd,
    }


def select_fastest_ddp_candidate(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    fitting = [row for row in rows if row.get("fit_decision") == "pass"]
    if not fitting:
        return None
    return max(fitting, key=lambda row: float(row["tokens_per_second"]))


def benchmark_minerva_7b_full_weight_ddp(
    *,
    repo_root: Path,
    config: Minerva7BFullWeightDdpBenchmarkConfig = (
        Minerva7BFullWeightDdpBenchmarkConfig()
    ),
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """Run a fixed two-rank benchmark and write one report from global rank zero."""
    validate_full_weight_ddp_benchmark_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Minerva full-weight DDP benchmark requires CUDA")
    if not dist.is_available():
        raise RuntimeError("PyTorch distributed support is unavailable")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != config.world_size:
        raise RuntimeError(
            f"benchmark requires exactly {config.world_size} torchrun ranks"
        )
    if torch.cuda.device_count() != config.world_size:
        raise RuntimeError(
            f"benchmark requires exactly {config.world_size} visible CUDA devices"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    try:
        hardware = _validate_hardware(config=config, device=device)
        communication = _measure_nccl_all_reduce(
            config=config,
            device=device,
            rank=rank,
        )
        if rank == 0:
            _report(
                progress,
                "NCCL all-reduce payload={payload:.3f}GiB mean={mean:.2f}ms "
                "bandwidth={bandwidth:.2f}GB/s".format(
                    payload=communication["payload_gib"],
                    mean=communication["mean_milliseconds"],
                    bandwidth=communication["algorithmic_gigabytes_per_second"],
                ),
            )

        dependencies = _load_dependencies()
        windows = load_full_weight_calibration_windows(
            _resolve(repo_root, config.calibration_windows_path)
        )
        maximum_sequences = max(config.global_sequence_counts)
        sequence_pool = _build_sequence_pool(
            windows["training_windows"],
            sequence_count=maximum_sequences,
        )
        torch.manual_seed(config.seed)
        prepare_cuda_memory_measurement(device)
        if rank == 0:
            _report(progress, "loading one unquantized BF16 model replica per GPU")
        model = dependencies["AutoModelForCausalLM"].from_pretrained(
            config.model_id,
            revision=config.revision,
            cache_dir=_resolve(repo_root, config.cache_dir),
            dtype=torch.bfloat16,
            device_map={"": local_rank},
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        model.config.use_cache = False
        model.gradient_checkpointing_disable()
        model_audit = audit_full_weight_model(model)
        if not (
            model_audit["all_weights_trainable"]
            and model_audit["adapter_free"]
            and model_audit["quantization_free"]
            and model_audit["parameter_dtype_counts"]
            == {"bfloat16": model_audit["total_parameter_count"]}
        ):
            raise ValueError("DDP benchmark model failed the full-weight BF16 audit")
        parameters = list(model.parameters())
        optimizer = dependencies["bitsandbytes"].optim.PagedAdamW8bit(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        candidates = build_ddp_throughput_candidates(config)
        rows = []
        benchmark_started_at = time.monotonic()
        for index, candidate in enumerate(candidates, start=1):
            if rank == 0:
                _report(
                    progress,
                    f"candidate {index}/{len(candidates)} start: "
                    f"{candidate.candidate_id}",
                )
            ddp_model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                bucket_cap_mb=candidate.bucket_cap_mib,
                gradient_as_bucket_view=True,
                find_unused_parameters=False,
            )
            row = _benchmark_candidate(
                ddp_model=ddp_model,
                optimizer=optimizer,
                parameters=parameters,
                sequence_pool=sequence_pool,
                candidate=candidate,
                config=config,
                device=device,
                rank=rank,
            )
            del ddp_model
            dist.barrier()
            if rank == 0:
                rows.append(row)
                _report(
                    progress,
                    f"candidate {candidate.candidate_id}: fit={row['fit_decision']} "
                    f"throughput={row['tokens_per_second']:.1f} tokens/s",
                )

        if rank != 0:
            return None
        selected = select_fastest_ddp_candidate(rows)
        report = {
            "benchmark_version": BENCHMARK_VERSION,
            "status": "complete",
            "config": asdict(config),
            "world_size": world_size,
            "hardware": hardware,
            "communication": communication,
            "model_audit": model_audit,
            "attention": {
                "model_attention_implementation": getattr(
                    model.config, "_attn_implementation", None
                ),
                "torch_flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
                "torch_mem_efficient_sdp_enabled": (
                    torch.backends.cuda.mem_efficient_sdp_enabled()
                ),
                "torch_math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
            },
            "candidates": rows,
            "selected_candidate": selected,
            "elapsed_seconds": time.monotonic() - benchmark_started_at,
            "retained_model_checkpoint": False,
            "long_training_automatically_authorized": False,
            "package_versions": _package_versions(dependencies),
        }
        output_path = _resolve(repo_root, config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _validate_hardware(
    *,
    config: Minerva7BFullWeightDdpBenchmarkConfig,
    device: torch.device,
) -> dict[str, Any]:
    properties = cuda_device_properties(device)
    local = {
        "rank": dist.get_rank(),
        "device_index": device.index,
        "gpu_name": cuda_device_name(device),
        "total_memory_mib": properties.total_memory / (1024**2),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "native_bf16_supported": torch.cuda.is_bf16_supported(),
    }
    gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    rows = [row for row in gathered if row is not None]
    if len(rows) != config.world_size:
        raise RuntimeError("failed to gather hardware details from every DDP rank")
    for row in rows:
        if (
            "rtx pro 6000 blackwell" not in str(row["gpu_name"]).lower()
            or float(row["total_memory_mib"]) < config.minimum_total_memory_mib
            or not bool(row["native_bf16_supported"])
        ):
            raise RuntimeError(
                "benchmark requires two RTX PRO 6000 Blackwell 96 GB BF16 GPUs"
            )
    if not torch.cuda.can_device_access_peer(0, 1):
        raise RuntimeError("CUDA peer access is unavailable between the two GPUs")
    return {
        "devices": rows,
        "cuda_peer_access": True,
        "nccl_version": list(torch.cuda.nccl.version()),
    }


def _measure_nccl_all_reduce(
    *,
    config: Minerva7BFullWeightDdpBenchmarkConfig,
    device: torch.device,
    rank: int,
) -> dict[str, Any]:
    element_count = config.communication_payload_mib * 1024**2 // 2
    payload = torch.ones(element_count, dtype=torch.bfloat16, device=device)
    for _ in range(config.communication_warmup_iterations):
        dist.all_reduce(payload)
    synchronize_cuda(device)
    dist.barrier()
    started_at = time.monotonic()
    for _ in range(config.communication_timed_iterations):
        dist.all_reduce(payload)
    synchronize_cuda(device)
    dist.barrier()
    elapsed = time.monotonic() - started_at
    local_elapsed = torch.tensor(elapsed, dtype=torch.float64, device=device)
    dist.all_reduce(local_elapsed, op=dist.ReduceOp.MAX)
    elapsed = float(local_elapsed.item())
    payload_bytes = payload.numel() * payload.element_size()
    del payload, local_elapsed
    torch.cuda.empty_cache()
    result = {
        "ranks": dist.get_world_size(),
        "payload_gib": payload_bytes / (1024**3),
        "warmup_iterations": config.communication_warmup_iterations,
        "timed_iterations": config.communication_timed_iterations,
        "mean_milliseconds": (
            elapsed / config.communication_timed_iterations * 1000
        ),
        "algorithmic_gigabytes_per_second": (
            payload_bytes
            / (elapsed / config.communication_timed_iterations)
            / 1e9
        ),
    }
    if rank != 0:
        return result
    return result


def _benchmark_candidate(
    *,
    ddp_model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
    sequence_pool: torch.Tensor,
    candidate: DdpThroughputCandidate,
    config: Minerva7BFullWeightDdpBenchmarkConfig,
    device: torch.device,
    rank: int,
) -> dict[str, Any]:
    local_sequences = _local_sequences(
        sequence_pool,
        candidate=candidate,
        rank=rank,
        world_size=config.world_size,
    )
    for _ in range(config.warmup_updates):
        _run_update(
            ddp_model=ddp_model,
            optimizer=optimizer,
            parameters=parameters,
            input_ids=local_sequences,
            config=config,
            device=device,
        )

    optimizer.zero_grad(set_to_none=True)
    prepare_cuda_memory_measurement(device)
    dist.barrier()
    timed_started_at = time.monotonic()
    losses = []
    gradient_norms = []
    for _ in range(config.timed_updates):
        loss, gradient_norm = _run_update(
            ddp_model=ddp_model,
            optimizer=optimizer,
            parameters=parameters,
            input_ids=local_sequences,
            config=config,
            device=device,
        )
        losses.append(loss)
        gradient_norms.append(gradient_norm)
    synchronize_cuda(device)
    dist.barrier()
    local_timed_seconds = time.monotonic() - timed_started_at
    timed_tensor = torch.tensor(local_timed_seconds, dtype=torch.float64, device=device)
    dist.all_reduce(timed_tensor, op=dist.ReduceOp.MAX)
    timed_seconds = float(timed_tensor.item())
    optimizer.zero_grad(set_to_none=True)
    synchronize_cuda(device)
    free_bytes, _ = cuda_memory_info(device)
    local_metrics = {
        "rank": rank,
        "peak_allocated_mib": max_cuda_memory_allocated(device) / (1024**2),
        "peak_reserved_mib": max_cuda_memory_reserved(device) / (1024**2),
        "free_memory_after_mib": free_bytes / (1024**2),
        "mean_loss": sum(losses) / len(losses),
        "mean_gradient_norm": sum(gradient_norms) / len(gradient_norms),
    }
    gathered: list[dict[str, Any] | None] = [None] * config.world_size
    dist.all_gather_object(gathered, local_metrics)
    rank_metrics = [row for row in gathered if row is not None]
    total_memory_mib = cuda_device_properties(device).total_memory / (1024**2)
    peak_reserved_mib = max(float(row["peak_reserved_mib"]) for row in rank_metrics)
    minimum_free_mib = min(float(row["free_memory_after_mib"]) for row in rank_metrics)
    reserved_headroom_mib = total_memory_mib - peak_reserved_mib
    numerical_pass = all(
        math.isfinite(float(row[key]))
        for row in rank_metrics
        for key in ("mean_loss", "mean_gradient_norm")
    )
    memory_pass = (
        minimum_free_mib >= config.minimum_headroom_mib
        and reserved_headroom_mib >= config.minimum_headroom_mib
    )
    tokens_per_second = (
        config.timed_updates * candidate.tokens_per_update / timed_seconds
    )
    return {
        **asdict(candidate),
        "status": "ok",
        "fit_decision": "pass" if numerical_pass and memory_pass else "reject",
        "timed_updates": config.timed_updates,
        "timed_seconds": timed_seconds,
        "tokens_per_second": tokens_per_second,
        "rank_metrics": rank_metrics,
        "peak_reserved_mib": peak_reserved_mib,
        "reserved_headroom_mib": reserved_headroom_mib,
        "minimum_free_memory_after_mib": minimum_free_mib,
        "numerical_gate_passed": numerical_pass,
        "memory_gate_passed": memory_pass,
        "projection": project_distributed_full_run(
            tokens_per_second=tokens_per_second,
            config=config,
        ),
        "error": None,
    }


def _run_update(
    *,
    ddp_model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
    input_ids: torch.Tensor,
    config: Minerva7BFullWeightDdpBenchmarkConfig,
    device: torch.device,
) -> tuple[float, float]:
    ddp_model.train()
    optimizer.zero_grad(set_to_none=True)
    batch = input_ids.to(device=device, dtype=torch.long)
    loss = ddp_model(input_ids=batch, labels=batch).loss
    loss_value = float(loss.detach().item())
    if not math.isfinite(loss_value):
        raise FloatingPointError("DDP benchmark produced a non-finite loss")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        config.max_gradient_norm,
    )
    gradient_norm_value = float(gradient_norm.detach().item())
    if not math.isfinite(gradient_norm_value):
        raise FloatingPointError("DDP benchmark produced a non-finite gradient norm")
    optimizer.step()
    synchronize_cuda(device)
    return loss_value, gradient_norm_value


def _local_sequences(
    sequence_pool: torch.Tensor,
    *,
    candidate: DdpThroughputCandidate,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    if candidate.global_sequences_per_update % world_size:
        raise ValueError("candidate cannot be divided across DDP ranks")
    local_count = candidate.global_sequences_per_update // world_size
    start = rank * local_count
    return sequence_pool[start:start + local_count].contiguous()


def _build_sequence_pool(windows: torch.Tensor, *, sequence_count: int) -> torch.Tensor:
    if windows.ndim != 2 or windows.shape[1] != 512:
        raise ValueError("benchmark windows must have shape (rows, 512)")
    repeats = math.ceil(sequence_count / windows.shape[0])
    return windows.repeat((repeats, 1))[:sequence_count].contiguous()


def _load_dependencies() -> dict[str, Any]:
    try:
        import accelerate
        import bitsandbytes
        import transformers
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError("Minerva DDP benchmark dependencies are missing") from error
    return {
        "accelerate": accelerate,
        "bitsandbytes": bitsandbytes,
        "transformers": transformers,
        "AutoModelForCausalLM": AutoModelForCausalLM,
    }


def _package_versions(dependencies: Mapping[str, Any]) -> dict[str, str]:
    return {
        "accelerate": dependencies["accelerate"].__version__,
        "bitsandbytes": dependencies["bitsandbytes"].__version__,
        "torch": torch.__version__,
        "transformers": dependencies["transformers"].__version__,
    }


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
