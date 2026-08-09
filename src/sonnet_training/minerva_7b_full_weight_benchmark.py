"""Fixed-effective-batch throughput benchmark for full-weight Minerva 7B."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

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
    MINIMUM_H100_MEMORY_MIB,
    MINIMUM_POST_OPTIMIZER_HEADROOM_MIB,
    audit_full_weight_model,
)
from sonnet_training.minerva_7b_full_weight_data import (
    load_full_weight_calibration_windows,
)
from sonnet_training.minerva_7b_qlora import (
    MINERVA_7B_INSTRUCT_MODEL_ID,
    MINERVA_7B_INSTRUCT_REVISION,
    is_cuda_out_of_memory,
)


BENCHMARK_VERSION = "minerva_7b_full_weight_h100_throughput_v1"
FULL_TRAINING_TOKEN_COUNT = 351_271_297


@dataclass(frozen=True)
class Minerva7BFullWeightBenchmarkConfig:
    """Freeze the bounded H100 comparison approved after memory calibration."""

    model_id: str = MINERVA_7B_INSTRUCT_MODEL_ID
    revision: str = MINERVA_7B_INSTRUCT_REVISION
    cache_dir: str = "data/local/minerva_qlora/huggingface"
    calibration_windows_path: str = (
        "data/local/minerva_7b_full_weight/encoded/calibration_windows.pt"
    )
    output_path: str = (
        "data/local/minerva_7b_full_weight/full_weight_h100_benchmark.json"
    )
    device: str = "cuda:0"
    context_length: int = 512
    effective_sequences_per_update: int = 8
    microbatch_sizes: tuple[int, ...] = (1, 2, 4, 8)
    gradient_checkpointing_modes: tuple[bool, ...] = (True, False)
    warmup_updates: int = 1
    timed_updates: int = 5
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    max_gradient_norm: float = 1.0
    minimum_total_memory_mib: int = MINIMUM_H100_MEMORY_MIB
    minimum_headroom_mib: int = MINIMUM_POST_OPTIMIZER_HEADROOM_MIB
    hourly_rate_usd: float = 2.12
    full_training_token_count: int = FULL_TRAINING_TOKEN_COUNT
    projected_overhead_multiplier: float = 1.15
    seed: int = 1337


@dataclass(frozen=True)
class ThroughputCandidate:
    candidate_id: str
    microbatch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    tokens_per_update: int


def validate_full_weight_benchmark_config(
    config: Minerva7BFullWeightBenchmarkConfig,
) -> None:
    if config != Minerva7BFullWeightBenchmarkConfig():
        raise ValueError("Minerva 7B full-weight benchmark configuration is locked")


def build_throughput_candidates(
    config: Minerva7BFullWeightBenchmarkConfig,
) -> tuple[ThroughputCandidate, ...]:
    """Build comparable candidates with the same effective token batch."""
    candidates = []
    for checkpointing in config.gradient_checkpointing_modes:
        mode = "gc_on" if checkpointing else "gc_off"
        for microbatch_size in config.microbatch_sizes:
            if config.effective_sequences_per_update % microbatch_size:
                raise ValueError("microbatch size must divide effective sequences")
            accumulation = config.effective_sequences_per_update // microbatch_size
            candidates.append(ThroughputCandidate(
                candidate_id=f"{mode}_micro{microbatch_size}",
                microbatch_size=microbatch_size,
                gradient_accumulation_steps=accumulation,
                gradient_checkpointing=checkpointing,
                tokens_per_update=(
                    config.effective_sequences_per_update * config.context_length
                ),
            ))
    return tuple(candidates)


def project_full_run(
    *,
    tokens_per_second: float,
    config: Minerva7BFullWeightBenchmarkConfig,
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


def select_fastest_fit_candidate(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    fitting = [row for row in rows if row.get("fit_decision") == "pass"]
    if not fitting:
        return None
    return max(fitting, key=lambda row: float(row["tokens_per_second"]))


def benchmark_minerva_7b_full_weight(
    *,
    repo_root: Path,
    config: Minerva7BFullWeightBenchmarkConfig = (
        Minerva7BFullWeightBenchmarkConfig()
    ),
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Benchmark eight candidates without retaining the modified model."""
    validate_full_weight_benchmark_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Minerva full-weight benchmark requires CUDA")
    device = torch.device(config.device)
    properties = cuda_device_properties(device)
    total_gpu_memory_mib = properties.total_memory / (1024**2)
    gpu_name = cuda_device_name(device)
    if (
        "h100" not in gpu_name.lower()
        or total_gpu_memory_mib < config.minimum_total_memory_mib
        or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("benchmark requires the approved H100 80 GB BF16 class")

    dependencies = _load_dependencies()
    windows = load_full_weight_calibration_windows(
        _resolve(repo_root, config.calibration_windows_path)
    )
    sequence_pool = _build_sequence_pool(
        windows["training_windows"],
        sequence_count=config.effective_sequences_per_update,
    )
    torch.manual_seed(config.seed)
    prepare_cuda_memory_measurement(device)
    _report(progress, "loading unquantized Minerva 7B in BF16 with SDPA")
    model = dependencies["AutoModelForCausalLM"].from_pretrained(
        config.model_id,
        revision=config.revision,
        cache_dir=_resolve(repo_root, config.cache_dir),
        dtype=torch.bfloat16,
        device_map={"": device.index if device.index is not None else 0},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model_audit = audit_full_weight_model(model)
    if not (
        model_audit["all_weights_trainable"]
        and model_audit["adapter_free"]
        and model_audit["quantization_free"]
        and model_audit["parameter_dtype_counts"]
        == {"bfloat16": model_audit["total_parameter_count"]}
    ):
        raise ValueError("benchmark model failed the full-weight BF16 audit")
    parameters = list(model.parameters())
    optimizer = dependencies["bitsandbytes"].optim.PagedAdamW8bit(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    attention = {
        "model_attention_implementation": getattr(
            model.config, "_attn_implementation", None
        ),
        "torch_flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "torch_mem_efficient_sdp_enabled": (
            torch.backends.cuda.mem_efficient_sdp_enabled()
        ),
        "torch_math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
    }

    candidates = build_throughput_candidates(config)
    rows = []
    benchmark_started_at = time.monotonic()
    for index, candidate in enumerate(candidates, start=1):
        _report(
            progress,
            f"candidate {index}/{len(candidates)} start: {candidate.candidate_id} "
            f"accumulation={candidate.gradient_accumulation_steps}",
        )
        row = _benchmark_candidate(
            model=model,
            optimizer=optimizer,
            parameters=parameters,
            sequence_pool=sequence_pool,
            candidate=candidate,
            config=config,
            device=device,
            total_gpu_memory_mib=total_gpu_memory_mib,
            progress=progress,
        )
        rows.append(row)
        _report(
            progress,
            "candidate {candidate}: status={status} fit={fit} throughput={speed}".format(
                candidate=candidate.candidate_id,
                status=row["status"],
                fit=row["fit_decision"],
                speed=(
                    f"{row['tokens_per_second']:.1f} tokens/s"
                    if row["tokens_per_second"] is not None
                    else "unavailable"
                ),
            ),
        )

    selected = select_fastest_fit_candidate(rows)
    report = {
        "benchmark_version": BENCHMARK_VERSION,
        "status": "complete",
        "config": asdict(config),
        "gpu_name": gpu_name,
        "total_gpu_memory_mib": total_gpu_memory_mib,
        "native_bf16_supported": torch.cuda.is_bf16_supported(),
        "model_audit": model_audit,
        "attention": attention,
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


def _benchmark_candidate(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
    sequence_pool: torch.Tensor,
    candidate: ThroughputCandidate,
    config: Minerva7BFullWeightBenchmarkConfig,
    device: torch.device,
    total_gpu_memory_mib: float,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    if candidate.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        model.gradient_checkpointing_disable()
    model.config.use_cache = False
    try:
        for warmup in range(config.warmup_updates):
            _run_update(
                model=model,
                optimizer=optimizer,
                parameters=parameters,
                sequence_pool=sequence_pool,
                candidate=candidate,
                config=config,
                device=device,
            )
            _report(
                progress,
                f"{candidate.candidate_id}: warmup {warmup + 1}/{config.warmup_updates}",
            )
        optimizer.zero_grad(set_to_none=True)
        prepare_cuda_memory_measurement(device)
        synchronize_cuda(device)
        timed_started_at = time.monotonic()
        losses = []
        gradient_norms = []
        for update in range(config.timed_updates):
            loss, gradient_norm = _run_update(
                model=model,
                optimizer=optimizer,
                parameters=parameters,
                sequence_pool=sequence_pool,
                candidate=candidate,
                config=config,
                device=device,
            )
            losses.append(loss)
            gradient_norms.append(gradient_norm)
            _report(
                progress,
                f"{candidate.candidate_id}: timed update "
                f"{update + 1}/{config.timed_updates}",
            )
        synchronize_cuda(device)
        timed_seconds = time.monotonic() - timed_started_at
        optimizer.zero_grad(set_to_none=True)
        synchronize_cuda(device)
        free_bytes, _ = cuda_memory_info(device)
        peak_allocated_mib = max_cuda_memory_allocated(device) / (1024**2)
        peak_reserved_mib = max_cuda_memory_reserved(device) / (1024**2)
        free_memory_mib = free_bytes / (1024**2)
        reserved_headroom_mib = total_gpu_memory_mib - peak_reserved_mib
        numerical_pass = all(math.isfinite(value) for value in losses + gradient_norms)
        memory_pass = (
            free_memory_mib >= config.minimum_headroom_mib
            and reserved_headroom_mib >= config.minimum_headroom_mib
        )
        tokens_per_second = (
            config.timed_updates * candidate.tokens_per_update / timed_seconds
        )
        fit = numerical_pass and memory_pass
        return {
            **asdict(candidate),
            "status": "ok",
            "fit_decision": "pass" if fit else "reject",
            "mean_loss": sum(losses) / len(losses),
            "mean_gradient_norm": sum(gradient_norms) / len(gradient_norms),
            "timed_updates": config.timed_updates,
            "timed_seconds": timed_seconds,
            "tokens_per_second": tokens_per_second,
            "peak_allocated_mib": peak_allocated_mib,
            "peak_reserved_mib": peak_reserved_mib,
            "reserved_headroom_mib": reserved_headroom_mib,
            "free_memory_after_mib": free_memory_mib,
            "numerical_gate_passed": numerical_pass,
            "memory_gate_passed": memory_pass,
            "projection": project_full_run(
                tokens_per_second=tokens_per_second,
                config=config,
            ),
            "error": None,
        }
    except (torch.OutOfMemoryError, RuntimeError) as error:
        if not is_cuda_out_of_memory(error):
            raise
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        synchronize_cuda(device)
        free_bytes, _ = cuda_memory_info(device)
        return {
            **asdict(candidate),
            "status": "out_of_memory",
            "fit_decision": "reject",
            "mean_loss": None,
            "mean_gradient_norm": None,
            "timed_updates": 0,
            "timed_seconds": None,
            "tokens_per_second": None,
            "peak_allocated_mib": max_cuda_memory_allocated(device) / (1024**2),
            "peak_reserved_mib": max_cuda_memory_reserved(device) / (1024**2),
            "reserved_headroom_mib": None,
            "free_memory_after_mib": free_bytes / (1024**2),
            "numerical_gate_passed": False,
            "memory_gate_passed": False,
            "projection": None,
            "error": str(error),
        }


def _run_update(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
    sequence_pool: torch.Tensor,
    candidate: ThroughputCandidate,
    config: Minerva7BFullWeightBenchmarkConfig,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for start in range(0, config.effective_sequences_per_update, candidate.microbatch_size):
        input_ids = sequence_pool[
            start:start + candidate.microbatch_size
        ].to(device=device, dtype=torch.long)
        loss = model(input_ids=input_ids, labels=input_ids).loss
        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value):
            raise FloatingPointError("benchmark produced a non-finite loss")
        (loss / candidate.gradient_accumulation_steps).backward()
        losses.append(loss_value)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        config.max_gradient_norm,
    )
    gradient_norm_value = float(gradient_norm.detach().item())
    if not math.isfinite(gradient_norm_value):
        raise FloatingPointError("benchmark produced a non-finite gradient norm")
    optimizer.step()
    synchronize_cuda(device)
    return sum(losses) / len(losses), gradient_norm_value


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
        raise RuntimeError("Minerva benchmark dependencies are missing") from error
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
