"""Exact-recipe calibration for Minerva 7B historical FP16 LoRA training."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
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
from sonnet_training.minerva_7b_historical_lora import (
    Minerva7BHistoricalLoRAConfig,
    _load_dependencies,
    _stream_window,
    build_historical_training_plan,
)
from sonnet_training.minerva_7b_staged_data import load_staged_tensor


@dataclass(frozen=True)
class HistoricalCalibrationConfig:
    """Freeze a short representative throughput and memory check."""

    warmup_updates: int = 2
    timed_updates: int = 10
    minimum_free_memory_mib: float = 4096.0


def validate_calibration_config(config: HistoricalCalibrationConfig) -> None:
    if config != HistoricalCalibrationConfig():
        raise ValueError("historical calibration configuration is locked")


def calibrate_historical_lora(
    *,
    repo_root: Path,
    output_path: Path,
    config: HistoricalCalibrationConfig = HistoricalCalibrationConfig(),
    recipe: Minerva7BHistoricalLoRAConfig = Minerva7BHistoricalLoRAConfig(),
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run twelve exact-shape updates without modifying the future run output."""
    validate_calibration_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("historical LoRA calibration requires CUDA")
    dependencies = _load_dependencies()
    device = torch.device(recipe.device)
    properties = cuda_device_properties(device)
    total_memory_mib = properties.total_memory / (1024**2)
    encoded_dir = repo_root / recipe.encoded_dir
    historical = load_staged_tensor(encoded_dir / "historical_train.pt", dimensions=1)
    replay = load_staged_tensor(encoded_dir / "modern_replay_train.pt", dimensions=1)
    plan = build_historical_training_plan(
        config=recipe,
        historical_token_count=historical.numel(),
        replay_token_count=replay.numel(),
    )

    _report(progress, "loading unquantized FP16 model")
    model = dependencies["AutoModelForCausalLM"].from_pretrained(
        recipe.model_id,
        revision=recipe.revision,
        cache_dir=repo_root / recipe.cache_dir,
        dtype=torch.float16,
        device_map={"": device.index or 0},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model = dependencies["get_peft_model"](
        model,
        dependencies["LoraConfig"](
            task_type="CAUSAL_LM",
            r=recipe.lora_rank,
            lora_alpha=recipe.lora_alpha,
            lora_dropout=recipe.lora_dropout,
            bias="none",
            target_modules=list(recipe.target_modules),
        ),
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=recipe.learning_rate,
        weight_decay=recipe.weight_decay,
        foreach=False,
    )
    prepare_cuda_memory_measurement(device)

    losses = []
    timed_seconds = 0.0
    total_updates = config.warmup_updates + config.timed_updates
    for update in range(total_updates):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        update_started_at = time.monotonic()
        microbatch_losses = []
        for microbatch in range(recipe.historical_microbatches_per_update):
            input_ids, labels = _stream_window(
                historical,
                window_index=(
                    update * recipe.historical_microbatches_per_update + microbatch
                ),
                context_length=recipe.context_length,
                device=device,
            )
            loss = model(input_ids=input_ids, labels=labels).loss
            (loss / 8).backward()
            microbatch_losses.append(float(loss.detach().item()))
        replay_input, replay_labels = _stream_window(
            replay,
            window_index=update,
            context_length=recipe.context_length,
            device=device,
        )
        replay_loss = model(input_ids=replay_input, labels=replay_labels).loss
        (replay_loss / 8).backward()
        microbatch_losses.append(float(replay_loss.detach().item()))
        torch.nn.utils.clip_grad_norm_(parameters, recipe.max_gradient_norm)
        optimizer.step()
        synchronize_cuda(device)
        update_seconds = time.monotonic() - update_started_at
        if update >= config.warmup_updates:
            timed_seconds += update_seconds
            losses.append(sum(microbatch_losses) / len(microbatch_losses))
        _report(progress, f"completed calibration update {update + 1}/{total_updates}")

    free_bytes, _ = cuda_memory_info(device)
    peak_allocated_mib = max_cuda_memory_allocated(device) / (1024**2)
    peak_reserved_mib = max_cuda_memory_reserved(device) / (1024**2)
    free_memory_mib = free_bytes / (1024**2)
    tokens_per_update = recipe.context_length * 8
    tokens_per_second = config.timed_updates * tokens_per_update / timed_seconds
    fit = free_memory_mib >= config.minimum_free_memory_mib
    report = {
        "calibration_version": "minerva_7b_historical_fp16_lora_exact_v1",
        "status": "pass" if fit else "reject",
        "gpu_name": cuda_device_name(device),
        "total_gpu_memory_mib": total_memory_mib,
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "free_memory_after_mib": free_memory_mib,
        "minimum_free_memory_mib": config.minimum_free_memory_mib,
        "mean_timed_loss": sum(losses) / len(losses),
        "timed_updates": config.timed_updates,
        "timed_seconds": timed_seconds,
        "tokens_per_update": tokens_per_update,
        "tokens_per_second": tokens_per_second,
        "estimated_seconds_for_full_plan": (
            timed_seconds * plan.planned_updates / config.timed_updates
        ),
        "planned_updates": plan.planned_updates,
        "calibration_config": asdict(config),
        "recipe_config": asdict(recipe),
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "optimizer": "torch.optim.AdamW",
        "base_weight_dtype": "float16",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
