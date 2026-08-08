"""Unquantized FP16 LoRA memory calibration for remote Minerva 7B training."""

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
from sonnet_training.minerva_7b_qlora import (
    MINERVA_7B_INSTRUCT_MODEL_ID,
    MINERVA_7B_INSTRUCT_REVISION,
    MINERVA_7B_QLORA_TARGET_MODULES,
    build_minerva_7b_calibration_batch,
    is_cuda_out_of_memory,
    load_minerva_7b_dependencies,
    minerva_7b_package_versions,
)


MINERVA_7B_FP16_MINIMUM_HEADROOM_MIB = 4096.0


@dataclass(frozen=True)
class Minerva7BFP16LoRACalibrationConfig:
    """Freeze the remote unquantized FP16 LoRA calibration recipe."""

    model_id: str = MINERVA_7B_INSTRUCT_MODEL_ID
    revision: str = MINERVA_7B_INSTRUCT_REVISION
    context_length: int = 512
    batch_size: int = 1
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = MINERVA_7B_QLORA_TARGET_MODULES
    learning_rate: float = 2e-5


def validate_minerva_7b_fp16_lora_config(
    config: Minerva7BFP16LoRACalibrationConfig,
) -> None:
    """Reject changes that would turn the remote calibration into a sweep."""
    if config != Minerva7BFP16LoRACalibrationConfig():
        raise ValueError("Minerva 7B FP16 LoRA calibration configuration is locked")


def build_minerva_7b_fp16_lora_report(
    *,
    config: Minerva7BFP16LoRACalibrationConfig,
    status: str,
    device: torch.device,
    gpu_name: str,
    total_gpu_memory_mib: float,
    peak_allocated_mib: float,
    peak_reserved_mib: float,
    free_memory_after_mib: float,
    loss: float | None,
    total_parameter_count: int | None,
    trainable_parameter_count: int | None,
    optimizer_update_seconds: float | None,
    processed_tokens: int | None,
    package_versions: dict[str, str],
    error: str | None = None,
) -> dict[str, Any]:
    """Build the successful measurement or a completed OOM result."""
    if status not in {"ok", "out_of_memory"}:
        raise ValueError("unsupported calibration status")
    reserved_headroom_mib = total_gpu_memory_mib - peak_reserved_mib
    fit = (
        status == "ok"
        and reserved_headroom_mib >= MINERVA_7B_FP16_MINIMUM_HEADROOM_MIB
        and free_memory_after_mib >= MINERVA_7B_FP16_MINIMUM_HEADROOM_MIB
    )
    return {
        "calibration_type": "minerva_7b_instruct_fp16_lora_one_update_v1",
        "status": status,
        "config": asdict(config),
        "device": str(device),
        "gpu_name": gpu_name,
        "loss": loss,
        "total_parameter_count": total_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_parameter_fraction": (
            trainable_parameter_count / total_parameter_count
            if trainable_parameter_count is not None
            and total_parameter_count is not None
            else None
        ),
        "total_gpu_memory_mib": total_gpu_memory_mib,
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "reserved_headroom_mib": reserved_headroom_mib,
        "free_memory_after_mib": free_memory_after_mib,
        "minimum_required_headroom_mib": MINERVA_7B_FP16_MINIMUM_HEADROOM_MIB,
        "remote_training_fit_decision": "pass" if fit else "reject",
        "weight_loading": {
            "quantized": False,
            "parameter_dtype": "float16",
            "compute_dtype": "float16",
        },
        "gradient_checkpointing": True,
        "gradient_checkpointing_use_reentrant": False,
        "optimizer": "PagedAdamW8bit",
        "optimizer_update_seconds": optimizer_update_seconds,
        "processed_tokens": processed_tokens,
        "tokens_per_second": (
            processed_tokens / optimizer_update_seconds
            if processed_tokens is not None
            and optimizer_update_seconds is not None
            and optimizer_update_seconds > 0
            else None
        ),
        "package_versions": package_versions,
        "error": error,
    }


def calibrate_minerva_7b_fp16_lora(
    *,
    config: Minerva7BFP16LoRACalibrationConfig,
    cache_dir: Path,
    output_path: Path,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    """Run one representative unquantized adapter update and record memory."""
    validate_minerva_7b_fp16_lora_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Minerva 7B FP16 LoRA calibration requires a CUDA GPU")
    dependencies = load_minerva_7b_dependencies()
    device = torch.device("cuda:0")
    properties = cuda_device_properties(device)
    total_gpu_memory_mib = properties.total_memory / (1024**2)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prepare_cuda_memory_measurement(device)

    try:
        report = _run_fp16_calibration(
            config=config,
            cache_dir=cache_dir,
            progress=progress,
            dependencies=dependencies,
            device=device,
            total_gpu_memory_mib=total_gpu_memory_mib,
        )
    except (torch.OutOfMemoryError, RuntimeError) as error:
        if not is_cuda_out_of_memory(error):
            raise
        peak_allocated = max_cuda_memory_allocated(device) / (1024**2)
        peak_reserved = max_cuda_memory_reserved(device) / (1024**2)
        torch.cuda.empty_cache()
        free_bytes, _ = cuda_memory_info(device)
        report = build_minerva_7b_fp16_lora_report(
            config=config,
            status="out_of_memory",
            device=device,
            gpu_name=cuda_device_name(device),
            total_gpu_memory_mib=total_gpu_memory_mib,
            peak_allocated_mib=peak_allocated,
            peak_reserved_mib=peak_reserved,
            free_memory_after_mib=free_bytes / (1024**2),
            loss=None,
            total_parameter_count=None,
            trainable_parameter_count=None,
            optimizer_update_seconds=None,
            processed_tokens=None,
            package_versions=minerva_7b_package_versions(dependencies),
            error=str(error),
        )
        progress("calibration reached the GPU memory limit")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _run_fp16_calibration(
    *,
    config: Minerva7BFP16LoRACalibrationConfig,
    cache_dir: Path,
    progress: Callable[[str], None],
    dependencies: dict[str, Any],
    device: torch.device,
    total_gpu_memory_mib: float,
) -> dict[str, Any]:
    progress("stage 1/5: loading tokenizer and rendering one chat example")
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        config.model_id,
        revision=config.revision,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    input_ids, attention_mask, labels = build_minerva_7b_calibration_batch(
        tokenizer=tokenizer,
        context_length=config.context_length,
        device=device,
    )
    processed_tokens = int(attention_mask.sum().item())

    progress("stage 2/5: loading unquantized 7B Instruct weights in FP16")
    model = dependencies["AutoModelForCausalLM"].from_pretrained(
        config.model_id,
        revision=config.revision,
        cache_dir=cache_dir,
        dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    progress("stage 3/5: attaching rank-8 attention adapters")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model = dependencies["get_peft_model"](
        model,
        dependencies["LoraConfig"](
            task_type="CAUSAL_LM",
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=list(config.target_modules),
        ),
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = dependencies["bitsandbytes"].optim.PagedAdamW8bit(
        trainable_parameters,
        lr=config.learning_rate,
    )

    progress("stage 4/5: running forward and backward passes")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    synchronize_cuda(device)
    update_started_at = time.monotonic()
    loss = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    ).loss
    loss.backward()

    progress("stage 5/5: running one adapter optimizer update")
    optimizer.step()
    synchronize_cuda(device)
    optimizer_update_seconds = time.monotonic() - update_started_at
    free_bytes, _ = cuda_memory_info(device)
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    return build_minerva_7b_fp16_lora_report(
        config=config,
        status="ok",
        device=device,
        gpu_name=cuda_device_name(device),
        total_gpu_memory_mib=total_gpu_memory_mib,
        peak_allocated_mib=max_cuda_memory_allocated(device) / (1024**2),
        peak_reserved_mib=max_cuda_memory_reserved(device) / (1024**2),
        free_memory_after_mib=free_bytes / (1024**2),
        loss=float(loss.item()),
        total_parameter_count=total_parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        optimizer_update_seconds=optimizer_update_seconds,
        processed_tokens=processed_tokens,
        package_versions=minerva_7b_package_versions(dependencies),
    )
