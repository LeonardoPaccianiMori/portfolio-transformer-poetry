"""Calibrate a conservative 4-bit QLoRA setup for Minerva 3B."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


MINERVA_3B_MODEL_ID = "sapienzanlp/Minerva-3B-base-v1.0"
MINERVA_3B_REVISION = "129ae5366bae3611a1c9f8c68606c38b7de8b055"
MINERVA_QLORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class MinervaQLoRACalibrationConfig:
    """Lock the one-batch hardware calibration before any full training run."""

    model_id: str = MINERVA_3B_MODEL_ID
    revision: str = MINERVA_3B_REVISION
    context_length: int = 512
    batch_size: int = 1
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = MINERVA_QLORA_TARGET_MODULES
    learning_rate: float = 1e-4
    calibration_text: str = "Nel mezzo del cammin di nostra vita"


def validate_calibration_config(config: MinervaQLoRACalibrationConfig) -> None:
    """Reject changes that would turn the calibration into a recipe search."""
    if config.model_id != MINERVA_3B_MODEL_ID:
        raise ValueError("calibration is locked to Minerva-3B-base-v1.0")
    if config.revision != MINERVA_3B_REVISION:
        raise ValueError("calibration must use the recorded Minerva 3B revision")
    if config.context_length != 512:
        raise ValueError("calibration context_length is locked to 512")
    if config.batch_size != 1:
        raise ValueError("calibration batch_size is locked to 1")
    if config.lora_rank != 16 or config.lora_alpha != 32:
        raise ValueError("calibration LoRA rank and alpha are locked to 16 and 32")
    if config.lora_dropout != 0.05:
        raise ValueError("calibration LoRA dropout is locked to 0.05")
    if config.target_modules != MINERVA_QLORA_TARGET_MODULES:
        raise ValueError("calibration target modules differ from the recorded plan")
    if config.learning_rate <= 0:
        raise ValueError("calibration learning_rate must be positive")
    if not config.calibration_text.strip():
        raise ValueError("calibration_text must not be empty")


def build_calibration_report(
    *,
    config: MinervaQLoRACalibrationConfig,
    device: torch.device,
    gpu_name: str,
    loss: float,
    total_parameter_count: int,
    trainable_parameter_count: int,
    peak_allocated_mib: float,
    peak_reserved_mib: float,
    package_versions: dict[str, str],
) -> dict[str, Any]:
    """Build the reproducible result record for one successful calibration."""
    return {
        "calibration_type": "minerva_3b_4bit_qlora_one_batch",
        "config": asdict(config),
        "device": str(device),
        "gpu_name": gpu_name,
        "loss": loss,
        "total_parameter_count": total_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_parameter_fraction": trainable_parameter_count
        / total_parameter_count,
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "quantization": {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "double_quantization": True,
            "compute_dtype": "float16",
        },
        "gradient_checkpointing": True,
        "optimizer": "PagedAdamW8bit",
        "package_versions": package_versions,
    }


def write_calibration_report(path: Path, report: dict[str, Any]) -> None:
    """Write the local calibration record without exposing model weights."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def calibrate_minerva_qlora(
    *,
    config: MinervaQLoRACalibrationConfig,
    cache_dir: Path,
    output_path: Path,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    """Load Minerva 3B in 4-bit form and execute one adapter optimizer update."""
    validate_calibration_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Minerva QLoRA calibration requires an available CUDA GPU")

    try:
        import accelerate
        import bitsandbytes as bnb
        import peft
        import transformers
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            "Minerva QLoRA dependencies are missing; install "
            "requirements/minerva_qlora.txt into .venv first"
        ) from error

    device = torch.device("cuda:0")
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    progress("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.revision,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    progress("loading Minerva 3B in 4-bit NF4")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.revision,
        cache_dir=cache_dir,
        quantization_config=quantization,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    model.config.use_cache = False

    progress("attaching LoRA adapters and enabling gradient checkpointing")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    model = get_peft_model(
        model,
        LoraConfig(
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
    optimizer = bnb.optim.PagedAdamW8bit(
        trainable_parameters,
        lr=config.learning_rate,
    )

    encoded = tokenizer(
        [config.calibration_text] * config.batch_size,
        max_length=config.context_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    progress("running one forward and backward pass")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    ).loss
    loss.backward()

    progress("running one adapter optimizer update")
    optimizer.step()
    torch.cuda.synchronize(device)

    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    report = build_calibration_report(
        config=config,
        device=device,
        gpu_name=torch.cuda.get_device_name(device),
        loss=float(loss.item()),
        total_parameter_count=total_parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        peak_allocated_mib=torch.cuda.max_memory_allocated(device) / (1024**2),
        peak_reserved_mib=torch.cuda.max_memory_reserved(device) / (1024**2),
        package_versions={
            "accelerate": accelerate.__version__,
            "bitsandbytes": bnb.__version__,
            "peft": peft.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    )
    write_calibration_report(output_path, report)
    return report
