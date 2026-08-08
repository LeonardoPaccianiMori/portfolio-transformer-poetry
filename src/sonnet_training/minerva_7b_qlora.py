"""Bounded 4-bit QLoRA memory calibration for Minerva 7B Instruct."""

from __future__ import annotations

import json
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

MINERVA_7B_INSTRUCT_MODEL_ID = "sapienzanlp/Minerva-7B-instruct-v1.0"
MINERVA_7B_INSTRUCT_REVISION = "d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d"
MINERVA_7B_QLORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)
MINIMUM_CUDA_HEADROOM_MIB = 512.0


@dataclass(frozen=True)
class Minerva7BQLoRACalibrationConfig:
    """Freeze the single permitted 7B Instruct training-memory calibration."""

    model_id: str = MINERVA_7B_INSTRUCT_MODEL_ID
    revision: str = MINERVA_7B_INSTRUCT_REVISION
    context_length: int = 512
    batch_size: int = 1
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = MINERVA_7B_QLORA_TARGET_MODULES
    learning_rate: float = 2e-5


def validate_minerva_7b_calibration_config(
    config: Minerva7BQLoRACalibrationConfig,
) -> None:
    """Reject changes that would turn one calibration into a hardware sweep."""
    expected = Minerva7BQLoRACalibrationConfig()
    if config != expected:
        raise ValueError("Minerva 7B QLoRA calibration configuration is locked")


def build_minerva_7b_calibration_report(
    *,
    config: Minerva7BQLoRACalibrationConfig,
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
    package_versions: dict[str, str],
    error: str | None = None,
) -> dict[str, Any]:
    """Build either the successful measurement or the completed OOM result."""
    if status not in {"ok", "out_of_memory"}:
        raise ValueError("unsupported calibration status")
    reserved_headroom_mib = total_gpu_memory_mib - peak_reserved_mib
    fit = (
        status == "ok"
        and reserved_headroom_mib >= MINIMUM_CUDA_HEADROOM_MIB
        and free_memory_after_mib >= MINIMUM_CUDA_HEADROOM_MIB
    )
    return {
        "calibration_type": "minerva_7b_instruct_4bit_qlora_one_update_v1",
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
        "minimum_required_headroom_mib": MINIMUM_CUDA_HEADROOM_MIB,
        "local_training_fit_decision": "pass" if fit else "reject",
        "quantization": {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "double_quantization": True,
            "compute_dtype": "float16",
        },
        "gradient_checkpointing": True,
        "optimizer": "PagedAdamW8bit",
        "package_versions": package_versions,
        "error": error,
    }


def write_minerva_7b_calibration_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def calibrate_minerva_7b_qlora(
    *,
    config: Minerva7BQLoRACalibrationConfig,
    cache_dir: Path,
    output_path: Path,
    progress: Callable[[str], None],
) -> dict[str, Any]:
    """Run one representative adapter update and record fit or clean OOM."""
    validate_minerva_7b_calibration_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Minerva 7B QLoRA calibration requires an available CUDA GPU")
    dependencies = _load_dependencies()
    device = torch.device("cuda:0")
    device_index = 0
    properties = cuda_device_properties(device)
    total_gpu_memory_mib = properties.total_memory / (1024**2)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prepare_cuda_memory_measurement(device)

    try:
        report = _run_calibration(
            config=config,
            cache_dir=cache_dir,
            progress=progress,
            dependencies=dependencies,
            device=device,
            device_index=device_index,
            total_gpu_memory_mib=total_gpu_memory_mib,
        )
    except (torch.OutOfMemoryError, RuntimeError) as error:
        if not _is_out_of_memory(error):
            raise
        peak_allocated = max_cuda_memory_allocated(device) / (1024**2)
        peak_reserved = max_cuda_memory_reserved(device) / (1024**2)
        torch.cuda.empty_cache()
        free_bytes, _ = cuda_memory_info(device)
        report = build_minerva_7b_calibration_report(
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
            package_versions=_package_versions(dependencies),
            error=str(error),
        )
        progress("calibration reached the fixed GPU memory limit")

    write_minerva_7b_calibration_report(output_path, report)
    return report


def _run_calibration(
    *,
    config: Minerva7BQLoRACalibrationConfig,
    cache_dir: Path,
    progress: Callable[[str], None],
    dependencies: dict[str, Any],
    device: torch.device,
    device_index: int,
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
    input_ids, attention_mask, labels = _build_calibration_batch(
        tokenizer=tokenizer,
        context_length=config.context_length,
        device=device,
    )

    progress("stage 2/5: loading 7B Instruct weights in 4-bit NF4")
    quantization = dependencies["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = dependencies["AutoModelForCausalLM"].from_pretrained(
        config.model_id,
        revision=config.revision,
        cache_dir=cache_dir,
        quantization_config=quantization,
        torch_dtype=torch.float16,
        device_map={"": device_index},
    )
    model.config.use_cache = False

    progress("stage 3/5: attaching rank-8 attention adapters")
    model = dependencies["prepare_model_for_kbit_training"](
        model,
        use_gradient_checkpointing=True,
    )
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
    loss = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    ).loss
    loss.backward()

    progress("stage 5/5: running one adapter optimizer update")
    optimizer.step()
    synchronize_cuda(device)
    free_bytes, _ = cuda_memory_info(device)
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    return build_minerva_7b_calibration_report(
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
        package_versions=_package_versions(dependencies),
    )


def _build_calibration_batch(
    *,
    tokenizer: Any,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    instruction = (
        "Componi un sonetto italiano di quattordici versi sul ritorno della "
        "luce dopo una lunga notte. Restituisci soltanto il sonetto."
    )
    response = "\n".join([
        "Dopo la notte torna chiara luce,",
        "e sopra i tetti il nuovo giorno appare;",
        "la mente stanca impara a respirare,",
        "mentre ogni ombra lentamente si riduce.",
        "Un vento lieve il primo canto adduce,",
        "e desta il campo, il colle e il largo mare;",
        "così nel petto ricomincia a stare",
        "la quieta speranza che conduce.",
        "Non fu perduto il tempo del dolore,",
        "se nel silenzio il cuore ebbe memoria",
        "di quanto resta oltre la paura.",
        "Ora il mattino schiude il suo colore,",
        "e fa del passo incerto una vittoria,",
        "serbando in noi la notte e la sua cura.",
    ])
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        max_length=context_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[:, :len(prompt_ids)] = -100
    labels[attention_mask == 0] = -100
    if not (labels != -100).any():
        raise ValueError("calibration example has no supervised response tokens")
    return input_ids, attention_mask, labels


def _load_dependencies() -> dict[str, Any]:
    try:
        import accelerate
        import bitsandbytes
        import peft
        import transformers
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            "Minerva QLoRA dependencies are missing; use the project .venv"
        ) from error
    return {
        "accelerate": accelerate,
        "bitsandbytes": bitsandbytes,
        "peft": peft,
        "transformers": transformers,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


def _package_versions(dependencies: dict[str, Any]) -> dict[str, str]:
    return {
        "accelerate": dependencies["accelerate"].__version__,
        "bitsandbytes": dependencies["bitsandbytes"].__version__,
        "peft": dependencies["peft"].__version__,
        "torch": torch.__version__,
        "transformers": dependencies["transformers"].__version__,
    }


def _is_out_of_memory(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()
