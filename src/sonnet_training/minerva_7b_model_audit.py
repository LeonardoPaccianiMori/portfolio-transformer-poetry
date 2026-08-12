"""Dependency-light audit proving a Minerva model is full-weight BF16."""

from __future__ import annotations

from typing import Any

import torch


def audit_full_weight_model(model: torch.nn.Module) -> dict[str, Any]:
    """Prove that a loaded model has no adapters, quantization, or frozen weights."""

    named_parameters = list(model.named_parameters())
    if not named_parameters:
        raise ValueError("Minerva model contains no parameters")
    total_parameter_count = sum(parameter.numel() for _, parameter in named_parameters)
    trainable_parameter_count = sum(
        parameter.numel()
        for _, parameter in named_parameters
        if parameter.requires_grad
    )
    frozen_parameter_names = [
        name for name, parameter in named_parameters if not parameter.requires_grad
    ]
    adapter_parameter_names = [
        name
        for name, _ in named_parameters
        if "lora_" in name.lower() or "adapter" in name.lower()
    ]
    quantized = bool(
        getattr(model, "is_loaded_in_4bit", False)
        or getattr(model, "is_loaded_in_8bit", False)
        or getattr(model, "quantization_method", None) is not None
    )
    dtype_counts: dict[str, int] = {}
    for _, parameter in named_parameters:
        key = str(parameter.dtype).removeprefix("torch.")
        dtype_counts[key] = dtype_counts.get(key, 0) + parameter.numel()
    return {
        "total_parameter_count": total_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_parameter_fraction": trainable_parameter_count
        / total_parameter_count,
        "frozen_parameter_names": frozen_parameter_names,
        "adapter_parameter_names": adapter_parameter_names,
        "quantized": quantized,
        "parameter_dtype_counts": dtype_counts,
        "all_weights_trainable": (
            trainable_parameter_count == total_parameter_count
            and not frozen_parameter_names
        ),
        "adapter_free": not adapter_parameter_names,
        "quantization_free": not quantized,
    }
