"""Explicit conversion from a LayerNorm parent to an RMSNorm fine-tuning model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer
from sonnet_model.transformer import CausalTransformerLanguageModel


VOCABULARY_STATE_KEYS = {
    "embedding.token_embedding.weight",
    "output_projection.weight",
    "output_projection.bias",
}


def initialize_rms_norm_conversion_from_parent(
    *,
    checkpoint_path: Path,
    tokenizer: BytePairEncodingTokenizer,
    learning_rate: float,
    device: torch.device,
) -> tuple[
    CausalTransformerLanguageModel,
    torch.optim.Optimizer,
    dict[str, Any],
    dict[str, Any],
]:
    """Convert a LayerNorm parent into an RMSNorm model with fresh AdamW state."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    _validate_layer_norm_parent(checkpoint, tokenizer)

    parent_config = checkpoint["config"]
    parent_vocab_size = int(checkpoint["vocab_size"])
    model = CausalTransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=int(parent_config["embedding_dim"]),
        num_layers=int(parent_config["num_layers"]),
        num_heads=int(parent_config["num_heads"]),
        head_dim=int(parent_config["head_dim"]),
        feed_forward_dim=int(parent_config["feed_forward_dim"]),
        max_context_length=int(parent_config["max_context_length"]),
        normalization_type="rms_norm",
        normalization_eps=float(parent_config.get("normalization_eps", 1e-5)),
        position_encoding_type=parent_config.get(
            "position_encoding_type",
            "learned_absolute",
        ),
        rope_theta=float(parent_config.get("rope_theta", 10_000.0)),
        feed_forward_type=parent_config.get("feed_forward_type", "relu"),
    ).to(device)
    conversion_metadata = copy_layer_norm_parent_state_to_rms_norm_model(
        model=model,
        parent_state_dict=checkpoint["model_state_dict"],
        parent_vocab_size=parent_vocab_size,
        checkpoint_path=checkpoint_path,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    return model, optimizer, checkpoint, conversion_metadata


def copy_layer_norm_parent_state_to_rms_norm_model(
    *,
    model: CausalTransformerLanguageModel,
    parent_state_dict: dict[str, torch.Tensor],
    parent_vocab_size: int,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Copy compatible parent state and explicitly omit LayerNorm bias tensors."""
    model_state_dict = model.state_dict()
    copied_state_keys = []
    extended_vocabulary_state_keys = []
    normalization_scale_mappings = []

    for name, model_value in model_state_dict.items():
        if name not in parent_state_dict:
            raise ValueError(f"parent checkpoint is missing state key: {name}")

        parent_value = parent_state_dict[name]
        if parent_value.shape == model_value.shape:
            model_state_dict[name] = parent_value
            copied_state_keys.append(name)
        elif _is_extendable_vocabulary_state(
            name=name,
            parent_value=parent_value,
            model_value=model_value,
            parent_vocab_size=parent_vocab_size,
        ):
            model_value[:parent_vocab_size] = parent_value
            extended_vocabulary_state_keys.append(name)
        else:
            raise ValueError(f"parent weight shape is incompatible: {name}")

        if name.endswith("_layer_norm.weight"):
            normalization_scale_mappings.append({"source": name, "target": name})

    dropped_layer_norm_bias_keys = sorted(
        name
        for name in parent_state_dict
        if name not in model_state_dict and _is_layer_norm_bias_key(name)
    )
    unexpected_parent_state_keys = sorted(
        name
        for name in parent_state_dict
        if name not in model_state_dict and name not in dropped_layer_norm_bias_keys
    )
    if unexpected_parent_state_keys:
        raise ValueError(
            "parent checkpoint has unexpected state keys: "
            + ", ".join(unexpected_parent_state_keys)
        )

    model.load_state_dict(model_state_dict)
    return {
        "conversion_type": "layer_norm_to_rms_norm",
        "source_checkpoint_path": str(checkpoint_path),
        "source_normalization_type": "layer_norm",
        "target_normalization_type": "rms_norm",
        "source_parameter_count": sum(
            value.numel()
            for name, value in parent_state_dict.items()
            if value.is_floating_point() and not name.endswith("causal_mask")
        ),
        "target_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
        "parent_vocab_size": parent_vocab_size,
        "target_vocab_size": model.output_projection.out_features,
        "copied_state_keys": copied_state_keys,
        "extended_vocabulary_state_keys": extended_vocabulary_state_keys,
        "normalization_scale_mappings": normalization_scale_mappings,
        "dropped_layer_norm_bias_keys": dropped_layer_norm_bias_keys,
        "optimizer_state_restored": False,
    }


def _validate_layer_norm_parent(
    checkpoint: Any,
    tokenizer: BytePairEncodingTokenizer,
) -> None:
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")

    required_fields = {
        "model_state_dict",
        "config",
        "vocab_size",
    }
    missing_fields = sorted(required_fields - checkpoint.keys())
    if missing_fields:
        raise ValueError(
            "parent checkpoint is missing required fields: "
            + ", ".join(missing_fields)
        )
    if not isinstance(checkpoint["config"], dict):
        raise ValueError("parent checkpoint config must contain a dictionary")
    if checkpoint["config"].get("normalization_type", "layer_norm") != "layer_norm":
        raise ValueError("RMSNorm conversion requires a LayerNorm parent checkpoint")
    if int(checkpoint["vocab_size"]) > tokenizer.vocab_size:
        raise ValueError(
            "parent checkpoint vocabulary size exceeds the fine-tuning tokenizer"
        )


def _is_extendable_vocabulary_state(
    *,
    name: str,
    parent_value: torch.Tensor,
    model_value: torch.Tensor,
    parent_vocab_size: int,
) -> bool:
    return (
        name in VOCABULARY_STATE_KEYS
        and parent_value.shape[0] == parent_vocab_size
        and model_value.shape[0] >= parent_vocab_size
        and parent_value.shape[1:] == model_value.shape[1:]
    )


def _is_layer_norm_bias_key(name: str) -> bool:
    return name.endswith("_layer_norm.bias")
