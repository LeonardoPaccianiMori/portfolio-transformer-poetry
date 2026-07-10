"""Fine-tune a broader-pretrained causal transformer on the sonnet corpus."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer
from sonnet_corpus.dataset_text import load_pretraining_bpe_encoded_splits
from sonnet_model.transformer import CausalTransformerLanguageModel
from sonnet_training.pretraining_run import count_parameters
from sonnet_training.steps import estimate_next_token_loss, train_next_token_step
from sonnet_training.transformer_run import resolve_device, write_json, write_jsonl


@dataclass(frozen=True)
class FineTuningRunConfig:
    """Settings for one sonnet fine-tuning continuation of a parent run."""

    dataset: str = "expanded_with_petrarch"
    pretraining_checkpoint_path: str = "runs/pretraining_larger_200k_001/model.pt"
    pretraining_tokenizer_path: str = "runs/pretraining_larger_200k_001/tokenizer.json"
    batch_size: int = 2
    context_length: int = 512
    train_steps: int = 20_000
    eval_interval: int = 250
    eval_batches: int = 5
    checkpoint_interval: int = 1_000
    learning_rate: float = 3e-5
    restore_optimizer_state: bool = True
    seed: int = 1337
    prompt: str = "Amor"
    max_new_tokens: int = 300
    device: str = "auto"


def train_finetuning_run(
    repo_root: Path,
    output_dir: Path,
    config: FineTuningRunConfig,
) -> dict[str, Path | list[dict[str, float | int]]]:
    """Continue a broader-pretrained model on train-split sonnet token IDs."""
    _validate_config(config)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)

    manifest_path = repo_root / "data" / "metadata" / "poems_manifest.csv"
    tokenizer_path = repo_root / config.pretraining_tokenizer_path
    train_tokens, validation_tokens, _, tokenizer = load_pretraining_bpe_encoded_splits(
        manifest_path=manifest_path,
        repo_root=repo_root,
        dataset=config.dataset,
        tokenizer_path=tokenizer_path,
    )
    model, optimizer, parent_checkpoint = load_parent_for_finetuning(
        checkpoint_path=repo_root / config.pretraining_checkpoint_path,
        tokenizer=tokenizer,
        learning_rate=config.learning_rate,
        restore_optimizer_state=config.restore_optimizer_state,
        device=device,
    )
    _validate_context_length(config, model)

    history = train_finetuning_steps(
        model=model,
        optimizer=optimizer,
        train_token_ids=train_tokens,
        validation_token_ids=validation_tokens,
        config=config,
        device=device,
        output_dir=output_dir,
        tokenizer=tokenizer,
        parent_checkpoint=parent_checkpoint,
    )
    generated_text = generate_finetuning_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=config.prompt,
        max_new_tokens=config.max_new_tokens,
        device=device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    log_path = output_dir / "loss_history.jsonl"
    tokenizer_output_path = output_dir / "tokenizer.json"
    sample_path = output_dir / "sample.txt"
    checkpoint_path = output_dir / "model.pt"
    parent_step = int(parent_checkpoint["step"])
    parent_parameter_count = int(parent_checkpoint["parameter_count"])

    write_json(
        config_path,
        {
            **asdict(config),
            "resolved_device": str(device),
            "vocab_size": tokenizer.vocab_size,
            "parent_vocab_size": int(parent_checkpoint["vocab_size"]),
            "added_token_strings": added_token_strings(
                tokenizer=tokenizer,
                parent_vocab_size=int(parent_checkpoint["vocab_size"]),
            ),
            "train_tokens": int(train_tokens.numel()),
            "validation_tokens": int(validation_tokens.numel()),
            "parameter_count": count_parameters(model),
            "parent_checkpoint_step": parent_step,
            "parent_parameter_count": parent_parameter_count,
            "parent_vocab_size": int(parent_checkpoint["vocab_size"]),
            "completed_steps": config.train_steps,
        },
    )
    write_jsonl(log_path, history)
    tokenizer.save(tokenizer_output_path)
    sample_path.write_text(generated_text, encoding="utf-8")
    save_finetuning_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        tokenizer=tokenizer,
        parent_checkpoint=parent_checkpoint,
        step=config.train_steps,
    )

    return {
        "config_path": config_path,
        "log_path": log_path,
        "tokenizer_path": tokenizer_output_path,
        "sample_path": sample_path,
        "checkpoint_path": checkpoint_path,
        "checkpoint_dir": output_dir / "checkpoints",
        "history": history,
    }


def load_parent_for_finetuning(
    *,
    checkpoint_path: Path,
    tokenizer: BytePairEncodingTokenizer,
    learning_rate: float,
    restore_optimizer_state: bool,
    device: torch.device,
) -> tuple[CausalTransformerLanguageModel, torch.optim.Optimizer, dict[str, Any]]:
    """Restore the parent architecture, weights, and optionally AdamW moments."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint must contain a dictionary: {checkpoint_path}")
    _validate_parent_checkpoint(checkpoint, tokenizer)

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
    ).to(device)
    _load_parent_weights_with_extended_vocabulary(
        model=model,
        parent_state_dict=checkpoint["model_state_dict"],
        parent_vocab_size=parent_vocab_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if restore_optimizer_state:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _extend_optimizer_state_for_vocabulary(
            optimizer=optimizer,
            parent_vocab_size=parent_vocab_size,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

    return model, optimizer, checkpoint


def train_finetuning_steps(
    *,
    model: CausalTransformerLanguageModel,
    optimizer: torch.optim.Optimizer,
    train_token_ids: torch.Tensor,
    validation_token_ids: torch.Tensor,
    config: FineTuningRunConfig,
    device: torch.device,
    output_dir: Path,
    tokenizer: BytePairEncodingTokenizer,
    parent_checkpoint: dict[str, Any],
) -> list[dict[str, float | int]]:
    """Run fine-tuning updates and write interval checkpoints when requested."""
    history: list[dict[str, float | int]] = []
    for step in range(1, config.train_steps + 1):
        train_loss = train_next_token_step(
            model=model,
            optimizer=optimizer,
            token_ids=train_token_ids,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=device,
        )
        if step == 1 or step % config.eval_interval == 0 or step == config.train_steps:
            validation_loss = estimate_next_token_loss(
                model=model,
                token_ids=validation_token_ids,
                batch_size=config.batch_size,
                context_length=config.context_length,
                eval_batches=config.eval_batches,
                device=device,
            )
            history.append({
                "step": step,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            })
        if config.checkpoint_interval and step % config.checkpoint_interval == 0:
            save_finetuning_checkpoint(
                checkpoint_path=output_dir / "checkpoints" / f"step_{step}.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                tokenizer=tokenizer,
                parent_checkpoint=parent_checkpoint,
                step=step,
            )
    return history


def save_finetuning_checkpoint(
    *,
    checkpoint_path: Path,
    model: CausalTransformerLanguageModel,
    optimizer: torch.optim.Optimizer,
    config: FineTuningRunConfig,
    tokenizer: BytePairEncodingTokenizer,
    parent_checkpoint: dict[str, Any],
    step: int,
) -> None:
    """Save a fine-tuned model with explicit lineage back to pretraining."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "vocab_size": tokenizer.vocab_size,
            "parameter_count": count_parameters(model),
            "parent_checkpoint_step": int(parent_checkpoint["step"]),
            "parent_parameter_count": int(parent_checkpoint["parameter_count"]),
            "parent_vocab_size": int(parent_checkpoint["vocab_size"]),
        },
        checkpoint_path,
    )


def generate_finetuning_sample(
    *,
    model: CausalTransformerLanguageModel,
    tokenizer: BytePairEncodingTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    """Generate one qualitative sample from the current fine-tuned weights."""
    prompt_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device,
    )
    generated_ids = model.generate(
        input_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
    )
    return tokenizer.decode(generated_ids[0].cpu().tolist())


def _validate_parent_checkpoint(
    checkpoint: dict[str, Any],
    tokenizer: BytePairEncodingTokenizer,
) -> None:
    required_fields = {
        "step",
        "model_state_dict",
        "optimizer_state_dict",
        "config",
        "vocab_size",
        "parameter_count",
    }
    missing_fields = sorted(required_fields - checkpoint.keys())
    if missing_fields:
        raise ValueError(
            "parent checkpoint is missing required fields: "
            + ", ".join(missing_fields)
        )
    if int(checkpoint["vocab_size"]) > tokenizer.vocab_size:
        raise ValueError(
            "parent checkpoint vocabulary size exceeds the fine-tuning tokenizer"
        )


def _load_parent_weights_with_extended_vocabulary(
    *,
    model: CausalTransformerLanguageModel,
    parent_state_dict: dict[str, torch.Tensor],
    parent_vocab_size: int,
) -> None:
    """Copy parent weights while retaining initialized rows for appended tokens."""
    model_state_dict = model.state_dict()
    vocabulary_parameter_names = {
        "embedding.token_embedding.weight",
        "output_projection.weight",
        "output_projection.bias",
    }
    for name, parent_value in parent_state_dict.items():
        model_value = model_state_dict[name]
        if parent_value.shape == model_value.shape:
            model_state_dict[name] = parent_value
            continue
        if (
            name in vocabulary_parameter_names
            and parent_value.shape[0] == parent_vocab_size
            and model_value.shape[0] >= parent_vocab_size
            and parent_value.shape[1:] == model_value.shape[1:]
        ):
            model_value[:parent_vocab_size] = parent_value
            continue
        raise ValueError(f"parent weight shape is incompatible: {name}")
    model.load_state_dict(model_state_dict)


def _extend_optimizer_state_for_vocabulary(
    *,
    optimizer: torch.optim.Optimizer,
    parent_vocab_size: int,
) -> None:
    """Pad AdamW moment tensors for appended vocabulary rows with zero moments."""
    for parameter_group in optimizer.param_groups:
        for parameter in parameter_group["params"]:
            state = optimizer.state[parameter]
            for state_name, value in list(state.items()):
                if (
                    not isinstance(value, torch.Tensor)
                    or value.ndim == 0
                    or value.shape == parameter.shape
                ):
                    continue
                if (
                    value.ndim == parameter.ndim
                    and value.shape[0] == parent_vocab_size
                    and parameter.shape[0] >= parent_vocab_size
                    and value.shape[1:] == parameter.shape[1:]
                ):
                    extended_value = torch.zeros_like(parameter)
                    extended_value[:parent_vocab_size] = value
                    state[state_name] = extended_value
                    continue
                raise ValueError(
                    f"optimizer state shape is incompatible: {state_name}"
                )


def added_token_strings(
    *,
    tokenizer: BytePairEncodingTokenizer,
    parent_vocab_size: int,
) -> list[str]:
    """Return appended token strings in token-ID order for run provenance."""
    return [
        tokenizer.id_to_token[token_id]
        for token_id in range(parent_vocab_size, tokenizer.vocab_size)
    ]


def _validate_context_length(
    config: FineTuningRunConfig,
    model: CausalTransformerLanguageModel,
) -> None:
    if config.context_length > model.max_context_length:
        raise ValueError(
            "context_length must be less than or equal to the parent model context"
        )


def _validate_config(config: FineTuningRunConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if config.context_length <= 0:
        raise ValueError("context_length must be greater than 0")
    if config.train_steps <= 0:
        raise ValueError("train_steps must be greater than 0")
    if config.eval_interval <= 0:
        raise ValueError("eval_interval must be greater than 0")
    if config.eval_batches <= 0:
        raise ValueError("eval_batches must be greater than 0")
    if config.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be greater than or equal to 0")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if config.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be greater than or equal to 0")
