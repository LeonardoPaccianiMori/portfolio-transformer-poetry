"""Run controlled sonnet-only and pretrained-initialization comparisons."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer
from sonnet_corpus.dataset_text import load_pretraining_bpe_encoded_splits
from sonnet_model.transformer import CausalTransformerLanguageModel
from sonnet_training.finetuning_run import generate_finetuning_sample, load_parent_for_finetuning
from sonnet_training.pretraining_run import count_parameters
from sonnet_training.steps import estimate_next_token_loss, train_next_token_step
from sonnet_training.transformer_run import resolve_device, write_json, write_jsonl


InitializationMode = Literal["pretrained", "random"]
LearningRateSchedule = Literal["constant", "warmup_cosine"]
MODEL_ARCHITECTURE_KEYS = (
    "vocab_size",
    "embedding_dim",
    "num_layers",
    "num_heads",
    "head_dim",
    "feed_forward_dim",
    "max_context_length",
)


@dataclass(frozen=True)
class SonnetControlRunConfig:
    """Shared data/training settings for one initialization-control arm."""

    initialization: InitializationMode = "random"
    dataset: str = "expanded_with_petrarch"
    model_architecture_path: str = (
        "runs/finetuning_larger_20k_001/selected_checkpoint.json"
    )
    pretraining_tokenizer_path: str = "runs/pretraining_larger_200k_001/tokenizer.json"
    pretraining_checkpoint_path: str = "runs/pretraining_larger_200k_001/model.pt"
    batch_size: int = 2
    context_length: int = 512
    train_steps: int = 20_000
    eval_interval: int = 250
    eval_batches: int = 5
    checkpoint_interval: int = 1_000
    learning_rate: float = 3e-5
    learning_rate_schedule: LearningRateSchedule = "constant"
    warmup_steps: int = 0
    min_learning_rate: float = 0.0
    seed: int = 1337
    prompt: str = "Amor"
    max_new_tokens: int = 300
    device: str = "auto"


def train_sonnet_control_run(
    repo_root: Path,
    output_dir: Path,
    config: SonnetControlRunConfig,
) -> dict[str, Path | list[dict[str, float | int]]]:
    """Train one controlled sonnet run with random or pretrained weights."""
    _validate_config(config)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    model_architecture = load_model_architecture(
        repo_root / config.model_architecture_path
    )
    manifest_path = repo_root / "data" / "metadata" / "poems_manifest.csv"
    train_tokens, validation_tokens, _, tokenizer = load_pretraining_bpe_encoded_splits(
        manifest_path=manifest_path,
        repo_root=repo_root,
        dataset=config.dataset,
        tokenizer_path=repo_root / config.pretraining_tokenizer_path,
    )
    _validate_tokenizer_architecture(tokenizer, model_architecture)
    model, optimizer, parent_checkpoint = initialize_control_model(
        repo_root=repo_root,
        config=config,
        tokenizer=tokenizer,
        model_architecture=model_architecture,
        device=device,
    )
    _validate_context_length(config, model)

    history, best_validation_row = train_control_steps(
        model=model,
        optimizer=optimizer,
        train_token_ids=train_tokens,
        validation_token_ids=validation_tokens,
        config=config,
        device=device,
        output_dir=output_dir,
        tokenizer=tokenizer,
        model_architecture=model_architecture,
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
    run_metadata = build_run_metadata(
        config=config,
        device=device,
        tokenizer=tokenizer,
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
        model=model,
        model_architecture=model_architecture,
        parent_checkpoint=parent_checkpoint,
        best_validation_row=best_validation_row,
    )
    write_json(config_path, run_metadata)
    write_jsonl(log_path, history)
    tokenizer.save(tokenizer_output_path)
    sample_path.write_text(generated_text, encoding="utf-8")
    save_control_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        tokenizer=tokenizer,
        model_architecture=model_architecture,
        parent_checkpoint=parent_checkpoint,
        step=config.train_steps,
        best_validation_row=best_validation_row,
    )

    return {
        "config_path": config_path,
        "log_path": log_path,
        "tokenizer_path": tokenizer_output_path,
        "sample_path": sample_path,
        "checkpoint_path": checkpoint_path,
        "checkpoint_dir": output_dir / "checkpoints",
        "best_checkpoint_path": output_dir / "best_validation.pt",
        "history": history,
    }


def load_model_architecture(path: Path) -> dict[str, int]:
    """Load the architecture manifest shared by every control arm."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    architecture = payload.get("model_architecture", payload)
    missing = [name for name in MODEL_ARCHITECTURE_KEYS if name not in architecture]
    if missing:
        raise ValueError("model architecture is missing fields: " + ", ".join(missing))
    return {name: int(architecture[name]) for name in MODEL_ARCHITECTURE_KEYS}


def initialize_control_model(
    *,
    repo_root: Path,
    config: SonnetControlRunConfig,
    tokenizer: BytePairEncodingTokenizer,
    model_architecture: dict[str, int],
    device: torch.device,
) -> tuple[
    CausalTransformerLanguageModel,
    torch.optim.Optimizer,
    dict[str, Any] | None,
]:
    """Build one arm while always creating fresh AdamW optimizer state."""
    if config.initialization == "random":
        model = CausalTransformerLanguageModel(**model_architecture).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        return model, optimizer, None

    if config.initialization == "pretrained":
        model, optimizer, parent_checkpoint = load_parent_for_finetuning(
            checkpoint_path=repo_root / config.pretraining_checkpoint_path,
            tokenizer=tokenizer,
            learning_rate=config.learning_rate,
            restore_optimizer_state=False,
            device=device,
        )
        _validate_model_architecture(model, model_architecture)
        return model, optimizer, parent_checkpoint

    raise ValueError("initialization must be 'pretrained' or 'random'")


def train_control_steps(
    *,
    model: CausalTransformerLanguageModel,
    optimizer: torch.optim.Optimizer,
    train_token_ids: torch.Tensor,
    validation_token_ids: torch.Tensor,
    config: SonnetControlRunConfig,
    device: torch.device,
    output_dir: Path,
    tokenizer: BytePairEncodingTokenizer,
    model_architecture: dict[str, int],
    parent_checkpoint: dict[str, Any] | None,
) -> tuple[list[dict[str, float | int]], dict[str, float | int]]:
    """Train, evaluate, and overwrite one exact best-validation checkpoint."""
    history: list[dict[str, float | int]] = []
    best_validation_row: dict[str, float | int] | None = None
    for step in range(1, config.train_steps + 1):
        current_learning_rate = learning_rate_for_step(config, step)
        set_optimizer_learning_rate(optimizer, current_learning_rate)
        train_loss = train_next_token_step(
            model=model,
            optimizer=optimizer,
            token_ids=train_token_ids,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=device,
        )
        should_evaluate = (
            step == 1
            or step % config.eval_interval == 0
            or step == config.train_steps
        )
        if should_evaluate:
            validation_loss = estimate_next_token_loss(
                model=model,
                token_ids=validation_token_ids,
                batch_size=config.batch_size,
                context_length=config.context_length,
                eval_batches=config.eval_batches,
                device=device,
            )
            row = {
                "step": step,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": current_learning_rate,
            }
            history.append(row)
            if (
                best_validation_row is None
                or row["validation_loss"] < best_validation_row["validation_loss"]
            ):
                best_validation_row = row
                save_control_checkpoint(
                    checkpoint_path=output_dir / "best_validation.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    tokenizer=tokenizer,
                    model_architecture=model_architecture,
                    parent_checkpoint=parent_checkpoint,
                    step=step,
                    best_validation_row=row,
                )
        if config.checkpoint_interval and step % config.checkpoint_interval == 0:
            save_control_checkpoint(
                checkpoint_path=output_dir / "checkpoints" / f"step_{step}.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                tokenizer=tokenizer,
                model_architecture=model_architecture,
                parent_checkpoint=parent_checkpoint,
                step=step,
                best_validation_row=best_validation_row,
            )

    if best_validation_row is None:
        raise RuntimeError("control run completed without a validation evaluation")
    return history, best_validation_row


def save_control_checkpoint(
    *,
    checkpoint_path: Path,
    model: CausalTransformerLanguageModel,
    optimizer: torch.optim.Optimizer,
    config: SonnetControlRunConfig,
    tokenizer: BytePairEncodingTokenizer,
    model_architecture: dict[str, int],
    parent_checkpoint: dict[str, Any] | None,
    step: int,
    best_validation_row: dict[str, float | int] | None,
) -> None:
    """Save one control checkpoint with explicit initialization provenance."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "initialization": config.initialization,
            "optimizer_state_restored": False,
            "vocab_size": tokenizer.vocab_size,
            "parameter_count": count_parameters(model),
            "model_architecture": model_architecture,
            "parent_checkpoint_step": (
                int(parent_checkpoint["step"])
                if parent_checkpoint is not None
                else None
            ),
            "best_validation_row": best_validation_row,
        },
        checkpoint_path,
    )


def build_run_metadata(
    *,
    config: SonnetControlRunConfig,
    device: torch.device,
    tokenizer: BytePairEncodingTokenizer,
    train_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    model: CausalTransformerLanguageModel,
    model_architecture: dict[str, int],
    parent_checkpoint: dict[str, Any] | None,
    best_validation_row: dict[str, float | int],
) -> dict[str, Any]:
    """Create reproducibility metadata shared by reports and selection tooling."""
    return {
        **asdict(config),
        "resolved_device": str(device),
        "optimizer_state_restored": False,
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": int(train_tokens.numel()),
        "validation_tokens": int(validation_tokens.numel()),
        "parameter_count": count_parameters(model),
        "model_architecture": model_architecture,
        "parent_checkpoint_step": (
            int(parent_checkpoint["step"])
            if parent_checkpoint is not None
            else None
        ),
        "completed_steps": config.train_steps,
        "best_validation_step": int(best_validation_row["step"]),
        "best_validation_loss": float(best_validation_row["validation_loss"]),
    }


def learning_rate_for_step(config: SonnetControlRunConfig, step: int) -> float:
    """Return the configured learning rate for one one-indexed training step."""
    if step <= 0 or step > config.train_steps:
        raise ValueError("step must be between 1 and train_steps")
    if config.learning_rate_schedule == "constant":
        return config.learning_rate

    if config.learning_rate_schedule == "warmup_cosine":
        if config.warmup_steps and step <= config.warmup_steps:
            return config.learning_rate * step / config.warmup_steps

        decay_steps = config.train_steps - config.warmup_steps
        decay_progress = (step - config.warmup_steps) / decay_steps
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return config.min_learning_rate + cosine_factor * (
            config.learning_rate - config.min_learning_rate
        )

    raise ValueError("unsupported learning_rate_schedule")


def set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    """Apply the current schedule value to every AdamW parameter group."""
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def _validate_tokenizer_architecture(
    tokenizer: BytePairEncodingTokenizer,
    model_architecture: dict[str, int],
) -> None:
    if tokenizer.vocab_size != model_architecture["vocab_size"]:
        raise ValueError("tokenizer vocabulary size does not match model architecture")


def _validate_model_architecture(
    model: CausalTransformerLanguageModel,
    model_architecture: dict[str, int],
) -> None:
    if model.output_projection.out_features != model_architecture["vocab_size"]:
        raise ValueError("pretrained model vocabulary does not match model architecture")
    if model.max_context_length != model_architecture["max_context_length"]:
        raise ValueError("pretrained model context does not match model architecture")


def _validate_context_length(
    config: SonnetControlRunConfig,
    model: CausalTransformerLanguageModel,
) -> None:
    if config.context_length > model.max_context_length:
        raise ValueError(
            "context_length must be less than or equal to model max_context_length"
        )


def _validate_config(config: SonnetControlRunConfig) -> None:
    if config.initialization not in {"pretrained", "random"}:
        raise ValueError("initialization must be 'pretrained' or 'random'")
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
    if config.learning_rate_schedule not in {"constant", "warmup_cosine"}:
        raise ValueError("unsupported learning_rate_schedule")
    if config.warmup_steps < 0 or config.warmup_steps >= config.train_steps:
        raise ValueError("warmup_steps must be at least 0 and less than train_steps")
    if config.min_learning_rate < 0:
        raise ValueError("min_learning_rate must be greater than or equal to 0")
    if config.min_learning_rate > config.learning_rate:
        raise ValueError("min_learning_rate must not exceed learning_rate")
    if config.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be greater than or equal to 0")
