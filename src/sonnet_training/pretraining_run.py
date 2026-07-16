"""Run broader Italian from-scratch transformer pretraining."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer
from sonnet_model.normalization import NormalizationType
from sonnet_model.positional_encoding import PositionEncodingType
from sonnet_model.transformer import CausalTransformerLanguageModel, FeedForwardType
from sonnet_training.learning_rate import (
    LearningRateSchedule,
    learning_rate_for_step,
    set_optimizer_learning_rate,
)
from sonnet_training.progress import TrainingProgressReporter
from sonnet_training.steps import (
    estimate_next_token_loss,
    estimate_next_token_loss_on_sequential_windows,
    sequential_next_token_window_count,
    train_next_token_step,
)
from sonnet_training.transformer_run import resolve_device, write_json, write_jsonl


ValidationMode = Literal["random_batches", "sequential_windows"]


@dataclass(frozen=True)
class PretrainingRunConfig:
    """Configuration for a broader-corpus pretraining run."""

    train_tokens_path: str = (
        "data/local/pretraining/expanded_italian_1200_1800_v1/encoded/"
        "bpe_8000_train.pt"
    )
    validation_tokens_path: str = (
        "data/local/pretraining/expanded_italian_1200_1800_v1/encoded/"
        "bpe_8000_validation.pt"
    )
    tokenizer_path: str = (
        "data/local/pretraining/expanded_italian_1200_1800_v1/tokenizers/"
        "bpe_8000.json"
    )
    batch_size: int = 8
    context_length: int = 512
    train_steps: int = 100
    eval_interval: int = 25
    eval_batches: int = 5
    validation_mode: ValidationMode = "random_batches"
    learning_rate: float = 3e-4
    learning_rate_schedule: LearningRateSchedule = "constant"
    warmup_steps: int = 0
    min_learning_rate: float = 0.0
    seed: int = 1337
    prompt: str = "Nel "
    max_new_tokens: int = 300
    device: str = "auto"
    embedding_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    head_dim: int = 32
    feed_forward_dim: int = 1024
    max_context_length: int = 512
    normalization_type: NormalizationType = "layer_norm"
    normalization_eps: float = 1e-5
    position_encoding_type: PositionEncodingType = "learned_absolute"
    rope_theta: float = 10_000.0
    feed_forward_type: FeedForwardType = "relu"
    tie_token_embeddings: bool = False
    checkpoint_interval: int = 0
    progress_interval: int = 100
    resume_from_checkpoint: str = ""


def train_pretraining_run(
    repo_root: Path,
    output_dir: Path,
    config: PretrainingRunConfig,
) -> dict[str, Path | list[dict[str, float | int]]]:
    """Train the transformer briefly on broader pretraining token tensors."""

    _validate_config(config)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)

    train_tokens = load_token_tensor(repo_root / config.train_tokens_path)
    validation_tokens = load_token_tensor(repo_root / config.validation_tokens_path)
    tokenizer = BytePairEncodingTokenizer.load(repo_root / config.tokenizer_path)

    model = CausalTransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        feed_forward_dim=config.feed_forward_dim,
        max_context_length=config.max_context_length,
        normalization_type=config.normalization_type,
        normalization_eps=config.normalization_eps,
        position_encoding_type=config.position_encoding_type,
        rope_theta=config.rope_theta,
        feed_forward_type=config.feed_forward_type,
        tie_token_embeddings=config.tie_token_embeddings,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )
    start_step = 0
    if config.resume_from_checkpoint:
        start_step = load_pretraining_checkpoint(
            checkpoint_path=repo_root / config.resume_from_checkpoint,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        if start_step >= config.train_steps:
            raise ValueError(
                "resume checkpoint step must be less than train_steps"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    prior_history = _read_history(output_dir / "loss_history.jsonl")
    initial_best_validation_row = _best_validation_row(
        prior_history,
        through_step=start_step,
    )

    history, best_validation_row = train_pretraining_steps(
        model=model,
        optimizer=optimizer,
        train_token_ids=train_tokens,
        validation_token_ids=validation_tokens,
        config=config,
        device=device,
        output_dir=output_dir,
        tokenizer=tokenizer,
        start_step=start_step,
        initial_best_validation_row=initial_best_validation_row,
    )

    generated_text = _generate_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=config.prompt,
        max_new_tokens=config.max_new_tokens,
        device=device,
    )

    config_path = output_dir / "config.json"
    log_path = output_dir / "loss_history.jsonl"
    tokenizer_output_path = output_dir / "tokenizer.json"
    sample_path = output_dir / "sample.txt"
    checkpoint_path = output_dir / "model.pt"

    write_json(
        path=config_path,
        payload={
            **asdict(config),
            "resolved_device": str(device),
            "vocab_size": tokenizer.vocab_size,
            "train_tokens": int(train_tokens.numel()),
            "validation_tokens": int(validation_tokens.numel()),
            "validation_window_count": sequential_next_token_window_count(
                validation_tokens,
                config.context_length,
            ),
            "parameter_count": count_parameters(model),
            "start_step": start_step,
            "completed_steps": config.train_steps,
            "best_validation_step": int(best_validation_row["step"]),
            "best_validation_loss": float(best_validation_row["validation_loss"]),
        },
    )
    saved_history = merge_existing_history(
        log_path=log_path,
        new_history=history,
        start_step=start_step,
    )
    write_jsonl(log_path, saved_history)
    tokenizer.save(tokenizer_output_path)
    sample_path.write_text(generated_text, encoding="utf-8")
    save_pretraining_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        tokenizer=tokenizer,
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
        "history": saved_history,
    }


def train_pretraining_steps(
    *,
    model: CausalTransformerLanguageModel,
    optimizer: torch.optim.Optimizer,
    train_token_ids: torch.Tensor,
    validation_token_ids: torch.Tensor,
    config: PretrainingRunConfig,
    device: torch.device,
    output_dir: Path,
    tokenizer: BytePairEncodingTokenizer,
    start_step: int,
    initial_best_validation_row: dict[str, float | int] | None = None,
) -> tuple[list[dict[str, float | int]], dict[str, float | int]]:
    """Train from start_step to config.train_steps with resumable checkpoints."""

    history: list[dict[str, float | int]] = []
    best_validation_row = initial_best_validation_row
    if start_step >= config.train_steps:
        if best_validation_row is None:
            raise ValueError("resumed run has no recorded validation evaluation")
        return history, best_validation_row
    progress = TrainingProgressReporter(
        total_steps=config.train_steps,
        progress_interval=config.progress_interval,
        start_step=start_step,
    )
    progress.write_start(label="pretraining", device=str(device))

    for step in range(start_step + 1, config.train_steps + 1):
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
            step == start_step + 1
            or step % config.eval_interval == 0
            or step == config.train_steps
        )
        if should_evaluate:
            validation_loss = estimate_pretraining_validation_loss(
                model=model,
                token_ids=validation_token_ids,
                config=config,
                device=device,
            )
            row = {
                "step": step,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": current_learning_rate,
            }
            history.append(row)
            best_validation_updated = (
                best_validation_row is None
                or row["validation_loss"] < best_validation_row["validation_loss"]
            )
            if best_validation_updated:
                best_validation_row = row
                save_pretraining_checkpoint(
                    checkpoint_path=output_dir / "best_validation.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    tokenizer=tokenizer,
                    step=step,
                    best_validation_row=best_validation_row,
                )
        else:
            validation_loss = None
            best_validation_updated = False

        checkpoint_written = False
        if config.checkpoint_interval and step % config.checkpoint_interval == 0:
            save_pretraining_checkpoint(
                checkpoint_path=output_dir / "checkpoints" / f"step_{step}.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                tokenizer=tokenizer,
                step=step,
                best_validation_row=best_validation_row,
            )
            checkpoint_written = True

        if progress.should_report(
            step,
            force=should_evaluate or checkpoint_written,
        ):
            progress.write_progress(
                step=step,
                train_loss=train_loss,
                validation_loss=validation_loss,
                learning_rate=current_learning_rate,
                checkpoint_written=checkpoint_written,
                best_validation=best_validation_updated,
            )

    if best_validation_row is None:
        raise RuntimeError("pretraining run completed without a validation evaluation")
    return history, best_validation_row


def estimate_pretraining_validation_loss(
    *,
    model: CausalTransformerLanguageModel,
    token_ids: torch.Tensor,
    config: PretrainingRunConfig,
    device: torch.device,
) -> float:
    """Evaluate pretraining loss using the configured reproducible policy."""

    if config.validation_mode == "random_batches":
        return estimate_next_token_loss(
            model=model,
            token_ids=token_ids,
            batch_size=config.batch_size,
            context_length=config.context_length,
            eval_batches=config.eval_batches,
            device=device,
        )
    if config.validation_mode == "sequential_windows":
        return estimate_next_token_loss_on_sequential_windows(
            model=model,
            token_ids=token_ids,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=device,
        )
    raise ValueError("unsupported validation_mode")


def save_pretraining_checkpoint(
    *,
    checkpoint_path: Path,
    model: CausalTransformerLanguageModel,
    optimizer: torch.optim.Optimizer,
    config: PretrainingRunConfig,
    tokenizer: BytePairEncodingTokenizer,
    step: int,
    best_validation_row: dict[str, float | int] | None = None,
) -> None:
    """Save model and optimizer state for later resume/fine-tuning."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "vocab_size": tokenizer.vocab_size,
            "parameter_count": count_parameters(model),
            "best_validation_row": best_validation_row,
        },
        checkpoint_path,
    )


def load_pretraining_checkpoint(
    *,
    checkpoint_path: Path,
    model: CausalTransformerLanguageModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    """Load model/optimizer state and return the completed checkpoint step."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint must contain a dictionary: {checkpoint_path}")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["step"])


def merge_existing_history(
    *,
    log_path: Path,
    new_history: list[dict[str, float | int]],
    start_step: int,
) -> list[dict[str, float | int]]:
    """Keep previous rows through start_step, then append new rows."""

    if not log_path.is_file():
        return new_history

    previous_history = _read_history(log_path)
    preserved_history = [
        row
        for row in previous_history
        if int(row["step"]) <= start_step
    ]
    return [*preserved_history, *new_history]


def _read_history(log_path: Path) -> list[dict[str, float | int]]:
    if not log_path.is_file():
        return []
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _best_validation_row(
    history: list[dict[str, float | int]],
    *,
    through_step: int,
) -> dict[str, float | int] | None:
    eligible_rows = [row for row in history if int(row["step"]) <= through_step]
    if not eligible_rows:
        return None
    return min(eligible_rows, key=lambda row: float(row["validation_loss"]))


def load_token_tensor(path: Path) -> torch.Tensor:
    """Load one 1D local token tensor for language-model training."""

    if not path.is_file():
        raise FileNotFoundError(f"token tensor file does not exist: {path}")

    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"token tensor file did not contain a torch.Tensor: {path}")
    if tensor.ndim != 1:
        raise ValueError(f"token tensor must be 1D: {path}")
    if tensor.dtype != torch.long:
        raise ValueError(f"token tensor must use dtype torch.long: {path}")
    return tensor


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable and non-trainable model parameters."""

    return sum(parameter.numel() for parameter in model.parameters())


def _validate_config(config: PretrainingRunConfig) -> None:
    if config.context_length > config.max_context_length:
        raise ValueError(
            "context_length must be less than or equal to max_context_length"
        )
    if config.context_length <= 0:
        raise ValueError("context_length must be greater than 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if config.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be greater than or equal to 0")
    if config.train_steps <= 0:
        raise ValueError("train_steps must be greater than 0")
    if config.eval_interval <= 0:
        raise ValueError("eval_interval must be greater than 0")
    if config.eval_batches <= 0:
        raise ValueError("eval_batches must be greater than 0")
    if config.validation_mode not in {"random_batches", "sequential_windows"}:
        raise ValueError("unsupported validation_mode")
    if config.learning_rate_schedule not in {"constant", "warmup_cosine"}:
        raise ValueError("unsupported learning_rate_schedule")
    if config.warmup_steps < 0 or config.warmup_steps > config.train_steps:
        raise ValueError("warmup_steps must be between 0 and train_steps")
    if config.min_learning_rate < 0:
        raise ValueError("min_learning_rate must be greater than or equal to 0")
    if config.min_learning_rate > config.learning_rate:
        raise ValueError("min_learning_rate must not exceed learning_rate")
    if config.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be greater than or equal to 0")
    if config.progress_interval <= 0:
        raise ValueError("progress_interval must be greater than 0")


def _generate_sample(
    *,
    model: CausalTransformerLanguageModel,
    tokenizer: BytePairEncodingTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
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
