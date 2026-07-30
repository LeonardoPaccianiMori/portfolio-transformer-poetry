"""Run resumable opening-line sonnet continuation post-training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer
from sonnet_corpus.task_format import (
    TASK_FORMAT_SPECIAL_TOKENS,
    EncodedSonnetContinuationExample,
    load_encoded_sonnet_continuation_splits,
)
from sonnet_model.transformer import CausalTransformerLanguageModel
from sonnet_training.finetuning_run import load_parent_for_finetuning
from sonnet_training.learning_rate import (
    LearningRateSchedule,
    learning_rate_for_step,
    set_optimizer_learning_rate,
)
from sonnet_training.pretraining_run import count_parameters, merge_existing_history
from sonnet_training.progress import TrainingProgressReporter
from sonnet_training.task_format_steps import (
    estimate_sonnet_continuation_loss,
    train_sonnet_continuation_step,
)
from sonnet_training.transformer_run import resolve_device, write_json, write_jsonl


TASK_FORMAT_VERSION = "opening_line_continuation_v1"
TASK_FORMAT_END_TOKEN = "<|endoftext|>"
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
class TaskFormatRunConfig:
    """Configuration for one masked opening-line continuation training run."""

    dataset: str = "expanded_with_petrarch"
    manifest_path: str = "data/metadata/sonnets_expanded_v5_manifest.csv"
    parent_checkpoint_path: str = (
        "runs/sonnet_control_historical_v2_xxl_v5_stable_eval_20k_001/"
        "best_validation.pt"
    )
    parent_tokenizer_path: str = (
        "runs/sonnet_control_historical_v2_xxl_v5_stable_eval_20k_001/"
        "tokenizer.json"
    )
    batch_size: int = 1
    gradient_accumulation_steps: int = 2
    context_length: int = 512
    train_steps: int = 12_000
    eval_interval: int = 250
    early_stopping_patience: int = 8
    min_validation_improvement: float = 0.01
    checkpoint_interval: int = 500
    progress_interval: int = 100
    learning_rate: float = 1e-5
    adamw_foreach: bool = False
    learning_rate_schedule: LearningRateSchedule = "constant"
    warmup_steps: int = 0
    min_learning_rate: float = 0.0
    max_gradient_norm: float | None = 1.0
    seed: int = 1337
    device: str = "auto"
    resume_from_checkpoint: str = ""


def train_task_format_run(
    repo_root: Path,
    output_dir: Path,
    config: TaskFormatRunConfig,
) -> dict[str, Path | list[dict[str, float | int | None]] | int | str]:
    """Train a selected sonnet model on masked opening-line continuations."""
    _validate_config(config)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    manifest_path = _resolve_repo_path(repo_root, config.manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"sonnet manifest does not exist: {manifest_path}")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    train_examples, validation_examples, test_examples, tokenizer = (
        load_encoded_sonnet_continuation_splits(
            manifest_path=manifest_path,
            repo_root=repo_root,
            dataset=config.dataset,
            tokenizer_path=_resolve_repo_path(repo_root, config.parent_tokenizer_path),
        )
    )
    pad_token_id = _single_token_id(tokenizer, TASK_FORMAT_END_TOKEN)
    model, optimizer, parent_checkpoint = load_parent_for_finetuning(
        checkpoint_path=_resolve_repo_path(repo_root, config.parent_checkpoint_path),
        tokenizer=tokenizer,
        learning_rate=config.learning_rate,
        restore_optimizer_state=False,
        device=device,
        adamw_foreach=config.adamw_foreach,
    )
    source_model_architecture = _source_model_architecture(parent_checkpoint)
    model_architecture = {
        **source_model_architecture,
        "vocab_size": tokenizer.vocab_size,
    }
    _validate_context_length(config, model)
    parent_provenance = _parent_provenance(
        config=config,
        parent_checkpoint=parent_checkpoint,
    )
    del parent_checkpoint

    output_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    best_validation_row: dict[str, float | int | None] | None = None
    non_improving_evaluations = 0
    if config.resume_from_checkpoint:
        (
            start_step,
            best_validation_row,
            non_improving_evaluations,
        ) = load_task_format_resume_checkpoint(
            checkpoint_path=_resolve_repo_path(repo_root, config.resume_from_checkpoint),
            model=model,
            optimizer=optimizer,
            manifest_sha256=manifest_sha256,
            expected_vocab_size=tokenizer.vocab_size,
            expected_model_architecture=model_architecture,
        )
        if start_step >= config.train_steps:
            raise ValueError("resume checkpoint step must be less than train_steps")

    (
        history,
        best_validation_row,
        completed_steps,
        stop_reason,
        non_improving_evaluations,
    ) = (
        train_task_format_steps(
            model=model,
            optimizer=optimizer,
            train_examples=train_examples,
            validation_examples=validation_examples,
            config=config,
            device=device,
            output_dir=output_dir,
            tokenizer=tokenizer,
            pad_token_id=pad_token_id,
            manifest_sha256=manifest_sha256,
            model_architecture=model_architecture,
            parent_provenance=parent_provenance,
            start_step=start_step,
            initial_best_validation_row=best_validation_row,
            initial_non_improving_evaluations=non_improving_evaluations,
        )
    )

    config_path = output_dir / "config.json"
    log_path = output_dir / "loss_history.jsonl"
    tokenizer_path = output_dir / "tokenizer.json"
    checkpoint_path = output_dir / "model.pt"
    saved_history = merge_existing_history(
        log_path=log_path,
        new_history=history,
        start_step=start_step,
    )
    write_json(
        config_path,
        _build_run_metadata(
            config=config,
            device=device,
            tokenizer=tokenizer,
            train_examples=train_examples,
            validation_examples=validation_examples,
            test_examples=test_examples,
            pad_token_id=pad_token_id,
            model=model,
            model_architecture=model_architecture,
            source_model_architecture=source_model_architecture,
            parent_provenance=parent_provenance,
            manifest_sha256=manifest_sha256,
            start_step=start_step,
            completed_steps=completed_steps,
            best_validation_row=best_validation_row,
            stop_reason=stop_reason,
        ),
    )
    write_jsonl(log_path, saved_history)
    tokenizer.save(tokenizer_path)
    save_task_format_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        tokenizer=tokenizer,
        manifest_sha256=manifest_sha256,
        model_architecture=model_architecture,
        parent_provenance=parent_provenance,
        step=completed_steps,
        best_validation_row=best_validation_row,
        non_improving_evaluations=non_improving_evaluations,
        stop_reason=stop_reason,
        include_optimizer_state=True,
    )

    return {
        "config_path": config_path,
        "log_path": log_path,
        "tokenizer_path": tokenizer_path,
        "checkpoint_path": checkpoint_path,
        "resume_checkpoint_path": output_dir / "resume.pt",
        "best_checkpoint_path": output_dir / "best_validation.pt",
        "history": saved_history,
        "completed_steps": completed_steps,
        "stop_reason": stop_reason,
    }


def train_task_format_steps(
    *,
    model: CausalTransformerLanguageModel,
    optimizer: torch.optim.Optimizer,
    train_examples: list[EncodedSonnetContinuationExample],
    validation_examples: list[EncodedSonnetContinuationExample],
    config: TaskFormatRunConfig,
    device: torch.device,
    output_dir: Path,
    tokenizer: BytePairEncodingTokenizer,
    pad_token_id: int,
    manifest_sha256: str,
    model_architecture: dict[str, int | float | str | bool],
    parent_provenance: dict[str, int | str | None],
    start_step: int,
    initial_best_validation_row: dict[str, float | int | None] | None,
    initial_non_improving_evaluations: int,
) -> tuple[
    list[dict[str, float | int | None]],
    dict[str, float | int | None],
    int,
    str,
    int,
]:
    """Train, select, and checkpoint task-format weights from one start step."""
    history: list[dict[str, float | int | None]] = []
    best_validation_row = initial_best_validation_row
    non_improving_evaluations = initial_non_improving_evaluations
    completed_steps = start_step
    stop_reason = "max_train_steps_reached"
    progress = TrainingProgressReporter(
        total_steps=config.train_steps,
        progress_interval=config.progress_interval,
        start_step=start_step,
    )
    progress.write_start(
        label="task-format post-training",
        device=str(device),
        tokens_per_step=estimated_supervised_targets_per_update(
            config,
            train_examples,
        ),
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    for step in range(start_step + 1, config.train_steps + 1):
        current_learning_rate = learning_rate_for_step(config, step)
        set_optimizer_learning_rate(optimizer, current_learning_rate)
        train_loss, pre_clipping_gradient_norm = train_sonnet_continuation_step(
            model=model,
            optimizer=optimizer,
            examples=train_examples,
            batch_size=config.batch_size,
            pad_token_id=pad_token_id,
            max_context_length=config.context_length,
            device=device,
            max_gradient_norm=config.max_gradient_norm,
            return_gradient_norm=True,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        should_evaluate = (
            step == start_step + 1
            or step % config.eval_interval == 0
            or step == config.train_steps
        )
        if should_evaluate:
            validation_loss = estimate_sonnet_continuation_loss(
                model=model,
                examples=validation_examples,
                batch_size=config.batch_size,
                pad_token_id=pad_token_id,
                max_context_length=config.context_length,
                device=device,
            )
            row: dict[str, float | int | None] = {
                "step": step,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": current_learning_rate,
                "pre_clipping_gradient_norm": pre_clipping_gradient_norm,
                "non_improving_evaluations": non_improving_evaluations,
            }
            history.append(row)
            best_validation_updated = (
                best_validation_row is None
                or validation_loss
                < float(best_validation_row["validation_loss"])
                - config.min_validation_improvement
            )
            if best_validation_updated:
                best_validation_row = row
                non_improving_evaluations = 0
                row["non_improving_evaluations"] = non_improving_evaluations
                save_task_format_checkpoint(
                    checkpoint_path=output_dir / "best_validation.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    tokenizer=tokenizer,
                    manifest_sha256=manifest_sha256,
                    model_architecture=model_architecture,
                    parent_provenance=parent_provenance,
                    step=step,
                    best_validation_row=best_validation_row,
                    non_improving_evaluations=non_improving_evaluations,
                    stop_reason=None,
                    include_optimizer_state=False,
                )
            else:
                non_improving_evaluations += 1
                row["non_improving_evaluations"] = non_improving_evaluations
            should_stop_early = (
                config.early_stopping_patience > 0
                and non_improving_evaluations >= config.early_stopping_patience
            )
        else:
            validation_loss = None
            best_validation_updated = False
            should_stop_early = False

        checkpoint_written = False
        if config.checkpoint_interval and step % config.checkpoint_interval == 0:
            save_task_format_checkpoint(
                checkpoint_path=output_dir / "resume.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                tokenizer=tokenizer,
                manifest_sha256=manifest_sha256,
                model_architecture=model_architecture,
                parent_provenance=parent_provenance,
                step=step,
                best_validation_row=best_validation_row,
                non_improving_evaluations=non_improving_evaluations,
                stop_reason=None,
                include_optimizer_state=True,
            )
            checkpoint_written = True

        if progress.should_report(step, force=should_evaluate or checkpoint_written):
            progress.write_progress(
                step=step,
                train_loss=train_loss,
                validation_loss=validation_loss,
                learning_rate=current_learning_rate,
                checkpoint_written=checkpoint_written,
                best_validation=best_validation_updated,
            )

        completed_steps = step
        if should_stop_early:
            stop_reason = "early_stopping_patience_exhausted"
            break

    if best_validation_row is None:
        raise RuntimeError("task-format run completed without validation")
    save_task_format_checkpoint(
        checkpoint_path=output_dir / "resume.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        tokenizer=tokenizer,
        manifest_sha256=manifest_sha256,
        model_architecture=model_architecture,
        parent_provenance=parent_provenance,
        step=completed_steps,
        best_validation_row=best_validation_row,
        non_improving_evaluations=non_improving_evaluations,
        stop_reason=stop_reason,
        include_optimizer_state=True,
    )
    return (
        history,
        best_validation_row,
        completed_steps,
        stop_reason,
        non_improving_evaluations,
    )


def save_task_format_checkpoint(
    *,
    checkpoint_path: Path,
    model: CausalTransformerLanguageModel,
    optimizer: torch.optim.Optimizer,
    config: TaskFormatRunConfig,
    tokenizer: BytePairEncodingTokenizer,
    manifest_sha256: str,
    model_architecture: dict[str, int | float | str | bool],
    parent_provenance: dict[str, int | str | None],
    step: int,
    best_validation_row: dict[str, float | int | None] | None,
    non_improving_evaluations: int,
    stop_reason: str | None,
    include_optimizer_state: bool,
) -> None:
    """Atomically save task-format state, with optimizer state when resumable."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(
        {
            "task_format_version": TASK_FORMAT_VERSION,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict() if include_optimizer_state else None
            ),
            "config": asdict(config),
            "manifest_sha256": manifest_sha256,
            "vocab_size": tokenizer.vocab_size,
            "parameter_count": count_parameters(model),
            "model_architecture": model_architecture,
            "parent_provenance": parent_provenance,
            "best_validation_row": best_validation_row,
            "non_improving_evaluations": non_improving_evaluations,
            "stop_reason": stop_reason,
        },
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)


def load_task_format_resume_checkpoint(
    *,
    checkpoint_path: Path,
    model: CausalTransformerLanguageModel,
    optimizer: torch.optim.Optimizer,
    manifest_sha256: str,
    expected_vocab_size: int,
    expected_model_architecture: dict[str, int | float | str | bool],
) -> tuple[int, dict[str, float | int | None] | None, int]:
    """Restore a task run without mapping its large checkpoint to CUDA first."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint must contain a dictionary: {checkpoint_path}")
    if checkpoint.get("task_format_version") != TASK_FORMAT_VERSION:
        raise ValueError("checkpoint is not a compatible task-format run")
    if checkpoint.get("optimizer_state_dict") is None:
        raise ValueError("checkpoint is not resumable")
    if checkpoint.get("manifest_sha256") != manifest_sha256:
        raise ValueError("resume checkpoint manifest does not match current manifest")
    if int(checkpoint.get("vocab_size", -1)) != expected_vocab_size:
        raise ValueError("resume checkpoint vocabulary size does not match tokenizer")
    if checkpoint.get("model_architecture") != expected_model_architecture:
        raise ValueError("resume checkpoint model architecture does not match parent")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_validation_row = _checkpoint_best_validation_row(checkpoint)
    non_improving_evaluations = int(checkpoint.get("non_improving_evaluations", 0))
    if non_improving_evaluations < 0:
        raise ValueError("resume checkpoint non_improving_evaluations is invalid")
    return int(checkpoint["step"]), best_validation_row, non_improving_evaluations


def estimated_supervised_targets_per_update(
    config: TaskFormatRunConfig,
    train_examples: list[EncodedSonnetContinuationExample],
) -> int:
    """Return the rounded expected target count shown in progress metadata."""
    train_targets = supervised_target_count(train_examples)
    return round(
        train_targets
        / len(train_examples)
        * config.batch_size
        * config.gradient_accumulation_steps
    )


def supervised_target_count(
    examples: list[EncodedSonnetContinuationExample],
) -> int:
    """Count target positions whose labels are not ignored by cross-entropy."""
    return sum(
        int((example.target_ids != -100).sum().item())
        for example in examples
    )


def _build_run_metadata(
    *,
    config: TaskFormatRunConfig,
    device: torch.device,
    tokenizer: BytePairEncodingTokenizer,
    train_examples: list[EncodedSonnetContinuationExample],
    validation_examples: list[EncodedSonnetContinuationExample],
    test_examples: list[EncodedSonnetContinuationExample],
    pad_token_id: int,
    model: CausalTransformerLanguageModel,
    model_architecture: dict[str, int | float | str | bool],
    source_model_architecture: dict[str, int | float | str | bool],
    parent_provenance: dict[str, int | str | None],
    manifest_sha256: str,
    start_step: int,
    completed_steps: int,
    best_validation_row: dict[str, float | int | None],
    stop_reason: str,
) -> dict[str, Any]:
    train_targets = supervised_target_count(train_examples)
    return {
        **asdict(config),
        "task_format_version": TASK_FORMAT_VERSION,
        "task_format_special_tokens": list(TASK_FORMAT_SPECIAL_TOKENS),
        "pad_token": TASK_FORMAT_END_TOKEN,
        "pad_token_id": pad_token_id,
        "manifest_sha256": manifest_sha256,
        "resolved_device": str(device),
        "vocab_size": tokenizer.vocab_size,
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "test_examples": len(test_examples),
        "train_supervised_targets": train_targets,
        "validation_supervised_targets": supervised_target_count(validation_examples),
        "test_supervised_targets": supervised_target_count(test_examples),
        "estimated_supervised_targets_per_update": (
            estimated_supervised_targets_per_update(config, train_examples)
        ),
        "planned_supervised_target_exposures": (
            config.train_steps
            * estimated_supervised_targets_per_update(config, train_examples)
        ),
        "parameter_count": count_parameters(model),
        "model_architecture": model_architecture,
        "source_model_architecture": source_model_architecture,
        "parent_provenance": parent_provenance,
        "start_step": start_step,
        "completed_steps": completed_steps,
        "best_validation_step": int(best_validation_row["step"]),
        "best_validation_loss": float(best_validation_row["validation_loss"]),
        "stop_reason": stop_reason,
    }


def _single_token_id(tokenizer: BytePairEncodingTokenizer, text: str) -> int:
    token_ids = tokenizer.encode(text)
    if len(token_ids) != 1:
        raise ValueError(f"{text} must encode to exactly one token")
    return token_ids[0]


def _source_model_architecture(
    parent_checkpoint: dict[str, Any],
) -> dict[str, int | float | str | bool]:
    raw_architecture = parent_checkpoint.get(
        "model_architecture",
        parent_checkpoint.get("config"),
    )
    if not isinstance(raw_architecture, dict):
        raise ValueError("parent checkpoint model architecture must be a dictionary")
    missing_fields = [
        field
        for field in MODEL_ARCHITECTURE_KEYS
        if field not in raw_architecture
    ]
    if missing_fields:
        raise ValueError(
            "parent checkpoint model architecture is missing fields: "
            + ", ".join(missing_fields)
        )
    return {
        **{field: int(raw_architecture[field]) for field in MODEL_ARCHITECTURE_KEYS},
        "normalization_type": str(raw_architecture.get("normalization_type", "layer_norm")),
        "normalization_eps": float(raw_architecture.get("normalization_eps", 1e-5)),
        "position_encoding_type": str(
            raw_architecture.get("position_encoding_type", "learned_absolute")
        ),
        "rope_theta": float(raw_architecture.get("rope_theta", 10_000.0)),
        "feed_forward_type": str(raw_architecture.get("feed_forward_type", "relu")),
        "tie_token_embeddings": bool(raw_architecture.get("tie_token_embeddings", False)),
    }


def _parent_provenance(
    *,
    config: TaskFormatRunConfig,
    parent_checkpoint: dict[str, Any],
) -> dict[str, int | str | None]:
    return {
        "checkpoint_path": config.parent_checkpoint_path,
        "step": int(parent_checkpoint["step"]),
        "vocab_size": int(parent_checkpoint["vocab_size"]),
        "parameter_count": int(parent_checkpoint["parameter_count"]),
        "manifest_sha256": _optional_string(parent_checkpoint.get("manifest_sha256")),
        "parent_checkpoint_step": _optional_int(
            parent_checkpoint.get("parent_checkpoint_step")
        ),
    }


def _checkpoint_best_validation_row(
    checkpoint: dict[str, Any],
) -> dict[str, float | int | None] | None:
    row = checkpoint.get("best_validation_row")
    if row is None:
        return None
    if not isinstance(row, dict):
        raise ValueError("checkpoint best_validation_row must be a dictionary")
    required_fields = {"step", "train_loss", "validation_loss", "learning_rate"}
    missing_fields = sorted(required_fields - row.keys())
    if missing_fields:
        raise ValueError(
            "checkpoint best_validation_row is missing fields: "
            + ", ".join(missing_fields)
        )
    return {
        "step": int(row["step"]),
        "train_loss": float(row["train_loss"]),
        "validation_loss": float(row["validation_loss"]),
        "learning_rate": float(row["learning_rate"]),
        "pre_clipping_gradient_norm": row.get("pre_clipping_gradient_norm"),
        "non_improving_evaluations": int(row.get("non_improving_evaluations", 0)),
    }


def _resolve_repo_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path


def _validate_context_length(
    config: TaskFormatRunConfig,
    model: CausalTransformerLanguageModel,
) -> None:
    if config.context_length > model.max_context_length:
        raise ValueError(
            "context_length must be less than or equal to model max_context_length"
        )


def _validate_config(config: TaskFormatRunConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be greater than 0")
    if config.context_length <= 0:
        raise ValueError("context_length must be greater than 0")
    if config.train_steps <= 0:
        raise ValueError("train_steps must be greater than 0")
    if config.eval_interval <= 0:
        raise ValueError("eval_interval must be greater than 0")
    if config.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be greater than or equal to 0")
    if config.min_validation_improvement < 0:
        raise ValueError("min_validation_improvement must be greater than or equal to 0")
    if config.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be greater than 0")
    if config.progress_interval <= 0:
        raise ValueError("progress_interval must be greater than 0")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if not isinstance(config.adamw_foreach, bool):
        raise ValueError("adamw_foreach must be a boolean")
    if config.learning_rate_schedule not in {"constant", "warmup_cosine"}:
        raise ValueError("unsupported learning_rate_schedule")
    if config.warmup_steps < 0 or config.warmup_steps >= config.train_steps:
        raise ValueError("warmup_steps must be at least 0 and less than train_steps")
    if config.min_learning_rate < 0:
        raise ValueError("min_learning_rate must be greater than or equal to 0")
    if config.min_learning_rate > config.learning_rate:
        raise ValueError("min_learning_rate must not exceed learning_rate")
    if config.max_gradient_norm is not None and config.max_gradient_norm <= 0:
        raise ValueError("max_gradient_norm must be greater than 0 when provided")


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None
