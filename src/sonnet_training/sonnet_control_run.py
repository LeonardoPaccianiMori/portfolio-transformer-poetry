"""Run controlled sonnet experiments with explicit initialization lineage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer
from sonnet_corpus.dataset_text import (
    load_pretraining_bpe_encoded_splits,
    read_manifest_rows,
    validate_manifest_rows,
)
from sonnet_model.transformer import CausalTransformerLanguageModel
from sonnet_training.finetuning_run import generate_finetuning_sample, load_parent_for_finetuning
from sonnet_training.learning_rate import (
    LearningRateSchedule,
    learning_rate_for_step,
    set_optimizer_learning_rate,
)
from sonnet_training.pretraining_run import count_parameters
from sonnet_training.progress import TrainingProgressReporter
from sonnet_training.rmsnorm_conversion import (
    initialize_rms_norm_conversion_from_parent,
)
from sonnet_training.steps import (
    estimate_next_token_loss,
    estimate_next_token_loss_on_sequential_windows,
    sequential_next_token_window_count,
    train_next_token_step,
)
from sonnet_training.transformer_run import resolve_device, write_json, write_jsonl


InitializationMode = Literal["pretrained", "random", "layernorm_to_rmsnorm"]
ValidationMode = Literal["random_batches", "sequential_windows"]
ModelArchitecture = dict[str, int | float | str | bool]
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
    manifest_path: str = "data/metadata/poems_manifest.csv"
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
    validation_mode: ValidationMode = "sequential_windows"
    early_stopping_patience: int = 8
    min_validation_improvement: float = 0.01
    checkpoint_interval: int = 1_000
    progress_interval: int = 100
    learning_rate: float = 3e-5
    learning_rate_schedule: LearningRateSchedule = "constant"
    warmup_steps: int = 0
    min_learning_rate: float = 0.0
    max_gradient_norm: float | None = None
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
    manifest_path = _resolve_repo_path(repo_root, config.manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"sonnet manifest does not exist: {manifest_path}")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    validate_manifest_rows(read_manifest_rows(manifest_path), config.dataset)
    source_model_architecture = load_model_architecture(
        _resolve_repo_path(repo_root, config.model_architecture_path)
    )
    train_tokens, validation_tokens, _, tokenizer = load_pretraining_bpe_encoded_splits(
        manifest_path=manifest_path,
        repo_root=repo_root,
        dataset=config.dataset,
        tokenizer_path=_resolve_repo_path(
            repo_root,
            config.pretraining_tokenizer_path,
        ),
    )
    model_architecture = target_model_architecture(
        initialization=config.initialization,
        source_model_architecture=source_model_architecture,
        target_vocab_size=tokenizer.vocab_size,
    )
    _validate_tokenizer_architecture(tokenizer, model_architecture)
    model, optimizer, parent_checkpoint, initialization_metadata = initialize_control_model(
        repo_root=repo_root,
        config=config,
        tokenizer=tokenizer,
        model_architecture=model_architecture,
        device=device,
    )
    _validate_context_length(config, model)

    history, best_validation_row, completed_steps, stop_reason = train_control_steps(
        model=model,
        optimizer=optimizer,
        train_token_ids=train_tokens,
        validation_token_ids=validation_tokens,
        config=config,
        device=device,
        output_dir=output_dir,
        tokenizer=tokenizer,
        model_architecture=model_architecture,
        initialization_metadata=initialization_metadata,
        parent_checkpoint=parent_checkpoint,
        manifest_sha256=manifest_sha256,
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
        source_model_architecture=source_model_architecture,
        initialization_metadata=initialization_metadata,
        parent_checkpoint=parent_checkpoint,
        manifest_sha256=manifest_sha256,
        best_validation_row=best_validation_row,
        completed_steps=completed_steps,
        stop_reason=stop_reason,
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
        initialization_metadata=initialization_metadata,
        parent_checkpoint=parent_checkpoint,
        manifest_sha256=manifest_sha256,
        step=completed_steps,
        best_validation_row=best_validation_row,
        stop_reason=stop_reason,
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
        "completed_steps": completed_steps,
        "stop_reason": stop_reason,
    }


def load_model_architecture(path: Path) -> ModelArchitecture:
    """Load the architecture manifest shared by every control arm."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    architecture = payload.get("model_architecture", payload)
    missing = [name for name in MODEL_ARCHITECTURE_KEYS if name not in architecture]
    if missing:
        raise ValueError("model architecture is missing fields: " + ", ".join(missing))
    return {
        **{name: int(architecture[name]) for name in MODEL_ARCHITECTURE_KEYS},
        "normalization_type": architecture.get("normalization_type", "layer_norm"),
        "normalization_eps": float(architecture.get("normalization_eps", 1e-5)),
        "position_encoding_type": architecture.get(
            "position_encoding_type",
            "learned_absolute",
        ),
        "rope_theta": float(architecture.get("rope_theta", 10_000.0)),
        "feed_forward_type": architecture.get("feed_forward_type", "relu"),
        "tie_token_embeddings": bool(architecture.get("tie_token_embeddings", False)),
    }


def target_model_architecture(
    *,
    initialization: InitializationMode,
    source_model_architecture: ModelArchitecture,
    target_vocab_size: int | None = None,
) -> ModelArchitecture:
    """Derive the model architecture trained by one controlled arm."""
    if target_vocab_size is not None and target_vocab_size <= 0:
        raise ValueError("target_vocab_size must be greater than 0")
    if (
        initialization == "layernorm_to_rmsnorm"
        and source_model_architecture["normalization_type"] != "layer_norm"
    ):
        raise ValueError("LayerNorm-to-RMSNorm conversion requires LayerNorm architecture")
    target_architecture = {
        **source_model_architecture,
    }
    if initialization == "layernorm_to_rmsnorm":
        target_architecture["normalization_type"] = "rms_norm"
    if target_vocab_size is not None:
        target_architecture["vocab_size"] = target_vocab_size
    return target_architecture


def initialize_control_model(
    *,
    repo_root: Path,
    config: SonnetControlRunConfig,
    tokenizer: BytePairEncodingTokenizer,
    model_architecture: ModelArchitecture,
    device: torch.device,
) -> tuple[
    CausalTransformerLanguageModel,
    torch.optim.Optimizer,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Build one arm while always creating fresh AdamW optimizer state."""
    if config.initialization == "random":
        model = CausalTransformerLanguageModel(**model_architecture).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        return model, optimizer, None, None

    if config.initialization == "pretrained":
        model, optimizer, parent_checkpoint = load_parent_for_finetuning(
            checkpoint_path=_resolve_repo_path(
                repo_root,
                config.pretraining_checkpoint_path,
            ),
            tokenizer=tokenizer,
            learning_rate=config.learning_rate,
            restore_optimizer_state=False,
            device=device,
        )
        _validate_model_architecture(model, model_architecture)
        return model, optimizer, parent_checkpoint, None

    if config.initialization == "layernorm_to_rmsnorm":
        model, optimizer, parent_checkpoint, conversion_metadata = (
            initialize_rms_norm_conversion_from_parent(
                checkpoint_path=_resolve_repo_path(
                    repo_root,
                    config.pretraining_checkpoint_path,
                ),
                tokenizer=tokenizer,
                learning_rate=config.learning_rate,
                device=device,
            )
        )
        _validate_model_architecture(model, model_architecture)
        return model, optimizer, parent_checkpoint, conversion_metadata

    raise ValueError(
        "initialization must be 'pretrained', 'random', or 'layernorm_to_rmsnorm'"
    )


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
    model_architecture: ModelArchitecture,
    initialization_metadata: dict[str, Any] | None,
    parent_checkpoint: dict[str, Any] | None,
    manifest_sha256: str,
) -> tuple[
    list[dict[str, float | int | None]],
    dict[str, float | int | None],
    int,
    str,
]:
    """Train, evaluate, and overwrite one exact best-validation checkpoint."""
    history: list[dict[str, float | int | None]] = []
    best_validation_row: dict[str, float | int | None] | None = None
    non_improving_evaluations = 0
    completed_steps = 0
    stop_reason = "max_train_steps_reached"
    progress = TrainingProgressReporter(
        total_steps=config.train_steps,
        progress_interval=config.progress_interval,
    )
    progress.write_start(label="sonnet control", device=str(device))
    for step in range(1, config.train_steps + 1):
        current_learning_rate = learning_rate_for_step(config, step)
        set_optimizer_learning_rate(optimizer, current_learning_rate)
        train_loss, pre_clipping_gradient_norm = train_next_token_step(
            model=model,
            optimizer=optimizer,
            token_ids=train_token_ids,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=device,
            max_gradient_norm=config.max_gradient_norm,
            return_gradient_norm=True,
        )
        should_evaluate = (
            step == 1
            or step % config.eval_interval == 0
            or step == config.train_steps
        )
        if should_evaluate:
            validation_loss = estimate_control_validation_loss(
                model=model,
                validation_token_ids=validation_token_ids,
                config=config,
                device=device,
            )
            row = {
                "step": step,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": current_learning_rate,
                "pre_clipping_gradient_norm": pre_clipping_gradient_norm,
                "non_improving_evaluations": non_improving_evaluations,
            }
            history.append(row)
            best_validation_updated = False
            if (
                best_validation_row is None
                or row["validation_loss"]
                < best_validation_row["validation_loss"]
                - config.min_validation_improvement
            ):
                best_validation_row = row
                best_validation_updated = True
                non_improving_evaluations = 0
                row["non_improving_evaluations"] = non_improving_evaluations
                save_control_checkpoint(
                    checkpoint_path=output_dir / "best_validation.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    tokenizer=tokenizer,
                    model_architecture=model_architecture,
                    initialization_metadata=initialization_metadata,
                    parent_checkpoint=parent_checkpoint,
                    manifest_sha256=manifest_sha256,
                    step=step,
                    best_validation_row=row,
                    stop_reason=None,
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
            save_control_checkpoint(
                checkpoint_path=output_dir / "checkpoints" / f"step_{step}.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                tokenizer=tokenizer,
                model_architecture=model_architecture,
                initialization_metadata=initialization_metadata,
                parent_checkpoint=parent_checkpoint,
                manifest_sha256=manifest_sha256,
                step=step,
                best_validation_row=best_validation_row,
                stop_reason=None,
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

        completed_steps = step
        if should_stop_early:
            stop_reason = "early_stopping_patience_exhausted"
            break

    if best_validation_row is None:
        raise RuntimeError("control run completed without a validation evaluation")
    return history, best_validation_row, completed_steps, stop_reason


def estimate_control_validation_loss(
    *,
    model: CausalTransformerLanguageModel,
    validation_token_ids: torch.Tensor,
    config: SonnetControlRunConfig,
    device: torch.device,
) -> float:
    """Return validation loss using the configured reproducible evaluation mode."""
    if config.validation_mode == "random_batches":
        return estimate_next_token_loss(
            model=model,
            token_ids=validation_token_ids,
            batch_size=config.batch_size,
            context_length=config.context_length,
            eval_batches=config.eval_batches,
            device=device,
        )
    if config.validation_mode == "sequential_windows":
        return estimate_next_token_loss_on_sequential_windows(
            model=model,
            token_ids=validation_token_ids,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=device,
        )
    raise ValueError("unsupported validation_mode")


def save_control_checkpoint(
    *,
    checkpoint_path: Path,
    model: CausalTransformerLanguageModel,
    optimizer: torch.optim.Optimizer,
    config: SonnetControlRunConfig,
    tokenizer: BytePairEncodingTokenizer,
    model_architecture: ModelArchitecture,
    initialization_metadata: dict[str, Any] | None,
    parent_checkpoint: dict[str, Any] | None,
    manifest_sha256: str,
    step: int,
    best_validation_row: dict[str, float | int | None] | None,
    stop_reason: str | None,
) -> None:
    """Save one control checkpoint with explicit initialization provenance."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "manifest_sha256": manifest_sha256,
            "initialization": config.initialization,
            "optimizer_state_restored": False,
            "vocab_size": tokenizer.vocab_size,
            "parameter_count": count_parameters(model),
            "model_architecture": model_architecture,
            "initialization_metadata": initialization_metadata,
            "parent_checkpoint_step": (
                int(parent_checkpoint["step"])
                if parent_checkpoint is not None
                else None
            ),
            "best_validation_row": best_validation_row,
            "completed_steps": step,
            "stop_reason": stop_reason,
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
    model_architecture: ModelArchitecture,
    source_model_architecture: ModelArchitecture,
    initialization_metadata: dict[str, Any] | None,
    parent_checkpoint: dict[str, Any] | None,
    manifest_sha256: str,
    best_validation_row: dict[str, float | int | None],
    completed_steps: int,
    stop_reason: str,
) -> dict[str, Any]:
    """Create reproducibility metadata shared by reports and selection tooling."""
    return {
        **asdict(config),
        "manifest_sha256": manifest_sha256,
        "resolved_device": str(device),
        "optimizer_state_restored": False,
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": int(train_tokens.numel()),
        "validation_tokens": int(validation_tokens.numel()),
        "validation_window_count": sequential_next_token_window_count(
            validation_tokens,
            config.context_length,
        ),
        "parameter_count": count_parameters(model),
        "model_architecture": model_architecture,
        "source_model_architecture": source_model_architecture,
        "initialization_metadata": initialization_metadata,
        "parent_checkpoint_step": (
            int(parent_checkpoint["step"])
            if parent_checkpoint is not None
            else None
        ),
        "completed_steps": completed_steps,
        "stop_reason": stop_reason,
        "best_validation_step": int(best_validation_row["step"]),
        "best_validation_loss": float(best_validation_row["validation_loss"]),
    }


def _validate_tokenizer_architecture(
    tokenizer: BytePairEncodingTokenizer,
    model_architecture: ModelArchitecture,
) -> None:
    if tokenizer.vocab_size != model_architecture["vocab_size"]:
        raise ValueError("tokenizer vocabulary size does not match model architecture")


def _resolve_repo_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path


def _validate_model_architecture(
    model: CausalTransformerLanguageModel,
    model_architecture: ModelArchitecture,
) -> None:
    if model.output_projection.out_features != model_architecture["vocab_size"]:
        raise ValueError("pretrained model vocabulary does not match model architecture")
    if model.max_context_length != model_architecture["max_context_length"]:
        raise ValueError("pretrained model context does not match model architecture")
    normalization_type = model_architecture.get("normalization_type", "layer_norm")
    if model.normalization_type != normalization_type:
        raise ValueError("pretrained model normalization does not match model architecture")
    normalization_eps = float(model_architecture.get("normalization_eps", 1e-5))
    if model.normalization_eps != normalization_eps:
        raise ValueError(
            "pretrained model normalization epsilon does not match model architecture"
        )
    position_encoding_type = model_architecture.get(
        "position_encoding_type",
        "learned_absolute",
    )
    if model.position_encoding_type != position_encoding_type:
        raise ValueError(
            "pretrained model position encoding does not match model architecture"
        )
    rope_theta = float(model_architecture.get("rope_theta", 10_000.0))
    if model.rope_theta != rope_theta:
        raise ValueError(
            "pretrained model RoPE theta does not match model architecture"
        )
    feed_forward_type = model_architecture.get("feed_forward_type", "relu")
    if model.feed_forward_type != feed_forward_type:
        raise ValueError(
            "pretrained model feed-forward type does not match model architecture"
        )
    tie_token_embeddings = bool(
        model_architecture.get("tie_token_embeddings", False)
    )
    if model.tie_token_embeddings != tie_token_embeddings:
        raise ValueError(
            "pretrained model tied-embedding setting does not match model architecture"
        )


def _validate_context_length(
    config: SonnetControlRunConfig,
    model: CausalTransformerLanguageModel,
) -> None:
    if config.context_length > model.max_context_length:
        raise ValueError(
            "context_length must be less than or equal to model max_context_length"
        )


def _validate_config(config: SonnetControlRunConfig) -> None:
    if config.initialization not in {"pretrained", "random", "layernorm_to_rmsnorm"}:
        raise ValueError(
            "initialization must be 'pretrained', 'random', or 'layernorm_to_rmsnorm'"
        )
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
    if config.validation_mode not in {"random_batches", "sequential_windows"}:
        raise ValueError("unsupported validation_mode")
    if config.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be greater than or equal to 0")
    if config.min_validation_improvement < 0:
        raise ValueError("min_validation_improvement must be greater than or equal to 0")
    if config.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be greater than or equal to 0")
    if config.progress_interval <= 0:
        raise ValueError("progress_interval must be greater than 0")
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
    if config.max_gradient_norm is not None and config.max_gradient_norm <= 0:
        raise ValueError("max_gradient_norm must be greater than 0 when provided")
    if config.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be greater than or equal to 0")
