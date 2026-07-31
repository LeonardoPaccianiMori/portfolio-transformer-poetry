"""Run broader Italian from-scratch transformer pretraining."""

from __future__ import annotations

import json
from hashlib import sha256
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer
from sonnet_corpus.paisa_historical_encoding import load_memory_mapped_token_ids
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
CheckpointRetention = Literal["all", "latest_only"]


PRETRAINING_DATASET_VERSION = "pretraining_historical_italian_v2"
PRETRAINING_TRAIN_TOKENS_PATH = (
    "data/local/pretraining/pretraining_historical_italian_v2/encoded/"
    "bpe_16000_train.pt"
)
PRETRAINING_VALIDATION_TOKENS_PATH = (
    "data/local/pretraining/pretraining_historical_italian_v2/encoded/"
    "bpe_16000_validation.pt"
)
PRETRAINING_TOKENIZER_PATH = (
    "data/metadata/pretraining_tokenizers/"
    "pretraining_historical_italian_v2_bpe_16000.json"
)
PRETRAINING_DATASET_REPORT_PATH = (
    "reports/pretraining_historical_italian_v2_encoded_report.json"
)


@dataclass(frozen=True)
class PretrainingRunConfig:
    """Configuration for a broader-corpus pretraining run."""

    dataset_version: str = PRETRAINING_DATASET_VERSION
    train_tokens_path: str = PRETRAINING_TRAIN_TOKENS_PATH
    validation_tokens_path: str = PRETRAINING_VALIDATION_TOKENS_PATH
    tokenizer_path: str = PRETRAINING_TOKENIZER_PATH
    dataset_report_path: str = PRETRAINING_DATASET_REPORT_PATH
    run_label: str = "pretraining"
    train_split_id: str = "train"
    validation_split_id: str = "validation"
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    context_length: int = 512
    train_steps: int = 100
    eval_interval: int = 25
    eval_batches: int = 5
    validation_mode: ValidationMode = "random_batches"
    learning_rate: float = 3e-4
    learning_rate_schedule: LearningRateSchedule = "constant"
    warmup_steps: int = 0
    stable_steps: int = 0
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
    checkpoint_retention: CheckpointRetention = "all"
    progress_interval: int = 100
    resume_from_checkpoint: str = ""
    initialization_checkpoint_path: str = ""


class PretrainingDatasetArtifactConfig(Protocol):
    """Paths and identity required to validate pretraining data artifacts."""

    dataset_version: str
    train_tokens_path: str | Path
    validation_tokens_path: str | Path
    tokenizer_path: str | Path
    dataset_report_path: str | Path
    train_split_id: str
    validation_split_id: str


def train_pretraining_run(
    repo_root: Path,
    output_dir: Path,
    config: PretrainingRunConfig,
) -> dict[str, Path | list[dict[str, float | int]]]:
    """Train the transformer briefly on broader pretraining token tensors."""

    _validate_config(config)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)

    train_tokens, validation_tokens = load_pretraining_token_splits(
        repo_root=repo_root,
        config=config,
    )
    tokenizer = BytePairEncodingTokenizer.load(repo_root / config.tokenizer_path)
    dataset_provenance = validate_pretraining_dataset_artifacts(
        repo_root=repo_root,
        config=config,
        tokenizer=tokenizer,
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
    )

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
    checkpoint_best_validation_row = None
    initialization_metadata = None
    if config.initialization_checkpoint_path and not config.resume_from_checkpoint:
        initialization_metadata = initialize_pretraining_model_from_checkpoint(
            checkpoint_path=repo_root / config.initialization_checkpoint_path,
            model=model,
            tokenizer=tokenizer,
        )
    if config.resume_from_checkpoint:
        start_step, checkpoint_best_validation_row = (
            load_pretraining_checkpoint_with_best_validation(
            checkpoint_path=repo_root / config.resume_from_checkpoint,
            model=model,
            optimizer=optimizer,
            device=device,
            )
        )
        if start_step >= config.train_steps:
            raise ValueError(
                "resume checkpoint step must be less than train_steps"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    prior_history = _read_history(output_dir / "loss_history.jsonl")
    initial_best_validation_row = _select_best_validation_row(
        _best_validation_row(prior_history, through_step=start_step),
        checkpoint_best_validation_row,
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
            "microbatch_tokens": microbatch_token_count(config),
            "tokens_per_optimizer_step": optimizer_step_token_count(config),
            "planned_train_token_exposures": planned_train_token_exposures(config),
            "dataset_provenance": dataset_provenance,
            "validation_window_count": sequential_next_token_window_count(
                validation_tokens,
                config.context_length,
            ),
            "parameter_count": count_parameters(model),
            "start_step": start_step,
            "completed_steps": config.train_steps,
            "best_validation_step": int(best_validation_row["step"]),
            "best_validation_loss": float(best_validation_row["validation_loss"]),
            "initialization": initialization_metadata,
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
        "resume_checkpoint_path": output_dir / "resume.pt",
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
    progress.write_start(
        label=config.run_label,
        device=str(device),
        tokens_per_step=optimizer_step_token_count(config),
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

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
            gradient_accumulation_steps=config.gradient_accumulation_steps,
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
                    include_optimizer_state=False,
                )
        else:
            validation_loss = None
            best_validation_updated = False

        checkpoint_written = False
        if config.checkpoint_interval and step % config.checkpoint_interval == 0:
            checkpoint_path = checkpoint_path_for_interval(
                output_dir=output_dir,
                step=step,
                retention=config.checkpoint_retention,
            )
            save_pretraining_checkpoint(
                checkpoint_path=checkpoint_path,
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
    include_optimizer_state: bool = True,
) -> None:
    """Save model state and optionally optimizer state using atomic replacement."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict() if include_optimizer_state else None
            ),
            "config": asdict(config),
            "vocab_size": tokenizer.vocab_size,
            "parameter_count": count_parameters(model),
            "best_validation_row": best_validation_row,
        },
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)


def load_pretraining_checkpoint(
    *,
    checkpoint_path: Path,
    model: CausalTransformerLanguageModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    """Load model/optimizer state and return the completed checkpoint step."""

    step, _ = load_pretraining_checkpoint_with_best_validation(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        device=device,
    )
    return step


def initialize_pretraining_model_from_checkpoint(
    *,
    checkpoint_path: Path,
    model: CausalTransformerLanguageModel,
    tokenizer: BytePairEncodingTokenizer,
) -> dict[str, object]:
    """Load model weights only and return immutable parent-checkpoint metadata."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint must contain a dictionary: {checkpoint_path}")
    if checkpoint.get("vocab_size") != tokenizer.vocab_size:
        raise ValueError("initialization checkpoint vocabulary size does not match")
    model_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise ValueError("initialization checkpoint is missing model_state_dict")

    model.load_state_dict(model_state_dict)
    source_config = checkpoint.get("config")
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "source_step": int(checkpoint.get("step", 0)),
        "source_parameter_count": int(checkpoint.get("parameter_count", 0)),
        "source_dataset_version": (
            source_config.get("dataset_version", "")
            if isinstance(source_config, dict)
            else ""
        ),
        "optimizer_state_reused": False,
    }


def load_pretraining_checkpoint_with_best_validation(
    *,
    checkpoint_path: Path,
    model: CausalTransformerLanguageModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, dict[str, float | int] | None]:
    """Restore a resumable checkpoint and return its selected validation row."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint must contain a dictionary: {checkpoint_path}")
    if checkpoint.get("optimizer_state_dict") is None:
        raise ValueError(f"checkpoint is not resumable: {checkpoint_path}")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["step"]), _checkpoint_best_validation_row(checkpoint)


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


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _best_validation_row(
    history: list[dict[str, float | int]],
    *,
    through_step: int,
) -> dict[str, float | int] | None:
    eligible_rows = [row for row in history if int(row["step"]) <= through_step]
    if not eligible_rows:
        return None
    return min(eligible_rows, key=lambda row: float(row["validation_loss"]))


def _select_best_validation_row(
    *rows: dict[str, float | int] | None,
) -> dict[str, float | int] | None:
    """Return the lowest-loss row available from history and resume state."""

    available_rows = [row for row in rows if row is not None]
    if not available_rows:
        return None
    return min(available_rows, key=lambda row: float(row["validation_loss"]))


def _checkpoint_best_validation_row(
    checkpoint: dict[str, object],
) -> dict[str, float | int] | None:
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
    }


def load_pretraining_token_splits(
    *,
    repo_root: Path,
    config: PretrainingDatasetArtifactConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load configured train/validation streams using their report token counts."""

    report_path = repo_root / config.dataset_report_path
    report = _read_dataset_report(report_path)
    train_token_count = _configured_split_token_count(
        report=report,
        config=config,
        role="train",
    )
    validation_token_count = _configured_split_token_count(
        report=report,
        config=config,
        role="validation",
    )
    return (
        load_token_tensor(
            repo_root / config.train_tokens_path,
            token_count=train_token_count,
        ),
        load_token_tensor(
            repo_root / config.validation_tokens_path,
            token_count=validation_token_count,
        ),
    )


def load_token_tensor(
    path: Path,
    *,
    token_count: int | None = None,
) -> torch.Tensor:
    """Load a legacy long tensor or map a compact uint16 token stream."""

    if not path.is_file():
        raise FileNotFoundError(f"token tensor file does not exist: {path}")

    if path.name.endswith(".uint16.bin"):
        if token_count is None:
            raise ValueError("uint16 token streams require an expected token_count")
        return load_memory_mapped_token_ids(path, token_count=token_count)

    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"token tensor file did not contain a torch.Tensor: {path}")
    if tensor.ndim != 1:
        raise ValueError(f"token tensor must be 1D: {path}")
    if tensor.dtype != torch.long:
        raise ValueError(f"token tensor must use dtype torch.long: {path}")
    return tensor


def validate_pretraining_dataset_artifacts(
    *,
    repo_root: Path,
    config: PretrainingDatasetArtifactConfig,
    tokenizer: BytePairEncodingTokenizer,
    train_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
) -> dict[str, int | str]:
    """Require loaded local tensors to match their committed dataset report."""

    report_path = repo_root / config.dataset_report_path
    report = _read_dataset_report(report_path)

    if isinstance(report.get("splits"), list):
        return _validate_memory_mapped_dataset_artifacts(
            repo_root=repo_root,
            config=config,
            tokenizer=tokenizer,
            report=report,
            train_tokens=train_tokens,
            validation_tokens=validation_tokens,
        )

    _require_report_path_match(
        report=report,
        field="train_path",
        expected_path=config.train_tokens_path,
        repo_root=repo_root,
    )
    _require_report_path_match(
        report=report,
        field="validation_path",
        expected_path=config.validation_tokens_path,
        repo_root=repo_root,
    )
    _require_report_path_match(
        report=report,
        field="tokenizer_path",
        expected_path=config.tokenizer_path,
        repo_root=repo_root,
    )
    _require_report_int_match(
        report=report,
        field="vocab_size",
        actual_value=tokenizer.vocab_size,
    )
    _require_report_int_match(
        report=report,
        field="train_tokens",
        actual_value=int(train_tokens.numel()),
    )
    _require_report_int_match(
        report=report,
        field="validation_tokens",
        actual_value=int(validation_tokens.numel()),
    )
    _require_report_int_match(
        report=report,
        field="total_tokens",
        actual_value=int(train_tokens.numel() + validation_tokens.numel()),
    )
    _require_report_string_match(
        report=report,
        field="train_dtype",
        actual_value=str(train_tokens.dtype),
    )
    _require_report_string_match(
        report=report,
        field="validation_dtype",
        actual_value=str(validation_tokens.dtype),
    )

    separator_ids = tokenizer.encode(
        str(report.get("document_separator", ""))
    )
    if len(separator_ids) != 1:
        raise ValueError("dataset report separator must encode to exactly one token")
    _require_report_int_match(
        report=report,
        field="document_separator_token_id",
        actual_value=separator_ids[0],
    )
    _validate_token_id_range(train_tokens, vocab_size=tokenizer.vocab_size, name="train")
    _validate_token_id_range(
        validation_tokens,
        vocab_size=tokenizer.vocab_size,
        name="validation",
    )

    source_count = _require_positive_report_int(report=report, field="source_count")
    sources = report.get("sources")
    if not isinstance(sources, list) or len(sources) != source_count:
        raise ValueError("dataset report source_count does not match its sources list")
    return {
        "dataset_version": config.dataset_version,
        "dataset_report_path": str(config.dataset_report_path),
        "source_count": source_count,
        "split_policy": str(report.get("split_policy", "")),
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": int(train_tokens.numel()),
        "validation_tokens": int(validation_tokens.numel()),
    }


def _require_report_path_match(
    *,
    report: dict[str, object],
    field: str,
    expected_path: str | Path,
    repo_root: Path,
) -> None:
    report_value = report.get(field)
    if not isinstance(report_value, str):
        raise ValueError(f"dataset report is missing string field: {field}")
    reported_path = (repo_root / report_value).resolve()
    configured_path = (repo_root / expected_path).resolve()
    if reported_path != configured_path:
        raise ValueError(f"dataset report {field} does not match run configuration")


def _read_dataset_report(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset report file does not exist: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"dataset report must contain a JSON object: {path}")
    return report


def _configured_split_token_count(
    *,
    report: dict[str, object],
    config: PretrainingDatasetArtifactConfig,
    role: str,
) -> int | None:
    if not isinstance(report.get("splits"), list):
        return _require_positive_report_int(report=report, field=f"{role}_tokens")
    split = _report_split_for_role(report=report, config=config, role=role)
    return _require_positive_mapping_int(mapping=split, field="tokens")


def _validate_memory_mapped_dataset_artifacts(
    *,
    repo_root: Path,
    config: PretrainingDatasetArtifactConfig,
    tokenizer: BytePairEncodingTokenizer,
    report: dict[str, object],
    train_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
) -> dict[str, int | str]:
    if report.get("status") != "complete":
        raise ValueError("memory-mapped dataset report is not complete")
    tokenizer_report = report.get("tokenizer")
    if not isinstance(tokenizer_report, dict):
        raise ValueError("memory-mapped dataset report is missing tokenizer metadata")
    _require_mapping_path_match(
        mapping=tokenizer_report,
        field="path",
        expected_path=config.tokenizer_path,
        repo_root=repo_root,
    )
    _require_mapping_int_match(
        mapping=tokenizer_report,
        field="vocab_size",
        actual_value=tokenizer.vocab_size,
    )

    train_split = _report_split_for_role(report=report, config=config, role="train")
    validation_split = _report_split_for_role(
        report=report,
        config=config,
        role="validation",
    )
    _validate_memory_mapped_split(
        split=train_split,
        expected_path=config.train_tokens_path,
        token_ids=train_tokens,
        tokenizer=tokenizer,
        name="train",
        repo_root=repo_root,
    )
    _validate_memory_mapped_split(
        split=validation_split,
        expected_path=config.validation_tokens_path,
        token_ids=validation_tokens,
        tokenizer=tokenizer,
        name="validation",
        repo_root=repo_root,
    )
    separator_ids = tokenizer.encode(
        str(tokenizer_report.get("document_separator", ""))
    )
    if len(separator_ids) != 1:
        raise ValueError("dataset report separator must encode to exactly one token")
    _require_mapping_int_match(
        mapping=tokenizer_report,
        field="document_separator_token_id",
        actual_value=separator_ids[0],
    )
    return {
        "dataset_version": config.dataset_version,
        "dataset_report_path": str(config.dataset_report_path),
        "stream_count": 2,
        "document_count": (
            _require_positive_mapping_int(mapping=train_split, field="documents")
            + _require_positive_mapping_int(
                mapping=validation_split,
                field="documents",
            )
        ),
        "split_policy": str(report.get("split_policy", "")),
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": int(train_tokens.numel()),
        "validation_tokens": int(validation_tokens.numel()),
    }


def _report_split_for_role(
    *,
    report: dict[str, object],
    config: PretrainingDatasetArtifactConfig,
    role: str,
) -> dict[str, object]:
    split_id = getattr(config, f"{role}_split_id", role)
    splits = report.get("splits")
    if not isinstance(splits, list):
        raise ValueError("memory-mapped dataset report is missing splits")
    for split in splits:
        if isinstance(split, dict) and split.get("split_id") == split_id:
            return split
    raise ValueError(
        f"dataset report does not contain configured {role} split: {split_id}"
    )


def _validate_memory_mapped_split(
    *,
    split: dict[str, object],
    expected_path: str | Path,
    token_ids: torch.Tensor,
    tokenizer: BytePairEncodingTokenizer,
    name: str,
    repo_root: Path,
) -> None:
    if split.get("status") != "complete":
        raise ValueError(f"{name} split is not complete")
    _require_mapping_path_match(
        mapping=split,
        field="output_path",
        expected_path=expected_path,
        repo_root=repo_root,
    )
    _require_mapping_int_match(
        mapping=split,
        field="tokens",
        actual_value=int(token_ids.numel()),
    )
    if split.get("dtype") != "torch.uint16" or token_ids.dtype != torch.uint16:
        raise ValueError(f"{name} split must use dtype torch.uint16")
    _validate_token_id_range(token_ids, vocab_size=tokenizer.vocab_size, name=name)


def _require_mapping_path_match(
    *,
    mapping: dict[str, object],
    field: str,
    expected_path: str | Path,
    repo_root: Path,
) -> None:
    value = mapping.get(field)
    if not isinstance(value, str):
        raise ValueError(f"dataset report is missing string field: {field}")
    reported_path = (repo_root / value).resolve()
    configured_path = (repo_root / expected_path).resolve()
    if reported_path != configured_path:
        raise ValueError(f"dataset report {field} does not match run configuration")


def _require_mapping_int_match(
    *,
    mapping: dict[str, object],
    field: str,
    actual_value: int,
) -> None:
    value = mapping.get(field)
    if not isinstance(value, int):
        raise ValueError(f"dataset report is missing integer field: {field}")
    if value != actual_value:
        raise ValueError(f"dataset report {field} does not match loaded artifact")


def _require_positive_mapping_int(*, mapping: dict[str, object], field: str) -> int:
    value = mapping.get(field)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"dataset report must contain a positive integer: {field}")
    return value


def _require_report_int_match(
    *,
    report: dict[str, object],
    field: str,
    actual_value: int,
) -> None:
    report_value = report.get(field)
    if not isinstance(report_value, int):
        raise ValueError(f"dataset report is missing integer field: {field}")
    if report_value != actual_value:
        raise ValueError(f"dataset report {field} does not match loaded artifact")


def _require_report_string_match(
    *,
    report: dict[str, object],
    field: str,
    actual_value: str,
) -> None:
    report_value = report.get(field)
    if report_value != actual_value:
        raise ValueError(f"dataset report {field} does not match loaded artifact")


def _require_positive_report_int(*, report: dict[str, object], field: str) -> int:
    report_value = report.get(field)
    if not isinstance(report_value, int) or report_value <= 0:
        raise ValueError(f"dataset report must contain a positive integer: {field}")
    return report_value


def _validate_token_id_range(
    token_ids: torch.Tensor,
    *,
    vocab_size: int,
    name: str,
) -> None:
    if token_ids.numel() == 0:
        raise ValueError(f"{name} token tensor must not be empty")
    if token_ids.dtype == torch.uint16:
        # PyTorch does not implement reductions directly for CPU uint16 tensors.
        # Convert bounded chunks instead of materializing a full int64 corpus.
        minimum, maximum = _uint16_token_id_range(token_ids)
    else:
        minimum = int(token_ids.min())
        maximum = int(token_ids.max())
    if minimum < 0 or maximum >= vocab_size:
        raise ValueError(
            f"{name} token IDs must be in [0, {vocab_size - 1}], "
            f"found [{minimum}, {maximum}]"
        )


def _uint16_token_id_range(token_ids: torch.Tensor) -> tuple[int, int]:
    chunk_size = 1_048_576
    minimum = 65_535
    maximum = 0
    for start in range(0, token_ids.numel(), chunk_size):
        chunk = token_ids[start : start + chunk_size].to(dtype=torch.long)
        minimum = min(minimum, int(chunk.min()))
        maximum = max(maximum, int(chunk.max()))
    return minimum, maximum


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable and non-trainable model parameters."""

    return sum(parameter.numel() for parameter in model.parameters())


def microbatch_token_count(config: PretrainingRunConfig) -> int:
    """Return next-token targets processed by one microbatch."""

    return config.batch_size * config.context_length


def optimizer_step_token_count(config: PretrainingRunConfig) -> int:
    """Return next-token targets represented by one optimizer update."""

    return microbatch_token_count(config) * config.gradient_accumulation_steps


def planned_train_token_exposures(config: PretrainingRunConfig) -> int:
    """Return the planned count of sampled next-token targets for one run."""

    return config.train_steps * optimizer_step_token_count(config)


def checkpoint_path_for_interval(
    *,
    output_dir: Path,
    step: int,
    retention: CheckpointRetention,
) -> Path:
    """Return an interval-checkpoint path under the configured retention policy."""

    if retention == "all":
        return output_dir / "checkpoints" / f"step_{step}.pt"
    if retention == "latest_only":
        return output_dir / "resume.pt"
    raise ValueError("unsupported checkpoint_retention")


def _validate_config(config: PretrainingRunConfig) -> None:
    if config.context_length > config.max_context_length:
        raise ValueError(
            "context_length must be less than or equal to max_context_length"
        )
    if config.context_length <= 0:
        raise ValueError("context_length must be greater than 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if not config.train_split_id:
        raise ValueError("train_split_id must not be empty")
    if not config.validation_split_id:
        raise ValueError("validation_split_id must not be empty")
    if config.train_split_id == config.validation_split_id:
        raise ValueError("train_split_id and validation_split_id must differ")
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
    if config.learning_rate_schedule not in {
        "constant",
        "warmup_cosine",
        "warmup_stable_cosine",
    }:
        raise ValueError("unsupported learning_rate_schedule")
    if config.warmup_steps < 0 or config.warmup_steps > config.train_steps:
        raise ValueError("warmup_steps must be between 0 and train_steps")
    if config.learning_rate_schedule == "warmup_stable_cosine":
        if config.stable_steps < config.warmup_steps:
            raise ValueError("stable_steps must not be less than warmup_steps")
        if config.stable_steps >= config.train_steps:
            raise ValueError("stable_steps must be less than train_steps")
    if config.min_learning_rate < 0:
        raise ValueError("min_learning_rate must be greater than or equal to 0")
    if config.min_learning_rate > config.learning_rate:
        raise ValueError("min_learning_rate must not exceed learning_rate")
    if config.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be greater than or equal to 0")
    if config.checkpoint_retention not in {"all", "latest_only"}:
        raise ValueError("unsupported checkpoint_retention")
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be greater than 0")
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
