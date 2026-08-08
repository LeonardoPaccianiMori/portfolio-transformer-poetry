"""Fixed adapter-only Minerva 3B sonnet-continuation fine-tuning run."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from sonnet_corpus.task_format import (
    IGNORE_INDEX,
    SonnetContinuationExample,
    build_sonnet_continuation_examples,
)
from sonnet_training.minerva_qlora import (
    MINERVA_3B_MODEL_ID,
    MINERVA_3B_REVISION,
    MINERVA_QLORA_TARGET_MODULES,
)
from sonnet_training.progress import TrainingProgressReporter
from sonnet_training.transformer_run import write_json, write_jsonl


MINERVA_TASK_FORMAT_VERSION = "opening_line_newline_continuation_v1"
RUN_LABEL = "minerva_3b_qlora_v5_opening_line_continuation"


@dataclass(frozen=True)
class MinervaQLoRAFineTuningConfig:
    """Lock the one approved Minerva comparison recipe before execution."""

    model_id: str = MINERVA_3B_MODEL_ID
    revision: str = MINERVA_3B_REVISION
    dataset: str = "expanded_with_petrarch"
    manifest_path: str = "data/metadata/sonnets_expanded_v5_manifest.csv"
    cache_dir: str = "data/local/minerva_qlora/huggingface"
    context_length: int = 512
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_epochs: int = 20
    learning_rate: float = 1e-4
    warmup_fraction: float = 0.05
    min_learning_rate: float = 1e-5
    early_stopping_patience: int = 3
    min_validation_improvement: float = 0.01
    max_gradient_norm: float = 1.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = MINERVA_QLORA_TARGET_MODULES
    seed: int = 1337
    progress_interval: int = 25
    device: str = "cuda:0"
    resume_from_checkpoint: str = ""


@dataclass(frozen=True)
class TokenizedMinervaContinuationExample:
    """One Minerva-tokenized first-line continuation example."""

    example: SonnetContinuationExample
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    continuation_target_start: int


@dataclass(frozen=True)
class MinervaContinuationBatch:
    """A right-padded causal-language-model batch with masked prompt labels."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    supervised_target_count: int


@dataclass(frozen=True)
class MinervaQLoRATrainingPlan:
    """Resolved update counts for the fixed epoch-based fine-tuning run."""

    updates_per_epoch: int
    planned_updates: int
    warmup_steps: int


def build_minerva_continuation_prompt(opening_line: str) -> str:
    """Return the user-visible prefix used for training and later generation."""
    if not opening_line.strip():
        raise ValueError("opening_line must not be empty")
    if "\n" in opening_line or "\r" in opening_line:
        raise ValueError("opening_line must contain exactly one line")
    return f"{opening_line}\n"


def tokenize_minerva_continuation_example(
    *,
    example: SonnetContinuationExample,
    tokenizer: Any,
    context_length: int,
) -> TokenizedMinervaContinuationExample:
    """Tokenize a complete sonnet while excluding its first line from loss."""
    if context_length <= 0:
        raise ValueError("context_length must be greater than 0")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token_id, int) or eos_token_id < 0:
        raise ValueError("Minerva tokenizer must define a non-negative eos_token_id")

    prompt = build_minerva_continuation_prompt(example.opening_line)
    full_text = f"{prompt}{example.continuation_text}\n"
    prompt_ids = _token_ids(tokenizer, prompt)
    full_ids = [*_token_ids(tokenizer, full_text), eos_token_id]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Minerva prompt tokenization must remain an exact prefix")
    if len(full_ids) > context_length:
        raise ValueError(
            "Minerva continuation example exceeds context_length: "
            f"{len(full_ids)} > {context_length} for {example.poem_id}"
        )
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("Minerva continuation example must include target tokens")

    labels = [
        *([IGNORE_INDEX] * len(prompt_ids)),
        *full_ids[len(prompt_ids):],
    ]
    if len(labels) != len(full_ids):
        raise AssertionError("Minerva labels must align with input_ids")

    return TokenizedMinervaContinuationExample(
        example=example,
        input_ids=tuple(full_ids),
        labels=tuple(labels),
        continuation_target_start=len(prompt_ids),
    )


def load_minerva_continuation_splits(
    *,
    manifest_path: Path,
    repo_root: Path,
    dataset: str,
    tokenizer: Any,
    context_length: int,
) -> tuple[
    list[TokenizedMinervaContinuationExample],
    list[TokenizedMinervaContinuationExample],
    list[TokenizedMinervaContinuationExample],
]:
    """Load all document-disjoint V5 splits with the exact Minerva tokenizer."""
    encoded_splits = []
    for split in ("train", "validation", "test"):
        examples = build_sonnet_continuation_examples(
            manifest_path=manifest_path,
            repo_root=repo_root,
            dataset=dataset,
            split=split,
        )
        encoded_splits.append([
            tokenize_minerva_continuation_example(
                example=example,
                tokenizer=tokenizer,
                context_length=context_length,
            )
            for example in examples
        ])
    return tuple(encoded_splits)  # type: ignore[return-value]


def collate_minerva_continuation_examples(
    *,
    examples: Sequence[TokenizedMinervaContinuationExample],
    pad_token_id: int,
    device: torch.device | str | None = None,
) -> MinervaContinuationBatch:
    """Right-pad examples while retaining labels only for continuation tokens."""
    if not examples:
        raise ValueError("examples must not be empty")
    if pad_token_id < 0:
        raise ValueError("pad_token_id must be non-negative")

    maximum_length = max(len(example.input_ids) for example in examples)
    input_ids = torch.full(
        (len(examples), maximum_length),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (len(examples), maximum_length),
        dtype=torch.long,
    )
    labels = torch.full(
        (len(examples), maximum_length),
        fill_value=IGNORE_INDEX,
        dtype=torch.long,
    )
    for index, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[index, :length] = torch.tensor(example.input_ids, dtype=torch.long)
        attention_mask[index, :length] = 1
        labels[index, :length] = torch.tensor(example.labels, dtype=torch.long)

    supervised_target_count = int((labels != IGNORE_INDEX).sum().item())
    if supervised_target_count == 0:
        raise ValueError("Minerva batch must contain at least one target token")
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

    return MinervaContinuationBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        supervised_target_count=supervised_target_count,
    )


def build_training_plan(
    *,
    config: MinervaQLoRAFineTuningConfig,
    train_example_count: int,
) -> MinervaQLoRATrainingPlan:
    """Resolve full-pass update counts before training begins."""
    validate_finetuning_config(config)
    if train_example_count <= 0:
        raise ValueError("train_example_count must be greater than 0")
    examples_per_update = config.batch_size * config.gradient_accumulation_steps
    updates_per_epoch = math.ceil(train_example_count / examples_per_update)
    planned_updates = updates_per_epoch * config.max_epochs
    warmup_steps = max(1, math.ceil(planned_updates * config.warmup_fraction))
    if warmup_steps >= planned_updates:
        raise ValueError("warmup_steps must be less than planned_updates")
    return MinervaQLoRATrainingPlan(
        updates_per_epoch=updates_per_epoch,
        planned_updates=planned_updates,
        warmup_steps=warmup_steps,
    )


def learning_rate_for_qlora_step(
    *,
    config: MinervaQLoRAFineTuningConfig,
    plan: MinervaQLoRATrainingPlan,
    step: int,
) -> float:
    """Return the fixed warmup-plus-cosine learning rate for one update."""
    if step <= 0 or step > plan.planned_updates:
        raise ValueError("step must be within the planned update range")
    if step <= plan.warmup_steps:
        return config.learning_rate * step / plan.warmup_steps

    decay_progress = (step - plan.warmup_steps) / (
        plan.planned_updates - plan.warmup_steps
    )
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return config.min_learning_rate + cosine_factor * (
        config.learning_rate - config.min_learning_rate
    )


def validate_finetuning_config(config: MinervaQLoRAFineTuningConfig) -> None:
    """Protect the predeclared one-run comparison from recipe drift."""
    expected_values = {
        "model_id": MINERVA_3B_MODEL_ID,
        "revision": MINERVA_3B_REVISION,
        "dataset": "expanded_with_petrarch",
        "context_length": 512,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_epochs": 20,
        "learning_rate": 1e-4,
        "warmup_fraction": 0.05,
        "min_learning_rate": 1e-5,
        "early_stopping_patience": 3,
        "min_validation_improvement": 0.01,
        "max_gradient_norm": 1.0,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": MINERVA_QLORA_TARGET_MODULES,
        "seed": 1337,
    }
    for name, expected in expected_values.items():
        if getattr(config, name) != expected:
            raise ValueError(
                f"Minerva QLoRA {name} is locked to {expected!r} for this comparison"
            )
    if config.progress_interval <= 0:
        raise ValueError("progress_interval must be greater than 0")


def train_minerva_qlora_run(
    *,
    repo_root: Path,
    output_dir: Path,
    config: MinervaQLoRAFineTuningConfig,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the single selected Minerva QLoRA sonnet comparison experiment."""
    validate_finetuning_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Minerva QLoRA fine-tuning requires an available CUDA GPU")
    dependencies = _load_qlora_dependencies()
    device = torch.device(config.device)
    if device.type != "cuda":
        raise ValueError("Minerva QLoRA fine-tuning requires a CUDA device")
    torch.manual_seed(config.seed)
    cache_dir = _resolve_repo_path(repo_root, config.cache_dir)
    manifest_path = _resolve_repo_path(repo_root, config.manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    _report(progress, "loading Minerva tokenizer")
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        config.model_id,
        revision=config.revision,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id
    if not isinstance(pad_token_id, int) or pad_token_id < 0:
        raise ValueError("Minerva tokenizer must define a non-negative pad_token_id")

    _report(progress, "tokenizing V5 sonnet continuation splits")
    train_examples, validation_examples, test_examples = load_minerva_continuation_splits(
        manifest_path=manifest_path,
        repo_root=repo_root,
        dataset=config.dataset,
        tokenizer=tokenizer,
        context_length=config.context_length,
    )
    plan = build_training_plan(config=config, train_example_count=len(train_examples))

    _report(progress, "loading Minerva 3B in 4-bit NF4 and attaching LoRA adapters")
    model = _load_trainable_qlora_model(
        dependencies=dependencies,
        config=config,
        cache_dir=cache_dir,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = dependencies["bitsandbytes"].optim.PagedAdamW8bit(
        trainable_parameters,
        lr=config.learning_rate,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    start_step = 0
    history: list[dict[str, Any]] = []
    best_validation_row: dict[str, Any] | None = None
    non_improving_evaluations = 0
    generator = torch.Generator().manual_seed(config.seed)
    if config.resume_from_checkpoint:
        _report(progress, "restoring adapter-only resume checkpoint")
        (
            start_epoch,
            start_step,
            history,
            best_validation_row,
            non_improving_evaluations,
            generator_state,
        ) = _load_resume_checkpoint(
            checkpoint_path=_resolve_repo_path(repo_root, config.resume_from_checkpoint),
            model=model,
            optimizer=optimizer,
            dependencies=dependencies,
            config=config,
            manifest_sha256=manifest_sha256,
            plan=plan,
        )
        generator.set_state(generator_state)
    if start_epoch >= config.max_epochs:
        raise ValueError("resume checkpoint has already completed the epoch ceiling")

    metadata_path = output_dir / "config.json"
    history_path = output_dir / "loss_history.jsonl"
    _write_run_metadata(
        path=metadata_path,
        config=config,
        plan=plan,
        device=device,
        tokenizer=tokenizer,
        train_examples=train_examples,
        validation_examples=validation_examples,
        test_examples=test_examples,
        manifest_sha256=manifest_sha256,
        start_epoch=start_epoch,
        start_step=start_step,
        completed_epoch=start_epoch,
        completed_step=start_step,
        best_validation_row=best_validation_row,
        stop_reason="in_progress",
        trainable_parameter_count=sum(parameter.numel() for parameter in trainable_parameters),
        package_versions=_package_versions(dependencies),
    )

    reporter = TrainingProgressReporter(
        total_steps=plan.planned_updates,
        progress_interval=config.progress_interval,
        start_step=start_step,
    )
    reporter.write_start(
        label=RUN_LABEL,
        device=str(device),
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    completed_epoch = start_epoch
    completed_step = start_step
    stop_reason = "epoch_ceiling_reached"
    for epoch in range(start_epoch + 1, config.max_epochs + 1):
        shuffled_indices = torch.randperm(len(train_examples), generator=generator).tolist()
        epoch_losses: list[float] = []
        epoch_gradient_norms: list[float] = []
        examples_per_update = config.batch_size * config.gradient_accumulation_steps
        for first_index in range(0, len(shuffled_indices), examples_per_update):
            selected_indices = shuffled_indices[
                first_index:first_index + examples_per_update
            ]
            selected_examples = [train_examples[index] for index in selected_indices]
            completed_step += 1
            learning_rate = learning_rate_for_qlora_step(
                config=config,
                plan=plan,
                step=completed_step,
            )
            _set_optimizer_learning_rate(optimizer, learning_rate)
            train_loss, gradient_norm = _train_adapter_update(
                model=model,
                optimizer=optimizer,
                examples=selected_examples,
                batch_size=config.batch_size,
                pad_token_id=pad_token_id,
                device=device,
                max_gradient_norm=config.max_gradient_norm,
            )
            epoch_losses.append(train_loss)
            epoch_gradient_norms.append(gradient_norm)
            if reporter.should_report(completed_step):
                reporter.write_progress(
                    step=completed_step,
                    train_loss=train_loss,
                    learning_rate=learning_rate,
                )

        _report(progress, f"evaluating full validation split after epoch {epoch}")
        validation_loss = evaluate_minerva_continuation_loss(
            model=model,
            examples=validation_examples,
            batch_size=config.batch_size,
            pad_token_id=pad_token_id,
            device=device,
        )
        train_loss = sum(epoch_losses) / len(epoch_losses)
        best_validation_updated = _is_meaningful_validation_improvement(
            candidate_loss=validation_loss,
            best_row=best_validation_row,
            minimum_improvement=config.min_validation_improvement,
        )
        if best_validation_updated:
            best_validation_row = {
                "epoch": epoch,
                "step": completed_step,
                "validation_loss": validation_loss,
            }
            non_improving_evaluations = 0
            _save_adapter_checkpoint(
                checkpoint_path=output_dir / "best_adapter.pt",
                model=model,
                dependencies=dependencies,
                config=config,
                manifest_sha256=manifest_sha256,
                epoch=epoch,
                step=completed_step,
                best_validation_row=best_validation_row,
                include_optimizer_state=False,
            )
        else:
            non_improving_evaluations += 1

        row = {
            "epoch": epoch,
            "step": completed_step,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": learning_rate,
            "mean_pre_clipping_gradient_norm": sum(epoch_gradient_norms)
            / len(epoch_gradient_norms),
            "non_improving_evaluations": non_improving_evaluations,
        }
        history.append(row)
        write_jsonl(history_path, history)
        _save_adapter_checkpoint(
            checkpoint_path=output_dir / "resume.pt",
            model=model,
            optimizer=optimizer,
            dependencies=dependencies,
            config=config,
            manifest_sha256=manifest_sha256,
            epoch=epoch,
            step=completed_step,
            best_validation_row=best_validation_row,
            history=history,
            non_improving_evaluations=non_improving_evaluations,
            generator_state=generator.get_state(),
            include_optimizer_state=True,
        )
        reporter.write_progress(
            step=completed_step,
            train_loss=train_loss,
            validation_loss=validation_loss,
            learning_rate=learning_rate,
            checkpoint_written=True,
            best_validation=best_validation_updated,
        )
        completed_epoch = epoch
        if non_improving_evaluations >= config.early_stopping_patience:
            stop_reason = "early_stopping_patience_exhausted"
            break

    _save_adapter_checkpoint(
        checkpoint_path=output_dir / "final_adapter.pt",
        model=model,
        dependencies=dependencies,
        config=config,
        manifest_sha256=manifest_sha256,
        epoch=completed_epoch,
        step=completed_step,
        best_validation_row=best_validation_row,
        include_optimizer_state=False,
    )
    _write_run_metadata(
        path=metadata_path,
        config=config,
        plan=plan,
        device=device,
        tokenizer=tokenizer,
        train_examples=train_examples,
        validation_examples=validation_examples,
        test_examples=test_examples,
        manifest_sha256=manifest_sha256,
        start_epoch=start_epoch,
        start_step=start_step,
        completed_epoch=completed_epoch,
        completed_step=completed_step,
        best_validation_row=best_validation_row,
        stop_reason=stop_reason,
        trainable_parameter_count=sum(parameter.numel() for parameter in trainable_parameters),
        package_versions=_package_versions(dependencies),
    )
    return {
        "config_path": metadata_path,
        "history_path": history_path,
        "best_checkpoint_path": output_dir / "best_adapter.pt",
        "resume_checkpoint_path": output_dir / "resume.pt",
        "final_checkpoint_path": output_dir / "final_adapter.pt",
        "history": history,
        "completed_epoch": completed_epoch,
        "completed_step": completed_step,
        "best_validation_row": best_validation_row,
        "stop_reason": stop_reason,
    }


def evaluate_minerva_continuation_loss(
    *,
    model: Any,
    examples: Sequence[TokenizedMinervaContinuationExample],
    batch_size: int,
    pad_token_id: int,
    device: torch.device | str,
) -> float:
    """Evaluate every validation sonnet once with continuation-token weighting."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if not examples:
        raise ValueError("examples must not be empty")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_targets = 0
    with torch.no_grad():
        for first_index in range(0, len(examples), batch_size):
            batch = collate_minerva_continuation_examples(
                examples=examples[first_index:first_index + batch_size],
                pad_token_id=pad_token_id,
                device=device,
            )
            loss = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                use_cache=False,
            ).loss
            total_loss += float(loss.item()) * batch.supervised_target_count
            total_targets += batch.supervised_target_count
    model.train(was_training)
    return total_loss / total_targets


def _train_adapter_update(
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    examples: Sequence[TokenizedMinervaContinuationExample],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    max_gradient_norm: float,
) -> tuple[float, float]:
    """Accumulate one variable-sized document group into one adapter update."""
    if not examples:
        raise ValueError("examples must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    microbatches = [
        collate_minerva_continuation_examples(
            examples=examples[index:index + batch_size],
            pad_token_id=pad_token_id,
            device=device,
        )
        for index in range(0, len(examples), batch_size)
    ]
    total_targets = sum(batch.supervised_target_count for batch in microbatches)
    total_loss = 0.0
    for batch in microbatches:
        loss = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            use_cache=False,
        ).loss
        target_weight = batch.supervised_target_count / total_targets
        (loss * target_weight).backward()
        total_loss += float(loss.item()) * batch.supervised_target_count

    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=max_gradient_norm,
        ).item()
    )
    optimizer.step()
    return total_loss / total_targets, gradient_norm


def _load_trainable_qlora_model(
    *,
    dependencies: dict[str, Any],
    config: MinervaQLoRAFineTuningConfig,
    cache_dir: Path,
) -> Any:
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
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = dependencies["prepare_model_for_kbit_training"](
        model,
        use_gradient_checkpointing=True,
    )
    return dependencies["get_peft_model"](
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


def _save_adapter_checkpoint(
    *,
    checkpoint_path: Path,
    model: Any,
    dependencies: dict[str, Any],
    config: MinervaQLoRAFineTuningConfig,
    manifest_sha256: str,
    epoch: int,
    step: int,
    best_validation_row: dict[str, Any] | None,
    optimizer: torch.optim.Optimizer | None = None,
    history: list[dict[str, Any]] | None = None,
    non_improving_evaluations: int = 0,
    generator_state: torch.Tensor | None = None,
    include_optimizer_state: bool,
) -> None:
    """Persist only adapter state, plus optimizer state only for resume files."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    adapter_state = {
        name: tensor.detach().cpu()
        for name, tensor in dependencies["get_peft_model_state_dict"](model).items()
    }
    torch.save(
        {
            "checkpoint_type": "minerva_qlora_adapter",
            "model_id": config.model_id,
            "revision": config.revision,
            "task_format_version": MINERVA_TASK_FORMAT_VERSION,
            "adapter_state_dict": adapter_state,
            "optimizer_state_dict": (
                optimizer.state_dict()
                if include_optimizer_state and optimizer is not None
                else None
            ),
            "recipe_config": _resume_recipe_config(config),
            "manifest_sha256": manifest_sha256,
            "epoch": epoch,
            "step": step,
            "history": history,
            "best_validation_row": best_validation_row,
            "non_improving_evaluations": non_improving_evaluations,
            "generator_state": generator_state,
        },
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)


def _load_resume_checkpoint(
    *,
    checkpoint_path: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    dependencies: dict[str, Any],
    config: MinervaQLoRAFineTuningConfig,
    manifest_sha256: str,
    plan: MinervaQLoRATrainingPlan,
) -> tuple[int, int, list[dict[str, Any]], dict[str, Any] | None, int, torch.Tensor]:
    """Restore adapter and optimizer state while rejecting incompatible resumes."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Minerva resume checkpoint must contain a dictionary")
    if checkpoint.get("checkpoint_type") != "minerva_qlora_adapter":
        raise ValueError("checkpoint is not a Minerva QLoRA adapter checkpoint")
    if checkpoint.get("model_id") != config.model_id:
        raise ValueError("resume checkpoint model_id does not match configuration")
    if checkpoint.get("revision") != config.revision:
        raise ValueError("resume checkpoint revision does not match configuration")
    if checkpoint.get("manifest_sha256") != manifest_sha256:
        raise ValueError("resume checkpoint manifest does not match current manifest")
    if checkpoint.get("recipe_config") != _resume_recipe_config(config):
        raise ValueError("resume checkpoint configuration does not match")
    if checkpoint.get("optimizer_state_dict") is None:
        raise ValueError("resume checkpoint does not contain optimizer state")
    epoch = _required_non_negative_int(checkpoint, "epoch")
    step = _required_non_negative_int(checkpoint, "step")
    if step > plan.planned_updates:
        raise ValueError("resume checkpoint step exceeds planned updates")
    dependencies["set_peft_model_state_dict"](model, checkpoint["adapter_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    history = checkpoint.get("history", [])
    if not isinstance(history, list):
        raise ValueError("resume checkpoint history must be a list")
    best_validation_row = checkpoint.get("best_validation_row")
    if best_validation_row is not None and not isinstance(best_validation_row, dict):
        raise ValueError("resume checkpoint best_validation_row must be a dictionary")
    generator_state = checkpoint.get("generator_state")
    if not isinstance(generator_state, torch.Tensor) or generator_state.dtype != torch.uint8:
        raise ValueError("resume checkpoint generator_state must be a uint8 tensor")
    return (
        epoch,
        step,
        history,
        best_validation_row,
        _required_non_negative_int(checkpoint, "non_improving_evaluations"),
        generator_state,
    )


def _write_run_metadata(
    *,
    path: Path,
    config: MinervaQLoRAFineTuningConfig,
    plan: MinervaQLoRATrainingPlan,
    device: torch.device,
    tokenizer: Any,
    train_examples: Sequence[TokenizedMinervaContinuationExample],
    validation_examples: Sequence[TokenizedMinervaContinuationExample],
    test_examples: Sequence[TokenizedMinervaContinuationExample],
    manifest_sha256: str,
    start_epoch: int,
    start_step: int,
    completed_epoch: int,
    completed_step: int,
    best_validation_row: dict[str, Any] | None,
    stop_reason: str,
    trainable_parameter_count: int,
    package_versions: dict[str, str],
) -> None:
    """Write reproducibility metadata without copying the frozen base model."""
    all_examples = [*train_examples, *validation_examples, *test_examples]
    write_json(
        path,
        {
            "run_label": RUN_LABEL,
            "task_format_version": MINERVA_TASK_FORMAT_VERSION,
            "config": asdict(config),
            "resolved_device": str(device),
            "tokenizer": {
                "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            },
            "manifest_sha256": manifest_sha256,
            "dataset_examples": {
                "train": len(train_examples),
                "validation": len(validation_examples),
                "test": len(test_examples),
                "maximum_sequence_tokens": max(len(example.input_ids) for example in all_examples),
            },
            "training_plan": asdict(plan),
            "trainable_parameter_count": trainable_parameter_count,
            "package_versions": package_versions,
            "start_epoch": start_epoch,
            "start_step": start_step,
            "completed_epoch": completed_epoch,
            "completed_step": completed_step,
            "best_validation_row": best_validation_row,
            "stop_reason": stop_reason,
        },
    )


def _is_meaningful_validation_improvement(
    *,
    candidate_loss: float,
    best_row: dict[str, Any] | None,
    minimum_improvement: float,
) -> bool:
    if best_row is None:
        return True
    best_loss = best_row.get("validation_loss")
    if not isinstance(best_loss, (int, float)):
        raise ValueError("best validation row must contain a numeric validation_loss")
    return candidate_loss < best_loss - minimum_improvement


def _load_qlora_dependencies() -> dict[str, Any]:
    """Import optional pretrained-model dependencies only for the GPU run."""
    try:
        import accelerate
        import bitsandbytes
        import peft
        import transformers
        from peft import (
            LoraConfig,
            get_peft_model,
            get_peft_model_state_dict,
            prepare_model_for_kbit_training,
            set_peft_model_state_dict,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            "Minerva QLoRA dependencies are missing; install "
            "requirements/minerva_qlora.txt into .venv first"
        ) from error
    return {
        "accelerate": accelerate,
        "bitsandbytes": bitsandbytes,
        "peft": peft,
        "transformers": transformers,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "get_peft_model_state_dict": get_peft_model_state_dict,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "set_peft_model_state_dict": set_peft_model_state_dict,
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


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    if not isinstance(encoded, dict) or not isinstance(encoded.get("input_ids"), list):
        raise ValueError("Minerva tokenizer must return a list of input_ids")
    token_ids = encoded["input_ids"]
    if not all(isinstance(token_id, int) and token_id >= 0 for token_id in token_ids):
        raise ValueError("Minerva tokenizer input_ids must be non-negative integers")
    return token_ids


def _set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def _resume_recipe_config(config: MinervaQLoRAFineTuningConfig) -> dict[str, Any]:
    """Return the locked recipe identity, excluding the local resume path."""
    recipe_config = asdict(config)
    recipe_config.pop("resume_from_checkpoint")
    return recipe_config


def _resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path


def _required_non_negative_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"checkpoint {key} must be a non-negative integer")
    return value


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(f"minerva-train | {message}")
