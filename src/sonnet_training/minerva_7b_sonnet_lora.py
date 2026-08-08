"""V6 sonnet specialization of the selected historical Minerva 7B adapter."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from sonnet_corpus.task_format import (
    IGNORE_INDEX,
    SonnetContinuationExample,
    build_sonnet_continuation_examples,
)
from sonnet_training.minerva_7b_historical_lora import (
    build_instruction_preservation_batches,
)
from sonnet_training.minerva_7b_qlora import (
    MINERVA_7B_INSTRUCT_MODEL_ID,
    MINERVA_7B_INSTRUCT_REVISION,
    MINERVA_7B_QLORA_TARGET_MODULES,
    minerva_7b_package_versions,
)
from sonnet_training.minerva_7b_staged_data import load_staged_tensor
from sonnet_training.progress import TrainingProgressReporter


SONNET_RUN_VERSION = "minerva_7b_v6_sonnet_fp16_lora_v1"
SONNET_TASK_FORMAT_VERSION = "minerva_chat_complete_sonnet_v1"
V6_MANIFEST_SHA256 = "994c4c374f42ba26f1c352d7ad7c3adec7ec4671507770bd7c485cb6f977a4fa"
SELECTED_STAGE_A_SHA256 = "acfad4d442ac8ea7349dcb1bd379c9b41859027ab45daac54c6b6aa35e0bbc63"


@dataclass(frozen=True)
class Minerva7BSonnetLoRAConfig:
    """Freeze the approved Stage B sonnet-specialization recipe."""

    model_id: str = MINERVA_7B_INSTRUCT_MODEL_ID
    revision: str = MINERVA_7B_INSTRUCT_REVISION
    cache_dir: str = "data/local/minerva_qlora/huggingface"
    manifest_path: str = "data/metadata/sonnets_expanded_v6_manifest.csv"
    dataset: str = "expanded_with_petrarch"
    selected_stage_a_path: str = (
        "runs/minerva_7b_historical_fp16_lora_001/checkpoints/"
        "adapter_step_004000.pt"
    )
    selected_stage_a_sha256: str = SELECTED_STAGE_A_SHA256
    encoded_dir: str = "data/local/minerva_7b_staged/encoded"
    preservation_prompts_path: str = "configs/minerva_7b_preservation_prompts.json"
    context_length: int = 512
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_epochs: int = 10
    learning_rate: float = 1e-5
    min_learning_rate: float = 1e-6
    warmup_fraction: float = 0.05
    weight_decay: float = 0.01
    max_gradient_norm: float = 1.0
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = MINERVA_7B_QLORA_TARGET_MODULES
    early_stopping_patience: int = 3
    min_validation_improvement: float = 0.01
    maximum_modern_loss_ratio: float = 1.05
    maximum_instruction_loss_ratio: float = 1.10
    progress_interval: int = 25
    seed: int = 1337
    device: str = "cuda:0"


@dataclass(frozen=True)
class TokenizedSonnetChatExample:
    poem_id: str
    split: str
    opening_line: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    response_start: int


@dataclass(frozen=True)
class SonnetBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    target_count: int


@dataclass(frozen=True)
class SonnetTrainingPlan:
    train_example_count: int
    validation_example_count: int
    updates_per_epoch: int
    planned_updates: int
    warmup_steps: int


def build_sonnet_user_message(opening_line: str) -> str:
    if not opening_line.strip() or "\n" in opening_line or "\r" in opening_line:
        raise ValueError("opening_line must contain exactly one non-empty line")
    return (
        "Componi un sonetto in italiano classico di esattamente quattordici "
        "versi. Usa come primo verso esattamente quello indicato, mantieni un "
        "tema coerente e una sintassi grammaticale, ed evita ripetizioni. "
        "Restituisci soltanto il sonetto, senza titolo, spiegazioni o commenti.\n\n"
        f"Primo verso: {opening_line}"
    )


def tokenize_sonnet_chat_example(
    *,
    example: SonnetContinuationExample,
    tokenizer: Any,
    context_length: int,
) -> TokenizedSonnetChatExample:
    """Render one chat example and supervise only the assistant's full sonnet."""
    if context_length <= 0:
        raise ValueError("context_length must be greater than zero")
    user_message = build_sonnet_user_message(example.opening_line)
    response = f"{example.opening_line}\n{example.continuation_text}"
    prompt_messages = [{"role": "user", "content": user_message}]
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": response},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    if not _valid_token_ids(prompt_ids) or not _valid_token_ids(full_ids):
        raise ValueError("Minerva chat template must return non-negative token ID lists")
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("assistant response must preserve the rendered prompt prefix")
    if len(full_ids) > context_length:
        raise ValueError(
            f"V6 example exceeds context length: {example.poem_id} "
            f"uses {len(full_ids)} tokens"
        )
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("chat example contains no supervised response tokens")
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return TokenizedSonnetChatExample(
        poem_id=example.poem_id,
        split=example.split,
        opening_line=example.opening_line,
        input_ids=tuple(full_ids),
        labels=tuple(labels),
        response_start=len(prompt_ids),
    )


def load_sonnet_chat_splits(
    *,
    repo_root: Path,
    manifest_path: Path,
    dataset: str,
    tokenizer: Any,
    context_length: int,
) -> tuple[list[TokenizedSonnetChatExample], list[TokenizedSonnetChatExample]]:
    """Load only V6 train and validation; final-test poems stay unavailable."""
    splits = []
    for split in ("train", "validation"):
        examples = build_sonnet_continuation_examples(
            manifest_path=manifest_path,
            repo_root=repo_root,
            dataset=dataset,
            split=split,
        )
        splits.append([
            tokenize_sonnet_chat_example(
                example=example,
                tokenizer=tokenizer,
                context_length=context_length,
            )
            for example in examples
        ])
    return splits[0], splits[1]


def collate_sonnet_examples(
    *,
    examples: Sequence[TokenizedSonnetChatExample],
    pad_token_id: int,
    device: torch.device | str | None = None,
) -> SonnetBatch:
    if not examples:
        raise ValueError("examples must not be empty")
    maximum_length = max(len(example.input_ids) for example in examples)
    input_ids = torch.full(
        (len(examples), maximum_length), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    for index, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[index, :length] = torch.tensor(example.input_ids)
        attention_mask[index, :length] = 1
        labels[index, :length] = torch.tensor(example.labels)
    target_count = int((labels != IGNORE_INDEX).sum().item())
    if target_count <= 0:
        raise ValueError("batch contains no supervised response tokens")
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
    return SonnetBatch(input_ids, attention_mask, labels, target_count)


def build_sonnet_training_plan(
    *, config: Minerva7BSonnetLoRAConfig, train_count: int, validation_count: int
) -> SonnetTrainingPlan:
    validate_sonnet_config(config)
    if train_count <= 0 or validation_count <= 0:
        raise ValueError("train and validation splits must be non-empty")
    examples_per_update = config.batch_size * config.gradient_accumulation_steps
    updates_per_epoch = math.ceil(train_count / examples_per_update)
    planned_updates = updates_per_epoch * config.max_epochs
    return SonnetTrainingPlan(
        train_example_count=train_count,
        validation_example_count=validation_count,
        updates_per_epoch=updates_per_epoch,
        planned_updates=planned_updates,
        warmup_steps=max(1, math.ceil(planned_updates * config.warmup_fraction)),
    )


def sonnet_learning_rate(
    *, config: Minerva7BSonnetLoRAConfig, plan: SonnetTrainingPlan, step: int
) -> float:
    if step <= 0 or step > plan.planned_updates:
        raise ValueError("step must be within the planned update range")
    if step <= plan.warmup_steps:
        return config.learning_rate * step / plan.warmup_steps
    progress = (step - plan.warmup_steps) / (plan.planned_updates - plan.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + cosine * (
        config.learning_rate - config.min_learning_rate
    )


def select_top_qualifying_candidates(
    history: Sequence[dict[str, Any]], *, limit: int = 3
) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    qualifying = [row for row in history if row.get("preservation_gate_passed") is True]
    return sorted(qualifying, key=lambda row: float(row["validation_loss"]))[:limit]


def validate_sonnet_config(config: Minerva7BSonnetLoRAConfig) -> None:
    if config != Minerva7BSonnetLoRAConfig():
        raise ValueError("Minerva 7B V6 sonnet LoRA recipe is locked")


def train_minerva_7b_sonnet_lora(
    *,
    repo_root: Path,
    output_dir: Path,
    config: Minerva7BSonnetLoRAConfig = Minerva7BSonnetLoRAConfig(),
    resume_from_checkpoint: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run or resume Stage B from the exact selected Stage A adapter."""
    validate_sonnet_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Minerva 7B sonnet specialization requires CUDA")
    device = torch.device(config.device)
    dependencies = _load_dependencies()
    torch.manual_seed(config.seed)

    manifest_path = _resolve(repo_root, config.manifest_path)
    if _sha256(manifest_path) != V6_MANIFEST_SHA256:
        raise ValueError("V6 manifest does not match the frozen audited version")
    parent_path = _resolve(repo_root, config.selected_stage_a_path)
    parent = _load_selected_parent(parent_path, config=config)
    original_baseline = dict(parent["baseline_metrics"])
    encoded_dir = _resolve(repo_root, config.encoded_dir)
    modern_validation = load_staged_tensor(
        encoded_dir / "modern_preservation_validation.pt", dimensions=2
    )

    _report(progress, "loading pinned Minerva tokenizer")
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        config.model_id,
        revision=config.revision,
        cache_dir=_resolve(repo_root, config.cache_dir),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id
    if not isinstance(pad_token_id, int) or pad_token_id < 0:
        raise ValueError("Minerva tokenizer must define a pad token ID")
    _report(progress, "tokenizing V6 train and validation sonnets")
    train_examples, validation_examples = load_sonnet_chat_splits(
        repo_root=repo_root,
        manifest_path=manifest_path,
        dataset=config.dataset,
        tokenizer=tokenizer,
        context_length=config.context_length,
    )
    plan = build_sonnet_training_plan(
        config=config,
        train_count=len(train_examples),
        validation_count=len(validation_examples),
    )
    instruction_batches = build_instruction_preservation_batches(
        tokenizer=tokenizer,
        path=_resolve(repo_root, config.preservation_prompts_path),
        context_length=config.context_length,
    )

    _report(progress, "loading unquantized Minerva 7B weights and Stage A adapter")
    model = dependencies["AutoModelForCausalLM"].from_pretrained(
        config.model_id,
        revision=config.revision,
        cache_dir=_resolve(repo_root, config.cache_dir),
        dtype=torch.float16,
        device_map={"": device.index or 0},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
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
    dependencies["set_peft_model_state_dict"](model, parent["adapter_state_dict"])
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        foreach=False,
    )

    output_dir = _resolve(repo_root, str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    history: list[dict[str, Any]] = []
    completed_epoch = 0
    completed_step = 0
    non_improving = 0
    patience_best_loss = math.inf
    generator = torch.Generator().manual_seed(config.seed)
    if resume_from_checkpoint is not None:
        (
            completed_epoch,
            completed_step,
            history,
            non_improving,
            patience_best_loss,
            generator_state,
        ) = _restore_resume(
            path=_resolve(repo_root, str(resume_from_checkpoint)),
            model=model,
            optimizer=optimizer,
            config=config,
            plan=plan,
            dependencies=dependencies,
        )
        generator.set_state(generator_state)

    _report(progress, "measuring Stage B starting losses")
    starting_metrics = {
        "sonnet_validation_loss": evaluate_sonnet_loss(
            model=model,
            examples=validation_examples,
            pad_token_id=pad_token_id,
            device=device,
        ),
        **evaluate_preservation(
            model=model,
            modern_validation=modern_validation,
            instruction_batches=instruction_batches,
            device=device,
        ),
    }
    _write_json(output_dir / "baseline_metrics.json", starting_metrics)
    _write_json(
        output_dir / "config.json",
        {
            "run_version": SONNET_RUN_VERSION,
            "task_format_version": SONNET_TASK_FORMAT_VERSION,
            "config": asdict(config),
            "plan": asdict(plan),
            "manifest_sha256": V6_MANIFEST_SHA256,
            "selected_stage_a_sha256": config.selected_stage_a_sha256,
            "selected_stage_a_row": parent["row"],
            "original_parent_baseline_metrics": original_baseline,
            "trainable_parameter_count": sum(p.numel() for p in trainable_parameters),
            "maximum_sequence_tokens": max(
                len(example.input_ids)
                for example in [*train_examples, *validation_examples]
            ),
            "package_versions": minerva_7b_package_versions(dependencies),
        },
    )

    reporter = TrainingProgressReporter(
        total_steps=plan.planned_updates,
        progress_interval=config.progress_interval,
        start_step=completed_step,
    )
    reporter.write_start(
        label="minerva_7b_v6_sonnet_lora",
        device=str(device),
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    stop_reason = "epoch_ceiling_reached"
    for epoch in range(completed_epoch + 1, config.max_epochs + 1):
        order = torch.randperm(len(train_examples), generator=generator).tolist()
        epoch_losses = []
        for start in range(0, len(order), config.gradient_accumulation_steps):
            completed_step += 1
            selected = [
                train_examples[index]
                for index in order[start:start + config.gradient_accumulation_steps]
            ]
            learning_rate = sonnet_learning_rate(
                config=config, plan=plan, step=completed_step
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            train_loss = _train_update(
                model=model,
                optimizer=optimizer,
                examples=selected,
                pad_token_id=pad_token_id,
                device=device,
                max_gradient_norm=config.max_gradient_norm,
            )
            epoch_losses.append(train_loss)
            if reporter.should_report(completed_step):
                reporter.write_progress(
                    step=completed_step,
                    train_loss=train_loss,
                    learning_rate=learning_rate,
                )

        _report(progress, f"evaluating V6 and preservation gates after epoch {epoch}")
        validation_loss = evaluate_sonnet_loss(
            model=model,
            examples=validation_examples,
            pad_token_id=pad_token_id,
            device=device,
        )
        preservation = evaluate_preservation(
            model=model,
            modern_validation=modern_validation,
            instruction_batches=instruction_batches,
            device=device,
        )
        gate_passed = (
            preservation["modern_validation_loss"]
            <= original_baseline["modern_validation_loss"]
            * config.maximum_modern_loss_ratio
            and preservation["instruction_validation_loss"]
            <= original_baseline["instruction_validation_loss"]
            * config.maximum_instruction_loss_ratio
        )
        meaningful = (
            gate_passed
            and validation_loss
            <= patience_best_loss - config.min_validation_improvement
        )
        if meaningful:
            patience_best_loss = validation_loss
            non_improving = 0
        else:
            non_improving += 1
        row = {
            "epoch": epoch,
            "step": completed_step,
            "train_loss": sum(epoch_losses) / len(epoch_losses),
            "validation_loss": validation_loss,
            "learning_rate": learning_rate,
            **preservation,
            "preservation_gate_passed": gate_passed,
            "non_improving_evaluations": non_improving,
        }
        history.append(row)
        _write_jsonl(output_dir / "loss_history.jsonl", history)
        candidate_path = output_dir / "checkpoints" / f"adapter_epoch_{epoch:02d}.pt"
        _save_adapter(
            path=candidate_path,
            model=model,
            dependencies=dependencies,
            config=config,
            plan=plan,
            row=row,
        )
        _save_resume(
            path=output_dir / "resume.pt",
            model=model,
            optimizer=optimizer,
            dependencies=dependencies,
            config=config,
            plan=plan,
            completed_epoch=epoch,
            completed_step=completed_step,
            history=history,
            non_improving=non_improving,
            patience_best_loss=patience_best_loss,
            generator_state=generator.get_state(),
        )
        reporter.write_progress(
            step=completed_step,
            train_loss=row["train_loss"],
            validation_loss=validation_loss,
            learning_rate=learning_rate,
            checkpoint_written=True,
            best_validation=meaningful,
        )
        print(
            "preservation | modern_loss={modern:.4f} | "
            "instruction_loss={instruction:.4f} | gate={gate}".format(
                modern=preservation["modern_validation_loss"],
                instruction=preservation["instruction_validation_loss"],
                gate="pass" if gate_passed else "fail",
            ),
            flush=True,
        )
        completed_epoch = epoch
        if non_improving >= config.early_stopping_patience:
            stop_reason = "early_stopping"
            break

    top_candidates = select_top_qualifying_candidates(history)
    result = {
        "run_version": SONNET_RUN_VERSION,
        "completed_epoch": completed_epoch,
        "completed_step": completed_step,
        "planned_updates": plan.planned_updates,
        "stop_reason": stop_reason,
        "baseline_metrics": starting_metrics,
        "top_qualifying_candidates": top_candidates,
        "candidate_checkpoint_paths": [
            str(output_dir / "checkpoints" / f"adapter_epoch_{row['epoch']:02d}.pt")
            for row in top_candidates
        ],
        "selection_frozen": False,
        "final_test_used": False,
    }
    _write_json(output_dir / "result.json", result)
    return result


@torch.no_grad()
def evaluate_sonnet_loss(
    *,
    model: Any,
    examples: Sequence[TokenizedSonnetChatExample],
    pad_token_id: int,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_targets = 0
    for example in examples:
        batch = collate_sonnet_examples(
            examples=[example], pad_token_id=pad_token_id, device=device
        )
        loss = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            use_cache=False,
        ).loss
        total_loss += float(loss.item()) * batch.target_count
        total_targets += batch.target_count
    return total_loss / total_targets


@torch.no_grad()
def evaluate_preservation(
    *,
    model: Any,
    modern_validation: torch.Tensor,
    instruction_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    modern_total = 0.0
    for row in modern_validation:
        input_ids = row[:-1].to(device=device, dtype=torch.long).unsqueeze(0)
        modern_total += float(model(input_ids=input_ids, labels=input_ids).loss.item())
    instruction_total = 0.0
    instruction_targets = 0
    for input_ids, labels in instruction_batches:
        target_count = int((labels != IGNORE_INDEX).sum().item())
        loss = model(
            input_ids=input_ids.to(device), labels=labels.to(device)
        ).loss
        instruction_total += float(loss.item()) * target_count
        instruction_targets += target_count
    return {
        "modern_validation_loss": modern_total / len(modern_validation),
        "instruction_validation_loss": instruction_total / instruction_targets,
    }


def _train_update(
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    examples: Sequence[TokenizedSonnetChatExample],
    pad_token_id: int,
    device: torch.device,
    max_gradient_norm: float,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batches = [
        collate_sonnet_examples(
            examples=[example], pad_token_id=pad_token_id, device=device
        )
        for example in examples
    ]
    total_targets = sum(batch.target_count for batch in batches)
    total_loss = 0.0
    for batch in batches:
        loss = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            use_cache=False,
        ).loss
        weight = batch.target_count / total_targets
        (loss * weight).backward()
        total_loss += float(loss.item()) * batch.target_count
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        max_gradient_norm,
    )
    optimizer.step()
    return total_loss / total_targets


def _load_selected_parent(
    path: Path, *, config: Minerva7BSonnetLoRAConfig
) -> dict[str, Any]:
    if _sha256(path) != config.selected_stage_a_sha256:
        raise ValueError("selected Stage A checkpoint hash does not match")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "checkpoint_type": "minerva_7b_historical_lora_adapter",
        "model_id": config.model_id,
        "revision": config.revision,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"selected Stage A checkpoint mismatch: {key}")
    row = checkpoint.get("row")
    if not isinstance(row, Mapping) or row.get("preservation_gate_passed") is not True:
        raise ValueError("selected Stage A checkpoint did not pass preservation")
    if int(row.get("step", -1)) != 4000:
        raise ValueError("selected Stage A checkpoint is not the pinned step 4000")
    return checkpoint


def _save_adapter(
    *,
    path: Path,
    model: Any,
    dependencies: dict[str, Any],
    config: Minerva7BSonnetLoRAConfig,
    plan: SonnetTrainingPlan,
    row: dict[str, Any],
) -> None:
    _atomic_save(path, {
        "checkpoint_type": "minerva_7b_v6_sonnet_lora_adapter",
        "run_version": SONNET_RUN_VERSION,
        "model_id": config.model_id,
        "revision": config.revision,
        "task_format_version": SONNET_TASK_FORMAT_VERSION,
        "selected_stage_a_sha256": config.selected_stage_a_sha256,
        "manifest_sha256": V6_MANIFEST_SHA256,
        "recipe_config": asdict(config),
        "plan": asdict(plan),
        "row": row,
        "adapter_state_dict": _cpu_state(
            dependencies["get_peft_model_state_dict"](model)
        ),
    })


def _save_resume(
    *,
    path: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    dependencies: dict[str, Any],
    config: Minerva7BSonnetLoRAConfig,
    plan: SonnetTrainingPlan,
    completed_epoch: int,
    completed_step: int,
    history: list[dict[str, Any]],
    non_improving: int,
    patience_best_loss: float,
    generator_state: torch.Tensor,
) -> None:
    _atomic_save(path, {
        "checkpoint_type": "minerva_7b_v6_sonnet_lora_resume",
        "run_version": SONNET_RUN_VERSION,
        "selected_stage_a_sha256": config.selected_stage_a_sha256,
        "manifest_sha256": V6_MANIFEST_SHA256,
        "recipe_config": asdict(config),
        "plan": asdict(plan),
        "completed_epoch": completed_epoch,
        "completed_step": completed_step,
        "history": history,
        "non_improving": non_improving,
        "patience_best_loss": patience_best_loss,
        "generator_state": generator_state,
        "adapter_state_dict": _cpu_state(
            dependencies["get_peft_model_state_dict"](model)
        ),
        "optimizer_state_dict": _to_cpu(optimizer.state_dict()),
    })


def _restore_resume(
    *,
    path: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    config: Minerva7BSonnetLoRAConfig,
    plan: SonnetTrainingPlan,
    dependencies: dict[str, Any],
) -> tuple[int, int, list[dict[str, Any]], int, float, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "checkpoint_type": "minerva_7b_v6_sonnet_lora_resume",
        "run_version": SONNET_RUN_VERSION,
        "selected_stage_a_sha256": config.selected_stage_a_sha256,
        "manifest_sha256": V6_MANIFEST_SHA256,
        "recipe_config": asdict(config),
        "plan": asdict(plan),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Stage B resume checkpoint mismatch: {key}")
    dependencies["set_peft_model_state_dict"](
        model, checkpoint["adapter_state_dict"]
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (
        int(checkpoint["completed_epoch"]),
        int(checkpoint["completed_step"]),
        list(checkpoint["history"]),
        int(checkpoint["non_improving"]),
        float(checkpoint["patience_best_loss"]),
        checkpoint["generator_state"],
    )


def _load_dependencies() -> dict[str, Any]:
    try:
        import accelerate
        import bitsandbytes
        import peft
        import transformers
        from peft import (
            LoraConfig,
            get_peft_model,
            get_peft_model_state_dict,
            set_peft_model_state_dict,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Minerva dependencies are missing; use the project .venv") from error
    return {
        "accelerate": accelerate,
        "bitsandbytes": bitsandbytes,
        "peft": peft,
        "transformers": transformers,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "get_peft_model_state_dict": get_peft_model_state_dict,
        "set_peft_model_state_dict": set_peft_model_state_dict,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def _valid_token_ids(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(token_id, int) and token_id >= 0 for token_id in value)
    )


def _cpu_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in state.items()}


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
