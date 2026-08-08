"""Resumable historical-Italian FP16 LoRA adaptation of Minerva 7B."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from sonnet_training.minerva_7b_qlora import (
    MINERVA_7B_INSTRUCT_MODEL_ID,
    MINERVA_7B_INSTRUCT_REVISION,
    MINERVA_7B_QLORA_TARGET_MODULES,
    minerva_7b_package_versions,
)
from sonnet_training.minerva_7b_staged_data import load_staged_tensor
from sonnet_training.progress import TrainingProgressReporter


HISTORICAL_RUN_VERSION = "minerva_7b_historical_fp16_lora_v1"
IGNORE_INDEX = -100


@dataclass(frozen=True)
class Minerva7BHistoricalLoRAConfig:
    """Freeze the approved historical-adaptation recipe."""

    model_id: str = MINERVA_7B_INSTRUCT_MODEL_ID
    revision: str = MINERVA_7B_INSTRUCT_REVISION
    cache_dir: str = "data/local/minerva_qlora/huggingface"
    encoded_dir: str = "data/local/minerva_7b_staged/encoded"
    preservation_prompts_path: str = "configs/minerva_7b_preservation_prompts.json"
    context_length: int = 512
    historical_microbatches_per_update: int = 7
    replay_microbatches_per_update: int = 1
    max_epochs: int = 2
    learning_rate: float = 2e-5
    min_learning_rate: float = 2e-6
    warmup_fraction: float = 0.03
    weight_decay: float = 0.01
    max_gradient_norm: float = 1.0
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = MINERVA_7B_QLORA_TARGET_MODULES
    validation_interval: int = 500
    resume_interval: int = 100
    progress_interval: int = 25
    early_stopping_patience: int = 3
    min_historical_improvement: float = 0.005
    maximum_modern_loss_ratio: float = 1.05
    maximum_instruction_loss_ratio: float = 1.10
    seed: int = 1337
    device: str = "cuda:0"


@dataclass(frozen=True)
class HistoricalTrainingPlan:
    historical_window_count: int
    replay_window_count: int
    updates_per_epoch: int
    planned_updates: int
    warmup_steps: int
    nominal_tokens_per_update: int


def build_historical_training_plan(
    *,
    config: Minerva7BHistoricalLoRAConfig,
    historical_token_count: int,
    replay_token_count: int,
) -> HistoricalTrainingPlan:
    validate_historical_config(config)
    historical_windows = max(0, (historical_token_count - 1) // config.context_length)
    replay_windows = max(0, (replay_token_count - 1) // config.context_length)
    if historical_windows == 0 or replay_windows == 0:
        raise ValueError("historical and replay streams must each contain one window")
    updates_per_epoch = math.ceil(
        historical_windows / config.historical_microbatches_per_update
    )
    planned_updates = updates_per_epoch * config.max_epochs
    warmup_steps = max(1, math.ceil(planned_updates * config.warmup_fraction))
    return HistoricalTrainingPlan(
        historical_window_count=historical_windows,
        replay_window_count=replay_windows,
        updates_per_epoch=updates_per_epoch,
        planned_updates=planned_updates,
        warmup_steps=warmup_steps,
        nominal_tokens_per_update=(
            config.context_length
            * (
                config.historical_microbatches_per_update
                + config.replay_microbatches_per_update
            )
        ),
    )


def historical_learning_rate(
    *, config: Minerva7BHistoricalLoRAConfig, plan: HistoricalTrainingPlan, step: int
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


def preservation_gate(
    *,
    modern_loss: float,
    instruction_loss: float,
    baseline_modern_loss: float,
    baseline_instruction_loss: float,
    config: Minerva7BHistoricalLoRAConfig,
) -> bool:
    return (
        modern_loss <= baseline_modern_loss * config.maximum_modern_loss_ratio
        and instruction_loss
        <= baseline_instruction_loss * config.maximum_instruction_loss_ratio
    )


def validate_historical_config(config: Minerva7BHistoricalLoRAConfig) -> None:
    expected = Minerva7BHistoricalLoRAConfig()
    if config != expected:
        raise ValueError("Minerva 7B historical LoRA recipe is locked")


def train_minerva_7b_historical_lora(
    *,
    repo_root: Path,
    output_dir: Path,
    config: Minerva7BHistoricalLoRAConfig = Minerva7BHistoricalLoRAConfig(),
    resume_from_checkpoint: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run or resume the approved historical adaptation stage."""
    validate_historical_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Minerva 7B historical adaptation requires CUDA")
    device = torch.device(config.device)
    if device.type != "cuda":
        raise ValueError("Minerva 7B historical adaptation requires a CUDA device")
    dependencies = _load_dependencies()
    torch.manual_seed(config.seed)

    encoded_dir = _resolve(repo_root, config.encoded_dir)
    data_report = _load_and_verify_data_report(encoded_dir)
    historical_train = load_staged_tensor(
        encoded_dir / "historical_train.pt", dimensions=1
    )
    historical_validation = load_staged_tensor(
        encoded_dir / "historical_validation.pt", dimensions=1
    )
    replay_train = load_staged_tensor(
        encoded_dir / "modern_replay_train.pt", dimensions=1
    )
    modern_validation = load_staged_tensor(
        encoded_dir / "modern_preservation_validation.pt", dimensions=2
    )
    plan = build_historical_training_plan(
        config=config,
        historical_token_count=historical_train.numel(),
        replay_token_count=replay_train.numel(),
    )

    output_dir = _resolve(repo_root, str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    _write_json(output_dir / "config.json", {
        "run_version": HISTORICAL_RUN_VERSION,
        "config": asdict(config),
        "plan": asdict(plan),
        "data_report_sha256": _sha256(encoded_dir / "report.json"),
        "data_summary": {
            key: data_report[key]
            for key in (
                "historical_train_tokens",
                "historical_validation_tokens",
                "modern_replay_train_tokens",
                "modern_preservation_tokens",
            )
        },
    })

    _report(progress, "loading pinned Minerva tokenizer")
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        config.model_id,
        revision=config.revision,
        cache_dir=_resolve(repo_root, config.cache_dir),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    instruction_batches = build_instruction_preservation_batches(
        tokenizer=tokenizer,
        path=_resolve(repo_root, config.preservation_prompts_path),
        context_length=config.context_length,
    )

    _report(progress, "loading unquantized Minerva 7B Instruct weights in FP16")
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
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        foreach=False,
    )

    completed_steps = 0
    history: list[dict[str, Any]] = []
    baseline_metrics: dict[str, float] | None = None
    best_row: dict[str, Any] | None = None
    non_improving_evaluations = 0
    if resume_from_checkpoint is not None:
        _report(progress, f"restoring resume checkpoint: {resume_from_checkpoint}")
        (
            completed_steps,
            history,
            baseline_metrics,
            best_row,
            non_improving_evaluations,
        ) = _restore_resume_checkpoint(
            path=_resolve(repo_root, str(resume_from_checkpoint)),
            model=model,
            optimizer=optimizer,
            config=config,
            plan=plan,
            data_report_sha256=_sha256(encoded_dir / "report.json"),
            set_peft_model_state_dict=dependencies["set_peft_model_state_dict"],
        )

    if baseline_metrics is None:
        _report(progress, "measuring stage-zero historical and preservation losses")
        baseline_metrics = evaluate_historical_and_preservation(
            model=model,
            historical_validation=historical_validation,
            modern_validation=modern_validation,
            instruction_batches=instruction_batches,
            context_length=config.context_length,
            device=device,
        )
        _write_json(output_dir / "baseline_metrics.json", baseline_metrics)

    reporter = TrainingProgressReporter(
        total_steps=plan.planned_updates,
        progress_interval=config.progress_interval,
        start_step=completed_steps,
    )
    reporter.write_start(
        label="minerva_7b_historical_lora",
        device=str(device),
        tokens_per_step=plan.nominal_tokens_per_update,
        gradient_accumulation_steps=(
            config.historical_microbatches_per_update
            + config.replay_microbatches_per_update
        ),
    )
    stop_reason = "completed_epoch_ceiling"
    last_train_loss = math.nan
    started_at = time.monotonic()
    for zero_based_step in range(completed_steps, plan.planned_updates):
        completed_step = zero_based_step + 1
        epoch_index = zero_based_step // plan.updates_per_epoch
        update_in_epoch = zero_based_step % plan.updates_per_epoch
        historical_order = _window_order(
            plan.historical_window_count,
            seed=config.seed + epoch_index,
        )
        start = update_in_epoch * config.historical_microbatches_per_update
        historical_indices = historical_order[
            start:start + config.historical_microbatches_per_update
        ]
        replay_order = _window_order(
            plan.replay_window_count,
            seed=config.seed + 10_000 + epoch_index,
        )
        replay_index = replay_order[update_in_epoch % len(replay_order)]
        microbatches = [
            (historical_train, index) for index in historical_indices
        ] + [(replay_train, replay_index)]

        learning_rate = historical_learning_rate(
            config=config,
            plan=plan,
            step=completed_step,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        optimizer.zero_grad(set_to_none=True)
        microbatch_losses: list[float] = []
        for stream, window_index in microbatches:
            input_ids, labels = _stream_window(
                stream,
                window_index=window_index,
                context_length=config.context_length,
                device=device,
            )
            loss = model(input_ids=input_ids, labels=labels).loss
            (loss / len(microbatches)).backward()
            microbatch_losses.append(float(loss.detach().item()))
        torch.nn.utils.clip_grad_norm_(trainable_parameters, config.max_gradient_norm)
        optimizer.step()
        last_train_loss = sum(microbatch_losses) / len(microbatch_losses)

        epoch_complete = update_in_epoch + 1 == plan.updates_per_epoch
        evaluation_due = (
            completed_step % config.validation_interval == 0 or epoch_complete
        )
        checkpoint_written = False
        best_updated = False
        validation_loss = None
        if evaluation_due:
            _report(progress, f"evaluating candidate at update {completed_step}")
            metrics = evaluate_historical_and_preservation(
                model=model,
                historical_validation=historical_validation,
                modern_validation=modern_validation,
                instruction_batches=instruction_batches,
                context_length=config.context_length,
                device=device,
            )
            validation_loss = metrics["historical_validation_loss"]
            gate_passed = preservation_gate(
                modern_loss=metrics["modern_validation_loss"],
                instruction_loss=metrics["instruction_validation_loss"],
                baseline_modern_loss=baseline_metrics["modern_validation_loss"],
                baseline_instruction_loss=baseline_metrics[
                    "instruction_validation_loss"
                ],
                config=config,
            )
            row = {
                "step": completed_step,
                "epoch": epoch_index + 1,
                "train_loss": last_train_loss,
                "learning_rate": learning_rate,
                **metrics,
                "preservation_gate_passed": gate_passed,
                "elapsed_seconds": time.monotonic() - started_at,
            }
            history.append(row)
            _write_jsonl(output_dir / "loss_history.jsonl", history)
            candidate_path = (
                output_dir / "checkpoints" / f"adapter_step_{completed_step:06d}.pt"
            )
            _save_adapter_checkpoint(
                path=candidate_path,
                model=model,
                config=config,
                plan=plan,
                data_report_sha256=_sha256(encoded_dir / "report.json"),
                row=row,
                baseline_metrics=baseline_metrics,
                get_peft_model_state_dict=dependencies["get_peft_model_state_dict"],
            )
            checkpoint_written = True
            qualifying = (
                gate_passed
                and validation_loss
                <= baseline_metrics["historical_validation_loss"]
                - config.min_historical_improvement
            )
            previous_best = (
                float(best_row["historical_validation_loss"])
                if best_row is not None
                else math.inf
            )
            if qualifying and validation_loss <= previous_best - config.min_historical_improvement:
                best_row = row
                _copy_checkpoint(candidate_path, output_dir / "best_adapter.pt")
                non_improving_evaluations = 0
                best_updated = True
            else:
                non_improving_evaluations += 1

        resume_due = (
            completed_step % config.resume_interval == 0
            or evaluation_due
            or completed_step == plan.planned_updates
        )
        if resume_due:
            _save_resume_checkpoint(
                path=output_dir / "resume.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                plan=plan,
                completed_steps=completed_step,
                data_report_sha256=_sha256(encoded_dir / "report.json"),
                history=history,
                baseline_metrics=baseline_metrics,
                best_row=best_row,
                non_improving_evaluations=non_improving_evaluations,
                get_peft_model_state_dict=dependencies["get_peft_model_state_dict"],
            )

        if reporter.should_report(completed_step, force=evaluation_due):
            reporter.write_progress(
                step=completed_step,
                train_loss=last_train_loss,
                validation_loss=validation_loss,
                learning_rate=learning_rate,
                checkpoint_written=checkpoint_written,
                best_validation=best_updated,
            )
            if evaluation_due:
                print(
                    "preservation | modern_loss={modern:.4f} | "
                    "instruction_loss={instruction:.4f} | gate={gate}".format(
                        modern=history[-1]["modern_validation_loss"],
                        instruction=history[-1]["instruction_validation_loss"],
                        gate="pass" if history[-1]["preservation_gate_passed"] else "fail",
                    ),
                    flush=True,
                )

        completed_steps = completed_step
        if (
            evaluation_due
            and completed_steps >= plan.updates_per_epoch
            and non_improving_evaluations >= config.early_stopping_patience
        ):
            stop_reason = "early_stopping"
            break

    _save_adapter_checkpoint(
        path=output_dir / "final_adapter.pt",
        model=model,
        config=config,
        plan=plan,
        data_report_sha256=_sha256(encoded_dir / "report.json"),
        row={
            "step": completed_steps,
            "train_loss": last_train_loss,
            "stop_reason": stop_reason,
        },
        baseline_metrics=baseline_metrics,
        get_peft_model_state_dict=dependencies["get_peft_model_state_dict"],
    )
    result = {
        "run_version": HISTORICAL_RUN_VERSION,
        "completed_steps": completed_steps,
        "planned_updates": plan.planned_updates,
        "stop_reason": stop_reason,
        "baseline_metrics": baseline_metrics,
        "best_validation_row": best_row,
        "qualified_checkpoint": best_row is not None,
        "best_checkpoint_path": (
            str(output_dir / "best_adapter.pt") if best_row is not None else None
        ),
        "final_checkpoint_path": str(output_dir / "final_adapter.pt"),
        "resume_checkpoint_path": str(output_dir / "resume.pt"),
        "package_versions": minerva_7b_package_versions(dependencies),
    }
    _write_json(output_dir / "result.json", result)
    return result


def build_instruction_preservation_batches(
    *, tokenizer: Any, path: Path, context_length: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("preservation prompt file must contain a non-empty list")
    batches = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("prompt") or not row.get("response"):
            raise ValueError("preservation prompt row is invalid")
        prompt_messages = [{"role": "user", "content": row["prompt"]}]
        full_messages = [
            *prompt_messages,
            {"role": "assistant", "content": row["response"]},
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
        if not isinstance(prompt_ids, list) or not isinstance(full_ids, list):
            raise ValueError("chat template must return token ID lists")
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("assistant response must preserve chat prompt token prefix")
        if len(full_ids) > context_length or len(full_ids) <= len(prompt_ids):
            raise ValueError("preservation prompt has an invalid tokenized length")
        labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
        batches.append((
            torch.tensor(full_ids, dtype=torch.long).unsqueeze(0),
            torch.tensor(labels, dtype=torch.long).unsqueeze(0),
        ))
    return batches


@torch.no_grad()
def evaluate_historical_and_preservation(
    *,
    model: Any,
    historical_validation: torch.Tensor,
    modern_validation: torch.Tensor,
    instruction_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    context_length: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    historical_loss = _evaluate_stream(
        model=model,
        stream=historical_validation,
        context_length=context_length,
        device=device,
    )
    modern_loss = _evaluate_fixed_windows(
        model=model,
        windows=modern_validation,
        device=device,
    )
    instruction_loss = _evaluate_instruction_batches(
        model=model,
        batches=instruction_batches,
        device=device,
    )
    return {
        "historical_validation_loss": historical_loss,
        "modern_validation_loss": modern_loss,
        "instruction_validation_loss": instruction_loss,
    }


def _evaluate_stream(
    *, model: Any, stream: torch.Tensor, context_length: int, device: torch.device
) -> float:
    window_count = (stream.numel() - 1) // context_length
    if window_count <= 0:
        raise ValueError("validation stream contains no complete windows")
    total = 0.0
    for index in range(window_count):
        input_ids, labels = _stream_window(
            stream,
            window_index=index,
            context_length=context_length,
            device=device,
        )
        total += float(model(input_ids=input_ids, labels=labels).loss.item())
    return total / window_count


def _evaluate_fixed_windows(
    *, model: Any, windows: torch.Tensor, device: torch.device
) -> float:
    total = 0.0
    for row in windows:
        input_ids = row[:-1].to(device=device, dtype=torch.long).unsqueeze(0)
        labels = input_ids.clone()
        total += float(model(input_ids=input_ids, labels=labels).loss.item())
    return total / len(windows)


def _evaluate_instruction_batches(
    *,
    model: Any,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    total_loss = 0.0
    total_targets = 0
    for input_ids, labels in batches:
        target_count = int((labels != IGNORE_INDEX).sum().item())
        loss = model(
            input_ids=input_ids.to(device),
            labels=labels.to(device),
        ).loss
        total_loss += float(loss.item()) * target_count
        total_targets += target_count
    return total_loss / total_targets


def _stream_window(
    stream: torch.Tensor,
    *,
    window_index: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = window_index * context_length
    window = stream[start:start + context_length + 1]
    if window.numel() != context_length + 1:
        raise ValueError("requested stream window is incomplete")
    input_ids = window[:-1].to(device=device, dtype=torch.long).unsqueeze(0)
    return input_ids, input_ids.clone()


def _window_order(count: int, *, seed: int) -> list[int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(count, generator=generator).tolist()


def _load_and_verify_data_report(encoded_dir: Path) -> dict[str, Any]:
    report_path = encoded_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("staged-data report is missing artifacts")
    if report.get("model_id") != MINERVA_7B_INSTRUCT_MODEL_ID:
        raise ValueError("staged-data report model does not match Minerva 7B")
    if report.get("revision") != MINERVA_7B_INSTRUCT_REVISION:
        raise ValueError("staged-data report revision does not match the pinned model")
    for row in artifacts.values():
        path = encoded_dir.parent.parent.parent.parent / row["path"]
        if not path.is_file() or _sha256(path) != row["sha256"]:
            raise ValueError(f"staged-data artifact failed integrity check: {path}")
    return report


def _save_adapter_checkpoint(
    *,
    path: Path,
    model: Any,
    config: Minerva7BHistoricalLoRAConfig,
    plan: HistoricalTrainingPlan,
    data_report_sha256: str,
    row: dict[str, Any],
    baseline_metrics: dict[str, float],
    get_peft_model_state_dict: Callable[..., dict[str, torch.Tensor]],
) -> None:
    payload = {
        "checkpoint_type": "minerva_7b_historical_lora_adapter",
        "run_version": HISTORICAL_RUN_VERSION,
        "model_id": config.model_id,
        "revision": config.revision,
        "recipe_config": asdict(config),
        "plan": asdict(plan),
        "data_report_sha256": data_report_sha256,
        "row": row,
        "baseline_metrics": baseline_metrics,
        "adapter_state_dict": _cpu_state(get_peft_model_state_dict(model)),
    }
    _atomic_torch_save(path, payload)


def _save_resume_checkpoint(
    *,
    path: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    config: Minerva7BHistoricalLoRAConfig,
    plan: HistoricalTrainingPlan,
    completed_steps: int,
    data_report_sha256: str,
    history: list[dict[str, Any]],
    baseline_metrics: dict[str, float],
    best_row: dict[str, Any] | None,
    non_improving_evaluations: int,
    get_peft_model_state_dict: Callable[..., dict[str, torch.Tensor]],
) -> None:
    payload = {
        "checkpoint_type": "minerva_7b_historical_lora_resume",
        "run_version": HISTORICAL_RUN_VERSION,
        "model_id": config.model_id,
        "revision": config.revision,
        "recipe_config": asdict(config),
        "plan": asdict(plan),
        "completed_steps": completed_steps,
        "data_report_sha256": data_report_sha256,
        "history": history,
        "baseline_metrics": baseline_metrics,
        "best_validation_row": best_row,
        "non_improving_evaluations": non_improving_evaluations,
        "adapter_state_dict": _cpu_state(get_peft_model_state_dict(model)),
        "optimizer_state_dict": _to_cpu(optimizer.state_dict()),
    }
    _atomic_torch_save(path, payload)


def _restore_resume_checkpoint(
    *,
    path: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    config: Minerva7BHistoricalLoRAConfig,
    plan: HistoricalTrainingPlan,
    data_report_sha256: str,
    set_peft_model_state_dict: Callable[..., Any],
) -> tuple[
    int,
    list[dict[str, Any]],
    dict[str, float],
    dict[str, Any] | None,
    int,
]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "checkpoint_type": "minerva_7b_historical_lora_resume",
        "run_version": HISTORICAL_RUN_VERSION,
        "model_id": config.model_id,
        "revision": config.revision,
        "recipe_config": asdict(config),
        "plan": asdict(plan),
        "data_report_sha256": data_report_sha256,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"resume checkpoint mismatch: {key}")
    completed_steps = int(checkpoint["completed_steps"])
    if completed_steps < 0 or completed_steps >= plan.planned_updates:
        raise ValueError("resume checkpoint has an invalid completed step")
    set_peft_model_state_dict(model, checkpoint["adapter_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (
        completed_steps,
        list(checkpoint["history"]),
        dict(checkpoint["baseline_metrics"]),
        checkpoint.get("best_validation_row"),
        int(checkpoint["non_improving_evaluations"]),
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


def _copy_checkpoint(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(source.read_bytes())
    temporary.replace(destination)


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


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
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


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
