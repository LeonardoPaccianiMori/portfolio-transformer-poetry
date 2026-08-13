"""Bounded PEFT DPO for the frozen Minerva V7 AI-majority preferences."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from sonnet_analysis.minerva_v7_prompt_intervention import build_intervention_prompt


EXPERIMENT_VERSION = "minerva_7b_v7_ai_judged_dpo_v1"
PARENT_STATE_ID = "stage_3_selected"
PARENT_IDENTITY = "478d5979e25a78375d7af0434db6a5432678762fac2d142af2d4798dda53a474"
PREFERENCE_VERSION = "minerva_7b_v7_dpo_preferences_v1"
TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


@dataclass(frozen=True)
class DPOExample:
    pair_id: str
    pair_type: str
    prompt_id: str
    opening_line: str
    chosen: str
    rejected: str
    vote_counts: dict[str, int]


@dataclass(frozen=True)
class TokenizedDPOExample:
    example: DPOExample
    chosen_ids: tuple[int, ...]
    rejected_ids: tuple[int, ...]
    prompt_length: int


def load_dpo_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "experiment_version": EXPERIMENT_VERSION,
        "scope": "exploratory_ai_judge_distillation_not_human_calibrated",
        "parent_state_id": PARENT_STATE_ID,
        "parent_state_identity_sha256": PARENT_IDENTITY,
        "preference_version": PREFERENCE_VERSION,
        "context_length": 1024,
        "epochs": 1,
        "microbatch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-5,
        "minimum_learning_rate": 1e-6,
        "warmup_fraction": 0.1,
        "weight_decay": 0.0,
        "maximum_gradient_norm": 1.0,
        "dpo_beta": 0.1,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": list(TARGET_MODULES),
        "split_seed": 11411,
        "validation_fraction": 0.1,
        "training_seed": 11413,
        "progress_interval": 5,
        "checkpoint_interval": 15,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"AI-judged DPO configuration mismatch: {key}")
    authorization = config.get("authorization", {})
    if (
        authorization.get("ai_judged_dpo_authorized") is not True
        or authorization.get("human_calibrated_claim_authorized") is not False
        or authorization.get("validation_calibration_pairs_eligible_for_training") is not False
        or authorization.get("v7_test_access_authorized") is not False
        or authorization.get("autonomous_completion_workflow_active") is not True
    ):
        raise PermissionError("AI-judged DPO authorization contract changed")
    if float(config.get("hourly_rate_usd", 0)) <= 0:
        raise ValueError("AI-judged DPO must record a positive hourly rate")
    return config


def load_ai_majority_examples(path: Path) -> list[DPOExample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("preference_version") != PREFERENCE_VERSION
        or payload.get("scope") != "exploratory_ai_judge_distillation_not_human_calibrated"
        or payload.get("source_split") != "sonnets_train"
        or payload.get("human_calibration_gate_passed") is not False
        or payload.get("validation_calibration_pairs_included") is not False
        or payload.get("v7_test_accessed") is not False
    ):
        raise ValueError("AI-majority preference dataset lineage mismatch")
    rows = []
    for raw in payload.get("examples", []):
        pair_id = str(raw.get("pair_id", ""))
        if not pair_id.startswith(("pair_", "completion_pair_")):
            raise ValueError("AI-majority dataset contains a non-training pair")
        counts = {key: int(raw["vote_counts"][key]) for key in ("A", "B", "tie")}
        if sum(counts.values()) != 3 or max(counts["A"], counts["B"]) < 2:
            raise ValueError("AI-majority dataset contains a non-decisive pair")
        rows.append(
            DPOExample(
                pair_id=pair_id,
                pair_type=str(raw["pair_type"]),
                prompt_id=str(raw["prompt_id"]),
                opening_line=str(raw["opening_line"]),
                chosen=str(raw["chosen"]),
                rejected=str(raw["rejected"]),
                vote_counts=counts,
            )
        )
    if not rows or len({row.pair_id for row in rows}) != len(rows):
        raise ValueError("AI-majority preference examples are empty or duplicated")
    return rows


def split_examples_by_prompt(
    examples: Sequence[DPOExample], *, validation_fraction: float, seed: int
) -> tuple[list[DPOExample], list[DPOExample]]:
    """Split by prompt identity so paired variants never cross train/validation."""

    if not 0 < validation_fraction < 0.5:
        raise ValueError("DPO validation fraction must be between 0 and 0.5")
    prompt_ids = sorted({row.prompt_id for row in examples})
    validation_count = max(1, round(len(prompt_ids) * validation_fraction))
    ranked = sorted(
        prompt_ids,
        key=lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest(),
    )
    validation_ids = set(ranked[:validation_count])
    train = [row for row in examples if row.prompt_id not in validation_ids]
    validation = [row for row in examples if row.prompt_id in validation_ids]
    if not train or not validation or {row.prompt_id for row in train} & validation_ids:
        raise ValueError("DPO prompt-group split failed")
    return train, validation


def tokenize_dpo_example(
    *, example: DPOExample, tokenizer: Any, context_length: int
) -> TokenizedDPOExample:
    """Tokenize one shared prompt with chosen/rejected continuation-only targets."""

    prompt = build_intervention_prompt(
        tokenizer, example.opening_line, "explicit_no_labels_or_prose"
    )
    prompt_ids = _token_ids(tokenizer, prompt)
    eos = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos, int) or eos < 0:
        raise ValueError("DPO tokenizer must define EOS")
    variants = []
    for text in (example.chosen, example.rejected):
        prefix = f"{example.opening_line}\n"
        if not text.startswith(prefix):
            raise ValueError("DPO candidate does not preserve the exact opening prefix")
        continuation = text[len(prefix):]
        full_ids = [*_token_ids(tokenizer, f"{prompt}{continuation}"), eos]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("DPO prompt tokenization is not an exact prefix")
        if len(full_ids) > context_length:
            raise ValueError(
                f"DPO example exceeds context length: {len(full_ids)} > {context_length}"
            )
        if len(full_ids) <= len(prompt_ids) + 1:
            raise ValueError("DPO continuation has no supervised content")
        variants.append(tuple(full_ids))
    return TokenizedDPOExample(
        example=example,
        chosen_ids=variants[0],
        rejected_ids=variants[1],
        prompt_length=len(prompt_ids),
    )


def collate_dpo_examples(
    examples: Sequence[TokenizedDPOExample], *, pad_token_id: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return chosen/rejected rows plus the response-token mask for log-probs."""

    if not examples:
        raise ValueError("DPO batch must not be empty")
    rows = [ids for row in examples for ids in (row.chosen_ids, row.rejected_ids)]
    prompt_lengths = [row.prompt_length for row in examples for _ in range(2)]
    maximum = max(map(len, rows))
    input_ids = torch.full((len(rows), maximum), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for index, (ids, prompt_length) in enumerate(zip(rows, prompt_lengths, strict=True)):
        length = len(ids)
        input_ids[index, :length] = torch.tensor(ids, dtype=torch.long)
        attention_mask[index, :length] = 1
        response_mask[index, prompt_length:length] = True
    return (
        input_ids.to(device), attention_mask.to(device), response_mask.to(device)
    )


def sequence_response_logps(
    logits: torch.Tensor, input_ids: torch.Tensor, response_mask: torch.Tensor
) -> torch.Tensor:
    """Sum next-token log-probability over response tokens only."""

    if logits.shape[:2] != input_ids.shape or response_mask.shape != input_ids.shape:
        raise ValueError("DPO log-probability tensors have incompatible shapes")
    token_logps = torch.gather(
        F.log_softmax(logits[:, :-1].float(), dim=-1),
        dim=-1,
        index=input_ids[:, 1:].unsqueeze(-1),
    ).squeeze(-1)
    shifted_mask = response_mask[:, 1:]
    if not shifted_mask.any(dim=1).all():
        raise ValueError("every DPO sequence must contain response targets")
    return (token_logps * shifted_mask).sum(dim=1)


def dpo_loss(
    *, policy_logps: torch.Tensor, reference_logps: torch.Tensor, beta: float
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute paired DPO loss for interleaved chosen/rejected sequence rows."""

    if beta <= 0 or policy_logps.shape != reference_logps.shape:
        raise ValueError("DPO log-probabilities or beta are invalid")
    if policy_logps.ndim != 1 or policy_logps.numel() % 2:
        raise ValueError("DPO rows must be interleaved chosen/rejected pairs")
    policy_margin = policy_logps[0::2] - policy_logps[1::2]
    reference_margin = reference_logps[0::2] - reference_logps[1::2]
    logits = beta * (policy_margin - reference_margin)
    losses = -F.logsigmoid(logits)
    return losses.mean(), {
        "policy_margin": float(policy_margin.detach().mean().item()),
        "reference_margin": float(reference_margin.detach().mean().item()),
        "reward_margin": float(logits.detach().mean().item()),
        "preference_accuracy": float((logits.detach() > 0).float().mean().item()),
    }


def learning_rate_for_step(config: Mapping[str, Any], *, step: int, total_steps: int) -> float:
    if step <= 0 or step > total_steps:
        raise ValueError("DPO step is outside the training plan")
    warmup = max(1, math.ceil(total_steps * float(config["warmup_fraction"])))
    peak = float(config["learning_rate"])
    floor = float(config["minimum_learning_rate"])
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return floor + cosine * (peak - floor)


def build_training_plan(
    examples: Sequence[DPOExample], *, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze the prompt-disjoint split and deterministic one-epoch order."""

    train, validation = split_examples_by_prompt(
        examples,
        validation_fraction=float(config["validation_fraction"]),
        seed=int(config["split_seed"]),
    )
    order = list(range(len(train)))
    random.Random(int(config["training_seed"])).shuffle(order)
    accumulation = int(config["gradient_accumulation_steps"])
    if accumulation <= 0:
        raise ValueError("DPO gradient accumulation must be positive")
    total_steps = math.ceil(len(order) / accumulation)
    return {
        "train": train,
        "validation": validation,
        "training_order": order,
        "total_steps": total_steps,
        "split_manifest": {
            "experiment_version": EXPERIMENT_VERSION,
            "split_seed": int(config["split_seed"]),
            "training_seed": int(config["training_seed"]),
            "validation_fraction": float(config["validation_fraction"]),
            "train_pair_ids": [row.pair_id for row in train],
            "validation_pair_ids": [row.pair_id for row in validation],
            "training_order_pair_ids": [train[index].pair_id for index in order],
            "train_prompt_ids": sorted({row.prompt_id for row in train}),
            "validation_prompt_ids": sorted({row.prompt_id for row in validation}),
            "v7_test_accessed": False,
        },
    }


def train_ai_judged_dpo(
    *,
    repo_root: Path,
    config: Mapping[str, Any],
    examples: Sequence[DPOExample],
    state: Mapping[str, Any],
    output_dir: Path,
    qualification: bool,
    resume_from: Path | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Run the single preregistered BF16 LoRA-DPO job or one disposable update."""

    if str(state.get("state_identity_sha256")) != PARENT_IDENTITY:
        raise ValueError("DPO parent is not the exact Stage-3 selected state")
    model_dir = Path(str(state["model_dir"]))
    if not model_dir.is_absolute():
        model_dir = repo_root / model_dir
    if not model_dir.is_dir():
        raise FileNotFoundError(f"DPO parent model is absent: {model_dir}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("AI-judged DPO requires exactly one CUDA GPU")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if properties.total_memory < 76_800 * 1024**2 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("AI-judged DPO requires the qualified H100-class BF16 GPU")

    training_seed = int(config["training_seed"])
    random.seed(training_seed)
    torch.manual_seed(training_seed)
    torch.cuda.manual_seed_all(training_seed)

    plan = build_training_plan(examples, config=config)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "split_manifest.json", plan["split_manifest"])
    _write_json_atomic(output_dir / "frozen_config.json", dict(config))
    dependencies = _load_training_dependencies()
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        model_dir, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized_train = [
        tokenize_dpo_example(
            example=row, tokenizer=tokenizer,
            context_length=int(config["context_length"]),
        )
        for row in plan["train"]
    ]
    tokenized_validation = [
        tokenize_dpo_example(
            example=row, tokenizer=tokenizer,
            context_length=int(config["context_length"]),
        )
        for row in plan["validation"]
    ]
    model = dependencies["AutoModelForCausalLM"].from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        device_map={"": 0},
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
            r=int(config["lora_rank"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            bias="none",
            target_modules=list(config["target_modules"]),
        ),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA attachment produced no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        foreach=False,
    )
    completed_steps = 0
    history: list[dict[str, Any]] = []
    best_validation_loss = math.inf
    if resume_from is not None:
        if qualification:
            raise ValueError("qualification cannot resume or persist mutated state")
        completed_steps, history, best_validation_loss = _restore_dpo_resume(
            path=resume_from, model=model, optimizer=optimizer,
            config=config, plan=plan, dependencies=dependencies,
        )

    total_steps = 1 if qualification else int(plan["total_steps"])
    order = list(plan["training_order"])
    accumulation = int(config["gradient_accumulation_steps"])
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    last_metrics: dict[str, float] = {}
    for step in range(completed_steps + 1, total_steps + 1):
        lr = learning_rate_for_step(config, step=step, total_steps=total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        begin = (step - 1) * accumulation
        indexes = order[begin : begin + accumulation]
        if qualification:
            indexes = order[:accumulation]
        if not indexes:
            raise RuntimeError("DPO training plan produced an empty update")
        weighted_loss = 0.0
        metric_sums = {
            "policy_margin": 0.0, "reference_margin": 0.0,
            "reward_margin": 0.0, "preference_accuracy": 0.0,
        }
        model.train()
        for index in indexes:
            tensors = collate_dpo_examples(
                [tokenized_train[index]], pad_token_id=int(tokenizer.pad_token_id),
                device=device,
            )
            input_ids, attention_mask, response_mask = tensors
            with torch.no_grad(), model.disable_adapter():
                reference_logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits
                reference_logps = sequence_response_logps(
                    reference_logits, input_ids, response_mask
                )
            del reference_logits
            policy_logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            policy_logps = sequence_response_logps(
                policy_logits, input_ids, response_mask
            )
            loss, metrics = dpo_loss(
                policy_logps=policy_logps, reference_logps=reference_logps,
                beta=float(config["dpo_beta"]),
            )
            (loss / len(indexes)).backward()
            weighted_loss += float(loss.detach().item()) / len(indexes)
            for key in metric_sums:
                metric_sums[key] += metrics[key] / len(indexes)
            del policy_logits, policy_logps, reference_logps, loss
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(config["maximum_gradient_norm"])
        )
        if not math.isfinite(float(gradient_norm)) or not math.isfinite(weighted_loss):
            raise FloatingPointError("DPO update produced a non-finite loss or gradient")
        optimizer.step()
        elapsed = time.monotonic() - started
        last_metrics = {
            "step": step, "train_loss": weighted_loss, "learning_rate": lr,
            "gradient_norm": float(gradient_norm), **metric_sums,
            "elapsed_seconds": elapsed,
        }
        history.append(last_metrics)
        due = (
            step == total_steps
            or step % int(config["progress_interval"]) == 0
            or step == 1
        )
        if progress is not None and due:
            eta = elapsed / max(step - completed_steps, 1) * (total_steps - step)
            progress(
                f"update={step}/{total_steps} progress={100 * step / total_steps:.1f}% "
                f"loss={weighted_loss:.4f} lr={lr:.2e} "
                f"reward_margin={metric_sums['reward_margin']:.4f} "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
            )
        if qualification:
            break
        checkpoint_due = (
            step % int(config["checkpoint_interval"]) == 0
            or step == total_steps
        )
        if checkpoint_due:
            validation = evaluate_dpo_preferences(
                model=model, rows=tokenized_validation, tokenizer=tokenizer,
                device=device, beta=float(config["dpo_beta"]),
            )
            history[-1]["validation"] = validation
            _write_jsonl_atomic(output_dir / "history.jsonl", history)
            checkpoint = output_dir / "checkpoints" / f"adapter_step_{step:06d}.pt"
            _save_dpo_checkpoint(
                path=checkpoint, model=model, optimizer=None, config=config,
                plan=plan, completed_steps=step, history=history,
                best_validation_loss=best_validation_loss,
                dependencies=dependencies, checkpoint_type="adapter",
            )
            if validation["loss"] < best_validation_loss:
                best_validation_loss = float(validation["loss"])
                _copy_file_atomic(checkpoint, output_dir / "best_adapter.pt")
            _save_dpo_checkpoint(
                path=output_dir / "resume.pt", model=model, optimizer=optimizer,
                config=config, plan=plan, completed_steps=step, history=history,
                best_validation_loss=best_validation_loss,
                dependencies=dependencies, checkpoint_type="resume",
            )
            if progress is not None:
                progress(
                    f"update={step} validation_loss={validation['loss']:.4f} "
                    f"validation_preference_accuracy="
                    f"{validation['preference_accuracy']:.3f} checkpoint={checkpoint}"
                )

    peak_memory = torch.cuda.max_memory_allocated(device)
    result = {
        "experiment_version": EXPERIMENT_VERSION,
        "scope": "qualification_disposable" if qualification else "authoritative",
        "parent_state_identity_sha256": PARENT_IDENTITY,
        "completed_steps": 1 if qualification else total_steps,
        "planned_steps": total_steps,
        "train_examples": len(plan["train"]),
        "validation_examples": len(plan["validation"]),
        "last_update": last_metrics,
        "peak_gpu_memory_bytes": peak_memory,
        "gpu_name": properties.name,
        "elapsed_seconds": time.monotonic() - started,
        "cost_usd": (time.monotonic() - started) / 3600 * float(config["hourly_rate_usd"]),
        "v7_test_accessed": False,
        "qualification_weights_persisted": False if qualification else None,
    }
    if qualification:
        _write_json_atomic(output_dir / "qualification_report.json", result)
    else:
        _save_dpo_checkpoint(
            path=output_dir / "final_adapter.pt", model=model, optimizer=None,
            config=config, plan=plan, completed_steps=total_steps, history=history,
            best_validation_loss=best_validation_loss,
            dependencies=dependencies, checkpoint_type="adapter",
        )
        result["best_validation_loss"] = best_validation_loss
        result["best_adapter_path"] = str(output_dir / "best_adapter.pt")
        result["final_adapter_path"] = str(output_dir / "final_adapter.pt")
        _write_json_atomic(output_dir / "result.json", result)
    return result


@torch.no_grad()
def evaluate_dpo_preferences(
    *, model: Any, rows: Sequence[TokenizedDPOExample], tokenizer: Any,
    device: torch.device, beta: float,
) -> dict[str, float]:
    """Evaluate the held-out prompt groups against the unchanged shared base."""

    if not rows:
        raise ValueError("DPO validation set is empty")
    model.eval()
    totals = {"loss": 0.0, "policy_margin": 0.0, "reference_margin": 0.0,
              "reward_margin": 0.0, "preference_accuracy": 0.0}
    for row in rows:
        input_ids, attention_mask, response_mask = collate_dpo_examples(
            [row], pad_token_id=int(tokenizer.pad_token_id), device=device,
        )
        with model.disable_adapter():
            reference = sequence_response_logps(
                model(input_ids=input_ids, attention_mask=attention_mask).logits,
                input_ids, response_mask,
            )
        policy = sequence_response_logps(
            model(input_ids=input_ids, attention_mask=attention_mask).logits,
            input_ids, response_mask,
        )
        loss, metrics = dpo_loss(
            policy_logps=policy, reference_logps=reference, beta=beta
        )
        totals["loss"] += float(loss.item())
        for key in metrics:
            totals[key] += metrics[key]
    return {key: value / len(rows) for key, value in totals.items()}


def _save_dpo_checkpoint(
    *, path: Path, model: Any, optimizer: Any | None,
    config: Mapping[str, Any], plan: Mapping[str, Any], completed_steps: int,
    history: Sequence[Mapping[str, Any]], best_validation_loss: float,
    dependencies: Mapping[str, Any], checkpoint_type: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_type": f"minerva_v7_ai_dpo_{checkpoint_type}",
        "experiment_version": EXPERIMENT_VERSION,
        "parent_state_identity_sha256": PARENT_IDENTITY,
        "config": dict(config),
        "split_manifest": plan["split_manifest"],
        "completed_steps": completed_steps,
        "history": list(history),
        "best_validation_loss": best_validation_loss,
        "adapter_state_dict": _cpu_state(
            dependencies["get_peft_model_state_dict"](model)
        ),
        "optimizer_state_dict": (
            _to_cpu(optimizer.state_dict()) if optimizer is not None else None
        ),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "v7_test_accessed": False,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _restore_dpo_resume(
    *, path: Path, model: Any, optimizer: Any, config: Mapping[str, Any],
    plan: Mapping[str, Any], dependencies: Mapping[str, Any],
) -> tuple[int, list[dict[str, Any]], float]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "checkpoint_type": "minerva_v7_ai_dpo_resume",
        "experiment_version": EXPERIMENT_VERSION,
        "parent_state_identity_sha256": PARENT_IDENTITY,
        "config": dict(config),
        "split_manifest": plan["split_manifest"],
        "v7_test_accessed": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"DPO resume checkpoint mismatch: {key}")
    completed = int(payload["completed_steps"])
    if completed <= 0 or completed >= int(plan["total_steps"]):
        raise ValueError("DPO resume completed step is invalid")
    dependencies["set_peft_model_state_dict"](model, payload["adapter_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    torch.set_rng_state(payload["torch_rng_state"])
    torch.cuda.set_rng_state(payload["cuda_rng_state"])
    return completed, list(payload["history"]), float(payload["best_validation_loss"])


def _load_training_dependencies() -> dict[str, Any]:
    try:
        from peft import (
            LoraConfig, get_peft_model, get_peft_model_state_dict,
            set_peft_model_state_dict,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("DPO requires the project Transformers and PEFT environment") from error
    return {
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "get_peft_model_state_dict": get_peft_model_state_dict,
        "set_peft_model_state_dict": set_peft_model_state_dict,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _cpu_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]
