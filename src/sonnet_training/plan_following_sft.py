"""Plan-following supervised fine-tuning for pre-committed line endings.

Training examples pair the frozen instruction prompt, the opening line, and a
numbered list of the sonnet's own endings with the continuation response. The
loss is computed only on the response. The adapter trains on top of the
verifier-DPO-merged Stage-3 model.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from sonnet_analysis.plan_then_poem_validation import adherence, planned_prompt
from sonnet_evaluation.rhyme_lexicon import line_final_word
from sonnet_training.form_targeted_data import read_sonnet_text
from sonnet_training.minerva_v7_ai_dpo import TARGET_MODULES
from sonnet_training.v8_stage3_retrain import stage3_learning_rate

TRAINER_VERSION = "plan_following_sft_v1"
Progress = Callable[[str], None]


def build_examples(
    rows: Iterable[Mapping[str, str]],
    root: Path,
    *,
    tokenizer: Any,
    max_sequence_tokens: int = 1024,
    limit: int | None = None,
    excluded_unit_ids: Iterable[str] = (),
    progress: Progress | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    excluded = set(excluded_unit_ids)
    examples: list[dict[str, Any]] = []
    skipped = {"excluded": 0, "malformed": 0, "too_long": 0}
    row_list = list(rows)
    for index, row in enumerate(row_list, start=1):
        if row["unit_id"] in excluded:
            skipped["excluded"] += 1
            continue
        text = read_sonnet_text(root, row)
        raw_lines = text.splitlines()
        nonempty = [
            (line_index, line.strip())
            for line_index, line in enumerate(raw_lines)
            if line.strip()
        ]
        if len(nonempty) != 14:
            skipped["malformed"] += 1
            continue
        lines = [line for _, line in nonempty]
        words = [line_final_word(line) for line in lines]
        if any(word is None for word in words):
            skipped["malformed"] += 1
            continue
        opening = lines[0]
        first_index = nonempty[0][0]
        continuation = "\n".join(raw_lines[first_index + 1 :]).strip("\n") + "\n"
        prompt = planned_prompt(tokenizer, opening, [str(word) for word in words])
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(
            continuation, add_special_tokens=False
        )["input_ids"] + [int(tokenizer.eos_token_id)]
        if len(prompt_ids) + len(target_ids) > max_sequence_tokens:
            skipped["too_long"] += 1
            continue
        examples.append(
            {
                "unit_id": row["unit_id"],
                "prompt_ids": list(prompt_ids),
                "target_ids": list(target_ids),
                "planned_words": [str(word) for word in words],
                "opening_line": opening,
            }
        )
        if limit is not None and len(examples) >= limit:
            break
        if progress and index % 4000 == 0:
            progress(f"built {len(examples)} examples from {index} rows")
    if not examples:
        raise ValueError("no plan-following examples could be built")
    return examples, skipped


def split_examples(
    examples: Sequence[Mapping[str, Any]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation fraction must be between zero and one half")
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    validation_count = max(1, int(len(examples) * validation_fraction))
    validation_indexes = set(order[:validation_count])
    train = [example for index, example in enumerate(examples) if index not in validation_indexes]
    validation = [examples[index] for index in order[:validation_count]]
    return train, validation


def pad_batch(
    examples: Sequence[Mapping[str, Any]], *, pad_token_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_length = max(
        len(example["prompt_ids"]) + len(example["target_ids"])
        for example in examples
    )
    input_ids = np.full((len(examples), max_length), pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((len(examples), max_length), dtype=np.int64)
    labels = np.full((len(examples), max_length), -100, dtype=np.int64)
    for row_index, example in enumerate(examples):
        prompt_length = len(example["prompt_ids"])
        ids = list(example["prompt_ids"]) + list(example["target_ids"])
        input_ids[row_index, : len(ids)] = ids
        attention_mask[row_index, : len(ids)] = 1
        labels[row_index, prompt_length : len(ids)] = list(example["target_ids"])
    return input_ids, attention_mask, labels


def _evaluate_loss(model: Any, examples: Sequence[Mapping[str, Any]], *, pad_token_id: int, device: Any, batch_size: int) -> float:
    import torch

    was_training = model.training
    model.eval()
    losses = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            input_ids, attention_mask, labels = pad_batch(
                batch, pad_token_id=pad_token_id
            )
            outputs = model(
                input_ids=torch.as_tensor(input_ids, device=device),
                attention_mask=torch.as_tensor(attention_mask, device=device),
                labels=torch.as_tensor(labels, device=device),
            )
            losses.append(float(outputs.loss))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def train_plan_following_sft(
    *,
    state: Mapping[str, Any],
    verifier_adapter_path: Path,
    examples: Sequence[Mapping[str, Any]],
    validation_examples: Sequence[Mapping[str, Any]],
    probe_jobs: Sequence[Mapping[str, Any]],
    output_dir: Path,
    config: Mapping[str, Any],
    progress: Progress | None = None,
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    training = config["training"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("plan-following SFT requires exactly one CUDA GPU")
    model_dir = Path(str(state["model_dir"]))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    if not verifier_adapter_path.is_file():
        raise FileNotFoundError(verifier_adapter_path)
    verifier = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(training["verifier_lora_rank"]),
        lora_alpha=int(training["verifier_lora_alpha"]),
        lora_dropout=float(training["verifier_lora_dropout"]),
        bias="none",
        target_modules=list(TARGET_MODULES),
    )
    model = get_peft_model(model, verifier)
    checkpoint = torch.load(verifier_adapter_path, map_location="cpu", weights_only=True)
    if checkpoint.get("parent_state_identity_sha256") != state["state_identity_sha256"]:
        raise ValueError("verifier adapter parent mismatch")
    set_peft_model_state_dict(model, checkpoint["adapter_state_dict"])
    model = model.merge_and_unload()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    plan_lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(training["lora_rank"]),
        lora_alpha=int(training["lora_alpha"]),
        lora_dropout=float(training["lora_dropout"]),
        bias="none",
        target_modules=list(TARGET_MODULES),
    )
    model = get_peft_model(model, plan_lora)
    model.enable_input_require_grads()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    steps_per_epoch = math.ceil(len(examples) / batch_size)
    total_steps = max(1, math.ceil(steps_per_epoch / accumulation)) * epochs
    warmup = max(1, int(total_steps * float(training["warmup_fraction"])))
    midpoint = max(1, total_steps // 2)
    eval_interval = int(training["evaluation_interval"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    started = time.monotonic()
    history: list[dict[str, Any]] = []
    adapters: list[dict[str, Any]] = []
    step = 0

    def save_adapter(label: str) -> dict[str, Any]:
        path = output_dir / f"adapter_{label}"
        model.save_pretrained(path, safe_serialization=True)
        adapter_file = path / "adapter_model.safetensors"
        return {
            "label": label,
            "step": step,
            "path": str(path),
            "sha256": hashlib.sha256(adapter_file.read_bytes()).hexdigest()
            if adapter_file.is_file()
            else None,
        }

    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(epochs):
            order = list(range(len(examples)))
            random.Random(int(training["seed"]) + epoch).shuffle(order)
            pending = 0
            for batch_index in range(steps_per_epoch):
                batch_examples = [
                    examples[index]
                    for index in order[
                        batch_index * batch_size : (batch_index + 1) * batch_size
                    ]
                ]
                learning_rate = stage3_learning_rate(
                    step,
                    total_updates=total_steps,
                    warmup_updates=warmup,
                    peak_learning_rate=float(training["learning_rate"]),
                    minimum_learning_rate=float(training["minimum_learning_rate"]),
                )
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                input_ids, attention_mask, labels = pad_batch(
                    batch_examples, pad_token_id=int(tokenizer.pad_token_id)
                )
                outputs = model(
                    input_ids=torch.as_tensor(input_ids, device=device),
                    attention_mask=torch.as_tensor(attention_mask, device=device),
                    labels=torch.as_tensor(labels, device=device),
                )
                (outputs.loss / accumulation).backward()
                pending += 1
                if pending < accumulation and batch_index < steps_per_epoch - 1:
                    continue
                preclip = float(
                    torch.nn.utils.clip_grad_norm_(
                        trainable, float(training["max_gradient_norm"])
                    )
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                step += 1
                recorded = float(outputs.loss.detach())
                if not math.isfinite(recorded) or preclip > float(
                    training["max_preclip_gradient_norm"]
                ):
                    log.write(
                        json.dumps(
                            {"step": step, "abort": "instability", "loss": recorded}
                        )
                        + "\n"
                    )
                    log.flush()
                    raise RuntimeError("plan-following SFT aborted on instability")
                if step % eval_interval == 0 or step == total_steps:
                    validation_loss = _evaluate_loss(
                        model,
                        validation_examples,
                        pad_token_id=int(tokenizer.pad_token_id),
                        device=device,
                        batch_size=batch_size,
                    )
                    history.append(
                        {
                            "step": step,
                            "train_loss": recorded,
                            "validation_loss": validation_loss,
                            "learning_rate": learning_rate,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    )
                    log.write(json.dumps({"history": history[-1]}) + "\n")
                    log.flush()
                    if progress:
                        progress(
                            f"step {step}/{total_steps} train={recorded:.4f} "
                            f"val={validation_loss:.4f}"
                        )
                if step == midpoint:
                    adapters.append(save_adapter("midpoint"))
                    if progress:
                        progress("midpoint adapter saved")
        adapters.append(save_adapter("final"))

    probe = None
    if probe_jobs:
        from sonnet_analysis.minerva_v7_high_volume_generation import generate_batch

        model.eval()
        results = generate_batch(
            model=model,
            tokenizer=tokenizer,
            jobs=[dict(job) for job in probe_jobs],
            recipe=config["probe"]["recipe"],
            device=device,
        )
        metrics = [
            adherence(str(result["text"]), job["planned_words"])
            for job, result in zip(probe_jobs, results, strict=True)
        ]
        probe = {
            "output_count": len(metrics),
            "mean_key_match_rate": sum(
                float(row["key_match_rate"] or 0.0) for row in metrics
            )
            / len(metrics),
            "mean_word_match_rate": sum(
                float(row["word_match_rate"] or 0.0) for row in metrics
            )
            / len(metrics),
        }
    model.train()
    merged_path = output_dir / "merged_model"
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_path, safe_serialization=True)
    merged_sha = None
    for candidate in sorted(merged_path.glob("model-*.safetensors")):
        merged_sha = candidate.name if merged_sha is None else merged_sha
    elapsed = time.monotonic() - started
    report = {
        "trainer_version": TRAINER_VERSION,
        "state_identity_sha256": state.get("state_identity_sha256"),
        "verifier_adapter_path": str(verifier_adapter_path),
        "verifier_adapter_sha256": hashlib.sha256(
            verifier_adapter_path.read_bytes()
        ).hexdigest(),
        "example_count": len(examples),
        "validation_count": len(validation_examples),
        "steps": step,
        "planned_steps": total_steps,
        "history": history,
        "adapters": adapters,
        "probe": probe,
        "merged_model_dir": str(merged_path),
        "merged_model_files": sorted(
            path.name for path in merged_path.glob("*.safetensors")
        ),
        "elapsed_seconds": elapsed,
        "estimated_cost_usd": elapsed
        / 3600
        * float(config["training"]["hourly_rate_usd"]),
        "v7_test_accessed": False,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
