"""One-arm form-targeted LoRA supervised fine-tuning on the V8 train sonnets.

The trainer packs the encoded sonnet stream into fixed-length next-token
windows, trains a PEFT LoRA adapter on the Stage-3 BF16 base, and writes the
adapter plus a run report. It is a research pilot: the adapter stays local and
is not released.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from sonnet_training.form_targeted_data import INT32_BYTES, load_token_shard

TRAINER_VERSION = "form_targeted_lora_trainer_v1"
Progress = Callable[[str], None]


def window_count(token_count: int, sequence_tokens: int) -> int:
    return max(0, (token_count - 1) // sequence_tokens)


def cosine_learning_rate(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    base_learning_rate: float,
) -> float:
    if step < warmup_steps:
        return base_learning_rate * (step + 1) / max(1, warmup_steps)
    if total_steps <= warmup_steps:
        return base_learning_rate
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def batch_starts(
    window_total: int, batch_size: int, *, seed: int, epoch: int
) -> list[list[int]]:
    order = np.random.default_rng(seed + epoch).permutation(window_total)
    return [
        order[start : start + batch_size].tolist()
        for start in range(0, order.size, batch_size)
    ]


def token_tensor(tokens: np.ndarray, start: int, sequence_tokens: int) -> np.ndarray:
    if start < 0 or start + sequence_tokens + 1 > tokens.size:
        raise ValueError("window outside the token stream")
    return tokens[start : start + sequence_tokens + 1]


def train_form_targeted_lora(
    *,
    base_model_dir: Path,
    token_shard_path: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    progress: Progress | None = None,
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    training = config["training"]
    data_config = config["data"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokens = load_token_shard(token_shard_path)
    sequence_tokens = int(data_config["max_sequence_tokens"])
    windows = window_count(int(tokens.size), sequence_tokens)
    if windows == 0:
        raise ValueError("token stream is too short for one window")

    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(training["lora_rank"]),
        lora_alpha=int(training["lora_alpha"]),
        lora_dropout=float(training["lora_dropout"]),
        bias="none",
        target_modules=list(training["target_modules"]),
    )
    model = get_peft_model(model, lora_config)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    max_steps = training.get("max_steps")
    batches_per_epoch = math.ceil(windows / batch_size)
    optimizer_steps_per_epoch = math.ceil(batches_per_epoch / accumulation)
    total_steps = optimizer_steps_per_epoch * epochs
    if max_steps:
        total_steps = min(total_steps, int(max_steps))
    warmup_steps = int(total_steps * float(training["warmup_ratio"]))
    seed = int(training["seed"])

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    started = time.monotonic()
    step = 0
    pending = 0
    tokens_seen = 0
    losses: list[float] = []

    def optimizer_update(learning_rate: float, recorded_loss: float) -> bool:
        nonlocal step, pending
        torch.nn.utils.clip_grad_norm_(
            trainable, float(training["max_grad_norm"])
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        pending = 0
        losses.append(recorded_loss)
        log.write(
            json.dumps(
                {
                    "step": step,
                    "loss": recorded_loss,
                    "learning_rate": learning_rate,
                    "tokens_seen": tokens_seen,
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            + "\n"
        )
        log.flush()
        if progress and step % 10 == 0:
            progress(
                f"step {step}/{total_steps} loss={recorded_loss:.4f} tokens={tokens_seen}"
            )
        return step >= total_steps

    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(epochs):
            for batch_index, starts in enumerate(
                batch_starts(windows, batch_size, seed=seed, epoch=epoch)
            ):
                learning_rate = cosine_learning_rate(
                    step,
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    base_learning_rate=float(training["learning_rate"]),
                )
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                inputs = torch.as_tensor(
                    np.stack([token_tensor(tokens, s, sequence_tokens)[:-1] for s in starts]),
                    dtype=torch.long,
                    device=device,
                )
                labels = torch.as_tensor(
                    np.stack([token_tensor(tokens, s, sequence_tokens)[1:] for s in starts]),
                    dtype=torch.long,
                    device=device,
                )
                outputs = model(input_ids=inputs, labels=labels)
                loss = outputs.loss / accumulation
                loss.backward()
                tokens_seen += int(inputs.numel())
                pending += 1
                if pending == accumulation:
                    if optimizer_update(learning_rate, float(loss.detach()) * accumulation):
                        break
            if pending:
                if optimizer_update(learning_rate, float(loss.detach()) * accumulation):
                    break
            if step >= total_steps:
                break

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_file = adapter_dir / "adapter_model.safetensors"
    report = {
        "trainer_version": TRAINER_VERSION,
        "base_model_dir": str(base_model_dir),
        "token_shard_path": str(token_shard_path),
        "token_shard_sha256": hashlib.sha256(token_shard_path.read_bytes()).hexdigest(),
        "sequence_tokens": sequence_tokens,
        "window_count": windows,
        "epochs": epochs,
        "completed_steps": step,
        "planned_steps": total_steps,
        "tokens_seen": tokens_seen,
        "final_loss": losses[-1] if losses else None,
        "adapter_dir": str(adapter_dir),
        "adapter_sha256": (
            hashlib.sha256(adapter_file.read_bytes()).hexdigest()
            if adapter_file.is_file()
            else None
        ),
        "elapsed_seconds": time.monotonic() - started,
        "config": dict(training),
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
