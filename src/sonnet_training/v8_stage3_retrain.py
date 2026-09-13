"""Full-weight V8 Stage-3 retrain with the approved safeguards.

The trainer starts from the published Stage-2 BF16 model, consumes the frozen
V8 train window plan with 5% preservation replay, records validation and
retention losses at fixed intervals, saves a midpoint and a final checkpoint,
and runs small generation probes with the reviewed prosody checker.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from sonnet_training.form_targeted_lora import token_tensor
from sonnet_training.v8_stage3_retrain_data import (
    WINDOW_KIND_REPLAY,
    load_plan_inputs,
)

TRAINER_VERSION = "v8_stage3_retrain_trainer_v1"
Progress = Callable[[str], None]


def stage3_learning_rate(
    step: int,
    *,
    total_updates: int,
    warmup_updates: int,
    peak_learning_rate: float,
    minimum_learning_rate: float,
) -> float:
    if step < warmup_updates:
        return peak_learning_rate * (step + 1) / max(1, warmup_updates)
    if total_updates <= warmup_updates:
        return peak_learning_rate
    progress = (step - warmup_updates) / (total_updates - warmup_updates)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_learning_rate + (peak_learning_rate - minimum_learning_rate) * cosine


def retention_ratio(baseline_loss: float, current_loss: float) -> float:
    if not math.isfinite(baseline_loss) or baseline_loss <= 0:
        raise ValueError("baseline loss must be finite and positive")
    return current_loss / baseline_loss


def abort_reason(
    *,
    loss: float,
    preclip_gradient_norm: float | None,
    elapsed_seconds: float,
    hourly_rate_usd: float,
    spend_ceiling_usd: float,
    max_preclip_gradient_norm: float,
) -> str | None:
    if not math.isfinite(loss):
        return "non_finite_loss"
    if (
        preclip_gradient_norm is not None
        and preclip_gradient_norm > max_preclip_gradient_norm
    ):
        return "gradient_spike"
    if elapsed_seconds / 3600.0 * hourly_rate_usd > spend_ceiling_usd:
        return "spend_ceiling"
    return None


def window_loss(model: Any, tokens: Any, start: int, context_length: int, device: Any) -> Any:
    import torch

    window = token_tensor(tokens, start, context_length)
    inputs = torch.as_tensor(window[:-1][None, :], dtype=torch.long, device=device)
    labels = torch.as_tensor(window[1:][None, :], dtype=torch.long, device=device)
    return model(input_ids=inputs, labels=labels).loss


def evaluate_stream_loss(
    model: Any,
    tokens: Any,
    *,
    context_length: int,
    device: Any,
    max_windows: int,
) -> float | None:
    import torch

    total_windows = int(tokens.size) // context_length
    if total_windows == 0:
        return None
    starts = [index * context_length for index in range(min(total_windows, max_windows))]
    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for start in starts:
            losses.append(float(window_loss(model, tokens, start, context_length, device)))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def run_probe(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Mapping[str, Any]],
    recipe: Mapping[str, Any],
    seed: int,
    device: Any,
    batch_size: int,
) -> dict[str, Any]:
    from sonnet_analysis.minerva_v7_high_volume_generation import generate_batch
    from sonnet_analysis.minerva_v7_prompt_intervention import build_intervention_prompt
    from sonnet_evaluation.sonnet_prosody_sealed import score_output

    jobs = [{"prompt": dict(prompt), "seed": int(seed)} for prompt in prompts]
    was_training = model.training
    model.eval()
    results = generate_batch(
        model=model,
        tokenizer=tokenizer,
        jobs=jobs,
        recipe=recipe,
        device=device,
        prompt_builder=lambda tok, opening: build_intervention_prompt(
            tok, opening, "explicit_no_labels_or_prose"
        ),
    )
    if was_training:
        model.train()
    scored = [score_output(str(result["text"])) for result in results]
    accepted = [row["hendecasyllable_lines"] for row in scored]
    return {
        "output_count": len(scored),
        "mean_accepted_lines": (sum(accepted) / len(accepted)) if accepted else None,
        "full_form_valid": sum(1 for row in scored if row["all_lines_valid"]),
        "mean_rhyme_score": (
            sum(row["rhyme_score"] for row in scored) / len(scored) if scored else None
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_checkpoint(model: Any, output_dir: Path, update: int) -> dict[str, Any]:
    path = output_dir / f"checkpoint_update_{update:04d}"
    model.save_pretrained(path, safe_serialization=True)
    files = {
        file.name: _sha256(file)
        for file in sorted(path.glob("*"))
        if file.is_file()
    }
    return {"update": update, "path": str(path), "files": files}


def train_v8_stage3_retrain(
    *,
    stage2_dir: Path,
    token_shard_path: Path,
    replay_shard_path: Path,
    validation_shard_path: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    prompts: Sequence[Mapping[str, Any]],
    progress: Progress | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sonnet_training.form_targeted_data import load_token_shard

    data = config["data"]
    training = config["training"]
    probe_config = config["probe"]
    evaluation = config["evaluation"]
    context_length = int(data["context_length"])
    windows_per_update = int(data["windows_per_update"])

    train_tokens, replay_tokens, plan, summary = load_plan_inputs(
        train_shard=token_shard_path,
        replay_shard=replay_shard_path,
        config=config,
    )
    validation_tokens = load_token_shard(validation_shard_path)
    replay_used = int(summary["replay_tokens_used"])
    replay_holdout = replay_tokens[replay_used:]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(stage2_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(stage2_dir),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)

    optimizer_name = "AdamW"
    try:
        from bitsandbytes.optim import PagedAdamW8bit

        optimizer = PagedAdamW8bit(
            model.parameters(),
            lr=float(training["peak_learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        optimizer_name = "PagedAdamW8bit"
    except Exception:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["peak_learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )

    eval_windows = int(training["evaluation_windows"])
    baseline_losses = {
        "validation_loss": evaluate_stream_loss(
            model, validation_tokens, context_length=context_length, device=device,
            max_windows=eval_windows,
        ),
        "replay_holdout_loss": evaluate_stream_loss(
            model, replay_holdout, context_length=context_length, device=device,
            max_windows=eval_windows,
        ),
    }
    model.train()
    if progress:
        progress(
            "baseline losses "
            f"validation={baseline_losses['validation_loss']} "
            f"replay={baseline_losses['replay_holdout_loss']}"
        )

    updates_total = int(summary["optimizer_updates"])
    warmup = int(training["warmup_updates"])
    peak = float(training["peak_learning_rate"])
    minimum = float(training["minimum_learning_rate"])
    eval_interval = int(training["evaluation_interval_updates"])
    checkpoint_updates = {int(value) for value in training.get("checkpoint_updates", [])}
    retention_gate = float(training["retention_loss_ratio_gate"])
    spend_ceiling = float(training["spend_ceiling_usd"])
    hourly_rate = float(training["hourly_rate_usd"])
    max_preclip = float(training["max_preclip_gradient_norm"])
    baseline_probe = float(probe_config["baseline_mean_accepted_lines"])
    probe_abort_ratio = float(probe_config["abort_ratio"])

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    started = time.monotonic()
    aborted: str | None = None
    losses: list[float] = []
    checkpoints: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    losses_eval: list[dict[str, Any]] = []
    micro = 0

    with log_path.open("w", encoding="utf-8") as log:
        for update in range(1, updates_total + 1):
            learning_rate = stage3_learning_rate(
                update - 1,
                total_updates=updates_total,
                warmup_updates=warmup,
                peak_learning_rate=peak,
                minimum_learning_rate=minimum,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            micro_losses: list[float] = []
            for _ in range(windows_per_update):
                entry = plan[micro]
                micro += 1
                tokens = (
                    train_tokens
                    if entry["kind"] != WINDOW_KIND_REPLAY
                    else replay_tokens
                )
                loss = window_loss(
                    model, tokens, int(entry["token_start"]), context_length, device
                )
                (loss / windows_per_update).backward()
                micro_losses.append(float(loss.detach()))
            preclip = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["max_gradient_norm"])
                )
            )
            optimizer.step()
            mean_loss = sum(micro_losses) / len(micro_losses)
            losses.append(mean_loss)
            log.write(
                json.dumps(
                    {
                        "update": update,
                        "loss": mean_loss,
                        "learning_rate": learning_rate,
                        "preclip_gradient_norm": preclip,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                )
                + "\n"
            )
            log.flush()

            aborted = abort_reason(
                loss=mean_loss,
                preclip_gradient_norm=preclip,
                elapsed_seconds=time.monotonic() - started,
                hourly_rate_usd=hourly_rate,
                spend_ceiling_usd=spend_ceiling,
                max_preclip_gradient_norm=max_preclip,
            )
            if aborted:
                break

            if update % eval_interval == 0 or update == updates_total:
                validation_loss = evaluate_stream_loss(
                    model, validation_tokens, context_length=context_length,
                    device=device, max_windows=eval_windows,
                )
                replay_loss = evaluate_stream_loss(
                    model, replay_holdout, context_length=context_length,
                    device=device, max_windows=eval_windows,
                )
                ratio = None
                if baseline_losses["replay_holdout_loss"] and replay_loss:
                    ratio = retention_ratio(
                        baseline_losses["replay_holdout_loss"], replay_loss
                    )
                losses_eval.append(
                    {
                        "update": update,
                        "validation_loss": validation_loss,
                        "replay_holdout_loss": replay_loss,
                        "retention_ratio": ratio,
                    }
                )
                log.write(json.dumps({"eval": losses_eval[-1]}) + "\n")
                log.flush()
                if progress:
                    progress(
                        f"update {update}/{updates_total} loss={mean_loss:.4f} "
                        f"val={validation_loss} retention={ratio}"
                    )

            if update in checkpoint_updates:
                checkpoints.append(_save_checkpoint(model, output_dir, update))
                model.train()
                probe = run_probe(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts[: int(probe_config["prompt_count"])],
                    recipe=evaluation["recipe"],
                    seed=int(probe_config["seed"]),
                    device=device,
                    batch_size=int(probe_config["batch_size"]),
                )
                probes.append({"update": update, **probe})
                log.write(json.dumps({"probe": probes[-1]}) + "\n")
                log.flush()
                if (
                    probe["mean_accepted_lines"] is not None
                    and probe["mean_accepted_lines"] < baseline_probe * probe_abort_ratio
                ):
                    aborted = "probe_degenerate"
                    break

        final_checkpoint = None
        if aborted is None:
            final_checkpoint = _save_checkpoint(model, output_dir, updates_total)
            model.train()
            probe = run_probe(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts[: int(probe_config["prompt_count"])],
                recipe=evaluation["recipe"],
                seed=int(probe_config["seed"]),
                device=device,
                batch_size=int(probe_config["batch_size"]),
            )
            probes.append({"update": updates_total, **probe})
            log.write(json.dumps({"probe": probes[-1]}) + "\n")
            log.flush()
            checkpoints.append(final_checkpoint)

    final_eval = losses_eval[-1] if losses_eval else None
    retention_passed = (
        final_eval is not None
        and final_eval["retention_ratio"] is not None
        and final_eval["retention_ratio"] <= retention_gate
    )
    elapsed = time.monotonic() - started
    report = {
        "trainer_version": TRAINER_VERSION,
        "stage2_dir": str(stage2_dir),
        "token_shard_path": str(token_shard_path),
        "token_shard_sha256": _sha256(token_shard_path),
        "replay_shard_path": str(replay_shard_path),
        "replay_shard_sha256": _sha256(replay_shard_path),
        "validation_shard_path": str(validation_shard_path),
        "validation_shard_sha256": _sha256(validation_shard_path),
        "optimizer": optimizer_name,
        "data": summary,
        "baseline_losses": baseline_losses,
        "final_eval": final_eval,
        "retention_loss_ratio_gate": retention_gate,
        "retention_gate_passed": retention_passed,
        "updates_completed": len(losses),
        "updates_planned": updates_total,
        "final_loss": losses[-1] if losses else None,
        "checkpoints": checkpoints,
        "probes": probes,
        "aborted": aborted,
        "elapsed_seconds": elapsed,
        "estimated_cost_usd": elapsed / 3600.0 * hourly_rate,
        "config": {
            "data": dict(data),
            "training": dict(training),
            "probe": dict(probe_config),
        },
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
