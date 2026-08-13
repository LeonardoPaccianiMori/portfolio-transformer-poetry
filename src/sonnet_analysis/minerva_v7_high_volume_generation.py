"""Batched, resumable high-volume exploratory generation for Minerva V7."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from sonnet_analysis.minerva_v7_exploratory_prompts import validate_exploratory_prompt_manifest
from sonnet_analysis.minerva_v7_generation import (
    banned_next_tokens,
    build_sonnet_candidate_prompt,
    completed_non_empty_line_count,
    prepare_minerva_sampling_logits,
)


HIGH_VOLUME_VERSION = "minerva_7b_v7_high_volume_generation_v1"
EXPECTED_PROMPT_SHA256 = "2f33aa518aa61c11193831e53b07fd3bd861a72bf68bb23c0e0e5b1a13b1d0c7"
EXPECTED_SEEDS = tuple(range(4200, 4208))
EXPECTED_RECIPE_IDS = ("conservative", "balanced", "creative")


def load_high_volume_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("generation_version") != HIGH_VOLUME_VERSION:
        raise ValueError("high-volume generation version mismatch")
    if config.get("prompt_manifest_sha256") != EXPECTED_PROMPT_SHA256:
        raise ValueError("high-volume prompt hash mismatch")
    if tuple(config.get("seeds", [])) != EXPECTED_SEEDS:
        raise ValueError("high-volume seed contract mismatch")
    recipes = config.get("recipes", [])
    if tuple(row.get("recipe_id") for row in recipes) != EXPECTED_RECIPE_IDS:
        raise ValueError("high-volume recipe contract mismatch")
    if config.get("prompt_count") != 120 or config.get("outputs_per_state") != 2880:
        raise ValueError("high-volume output-count contract mismatch")
    authorization = config.get("authorization", {})
    if (
        authorization.get("exploratory_only") is not True
        or authorization.get("gpu_execution_requires_user_manual_launch") is not True
        or any(
            authorization.get(key) is not False
            for key in (
                "v7_test_access_authorized", "training_authorized",
                "causal_experiments_authorized", "instance_lifecycle_action_authorized",
            )
        )
    ):
        raise PermissionError("high-volume authorization boundary changed")
    return config


def generate_batch(
    *, model: Any, tokenizer: Any, jobs: Sequence[Mapping[str, Any]],
    recipe: Mapping[str, Any], device: torch.device | str,
) -> list[dict[str, Any]]:
    """Decode one fixed-recipe batch with an independent RNG stream per job."""

    if not jobs:
        raise ValueError("generation batch must not be empty")
    resolved_device = torch.device(device)
    rendered = [build_sonnet_candidate_prompt(tokenizer, str(job["prompt"]["opening_line"])) for job in jobs]
    previous_side = getattr(tokenizer, "padding_side", "right")
    previous_pad = getattr(tokenizer, "pad_token_id", None)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    try:
        encoded = tokenizer(
            rendered, add_special_tokens=False, return_tensors="pt", padding=True,
        )
    finally:
        tokenizer.padding_side = previous_side
        tokenizer.pad_token_id = previous_pad
    input_ids = encoded["input_ids"].to(resolved_device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(resolved_device)
    conditioning_ids = [
        input_ids[index][attention_mask[index].bool()].cpu().tolist()
        for index in range(len(jobs))
    ]
    generators = [torch.Generator(device=resolved_device).manual_seed(int(job["seed"])) for job in jobs]
    generated: list[list[int]] = [[] for _ in jobs]
    continuations = ["" for _ in jobs]
    finished = [False for _ in jobs]
    stop_reasons = ["max_new_tokens" for _ in jobs]
    current = input_ids
    past = None
    special_ids = {
        int(value) for value in getattr(tokenizer, "all_special_ids", [])
        if isinstance(value, int) and value >= 0
    }
    started = time.monotonic()
    model.eval()
    with torch.inference_mode():
        for _step in range(int(recipe["max_new_tokens"])):
            if all(finished):
                break
            outputs = model(
                input_ids=current, attention_mask=attention_mask,
                past_key_values=past, use_cache=True, return_dict=True,
            )
            logits = outputs.logits[:, -1, :].float()
            past = outputs.past_key_values
            next_tokens = []
            active_mask = []
            for index, job in enumerate(jobs):
                if finished[index]:
                    next_tokens.append(int(tokenizer.eos_token_id))
                    active_mask.append(0)
                    continue
                row = logits[index : index + 1]
                if special_ids:
                    row[:, list(special_ids)] = -torch.inf
                banned = banned_next_tokens(generated[index], int(recipe["no_repeat_ngram_size"]))
                if banned:
                    row[:, list(banned)] = -torch.inf
                row = prepare_minerva_sampling_logits(
                    row, generated_token_ids=generated[index],
                    temperature=float(recipe["temperature"]), top_k=recipe["top_k"],
                    top_p=float(recipe["top_p"]),
                    repetition_penalty=float(recipe["repetition_penalty"]),
                )
                probabilities = torch.softmax(row, dim=-1)
                if not torch.isfinite(probabilities).all():
                    raise RuntimeError("high-volume generation produced invalid probabilities")
                token = int(torch.multinomial(probabilities, 1, generator=generators[index]).item())
                next_tokens.append(token)
                active_mask.append(1)
                generated[index].append(token)
                continuations[index] = tokenizer.decode(
                    generated[index], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if completed_non_empty_line_count(continuations[index]) >= int(recipe["continuation_line_target"]):
                    finished[index] = True
                    stop_reasons[index] = "target_lines"
            current = torch.tensor(next_tokens, dtype=torch.long, device=resolved_device).unsqueeze(1)
            attention_mask = torch.cat(
                [attention_mask, torch.tensor(active_mask, dtype=attention_mask.dtype, device=resolved_device).unsqueeze(1)],
                dim=1,
            )
    elapsed = time.monotonic() - started
    return [
        {
            "text": f"{job['prompt']['opening_line']}\n{continuations[index]}",
            "opening_line": str(job["prompt"]["opening_line"]),
            "conditioning_prompt": rendered[index],
            "conditioning_input_ids": conditioning_ids[index],
            "generated_token_ids": generated[index],
            "seed": int(job["seed"]),
            "stop_reason": stop_reasons[index],
            "generated_new_tokens": len(generated[index]),
            "completed_continuation_lines": completed_non_empty_line_count(continuations[index]),
            "batch_elapsed_seconds": elapsed,
            "batch_size": len(jobs),
        }
        for index, job in enumerate(jobs)
    ]


def generate_high_volume_state(
    *, model: Any, tokenizer: Any, state_id: str, state_identity_sha256: str,
    prompts: Sequence[Mapping[str, Any]], seeds: Sequence[int],
    recipes: Sequence[Mapping[str, Any]], output_dir: Path,
    device: torch.device | str, batch_size: int,
    maximum_batches: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate/resume the 2,880-output exploratory grid in fixed recipe batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if tuple(seeds) != EXPECTED_SEEDS or tuple(row["recipe_id"] for row in recipes) != EXPECTED_RECIPE_IDS:
        raise ValueError("high-volume grid differs from its frozen seed/recipe contract")
    if len(prompts) != 120:
        raise ValueError("high-volume generation requires exactly 120 prompts")
    output_dir.mkdir(parents=True, exist_ok=True)
    planned = [
        {"prompt": dict(prompt), "seed": int(seed), "recipe": dict(recipe)}
        for recipe in recipes for prompt in prompts for seed in seeds
    ]
    rows = []
    pending_by_recipe: dict[str, list[dict[str, Any]]] = {
        recipe_id: [] for recipe_id in EXPECTED_RECIPE_IDS
    }
    started = time.monotonic()
    executed_batches = 0
    for job in planned:
        path = output_dir / _output_name(state_id, job)
        if path.is_file():
            payload = _load_and_verify_output(
                path, state_id=state_id, state_identity_sha256=state_identity_sha256,
                job=job,
            )
            rows.append(_completion_row(path, payload))
        else:
            pending_by_recipe[str(job["recipe"]["recipe_id"])].append(job)
    stop = False
    for recipe_id in EXPECTED_RECIPE_IDS:
        pending = pending_by_recipe[recipe_id]
        for start in range(0, len(pending), batch_size):
            if maximum_batches is not None and executed_batches >= maximum_batches:
                stop = True
                break
            batch = pending[start : start + batch_size]
            results = generate_batch(
                model=model, tokenizer=tokenizer, jobs=batch,
                recipe=batch[0]["recipe"], device=device,
            )
            for job, result in zip(batch, results, strict=True):
                path = output_dir / _output_name(state_id, job)
                payload = {
                    "generation_version": HIGH_VOLUME_VERSION,
                    "analysis_role": "exploratory_high_volume",
                    "state_id": state_id,
                    "state_identity_sha256": state_identity_sha256,
                    "prompt": job["prompt"], "recipe": job["recipe"],
                    **result, "v7_test_accessed": False,
                }
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(path)
                verified = _load_and_verify_output(
                    path, state_id=state_id, state_identity_sha256=state_identity_sha256,
                    job=job,
                )
                rows.append(_completion_row(path, verified))
            executed_batches += 1
            if progress:
                completed = len(rows)
                elapsed = time.monotonic() - started
                progress(
                    f"batch={executed_batches} completed={completed}/2880 "
                    f"progress={100 * completed / 2880:.1f}% elapsed={elapsed:.1f}s "
                    f"eta={elapsed/max(completed,1)*(2880-completed):.1f}s"
                )
        if stop:
            break
    authoritative = len(rows) == 2880
    completion = {
        "generation_version": HIGH_VOLUME_VERSION,
        "analysis_role": "exploratory_high_volume",
        "state_id": state_id,
        "state_identity_sha256": state_identity_sha256,
        "prompt_count": 120, "seeds": list(seeds),
        "recipe_ids": [row["recipe_id"] for row in recipes],
        "planned_output_count": 2880, "completed_output_count": len(rows),
        "completion_scope": (
            "authoritative_120_prompt_8_seed_3_recipe_grid"
            if authoritative else "qualification_or_incomplete_prefix"
        ),
        "outputs": sorted(rows, key=lambda row: row["path"]),
        "elapsed_seconds": time.monotonic() - started,
        "v7_test_accessed": False,
        "causal_experiments_performed": False,
    }
    marker = "complete.json" if authoritative else "qualification_or_progress.json"
    (output_dir / marker).write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return completion


def _output_name(state_id: str, job: Mapping[str, Any]) -> str:
    key = (
        f"{HIGH_VOLUME_VERSION}|{state_id}|{job['prompt']['id']}|"
        f"{job['seed']}|{job['recipe']['recipe_id']}"
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20] + ".json"


def _load_and_verify_output(
    path: Path, *, state_id: str, state_identity_sha256: str,
    job: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed high-volume output: {path}") from error
    expected = {
        "generation_version": HIGH_VOLUME_VERSION,
        "analysis_role": "exploratory_high_volume",
        "state_id": state_id,
        "state_identity_sha256": state_identity_sha256,
        "prompt": job["prompt"], "seed": int(job["seed"]),
        "recipe": job["recipe"], "v7_test_accessed": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"high-volume output lineage mismatch: {key}")
    return payload


def _completion_row(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "prompt_id": payload["prompt"]["id"], "seed": payload["seed"],
        "recipe_id": payload["recipe"]["recipe_id"],
    }
