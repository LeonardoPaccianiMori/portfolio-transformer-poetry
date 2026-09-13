"""Matched validation generation for the form-targeted LoRA pilot.

The baseline system is Stage-3 with its LoRA adapter disabled. The candidate
system is the same Stage-3 base with the pilot adapter enabled. Everything
else, including the prompt builder, the recipe, the openings, and the seeds,
is identical, so the paired comparison isolates the adapter.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from sonnet_analysis.minerva_v7_high_volume_generation import generate_batch
from sonnet_analysis.minerva_v7_prompt_intervention import build_intervention_prompt

GENERATION_VERSION = "form_targeted_lora_pilot_validation_v1"
CANDIDATE_GENERATION_VERSION = "v8_stage3_retrain_validation_v1"
SYSTEM_IDS = ("baseline", "candidate")


def output_name(
    system_id: str,
    prompt_id: str,
    seed: int,
    *,
    version: str = GENERATION_VERSION,
) -> str:
    key = f"{version}|{system_id}|{prompt_id}|{seed}"
    return hashlib.sha256(key.encode()).hexdigest()[:20] + ".json"


def generate_form_targeted_validation(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    recipe: Mapping[str, Any],
    output_dir: Path,
    adapter_identity: str,
    device: Any,
    batch_size: int,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not prompts or not seeds or batch_size <= 0:
        raise ValueError("validation grid is empty or invalid")
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        {"prompt": dict(prompt), "seed": int(seed)}
        for prompt in prompts
        for seed in seeds
    ]
    started = time.monotonic()
    for system_id in SYSTEM_IDS:
        pending = [
            job
            for job in jobs
            if not (output_dir / output_name(system_id, job["prompt"]["id"], job["seed"])).is_file()
        ]
        context = model.disable_adapter if system_id == "baseline" else nullcontext
        with context():
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                results = generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    jobs=batch,
                    recipe=recipe,
                    device=device,
                    prompt_builder=lambda tok, opening: build_intervention_prompt(
                        tok, opening, "explicit_no_labels_or_prose"
                    ),
                )
                for job, result in zip(batch, results, strict=True):
                    payload = {
                        "generation_version": GENERATION_VERSION,
                        "analysis_role": "form_targeted_lora_pilot_validation",
                        "system_id": system_id,
                        "adapter_identity_sha256": adapter_identity,
                        "prompt": job["prompt"],
                        "recipe": dict(recipe),
                        **result,
                        "v7_test_accessed": False,
                    }
                    _write_json_atomic(
                        output_dir
                        / output_name(system_id, job["prompt"]["id"], job["seed"]),
                        payload,
                    )
                if progress is not None:
                    count = len(list(output_dir.glob("*.json")))
                    planned = len(jobs) * len(SYSTEM_IDS)
                    elapsed = time.monotonic() - started
                    progress(
                        f"system={system_id} completed={count}/{planned} "
                        f"elapsed={elapsed:.1f}s"
                    )
    outputs = []
    for system_id in SYSTEM_IDS:
        for job in jobs:
            path = output_dir / output_name(system_id, job["prompt"]["id"], job["seed"])
            outputs.append(
                {
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "system_id": system_id,
                    "prompt_id": job["prompt"]["id"],
                    "seed": int(job["seed"]),
                }
            )
    result = {
        "generation_version": GENERATION_VERSION,
        "analysis_role": "form_targeted_lora_pilot_validation",
        "adapter_identity_sha256": adapter_identity,
        "system_ids": list(SYSTEM_IDS),
        "prompt_count": len(prompts),
        "seeds": list(seeds),
        "planned_output_count": len(jobs) * len(SYSTEM_IDS),
        "completed_output_count": len(outputs),
        "outputs": sorted(outputs, key=lambda row: row["path"]),
        "v7_test_accessed": False,
    }
    _write_json_atomic(output_dir / "complete.json", result)
    return result


def load_generation_records(generation_dir: Path) -> list[dict[str, Any]]:
    complete = json.loads((generation_dir / "complete.json").read_text(encoding="utf-8"))
    records = []
    for row in complete["outputs"]:
        path = generation_dir / row["path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("system_id") != row["system_id"]:
            raise ValueError(f"system mismatch in {row['path']}")
        records.append(
            {
                "path": row["path"],
                "prompt_id": row["prompt_id"],
                "seed": int(row["seed"]),
                "system_id": row["system_id"],
                "opening_line": payload.get("opening_line"),
                "text": payload["text"],
            }
        )
    if len(records) != int(complete["completed_output_count"]):
        raise ValueError("generation count does not match complete.json")
    return records


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def generate_single_system_validation(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    recipe: Mapping[str, Any],
    output_dir: Path,
    device: Any,
    batch_size: int,
    system_id: str = "candidate",
    generation_version: str = CANDIDATE_GENERATION_VERSION,
    analysis_role: str = "v8_stage3_retrain_matched_validation",
    identity_sha256: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not prompts or not seeds or batch_size <= 0:
        raise ValueError("validation grid is empty or invalid")
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        {"prompt": dict(prompt), "seed": int(seed)}
        for prompt in prompts
        for seed in seeds
    ]
    started = time.monotonic()
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        pending = [
            job
            for job in batch
            if not (
                output_dir
                / output_name(
                    system_id, job["prompt"]["id"], job["seed"], version=generation_version
                )
            ).is_file()
        ]
        if not pending:
            continue
        results = generate_batch(
            model=model,
            tokenizer=tokenizer,
            jobs=pending,
            recipe=recipe,
            device=device,
            prompt_builder=lambda tok, opening: build_intervention_prompt(
                tok, opening, "explicit_no_labels_or_prose"
            ),
        )
        for job, result in zip(pending, results, strict=True):
            payload = {
                "generation_version": generation_version,
                "analysis_role": analysis_role,
                "system_id": system_id,
                "identity_sha256": identity_sha256,
                "prompt": job["prompt"],
                "recipe": dict(recipe),
                **result,
                "v7_test_accessed": False,
            }
            _write_json_atomic(
                output_dir
                / output_name(
                    system_id, job["prompt"]["id"], job["seed"], version=generation_version
                ),
                payload,
            )
        if progress is not None:
            completed = len(list(output_dir.glob("*.json")))
            elapsed = time.monotonic() - started
            progress(f"completed={completed}/{len(jobs)} elapsed={elapsed:.1f}s")
    outputs = []
    for job in jobs:
        path = output_dir / output_name(
            system_id, job["prompt"]["id"], job["seed"], version=generation_version
        )
        outputs.append(
            {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "system_id": system_id,
                "prompt_id": job["prompt"]["id"],
                "seed": int(job["seed"]),
            }
        )
    result = {
        "generation_version": generation_version,
        "analysis_role": analysis_role,
        "identity_sha256": identity_sha256,
        "system_ids": [system_id],
        "prompt_count": len(prompts),
        "seeds": list(seeds),
        "planned_output_count": len(jobs),
        "completed_output_count": len(outputs),
        "outputs": sorted(outputs, key=lambda row: row["path"]),
        "v7_test_accessed": False,
    }
    _write_json_atomic(output_dir / "complete.json", result)
    return result


def merge_matched_generation(
    *, baseline_dir: Path, candidate_dir: Path, output_dir: Path
) -> dict[str, Any]:
    import shutil

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_dir, system_id in ((baseline_dir, "baseline"), (candidate_dir, "candidate")):
        complete = json.loads((source_dir / "complete.json").read_text(encoding="utf-8"))
        for row in complete["outputs"]:
            if row["system_id"] != system_id:
                raise ValueError(f"unexpected system in {source_dir.name}")
            if row["path"] in seen:
                raise ValueError(f"duplicate output name {row['path']}")
            seen.add(row["path"])
            shutil.copy2(source_dir / row["path"], output_dir / row["path"])
            rows.append(
                {
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "system_id": system_id,
                    "prompt_id": row["prompt_id"],
                    "seed": int(row["seed"]),
                }
            )
    if not rows:
        raise ValueError("no generation rows to merge")
    result = {
        "generation_version": "v8_stage3_retrain_validation_merged_v1",
        "analysis_role": "v8_stage3_retrain_matched_validation",
        "system_ids": ["baseline", "candidate"],
        "completed_output_count": len(rows),
        "outputs": sorted(rows, key=lambda row: row["path"]),
        "v7_test_accessed": False,
    }
    _write_json_atomic(output_dir / "complete.json", result)
    return result
