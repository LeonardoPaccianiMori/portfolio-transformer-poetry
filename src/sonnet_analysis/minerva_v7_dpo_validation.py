"""Matched validation generation for Stage 3 and its AI-judged DPO adapter."""

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


VALIDATION_VERSION = "minerva_7b_v7_dpo_matched_validation_v1"
SYSTEM_IDS = ("stage_3", "dpo")


def generate_matched_validation(
    *, model: Any, tokenizer: Any, prompts: Sequence[Mapping[str, Any]],
    seeds: Sequence[int], recipe: Mapping[str, Any], output_dir: Path,
    state_identity: str, adapter_identity: str, device: Any, batch_size: int,
    generation_version: str = VALIDATION_VERSION,
    analysis_role: str = "validation_only_dpo_model_selection",
    v7_test_accessed: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate a resumable paired grid with adapter disablement as the only change."""

    if not prompts or not seeds or batch_size <= 0:
        raise ValueError("DPO matched validation grid is empty or invalid")
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        {"prompt": dict(prompt), "seed": int(seed)}
        for prompt in prompts for seed in seeds
    ]
    started = time.monotonic()
    completed = []
    for system_id in SYSTEM_IDS:
        pending = [
            job for job in jobs
            if not (output_dir / _name(system_id, job, generation_version)).is_file()
        ]
        context = model.disable_adapter if system_id == "stage_3" else nullcontext
        with context():
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                results = generate_batch(
                    model=model, tokenizer=tokenizer, jobs=batch, recipe=recipe,
                    device=device,
                    prompt_builder=lambda tok, opening: build_intervention_prompt(
                        tok, opening, "explicit_no_labels_or_prose"
                    ),
                )
                for job, result in zip(batch, results, strict=True):
                    payload = {
                        "validation_version": generation_version,
                        "analysis_role": analysis_role,
                        "system_id": system_id,
                        "stage_3_state_identity_sha256": state_identity,
                        "dpo_adapter_identity_sha256": adapter_identity,
                        "prompt": job["prompt"], "recipe": dict(recipe),
                        **result,
                        "v7_test_accessed": v7_test_accessed,
                    }
                    _write_json_atomic(
                        output_dir / _name(system_id, job, generation_version), payload
                    )
                if progress is not None:
                    count = len(list(output_dir.glob("*.json")))
                    planned = len(jobs) * len(SYSTEM_IDS)
                    elapsed = time.monotonic() - started
                    progress(
                        f"system={system_id} completed={count}/{planned} "
                        f"progress={100 * count / planned:.1f}% elapsed={elapsed:.1f}s "
                        f"eta={elapsed / max(count, 1) * (planned - count):.1f}s"
                    )
    for system_id in SYSTEM_IDS:
        for job in jobs:
            path = output_dir / _name(system_id, job, generation_version)
            payload = _load_and_validate(
                path, system_id=system_id, job=job,
                state_identity=state_identity, adapter_identity=adapter_identity,
                recipe=recipe, generation_version=generation_version,
                analysis_role=analysis_role, v7_test_accessed=v7_test_accessed,
            )
            completed.append({
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "system_id": system_id,
                "prompt_id": payload["prompt"]["id"],
                "seed": payload["seed"],
            })
    result = {
        "validation_version": generation_version,
        "analysis_role": analysis_role,
        "stage_3_state_identity_sha256": state_identity,
        "dpo_adapter_identity_sha256": adapter_identity,
        "system_ids": list(SYSTEM_IDS),
        "prompt_count": len(prompts), "seeds": list(seeds),
        "planned_output_count": len(jobs) * len(SYSTEM_IDS),
        "completed_output_count": len(completed),
        "outputs": sorted(completed, key=lambda row: row["path"]),
        "v7_test_accessed": v7_test_accessed,
    }
    _write_json_atomic(output_dir / "complete.json", result)
    return result


def _name(
    system_id: str, job: Mapping[str, Any], generation_version: str = VALIDATION_VERSION
) -> str:
    key = (
        f"{generation_version}|{system_id}|{job['prompt']['id']}|{job['seed']}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:20] + ".json"


def _load_and_validate(
    path: Path, *, system_id: str, job: Mapping[str, Any], state_identity: str,
    adapter_identity: str, recipe: Mapping[str, Any],
    generation_version: str = VALIDATION_VERSION,
    analysis_role: str = "validation_only_dpo_model_selection",
    v7_test_accessed: bool = False,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "validation_version": generation_version,
        "analysis_role": analysis_role,
        "system_id": system_id,
        "stage_3_state_identity_sha256": state_identity,
        "dpo_adapter_identity_sha256": adapter_identity,
        "prompt": job["prompt"], "seed": int(job["seed"]),
        "recipe": dict(recipe), "v7_test_accessed": v7_test_accessed,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"DPO validation output lineage mismatch: {key}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
