"""Resumable Stage-3 no-labels prompt plus creative-decoding experiment."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sonnet_analysis.minerva_v7_exploratory_prompts import (
    validate_exploratory_prompt_manifest,
)
from sonnet_analysis.minerva_v7_high_volume_generation import generate_batch
from sonnet_analysis.minerva_v7_prompt_intervention import (
    EXPECTED_PROMPT_SHA256,
    EXPECTED_SEEDS,
    EXPECTED_STATE_ID,
    build_intervention_prompt,
    score_attempt,
)


NO_LABELS_CREATIVE_VERSION = "minerva_7b_v7_stage_3_no_labels_creative_v1"
PROMPT_ARM_ID = "explicit_no_labels_or_prose"
RECIPE_ID = "creative"
EXPECTED_RECIPE = {
    "recipe_id": RECIPE_ID,
    "temperature": 0.85,
    "top_p": 0.95,
    "top_k": None,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 4,
    "max_new_tokens": 512,
    "continuation_line_target": 13,
}


def load_no_labels_creative_config(path: Path) -> dict[str, Any]:
    """Load and fail closed on the approved one-cell experiment contract."""

    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "experiment_version": NO_LABELS_CREATIVE_VERSION,
        "state_id": EXPECTED_STATE_ID,
        "prompt_manifest_sha256": EXPECTED_PROMPT_SHA256,
        "prompt_count": 120,
        "prompt_arm_id": PROMPT_ARM_ID,
        "sampling_recipe": EXPECTED_RECIPE,
        "final_outputs": 960,
        "qualification_output_count": 8,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"no-labels/creative contract mismatch: {key}")
    if tuple(config.get("seeds", [])) != EXPECTED_SEEDS:
        raise ValueError("no-labels/creative seed contract mismatch")
    authorization = config.get("authorization", {})
    if (
        authorization.get("post_hoc_experiment") is not True
        or authorization.get("user_approved") is not True
        or authorization.get("v7_test_access_authorized") is not False
        or authorization.get("training_authorized") is not False
        or authorization.get("instance_lifecycle_action_authorized") is not False
    ):
        raise PermissionError("no-labels/creative authorization boundary changed")
    return config


def load_no_labels_creative_prompts(
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Load the existing validation-only prompt grid without touching V7 test."""

    manifest = validate_exploratory_prompt_manifest(
        Path(str(config["prompt_manifest_path"])),
        expected_sha256=str(config["prompt_manifest_sha256"]),
    )
    return [dict(row) for row in manifest["prompts"]]


def generate_no_labels_creative(
    *,
    model: Any,
    tokenizer: Any,
    state_identity_sha256: str,
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    output_dir: Path,
    device: Any,
    batch_size: int,
    maximum_outputs: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate or resume the frozen 120-prompt by eight-seed experiment cell."""

    if len(prompts) != 120 or tuple(config["seeds"]) != EXPECTED_SEEDS:
        raise ValueError("no-labels/creative prompt/seed grid mismatch")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if maximum_outputs is not None and maximum_outputs <= 0:
        raise ValueError("maximum_outputs must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        {"prompt": dict(prompt), "seed": int(seed)}
        for prompt in prompts
        for seed in config["seeds"]
    ]
    recipe = dict(config["sampling_recipe"])
    pending = [
        job for job in jobs if not (output_dir / _output_name(job)).is_file()
    ]
    if maximum_outputs is not None:
        pending = pending[:maximum_outputs]

    started = time.monotonic()
    newly_completed = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        results = generate_batch(
            model=model,
            tokenizer=tokenizer,
            jobs=batch,
            recipe=recipe,
            device=device,
            prompt_builder=lambda selected_tokenizer, opening: build_intervention_prompt(
                selected_tokenizer, opening, PROMPT_ARM_ID
            ),
        )
        for job, result in zip(batch, results, strict=True):
            scored = score_attempt(result)
            payload = {
                "experiment_version": NO_LABELS_CREATIVE_VERSION,
                "analysis_role": "post_hoc_stage_3_combined_prompt_decoding_experiment",
                "state_id": EXPECTED_STATE_ID,
                "state_identity_sha256": state_identity_sha256,
                "prompt_manifest_sha256": EXPECTED_PROMPT_SHA256,
                "prompt_arm_id": PROMPT_ARM_ID,
                "prompt": dict(job["prompt"]),
                "seed": int(job["seed"]),
                "sampling_recipe": recipe,
                **scored,
                "v7_test_accessed": False,
                "training_performed": False,
            }
            _write_output(output_dir / _output_name(job), payload)
            newly_completed += 1
        if progress is not None:
            completed = _count_outputs(output_dir)
            elapsed = time.monotonic() - started
            progress(
                f"completed={completed}/960 progress={100 * completed / 960:.1f}% "
                f"elapsed={elapsed:.1f}s "
                f"eta={elapsed / max(newly_completed, 1) * (960 - completed):.1f}s"
            )

    rows = []
    for job in jobs:
        path = output_dir / _output_name(job)
        if path.is_file():
            payload = _load_and_verify_output(
                path,
                state_identity_sha256=state_identity_sha256,
                job=job,
                recipe=recipe,
            )
            rows.append(
                {
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "prompt_id": payload["prompt"]["id"],
                    "seed": payload["seed"],
                    "surface_screen_pass": payload["surface_diagnostics"][
                        "surface_screen_pass"
                    ],
                }
            )
    authoritative = len(rows) == int(config["final_outputs"])
    completion = {
        "experiment_version": NO_LABELS_CREATIVE_VERSION,
        "analysis_role": "post_hoc_stage_3_combined_prompt_decoding_experiment",
        "state_id": EXPECTED_STATE_ID,
        "state_identity_sha256": state_identity_sha256,
        "prompt_manifest_sha256": EXPECTED_PROMPT_SHA256,
        "prompt_arm_id": PROMPT_ARM_ID,
        "recipe_id": RECIPE_ID,
        "planned_output_count": int(config["final_outputs"]),
        "completed_output_count": len(rows),
        "completion_scope": (
            "authoritative_120_prompt_8_seed_cell"
            if authoritative
            else "qualification_or_incomplete_prefix"
        ),
        "outputs": rows,
        "v7_test_accessed": False,
        "training_performed": False,
    }
    marker = "complete.json" if authoritative else "qualification_or_progress.json"
    (output_dir / marker).write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return completion


def _output_name(job: Mapping[str, Any]) -> str:
    key = (
        f"{NO_LABELS_CREATIVE_VERSION}|{EXPECTED_STATE_ID}|{PROMPT_ARM_ID}|"
        f"{RECIPE_ID}|{job['prompt']['id']}|{job['seed']}"
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20] + ".json"


def _write_output(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_and_verify_output(
    path: Path,
    *,
    state_identity_sha256: str,
    job: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed no-labels/creative output: {path}") from error
    expected = {
        "experiment_version": NO_LABELS_CREATIVE_VERSION,
        "state_id": EXPECTED_STATE_ID,
        "state_identity_sha256": state_identity_sha256,
        "prompt_manifest_sha256": EXPECTED_PROMPT_SHA256,
        "prompt_arm_id": PROMPT_ARM_ID,
        "prompt": dict(job["prompt"]),
        "seed": int(job["seed"]),
        "sampling_recipe": dict(recipe),
        "v7_test_accessed": False,
        "training_performed": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"no-labels/creative output lineage mismatch: {key}")
    return payload


def _count_outputs(output_dir: Path) -> int:
    return sum(
        path.name not in {"complete.json", "qualification_or_progress.json"}
        for path in output_dir.glob("*.json")
    )
