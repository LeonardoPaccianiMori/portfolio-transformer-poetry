"""Bounded Stage-3 prompt and deterministic-retry quality experiment."""

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
from sonnet_analysis.minerva_v7_quality import generated_sonnet_surface_diagnostics
from sonnet_evaluation.metrics import score_generated_text


PROMPT_INTERVENTION_VERSION = "minerva_7b_v7_stage_3_prompt_intervention_v1"
EXPECTED_PROMPT_SHA256 = (
    "2f33aa518aa61c11193831e53b07fd3bd861a72bf68bb23c0e0e5b1a13b1d0c7"
)
EXPECTED_STATE_ID = "stage_3_selected"
EXPECTED_SEEDS = tuple(range(4200, 4208))
EXPECTED_ARM_IDS = (
    "current_prompt_control",
    "explicit_no_labels_or_prose",
    "explicit_4433_structure",
    "structured_with_deterministic_retries",
)


def load_prompt_intervention_config(path: Path) -> dict[str, Any]:
    """Load and fail closed on the preregistered intervention contract."""

    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "experiment_version": PROMPT_INTERVENTION_VERSION,
        "state_id": EXPECTED_STATE_ID,
        "prompt_manifest_sha256": EXPECTED_PROMPT_SHA256,
        "prompt_count": 120,
        "final_outputs": 3840,
        "maximum_attempts_per_retry_output": 3,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"prompt-intervention contract mismatch: {key}")
    if tuple(config.get("seeds", [])) != EXPECTED_SEEDS:
        raise ValueError("prompt-intervention seed contract mismatch")
    if tuple(row.get("arm_id") for row in config.get("arms", [])) != EXPECTED_ARM_IDS:
        raise ValueError("prompt-intervention arm contract mismatch")
    recipe = config.get("sampling_recipe", {})
    expected_recipe = {
        "temperature": 0.7,
        "top_p": 0.92,
        "top_k": None,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 4,
        "max_new_tokens": 512,
        "continuation_line_target": 13,
    }
    if recipe != expected_recipe:
        raise ValueError("prompt-intervention sampling contract mismatch")
    authorization = config.get("authorization", {})
    if (
        authorization.get("post_hoc_experiment") is not True
        or authorization.get("autonomous_workflow_active") is not True
        or authorization.get("v7_test_access_authorized") is not False
        or authorization.get("training_authorized") is not False
        or authorization.get("instance_lifecycle_action_authorized") is not False
    ):
        raise PermissionError("prompt-intervention authorization boundary changed")
    return config


def build_intervention_prompt(tokenizer: Any, opening_line: str, arm_id: str) -> str:
    """Render one frozen prompt arm and preserve the exact opening prefill."""

    if not opening_line.strip() or "\n" in opening_line or "\r" in opening_line:
        raise ValueError("opening_line must contain exactly one non-empty line")
    common = (
        "Componi un sonetto in italiano classico di esattamente quattordici "
        "versi. Usa come primo verso esattamente quello indicato, mantieni un "
        "tema coerente e una sintassi grammaticale, ed evita ripetizioni. "
    )
    if arm_id == "current_prompt_control":
        instruction = (
            common
            + "Restituisci soltanto il sonetto, senza titolo, spiegazioni o "
            "commenti."
        )
    elif arm_id == "explicit_no_labels_or_prose":
        instruction = (
            common
            + "Restituisci esclusivamente i quattordici versi poetici: non "
            "scrivere titolo, introduzione, spiegazione, commento, numeri, "
            "etichette come 'Primo verso' o testo in prosa."
        )
    elif arm_id in {
        "explicit_4433_structure",
        "structured_with_deterministic_retries",
    }:
        instruction = (
            common
            + "Organizza i quattordici versi nella struttura 4+4+3+3: due "
            "quartine seguite da due terzine. Restituisci esclusivamente i "
            "versi poetici, senza titolo, introduzione, spiegazione, commento, "
            "numeri, etichette dei versi o testo in prosa."
        )
    else:
        raise ValueError(f"unknown prompt-intervention arm: {arm_id}")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{instruction}\n\nPrimo verso: {opening_line}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("Minerva chat template must render a non-empty string")
    return f"{rendered}{opening_line}\n"


def retry_seed(base_seed: int, attempt_index: int, stride: int) -> int:
    """Return the preregistered deterministic RNG seed for one attempt."""

    if attempt_index < 0 or stride <= 0:
        raise ValueError("attempt_index must be non-negative and stride positive")
    return int(base_seed) + attempt_index * stride


def score_attempt(result: Mapping[str, Any]) -> dict[str, Any]:
    """Attach the literal retry screen; it is not a poetic-quality judgment."""

    metrics = score_generated_text(str(result["text"]), str(result["opening_line"]))
    diagnostics = generated_sonnet_surface_diagnostics(
        str(result["text"]),
        non_empty_line_count=int(metrics["non_empty_line_count"]),
        repetition_ratio=float(metrics["repetition_ratio"]),
    )
    return {**dict(result), "metrics": metrics, "surface_diagnostics": diagnostics}


def generate_prompt_intervention(
    *,
    model: Any,
    tokenizer: Any,
    state_identity_sha256: str,
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    output_dir: Path,
    device: Any,
    batch_size: int,
    maximum_batches_per_arm: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate or resume all four arms, reusing arm 3 as arm 4 attempt zero."""

    if len(prompts) != 120 or tuple(config["seeds"]) != EXPECTED_SEEDS:
        raise ValueError("prompt-intervention prompt/seed grid mismatch")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    recipe = dict(config["sampling_recipe"])
    arms = [str(row["arm_id"]) for row in config["arms"]]
    jobs = [
        {"prompt": dict(prompt), "seed": int(seed)}
        for prompt in prompts
        for seed in config["seeds"]
    ]
    started = time.monotonic()
    completed: list[dict[str, Any]] = []
    executed_batches = 0

    for arm_id in arms[:3]:
        pending = [
            job for job in jobs
            if not (output_dir / _output_name(arm_id, job)).is_file()
        ]
        arm_batches = 0
        for start in range(0, len(pending), batch_size):
            if maximum_batches_per_arm is not None and arm_batches >= maximum_batches_per_arm:
                break
            batch = pending[start : start + batch_size]
            results = generate_batch(
                model=model,
                tokenizer=tokenizer,
                jobs=batch,
                recipe=recipe,
                device=device,
                prompt_builder=lambda tok, opening, selected=arm_id: (
                    build_intervention_prompt(tok, opening, selected)
                ),
            )
            for job, result in zip(batch, results, strict=True):
                attempt = score_attempt(result)
                payload = _payload(
                    arm_id=arm_id,
                    state_identity_sha256=state_identity_sha256,
                    job=job,
                    recipe=recipe,
                    attempts=[attempt],
                    selected_attempt_index=0,
                )
                _write_output(output_dir / _output_name(arm_id, job), payload)
            arm_batches += 1
            executed_batches += 1
            _report_progress(progress, arm_id, output_dir, started, executed_batches)

    retry_arm = EXPECTED_ARM_IDS[-1]
    pending_retry = [
        job for job in jobs
        if not (output_dir / _output_name(retry_arm, job)).is_file()
        and (output_dir / _output_name("explicit_4433_structure", job)).is_file()
    ]
    retry_batches = 0
    for start in range(0, len(pending_retry), batch_size):
        if maximum_batches_per_arm is not None and retry_batches >= maximum_batches_per_arm:
            break
        batch = pending_retry[start : start + batch_size]
        attempts_by_job: list[list[dict[str, Any]]] = []
        unresolved: list[tuple[int, Mapping[str, Any]]] = []
        for index, job in enumerate(batch):
            source = _load_output(
                output_dir / _output_name("explicit_4433_structure", job)
            )
            first = dict(source["attempts"][0])
            attempts_by_job.append([first])
            if not first["surface_diagnostics"]["surface_screen_pass"]:
                unresolved.append((index, job))
        for attempt_index in range(1, int(config["maximum_attempts_per_retry_output"])):
            if not unresolved:
                break
            attempt_jobs = [
                {
                    "prompt": dict(job["prompt"]),
                    "seed": retry_seed(
                        int(job["seed"]), attempt_index, int(config["retry_seed_stride"])
                    ),
                }
                for _, job in unresolved
            ]
            results = generate_batch(
                model=model,
                tokenizer=tokenizer,
                jobs=attempt_jobs,
                recipe=recipe,
                device=device,
                prompt_builder=lambda tok, opening: build_intervention_prompt(
                    tok, opening, retry_arm
                ),
            )
            still_unresolved = []
            for (batch_index, original_job), result in zip(unresolved, results, strict=True):
                attempt = score_attempt(result)
                attempts_by_job[batch_index].append(attempt)
                if not attempt["surface_diagnostics"]["surface_screen_pass"]:
                    still_unresolved.append((batch_index, original_job))
            unresolved = still_unresolved
        for job, attempts in zip(batch, attempts_by_job, strict=True):
            selected_index = next(
                (
                    index for index, attempt in enumerate(attempts)
                    if attempt["surface_diagnostics"]["surface_screen_pass"]
                ),
                len(attempts) - 1,
            )
            payload = _payload(
                arm_id=retry_arm,
                state_identity_sha256=state_identity_sha256,
                job=job,
                recipe=recipe,
                attempts=attempts,
                selected_attempt_index=selected_index,
            )
            _write_output(output_dir / _output_name(retry_arm, job), payload)
        retry_batches += 1
        executed_batches += 1
        _report_progress(progress, retry_arm, output_dir, started, executed_batches)

    for arm_id in arms:
        for job in jobs:
            path = output_dir / _output_name(arm_id, job)
            if path.is_file():
                payload = _load_and_verify_output(
                    path,
                    arm_id=arm_id,
                    state_identity_sha256=state_identity_sha256,
                    job=job,
                    recipe=recipe,
                )
                completed.append(_completion_row(path, payload))
    authoritative = len(completed) == int(config["final_outputs"])
    completion = {
        "experiment_version": PROMPT_INTERVENTION_VERSION,
        "analysis_role": "post_hoc_stage_3_prompt_and_stopping_experiment",
        "state_id": EXPECTED_STATE_ID,
        "state_identity_sha256": state_identity_sha256,
        "prompt_manifest_sha256": EXPECTED_PROMPT_SHA256,
        "prompt_count": len(prompts),
        "seeds": list(config["seeds"]),
        "arm_ids": arms,
        "planned_final_output_count": int(config["final_outputs"]),
        "completed_final_output_count": len(completed),
        "completion_scope": (
            "authoritative_120_prompt_8_seed_4_arm_grid"
            if authoritative else "qualification_or_incomplete_prefix"
        ),
        "outputs": sorted(completed, key=lambda row: row["path"]),
        "v7_test_accessed": False,
        "training_performed": False,
        "surface_retry_is_not_poetic_quality_selection": True,
    }
    marker = "complete.json" if authoritative else "qualification_or_progress.json"
    (output_dir / marker).write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return completion


def load_experiment_prompts(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = validate_exploratory_prompt_manifest(
        Path(str(config["prompt_manifest_path"])),
        expected_sha256=str(config["prompt_manifest_sha256"]),
    )
    return [dict(row) for row in manifest["prompts"]]


def _payload(
    *, arm_id: str, state_identity_sha256: str, job: Mapping[str, Any],
    recipe: Mapping[str, Any], attempts: Sequence[Mapping[str, Any]],
    selected_attempt_index: int,
) -> dict[str, Any]:
    selected = dict(attempts[selected_attempt_index])
    return {
        "experiment_version": PROMPT_INTERVENTION_VERSION,
        "analysis_role": "post_hoc_stage_3_prompt_and_stopping_experiment",
        "state_id": EXPECTED_STATE_ID,
        "state_identity_sha256": state_identity_sha256,
        "prompt_manifest_sha256": EXPECTED_PROMPT_SHA256,
        "arm_id": arm_id,
        "prompt": dict(job["prompt"]),
        "base_seed": int(job["seed"]),
        "sampling_recipe": dict(recipe),
        "attempts": [dict(row) for row in attempts],
        "selected_attempt_index": selected_attempt_index,
        "text": selected["text"],
        "opening_line": selected["opening_line"],
        "metrics": selected["metrics"],
        "surface_diagnostics": selected["surface_diagnostics"],
        "v7_test_accessed": False,
        "training_performed": False,
    }


def _output_name(arm_id: str, job: Mapping[str, Any]) -> str:
    key = (
        f"{PROMPT_INTERVENTION_VERSION}|{EXPECTED_STATE_ID}|{arm_id}|"
        f"{job['prompt']['id']}|{job['seed']}"
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20] + ".json"


def _write_output(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_output(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed prompt-intervention output: {path}") from error


def _load_and_verify_output(
    path: Path, *, arm_id: str, state_identity_sha256: str,
    job: Mapping[str, Any], recipe: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _load_output(path)
    expected = {
        "experiment_version": PROMPT_INTERVENTION_VERSION,
        "state_id": EXPECTED_STATE_ID,
        "state_identity_sha256": state_identity_sha256,
        "prompt_manifest_sha256": EXPECTED_PROMPT_SHA256,
        "arm_id": arm_id,
        "prompt": dict(job["prompt"]),
        "base_seed": int(job["seed"]),
        "sampling_recipe": dict(recipe),
        "v7_test_accessed": False,
        "training_performed": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"prompt-intervention output lineage mismatch: {key}")
    attempts = payload.get("attempts", [])
    selected = payload.get("selected_attempt_index")
    if not attempts or not isinstance(selected, int) or not 0 <= selected < len(attempts):
        raise ValueError("prompt-intervention attempt lineage is invalid")
    return payload


def _completion_row(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "arm_id": payload["arm_id"],
        "prompt_id": payload["prompt"]["id"],
        "base_seed": payload["base_seed"],
        "attempt_count": len(payload["attempts"]),
    }


def _report_progress(
    progress: Callable[[str], None] | None,
    arm_id: str,
    output_dir: Path,
    started: float,
    executed_batches: int,
) -> None:
    if progress is None:
        return
    completed = sum(
        1 for path in output_dir.glob("*.json")
        if path.name not in {"complete.json", "qualification_or_progress.json"}
    )
    elapsed = time.monotonic() - started
    progress(
        f"arm={arm_id} batch={executed_batches} completed={completed}/3840 "
        f"progress={100 * completed / 3840:.1f}% elapsed={elapsed:.1f}s"
    )
