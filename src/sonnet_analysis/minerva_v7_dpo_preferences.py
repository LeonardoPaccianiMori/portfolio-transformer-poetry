"""Bounded, training-only preference-data preparation for Minerva V7 DPO.

This module prepares evidence and preference pairs.  It never trains a model,
never reads the sealed V7 test split, and keeps raw candidates even when a
deterministic screen rejects them.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sonnet_analysis.minerva_v7_memorization import (
    EXPECTED_INDEX_SHA256,
    EXPECTED_SHARD_SHA256,
    EXPECTED_TOKENIZER_SHA256,
    load_verified_sonnet_train_reference,
)
from sonnet_analysis.minerva_v7_quality import generated_sonnet_surface_diagnostics
from sonnet_analysis.minerva_v7_high_volume_generation import generate_batch
from sonnet_analysis.minerva_v7_prompt_intervention import build_intervention_prompt
from sonnet_evaluation.metrics import score_generated_text


PREFERENCE_VERSION = "minerva_7b_v7_dpo_preferences_v1"
PROMPT_VERSION = "minerva_7b_v7_dpo_training_prompts_v1"
PROMPT_COUNT = 512
SELECTION_SEED = 10_937
PAIRING_SEED = 10_943
EXPECTED_TRAIN_DOCUMENTS = 19_899
EXPECTED_TRAIN_TOKENS = 3_551_021
EXPECTED_REFERENCE_MANIFEST_SHA256 = (
    "5a4223d00fd6e09604340ebbe8d24f2f90588dfd1aa86c7abb2165a8215d6ad8"
)
EXPECTED_REFERENCE_RECORDS_SHA256 = (
    "308cdefb6e5fba3d5e1170e2d3d90ee62f693a2f843348e268be0afeede07c76"
)
EXPECTED_VALIDATION_PROMPT_SHA256 = (
    "2f33aa518aa61c11193831e53b07fd3bd861a72bf68bb23c0e0e5b1a13b1d0c7"
)
STATE_ID = "stage_3_selected"
STATE_IDENTITY_SHA256 = (
    "478d5979e25a78375d7af0434db6a5432678762fac2d142af2d4798dda53a474"
)
RECIPE_IDS = ("no_labels_balanced", "no_labels_creative")
SEEDS = (7300, 7301, 7302, 7303)
EXPECTED_PROMPT_MANIFEST_SHA256 = (
    "45631fdecce259488e134cd67ef1d56f29f0176d6cc18eab90fd8b2b442703e8"
)


def load_preference_config(path: Path) -> dict[str, Any]:
    """Load the frozen candidate-data contract and fail closed on drift."""

    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "preference_version": PREFERENCE_VERSION,
        "state_id": STATE_ID,
        "state_identity_sha256": STATE_IDENTITY_SHA256,
        "prompt_manifest_sha256": EXPECTED_PROMPT_MANIFEST_SHA256,
        "prompt_count": PROMPT_COUNT,
        "candidate_count": 4096,
        "qualification_prompt_count": 8,
        "qualification_candidate_count": 64,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"DPO preference config mismatch: {key}")
    if tuple(config.get("seeds", [])) != SEEDS:
        raise ValueError("DPO preference seed contract mismatch")
    recipes = config.get("recipes", [])
    if tuple(row.get("recipe_id") for row in recipes) != RECIPE_IDS:
        raise ValueError("DPO preference recipe contract mismatch")
    expected_recipes = {
        "no_labels_balanced": {
            "recipe_id": "no_labels_balanced", "temperature": 0.7,
            "top_p": 0.92, "top_k": None, "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 4, "max_new_tokens": 512,
            "continuation_line_target": 13,
        },
        "no_labels_creative": {
            "recipe_id": "no_labels_creative", "temperature": 0.85,
            "top_p": 0.95, "top_k": None, "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 4, "max_new_tokens": 512,
            "continuation_line_target": 13,
        },
    }
    if {row["recipe_id"]: row for row in recipes} != expected_recipes:
        raise ValueError("DPO preference sampling recipes changed")
    authorization = config.get("authorization", {})
    if (
        authorization.get("user_approved_preference_data_experiment") is not True
        or authorization.get("candidate_generation_authorized") is not True
        or authorization.get("dpo_training_authorized") is not False
        or authorization.get("v7_test_access_authorized") is not False
        or authorization.get("instance_lifecycle_action_authorized") is not False
    ):
        raise PermissionError("DPO preference authorization boundary changed")
    return config


def generate_preference_candidates(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    output_dir: Path,
    device: Any,
    batch_size: int,
    maximum_candidates: int | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Generate or resume raw candidates; rejected evidence is never removed."""

    if batch_size <= 0:
        raise ValueError("DPO generation batch_size must be positive")
    jobs = build_candidate_jobs(prompts)
    recipes = {str(row["recipe_id"]): dict(row) for row in config["recipes"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = [job for job in jobs if not (output_dir / _candidate_name(job)).is_file()]
    if maximum_candidates is not None:
        if maximum_candidates <= 0:
            raise ValueError("maximum_candidates must be positive")
        pending = pending[:maximum_candidates]
    started = time.monotonic()
    newly_completed = 0
    for recipe_id in RECIPE_IDS:
        recipe_jobs = [row for row in pending if row["recipe_id"] == recipe_id]
        for start in range(0, len(recipe_jobs), batch_size):
            batch = recipe_jobs[start : start + batch_size]
            results = generate_batch(
                model=model,
                tokenizer=tokenizer,
                jobs=batch,
                recipe=recipes[recipe_id],
                device=device,
                prompt_builder=lambda tok, opening: build_intervention_prompt(
                    tok, opening, "explicit_no_labels_or_prose"
                ),
            )
            for job, result in zip(batch, results, strict=True):
                candidate_id = _candidate_id(job)
                payload = {
                    "preference_version": PREFERENCE_VERSION,
                    "analysis_role": "training_only_dpo_preference_candidate",
                    "candidate_id": candidate_id,
                    "state_id": STATE_ID,
                    "state_identity_sha256": STATE_IDENTITY_SHA256,
                    "prompt_manifest_sha256": EXPECTED_PROMPT_MANIFEST_SHA256,
                    "prompt_id": str(job["prompt"]["id"]),
                    "source_identity": str(job["prompt"]["source_identity"]),
                    "source_split": "sonnets_train",
                    "recipe_id": recipe_id,
                    "recipe": recipes[recipe_id],
                    **result,
                    "v7_test_accessed": False,
                    "training_performed": False,
                }
                write_json_atomic(output_dir / _candidate_name(job), payload)
                newly_completed += 1
            if progress:
                completed = _count_candidates(output_dir)
                elapsed = time.monotonic() - started
                progress(
                    f"completed={completed}/4096 progress={100 * completed / 4096:.1f}% "
                    f"elapsed={elapsed:.1f}s "
                    f"eta={elapsed / max(newly_completed, 1) * (4096 - completed):.1f}s"
                )

    rows = []
    for job in jobs:
        path = output_dir / _candidate_name(job)
        if path.is_file():
            payload = _load_candidate(path, job=job, recipe=recipes[job["recipe_id"]])
            rows.append(
                {
                    "candidate_id": payload["candidate_id"],
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "prompt_id": payload["prompt_id"],
                    "recipe_id": payload["recipe_id"],
                    "seed": payload["seed"],
                }
            )
    authoritative = len(rows) == 4096
    completion = {
        "preference_version": PREFERENCE_VERSION,
        "analysis_role": "training_only_dpo_preference_candidates",
        "state_id": STATE_ID,
        "state_identity_sha256": STATE_IDENTITY_SHA256,
        "prompt_manifest_sha256": EXPECTED_PROMPT_MANIFEST_SHA256,
        "planned_candidate_count": 4096,
        "completed_candidate_count": len(rows),
        "completion_scope": (
            "authoritative_512_prompt_2_recipe_4_seed_grid"
            if authoritative else "qualification_or_incomplete_prefix"
        ),
        "candidates": sorted(rows, key=lambda row: row["path"]),
        "generation_elapsed_seconds": time.monotonic() - started,
        "v7_test_accessed": False,
        "training_performed": False,
    }
    marker = "complete.json" if authoritative else "qualification_or_progress.json"
    write_json_atomic(output_dir / marker, completion)
    return completion


def load_verified_candidates(
    output_dir: Path, *, require_complete: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    marker = output_dir / ("complete.json" if require_complete else "qualification_or_progress.json")
    if not marker.is_file() and not require_complete:
        marker = output_dir / "complete.json"
    completion = json.loads(marker.read_text(encoding="utf-8"))
    if require_complete and completion.get("completed_candidate_count") != 4096:
        raise ValueError("DPO candidate grid is incomplete")
    rows = []
    for declared in completion.get("candidates", []):
        path = output_dir / str(declared["path"])
        if not path.is_file() or sha256_file(path) != declared["sha256"]:
            raise ValueError("DPO candidate evidence hash mismatch")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("candidate_id") != declared["candidate_id"]
            or payload.get("source_split") != "sonnets_train"
            or payload.get("v7_test_accessed") is not False
            or payload.get("training_performed") is not False
        ):
            raise ValueError("DPO candidate evidence lineage mismatch")
        rows.append(payload)
    return rows, completion


def analyze_preference_candidates(
    *,
    candidates: Sequence[Mapping[str, Any]],
    memorization_scores: Sequence[Mapping[str, Any]],
    generation_elapsed_seconds: float,
    hourly_rate_usd: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Screen candidates, build blind pairs, and project frozen abort gates."""

    if len(candidates) != len(memorization_scores) or not candidates:
        raise ValueError("DPO candidates and memorization scores must align")
    if generation_elapsed_seconds <= 0 or hourly_rate_usd <= 0:
        raise ValueError("DPO analysis requires positive measured time and hourly rate")
    assessments = [
        deterministic_candidate_screen(candidate, memorization=memorization)
        for candidate, memorization in zip(
            candidates, memorization_scores, strict=True
        )
    ]
    public_pairs, private_mapping = build_blinded_pairs(
        candidates=candidates, assessments=assessments
    )
    completion_pairs, completion_mapping = build_completion_contrast_pairs(
        candidates=candidates, assessments=assessments
    )
    completed = len(candidates)
    eligible = sum(row["eligible_for_blind_pairing"] for row in assessments)
    prompt_count = len({str(row["prompt_id"]) for row in candidates})
    projected_pairs = round(
        int(public_pairs["pair_count"]) / max(prompt_count, 1) * PROMPT_COUNT
    )
    projected_completion_pairs = round(
        int(completion_pairs["pair_count"]) / max(prompt_count, 1) * PROMPT_COUNT
    )
    projected_seconds = generation_elapsed_seconds / completed * 4096
    gates = evaluate_preference_gates(
        completed_candidates=completed,
        eligible_candidates=eligible,
        pair_count=projected_pairs,
        high_risk_memorization_count=sum(
            row["memorization"].get("risk_level") == "high"
            for row in assessments
        ),
        projected_full_minutes=projected_seconds / 60,
        projected_full_cost_usd=projected_seconds / 3600 * hourly_rate_usd,
    )
    rejection_counts = Counter(
        reason for row in assessments for reason in row["rejection_reasons"]
    )
    report = {
        "preference_version": PREFERENCE_VERSION,
        "analysis_role": "dpo_preference_candidate_qualification_or_analysis",
        "candidate_count": completed,
        "prompt_count": prompt_count,
        "eligible_candidate_count": eligible,
        "eligible_candidate_yield": eligible / completed,
        "observed_pair_count": int(public_pairs["pair_count"]),
        "projected_full_pair_count": projected_pairs,
        "observed_completion_contrast_pair_count": int(
            completion_pairs["pair_count"]
        ),
        "projected_full_completion_contrast_pair_count": (
            projected_completion_pairs
        ),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "measured_generation_seconds": generation_elapsed_seconds,
        "projected_full_generation_minutes": projected_seconds / 60,
        "projected_full_generation_cost_usd": (
            projected_seconds / 3600 * hourly_rate_usd
        ),
        "hourly_rate_usd": hourly_rate_usd,
        "gates": gates,
        "dpo_training_performed": False,
        "v7_test_accessed": False,
        "completion_contrast_pairs": completion_pairs,
        "completion_contrast_mapping_private": completion_mapping,
    }
    return report, assessments, public_pairs, private_mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> str:
    """Write canonical JSON atomically and return its byte identity."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return sha256_file(path)


def build_training_prompt_manifest(
    *,
    document_index_path: Path,
    reference_manifest_path: Path,
    validation_prompt_path: Path,
) -> dict[str, Any]:
    """Select 512 balanced openings from the exact frozen training pool."""

    if sha256_file(document_index_path) != EXPECTED_INDEX_SHA256:
        raise ValueError("DPO training document index hash mismatch")
    if sha256_file(reference_manifest_path) != EXPECTED_REFERENCE_MANIFEST_SHA256:
        raise ValueError("DPO decoded training-reference manifest hash mismatch")
    if sha256_file(validation_prompt_path) != EXPECTED_VALIDATION_PROMPT_SHA256:
        raise ValueError("DPO validation exclusion manifest hash mismatch")

    documents = _jsonl(document_index_path)
    records, reference = load_verified_sonnet_train_reference(reference_manifest_path)
    validation = json.loads(validation_prompt_path.read_text(encoding="utf-8"))
    if len(documents) != EXPECTED_TRAIN_DOCUMENTS or len(records) != len(documents):
        raise ValueError("DPO training source count mismatch")
    if reference.get("records_sha256") != EXPECTED_REFERENCE_RECORDS_SHA256:
        raise ValueError("DPO decoded training records identity mismatch")
    if validation.get("source_split") != "validation" or validation.get(
        "v7_test_accessed"
    ) is not False:
        raise ValueError("DPO validation exclusions have invalid lineage")

    forbidden_identities = {
        str(row["source_identity"]) for row in validation.get("prompts", [])
    }
    forbidden_hashes = {
        str(row["source_logical_sha256"]) for row in validation.get("prompts", [])
    }
    forbidden_openings = {
        _normalize_opening(str(row["opening_line"]))
        for row in validation.get("prompts", [])
    }
    candidates: list[dict[str, Any]] = []
    excluded_validation_openings = []
    for index, (document, record) in enumerate(zip(documents, records, strict=True)):
        _validate_training_row(document, record, index)
        opening = next(
            (line.strip() for line in str(record["text"]).splitlines() if line.strip()),
            "",
        )
        if not opening or "\n" in opening or "\r" in opening:
            raise ValueError("DPO training document lacks one usable opening")
        if (
            str(document["unit_id"]) in forbidden_identities
            or str(document["logical_sha256"]) in forbidden_hashes
        ):
            raise ValueError("DPO training candidate duplicates validation identity")
        if _normalize_opening(opening) in forbidden_openings:
            excluded_validation_openings.append(
                {
                    "source_identity": str(document["unit_id"]),
                    "source_logical_sha256": str(document["logical_sha256"]),
                    "opening_sha256": hashlib.sha256(
                        _normalize_opening(opening).encode("utf-8")
                    ).hexdigest(),
                }
            )
            continue
        candidates.append({**document, "opening_line": opening})

    selected = _balanced_selection(candidates, PROMPT_COUNT)
    prompts = []
    for row in selected:
        identity = hashlib.sha256(
            f"{PROMPT_VERSION}|{row['unit_id']}|{row['logical_sha256']}".encode()
        ).hexdigest()
        prompts.append(
            {
                "id": f"dpo_train_{identity[:16]}",
                "source_identity": str(row["unit_id"]),
                "source_split": "sonnets_train",
                "source_logical_sha256": str(row["logical_sha256"]),
                "author_key": str(row["author_key"]),
                "work_key": str(row["work_key"]),
                "period": str(row["epoch_key"]),
                "opening_line": str(row["opening_line"]),
            }
        )
    return {
        "prompt_version": PROMPT_VERSION,
        "preference_version": PREFERENCE_VERSION,
        "prompt_count": len(prompts),
        "selection_seed": SELECTION_SEED,
        "selection_policy": (
            "waterfill_period_quota_then_minimize_global_author_and_work_reuse"
        ),
        "source_pool_id": "sonnets_train",
        "source_split": "train",
        "source_document_count": EXPECTED_TRAIN_DOCUMENTS,
        "source_token_count": EXPECTED_TRAIN_TOKENS,
        "source_document_index_sha256": EXPECTED_INDEX_SHA256,
        "source_token_shard_sha256": EXPECTED_SHARD_SHA256,
        "source_reference_manifest_sha256": EXPECTED_REFERENCE_MANIFEST_SHA256,
        "source_reference_records_sha256": EXPECTED_REFERENCE_RECORDS_SHA256,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "validation_exclusion_manifest_sha256": EXPECTED_VALIDATION_PROMPT_SHA256,
        "period_counts": dict(sorted(Counter(row["period"] for row in prompts).items())),
        "unique_authors": len({row["author_key"] for row in prompts}),
        "maximum_prompts_per_author": max(
            Counter(row["author_key"] for row in prompts).values()
        ),
        "unique_works": len({row["work_key"] for row in prompts}),
        "maximum_prompts_per_work": max(
            Counter(row["work_key"] for row in prompts).values()
        ),
        "prompts": prompts,
        "excluded_validation_opening_count": len(excluded_validation_openings),
        "excluded_validation_openings": excluded_validation_openings,
        "selected_validation_prompt_overlap_count": 0,
        "v7_test_accessed": False,
        "training_performed": False,
    }


def validate_training_prompt_manifest(
    path: Path, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise ValueError("DPO training prompt manifest hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "prompt_version": PROMPT_VERSION,
        "preference_version": PREFERENCE_VERSION,
        "prompt_count": PROMPT_COUNT,
        "source_pool_id": "sonnets_train",
        "source_split": "train",
        "source_document_index_sha256": EXPECTED_INDEX_SHA256,
        "source_token_shard_sha256": EXPECTED_SHARD_SHA256,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "validation_exclusion_manifest_sha256": EXPECTED_VALIDATION_PROMPT_SHA256,
        "selected_validation_prompt_overlap_count": 0,
        "v7_test_accessed": False,
        "training_performed": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"DPO prompt contract mismatch: {key}")
    prompts = manifest.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != PROMPT_COUNT:
        raise ValueError("DPO training prompt rows are incomplete")
    seen: set[tuple[Any, ...]] = set()
    for row in prompts:
        if row.get("source_split") != "sonnets_train":
            raise ValueError("DPO prompt is not training-only")
        source_identity = str(row.get("source_identity", "")).casefold()
        if any(
            marker in source_identity
            for marker in ("sonnets_validation", "sonnets_test", "/validation/", "/test/")
        ):
            raise ValueError("DPO prompt contains held-out lineage")
        identity = (row.get("id"), row.get("source_identity"), row.get("source_logical_sha256"))
        if identity in seen:
            raise ValueError("duplicate DPO prompt identity")
        seen.add(identity)
        if not str(row.get("opening_line", "")).strip() or "\n" in row["opening_line"]:
            raise ValueError("invalid DPO opening line")
    return manifest


def build_candidate_jobs(prompts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the frozen two-recipe by four-seed candidate grid."""

    if len(prompts) != PROMPT_COUNT:
        raise ValueError("DPO candidate grid requires exactly 512 prompts")
    jobs = [
        {"prompt": dict(prompt), "recipe_id": recipe_id, "seed": seed}
        for prompt in prompts
        for recipe_id in RECIPE_IDS
        for seed in SEEDS
    ]
    if len(jobs) != 4096:
        raise AssertionError("DPO candidate grid size drifted")
    return jobs


def _candidate_id(job: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        f"{PREFERENCE_VERSION}|{STATE_ID}|{job['prompt']['id']}|"
        f"{job['recipe_id']}|{job['seed']}".encode()
    ).hexdigest()
    return f"candidate_{digest[:20]}"


def _candidate_name(job: Mapping[str, Any]) -> str:
    return f"{_candidate_id(job)}.json"


def _count_candidates(output_dir: Path) -> int:
    return sum(1 for path in output_dir.glob("candidate_*.json") if path.is_file())


def _load_candidate(
    path: Path, *, job: Mapping[str, Any], recipe: Mapping[str, Any]
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "preference_version": PREFERENCE_VERSION,
        "candidate_id": _candidate_id(job),
        "state_id": STATE_ID,
        "state_identity_sha256": STATE_IDENTITY_SHA256,
        "prompt_manifest_sha256": EXPECTED_PROMPT_MANIFEST_SHA256,
        "prompt_id": str(job["prompt"]["id"]),
        "source_identity": str(job["prompt"]["source_identity"]),
        "source_split": "sonnets_train",
        "recipe_id": str(job["recipe_id"]),
        "recipe": dict(recipe),
        "seed": int(job["seed"]),
        "v7_test_accessed": False,
        "training_performed": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"DPO candidate lineage mismatch: {key}")
    return payload


def deterministic_candidate_screen(
    candidate: Mapping[str, Any], *, memorization: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply predeclared hard filters while preserving every reason."""

    text = str(candidate.get("text", ""))
    opening = str(candidate.get("opening_line", ""))
    metrics = score_generated_text(text, opening)
    diagnostics = generated_sonnet_surface_diagnostics(
        text,
        non_empty_line_count=int(metrics["non_empty_line_count"]),
        repetition_ratio=float(metrics["repetition_ratio"]),
    )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated_line_collapse = len(lines) != len(set(lines))
    reasons = []
    checks = {
        "exact_opening": bool(metrics["prompt_preserved"]),
        "exact_fourteen_lines": int(metrics["non_empty_line_count"]) == 14,
        "meta_text_free": bool(diagnostics["meta_text_free"]),
        "complete_terminal_syntax": bool(diagnostics["ends_with_terminal_punctuation"]),
        "no_repeated_line_collapse": not repeated_line_collapse,
        "no_very_long_line": bool(diagnostics["no_line_at_or_above_120_characters"]),
        "below_repetition_threshold": bool(diagnostics["below_035_repetition_ratio"]),
        "no_high_risk_memorization": memorization.get("risk_level") != "high",
    }
    for name, passed in checks.items():
        if not passed:
            reasons.append(name)
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "prompt_id": str(candidate["prompt_id"]),
        "recipe_id": str(candidate["recipe_id"]),
        "seed": int(candidate["seed"]),
        "checks": checks,
        "eligible_for_blind_pairing": not reasons,
        "rejection_reasons": reasons,
        "metrics": metrics,
        "surface_diagnostics": diagnostics,
        "memorization": dict(memorization),
    }


def build_blinded_pairs(
    *,
    candidates: Sequence[Mapping[str, Any]],
    assessments: Sequence[Mapping[str, Any]],
    maximum_pairs_per_prompt: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build deterministic same-prompt A/B pairs and a separate private mapping."""

    if maximum_pairs_per_prompt <= 0:
        raise ValueError("maximum_pairs_per_prompt must be positive")
    by_id = {str(row["candidate_id"]): row for row in candidates}
    eligible: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in assessments:
        if row.get("eligible_for_blind_pairing") is True:
            candidate = by_id.get(str(row["candidate_id"]))
            if candidate is None or str(candidate["prompt_id"]) != str(row["prompt_id"]):
                raise ValueError("DPO candidate/assessment lineage mismatch")
            eligible[str(row["prompt_id"])].append(candidate)

    public_rows = []
    private_rows = []
    for prompt_id in sorted(eligible):
        rows = sorted(eligible[prompt_id], key=lambda row: str(row["candidate_id"]))
        cross_recipe = [
            (left, right)
            for left_index, left in enumerate(rows)
            for right in rows[left_index + 1 :]
            if left["recipe_id"] != right["recipe_id"]
        ]
        same_recipe = [
            (left, right)
            for left_index, left in enumerate(rows)
            for right in rows[left_index + 1 :]
            if left["recipe_id"] == right["recipe_id"]
        ]
        ranked = sorted(
            cross_recipe + same_recipe,
            key=lambda pair: hashlib.sha256(
                f"{PAIRING_SEED}|{pair[0]['candidate_id']}|{pair[1]['candidate_id']}".encode()
            ).hexdigest(),
        )
        used: set[str] = set()
        emitted = 0
        for left, right in ranked:
            identities = {str(left["candidate_id"]), str(right["candidate_id"])}
            if identities & used:
                continue
            pair_digest = hashlib.sha256(
                f"{PREFERENCE_VERSION}|{prompt_id}|{'|'.join(sorted(identities))}".encode()
            ).hexdigest()
            rng = random.Random(int(pair_digest[:16], 16) ^ PAIRING_SEED)
            a, b = (left, right) if rng.randrange(2) == 0 else (right, left)
            pair_id = f"pair_{pair_digest[:16]}"
            public_rows.append(
                {
                    "pair_id": pair_id,
                    "prompt_id": prompt_id,
                    "opening_line": str(a["opening_line"]),
                    "candidate_a": str(a["text"]),
                    "candidate_b": str(b["text"]),
                }
            )
            private_rows.append(
                {
                    "pair_id": pair_id,
                    "prompt_id": prompt_id,
                    "candidate_a_id": str(a["candidate_id"]),
                    "candidate_b_id": str(b["candidate_id"]),
                    "candidate_a_recipe_id": str(a["recipe_id"]),
                    "candidate_b_recipe_id": str(b["recipe_id"]),
                }
            )
            used.update(identities)
            emitted += 1
            if emitted >= maximum_pairs_per_prompt:
                break
    common = {
        "preference_version": PREFERENCE_VERSION,
        "pairing_seed": PAIRING_SEED,
        "pair_count": len(public_rows),
        "same_prompt_only": True,
        "v7_test_accessed": False,
        "training_performed": False,
    }
    return ({**common, "pairs": public_rows}, {**common, "mapping": private_rows})


def build_completion_contrast_pairs(
    *,
    candidates: Sequence[Mapping[str, Any]],
    assessments: Sequence[Mapping[str, Any]],
    maximum_pairs_per_prompt: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pair complete endings against otherwise-safe incomplete endings."""

    if maximum_pairs_per_prompt <= 0:
        raise ValueError("maximum completion pairs per prompt must be positive")
    by_id = {str(row["candidate_id"]): row for row in candidates}
    complete: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    incomplete: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    other_checks = (
        "exact_opening", "exact_fourteen_lines", "meta_text_free",
        "no_repeated_line_collapse", "no_very_long_line",
        "below_repetition_threshold", "no_high_risk_memorization",
    )
    for assessment in assessments:
        candidate = by_id.get(str(assessment["candidate_id"]))
        if candidate is None:
            raise ValueError("completion contrast lacks candidate evidence")
        checks = assessment.get("checks", {})
        if not all(checks.get(key) is True for key in other_checks):
            continue
        destination = (
            complete
            if checks.get("complete_terminal_syntax") is True
            else incomplete
        )
        destination[str(assessment["prompt_id"])].append(candidate)

    public_rows = []
    private_rows = []
    for prompt_id in sorted(set(complete) & set(incomplete)):
        ranked = sorted(
            (
                (good, bad)
                for good in complete[prompt_id]
                for bad in incomplete[prompt_id]
            ),
            key=lambda pair: hashlib.sha256(
                f"completion|{PAIRING_SEED}|{pair[0]['candidate_id']}|"
                f"{pair[1]['candidate_id']}".encode()
            ).hexdigest(),
        )
        used: set[str] = set()
        emitted = 0
        for good, bad in ranked:
            identities = {str(good["candidate_id"]), str(bad["candidate_id"])}
            if identities & used:
                continue
            pair_digest = hashlib.sha256(
                f"{PREFERENCE_VERSION}|completion|{prompt_id}|"
                f"{'|'.join(sorted(identities))}".encode()
            ).hexdigest()
            good_is_a = random.Random(
                int(pair_digest[:16], 16) ^ PAIRING_SEED
            ).randrange(2) == 0
            a, b = (good, bad) if good_is_a else (bad, good)
            pair_id = f"completion_pair_{pair_digest[:16]}"
            public_rows.append(
                {
                    "pair_id": pair_id,
                    "pair_type": "terminal_completion_contrast",
                    "prompt_id": prompt_id,
                    "opening_line": str(a["opening_line"]),
                    "candidate_a": str(a["text"]),
                    "candidate_b": str(b["text"]),
                    "judge_instruction": (
                        "Prefer the poem with genuinely complete final syntax and "
                        "coherent closure; do not reward punctuation alone. Also "
                        "consider grammar, progression, register, poetry, form, "
                        "and volta."
                    ),
                }
            )
            private_rows.append(
                {
                    "pair_id": pair_id,
                    "pair_type": "terminal_completion_contrast",
                    "prompt_id": prompt_id,
                    "candidate_a_id": str(a["candidate_id"]),
                    "candidate_b_id": str(b["candidate_id"]),
                    "candidate_a_recipe_id": str(a["recipe_id"]),
                    "candidate_b_recipe_id": str(b["recipe_id"]),
                    "expected_complete_side": "A" if good_is_a else "B",
                    "automatic_expected_side_is_not_a_preference_label": True,
                }
            )
            used.update(identities)
            emitted += 1
            if emitted >= maximum_pairs_per_prompt:
                break
    common = {
        "preference_version": PREFERENCE_VERSION,
        "pair_type": "terminal_completion_contrast",
        "pairing_seed": PAIRING_SEED,
        "pair_count": len(public_rows),
        "same_prompt_only": True,
        "automatic_completion_side_requires_blinded_judge_confirmation": True,
        "v7_test_accessed": False,
        "training_performed": False,
    }
    return ({**common, "pairs": public_rows}, {**common, "mapping": private_rows})


def evaluate_preference_gates(
    *,
    completed_candidates: int,
    eligible_candidates: int,
    pair_count: int,
    high_risk_memorization_count: int,
    projected_full_minutes: float,
    projected_full_cost_usd: float,
    votes: Sequence[Mapping[str, Any]] | None = None,
    user_calibration_accuracy: float | None = None,
) -> dict[str, Any]:
    """Evaluate frozen qualification and judge gates without authorizing training."""

    yield_rate = eligible_candidates / completed_candidates if completed_candidates else 0.0
    checks: dict[str, bool | None] = {
        "all_qualification_candidates_complete": completed_candidates >= 64,
        "eligible_candidate_yield_at_least_005": yield_rate >= 0.05,
        "projected_pair_count_at_least_96": pair_count >= 96,
        "zero_high_risk_memorization": high_risk_memorization_count == 0,
        "projected_generation_at_most_40_minutes": projected_full_minutes <= 40.0,
        "projected_generation_cost_at_most_1_60_usd": projected_full_cost_usd <= 1.60,
        "three_votes_per_pair": None,
        "majority_non_tie_rate_at_least_070": None,
        "user_calibration_accuracy_at_least_080": None,
    }
    if votes is not None:
        counts = Counter(str(row["pair_id"]) for row in votes)
        checks["three_votes_per_pair"] = bool(counts) and all(value >= 3 for value in counts.values())
        grouped: dict[str, Counter[str]] = defaultdict(Counter)
        for row in votes:
            grouped[str(row["pair_id"])][str(row["preference"])] += 1
        decisive = sum(
            max(counter.get("A", 0), counter.get("B", 0)) >= 2
            for counter in grouped.values()
        )
        checks["majority_non_tie_rate_at_least_070"] = (
            decisive / len(grouped) >= 0.70 if grouped else False
        )
    if user_calibration_accuracy is not None:
        checks["user_calibration_accuracy_at_least_080"] = (
            user_calibration_accuracy >= 0.80
        )
    ready = bool(checks) and all(value is True for value in checks.values())
    return {
        "preference_version": PREFERENCE_VERSION,
        "thresholds_are_preregistered": True,
        "completed_candidates": completed_candidates,
        "eligible_candidates": eligible_candidates,
        "eligible_candidate_yield": yield_rate,
        "pair_count": pair_count,
        "checks": checks,
        "dpo_training_gate_passed": ready,
        "dpo_training_authorized": False,
        "v7_test_accessed": False,
    }


def validate_vote(row: Mapping[str, Any]) -> dict[str, Any]:
    required_scores = ("grammar", "coherence", "historical_register", "poetic_force", "form", "volta_closure")
    if row.get("preference") not in {"A", "B", "tie"}:
        raise ValueError("DPO vote preference must be A, B, or tie")
    pair_id = str(row.get("pair_id", ""))
    if (
        not pair_id.startswith(("pair_", "completion_pair_"))
        or not str(row.get("judge_id", "")).strip()
    ):
        raise ValueError("DPO vote lacks pair or judge identity")
    scores = row.get("scores")
    if not isinstance(scores, Mapping):
        raise ValueError("DPO vote lacks rubric scores")
    for candidate_side in ("A", "B"):
        side = scores.get(candidate_side)
        if not isinstance(side, Mapping) or any(
            not isinstance(side.get(metric), int) or not 1 <= side[metric] <= 5
            for metric in required_scores
        ):
            raise ValueError("DPO vote contains invalid rubric scores")
    if not str(row.get("evidence", "")).strip():
        raise ValueError("DPO vote requires concrete evidence")
    return dict(row)


def build_user_calibration_packet(
    pairs: Mapping[str, Any], *, count: int = 20, seed: int = 10_951
) -> dict[str, Any]:
    rows = list(pairs.get("pairs", []))
    if count <= 0 or len(rows) < count:
        raise ValueError("insufficient pairs for the user calibration packet")
    selected = sorted(random.Random(seed).sample(rows, count), key=lambda row: row["pair_id"])
    return {
        "preference_version": PREFERENCE_VERSION,
        "calibration_seed": seed,
        "pair_count": count,
        "pairs": selected,
        "answers_present": False,
        "v7_test_accessed": False,
    }


def build_judge_assignments(
    pairs: Mapping[str, Any], *, judge_ids: Sequence[str]
) -> dict[str, Any]:
    """Assign every blind pair to exactly three distinct recorded judges."""

    normalized_judges = tuple(str(value).strip() for value in judge_ids)
    if len(normalized_judges) != 3 or len(set(normalized_judges)) != 3 or not all(normalized_judges):
        raise ValueError("DPO judging requires exactly three distinct judge identities")
    rows = list(pairs.get("pairs", []))
    if int(pairs.get("pair_count", -1)) != len(rows) or not rows:
        raise ValueError("DPO blind pair packet is empty or inconsistent")
    assignments = [
        {
            "assignment_id": hashlib.sha256(
                f"{PREFERENCE_VERSION}|{row['pair_id']}|{judge_id}".encode()
            ).hexdigest()[:20],
            "judge_id": judge_id,
            "pair": dict(row),
        }
        for row in sorted(rows, key=lambda item: str(item["pair_id"]))
        for judge_id in normalized_judges
    ]
    return {
        "preference_version": PREFERENCE_VERSION,
        "judge_ids": list(normalized_judges),
        "pair_count": len(rows),
        "assignment_count": len(assignments),
        "assignments": assignments,
        "blind_to_candidate_identity_and_recipe": True,
        "v7_test_accessed": False,
    }


def aggregate_judge_votes(
    *,
    pairs: Mapping[str, Any],
    private_mapping: Mapping[str, Any],
    votes: Sequence[Mapping[str, Any]],
    expected_judge_ids: Sequence[str],
) -> dict[str, Any]:
    """Freeze majority decisions while preserving every score and disagreement."""

    pair_rows = {str(row["pair_id"]): row for row in pairs.get("pairs", [])}
    mapping_rows = {
        str(row["pair_id"]): row for row in private_mapping.get("mapping", [])
    }
    if not pair_rows or set(pair_rows) != set(mapping_rows):
        raise ValueError("DPO public/private pair identities differ")
    judges = {str(value).strip() for value in expected_judge_ids}
    if len(judges) != 3 or not all(judges):
        raise ValueError("DPO vote aggregation requires three judge identities")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_assignments: set[tuple[str, str]] = set()
    for raw in votes:
        row = validate_vote(raw)
        pair_id = str(row["pair_id"])
        judge_id = str(row["judge_id"])
        if pair_id not in pair_rows or judge_id not in judges:
            raise ValueError("DPO vote references an unknown pair or judge")
        assignment = (pair_id, judge_id)
        if assignment in seen_assignments:
            raise ValueError("duplicate DPO judge vote")
        seen_assignments.add(assignment)
        grouped[pair_id].append(row)
    expected_assignments = {(pair_id, judge) for pair_id in pair_rows for judge in judges}
    if seen_assignments != expected_assignments:
        raise ValueError("DPO vote set is incomplete")

    decisions = []
    for pair_id in sorted(pair_rows):
        pair_votes = sorted(grouped[pair_id], key=lambda row: str(row["judge_id"]))
        counts = Counter(str(row["preference"]) for row in pair_votes)
        majority = "A" if counts["A"] >= 2 else "B" if counts["B"] >= 2 else "no_majority"
        pair = pair_rows[pair_id]
        mapping = mapping_rows[pair_id]
        chosen_side = majority if majority in {"A", "B"} else None
        rejected_side = ({"A": "B", "B": "A"}.get(majority))
        decisions.append(
            {
                "pair_id": pair_id,
                "prompt_id": str(pair["prompt_id"]),
                "vote_counts": {key: counts[key] for key in ("A", "B", "tie")},
                "majority_preference": majority,
                "decisive": chosen_side is not None,
                "unanimous": counts[majority] == 3 if chosen_side else False,
                "chosen_candidate_id": (
                    str(mapping[f"candidate_{chosen_side.lower()}_id"])
                    if chosen_side else None
                ),
                "rejected_candidate_id": (
                    str(mapping[f"candidate_{rejected_side.lower()}_id"])
                    if rejected_side else None
                ),
                "chosen_text": str(pair[f"candidate_{chosen_side.lower()}"])
                if chosen_side else None,
                "rejected_text": str(pair[f"candidate_{rejected_side.lower()}"])
                if rejected_side else None,
                "votes": pair_votes,
            }
        )
    decisive_count = sum(row["decisive"] for row in decisions)
    unanimous_count = sum(row["unanimous"] for row in decisions)
    return {
        "preference_version": PREFERENCE_VERSION,
        "pair_count": len(decisions),
        "vote_count": len(votes),
        "decisive_pair_count": decisive_count,
        "decisive_pair_rate": decisive_count / len(decisions),
        "unanimous_pair_count": unanimous_count,
        "unanimous_pair_rate": unanimous_count / len(decisions),
        "decisions": decisions,
        "correlated_ai_judges_are_not_independent_replicates": True,
        "v7_test_accessed": False,
    }


def score_user_calibration(
    *,
    aggregation: Mapping[str, Any],
    calibration_packet: Mapping[str, Any],
    user_votes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare one human vote per calibration pair to the frozen AI majority."""

    calibration_ids = {
        str(row["pair_id"]) for row in calibration_packet.get("pairs", [])
    }
    if len(calibration_ids) != int(calibration_packet.get("pair_count", -1)):
        raise ValueError("DPO calibration packet has duplicate or missing pairs")
    decisions = {
        str(row["pair_id"]): row for row in aggregation.get("decisions", [])
    }
    by_pair: dict[str, str] = {}
    for row in user_votes:
        pair_id = str(row.get("pair_id", ""))
        preference = str(row.get("preference", ""))
        if pair_id not in calibration_ids or preference not in {"A", "B", "tie"}:
            raise ValueError("invalid user calibration vote")
        if pair_id in by_pair:
            raise ValueError("duplicate user calibration vote")
        by_pair[pair_id] = preference
    if set(by_pair) != calibration_ids:
        raise ValueError("user calibration vote set is incomplete")
    comparable = [
        pair_id for pair_id in sorted(calibration_ids)
        if decisions.get(pair_id, {}).get("majority_preference") in {"A", "B"}
    ]
    agreements = sum(
        by_pair[pair_id] == decisions[pair_id]["majority_preference"]
        for pair_id in comparable
    )
    accuracy = agreements / len(comparable) if comparable else 0.0
    return {
        "preference_version": PREFERENCE_VERSION,
        "calibration_pair_count": len(calibration_ids),
        "comparable_decisive_pair_count": len(comparable),
        "agreement_count": agreements,
        "agreement_rate": accuracy,
        "agreement_gate_at_least_080": accuracy >= 0.80,
        "user_votes": [
            {"pair_id": pair_id, "preference": by_pair[pair_id]}
            for pair_id in sorted(by_pair)
        ],
        "v7_test_accessed": False,
    }


def build_chosen_rejected_dataset(
    *, aggregation: Mapping[str, Any], calibration: Mapping[str, Any]
) -> dict[str, Any]:
    """Export preference examples only after majority and human-calibration gates."""

    if float(aggregation.get("decisive_pair_rate", 0.0)) < 0.70:
        raise PermissionError("DPO majority-decision gate failed")
    if calibration.get("agreement_gate_at_least_080") is not True:
        raise PermissionError("DPO human-AI calibration gate failed")
    examples = [
        {
            "pair_id": row["pair_id"],
            "prompt_id": row["prompt_id"],
            "chosen_candidate_id": row["chosen_candidate_id"],
            "rejected_candidate_id": row["rejected_candidate_id"],
            "chosen": row["chosen_text"],
            "rejected": row["rejected_text"],
            "vote_counts": row["vote_counts"],
        }
        for row in aggregation.get("decisions", [])
        if row.get("decisive") is True
    ]
    return {
        "preference_version": PREFERENCE_VERSION,
        "example_count": len(examples),
        "examples": examples,
        "human_ai_calibration_agreement_rate": calibration["agreement_rate"],
        "dpo_training_authorized": False,
        "v7_test_accessed": False,
    }


def _validate_training_row(
    document: Mapping[str, Any], record: Mapping[str, Any], index: int
) -> None:
    if (
        document.get("pool_id") != "sonnets_train"
        or document.get("split") != "train"
        or int(document.get("document_index", -1)) != index
        or int(record.get("document_index", -1)) != index
        or str(record.get("record_id")) != str(document.get("unit_id"))
        or str(record.get("logical_sha256")) != str(document.get("logical_sha256"))
    ):
        raise ValueError("DPO training document/reference lineage mismatch")


def _balanced_selection(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    by_period: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_period[str(row["epoch_key"])].append(row)
    quotas = {period: 0 for period in by_period}
    while sum(quotas.values()) < count:
        available = [period for period in sorted(by_period) if quotas[period] < len(by_period[period])]
        if not available:
            raise ValueError("insufficient DPO training prompts")
        selected_period = min(available, key=lambda value: (quotas[value], value))
        quotas[selected_period] += 1
    authors: Counter[str] = Counter()
    works: Counter[str] = Counter()
    selected = []
    for period in sorted(by_period):
        remaining = list(by_period[period])
        for _ in range(quotas[period]):
            row = min(
                remaining,
                key=lambda item: (
                    works[str(item["work_key"])],
                    authors[str(item["author_key"])],
                    hashlib.sha256(f"{SELECTION_SEED}|{item['unit_id']}".encode()).hexdigest(),
                ),
            )
            remaining.remove(row)
            selected.append(row)
            authors[str(row["author_key"])] += 1
            works[str(row["work_key"])] += 1
    return sorted(selected, key=lambda row: (str(row["epoch_key"]), str(row["unit_id"])))


def _normalize_opening(value: str) -> str:
    return " ".join(value.casefold().split())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
