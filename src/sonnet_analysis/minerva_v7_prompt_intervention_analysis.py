"""Paired statistics and blinded review for the Stage-3 prompt experiment."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sonnet_analysis.minerva_v7_memorization import score_texts_against_reference
from sonnet_analysis.minerva_v7_prompt_intervention import (
    EXPECTED_ARM_IDS,
    EXPECTED_PROMPT_SHA256,
    EXPECTED_SEEDS,
    PROMPT_INTERVENTION_VERSION,
)


PROMPT_INTERVENTION_ANALYSIS_VERSION = (
    "minerva_7b_v7_stage_3_prompt_intervention_analysis_v1"
)
METRICS = (
    "fourteen_line",
    "prompt_preserved",
    "repetition_ratio",
    "unique_character_ratio",
    "character_count",
    "meta_text_free",
    "ends_with_terminal_punctuation",
    "no_line_at_or_above_120_characters",
    "surface_screen_pass",
    "explicit_4433_stanza_pattern",
)


def analyze_prompt_intervention(
    *,
    output_dir: Path,
    expected_state_identity: str,
    memorization_records: Sequence[Mapping[str, str]] | None,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    confidence_level: float,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Verify the 3,840-row matched grid and compare each arm with control."""

    if bootstrap_resamples <= 0 or not 0 < confidence_level < 1:
        raise ValueError("invalid prompt-intervention bootstrap contract")
    completion = json.loads((output_dir / "complete.json").read_text(encoding="utf-8"))
    if (
        completion.get("experiment_version") != PROMPT_INTERVENTION_VERSION
        or completion.get("completion_scope")
        != "authoritative_120_prompt_8_seed_4_arm_grid"
        or completion.get("completed_final_output_count") != 3840
        or len(completion.get("outputs", [])) != 3840
        or completion.get("state_id") != "stage_3_selected"
        or completion.get("state_identity_sha256") != expected_state_identity
        or completion.get("prompt_manifest_sha256") != EXPECTED_PROMPT_SHA256
        or completion.get("v7_test_accessed") is not False
        or completion.get("training_performed") is not False
    ):
        raise ValueError("prompt-intervention completion contract mismatch")
    rows = []
    grid = set()
    for item in completion["outputs"]:
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("prompt-intervention completion contains unsafe path")
        path = output_dir / relative
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item.get("sha256"):
            raise ValueError("prompt-intervention output hash mismatch")
        payload = json.loads(raw)
        if (
            payload.get("experiment_version") != PROMPT_INTERVENTION_VERSION
            or payload.get("state_identity_sha256") != expected_state_identity
            or payload.get("prompt_manifest_sha256") != EXPECTED_PROMPT_SHA256
            or payload.get("v7_test_accessed") is not False
            or payload.get("training_performed") is not False
        ):
            raise ValueError("prompt-intervention output lineage mismatch")
        arm_id = str(payload["arm_id"])
        prompt_id = str(payload["prompt"]["id"])
        seed = int(payload["base_seed"])
        identity = (arm_id, prompt_id, seed)
        if identity in grid:
            raise ValueError("duplicate prompt-intervention output")
        grid.add(identity)
        metrics = payload["metrics"]
        diagnostics = payload["surface_diagnostics"]
        rows.append(
            {
                "arm_id": arm_id,
                "prompt_id": prompt_id,
                "seed": seed,
                "period": payload["prompt"].get("period", ""),
                "text": payload["text"],
                "attempt_count": len(payload["attempts"]),
                "selected_attempt_index": int(payload["selected_attempt_index"]),
                "fourteen_line": int(metrics["non_empty_line_count"]) == 14,
                "prompt_preserved": bool(metrics["prompt_preserved"]),
                "repetition_ratio": float(metrics["repetition_ratio"]),
                "unique_character_ratio": float(metrics["unique_character_ratio"]),
                "character_count": int(metrics["character_count"]),
                "meta_text_free": bool(diagnostics["meta_text_free"]),
                "ends_with_terminal_punctuation": bool(
                    diagnostics["ends_with_terminal_punctuation"]
                ),
                "no_line_at_or_above_120_characters": bool(
                    diagnostics["no_line_at_or_above_120_characters"]
                ),
                "surface_screen_pass": bool(diagnostics["surface_screen_pass"]),
                "explicit_4433_stanza_pattern": bool(
                    diagnostics["explicit_4433_stanza_pattern"]
                ),
                "memorization": None,
            }
        )
    expected_grid = {
        (arm, f_prompt, seed)
        for arm in EXPECTED_ARM_IDS
        for f_prompt in {row["prompt_id"] for row in rows}
        for seed in EXPECTED_SEEDS
    }
    if len({row["prompt_id"] for row in rows}) != 120 or grid != expected_grid:
        raise ValueError("prompt-intervention matched grid is incomplete")
    if memorization_records:
        scored = score_texts_against_reference(
            [str(row["text"]) for row in rows],
            memorization_records,
            progress=progress,
        )
        for row, memorization in zip(rows, scored, strict=True):
            row["memorization"] = memorization
    summaries = _summaries(rows, bootstrap_resamples, bootstrap_seed, confidence_level)
    comparisons = _comparisons(
        rows, bootstrap_resamples, bootstrap_seed + 10_000, confidence_level
    )
    return {
        "analysis_version": PROMPT_INTERVENTION_ANALYSIS_VERSION,
        "analysis_role": "post_hoc_stage_3_prompt_and_stopping_experiment",
        "rows": rows,
        "summaries": summaries,
        "comparisons_to_control": comparisons,
        "bootstrap": {
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "cluster_unit": "prompt_id",
        },
        "memorization_scored": memorization_records is not None,
        "surface_retry_is_not_poetic_quality_selection": True,
        "v7_test_accessed": False,
        "training_performed": False,
    }


def build_prompt_intervention_blinded_sample(
    *, analysis: Mapping[str, Any], selected_prompt_count: int, selection_seed: int
) -> dict[str, Any]:
    """Select prompt clusters and one seed while retaining every arm."""

    prompts = sorted({str(row["prompt_id"]) for row in analysis["rows"]})
    if selected_prompt_count <= 0 or selected_prompt_count > len(prompts):
        raise ValueError("invalid prompt-intervention blinded prompt count")
    rng = random.Random(selection_seed)
    selected = set(rng.sample(prompts, selected_prompt_count))
    seed_by_prompt = {prompt: rng.choice(EXPECTED_SEEDS) for prompt in selected}
    sampled = [
        row for row in analysis["rows"]
        if row["prompt_id"] in selected
        and row["seed"] == seed_by_prompt[row["prompt_id"]]
    ]
    expected = selected_prompt_count * len(EXPECTED_ARM_IDS)
    if len(sampled) != expected:
        raise ValueError("prompt-intervention blinded sample is incomplete")
    mapping = []
    for row in sampled:
        blind_id = hashlib.sha256(
            f"{PROMPT_INTERVENTION_ANALYSIS_VERSION}|{row['arm_id']}|"
            f"{row['prompt_id']}|{row['seed']}".encode()
        ).hexdigest()[:16]
        mapping.append({"blind_id": blind_id, **row})
    return {
        "analysis_version": PROMPT_INTERVENTION_ANALYSIS_VERSION,
        "selection_seed": selection_seed,
        "sample_rows": expected,
        "mapping": sorted(mapping, key=lambda row: row["blind_id"]),
        "v7_test_accessed": False,
    }


def prompt_intervention_review_markdown(sample: Mapping[str, Any]) -> str:
    """Render the review without arm identities or retry counts."""

    lines = [
        "# Minerva V7 Stage-3 Prompt Experiment Blinded Review",
        "",
        "Score 1 (poor) through 5 (strong) without consulting the private mapping.",
        "",
        "| Blind ID | Grammar | Historical Register | Poetic Quality | Sonnet/Form Coherence | Volta/Argument | Meta-text | Truncation | Evidence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in sample["mapping"]:
        lines.append(
            f"| `{row['blind_id']}` | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |"
        )
    lines.extend(["", "## Outputs", ""])
    for row in sample["mapping"]:
        lines.extend(
            [f"### `{row['blind_id']}`", "", "```text", str(row["text"]), "```", ""]
        )
    return "\n".join(lines)


def _summaries(
    rows: Sequence[Mapping[str, Any]], resamples: int, seed: int,
    confidence_level: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["arm_id"])].append(row)
    output = []
    for group_index, arm_id in enumerate(EXPECTED_ARM_IDS):
        values = grouped[arm_id]
        by_prompt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in values:
            by_prompt[str(row["prompt_id"])].append(row)
        prompt_ids = sorted(by_prompt)
        rng = random.Random(seed + group_index)
        summary: dict[str, Any] = {
            "arm_id": arm_id,
            "outputs": len(values),
            "prompt_clusters": len(prompt_ids),
            "mean_attempt_count": statistics.fmean(
                int(row["attempt_count"]) for row in values
            ),
            "high_memorization_risk_count": sum(
                row["memorization"] is not None
                and row["memorization"]["risk_level"] == "high"
                for row in values
            ),
        }
        for metric in METRICS:
            prompt_means = {
                prompt: statistics.fmean(float(row[metric]) for row in by_prompt[prompt])
                for prompt in prompt_ids
            }
            distribution = sorted(
                statistics.fmean(prompt_means[rng.choice(prompt_ids)] for _ in prompt_ids)
                for _ in range(resamples)
            )
            low, high = _bounds(distribution, confidence_level)
            summary[metric] = {
                "mean": statistics.fmean(prompt_means.values()),
                "cluster_bootstrap_ci_low": low,
                "cluster_bootstrap_ci_high": high,
            }
        output.append(summary)
    return output


def _comparisons(
    rows: Sequence[Mapping[str, Any]], resamples: int, seed: int,
    confidence_level: float,
) -> list[dict[str, Any]]:
    lookup = {
        (row["arm_id"], row["prompt_id"], row["seed"]): row for row in rows
    }
    prompts = sorted({str(row["prompt_id"]) for row in rows})
    output = []
    control = EXPECTED_ARM_IDS[0]
    for group_index, arm_id in enumerate(EXPECTED_ARM_IDS[1:]):
        rng = random.Random(seed + group_index)
        summary: dict[str, Any] = {
            "control_arm_id": control,
            "intervention_arm_id": arm_id,
            "paired_outputs": len(prompts) * len(EXPECTED_SEEDS),
            "prompt_clusters": len(prompts),
        }
        for metric in METRICS:
            differences = {
                prompt: statistics.fmean(
                    float(lookup[(arm_id, prompt, seed_value)][metric])
                    - float(lookup[(control, prompt, seed_value)][metric])
                    for seed_value in EXPECTED_SEEDS
                )
                for prompt in prompts
            }
            distribution = sorted(
                statistics.fmean(differences[rng.choice(prompts)] for _ in prompts)
                for _ in range(resamples)
            )
            low, high = _bounds(distribution, confidence_level)
            summary[metric] = {
                "mean_paired_change": statistics.fmean(differences.values()),
                "cluster_bootstrap_ci_low": low,
                "cluster_bootstrap_ci_high": high,
            }
        output.append(summary)
    return output


def _bounds(distribution: Sequence[float], confidence_level: float) -> tuple[float, float]:
    alpha = (1 - confidence_level) / 2
    last = len(distribution) - 1
    return distribution[int(alpha * last)], distribution[int((1 - alpha) * last)]
