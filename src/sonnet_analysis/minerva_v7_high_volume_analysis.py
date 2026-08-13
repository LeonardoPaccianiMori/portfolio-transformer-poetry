"""Clustered statistical summaries and blinded sampling for high-volume outputs."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sonnet_analysis.minerva_v7_high_volume_generation import HIGH_VOLUME_VERSION
from sonnet_analysis.minerva_v7_registry import MODEL_STATES
from sonnet_analysis.minerva_v7_registry import COMPARISONS
from sonnet_evaluation.metrics import score_generated_text


ANALYSIS_VERSION = "minerva_7b_v7_high_volume_analysis_v1"
METRICS = (
    "fourteen_line", "prompt_preserved", "repetition_ratio",
    "unique_character_ratio", "character_count",
)


def analyze_high_volume_outputs(
    *, state_directories: Mapping[str, Path], expected_state_identities: Mapping[str, str],
    bootstrap_resamples: int, bootstrap_seed: int, confidence_level: float,
) -> dict[str, Any]:
    """Verify 20,160 outputs and bootstrap state/recipe means by prompt cluster."""

    expected_states = {row.state_id for row in MODEL_STATES}
    if set(state_directories) != expected_states:
        raise ValueError("high-volume analysis requires all seven frozen states")
    if bootstrap_resamples <= 0 or not 0 < confidence_level < 1:
        raise ValueError("invalid bootstrap contract")
    rows = []
    grids = {}
    for state_id, directory in state_directories.items():
        completion = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
        if (
            completion.get("generation_version") != HIGH_VOLUME_VERSION
            or completion.get("completion_scope") != "authoritative_120_prompt_8_seed_3_recipe_grid"
            or completion.get("completed_output_count") != 2880
            or len(completion.get("outputs", [])) != 2880
            or completion.get("v7_test_accessed") is not False
            or completion.get("state_identity_sha256") != expected_state_identities[state_id]
        ):
            raise ValueError("high-volume state completion contract mismatch")
        grid = set()
        for item in completion["outputs"]:
            relative = Path(str(item["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("high-volume completion contains unsafe path")
            path = directory / relative
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != item.get("sha256"):
                raise ValueError("high-volume output hash mismatch")
            payload = json.loads(raw)
            if (
                payload.get("state_id") != state_id
                or payload.get("state_identity_sha256") != expected_state_identities[state_id]
                or payload.get("analysis_role") != "exploratory_high_volume"
                or payload.get("v7_test_accessed") is not False
            ):
                raise ValueError("high-volume output lineage mismatch")
            prompt_id = str(payload["prompt"]["id"])
            seed = int(payload["seed"])
            recipe_id = str(payload["recipe"]["recipe_id"])
            identity = (prompt_id, seed, recipe_id)
            if identity in grid:
                raise ValueError("duplicate high-volume prompt/seed/recipe output")
            grid.add(identity)
            metrics = score_generated_text(payload["text"], payload["opening_line"])
            rows.append(
                {
                    "state_id": state_id, "prompt_id": prompt_id,
                    "seed": seed, "recipe_id": recipe_id,
                    "period": payload["prompt"]["period"],
                    "text": payload["text"],
                    "fourteen_line": metrics["non_empty_line_count"] == 14,
                    **metrics,
                }
            )
        grids[state_id] = grid
    if len({frozenset(grid) for grid in grids.values()}) != 1 or any(len(grid) != 2880 for grid in grids.values()):
        raise ValueError("high-volume states do not share the exact matched grid")
    summaries = _summaries(rows, bootstrap_resamples, bootstrap_seed, confidence_level)
    comparisons = _comparison_summaries(
        rows, bootstrap_resamples, bootstrap_seed + 10_000, confidence_level
    )
    return {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_role": "exploratory_high_volume",
        "state_count": 7, "rows": rows, "summaries": summaries,
        "comparison_summaries": comparisons,
        "bootstrap": {
            "resamples": bootstrap_resamples, "seed": bootstrap_seed,
            "confidence_level": confidence_level, "cluster_unit": "prompt_id",
        },
        "confirmatory_grid_unchanged": True,
        "v7_test_accessed": False,
        "causal_experiments_performed": False,
    }


def build_high_volume_blinded_sample(
    *, analysis_report: Mapping[str, Any], selected_prompt_count: int,
    selection_seed: int,
) -> dict[str, Any]:
    """Select 24 prompt clusters, one seed, all recipes/states: exactly 504 rows."""

    rows = analysis_report["rows"]
    prompts = sorted({str(row["prompt_id"]) for row in rows})
    if selected_prompt_count <= 0 or selected_prompt_count > len(prompts):
        raise ValueError("invalid blinded prompt sample count")
    rng = random.Random(selection_seed)
    selected = set(rng.sample(prompts, selected_prompt_count))
    seed_by_prompt = {prompt: rng.choice(list(range(4200, 4208))) for prompt in selected}
    sampled = [
        row for row in rows
        if row["prompt_id"] in selected and row["seed"] == seed_by_prompt[row["prompt_id"]]
    ]
    expected = selected_prompt_count * 7 * 3
    if len(sampled) != expected:
        raise ValueError("blinded high-volume sample is incomplete")
    mapping = []
    for row in sampled:
        blind_id = hashlib.sha256(
            f"{ANALYSIS_VERSION}|{row['state_id']}|{row['prompt_id']}|{row['seed']}|{row['recipe_id']}".encode()
        ).hexdigest()[:16]
        mapping.append({"blind_id": blind_id, **row})
    return {
        "analysis_version": ANALYSIS_VERSION,
        "selection_seed": selection_seed,
        "selected_prompt_count": selected_prompt_count,
        "sample_rows": expected,
        "mapping": sorted(mapping, key=lambda row: row["blind_id"]),
        "v7_test_accessed": False,
    }


def blinded_review_markdown(blind_sample: Mapping[str, Any]) -> str:
    """Render only blind IDs and texts; state/recipe identities stay in private mapping."""

    lines = [
        "# Minerva V7 High-Volume Blinded Review", "",
        "Score 1 (poor) through 5 (strong) without consulting the private mapping.", "",
        "| Blind ID | Grammar | Historical Register | Poetic Quality | Sonnet/Form Coherence | Volta/Argument | Collapse | Evidence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in blind_sample["mapping"]:
        lines.append(f"| `{row['blind_id']}` | TODO | TODO | TODO | TODO | TODO | TODO | TODO |")
    lines.extend(["", "## Outputs", ""])
    for row in blind_sample["mapping"]:
        lines.extend([f"### `{row['blind_id']}`", "", "```text", str(row["text"]), "```", ""])
    return "\n".join(lines)


def _summaries(
    rows: Sequence[Mapping[str, Any]], resamples: int, seed: int,
    confidence_level: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["state_id"]), str(row["recipe_id"]))].append(row)
    output = []
    for group_index, ((state_id, recipe_id), values) in enumerate(sorted(grouped.items())):
        by_prompt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in values:
            by_prompt[str(row["prompt_id"])].append(row)
        prompt_ids = sorted(by_prompt)
        rng = random.Random(seed + group_index)
        row = {"state_id": state_id, "recipe_id": recipe_id, "outputs": len(values), "prompt_clusters": len(prompt_ids)}
        for metric in METRICS:
            prompt_means = {
                prompt: statistics.fmean(float(item[metric]) for item in by_prompt[prompt])
                for prompt in prompt_ids
            }
            estimate = statistics.fmean(prompt_means.values())
            distribution = sorted(
                statistics.fmean(prompt_means[rng.choice(prompt_ids)] for _ in prompt_ids)
                for _ in range(resamples)
            )
            alpha = (1 - confidence_level) / 2
            low = distribution[int(alpha * (resamples - 1))]
            high = distribution[int((1 - alpha) * (resamples - 1))]
            row[metric] = {"mean": estimate, "cluster_bootstrap_ci_low": low, "cluster_bootstrap_ci_high": high}
        output.append(row)
    return output


def _comparison_summaries(
    rows: Sequence[Mapping[str, Any]], resamples: int, seed: int,
    confidence_level: float,
) -> list[dict[str, Any]]:
    lookup = {
        (row["state_id"], row["prompt_id"], row["seed"], row["recipe_id"]): row
        for row in rows
    }
    prompt_ids = sorted({str(row["prompt_id"]) for row in rows})
    recipe_ids = sorted({str(row["recipe_id"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    output = []
    group_index = 0
    for comparison in COMPARISONS:
        for recipe_id in recipe_ids:
            prompt_differences: dict[str, dict[str, float]] = {}
            for prompt_id in prompt_ids:
                metric_values = {}
                for metric in METRICS:
                    differences = [
                        float(lookup[(comparison["right"], prompt_id, current_seed, recipe_id)][metric])
                        - float(lookup[(comparison["left"], prompt_id, current_seed, recipe_id)][metric])
                        for current_seed in seeds
                    ]
                    metric_values[metric] = statistics.fmean(differences)
                prompt_differences[prompt_id] = metric_values
            rng = random.Random(seed + group_index)
            row: dict[str, Any] = {
                "comparison_id": comparison["comparison_id"],
                "left_state_id": comparison["left"],
                "right_state_id": comparison["right"],
                "recipe_id": recipe_id,
                "prompt_clusters": len(prompt_ids),
                "paired_outputs": len(prompt_ids) * len(seeds),
            }
            for metric in METRICS:
                values = {prompt: prompt_differences[prompt][metric] for prompt in prompt_ids}
                estimate = statistics.fmean(values.values())
                distribution = sorted(
                    statistics.fmean(values[rng.choice(prompt_ids)] for _ in prompt_ids)
                    for _ in range(resamples)
                )
                alpha = (1 - confidence_level) / 2
                row[metric] = {
                    "mean_paired_change": estimate,
                    "cluster_bootstrap_ci_low": distribution[int(alpha * (resamples - 1))],
                    "cluster_bootstrap_ci_high": distribution[int((1 - alpha) * (resamples - 1))],
                }
            output.append(row)
            group_index += 1
    return output
