"""Paired analysis of the Stage-3 no-labels plus creative-decoding cell."""

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
from sonnet_analysis.minerva_v7_memorization import score_texts_against_reference
from sonnet_analysis.minerva_v7_no_labels_creative import (
    NO_LABELS_CREATIVE_VERSION,
)
from sonnet_analysis.minerva_v7_prompt_intervention import (
    PROMPT_INTERVENTION_VERSION,
)
from sonnet_analysis.minerva_v7_quality import generated_sonnet_surface_diagnostics
from sonnet_evaluation.metrics import score_generated_text


ANALYSIS_VERSION = "minerva_7b_v7_stage_3_no_labels_creative_analysis_v1"
CELL_IDS = (
    "current_prompt_creative",
    "no_labels_balanced",
    "no_labels_creative",
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
)


def analyze_no_labels_creative(
    *,
    new_cell_dir: Path,
    high_volume_stage_3_dir: Path,
    prompt_intervention_dir: Path,
    expected_state_identity: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    confidence_level: float,
    memorization_records: Sequence[Mapping[str, str]] | None = None,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Verify three matched cells and estimate prompt-clustered paired changes."""

    if bootstrap_resamples <= 0 or not 0 < confidence_level < 1:
        raise ValueError("invalid no-labels/creative bootstrap contract")
    cells = {
        "current_prompt_creative": _load_high_volume_creative(
            high_volume_stage_3_dir, expected_state_identity
        ),
        "no_labels_balanced": _load_prompt_balanced(
            prompt_intervention_dir, expected_state_identity
        ),
        "no_labels_creative": _load_new_cell(new_cell_dir, expected_state_identity),
    }
    grids = [{(row["prompt_id"], row["seed"]) for row in rows} for rows in cells.values()]
    if any(len(grid) != 960 for grid in grids) or len({frozenset(grid) for grid in grids}) != 1:
        raise ValueError("no-labels/creative comparator grids are not exactly matched")
    rows = [row for cell_rows in cells.values() for row in cell_rows]
    new_rows = [row for row in rows if row["cell_id"] == "no_labels_creative"]
    if memorization_records is not None:
        memorization = score_texts_against_reference(
            [str(row["text"]) for row in new_rows],
            memorization_records,
            progress=progress,
        )
        for row, score in zip(new_rows, memorization, strict=True):
            row["memorization"] = score
    summaries = _summaries(rows, bootstrap_resamples, bootstrap_seed, confidence_level)
    comparisons = [
        _paired_comparison(
            rows,
            left_cell_id=left,
            right_cell_id="no_labels_creative",
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + 10_000 + index,
            confidence_level=confidence_level,
        )
        for index, left in enumerate(CELL_IDS[:2])
    ]
    return {
        "analysis_version": ANALYSIS_VERSION,
        "analysis_role": "post_hoc_stage_3_combined_prompt_decoding_experiment",
        "state_id": "stage_3_selected",
        "state_identity_sha256": expected_state_identity,
        "cell_ids": list(CELL_IDS),
        "rows": rows,
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "bootstrap": {
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "cluster_unit": "prompt_id",
        },
        "surface_screen_is_not_poetic_quality_judgment": True,
        "memorization_scored_for_new_cell": memorization_records is not None,
        "v7_test_accessed": False,
        "training_performed": False,
    }


def build_no_labels_creative_blinded_sample(
    *, analysis: Mapping[str, Any], prompt_count: int, selection_seed: int
) -> dict[str, Any]:
    """Freeze matched blinded rows before any literary-quality claims."""

    prompts = sorted({str(row["prompt_id"]) for row in analysis["rows"]})
    if prompt_count <= 0 or prompt_count > len(prompts):
        raise ValueError("invalid no-labels/creative blind prompt count")
    rng = random.Random(selection_seed)
    selected_prompts = sorted(rng.sample(prompts, prompt_count))
    selected = set(selected_prompts)
    seed_by_prompt = {
        prompt: rng.choice(list(range(4200, 4208))) for prompt in selected_prompts
    }
    sampled = [
        row
        for row in analysis["rows"]
        if row["prompt_id"] in selected and row["seed"] == seed_by_prompt[row["prompt_id"]]
    ]
    expected = prompt_count * len(CELL_IDS)
    if len(sampled) != expected:
        raise ValueError("no-labels/creative blinded sample is incomplete")
    mapping = []
    for row in sampled:
        blind_id = hashlib.sha256(
            f"{ANALYSIS_VERSION}|{row['cell_id']}|{row['prompt_id']}|{row['seed']}".encode()
        ).hexdigest()[:16]
        mapping.append({"blind_id": blind_id, **row})
    return {
        "analysis_version": ANALYSIS_VERSION,
        "selection_seed": selection_seed,
        "selected_prompt_count": prompt_count,
        "sample_rows": expected,
        "mapping": sorted(mapping, key=lambda row: row["blind_id"]),
        "v7_test_accessed": False,
    }


def no_labels_creative_review_markdown(sample: Mapping[str, Any]) -> str:
    """Render a review form without cell identities or automatic scores."""

    lines = [
        "# Stage-3 Combined-Prompt/Decoder Blinded Review",
        "",
        "Score 1 (poor) through 5 (strong) without consulting the private mapping.",
        "",
        "| Blind ID | Grammar | Historical Register | Poetic Quality | Sonnet/Form | Volta/Argument | Complete | Meta-text | Evidence |",
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


def _load_new_cell(directory: Path, identity: str) -> list[dict[str, Any]]:
    completion = _completion(directory, "complete.json")
    if (
        completion.get("experiment_version") != NO_LABELS_CREATIVE_VERSION
        or completion.get("completion_scope") != "authoritative_120_prompt_8_seed_cell"
        or completion.get("completed_output_count") != 960
        or completion.get("state_identity_sha256") != identity
        or completion.get("v7_test_accessed") is not False
        or completion.get("training_performed") is not False
    ):
        raise ValueError("new no-labels/creative completion contract mismatch")
    return _verified_rows(
        directory, completion["outputs"], identity, "no_labels_creative", _new_validator
    )


def _load_high_volume_creative(directory: Path, identity: str) -> list[dict[str, Any]]:
    completion = _completion(directory, "complete.json")
    if (
        completion.get("generation_version") != HIGH_VOLUME_VERSION
        or completion.get("completed_output_count") != 2880
        or completion.get("state_id") != "stage_3_selected"
        or completion.get("state_identity_sha256") != identity
        or completion.get("v7_test_accessed") is not False
    ):
        raise ValueError("high-volume comparator completion contract mismatch")
    items = [row for row in completion["outputs"] if row["recipe_id"] == "creative"]
    if len(items) != 960:
        raise ValueError("high-volume creative comparator is incomplete")
    return _verified_rows(
        directory, items, identity, "current_prompt_creative", _high_volume_validator
    )


def _load_prompt_balanced(directory: Path, identity: str) -> list[dict[str, Any]]:
    completion = _completion(directory, "complete.json")
    if (
        completion.get("experiment_version") != PROMPT_INTERVENTION_VERSION
        or completion.get("completed_final_output_count") != 3840
        or completion.get("state_identity_sha256") != identity
        or completion.get("v7_test_accessed") is not False
        or completion.get("training_performed") is not False
    ):
        raise ValueError("prompt comparator completion contract mismatch")
    items = [
        row for row in completion["outputs"]
        if row["arm_id"] == "explicit_no_labels_or_prose"
    ]
    if len(items) != 960:
        raise ValueError("no-labels balanced comparator is incomplete")
    return _verified_rows(
        directory, items, identity, "no_labels_balanced", _prompt_validator
    )


def _completion(directory: Path, name: str) -> dict[str, Any]:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def _verified_rows(
    directory: Path,
    items: Sequence[Mapping[str, Any]],
    identity: str,
    cell_id: str,
    validator: Any,
) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for item in items:
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe comparator output path")
        path = directory / relative
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item.get("sha256"):
            raise ValueError("comparator output hash mismatch")
        payload = json.loads(raw)
        validator(payload, identity)
        prompt_id = str(payload["prompt"]["id"])
        seed = int(payload.get("seed", payload.get("base_seed")))
        key = (prompt_id, seed)
        if key in seen:
            raise ValueError("duplicate comparator prompt/seed output")
        seen.add(key)
        metrics = score_generated_text(payload["text"], payload["opening_line"])
        diagnostics = generated_sonnet_surface_diagnostics(
            payload["text"],
            non_empty_line_count=int(metrics["non_empty_line_count"]),
            repetition_ratio=float(metrics["repetition_ratio"]),
        )
        rows.append(
            {
                "cell_id": cell_id,
                "prompt_id": prompt_id,
                "seed": seed,
                "text": payload["text"],
                "fourteen_line": int(metrics["non_empty_line_count"]) == 14,
                **metrics,
                **diagnostics,
            }
        )
    return rows


def _new_validator(payload: Mapping[str, Any], identity: str) -> None:
    if (
        payload.get("experiment_version") != NO_LABELS_CREATIVE_VERSION
        or payload.get("state_identity_sha256") != identity
        or payload.get("prompt_arm_id") != "explicit_no_labels_or_prose"
        or payload.get("sampling_recipe", {}).get("recipe_id") != "creative"
        or payload.get("v7_test_accessed") is not False
        or payload.get("training_performed") is not False
    ):
        raise ValueError("new-cell output lineage mismatch")


def _high_volume_validator(payload: Mapping[str, Any], identity: str) -> None:
    if (
        payload.get("generation_version") != HIGH_VOLUME_VERSION
        or payload.get("state_id") != "stage_3_selected"
        or payload.get("state_identity_sha256") != identity
        or payload.get("recipe", {}).get("recipe_id") != "creative"
        or payload.get("v7_test_accessed") is not False
    ):
        raise ValueError("high-volume comparator output lineage mismatch")


def _prompt_validator(payload: Mapping[str, Any], identity: str) -> None:
    if (
        payload.get("experiment_version") != PROMPT_INTERVENTION_VERSION
        or payload.get("state_identity_sha256") != identity
        or payload.get("arm_id") != "explicit_no_labels_or_prose"
        or payload.get("sampling_recipe", {}).get("temperature") != 0.7
        or payload.get("v7_test_accessed") is not False
        or payload.get("training_performed") is not False
    ):
        raise ValueError("prompt comparator output lineage mismatch")


def _summaries(
    rows: Sequence[Mapping[str, Any]], resamples: int, seed: int,
    confidence_level: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["cell_id"])].append(row)
    output = []
    for index, cell_id in enumerate(CELL_IDS):
        by_prompt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in grouped[cell_id]:
            by_prompt[str(row["prompt_id"])].append(row)
        output.append(
            _metric_summary(
                by_prompt, resamples, seed + index, confidence_level,
                {"cell_id": cell_id, "outputs": len(grouped[cell_id])},
            )
        )
    return output


def _paired_comparison(
    rows: Sequence[Mapping[str, Any]], *, left_cell_id: str, right_cell_id: str,
    resamples: int, seed: int, confidence_level: float,
) -> dict[str, Any]:
    lookup = {
        (str(row["cell_id"]), str(row["prompt_id"]), int(row["seed"])): row
        for row in rows
    }
    prompts = sorted({str(row["prompt_id"]) for row in rows})
    by_prompt: dict[str, list[dict[str, float]]] = defaultdict(list)
    for prompt in prompts:
        for current_seed in range(4200, 4208):
            left = lookup[(left_cell_id, prompt, current_seed)]
            right = lookup[(right_cell_id, prompt, current_seed)]
            by_prompt[prompt].append(
                {metric: float(right[metric]) - float(left[metric]) for metric in METRICS}
            )
    return _metric_summary(
        by_prompt, resamples, seed, confidence_level,
        {
            "left_cell_id": left_cell_id,
            "right_cell_id": right_cell_id,
            "paired_outputs": 960,
        },
    )


def _metric_summary(
    by_prompt: Mapping[str, Sequence[Mapping[str, Any]]], resamples: int, seed: int,
    confidence_level: float, base: Mapping[str, Any],
) -> dict[str, Any]:
    prompt_ids = sorted(by_prompt)
    result = {**base, "prompt_clusters": len(prompt_ids)}
    for metric_index, metric in enumerate(METRICS):
        means = {
            prompt: statistics.fmean(float(row[metric]) for row in by_prompt[prompt])
            for prompt in prompt_ids
        }
        rng = random.Random(seed + metric_index)
        distribution = sorted(
            statistics.fmean(means[rng.choice(prompt_ids)] for _ in prompt_ids)
            for _ in range(resamples)
        )
        alpha = (1 - confidence_level) / 2
        last = resamples - 1
        label = "mean_paired_change" if "left_cell_id" in base else "mean"
        result[metric] = {
            label: statistics.fmean(means.values()),
            "cluster_bootstrap_ci_low": distribution[int(alpha * last)],
            "cluster_bootstrap_ci_high": distribution[int((1 - alpha) * last)],
        }
    return result
