"""Summarize fixed-prompt checkpoint-neighborhood generation batches."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from sonnet_evaluation.metrics import score_generation_directory


def summarize_checkpoint_neighborhoods(
    metadata_path: Path,
    ngram_size: int = 4,
) -> list[dict[str, Any]]:
    """Score every checkpoint batch recorded in neighborhood metadata."""

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    runs = metadata.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("neighborhood metadata must contain one or more runs")

    summaries = []
    for run in runs:
        _require_mapping(run, "run")
        run_id = _require_string(run, "id")
        selected_checkpoint_id = _require_string(run, "selected_checkpoint_id")
        checkpoints = run.get("checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints:
            raise ValueError(f"run {run_id} must contain one or more checkpoints")

        checkpoint_ids = set()
        for checkpoint in checkpoints:
            _require_mapping(checkpoint, "checkpoint")
            checkpoint_id = _require_string(checkpoint, "id")
            checkpoint_ids.add(checkpoint_id)
            output_dir = Path(_require_string(checkpoint, "output_dir"))
            metrics_rows = score_generation_directory(
                generation_dir=output_dir,
                ngram_size=ngram_size,
            )
            if not metrics_rows:
                raise ValueError(
                    f"checkpoint batch contains no generated files: {output_dir}"
                )
            summaries.append({
                "run_id": run_id,
                "checkpoint_id": checkpoint_id,
                "selected_by_validation": checkpoint_id == selected_checkpoint_id,
                "step": _require_number(checkpoint, "step"),
                "validation_loss": _require_number(checkpoint, "validation_loss"),
                "generation_count": len(metrics_rows),
                "average_character_count": _average_metric(
                    metrics_rows,
                    "character_count",
                ),
                "average_non_empty_line_count": _average_metric(
                    metrics_rows,
                    "non_empty_line_count",
                ),
                "average_repetition_ratio": _average_metric(
                    metrics_rows,
                    "repetition_ratio",
                ),
                "average_unique_character_ratio": _average_metric(
                    metrics_rows,
                    "unique_character_ratio",
                ),
                "all_prompts_preserved": all(
                    row["prompt_preserved"] for row in metrics_rows
                ),
            })

        if selected_checkpoint_id not in checkpoint_ids:
            raise ValueError(
                f"run {run_id} selects a checkpoint that was not generated: "
                f"{selected_checkpoint_id}"
            )

    return summaries


def build_checkpoint_neighborhood_report(
    summaries: list[dict[str, Any]],
    *,
    ngram_size: int = 4,
) -> str:
    """Render automatic diagnostics for all neighborhood checkpoint batches."""

    if not summaries:
        raise ValueError("neighborhood summaries must not be empty")

    rows = []
    for summary in summaries:
        rows.append([
            summary["run_id"],
            summary["checkpoint_id"],
            "yes" if summary["selected_by_validation"] else "no",
            f"{summary['step']:,}",
            f"{summary['validation_loss']:.4f}",
            str(summary["generation_count"]),
            f"{summary['average_character_count']:.1f}",
            f"{summary['average_non_empty_line_count']:.1f}",
            f"{summary['average_repetition_ratio']:.4f}",
            f"{summary['average_unique_character_ratio']:.4f}",
            "yes" if summary["all_prompts_preserved"] else "no",
        ])

    table = "\n".join([
        "| Parent Run | Checkpoint | Validation Selected | Step | Validation Loss | Prompts | Avg Chars | Avg Non-empty Lines | Avg Repeated Character-4-gram Ratio | Avg Unique-Character Ratio | Prompts Preserved |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])

    return "\n\n".join([
        "# Pretraining Checkpoint-Neighborhood Evaluation",
        (
            "Each parent run is evaluated at the checkpoint selected by its lowest "
            "deterministic validation loss and at the nearest planned checkpoints "
            "before and after it. Every batch uses the same five prompts, fixed "
            "seeds, temperature 1.0, and a 300-token limit."
        ),
        "## Automatic Diagnostics\n\n" + table,
        "## Interpretation Rules\n\n"
        + "\n".join([
            "- The validation-selected checkpoint remains the model-selection checkpoint. Neighbor outputs are a stability diagnostic, not a basis for cherry-picking a different checkpoint.",
            f"- Repetition is measured as the proportion of repeated character {ngram_size}-grams within each output, then averaged across the five prompts. Lower values can indicate less local looping, but do not by themselves establish better prose.",
            "- Prompt preservation must be `yes`; otherwise the generation procedure is invalid for that batch.",
            "- These automatic measurements must be read together with qualitative inspection of the matched outputs. They do not measure grammaticality, historical style, factual consistency, or literary quality.",
        ]),
        "## Selection Scope\n\n"
        + (
            "This report evaluates checkpoint stability within each already-trained "
            "parent run. It does not compare training cost, training-corpus coverage, "
            "or fine-tuned sonnet quality, which remain separate selection criteria."
        ),
    ]) + "\n"


def write_checkpoint_neighborhood_report(
    *,
    metadata_path: Path,
    output_path: Path,
    ngram_size: int = 4,
) -> list[dict[str, Any]]:
    """Score local generations and write their public aggregate report."""

    summaries = summarize_checkpoint_neighborhoods(
        metadata_path=metadata_path,
        ngram_size=ngram_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_checkpoint_neighborhood_report(
            summaries,
            ngram_size=ngram_size,
        ),
        encoding="utf-8",
    )
    return summaries


def _average_metric(rows: list[dict[str, Any]], field: str) -> float:
    return mean(float(row[field]) for row in rows)


def _require_mapping(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")


def _require_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing or invalid {field}")
    return value


def _require_number(payload: dict[str, Any], field: str) -> float | int:
    value = payload.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"missing or invalid {field}")
    return value
