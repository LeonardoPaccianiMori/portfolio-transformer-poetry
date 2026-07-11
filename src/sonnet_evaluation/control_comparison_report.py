"""Build a reproducible report for the two-arm sonnet initialization control."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from sonnet_evaluation.memorization import (
    load_training_records,
    score_generation_memorization,
)
from sonnet_evaluation.metrics import score_generation_directory


COMPARABILITY_FIELDS = (
    "dataset",
    "vocab_size",
    "train_tokens",
    "validation_tokens",
    "model_architecture",
    "batch_size",
    "context_length",
    "train_steps",
    "learning_rate",
    "seed",
    "optimizer_state_restored",
    "learning_rate_schedule",
    "warmup_steps",
    "min_learning_rate",
)
COMPARABILITY_DEFAULTS = {
    "learning_rate_schedule": "constant",
    "warmup_steps": 0,
    "min_learning_rate": 0.0,
}


def summarize_control_arm(
    *,
    run_dir: Path,
    generation_dir: Path,
    manifest_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Collect loss, generation, and copying diagnostics for one control arm."""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    history = [
        json.loads(line)
        for line in (run_dir / "loss_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not history:
        raise ValueError(f"loss history is empty: {run_dir}")

    metrics_rows = score_generation_directory(generation_dir)
    training_records = load_training_records(
        manifest_path=manifest_path,
        repo_root=repo_root,
        dataset=config["dataset"],
        split="train",
    )
    memorization_rows = score_generation_memorization(
        generation_dir=generation_dir,
        training_records=training_records,
    )
    best_row = min(history, key=lambda row: row["validation_loss"])
    final_row = history[-1]

    return {
        "name": run_dir.name,
        "config": config,
        "best_row": best_row,
        "final_row": final_row,
        "metrics_rows": metrics_rows,
        "memorization_rows": memorization_rows,
    }


def control_arms_are_comparable(
    pretrained: dict[str, Any],
    random: dict[str, Any],
    allowed_difference_fields: set[str] | None = None,
) -> bool:
    """Check that all declared shared experimental settings match."""
    allowed_difference_fields = allowed_difference_fields or set()
    return all(
        _config_value(pretrained["config"], field)
        == _config_value(random["config"], field)
        for field in COMPARABILITY_FIELDS
        if field not in allowed_difference_fields
    )


def _config_value(config: dict[str, Any], field: str) -> Any:
    return config.get(field, COMPARABILITY_DEFAULTS.get(field))


def average_metric(rows: list[dict[str, Any]], field: str) -> float:
    """Return a mean metric across fixed prompts."""
    if not rows:
        raise ValueError("metric rows must not be empty")
    return mean(float(row[field]) for row in rows)


def risk_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count heuristic memorization labels across fixed prompts."""
    return {
        level: sum(row["risk_level"] == level for row in rows)
        for level in ("low", "medium", "high")
    }


def build_control_comparison_markdown(
    pretrained: dict[str, Any],
    random: dict[str, Any],
    *,
    experiment_name: str = "Initialization",
    left_label: str = "pretrained",
    right_label: str = "random",
    intended_difference: str = (
        "broader-pretrained weights versus random weights"
    ),
    allowed_difference_fields: set[str] | None = None,
) -> str:
    """Render the causal comparison while preserving its experimental limits."""
    comparable = control_arms_are_comparable(
        pretrained,
        random,
        allowed_difference_fields=allowed_difference_fields,
    )
    pretrained_best = pretrained["best_row"]
    random_best = random["best_row"]
    validation_gap = (
        float(random_best["validation_loss"])
        - float(pretrained_best["validation_loss"])
    )
    if validation_gap >= 0:
        validation_interpretation = (
            f"The {left_label} arm achieved a lower best validation loss by "
            f"{validation_gap:.4f} nats per BPE token."
        )
    else:
        validation_interpretation = (
            f"The {right_label} arm achieved a lower best validation loss by "
            f"{-validation_gap:.4f} nats per BPE token."
        )
    rows = []
    for label, arm in ((left_label, pretrained), (right_label, random)):
        metrics_rows = arm["metrics_rows"]
        risks = risk_counts(arm["memorization_rows"])
        rows.append([
            label,
            f"{arm['best_row']['step']:,}",
            f"{arm['best_row']['validation_loss']:.4f}",
            f"{arm['final_row']['validation_loss']:.4f}",
            f"{average_metric(metrics_rows, 'character_count'):.1f}",
            f"{average_metric(metrics_rows, 'non_empty_line_count'):.1f}",
            f"{average_metric(metrics_rows, 'repetition_ratio'):.4f}",
            f"{average_metric(metrics_rows, 'unique_character_ratio'):.4f}",
            f"{risks['low']}/5",
        ])

    table_lines = [
        "| Initialization | Best Step | Best Val | Final Val | Avg Chars | Avg Lines | Avg Repeated 4-gram Ratio | Avg Unique-Character Ratio | Low Memorization Risk |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]
    return "\n\n".join([
        f"# Controlled Experiment Comparison: {experiment_name}",
        (
            "This report compares two sonnet-corpus training runs with matching "
            "architecture, tokenizer, data, optimizer initialization, seed, and "
            "training schedule except for the declared experimental factor. The "
            f"intended difference is {intended_difference}."
        ),
        "## Comparability\n\n"
        + (
            "All declared shared settings match."
            if comparable
            else "Warning: one or more declared shared settings differ."
        ),
        "## Results\n\n" + "\n".join(table_lines),
        "## Interpretation\n\n"
        + validation_interpretation
        + " Both arms use the same tokenizer, so this loss difference is directly comparable.",
        (
            "The 14-line generation target is decoder enforced in both arms. "
            "Automatic format metrics therefore validate the control procedure, not "
            "learned sonnet structure. Memorization labels are heuristic surface-copying "
            "checks. Qualitative reviews must be read alongside this report."
        ),
        "## Limits\n\n"
        + (
            "This is one random seed with validation estimated from five sampled "
            "batches at each evaluation. It supports the value of broader pretraining "
            "under this setup, but it does not establish a general result without "
            "additional seeds and more stable validation estimates."
        ),
    ]) + "\n"


def write_control_comparison_report(
    *,
    pretrained_run_dir: Path,
    pretrained_generation_dir: Path,
    random_run_dir: Path,
    random_generation_dir: Path,
    manifest_path: Path,
    repo_root: Path,
    output_path: Path,
    experiment_name: str = "Initialization",
    left_label: str = "pretrained",
    right_label: str = "random",
    intended_difference: str = (
        "broader-pretrained weights versus random weights"
    ),
    allowed_difference_fields: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Write one public comparison report from two local control-run artifacts."""
    pretrained = summarize_control_arm(
        run_dir=pretrained_run_dir,
        generation_dir=pretrained_generation_dir,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    random = summarize_control_arm(
        run_dir=random_run_dir,
        generation_dir=random_generation_dir,
        manifest_path=manifest_path,
        repo_root=repo_root,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_control_comparison_markdown(
            pretrained,
            random,
            experiment_name=experiment_name,
            left_label=left_label,
            right_label=right_label,
            intended_difference=intended_difference,
            allowed_difference_fields=allowed_difference_fields,
        ),
        encoding="utf-8",
    )
    return {"pretrained": pretrained, "random": random}
