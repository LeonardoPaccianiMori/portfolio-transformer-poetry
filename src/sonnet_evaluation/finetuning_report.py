"""Write public reports for selected sonnet fine-tuning checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sonnet_training.finetuning_selection import load_loss_history


def summarize_finetuning_run(run_dir: Path, selection_path: Path) -> dict[str, Any]:
    """Load the facts needed to describe a fine-tuning run and its selection."""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    history = load_loss_history(run_dir / "loss_history.jsonl")
    if not history:
        raise ValueError("loss history is empty")

    return {
        "run_name": run_dir.name,
        "config": config,
        "selection": selection,
        "first_row": history[0],
        "final_row": history[-1],
        "history": history,
    }


def build_finetuning_markdown_report(summary: dict[str, Any]) -> str:
    """Render selection evidence without treating the final checkpoint as best."""
    config = summary["config"]
    selection = summary["selection"]
    first_row = summary["first_row"]
    final_row = summary["final_row"]
    parent_vocab_size = _parent_vocab_size(config)
    added_vocabulary = _added_vocabulary_description(config, parent_vocab_size)
    evaluation_description = _evaluation_description(config)
    selection_description = _selection_description(selection)
    selected_validation_loss = (
        f"{selection['best_validation_loss']:.4f}"
        if selection["exact_best_checkpoint_available"]
        else "not measured at this exact step"
    )
    history_rows = [
        "| Step | Training loss | Validation loss |",
        "| ---: | ---: | ---: |",
        *[
            "| {step:,} | {train_loss:.4f} | {validation_loss:.4f} |".format(**row)
            for row in summary["history"]
        ],
    ]
    return "\n\n".join([
        f"# Fine-Tuning Run: {summary['run_name']}",
        (
            "This report records sonnet specialization from the broader-Italian "
            "pretraining checkpoint. Checkpoints remain local-only; this report "
            "preserves the configuration, validation evidence, and selection rule."
        ),
        "## Configuration\n\n"
        + "\n".join([
            "| Setting | Value |",
            "| --- | --- |",
            f"| Parent checkpoint step | {config['parent_checkpoint_step']:,} |",
            f"| Parent vocabulary | {parent_vocab_size:,} |",
            f"| Fine-tuning vocabulary | {config['vocab_size']:,} |",
            f"| Added vocabulary entries | {added_vocabulary} |",
            f"| Context length | {config['context_length']} |",
            f"| Batch size | {config['batch_size']} |",
            f"| Learning rate | {config['learning_rate']:.1e} |",
            f"| Train steps | {config['completed_steps']:,} |",
            f"| Evaluation | {evaluation_description} |",
        ]),
        "## Checkpoint Selection\n\n"
        + "\n".join([
            "| Measurement | Step | Validation loss |",
            "| --- | ---: | ---: |",
            f"| Best recorded validation | {selection['best_validation_step']:,} | {selection['best_validation_loss']:.4f} |",
            f"| Selected saved checkpoint | {selection['selected_checkpoint_step']:,} | {selected_validation_loss} |",
        ])
        + "\n\n" + selection_description,
        "## Overfitting Evidence\n\n"
        + "\n".join([
            "| Measurement | Step | Training loss | Validation loss |",
            "| --- | ---: | ---: | ---: |",
            f"| First recorded evaluation | {first_row['step']:,} | {first_row['train_loss']:.4f} | {first_row['validation_loss']:.4f} |",
            f"| Final evaluation | {final_row['step']:,} | {final_row['train_loss']:.4f} | {final_row['validation_loss']:.4f} |",
        ]),
        "## Full Loss History\n\n" + "\n".join(history_rows),
    ]) + "\n"


def _parent_vocab_size(config: dict[str, Any]) -> int:
    if "parent_vocab_size" in config:
        return int(config["parent_vocab_size"])
    source_architecture = config.get("source_model_architecture", {})
    if "vocab_size" in source_architecture:
        return int(source_architecture["vocab_size"])
    return int(config["vocab_size"])


def _added_vocabulary_description(
    config: dict[str, Any],
    parent_vocab_size: int,
) -> str:
    literal_tokens = config.get("added_token_strings")
    if literal_tokens is not None:
        return ", ".join(literal_tokens)
    added_count = int(config["vocab_size"]) - parent_vocab_size
    return f"{added_count} (literal strings not recorded)"


def _evaluation_description(config: dict[str, Any]) -> str:
    if config.get("validation_mode") == "sequential_windows":
        return (
            f"every {config['eval_interval']:,} steps; "
            f"all {config['validation_window_count']:,} fixed sequential windows"
        )
    return (
        f"every {config['eval_interval']:,} steps; "
        f"{config['eval_batches']} random batches"
    )


def _selection_description(selection: dict[str, Any]) -> str:
    if selection["exact_best_checkpoint_available"]:
        return (
            "The selected saved checkpoint is the exact checkpoint from the best "
            "recorded validation step. It is selected for evaluation instead of the "
            "final checkpoint because validation worsened after the best step."
        )
    return (
        "The selected checkpoint is the latest interval checkpoint at or before "
        "the best validation measurement. It is selected for evaluation because "
        "the final checkpoint is strongly overfit."
    )


def write_finetuning_markdown_report(
    run_dir: Path,
    selection_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write one public Markdown fine-tuning report."""
    summary = summarize_finetuning_run(run_dir, selection_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_finetuning_markdown_report(summary),
        encoding="utf-8",
    )
    return summary
