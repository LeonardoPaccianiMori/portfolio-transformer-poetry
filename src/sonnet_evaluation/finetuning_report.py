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
            f"| Parent vocabulary | {config['parent_vocab_size']:,} |",
            f"| Fine-tuning vocabulary | {config['vocab_size']:,} |",
            f"| Added literal tokens | {', '.join(config['added_token_strings'])} |",
            f"| Context length | {config['context_length']} |",
            f"| Batch size | {config['batch_size']} |",
            f"| Learning rate | {config['learning_rate']:.1e} |",
            f"| Train steps | {config['completed_steps']:,} |",
            f"| Evaluation | every {config['eval_interval']:,} steps; {config['eval_batches']} batches |",
        ]),
        "## Checkpoint Selection\n\n"
        + "\n".join([
            "| Measurement | Step | Validation loss |",
            "| --- | ---: | ---: |",
            f"| Best recorded validation | {selection['best_validation_step']:,} | {selection['best_validation_loss']:.4f} |",
            f"| Selected saved checkpoint | {selection['selected_checkpoint_step']:,} | not measured at this exact step |",
        ])
        + "\n\n"
        + (
            "The selected checkpoint is the latest interval checkpoint at or before "
            "the best validation measurement. It is selected for evaluation because "
            "the final checkpoint is strongly overfit."
        ),
        "## Overfitting Evidence\n\n"
        + "\n".join([
            "| Measurement | Step | Training loss | Validation loss |",
            "| --- | ---: | ---: | ---: |",
            f"| First recorded evaluation | {first_row['step']:,} | {first_row['train_loss']:.4f} | {first_row['validation_loss']:.4f} |",
            f"| Final evaluation | {final_row['step']:,} | {final_row['train_loss']:.4f} | {final_row['validation_loss']:.4f} |",
        ]),
        "## Full Loss History\n\n" + "\n".join(history_rows),
    ]) + "\n"


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
