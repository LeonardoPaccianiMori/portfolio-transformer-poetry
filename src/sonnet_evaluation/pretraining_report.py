"""Create public summaries of ignored broader-corpus pretraining runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _required_file(run_dir: Path, filename: str) -> Path:
    """Return one required run artifact or raise a clear error."""
    path = run_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"required run artifact is missing: {path}")
    return path


def load_loss_history(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL loss records in the order the runner wrote them."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def checkpoint_count(run_dir: Path) -> int:
    """Count interval checkpoints without treating the final model as one."""
    checkpoint_dir = run_dir / "checkpoints"
    return len(list(checkpoint_dir.glob("step_*.pt")))


def sample_excerpt(path: Path, max_characters: int = 800) -> str:
    """Read a bounded sample excerpt suitable for a checked-in Markdown report."""
    if max_characters <= 0:
        raise ValueError("max_characters must be greater than 0")

    text = path.read_text(encoding="utf-8").strip()
    if len(text) <= max_characters:
        return text

    return text[:max_characters].rstrip() + "\n[excerpt truncated]"


def summarize_pretraining_run(
    run_dir: Path,
    sample_excerpt_characters: int = 800,
) -> dict[str, Any]:
    """Extract the public facts needed to interpret one local pretraining run."""
    config_path = _required_file(run_dir, "config.json")
    history_path = _required_file(run_dir, "loss_history.jsonl")
    sample_path = _required_file(run_dir, "sample.txt")
    model_path = _required_file(run_dir, "model.pt")
    _required_file(run_dir, "tokenizer.json")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    history = load_loss_history(history_path)
    if not history:
        raise ValueError(f"loss history is empty: {history_path}")

    first_row = history[0]
    final_row = history[-1]
    best_validation_row = min(history, key=lambda row: row["validation_loss"])

    return {
        "run_name": run_dir.name,
        "resolved_device": config["resolved_device"],
        "vocab_size": config["vocab_size"],
        "train_tokens": config["train_tokens"],
        "validation_tokens": config["validation_tokens"],
        "context_length": config["context_length"],
        "batch_size": config["batch_size"],
        "train_steps": config["train_steps"],
        "completed_steps": config["completed_steps"],
        "eval_interval": config["eval_interval"],
        "eval_batches": config["eval_batches"],
        "checkpoint_interval": config["checkpoint_interval"],
        "learning_rate": config["learning_rate"],
        "embedding_dim": config["embedding_dim"],
        "num_layers": config["num_layers"],
        "num_heads": config["num_heads"],
        "head_dim": config["head_dim"],
        "feed_forward_dim": config["feed_forward_dim"],
        "normalization_type": config.get("normalization_type") or "layer_norm",
        "normalization_eps": float(config.get("normalization_eps") or 1e-5),
        "position_encoding_type": (
            config.get("position_encoding_type") or "learned_absolute"
        ),
        "rope_theta": float(config.get("rope_theta") or 10_000.0),
        "parameter_count": config["parameter_count"],
        "first_step": first_row["step"],
        "first_train_loss": first_row["train_loss"],
        "first_validation_loss": first_row["validation_loss"],
        "final_train_loss": final_row["train_loss"],
        "final_validation_loss": final_row["validation_loss"],
        "best_validation_step": best_validation_row["step"],
        "best_validation_train_loss": best_validation_row["train_loss"],
        "best_validation_loss": best_validation_row["validation_loss"],
        "loss_records": len(history),
        "interval_checkpoint_count": checkpoint_count(run_dir),
        "final_checkpoint_size_mib": model_path.stat().st_size / (1024 * 1024),
        "sample_excerpt": sample_excerpt(
            sample_path,
            max_characters=sample_excerpt_characters,
        ),
        "loss_history": history,
    }


def _configuration_table(summary: dict[str, Any]) -> str:
    rows = [
        ("Device", summary["resolved_device"]),
        ("Vocabulary size", f"{summary['vocab_size']:,}"),
        ("Training tokens", f"{summary['train_tokens']:,}"),
        ("Validation tokens", f"{summary['validation_tokens']:,}"),
        ("Context length", summary["context_length"]),
        ("Batch size", summary["batch_size"]),
        ("Completed steps", f"{summary['completed_steps']:,}"),
        ("Learning rate", f"{summary['learning_rate']:.1e}"),
        ("Evaluation", f"every {summary['eval_interval']:,} steps; {summary['eval_batches']} batches"),
        ("Interval checkpoints", f"every {summary['checkpoint_interval']:,} steps"),
        ("Parameters", f"{summary['parameter_count']:,}"),
        ("Embedding dimension", summary["embedding_dim"]),
        ("Transformer layers", summary["num_layers"]),
        ("Attention heads", summary["num_heads"]),
        ("Head dimension", summary["head_dim"]),
        ("Feed-forward dimension", summary["feed_forward_dim"]),
        ("Normalization", summary["normalization_type"]),
        ("Normalization epsilon", f"{summary['normalization_eps']:.1e}"),
        ("Position encoding", summary["position_encoding_type"]),
        ("RoPE theta", f"{summary['rope_theta']:g}"),
    ]
    lines = ["| Setting | Value |", "| --- | --- |"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return "\n".join(lines)


def _loss_history_table(history: list[dict[str, Any]]) -> str:
    lines = [
        "| Step | Training loss | Validation loss |",
        "| ---: | ---: | ---: |",
    ]
    lines.extend(
        "| {step:,} | {train_loss:.4f} | {validation_loss:.4f} |".format(**row)
        for row in history
    )
    return "\n".join(lines)


def build_pretraining_markdown_report(summary: dict[str, Any]) -> str:
    """Render one summarized run as a self-contained public Markdown report."""
    return "\n\n".join([
        f"# Pretraining Run: {summary['run_name']}",
        (
            "This report records a from-scratch broader-Italian-corpus pretraining "
            "run. Raw corpus files, interval checkpoints, and the final checkpoint "
            "are intentionally local-only; the configuration and observed results are "
            "preserved here."
        ),
        "## Configuration\n\n" + _configuration_table(summary),
        "## Loss Summary\n\n"
        + "\n".join([
            "| Measurement | Step | Training loss | Validation loss |",
            "| --- | ---: | ---: | ---: |",
            (
                f"| First recorded evaluation | {summary['first_step']:,} | "
                f"{summary['first_train_loss']:.4f} | "
                f"{summary['first_validation_loss']:.4f} |"
            ),
            (
                f"| Best validation evaluation | {summary['best_validation_step']:,} | "
                f"{summary['best_validation_train_loss']:.4f} | "
                f"{summary['best_validation_loss']:.4f} |"
            ),
            (
                f"| Final evaluation | {summary['completed_steps']:,} | "
                f"{summary['final_train_loss']:.4f} | "
                f"{summary['final_validation_loss']:.4f} |"
            ),
        ]),
        "## Saved Local Artifacts\n\n"
        + "\n".join([
            f"- Interval checkpoints: {summary['interval_checkpoint_count']}",
            f"- Final checkpoint size: {summary['final_checkpoint_size_mib']:.1f} MiB",
            f"- Loss-history records: {summary['loss_records']}",
        ]),
        "## Final Sample Excerpt\n\n```text\n"
        + summary["sample_excerpt"]
        + "\n```",
        "## Interpretation\n\n"
        + (
            "The loss fell substantially from the first recorded evaluation, and the "
            "sample has learned historical Italian prose-like texture. It is not "
            "sonnet-specialized: that is the intended role of the next fine-tuning "
            "stage. The best validation value is a noisy estimate because each "
            "evaluation used only a small random batch sample; it should guide, not "
            "replace, later controlled evaluation."
        ),
        "## Full Loss History\n\n" + _loss_history_table(summary["loss_history"]),
    ]) + "\n"


def write_pretraining_markdown_report(run_dir: Path, output_path: Path) -> dict[str, Any]:
    """Create a public report from one ignored pretraining-run directory."""
    summary = summarize_pretraining_run(run_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_pretraining_markdown_report(summary),
        encoding="utf-8",
    )
    return summary
