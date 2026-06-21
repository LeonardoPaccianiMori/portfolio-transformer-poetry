import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sample_preview(path: Path, max_characters: int = 200) -> str:
    if max_characters <= 0:
        raise ValueError("max_characters must be greater than 0")

    text = path.read_text(encoding="utf-8")
    preview = text[:max_characters]

    return preview.replace("\n", "\\n")


def checkpoint_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def summarize_training_run(
    run_dir: Path,
    sample_preview_characters: int = 200,
) -> dict[str, Any]:
    config = read_json(run_dir / "config.json")
    history = read_jsonl(run_dir / "loss_history.jsonl")

    if not history:
        raise ValueError(f"loss history is empty: {run_dir}")

    final_row = history[-1]
    best_validation_row = min(
        history,
        key=lambda row: row["validation_loss"],
    )

    return {
        "run_name": run_dir.name,
        "dataset": config["dataset"],
        "context_length": config["context_length"],
        "batch_size": config["batch_size"],
        "train_steps": config["train_steps"],
        "learning_rate": config["learning_rate"],
        "embedding_dim": config["embedding_dim"],
        "num_layers": config["num_layers"],
        "num_heads": config["num_heads"],
        "head_dim": config["head_dim"],
        "feed_forward_dim": config["feed_forward_dim"],
        "vocab_size": config["vocab_size"],
        "train_tokens": config["train_tokens"],
        "validation_tokens": config["validation_tokens"],
        "final_train_loss": final_row["train_loss"],
        "final_validation_loss": final_row["validation_loss"],
        "best_validation_loss": best_validation_row["validation_loss"],
        "best_validation_step": best_validation_row["step"],
        "checkpoint_size_mb": checkpoint_size_mb(run_dir / "model.pt"),
        "sample_preview": sample_preview(
            run_dir / "sample.txt",
            max_characters=sample_preview_characters,
        ),
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Run",
        "Ctx",
        "Batch",
        "Steps",
        "LR",
        "Emb",
        "Layers",
        "Heads",
        "FF",
        "Final Train",
        "Final Val",
        "Best Val",
        "Best Step",
        "Ckpt MB",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        values = [
            row["run_name"],
            str(row["context_length"]),
            str(row["batch_size"]),
            str(row["train_steps"]),
            f"{row['learning_rate']:.1e}",
            str(row["embedding_dim"]),
            str(row["num_layers"]),
            str(row["num_heads"]),
            str(row["feed_forward_dim"]),
            f"{row['final_train_loss']:.4f}",
            f"{row['final_validation_loss']:.4f}",
            f"{row['best_validation_loss']:.4f}",
            str(row["best_validation_step"]),
            f"{row['checkpoint_size_mb']:.2f}",
        ]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def markdown_sample_sections(rows: list[dict[str, Any]]) -> str:
    sections = []

    for row in rows:
        sections.append(f"### {row['run_name']}")
        sections.append("")
        sections.append("```text")
        sections.append(row["sample_preview"])
        sections.append("```")

    return "\n".join(sections)


def build_training_report(rows: list[dict[str, Any]]) -> str:
    sorted_rows = sorted(
        rows,
        key=lambda row: row["final_validation_loss"],
    )

    return "\n\n".join([
        "# Training Runs",
        "This report summarizes ignored raw training runs from `runs/`.",
        markdown_table(sorted_rows),
        "## Sample Previews",
        markdown_sample_sections(sorted_rows),
        "",
    ])


def write_training_report(
    run_dirs: list[Path],
    output_path: Path,
    sample_preview_characters: int = 200,
) -> list[dict[str, Any]]:
    rows = [
        summarize_training_run(
            run_dir=run_dir,
            sample_preview_characters=sample_preview_characters,
        )
        for run_dir in run_dirs
    ]
    report = build_training_report(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    return rows
