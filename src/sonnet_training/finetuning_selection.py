"""Select a saved fine-tuning checkpoint from validation-loss evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_PATTERN = re.compile(r"step_(\d+)\.pt$")
MODEL_ARCHITECTURE_KEYS = (
    "embedding_dim",
    "num_layers",
    "num_heads",
    "head_dim",
    "feed_forward_dim",
    "max_context_length",
)


def load_loss_history(path: Path) -> list[dict[str, Any]]:
    """Load ordered non-empty fine-tuning loss records from JSONL."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def interval_checkpoints(checkpoint_dir: Path) -> dict[int, Path]:
    """Map saved interval checkpoint steps to paths."""
    checkpoints: dict[int, Path] = {}
    for path in checkpoint_dir.glob("step_*.pt"):
        match = CHECKPOINT_PATTERN.match(path.name)
        if match is not None:
            checkpoints[int(match.group(1))] = path
    return checkpoints


def select_checkpoint_at_or_before(
    history: list[dict[str, Any]],
    checkpoints: dict[int, Path],
) -> tuple[dict[str, Any], int, Path]:
    """Select the latest available checkpoint no later than best validation."""
    if not history:
        raise ValueError("loss history is empty")
    if not checkpoints:
        raise ValueError("no interval checkpoints are available")

    best_validation_row = min(history, key=lambda row: row["validation_loss"])
    best_step = int(best_validation_row["step"])
    eligible_steps = [step for step in checkpoints if step <= best_step]
    if not eligible_steps:
        raise ValueError("no checkpoint exists at or before the best validation step")

    selected_step = max(eligible_steps)
    return best_validation_row, selected_step, checkpoints[selected_step]


def build_finetuning_checkpoint_selection(
    *,
    repo_root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """Create a selection manifest with parent architecture and checkpoint lineage."""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    history = load_loss_history(run_dir / "loss_history.jsonl")
    best_row = min(history, key=lambda row: row["validation_loss"])
    best_checkpoint_path = run_dir / "best_validation.pt"
    if best_checkpoint_path.is_file():
        selected_step = int(best_row["step"])
        selected_path = best_checkpoint_path
        selection_rule = "exact_best_validation_checkpoint"
    else:
        _, selected_step, selected_path = select_checkpoint_at_or_before(
            history,
            interval_checkpoints(run_dir / "checkpoints"),
        )
        selection_rule = "latest_interval_checkpoint_at_or_before_best_validation"
    model_architecture = _model_architecture_from_run_config(
        repo_root=repo_root,
        config=config,
    )

    return {
        "fine_tuning_run_dir": str(run_dir),
        "parent_checkpoint_path": config.get("pretraining_checkpoint_path"),
        "parent_checkpoint_step": config.get("parent_checkpoint_step"),
        "best_validation_step": int(best_row["step"]),
        "best_validation_loss": float(best_row["validation_loss"]),
        "selected_checkpoint_step": selected_step,
        "selected_checkpoint_path": str(selected_path),
        "selection_rule": selection_rule,
        "exact_best_checkpoint_available": selected_step == int(best_row["step"]),
        "model_architecture": model_architecture,
    }


def _model_architecture_from_run_config(
    *,
    repo_root: Path,
    config: dict[str, Any],
) -> dict[str, int | float | str]:
    if "model_architecture" in config:
        return {
            name: int(config["model_architecture"][name])
            for name in ("vocab_size", *MODEL_ARCHITECTURE_KEYS)
        } | {
            "normalization_type": config["model_architecture"].get(
                "normalization_type",
                "layer_norm",
            ),
            "normalization_eps": float(
                config["model_architecture"].get("normalization_eps", 1e-5)
            ),
        }

    parent_path = repo_root / config["pretraining_checkpoint_path"]
    parent_checkpoint = torch.load(parent_path, map_location="cpu")
    parent_config = parent_checkpoint["config"]
    return {
        "vocab_size": int(config["vocab_size"]),
        **{
            name: int(parent_config[name])
            for name in MODEL_ARCHITECTURE_KEYS
        },
        "normalization_type": parent_config.get("normalization_type", "layer_norm"),
        "normalization_eps": float(parent_config.get("normalization_eps", 1e-5)),
    }


def write_finetuning_checkpoint_selection(
    *,
    repo_root: Path,
    run_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write one ignored selection manifest for later generation/evaluation."""
    selection = build_finetuning_checkpoint_selection(
        repo_root=repo_root,
        run_dir=run_dir,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return selection
