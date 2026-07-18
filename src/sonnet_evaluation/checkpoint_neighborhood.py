"""Generate fixed-prompt checkpoint-neighborhood evaluation batches."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from sonnet_evaluation.generation import generate_for_prompts, load_prompts


def load_checkpoint_neighborhood_plan(path: Path) -> dict[str, Any]:
    """Load and validate one reproducible checkpoint-neighborhood plan."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
        raise ValueError("checkpoint neighborhood plan must contain a runs list")

    for run in payload["runs"]:
        if not isinstance(run, dict):
            raise ValueError("each planned run must be a JSON object")
        _require_keys(run, ("id", "run_dir", "selected_checkpoint_id", "checkpoints"))
        checkpoints = run["checkpoints"]
        if not isinstance(checkpoints, list) or not checkpoints:
            raise ValueError("each planned run must contain checkpoints")
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict):
                raise ValueError("each checkpoint must be a JSON object")
            _require_keys(
                checkpoint,
                ("id", "checkpoint_path", "step", "validation_loss"),
            )
        if len({checkpoint["id"] for checkpoint in checkpoints}) != len(checkpoints):
            raise ValueError("checkpoint identifiers must be unique within each run")
        checkpoint_ids = {checkpoint["id"] for checkpoint in checkpoints}
        if run["selected_checkpoint_id"] not in checkpoint_ids:
            raise ValueError("selected_checkpoint_id must name a planned checkpoint")

    return payload


def generate_checkpoint_neighborhoods(
    *,
    repo_root: Path,
    plan_path: Path,
    prompts_path: Path,
    output_root: Path,
    max_new_tokens: int,
    seed: int,
    device: torch.device | str,
    temperature: float = 1.0,
    top_k: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate every planned checkpoint batch and write local run metadata."""

    plan = load_checkpoint_neighborhood_plan(plan_path)
    prompts = load_prompts(prompts_path)
    output_root.mkdir(parents=True, exist_ok=True)
    planned_runs = plan["runs"]
    generated_runs = []

    for run_index, run in enumerate(planned_runs, start=1):
        run_id = run["id"]
        run_dir = _resolve_path(repo_root, run["run_dir"])
        checkpoints = run["checkpoints"]
        _report_progress(
            progress,
            f"run {run_index}/{len(planned_runs)}: {run_id}",
        )
        generated_checkpoints = []

        for checkpoint_index, checkpoint in enumerate(checkpoints, start=1):
            checkpoint_id = checkpoint["id"]
            output_dir = output_root / run_id / checkpoint_id
            _report_progress(
                progress,
                f"{run_id} checkpoint {checkpoint_index}/{len(checkpoints)}: "
                f"{checkpoint_id} (step {checkpoint['step']})",
            )
            generation_metadata = generate_for_prompts(
                run_dir=run_dir,
                prompts=prompts,
                output_dir=output_dir,
                max_new_tokens=max_new_tokens,
                seed=seed,
                device=device,
                temperature=temperature,
                top_k=top_k,
                checkpoint_path=_resolve_path(repo_root, checkpoint["checkpoint_path"]),
                model_config_path=run_dir / "config.json",
                progress=(
                    lambda message, run_id=run_id, checkpoint_id=checkpoint_id: (
                        _report_progress(
                            progress,
                            f"{run_id}/{checkpoint_id} | {message}",
                        )
                    )
                ),
            )
            generated_checkpoints.append({
                **checkpoint,
                "output_dir": str(output_dir),
                "generation_metadata": generation_metadata,
            })

        generated_runs.append({
            "id": run_id,
            "run_dir": str(run_dir),
            "selected_checkpoint_id": run["selected_checkpoint_id"],
            "checkpoints": generated_checkpoints,
        })

    metadata = {
        "plan_path": str(plan_path),
        "prompts_path": str(prompts_path),
        "output_root": str(output_root),
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "device": str(device),
        "temperature": temperature,
        "top_k": top_k,
        "runs": generated_runs,
    }
    metadata_path = output_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _report_progress(progress, f"wrote neighborhood metadata: {metadata_path}")
    return metadata


def _require_keys(payload: dict[str, Any], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError("plan item is missing fields: " + ", ".join(missing))


def _resolve_path(repo_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_root / path


def _report_progress(
    progress: Callable[[str], None] | None,
    message: str,
) -> None:
    if progress is not None:
        progress(message)
