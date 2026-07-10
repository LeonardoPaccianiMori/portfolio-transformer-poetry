import json
from pathlib import Path

import pytest
import torch

from sonnet_evaluation.finetuning_report import (
    build_finetuning_markdown_report,
    summarize_finetuning_run,
    write_finetuning_markdown_report,
)
from sonnet_training.finetuning_selection import (
    build_finetuning_checkpoint_selection,
    select_checkpoint_at_or_before,
    write_finetuning_checkpoint_selection,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_fake_finetuning_run(repo_root: Path) -> Path:
    parent_path = repo_root / "runs" / "parent" / "model.pt"
    parent_path.parent.mkdir(parents=True)
    torch.save(
        {
            "config": {
                "embedding_dim": 8,
                "num_layers": 1,
                "num_heads": 2,
                "head_dim": 4,
                "feed_forward_dim": 16,
                "max_context_length": 8,
            },
        },
        parent_path,
    )
    run_dir = repo_root / "runs" / "finetuning"
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "config.json",
        {
            "pretraining_checkpoint_path": "runs/parent/model.pt",
            "parent_checkpoint_step": 100,
            "parent_vocab_size": 50,
            "vocab_size": 52,
            "added_token_strings": ["ü", "ʼ"],
            "context_length": 8,
            "batch_size": 2,
            "learning_rate": 3e-5,
            "completed_steps": 3_000,
            "eval_interval": 250,
            "eval_batches": 5,
        },
    )
    (run_dir / "loss_history.jsonl").write_text(
        "\n".join([
            json.dumps({"step": 1, "train_loss": 3.0, "validation_loss": 3.2}),
            json.dumps({"step": 2_000, "train_loss": 2.0, "validation_loss": 2.4}),
            json.dumps({"step": 2_500, "train_loss": 1.8, "validation_loss": 2.1}),
            json.dumps({"step": 3_000, "train_loss": 0.1, "validation_loss": 4.0}),
        ])
        + "\n",
        encoding="utf-8",
    )
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    for step in (1_000, 2_000, 3_000):
        (checkpoint_dir / f"step_{step}.pt").write_bytes(b"checkpoint")
    return run_dir


def test_select_checkpoint_at_or_before_uses_best_validation_and_prior_checkpoint(tmp_path):
    run_dir = write_fake_finetuning_run(tmp_path)
    history = [
        json.loads(line)
        for line in (run_dir / "loss_history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    checkpoints = {
        1_000: run_dir / "checkpoints" / "step_1000.pt",
        2_000: run_dir / "checkpoints" / "step_2000.pt",
        3_000: run_dir / "checkpoints" / "step_3000.pt",
    }

    best_row, selected_step, selected_path = select_checkpoint_at_or_before(
        history,
        checkpoints,
    )

    assert best_row["step"] == 2_500
    assert selected_step == 2_000
    assert selected_path.name == "step_2000.pt"


def test_select_checkpoint_at_or_before_rejects_missing_checkpoints():
    with pytest.raises(ValueError, match="no interval checkpoints"):
        select_checkpoint_at_or_before(
            [{"step": 1, "validation_loss": 1.0}],
            {},
        )


def test_selection_manifest_contains_parent_architecture_and_lineage(tmp_path):
    run_dir = write_fake_finetuning_run(tmp_path)

    selection = build_finetuning_checkpoint_selection(
        repo_root=tmp_path,
        run_dir=run_dir,
    )

    assert selection["selected_checkpoint_step"] == 2_000
    assert selection["best_validation_step"] == 2_500
    assert selection["exact_best_checkpoint_available"] is False
    assert selection["model_architecture"]["vocab_size"] == 52
    assert selection["model_architecture"]["embedding_dim"] == 8


def test_write_selection_and_public_report(tmp_path):
    run_dir = write_fake_finetuning_run(tmp_path)
    selection_path = run_dir / "selected_checkpoint.json"
    report_path = tmp_path / "reports" / "finetuning.md"

    write_finetuning_checkpoint_selection(
        repo_root=tmp_path,
        run_dir=run_dir,
        output_path=selection_path,
    )
    summary = write_finetuning_markdown_report(
        run_dir,
        selection_path,
        report_path,
    )

    report = report_path.read_text(encoding="utf-8")
    assert summary["selection"]["selected_checkpoint_step"] == 2_000
    assert "| Best recorded validation | 2,500 | 2.1000 |" in report
    assert "| Final evaluation | 3,000 | 0.1000 | 4.0000 |" in report


def test_build_finetuning_markdown_report_uses_loaded_summary(tmp_path):
    run_dir = write_fake_finetuning_run(tmp_path)
    selection_path = run_dir / "selected_checkpoint.json"
    write_finetuning_checkpoint_selection(
        repo_root=tmp_path,
        run_dir=run_dir,
        output_path=selection_path,
    )

    report = build_finetuning_markdown_report(
        summarize_finetuning_run(run_dir, selection_path)
    )

    assert "# Fine-Tuning Run: finetuning" in report
    assert "latest interval checkpoint at or before" in report
