import json
from pathlib import Path

import pytest

from sonnet_evaluation.training_report import (
    build_training_report,
    read_json,
    read_jsonl,
    sample_preview,
    summarize_training_run,
    write_training_report,
)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_fake_run(run_dir: Path, final_validation_loss: float = 2.0) -> None:
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "config.json",
        {
            "dataset": "expanded_with_petrarch",
            "context_length": 128,
            "batch_size": 32,
            "train_steps": 3,
            "learning_rate": 3e-4,
            "embedding_dim": 32,
            "num_layers": 2,
            "num_heads": 2,
            "head_dim": 16,
            "feed_forward_dim": 128,
            "vocab_size": 97,
            "train_tokens": 1000,
            "validation_tokens": 100,
        },
    )
    write_jsonl(
        run_dir / "loss_history.jsonl",
        [
            {
                "step": 1,
                "train_loss": 4.0,
                "validation_loss": 4.0,
            },
            {
                "step": 2,
                "train_loss": 2.5,
                "validation_loss": 1.9,
            },
            {
                "step": 3,
                "train_loss": 2.1,
                "validation_loss": final_validation_loss,
            },
        ],
    )
    (run_dir / "sample.txt").write_text(
        "Amor\nche move",
        encoding="utf-8",
    )
    (run_dir / "model.pt").write_bytes(b"checkpoint")


def test_read_json_loads_json_file(tmp_path):
    path = tmp_path / "config.json"
    write_json(path, {"run": "demo"})

    assert read_json(path) == {"run": "demo"}


def test_read_jsonl_loads_json_lines(tmp_path):
    path = tmp_path / "loss_history.jsonl"
    write_jsonl(
        path,
        [
            {"step": 1},
            {"step": 2},
        ],
    )

    assert read_jsonl(path) == [{"step": 1}, {"step": 2}]


def test_sample_preview_limits_characters_and_escapes_newlines(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("Amor\nche move", encoding="utf-8")

    preview = sample_preview(path, max_characters=8)

    assert preview == "Amor\\nche"


def test_sample_preview_rejects_invalid_length(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("Amor", encoding="utf-8")

    with pytest.raises(ValueError, match="max_characters"):
        sample_preview(path, max_characters=0)


def test_summarize_training_run_extracts_config_losses_and_artifacts(tmp_path):
    run_dir = tmp_path / "run_a"
    write_fake_run(run_dir)

    summary = summarize_training_run(run_dir)

    assert summary["run_name"] == "run_a"
    assert summary["dataset"] == "expanded_with_petrarch"
    assert summary["context_length"] == 128
    assert summary["final_train_loss"] == 2.1
    assert summary["final_validation_loss"] == 2.0
    assert summary["best_validation_loss"] == 1.9
    assert summary["best_validation_step"] == 2
    assert summary["checkpoint_size_mb"] > 0
    assert summary["sample_preview"] == "Amor\\nche move"


def test_summarize_training_run_rejects_empty_loss_history(tmp_path):
    run_dir = tmp_path / "run_a"
    write_fake_run(run_dir)
    (run_dir / "loss_history.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="loss history"):
        summarize_training_run(run_dir)


def test_build_training_report_sorts_by_final_validation_loss():
    rows = [
        {
            "run_name": "worse",
            "context_length": 128,
            "batch_size": 32,
            "train_steps": 3,
            "learning_rate": 3e-4,
            "embedding_dim": 32,
            "num_layers": 2,
            "num_heads": 2,
            "feed_forward_dim": 128,
            "final_train_loss": 2.0,
            "final_validation_loss": 2.5,
            "best_validation_loss": 2.4,
            "best_validation_step": 3,
            "checkpoint_size_mb": 1.0,
            "sample_preview": "worse",
        },
        {
            "run_name": "better",
            "context_length": 128,
            "batch_size": 32,
            "train_steps": 3,
            "learning_rate": 3e-4,
            "embedding_dim": 32,
            "num_layers": 2,
            "num_heads": 2,
            "feed_forward_dim": 128,
            "final_train_loss": 2.0,
            "final_validation_loss": 1.5,
            "best_validation_loss": 1.4,
            "best_validation_step": 3,
            "checkpoint_size_mb": 1.0,
            "sample_preview": "better",
        },
    ]

    report = build_training_report(rows)

    assert report.index("better") < report.index("worse")
    assert "| Run | Ctx | Batch |" in report
    assert "## Sample Previews" in report


def test_write_training_report_writes_markdown(tmp_path):
    first_run = tmp_path / "first_run"
    second_run = tmp_path / "second_run"
    output_path = tmp_path / "reports" / "training_runs.md"
    write_fake_run(first_run, final_validation_loss=2.0)
    write_fake_run(second_run, final_validation_loss=1.8)

    rows = write_training_report(
        run_dirs=[first_run, second_run],
        output_path=output_path,
    )

    report = output_path.read_text(encoding="utf-8")

    assert len(rows) == 2
    assert output_path.is_file()
    assert "first_run" in report
    assert "second_run" in report
    assert "1.8000" in report
