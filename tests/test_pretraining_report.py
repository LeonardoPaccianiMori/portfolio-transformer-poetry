import json
from pathlib import Path

import pytest

from sonnet_evaluation.pretraining_report import (
    build_pretraining_markdown_report,
    checkpoint_count,
    sample_excerpt,
    summarize_pretraining_run,
    write_pretraining_markdown_report,
)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_fake_pretraining_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "config.json",
        {
            "resolved_device": "cuda:0",
            "vocab_size": 8000,
            "train_tokens": 100_000,
            "validation_tokens": 10_000,
            "context_length": 512,
            "batch_size": 2,
            "train_steps": 20_000,
            "completed_steps": 20_000,
            "eval_interval": 1_000,
            "eval_batches": 5,
            "checkpoint_interval": 5_000,
            "learning_rate": 3e-4,
            "embedding_dim": 512,
            "num_layers": 8,
            "num_heads": 8,
            "head_dim": 64,
            "feed_forward_dim": 2048,
            "parameter_count": 33_669_952,
        },
    )
    write_jsonl(
        run_dir / "loss_history.jsonl",
        [
            {"step": 1, "train_loss": 8.0, "validation_loss": 7.5},
            {"step": 10_000, "train_loss": 2.2, "validation_loss": 2.0},
            {"step": 20_000, "train_loss": 1.8, "validation_loss": 2.1},
        ],
    )
    (run_dir / "sample.txt").write_text("Nel monte\ndi San Benedetto", encoding="utf-8")
    (run_dir / "model.pt").write_bytes(b"final checkpoint")
    (run_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "step_5000.pt").write_bytes(b"checkpoint")
    (checkpoint_dir / "step_10000.pt").write_bytes(b"checkpoint")


def test_sample_excerpt_limits_text_and_marks_truncation(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("Amor che move", encoding="utf-8")

    assert sample_excerpt(path, max_characters=6) == "Amor c\n[excerpt truncated]"


def test_sample_excerpt_rejects_invalid_length(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("Amor", encoding="utf-8")

    with pytest.raises(ValueError, match="max_characters"):
        sample_excerpt(path, max_characters=0)


def test_checkpoint_count_returns_zero_when_no_checkpoint_directory_exists(tmp_path):
    assert checkpoint_count(tmp_path) == 0


def test_summarize_pretraining_run_extracts_losses_configuration_and_artifacts(tmp_path):
    run_dir = tmp_path / "pretraining_run"
    write_fake_pretraining_run(run_dir)

    summary = summarize_pretraining_run(run_dir)

    assert summary["run_name"] == "pretraining_run"
    assert summary["resolved_device"] == "cuda:0"
    assert summary["parameter_count"] == 33_669_952
    assert summary["normalization_type"] == "layer_norm"
    assert summary["normalization_eps"] == 1e-5
    assert summary["position_encoding_type"] == "learned_absolute"
    assert summary["rope_theta"] == 10_000.0
    assert summary["feed_forward_type"] == "relu"
    assert summary["first_validation_loss"] == 7.5
    assert summary["final_validation_loss"] == 2.1
    assert summary["best_validation_loss"] == 2.0
    assert summary["best_validation_step"] == 10_000
    assert summary["best_validation_train_loss"] == 2.2
    assert summary["validation_mode"] == "random_batches"
    assert summary["validation_window_count"] == 19
    assert summary["learning_rate_schedule"] == "constant"
    assert summary["interval_checkpoint_count"] == 2
    assert summary["loss_records"] == 3
    assert summary["sample_excerpt"] == "Nel monte\ndi San Benedetto"


def test_summarize_pretraining_run_rejects_missing_required_artifact(tmp_path):
    run_dir = tmp_path / "pretraining_run"
    write_fake_pretraining_run(run_dir)
    (run_dir / "model.pt").unlink()

    with pytest.raises(FileNotFoundError, match="required run artifact"):
        summarize_pretraining_run(run_dir)


def test_summarize_pretraining_run_rejects_empty_loss_history(tmp_path):
    run_dir = tmp_path / "pretraining_run"
    write_fake_pretraining_run(run_dir)
    (run_dir / "loss_history.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="loss history"):
        summarize_pretraining_run(run_dir)


def test_markdown_report_includes_summary_sample_and_full_history(tmp_path):
    run_dir = tmp_path / "pretraining_run"
    write_fake_pretraining_run(run_dir)

    report = build_pretraining_markdown_report(summarize_pretraining_run(run_dir))

    assert "# Pretraining Run: pretraining_run" in report
    assert "| Best validation evaluation | 10,000 | 2.2000 | 2.0000 |" in report
    assert "Nel monte\ndi San Benedetto" in report
    assert "| Normalization | layer_norm |" in report
    assert "| Position encoding | learned_absolute |" in report
    assert "| Feed-forward type | relu |" in report
    assert "| Evaluation | every 1,000 steps; 5 random batches |" in report
    assert "| 20,000 | 1.8000 | 2.1000 |" in report


def test_pretraining_report_describes_deterministic_sequential_validation(tmp_path):
    run_dir = tmp_path / "pretraining_run"
    write_fake_pretraining_run(run_dir)
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update({
        "validation_mode": "sequential_windows",
        "validation_window_count": 19,
        "learning_rate_schedule": "warmup_cosine",
        "warmup_steps": 500,
        "min_learning_rate": 3e-5,
    })
    write_json(config_path, config)

    report = build_pretraining_markdown_report(summarize_pretraining_run(run_dir))

    assert "| Learning-rate schedule | warmup_cosine |" in report
    assert "| Warmup steps | 500 |" in report
    assert "| Evaluation | every 1,000 steps; all 19 sequential windows |" in report
    assert "deterministic selection within this run" in report
    assert "should use `best_validation.pt`, not `model.pt`" in report


def test_summarize_pretraining_run_reads_explicit_rms_norm_configuration(tmp_path):
    run_dir = tmp_path / "pretraining_run"
    write_fake_pretraining_run(run_dir)
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["normalization_type"] = "rms_norm"
    config["normalization_eps"] = 1e-6
    write_json(config_path, config)

    summary = summarize_pretraining_run(run_dir)

    assert summary["normalization_type"] == "rms_norm"
    assert summary["normalization_eps"] == 1e-6


def test_summarize_pretraining_run_reads_explicit_rope_configuration(tmp_path):
    run_dir = tmp_path / "pretraining_run"
    write_fake_pretraining_run(run_dir)
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["position_encoding_type"] = "rope"
    config["rope_theta"] = 20_000.0
    write_json(config_path, config)

    summary = summarize_pretraining_run(run_dir)

    assert summary["position_encoding_type"] == "rope"
    assert summary["rope_theta"] == 20_000.0


def test_summarize_pretraining_run_reads_explicit_swiglu_configuration(tmp_path):
    run_dir = tmp_path / "pretraining_run"
    write_fake_pretraining_run(run_dir)
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["feed_forward_type"] = "swiglu"
    write_json(config_path, config)

    summary = summarize_pretraining_run(run_dir)

    assert summary["feed_forward_type"] == "swiglu"


def test_write_pretraining_markdown_report_writes_public_file(tmp_path):
    run_dir = tmp_path / "pretraining_run"
    output_path = tmp_path / "reports" / "pretraining.md"
    write_fake_pretraining_run(run_dir)

    summary = write_pretraining_markdown_report(run_dir, output_path)

    assert output_path.is_file()
    assert summary["run_name"] == "pretraining_run"
    assert "## Configuration" in output_path.read_text(encoding="utf-8")
