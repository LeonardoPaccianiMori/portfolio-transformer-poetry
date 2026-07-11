import csv
import json
from pathlib import Path

from sonnet_evaluation.control_comparison_report import (
    build_control_comparison_markdown,
    control_arms_are_comparable,
    summarize_control_arm,
    write_control_comparison_report,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_manifest(repo_root: Path) -> Path:
    manifest_path = repo_root / "data" / "metadata" / "poems_manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    row = {
        "poem_id": "train_poem",
        "title_or_first_line": "Amor che move",
        "author": "Test Author",
        "clean_text_path": "data/processed/poems/train_poem.txt",
        "include_in_core_pre_petrarch": "True",
        "include_in_expanded_with_petrarch": "True",
        "split_core_pre_petrarch": "train",
        "split_expanded_with_petrarch": "train",
    }
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    poem_path = repo_root / row["clean_text_path"]
    poem_path.parent.mkdir(parents=True)
    poem_path.write_text("Amor che move il sole\n", encoding="utf-8")
    return manifest_path


def write_arm(repo_root: Path, name: str, initialization: str, best_loss: float) -> tuple[Path, Path]:
    run_dir = repo_root / "runs" / name
    generation_dir = repo_root / "outputs" / name
    run_dir.mkdir(parents=True)
    generation_dir.mkdir(parents=True)
    config = {
        "initialization": initialization,
        "dataset": "expanded_with_petrarch",
        "vocab_size": 8,
        "train_tokens": 100,
        "validation_tokens": 20,
        "model_architecture": {"vocab_size": 8},
        "batch_size": 2,
        "context_length": 8,
        "train_steps": 10,
        "learning_rate": 3e-5,
        "seed": 1337,
        "optimizer_state_restored": False,
    }
    write_json(run_dir / "config.json", config)
    (run_dir / "loss_history.jsonl").write_text(
        "\n".join([
            json.dumps({"step": 1, "train_loss": 4.0, "validation_loss": 4.1}),
            json.dumps({"step": 5, "train_loss": 2.0, "validation_loss": best_loss}),
            json.dumps({"step": 10, "train_loss": 1.0, "validation_loss": best_loss + 0.5}),
        ])
        + "\n",
        encoding="utf-8",
    )
    generated_path = generation_dir / "amor.txt"
    generated_path.write_text("Amor che move\n", encoding="utf-8")
    write_json(
        generation_dir / "metadata.json",
        {
            "stop_text": "<|endoftext|>",
            "generated_files": [
                {
                    "prompt_id": "amor",
                    "prompt_text": "Amor",
                    "path": str(generated_path),
                    "seed": 1337,
                },
            ],
        },
    )
    return run_dir, generation_dir


def test_control_comparison_summarizes_matching_arms_and_writes_report(tmp_path):
    manifest_path = write_manifest(tmp_path)
    pretrained_run, pretrained_generation = write_arm(
        tmp_path,
        "pretrained",
        "pretrained",
        2.0,
    )
    random_run, random_generation = write_arm(
        tmp_path,
        "random",
        "random",
        3.0,
    )

    pretrained = summarize_control_arm(
        run_dir=pretrained_run,
        generation_dir=pretrained_generation,
        manifest_path=manifest_path,
        repo_root=tmp_path,
    )
    random = summarize_control_arm(
        run_dir=random_run,
        generation_dir=random_generation,
        manifest_path=manifest_path,
        repo_root=tmp_path,
    )
    output_path = tmp_path / "reports" / "comparison.md"
    summaries = write_control_comparison_report(
        pretrained_run_dir=pretrained_run,
        pretrained_generation_dir=pretrained_generation,
        random_run_dir=random_run,
        random_generation_dir=random_generation,
        manifest_path=manifest_path,
        repo_root=tmp_path,
        output_path=output_path,
    )

    report = build_control_comparison_markdown(pretrained, random)
    assert control_arms_are_comparable(pretrained, random)
    assert pretrained["best_row"]["step"] == 5
    assert "lower best validation loss by 1.0000" in report
    assert output_path.is_file()
    assert summaries["random"]["best_row"]["validation_loss"] == 3.0


def test_control_comparison_allows_declared_schedule_difference(tmp_path):
    manifest_path = write_manifest(tmp_path)
    left_run, left_generation = write_arm(tmp_path, "constant", "pretrained", 2.0)
    right_run, right_generation = write_arm(tmp_path, "cosine", "pretrained", 2.1)
    for run_dir, schedule, warmup, minimum in (
        (left_run, "constant", 0, 0.0),
        (right_run, "warmup_cosine", 2, 1e-6),
    ):
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["learning_rate_schedule"] = schedule
        config["warmup_steps"] = warmup
        config["min_learning_rate"] = minimum
        config_path.write_text(json.dumps(config), encoding="utf-8")

    left = summarize_control_arm(
        run_dir=left_run,
        generation_dir=left_generation,
        manifest_path=manifest_path,
        repo_root=tmp_path,
    )
    right = summarize_control_arm(
        run_dir=right_run,
        generation_dir=right_generation,
        manifest_path=manifest_path,
        repo_root=tmp_path,
    )
    report = build_control_comparison_markdown(
        left,
        right,
        experiment_name="Learning-Rate Schedule",
        left_label="constant",
        right_label="warmup cosine",
        intended_difference="the learning-rate schedule",
        allowed_difference_fields={
            "learning_rate_schedule",
            "warmup_steps",
            "min_learning_rate",
        },
    )

    assert control_arms_are_comparable(
        left,
        right,
        allowed_difference_fields={
            "learning_rate_schedule",
            "warmup_steps",
            "min_learning_rate",
        },
    )
    assert "# Controlled Experiment Comparison: Learning-Rate Schedule" in report
