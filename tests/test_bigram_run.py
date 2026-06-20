import json
from pathlib import Path

import torch

from sonnet_training.bigram_run import (
    BigramTrainingConfig,
    resolve_device,
    train_bigram_run,
)


def write_tiny_manifest(repo_root: Path) -> None:
    manifest_path = repo_root / "data" / "metadata" / "poems_manifest.csv"
    manifest_path.parent.mkdir(parents=True)

    manifest_path.write_text(
        "\n".join([
            "poem_id,clean_text_path,include_in_core_pre_petrarch,"
            "include_in_expanded_with_petrarch,split_core_pre_petrarch,"
            "split_expanded_with_petrarch",
            "train,data/processed/poems/train.txt,True,True,train,train",
            "validation,data/processed/poems/validation.txt,True,True,"
            "validation,validation",
            "test,data/processed/poems/test.txt,True,True,test,test",
        ])
        + "\n",
        encoding="utf-8",
    )


def write_tiny_poems(repo_root: Path) -> None:
    poems_dir = repo_root / "data" / "processed" / "poems"
    poems_dir.mkdir(parents=True)
    poems_dir.joinpath("train.txt").write_text("ab" * 100, encoding="utf-8")
    poems_dir.joinpath("validation.txt").write_text(
        "ab" * 100,
        encoding="utf-8",
    )
    poems_dir.joinpath("test.txt").write_text("ab" * 100, encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_resolve_device_accepts_explicit_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_train_bigram_run_writes_reproducible_artifacts(tmp_path):
    write_tiny_manifest(tmp_path)
    write_tiny_poems(tmp_path)

    output_dir = tmp_path / "runs" / "bigram"
    config = BigramTrainingConfig(
        batch_size=4,
        context_length=8,
        train_steps=20,
        eval_interval=10,
        eval_batches=1,
        learning_rate=1e-1,
        seed=123,
        prompt="a",
        max_new_tokens=5,
        device="cpu",
    )

    result = train_bigram_run(
        repo_root=tmp_path,
        output_dir=output_dir,
        config=config,
    )

    assert result["config_path"].is_file()
    assert result["log_path"].is_file()
    assert result["tokenizer_path"].is_file()
    assert result["sample_path"].is_file()
    assert result["checkpoint_path"].is_file()

    saved_config = read_json(result["config_path"])
    loss_history = read_jsonl(result["log_path"])
    saved_tokenizer = read_json(result["tokenizer_path"])
    generated_sample = result["sample_path"].read_text(encoding="utf-8")
    checkpoint = torch.load(
        result["checkpoint_path"],
        map_location="cpu",
    )

    assert saved_config["dataset"] == "expanded_with_petrarch"
    assert saved_config["resolved_device"] == "cpu"
    assert saved_config["vocab_size"] > 0
    assert len(loss_history) == 3
    assert loss_history[-1]["step"] == 20
    assert loss_history[-1]["train_loss"] < loss_history[0]["train_loss"]
    assert loss_history[-1]["train_loss"] > 0.0
    assert loss_history[-1]["validation_loss"] > 0.0
    assert saved_tokenizer["type"] == "character"
    assert saved_tokenizer["char_to_id"]["a"] >= 0
    assert generated_sample.startswith("a")
    assert len(generated_sample) == len(config.prompt) + config.max_new_tokens
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert checkpoint["vocab_size"] == saved_config["vocab_size"]
