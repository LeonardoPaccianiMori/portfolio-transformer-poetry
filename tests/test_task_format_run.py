import json
from pathlib import Path

import pytest
import torch

from sonnet_corpus.pretraining_tokenizer import train_weighted_pretoken_bpe_tokenizer
from sonnet_model.transformer import CausalTransformerLanguageModel
from sonnet_training.pretraining_run import count_parameters
from sonnet_training.task_format_run import (
    TASK_FORMAT_VERSION,
    TaskFormatRunConfig,
    train_task_format_run,
)


def make_sonnet(prefix: str) -> str:
    return "\n".join(f"{prefix} {number}" for number in range(1, 15))


def write_manifest(path: Path) -> None:
    path.write_text(
        "poem_id,clean_text_path,include_in_expanded_with_petrarch,split_expanded_with_petrarch\n"
        "train_a,data/processed/train_a.txt,True,train\n"
        "train_b,data/processed/train_b.txt,True,train\n"
        "validation,data/processed/validation.txt,True,validation\n"
        "test,data/processed/test.txt,True,test\n",
        encoding="utf-8",
    )


def write_tiny_task_artifacts(repo_root: Path) -> tuple[Path, Path]:
    processed_dir = repo_root / "data" / "processed"
    processed_dir.mkdir(parents=True)
    texts = {
        "train_a": make_sonnet("train a"),
        "train_b": make_sonnet("train b"),
        "validation": make_sonnet("validation"),
        "test": make_sonnet("test"),
    }
    for name, text in texts.items():
        (processed_dir / f"{name}.txt").write_text(text, encoding="utf-8")
    manifest_path = repo_root / "data" / "metadata" / "manifest.csv"
    manifest_path.parent.mkdir()
    write_manifest(manifest_path)

    tokenizer = train_weighted_pretoken_bpe_tokenizer(
        training_text="\n".join(texts.values()),
        base_text="\n".join([*texts.values(), "<|endoftext|>"]),
        vocab_size=64,
        special_tokens=["<|endoftext|>"],
    )
    tokenizer_path = repo_root / "runs" / "parent" / "tokenizer.json"
    tokenizer_path.parent.mkdir(parents=True)
    tokenizer.save(tokenizer_path)

    model = CausalTransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        feed_forward_dim=16,
        max_context_length=128,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    architecture = {
        "vocab_size": tokenizer.vocab_size,
        "embedding_dim": 8,
        "num_layers": 1,
        "num_heads": 2,
        "head_dim": 4,
        "feed_forward_dim": 16,
        "max_context_length": 128,
        "normalization_type": "layer_norm",
        "normalization_eps": 1e-5,
        "position_encoding_type": "learned_absolute",
        "rope_theta": 10_000.0,
        "feed_forward_type": "relu",
        "tie_token_embeddings": False,
    }
    checkpoint_path = repo_root / "runs" / "parent" / "best_validation.pt"
    torch.save(
        {
            "step": 7,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {"initialization": "pretrained"},
            "vocab_size": tokenizer.vocab_size,
            "parameter_count": count_parameters(model),
            "model_architecture": architecture,
            "manifest_sha256": "parent-manifest",
            "parent_checkpoint_step": 3,
        },
        checkpoint_path,
    )
    return manifest_path, checkpoint_path


def tiny_task_config() -> TaskFormatRunConfig:
    return TaskFormatRunConfig(
        manifest_path="data/metadata/manifest.csv",
        parent_checkpoint_path="runs/parent/best_validation.pt",
        parent_tokenizer_path="runs/parent/tokenizer.json",
        batch_size=1,
        gradient_accumulation_steps=1,
        context_length=128,
        train_steps=3,
        eval_interval=1,
        early_stopping_patience=0,
        checkpoint_interval=2,
        progress_interval=1,
        learning_rate=1e-3,
        max_gradient_norm=1.0,
        device="cpu",
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_task_format_run_writes_resumable_artifacts_from_nested_parent_architecture(
    tmp_path: Path,
):
    write_tiny_task_artifacts(tmp_path)
    output_dir = tmp_path / "runs" / "task"

    result = train_task_format_run(
        repo_root=tmp_path,
        output_dir=output_dir,
        config=tiny_task_config(),
    )

    saved_config = read_json(result["config_path"])
    history = read_jsonl(result["log_path"])
    saved_tokenizer = read_json(result["tokenizer_path"])
    best_checkpoint = torch.load(result["best_checkpoint_path"], map_location="cpu")
    resume_checkpoint = torch.load(result["resume_checkpoint_path"], map_location="cpu")
    final_checkpoint = torch.load(result["checkpoint_path"], map_location="cpu")
    parent_checkpoint = torch.load(
        tmp_path / "runs" / "parent" / "best_validation.pt",
        map_location="cpu",
    )
    expected_task_vocab_size = int(parent_checkpoint["vocab_size"]) + 2

    assert result["completed_steps"] == 3
    assert result["stop_reason"] == "max_train_steps_reached"
    assert [row["step"] for row in history] == [1, 2, 3]
    assert saved_config["task_format_version"] == TASK_FORMAT_VERSION
    assert saved_config["vocab_size"] == expected_task_vocab_size
    assert saved_config["pad_token_id"] == 0
    assert saved_config["train_examples"] == 2
    assert saved_config["validation_examples"] == 1
    assert saved_config["train_supervised_targets"] > 0
    assert saved_config["parent_provenance"]["step"] == 7
    assert saved_tokenizer["special_tokens"][-2:] == [
        "<|sonnet_opening|>",
        "<|sonnet_continuation|>",
    ]
    assert best_checkpoint["optimizer_state_dict"] is None
    assert resume_checkpoint["optimizer_state_dict"] is not None
    assert resume_checkpoint["step"] == 3
    assert final_checkpoint["optimizer_state_dict"] is not None
    assert final_checkpoint["model_architecture"]["vocab_size"] == expected_task_vocab_size


def test_task_format_run_resumes_from_latest_checkpoint_and_preserves_history(
    tmp_path: Path,
):
    write_tiny_task_artifacts(tmp_path)
    output_dir = tmp_path / "runs" / "task"
    first_config = TaskFormatRunConfig(
        **{
            **tiny_task_config().__dict__,
            "train_steps": 2,
            "checkpoint_interval": 1,
        }
    )
    train_task_format_run(
        repo_root=tmp_path,
        output_dir=output_dir,
        config=first_config,
    )
    resumed_config = TaskFormatRunConfig(
        **{
            **tiny_task_config().__dict__,
            "train_steps": 3,
            "checkpoint_interval": 1,
            "resume_from_checkpoint": "runs/task/resume.pt",
        }
    )

    result = train_task_format_run(
        repo_root=tmp_path,
        output_dir=output_dir,
        config=resumed_config,
    )

    history = read_jsonl(result["log_path"])
    saved_config = read_json(result["config_path"])
    assert [row["step"] for row in history] == [1, 2, 3]
    assert saved_config["start_step"] == 2
    assert saved_config["completed_steps"] == 3


def test_task_format_run_rejects_invalid_configuration(tmp_path: Path):
    write_tiny_task_artifacts(tmp_path)
    config = TaskFormatRunConfig(
        **{
            **tiny_task_config().__dict__,
            "batch_size": 0,
        }
    )

    with pytest.raises(ValueError, match="batch_size"):
        train_task_format_run(
            repo_root=tmp_path,
            output_dir=tmp_path / "runs" / "invalid",
            config=config,
        )


def test_task_format_resume_rejects_manifest_mismatch(tmp_path: Path):
    write_tiny_task_artifacts(tmp_path)
    output_dir = tmp_path / "runs" / "task"
    first_config = TaskFormatRunConfig(
        **{
            **tiny_task_config().__dict__,
            "train_steps": 2,
            "checkpoint_interval": 1,
        }
    )
    train_task_format_run(
        repo_root=tmp_path,
        output_dir=output_dir,
        config=first_config,
    )
    manifest_path = tmp_path / "data" / "metadata" / "manifest.csv"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    resumed_config = TaskFormatRunConfig(
        **{
            **tiny_task_config().__dict__,
            "train_steps": 3,
            "checkpoint_interval": 1,
            "resume_from_checkpoint": "runs/task/resume.pt",
        }
    )

    with pytest.raises(ValueError, match="manifest"):
        train_task_format_run(
            repo_root=tmp_path,
            output_dir=output_dir,
            config=resumed_config,
        )
