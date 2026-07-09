import json
from pathlib import Path

import pytest
import torch

from sonnet_corpus.pretraining_tokenizer import train_weighted_pretoken_bpe_tokenizer
from sonnet_training.pretraining_run import (
    PretrainingRunConfig,
    count_parameters,
    load_token_tensor,
    train_pretraining_run,
)


def write_tiny_tokenizer(path: Path) -> None:
    text = "amor antico memoria cronica virtute novella lingua storia\n"
    tokenizer = train_weighted_pretoken_bpe_tokenizer(
        training_text=text,
        base_text=text,
        vocab_size=50,
        special_tokens=["<|endoftext|>"],
    )
    tokenizer.save(path)


def tiny_pretraining_config() -> PretrainingRunConfig:
    return PretrainingRunConfig(
        train_tokens_path="data/local/pretraining/encoded/bpe_8000_train.pt",
        validation_tokens_path="data/local/pretraining/encoded/bpe_8000_validation.pt",
        tokenizer_path="data/local/pretraining/tokenizers/bpe_8000.json",
        batch_size=4,
        context_length=8,
        train_steps=3,
        eval_interval=1,
        eval_batches=1,
        learning_rate=1e-3,
        seed=123,
        prompt="amor",
        max_new_tokens=5,
        device="cpu",
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        feed_forward_dim=16,
        max_context_length=8,
    )


def write_tiny_pretraining_artifacts(repo_root: Path) -> None:
    encoded_dir = repo_root / "data" / "local" / "pretraining" / "encoded"
    tokenizer_dir = repo_root / "data" / "local" / "pretraining" / "tokenizers"
    encoded_dir.mkdir(parents=True)
    tokenizer_dir.mkdir(parents=True)
    write_tiny_tokenizer(tokenizer_dir / "bpe_8000.json")
    torch.save(
        torch.tensor(([1, 2, 3, 4, 5, 6] * 40), dtype=torch.long),
        encoded_dir / "bpe_8000_train.pt",
    )
    torch.save(
        torch.tensor(([1, 2, 3, 4, 5, 6] * 40), dtype=torch.long),
        encoded_dir / "bpe_8000_validation.pt",
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_load_token_tensor_returns_1d_long_tensor(tmp_path: Path):
    path = tmp_path / "tokens.pt"
    torch.save(torch.tensor([1, 2, 3], dtype=torch.long), path)

    tokens = load_token_tensor(path)

    assert tokens.shape == (3,)
    assert tokens.dtype == torch.long


def test_load_token_tensor_rejects_wrong_dtype(tmp_path: Path):
    path = tmp_path / "tokens.pt"
    torch.save(torch.tensor([1.0, 2.0]), path)

    with pytest.raises(ValueError, match="torch.long"):
        load_token_tensor(path)


def test_train_pretraining_run_writes_reproducible_artifacts(tmp_path: Path):
    write_tiny_pretraining_artifacts(tmp_path)
    output_dir = tmp_path / "runs" / "pretraining"
    config = tiny_pretraining_config()

    result = train_pretraining_run(
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
    checkpoint = torch.load(result["checkpoint_path"], map_location="cpu")

    assert saved_config["resolved_device"] == "cpu"
    assert saved_config["vocab_size"] == 50
    assert saved_config["train_tokens"] == 240
    assert saved_config["validation_tokens"] == 240
    assert saved_config["parameter_count"] > 0
    assert len(loss_history) == 3
    assert loss_history[-1]["step"] == 3
    assert loss_history[-1]["train_loss"] > 0.0
    assert loss_history[-1]["validation_loss"] > 0.0
    assert saved_tokenizer["type"] == "unicode_bpe"
    assert generated_sample.startswith("amor")
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert checkpoint["parameter_count"] == saved_config["parameter_count"]


def test_train_pretraining_run_rejects_context_longer_than_model_context(
    tmp_path: Path,
):
    write_tiny_pretraining_artifacts(tmp_path)
    config = PretrainingRunConfig(
        **{
            **tiny_pretraining_config().__dict__,
            "context_length": 9,
            "max_context_length": 8,
        }
    )

    with pytest.raises(ValueError, match="context_length"):
        train_pretraining_run(
            repo_root=tmp_path,
            output_dir=tmp_path / "runs" / "pretraining",
            config=config,
        )


def test_count_parameters_counts_all_model_parameters():
    model = torch.nn.Linear(3, 2)

    assert count_parameters(model) == 8
