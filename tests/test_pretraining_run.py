import json
from pathlib import Path

import pytest
import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer
from sonnet_corpus.pretraining_tokenizer import train_weighted_pretoken_bpe_tokenizer
from sonnet_model.transformer import CausalTransformerLanguageModel
from sonnet_training.pretraining_run import (
    PretrainingRunConfig,
    count_parameters,
    load_pretraining_checkpoint,
    load_token_tensor,
    merge_existing_history,
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
    assert saved_config["progress_interval"] == 100
    assert len(loss_history) == 3
    assert loss_history[-1]["step"] == 3
    assert loss_history[-1]["train_loss"] > 0.0
    assert loss_history[-1]["validation_loss"] > 0.0
    assert saved_tokenizer["type"] == "unicode_bpe"
    assert generated_sample.startswith("amor")
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert checkpoint["parameter_count"] == saved_config["parameter_count"]
    assert checkpoint["step"] == config.train_steps


def test_train_pretraining_run_supports_rope_and_records_its_configuration(
    tmp_path: Path,
):
    write_tiny_pretraining_artifacts(tmp_path)
    config = PretrainingRunConfig(
        **{
            **tiny_pretraining_config().__dict__,
            "position_encoding_type": "rope",
            "rope_theta": 10_000.0,
        }
    )

    result = train_pretraining_run(
        repo_root=tmp_path,
        output_dir=tmp_path / "runs" / "pretraining_rope",
        config=config,
    )
    saved_config = read_json(result["config_path"])
    checkpoint = torch.load(result["checkpoint_path"], map_location="cpu")

    assert saved_config["position_encoding_type"] == "rope"
    assert saved_config["rope_theta"] == 10_000.0
    assert checkpoint["config"]["position_encoding_type"] == "rope"
    assert "embedding.position_embedding.weight" not in checkpoint["model_state_dict"]


def test_train_pretraining_run_supports_swiglu_and_records_its_configuration(
    tmp_path: Path,
):
    write_tiny_pretraining_artifacts(tmp_path)
    config = PretrainingRunConfig(
        **{
            **tiny_pretraining_config().__dict__,
            "feed_forward_dim": 5,
            "feed_forward_type": "swiglu",
        }
    )

    result = train_pretraining_run(
        repo_root=tmp_path,
        output_dir=tmp_path / "runs" / "pretraining_swiglu",
        config=config,
    )
    saved_config = read_json(result["config_path"])
    checkpoint = torch.load(result["checkpoint_path"], map_location="cpu")

    assert saved_config["feed_forward_type"] == "swiglu"
    assert checkpoint["config"]["feed_forward_type"] == "swiglu"
    assert "blocks.0.feed_forward.gate_projection.weight" in checkpoint[
        "model_state_dict"
    ]


def test_train_pretraining_run_records_tied_token_embeddings(tmp_path: Path):
    write_tiny_pretraining_artifacts(tmp_path)
    config = PretrainingRunConfig(
        **{
            **tiny_pretraining_config().__dict__,
            "tie_token_embeddings": True,
        }
    )

    result = train_pretraining_run(
        repo_root=tmp_path,
        output_dir=tmp_path / "runs" / "pretraining_tied",
        config=config,
    )
    saved_config = read_json(result["config_path"])
    checkpoint = torch.load(result["checkpoint_path"], map_location="cpu")

    assert saved_config["tie_token_embeddings"] is True
    assert checkpoint["config"]["tie_token_embeddings"] is True
    assert saved_config["parameter_count"] == checkpoint["parameter_count"]


def test_train_pretraining_run_writes_interval_checkpoints(tmp_path: Path):
    write_tiny_pretraining_artifacts(tmp_path)
    output_dir = tmp_path / "runs" / "pretraining"
    config = PretrainingRunConfig(
        **{
            **tiny_pretraining_config().__dict__,
            "train_steps": 4,
            "checkpoint_interval": 2,
        }
    )

    result = train_pretraining_run(
        repo_root=tmp_path,
        output_dir=output_dir,
        config=config,
    )

    checkpoint_dir = result["checkpoint_dir"]
    step_2 = checkpoint_dir / "step_2.pt"
    step_4 = checkpoint_dir / "step_4.pt"
    final_checkpoint = result["checkpoint_path"]
    assert step_2.is_file()
    assert step_4.is_file()
    assert final_checkpoint.is_file()
    assert torch.load(step_2, map_location="cpu")["step"] == 2
    assert torch.load(step_4, map_location="cpu")["step"] == 4
    assert torch.load(final_checkpoint, map_location="cpu")["step"] == 4


def test_train_pretraining_run_resumes_from_checkpoint(tmp_path: Path):
    write_tiny_pretraining_artifacts(tmp_path)
    first_output_dir = tmp_path / "runs" / "pretraining_first"
    first_config = PretrainingRunConfig(
        **{
            **tiny_pretraining_config().__dict__,
            "train_steps": 2,
            "checkpoint_interval": 2,
        }
    )
    first_result = train_pretraining_run(
        repo_root=tmp_path,
        output_dir=first_output_dir,
        config=first_config,
    )
    resume_path = first_result["checkpoint_dir"] / "step_2.pt"
    resume_relative_path = resume_path.relative_to(tmp_path)

    second_output_dir = tmp_path / "runs" / "pretraining_resumed"
    second_config = PretrainingRunConfig(
        **{
            **tiny_pretraining_config().__dict__,
            "train_steps": 4,
            "checkpoint_interval": 2,
            "resume_from_checkpoint": str(resume_relative_path),
        }
    )
    second_result = train_pretraining_run(
        repo_root=tmp_path,
        output_dir=second_output_dir,
        config=second_config,
    )

    saved_config = read_json(second_result["config_path"])
    loss_history = read_jsonl(second_result["log_path"])
    final_checkpoint = torch.load(
        second_result["checkpoint_path"],
        map_location="cpu",
    )
    assert saved_config["start_step"] == 2
    assert saved_config["completed_steps"] == 4
    assert loss_history[0]["step"] == 3
    assert loss_history[-1]["step"] == 4
    assert final_checkpoint["step"] == 4


def test_train_pretraining_run_rejects_resume_checkpoint_at_target_step(
    tmp_path: Path,
):
    write_tiny_pretraining_artifacts(tmp_path)
    first_result = train_pretraining_run(
        repo_root=tmp_path,
        output_dir=tmp_path / "runs" / "pretraining_first",
        config=PretrainingRunConfig(
            **{
                **tiny_pretraining_config().__dict__,
                "train_steps": 2,
                "checkpoint_interval": 2,
            }
        ),
    )
    resume_path = first_result["checkpoint_dir"] / "step_2.pt"

    with pytest.raises(ValueError, match="less than train_steps"):
        train_pretraining_run(
            repo_root=tmp_path,
            output_dir=tmp_path / "runs" / "pretraining_resumed",
            config=PretrainingRunConfig(
                **{
                    **tiny_pretraining_config().__dict__,
                    "train_steps": 2,
                    "resume_from_checkpoint": str(resume_path.relative_to(tmp_path)),
                }
            ),
        )


def test_merge_existing_history_preserves_rows_through_resume_step(tmp_path: Path):
    log_path = tmp_path / "loss_history.jsonl"
    log_path.write_text(
        "\n".join([
            json.dumps({"step": 1, "train_loss": 3.0, "validation_loss": 3.1}),
            json.dumps({"step": 2, "train_loss": 2.0, "validation_loss": 2.1}),
            json.dumps({"step": 3, "train_loss": 1.0, "validation_loss": 1.1}),
        ])
        + "\n",
        encoding="utf-8",
    )

    history = merge_existing_history(
        log_path=log_path,
        start_step=2,
        new_history=[
            {"step": 3, "train_loss": 0.9, "validation_loss": 1.0},
            {"step": 4, "train_loss": 0.8, "validation_loss": 0.9},
        ],
    )

    assert [row["step"] for row in history] == [1, 2, 3, 4]
    assert history[2]["train_loss"] == 0.9


def test_load_pretraining_checkpoint_returns_completed_step(tmp_path: Path):
    write_tiny_pretraining_artifacts(tmp_path)
    result = train_pretraining_run(
        repo_root=tmp_path,
        output_dir=tmp_path / "runs" / "pretraining",
        config=tiny_pretraining_config(),
    )
    config = tiny_pretraining_config()
    tokenizer_path = tmp_path / config.tokenizer_path

    tokenizer = BytePairEncodingTokenizer.load(tokenizer_path)
    model = CausalTransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        feed_forward_dim=config.feed_forward_dim,
        max_context_length=config.max_context_length,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    step = load_pretraining_checkpoint(
        checkpoint_path=result["checkpoint_path"],
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert step == config.train_steps


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
