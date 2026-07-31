import json
from array import array
from pathlib import Path

import pytest
import torch

import sonnet_training.pretraining_run as pretraining_run
from sonnet_corpus.bpe import BytePairEncodingTokenizer
from sonnet_corpus.pretraining_tokenizer import train_weighted_pretoken_bpe_tokenizer
from sonnet_model.transformer import CausalTransformerLanguageModel
from sonnet_training.pretraining_run import (
    PretrainingRunConfig,
    checkpoint_path_for_interval,
    count_parameters,
    learning_rate_for_step,
    load_pretraining_token_splits,
    load_pretraining_checkpoint,
    load_pretraining_checkpoint_with_best_validation,
    load_token_tensor,
    merge_existing_history,
    optimizer_step_token_count,
    planned_train_token_exposures,
    train_pretraining_run,
    validate_pretraining_dataset_artifacts,
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
        dataset_report_path="reports/tiny_pretraining_dataset_report.json",
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
    report_dir = repo_root / "reports"
    report_dir.mkdir()
    report_dir.joinpath("tiny_pretraining_dataset_report.json").write_text(
        json.dumps(
            {
                "train_path": "data/local/pretraining/encoded/bpe_8000_train.pt",
                "validation_path": "data/local/pretraining/encoded/bpe_8000_validation.pt",
                "tokenizer_path": "data/local/pretraining/tokenizers/bpe_8000.json",
                "train_tokens": 240,
                "validation_tokens": 240,
                "total_tokens": 480,
                "train_dtype": "torch.int64",
                "validation_dtype": "torch.int64",
                "vocab_size": 50,
                "document_separator": "<|endoftext|>",
                "document_separator_token_id": 0,
                "source_count": 2,
                "split_policy": "final_token_fraction_per_source",
                "sources": [{"source_id": "a"}, {"source_id": "b"}],
            }
        ),
        encoding="utf-8",
    )


def write_tiny_memory_mapped_pretraining_artifacts(repo_root: Path) -> None:
    encoded_dir = repo_root / "data/local/pretraining/paisa_historical_rescue_v1/encoded"
    tokenizer_dir = repo_root / "data/local/pretraining/paisa_historical_rescue_v1"
    encoded_dir.mkdir(parents=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    write_tiny_tokenizer(tokenizer_dir / "tokenizer.json")
    token_values = [1, 2, 3, 4, 5, 6] * 40
    for split_id in ("paisa_train", "paisa_validation"):
        with (encoded_dir / f"{split_id}.uint16.bin").open("wb") as handle:
            array("H", token_values).tofile(handle)
    report_dir = repo_root / "reports"
    report_dir.mkdir(exist_ok=True)
    report_dir.joinpath("paisa_historical_rescue_v1_encoded_report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "split_policy": "fixed splits only",
                "tokenizer": {
                    "path": "data/local/pretraining/paisa_historical_rescue_v1/tokenizer.json",
                    "vocab_size": 50,
                    "document_separator": "<|endoftext|>",
                    "document_separator_token_id": 0,
                },
                "splits": [
                    {
                        "split_id": "paisa_train",
                        "status": "complete",
                        "output_path": "data/local/pretraining/paisa_historical_rescue_v1/encoded/paisa_train.uint16.bin",
                        "dtype": "torch.uint16",
                        "documents": 2,
                        "tokens": len(token_values),
                    },
                    {
                        "split_id": "paisa_validation",
                        "status": "complete",
                        "output_path": "data/local/pretraining/paisa_historical_rescue_v1/encoded/paisa_validation.uint16.bin",
                        "dtype": "torch.uint16",
                        "documents": 2,
                        "tokens": len(token_values),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def tiny_memory_mapped_pretraining_config() -> PretrainingRunConfig:
    return PretrainingRunConfig(
        **{
            **tiny_pretraining_config().__dict__,
            "dataset_version": "paisa_historical_rescue_v1",
            "train_tokens_path": (
                "data/local/pretraining/paisa_historical_rescue_v1/encoded/"
                "paisa_train.uint16.bin"
            ),
            "validation_tokens_path": (
                "data/local/pretraining/paisa_historical_rescue_v1/encoded/"
                "paisa_validation.uint16.bin"
            ),
            "tokenizer_path": (
                "data/local/pretraining/paisa_historical_rescue_v1/tokenizer.json"
            ),
            "dataset_report_path": "reports/paisa_historical_rescue_v1_encoded_report.json",
            "train_split_id": "paisa_train",
            "validation_split_id": "paisa_validation",
        }
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


def test_pretraining_run_rejects_the_same_train_and_validation_split_id():
    config = PretrainingRunConfig(
        **{
            **tiny_pretraining_config().__dict__,
            "validation_split_id": "train",
        }
    )

    with pytest.raises(ValueError, match="must differ"):
        pretraining_run._validate_config(config)


def test_memory_mapped_pretraining_artifacts_load_and_train_on_cpu(tmp_path: Path):
    write_tiny_memory_mapped_pretraining_artifacts(tmp_path)
    config = tiny_memory_mapped_pretraining_config()

    train_tokens, validation_tokens = load_pretraining_token_splits(
        repo_root=tmp_path,
        config=config,
    )
    tokenizer = BytePairEncodingTokenizer.load(tmp_path / config.tokenizer_path)
    provenance = validate_pretraining_dataset_artifacts(
        repo_root=tmp_path,
        config=config,
        tokenizer=tokenizer,
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
    )
    result = train_pretraining_run(
        repo_root=tmp_path,
        output_dir=tmp_path / "runs/memory_mapped_pretraining",
        config=config,
    )

    assert train_tokens.dtype == torch.uint16
    assert validation_tokens.dtype == torch.uint16
    assert provenance["stream_count"] == 2
    assert provenance["document_count"] == 4
    assert result["history"][-1]["step"] == config.train_steps


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
    assert result["best_checkpoint_path"].is_file()

    saved_config = read_json(result["config_path"])
    loss_history = read_jsonl(result["log_path"])
    saved_tokenizer = read_json(result["tokenizer_path"])
    generated_sample = result["sample_path"].read_text(encoding="utf-8")
    checkpoint = torch.load(result["checkpoint_path"], map_location="cpu")
    best_checkpoint = torch.load(result["best_checkpoint_path"], map_location="cpu")

    assert saved_config["resolved_device"] == "cpu"
    assert saved_config["vocab_size"] == 50
    assert saved_config["train_tokens"] == 240
    assert saved_config["validation_tokens"] == 240
    assert saved_config["microbatch_tokens"] == 32
    assert saved_config["tokens_per_optimizer_step"] == 32
    assert saved_config["planned_train_token_exposures"] == 96
    assert saved_config["dataset_provenance"] == {
        "dataset_version": "pretraining_historical_italian_v2",
        "dataset_report_path": "reports/tiny_pretraining_dataset_report.json",
        "source_count": 2,
        "split_policy": "final_token_fraction_per_source",
        "vocab_size": 50,
        "train_tokens": 240,
        "validation_tokens": 240,
    }
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
    best_history_row = min(loss_history, key=lambda row: row["validation_loss"])
    assert saved_config["best_validation_step"] == best_history_row["step"]
    assert saved_config["best_validation_loss"] == best_history_row["validation_loss"]
    assert best_checkpoint["step"] == best_history_row["step"]
    assert best_checkpoint["best_validation_row"] == best_history_row


def test_pretraining_dataset_preflight_rejects_token_ids_outside_vocabulary(
    tmp_path: Path,
):
    write_tiny_pretraining_artifacts(tmp_path)
    config = tiny_pretraining_config()
    train_tokens = load_token_tensor(tmp_path / config.train_tokens_path)
    validation_tokens = load_token_tensor(tmp_path / config.validation_tokens_path)
    tokenizer = BytePairEncodingTokenizer.load(tmp_path / config.tokenizer_path)
    train_tokens[-1] = tokenizer.vocab_size

    with pytest.raises(ValueError, match="train token IDs"):
        validate_pretraining_dataset_artifacts(
            repo_root=tmp_path,
            config=config,
            tokenizer=tokenizer,
            train_tokens=train_tokens,
            validation_tokens=validation_tokens,
        )


def test_pretraining_dataset_preflight_rejects_mismatched_report_count(
    tmp_path: Path,
):
    write_tiny_pretraining_artifacts(tmp_path)
    config = tiny_pretraining_config()
    report_path = tmp_path / config.dataset_report_path
    report = read_json(report_path)
    report["validation_tokens"] = 239
    report_path.write_text(json.dumps(report), encoding="utf-8")
    tokenizer = BytePairEncodingTokenizer.load(tmp_path / config.tokenizer_path)

    with pytest.raises(ValueError, match="validation_tokens"):
        validate_pretraining_dataset_artifacts(
            repo_root=tmp_path,
            config=config,
            tokenizer=tokenizer,
            train_tokens=load_token_tensor(tmp_path / config.train_tokens_path),
            validation_tokens=load_token_tensor(tmp_path / config.validation_tokens_path),
        )


def test_pretraining_warmup_cosine_schedule_is_recorded_in_history(tmp_path: Path):
    write_tiny_pretraining_artifacts(tmp_path)
    config = PretrainingRunConfig(
        **{
            **tiny_pretraining_config().__dict__,
            "learning_rate_schedule": "warmup_cosine",
            "warmup_steps": 2,
            "min_learning_rate": 1e-4,
        }
    )

    result = train_pretraining_run(
        repo_root=tmp_path,
        output_dir=tmp_path / "runs" / "scheduled_pretraining",
        config=config,
    )
    history = read_jsonl(result["log_path"])

    assert learning_rate_for_step(config, 1) == 5e-4
    assert learning_rate_for_step(config, 2) == 1e-3
    assert learning_rate_for_step(config, 3) == 1e-4
    assert [row["learning_rate"] for row in history] == [5e-4, 1e-3, 1e-4]


def test_pretraining_run_uses_full_sequential_validation_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    write_tiny_pretraining_artifacts(tmp_path)
    calls = []

    def fake_sequential_loss(**kwargs) -> float:
        calls.append(kwargs)
        return 2.5

    monkeypatch.setattr(
        pretraining_run,
        "estimate_next_token_loss_on_sequential_windows",
        fake_sequential_loss,
    )
    config = PretrainingRunConfig(
        **{
            **tiny_pretraining_config().__dict__,
            "validation_mode": "sequential_windows",
        }
    )

    result = train_pretraining_run(
        repo_root=tmp_path,
        output_dir=tmp_path / "runs" / "sequential_pretraining",
        config=config,
    )
    saved_config = read_json(result["config_path"])
    history = read_jsonl(result["log_path"])

    assert len(calls) == config.train_steps
    assert all(call["batch_size"] == config.batch_size for call in calls)
    assert saved_config["validation_mode"] == "sequential_windows"
    assert saved_config["validation_window_count"] == (
        saved_config["validation_tokens"] - 1
    ) // config.context_length
    assert all(row["validation_loss"] == 2.5 for row in history)


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


def test_train_pretraining_run_keeps_one_resumable_checkpoint_when_requested(
    tmp_path: Path,
):
    write_tiny_pretraining_artifacts(tmp_path)
    output_dir = tmp_path / "runs" / "pretraining"
    config = PretrainingRunConfig(
        **{
            **tiny_pretraining_config().__dict__,
            "train_steps": 4,
            "checkpoint_interval": 2,
            "checkpoint_retention": "latest_only",
            "gradient_accumulation_steps": 2,
        }
    )

    result = train_pretraining_run(
        repo_root=tmp_path,
        output_dir=output_dir,
        config=config,
    )

    resume_checkpoint = result["resume_checkpoint_path"]
    assert resume_checkpoint.is_file()
    assert torch.load(resume_checkpoint, map_location="cpu")["step"] == 4
    assert not (output_dir / "checkpoints").exists()
    assert optimizer_step_token_count(config) == 64
    assert planned_train_token_exposures(config) == 256


def test_checkpoint_path_for_interval_rejects_unknown_retention(tmp_path: Path):
    with pytest.raises(ValueError, match="checkpoint_retention"):
        checkpoint_path_for_interval(
            output_dir=tmp_path,
            step=1,
            retention="unknown",  # type: ignore[arg-type]
        )


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


def test_pretraining_resume_preserves_checkpoint_best_when_history_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    write_tiny_pretraining_artifacts(tmp_path)
    output_dir = tmp_path / "runs" / "interrupted_pretraining"
    first_config = PretrainingRunConfig(
        **{
            **tiny_pretraining_config().__dict__,
            "train_steps": 2,
            "checkpoint_interval": 2,
            "checkpoint_retention": "latest_only",
        }
    )
    train_pretraining_run(
        repo_root=tmp_path,
        output_dir=output_dir,
        config=first_config,
    )
    best_before = torch.load(output_dir / "best_validation.pt", map_location="cpu")
    (output_dir / "loss_history.jsonl").unlink()

    monkeypatch.setattr(
        pretraining_run,
        "estimate_pretraining_validation_loss",
        lambda **kwargs: float(best_before["best_validation_row"]["validation_loss"]) + 1.0,
    )
    resumed_config = PretrainingRunConfig(
        **{
            **tiny_pretraining_config().__dict__,
            "train_steps": 4,
            "checkpoint_interval": 2,
            "checkpoint_retention": "latest_only",
            "resume_from_checkpoint": str(
                (output_dir / "resume.pt").relative_to(tmp_path)
            ),
        }
    )
    result = train_pretraining_run(
        repo_root=tmp_path,
        output_dir=output_dir,
        config=resumed_config,
    )
    best_after = torch.load(output_dir / "best_validation.pt", map_location="cpu")
    saved_config = read_json(result["config_path"])

    assert best_after["step"] == best_before["step"]
    assert saved_config["best_validation_step"] == best_before["step"]


def test_load_pretraining_checkpoint_returns_saved_best_validation_row(tmp_path: Path):
    write_tiny_pretraining_artifacts(tmp_path)
    result = train_pretraining_run(
        repo_root=tmp_path,
        output_dir=tmp_path / "runs" / "pretraining",
        config=tiny_pretraining_config(),
    )
    config = tiny_pretraining_config()
    tokenizer = BytePairEncodingTokenizer.load(tmp_path / config.tokenizer_path)
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

    step, best_row = load_pretraining_checkpoint_with_best_validation(
        checkpoint_path=result["checkpoint_path"],
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert step == config.train_steps
    assert best_row is not None
    assert best_row["step"] <= config.train_steps


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
