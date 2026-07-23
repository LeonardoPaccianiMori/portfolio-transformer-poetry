import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer, train_bpe_tokenizer
from sonnet_corpus.dataset_text import PRETRAINING_DOCUMENT_SEPARATOR
from sonnet_model.transformer import CausalTransformerLanguageModel
from sonnet_training.sonnet_control_run import (
    SonnetControlRunConfig,
    initialize_control_model,
    learning_rate_for_step,
    load_model_architecture,
    set_optimizer_learning_rate,
    target_model_architecture,
    train_sonnet_control_run,
)


def write_manifest(path: Path) -> None:
    rows = []
    for split, poem_id in (
        ("train", "train_one"),
        ("train", "train_two"),
        ("validation", "validation"),
        ("test", "test"),
    ):
        rows.append({
            "poem_id": poem_id,
            "clean_text_path": f"data/processed/poems/{poem_id}.txt",
            "include_in_core_pre_petrarch": "True",
            "include_in_expanded_with_petrarch": "True",
            "split_core_pre_petrarch": split,
            "split_expanded_with_petrarch": split,
        })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_control_inputs(repo_root: Path) -> tuple[Path, Path, dict[str, int]]:
    manifest_path = repo_root / "data" / "metadata" / "poems_manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    write_manifest(manifest_path)
    poems_dir = repo_root / "data" / "processed" / "poems"
    poems_dir.mkdir(parents=True)
    texts = {
        "train_one": "Amor antico nel core gentile\n" * 3,
        "train_two": "Donna cortese muove il mio disio\n" * 3,
        "validation": "Virtute chiara guida ogni pensiero\n" * 3,
        "test": "Memoria antica torna nel mio canto\n" * 3,
    }
    for poem_id, text in texts.items():
        (poems_dir / f"{poem_id}.txt").write_text(text, encoding="utf-8")

    tokenizer = train_bpe_tokenizer(
        texts=[texts["train_one"]],
        base_texts=list(texts.values()),
        vocab_size=80,
        special_tokens=[PRETRAINING_DOCUMENT_SEPARATOR],
    )
    parent_dir = repo_root / "runs" / "parent"
    parent_dir.mkdir(parents=True)
    tokenizer_path = parent_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    architecture = {
        "vocab_size": tokenizer.vocab_size,
        "embedding_dim": 8,
        "num_layers": 1,
        "num_heads": 2,
        "head_dim": 4,
        "feed_forward_dim": 16,
        "max_context_length": 8,
    }
    model = CausalTransformerLanguageModel(**architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    target_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    _, loss = model(input_ids, target_ids)
    assert loss is not None
    loss.backward()
    optimizer.step()
    parent_checkpoint_path = parent_dir / "model.pt"
    torch.save(
        {
            "step": 100,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": architecture,
            "vocab_size": tokenizer.vocab_size,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        parent_checkpoint_path,
    )
    architecture_path = repo_root / "runs" / "architecture.json"
    architecture_path.write_text(
        json.dumps({"model_architecture": architecture}),
        encoding="utf-8",
    )
    return parent_checkpoint_path, tokenizer_path, architecture


def tiny_control_config(
    repo_root: Path,
    initialization: str,
) -> SonnetControlRunConfig:
    return SonnetControlRunConfig(
        initialization=initialization,
        model_architecture_path="runs/architecture.json",
        pretraining_tokenizer_path="runs/parent/tokenizer.json",
        pretraining_checkpoint_path="runs/parent/model.pt",
        batch_size=2,
        context_length=8,
        train_steps=3,
        eval_interval=1,
        eval_batches=1,
        checkpoint_interval=2,
        learning_rate=3e-5,
        max_new_tokens=3,
        device="cpu",
    )


def test_load_model_architecture_reads_selection_manifest(tmp_path):
    _, _, architecture = write_control_inputs(tmp_path)

    loaded = load_model_architecture(tmp_path / "runs" / "architecture.json")

    assert loaded == {
        **architecture,
        "normalization_type": "layer_norm",
        "normalization_eps": 1e-5,
        "position_encoding_type": "learned_absolute",
        "rope_theta": 10_000.0,
        "feed_forward_type": "relu",
        "tie_token_embeddings": False,
    }


def test_pretrained_control_uses_parent_weights_with_fresh_adamw_state(tmp_path):
    parent_path, tokenizer_path, architecture = write_control_inputs(tmp_path)
    tokenizer = BytePairEncodingTokenizer.load(tokenizer_path)
    parent_checkpoint = torch.load(parent_path, map_location="cpu")

    model, optimizer, loaded_parent, initialization_metadata = initialize_control_model(
        repo_root=tmp_path,
        config=tiny_control_config(tmp_path, "pretrained"),
        tokenizer=tokenizer,
        model_architecture=architecture,
        device=torch.device("cpu"),
    )

    assert loaded_parent is not None
    assert initialization_metadata is None
    assert optimizer.state == {}
    assert torch.equal(
        model.embedding.token_embedding.weight,
        parent_checkpoint["model_state_dict"]["embedding.token_embedding.weight"],
    )


def test_learning_rate_schedule_returns_constant_or_warmup_cosine_values(tmp_path):
    constant_config = tiny_control_config(tmp_path, "random")
    scheduled_config = SonnetControlRunConfig(
        **{
            **constant_config.__dict__,
            "train_steps": 10,
            "learning_rate": 1e-3,
            "learning_rate_schedule": "warmup_cosine",
            "warmup_steps": 2,
            "min_learning_rate": 1e-4,
        }
    )

    assert learning_rate_for_step(constant_config, 1) == 3e-5
    assert learning_rate_for_step(scheduled_config, 1) == 5e-4
    assert learning_rate_for_step(scheduled_config, 2) == 1e-3
    assert learning_rate_for_step(scheduled_config, 10) == 1e-4


def test_set_optimizer_learning_rate_updates_adamw_parameter_groups():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    set_optimizer_learning_rate(optimizer, 2e-4)

    assert optimizer.param_groups[0]["lr"] == 2e-4


def test_control_run_logs_pre_clipping_gradient_norm_when_enabled(tmp_path):
    write_control_inputs(tmp_path)
    config = SonnetControlRunConfig(
        **{
            **tiny_control_config(tmp_path, "random").__dict__,
            "max_gradient_norm": 0.01,
        }
    )

    result = train_sonnet_control_run(tmp_path, tmp_path / "runs" / "clipped", config)
    history = [
        json.loads(line)
        for line in result["log_path"].read_text(encoding="utf-8").splitlines()
    ]

    assert all(row["pre_clipping_gradient_norm"] is not None for row in history)
    assert all(row["pre_clipping_gradient_norm"] > 0.01 for row in history)


def test_conversion_control_run_records_rms_norm_provenance(tmp_path):
    write_control_inputs(tmp_path)
    config = tiny_control_config(tmp_path, "layernorm_to_rmsnorm")

    result = train_sonnet_control_run(tmp_path, tmp_path / "runs" / "converted", config)
    run_metadata = json.loads(result["config_path"].read_text(encoding="utf-8"))
    checkpoint = torch.load(result["best_checkpoint_path"], map_location="cpu")

    assert run_metadata["model_architecture"]["normalization_type"] == "rms_norm"
    assert run_metadata["source_model_architecture"]["normalization_type"] == "layer_norm"
    assert run_metadata["initialization_metadata"]["conversion_type"] == (
        "layer_norm_to_rms_norm"
    )
    assert run_metadata["initialization_metadata"]["optimizer_state_restored"] is False
    assert checkpoint["initialization_metadata"] == run_metadata["initialization_metadata"]
    assert "final_layer_norm.bias" not in checkpoint["model_state_dict"]


def test_target_model_architecture_requires_layer_norm_source():
    source_architecture = {
        "vocab_size": 10,
        "embedding_dim": 8,
        "num_layers": 1,
        "num_heads": 2,
        "head_dim": 4,
        "feed_forward_dim": 16,
        "max_context_length": 8,
        "normalization_type": "rms_norm",
        "normalization_eps": 1e-5,
    }

    with pytest.raises(ValueError, match="LayerNorm architecture"):
        target_model_architecture(
            initialization="layernorm_to_rmsnorm",
            source_model_architecture=source_architecture,
        )


def test_target_model_architecture_uses_expanded_finetuning_vocabulary():
    source_architecture = {
        "vocab_size": 10,
        "embedding_dim": 8,
        "num_layers": 1,
        "num_heads": 2,
        "head_dim": 4,
        "feed_forward_dim": 16,
        "max_context_length": 8,
        "normalization_type": "rms_norm",
        "normalization_eps": 1e-5,
    }

    target_architecture = target_model_architecture(
        initialization="pretrained",
        source_model_architecture=source_architecture,
        target_vocab_size=13,
    )

    assert source_architecture["vocab_size"] == 10
    assert target_architecture["vocab_size"] == 13
    assert target_architecture["normalization_type"] == "rms_norm"


def test_control_arms_write_matching_data_and_architecture_metadata(tmp_path):
    write_control_inputs(tmp_path)
    random_result = train_sonnet_control_run(
        tmp_path,
        tmp_path / "runs" / "random",
        tiny_control_config(tmp_path, "random"),
    )
    pretrained_result = train_sonnet_control_run(
        tmp_path,
        tmp_path / "runs" / "pretrained",
        tiny_control_config(tmp_path, "pretrained"),
    )
    random_config = json.loads(random_result["config_path"].read_text(encoding="utf-8"))
    pretrained_config = json.loads(
        pretrained_result["config_path"].read_text(encoding="utf-8")
    )
    best_checkpoint = torch.load(
        random_result["best_checkpoint_path"],
        map_location="cpu",
    )

    for field in (
        "dataset",
        "vocab_size",
        "train_tokens",
        "validation_tokens",
        "model_architecture",
        "batch_size",
        "context_length",
        "train_steps",
        "learning_rate",
    ):
        assert random_config[field] == pretrained_config[field]
    assert random_config["initialization"] == "random"
    assert pretrained_config["initialization"] == "pretrained"
    assert random_config["optimizer_state_restored"] is False
    assert pretrained_config["optimizer_state_restored"] is False
    assert random_result["checkpoint_dir"].joinpath("step_2.pt").is_file()
    assert random_result["best_checkpoint_path"].is_file()
    assert best_checkpoint["step"] == random_config["best_validation_step"]


def test_control_run_uses_explicit_manifest_and_records_its_hash(tmp_path):
    write_control_inputs(tmp_path)
    default_manifest = tmp_path / "data" / "metadata" / "poems_manifest.csv"
    versioned_manifest = (
        tmp_path / "data" / "metadata" / "sonnets_expanded_test_manifest.csv"
    )
    default_manifest.replace(versioned_manifest)
    config = SonnetControlRunConfig(
        **{
            **tiny_control_config(tmp_path, "random").__dict__,
            "manifest_path": (
                "data/metadata/sonnets_expanded_test_manifest.csv"
            ),
        }
    )

    result = train_sonnet_control_run(
        tmp_path,
        tmp_path / "runs" / "versioned_manifest",
        config,
    )
    run_metadata = json.loads(result["config_path"].read_text(encoding="utf-8"))
    checkpoint = torch.load(result["best_checkpoint_path"], map_location="cpu")
    expected_hash = hashlib.sha256(versioned_manifest.read_bytes()).hexdigest()

    assert run_metadata["manifest_path"] == config.manifest_path
    assert run_metadata["manifest_sha256"] == expected_hash
    assert checkpoint["config"]["manifest_path"] == config.manifest_path
    assert checkpoint["manifest_sha256"] == expected_hash


def test_control_run_keeps_historical_manifest_as_default():
    config = SonnetControlRunConfig()

    assert config.manifest_path == "data/metadata/poems_manifest.csv"


def test_control_run_rejects_missing_explicit_manifest_before_training(tmp_path):
    config = SonnetControlRunConfig(
        **{
            **tiny_control_config(tmp_path, "random").__dict__,
            "manifest_path": "data/metadata/missing.csv",
        }
    )

    with pytest.raises(FileNotFoundError, match="sonnet manifest does not exist"):
        train_sonnet_control_run(
            tmp_path,
            tmp_path / "runs" / "missing_manifest",
            config,
        )


def test_control_run_rejects_invalid_manifest_before_training(tmp_path):
    invalid_manifest = tmp_path / "data" / "metadata" / "invalid.csv"
    invalid_manifest.parent.mkdir(parents=True)
    invalid_manifest.write_text("poem_id\npoem\n", encoding="utf-8")
    config = SonnetControlRunConfig(
        **{
            **tiny_control_config(tmp_path, "random").__dict__,
            "manifest_path": "data/metadata/invalid.csv",
        }
    )

    with pytest.raises(ValueError, match="sonnet manifest is missing columns"):
        train_sonnet_control_run(
            tmp_path,
            tmp_path / "runs" / "invalid_manifest",
            config,
        )


def test_control_run_uses_sequential_validation_and_records_its_coverage(tmp_path):
    write_control_inputs(tmp_path)

    result = train_sonnet_control_run(
        tmp_path,
        tmp_path / "runs" / "sequential_validation",
        tiny_control_config(tmp_path, "random"),
    )
    run_metadata = json.loads(result["config_path"].read_text(encoding="utf-8"))

    assert run_metadata["validation_mode"] == "sequential_windows"
    assert run_metadata["validation_window_count"] == (
        run_metadata["validation_tokens"] - 1
    ) // run_metadata["context_length"]
    assert run_metadata["stop_reason"] == "max_train_steps_reached"


def test_control_run_stops_early_and_records_actual_completion(tmp_path):
    write_control_inputs(tmp_path)
    config = SonnetControlRunConfig(
        **{
            **tiny_control_config(tmp_path, "random").__dict__,
            "train_steps": 10,
            "checkpoint_interval": 0,
            "early_stopping_patience": 1,
            "min_validation_improvement": 1_000.0,
        }
    )

    result = train_sonnet_control_run(
        tmp_path,
        tmp_path / "runs" / "early_stopped",
        config,
    )
    run_metadata = json.loads(result["config_path"].read_text(encoding="utf-8"))
    final_checkpoint = torch.load(result["checkpoint_path"], map_location="cpu")
    best_checkpoint = torch.load(result["best_checkpoint_path"], map_location="cpu")

    assert result["completed_steps"] == 2
    assert result["stop_reason"] == "early_stopping_patience_exhausted"
    assert len(result["history"]) == 2
    assert result["history"][-1]["non_improving_evaluations"] == 1
    assert run_metadata["completed_steps"] == 2
    assert run_metadata["stop_reason"] == "early_stopping_patience_exhausted"
    assert final_checkpoint["completed_steps"] == 2
    assert final_checkpoint["stop_reason"] == "early_stopping_patience_exhausted"
    assert best_checkpoint["step"] == 1
    assert best_checkpoint["stop_reason"] is None
