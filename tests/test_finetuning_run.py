import csv
import json
from pathlib import Path

import pytest
import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer, train_bpe_tokenizer
from sonnet_corpus.dataset_text import (
    PRETRAINING_DOCUMENT_SEPARATOR,
    encode_poem_texts_with_pretraining_tokenizer,
    extend_tokenizer_for_character_coverage,
    load_pretraining_bpe_encoded_splits,
)
from sonnet_model.transformer import CausalTransformerLanguageModel
from sonnet_training.finetuning_run import (
    FineTuningRunConfig,
    load_parent_for_finetuning,
    train_finetuning_run,
)


def write_manifest(path: Path) -> None:
    rows = [
        {
            "poem_id": "train_one",
            "clean_text_path": "data/processed/poems/train_one.txt",
            "include_in_core_pre_petrarch": "True",
            "include_in_expanded_with_petrarch": "True",
            "split_core_pre_petrarch": "train",
            "split_expanded_with_petrarch": "train",
        },
        {
            "poem_id": "train_two",
            "clean_text_path": "data/processed/poems/train_two.txt",
            "include_in_core_pre_petrarch": "True",
            "include_in_expanded_with_petrarch": "True",
            "split_core_pre_petrarch": "train",
            "split_expanded_with_petrarch": "train",
        },
        {
            "poem_id": "validation",
            "clean_text_path": "data/processed/poems/validation.txt",
            "include_in_core_pre_petrarch": "True",
            "include_in_expanded_with_petrarch": "True",
            "split_core_pre_petrarch": "validation",
            "split_expanded_with_petrarch": "validation",
        },
        {
            "poem_id": "test",
            "clean_text_path": "data/processed/poems/test.txt",
            "include_in_core_pre_petrarch": "True",
            "include_in_expanded_with_petrarch": "True",
            "split_core_pre_petrarch": "test",
            "split_expanded_with_petrarch": "test",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_tiny_finetuning_repository(
    repo_root: Path,
    *,
    tie_token_embeddings: bool = False,
) -> tuple[Path, Path]:
    manifest_path = repo_root / "data" / "metadata" / "poems_manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    write_manifest(manifest_path)
    poems_dir = repo_root / "data" / "processed" / "poems"
    poems_dir.mkdir(parents=True)
    texts = {
        "train_one.txt": "Amor antico nel core gentile\n" * 3,
        "train_two.txt": "Donna cortese muove il mio disio\n" * 3,
        "validation.txt": "Virtüte chiara guida ogni pensiero\n" * 3,
        "test.txt": "Memoria antica torna nel mio canto\n" * 3,
    }
    for filename, text in texts.items():
        (poems_dir / filename).write_text(text, encoding="utf-8")

    tokenizer = train_bpe_tokenizer(
        texts=[texts["train_one.txt"]],
        base_texts=[texts["train_one.txt"], texts["train_two.txt"]],
        vocab_size=80,
        special_tokens=[PRETRAINING_DOCUMENT_SEPARATOR],
    )
    parent_dir = repo_root / "runs" / "parent"
    parent_dir.mkdir(parents=True)
    tokenizer_path = parent_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    model = CausalTransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        feed_forward_dim=16,
        max_context_length=8,
        tie_token_embeddings=tie_token_embeddings,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    target_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    _, loss = model(input_ids, target_ids)
    assert loss is not None
    loss.backward()
    optimizer.step()

    checkpoint_path = parent_dir / "model.pt"
    torch.save(
        {
            "step": 100,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "embedding_dim": 8,
                "num_layers": 1,
                "num_heads": 2,
                "head_dim": 4,
                "feed_forward_dim": 16,
                "max_context_length": 8,
                "tie_token_embeddings": tie_token_embeddings,
            },
            "vocab_size": tokenizer.vocab_size,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        checkpoint_path,
    )
    return checkpoint_path, tokenizer_path


def test_encode_poem_texts_uses_one_pretraining_separator_between_poems():
    tokenizer = train_bpe_tokenizer(
        texts=["Amor\nDonna\n"],
        vocab_size=30,
        special_tokens=[PRETRAINING_DOCUMENT_SEPARATOR],
    )

    tokens = encode_poem_texts_with_pretraining_tokenizer(
        texts=["Amor\n", "Donna\n"],
        tokenizer=tokenizer,
    )

    assert tokenizer.decode(tokens.tolist()) == "Amor\n<|endoftext|>Donna\n"
    separator_id = tokenizer.encode(PRETRAINING_DOCUMENT_SEPARATOR)[0]
    assert tokens.tolist().count(separator_id) == 1


def test_extend_tokenizer_for_character_coverage_appends_missing_characters():
    tokenizer = train_bpe_tokenizer(
        texts=["Amor virtlma\n"],
        vocab_size=20,
        special_tokens=[PRETRAINING_DOCUMENT_SEPARATOR],
    )

    extended_tokenizer, added_characters = extend_tokenizer_for_character_coverage(
        tokenizer,
        ["Amor\n", "virtü\n", "lʼalma\n"],
    )

    assert added_characters == ["ü", "ʼ"]
    assert extended_tokenizer.vocab_size == tokenizer.vocab_size + 2
    assert extended_tokenizer.decode(extended_tokenizer.encode("virtü\n")) == "virtü\n"


def test_load_pretraining_bpe_encoded_splits_preserves_manifest_splits(tmp_path):
    _, tokenizer_path = write_tiny_finetuning_repository(tmp_path)

    train_tokens, validation_tokens, test_tokens, tokenizer = (
        load_pretraining_bpe_encoded_splits(
            manifest_path=tmp_path / "data" / "metadata" / "poems_manifest.csv",
            repo_root=tmp_path,
            dataset="expanded_with_petrarch",
            tokenizer_path=tokenizer_path,
        )
    )

    assert PRETRAINING_DOCUMENT_SEPARATOR in tokenizer.decode(train_tokens.tolist())
    assert PRETRAINING_DOCUMENT_SEPARATOR not in tokenizer.decode(validation_tokens.tolist())
    assert "Virtüte" in tokenizer.decode(validation_tokens.tolist())
    assert "Memoria" in tokenizer.decode(test_tokens.tolist())
    assert "ü" in tokenizer.token_to_id


def test_load_parent_for_finetuning_restores_weights_and_approved_learning_rate(tmp_path):
    checkpoint_path, tokenizer_path = write_tiny_finetuning_repository(tmp_path)
    tokenizer = BytePairEncodingTokenizer.load(tokenizer_path)

    model, optimizer, checkpoint = load_parent_for_finetuning(
        checkpoint_path=checkpoint_path,
        tokenizer=tokenizer,
        learning_rate=3e-5,
        restore_optimizer_state=True,
        device=torch.device("cpu"),
    )

    assert checkpoint["step"] == 100
    assert model.max_context_length == 8
    assert optimizer.param_groups[0]["lr"] == 3e-5
    assert optimizer.state


def test_load_parent_for_finetuning_expands_vocabulary_weights_and_adamw_state(tmp_path):
    checkpoint_path, tokenizer_path = write_tiny_finetuning_repository(tmp_path)
    parent_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    tokenizer = BytePairEncodingTokenizer.load(tokenizer_path)
    extended_tokenizer, added_characters = extend_tokenizer_for_character_coverage(
        tokenizer,
        ["virtü\n"],
    )

    model, optimizer, _ = load_parent_for_finetuning(
        checkpoint_path=checkpoint_path,
        tokenizer=extended_tokenizer,
        learning_rate=3e-5,
        restore_optimizer_state=True,
        device=torch.device("cpu"),
    )

    parent_vocab_size = parent_checkpoint["vocab_size"]
    assert added_characters == ["ü"]
    assert model.output_projection.out_features == parent_vocab_size + 1
    assert torch.equal(
        model.embedding.token_embedding.weight[:parent_vocab_size],
        parent_checkpoint["model_state_dict"]["embedding.token_embedding.weight"],
    )
    assert (
        optimizer.state[model.embedding.token_embedding.weight]["exp_avg"].shape
        == model.embedding.token_embedding.weight.shape
    )
    assert (
        optimizer.state[model.output_projection.weight]["exp_avg_sq"].shape
        == model.output_projection.weight.shape
    )


def test_load_parent_for_finetuning_preserves_tied_embeddings_when_expanding_vocab(
    tmp_path,
):
    checkpoint_path, tokenizer_path = write_tiny_finetuning_repository(
        tmp_path,
        tie_token_embeddings=True,
    )
    parent_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    tokenizer = BytePairEncodingTokenizer.load(tokenizer_path)
    extended_tokenizer, _ = extend_tokenizer_for_character_coverage(
        tokenizer,
        ["virtü\n"],
    )

    model, optimizer, _ = load_parent_for_finetuning(
        checkpoint_path=checkpoint_path,
        tokenizer=extended_tokenizer,
        learning_rate=3e-5,
        restore_optimizer_state=True,
        device=torch.device("cpu"),
    )

    parent_vocab_size = parent_checkpoint["vocab_size"]
    assert model.tie_token_embeddings is True
    assert model.embedding.token_embedding.weight is model.output_projection.weight
    assert torch.equal(
        model.embedding.token_embedding.weight[:parent_vocab_size],
        parent_checkpoint["model_state_dict"]["embedding.token_embedding.weight"],
    )
    assert (
        optimizer.state[model.embedding.token_embedding.weight]["exp_avg"].shape
        == model.embedding.token_embedding.weight.shape
    )


def test_load_parent_for_finetuning_rejects_vocab_mismatch(tmp_path):
    checkpoint_path, _ = write_tiny_finetuning_repository(tmp_path)
    mismatched_tokenizer = BytePairEncodingTokenizer(
        token_to_id={"a": 0, "b": 1, PRETRAINING_DOCUMENT_SEPARATOR: 2},
        merges=[],
        special_tokens=[PRETRAINING_DOCUMENT_SEPARATOR],
    )

    with pytest.raises(ValueError, match="vocabulary size"):
        load_parent_for_finetuning(
            checkpoint_path=checkpoint_path,
            tokenizer=mismatched_tokenizer,
            learning_rate=3e-5,
            restore_optimizer_state=True,
            device=torch.device("cpu"),
        )


def test_train_finetuning_run_writes_lineage_and_interval_checkpoints(tmp_path):
    checkpoint_path, tokenizer_path = write_tiny_finetuning_repository(tmp_path)
    output_dir = tmp_path / "runs" / "finetuning"
    config = FineTuningRunConfig(
        pretraining_checkpoint_path=str(checkpoint_path.relative_to(tmp_path)),
        pretraining_tokenizer_path=str(tokenizer_path.relative_to(tmp_path)),
        batch_size=2,
        context_length=8,
        train_steps=3,
        eval_interval=1,
        eval_batches=1,
        checkpoint_interval=2,
        learning_rate=3e-5,
        prompt="Amor",
        max_new_tokens=3,
        device="cpu",
    )

    result = train_finetuning_run(tmp_path, output_dir, config)

    saved_config = json.loads(result["config_path"].read_text(encoding="utf-8"))
    saved_checkpoint = torch.load(result["checkpoint_path"], map_location="cpu")
    assert result["checkpoint_dir"].joinpath("step_2.pt").is_file()
    assert len(result["history"]) == 3
    assert saved_config["parent_checkpoint_step"] == 100
    assert saved_config["restore_optimizer_state"] is True
    assert "ü" in saved_config["added_token_strings"]
    assert saved_checkpoint["parent_parameter_count"] == saved_config["parent_parameter_count"]
    assert (
        saved_checkpoint["parent_vocab_size"]
        + len(saved_config["added_token_strings"])
        == saved_config["vocab_size"]
    )


def test_train_finetuning_run_rejects_context_longer_than_parent_model(tmp_path):
    checkpoint_path, tokenizer_path = write_tiny_finetuning_repository(tmp_path)
    config = FineTuningRunConfig(
        pretraining_checkpoint_path=str(checkpoint_path.relative_to(tmp_path)),
        pretraining_tokenizer_path=str(tokenizer_path.relative_to(tmp_path)),
        context_length=9,
        train_steps=1,
        eval_interval=1,
        eval_batches=1,
        device="cpu",
    )

    with pytest.raises(ValueError, match="parent model context"):
        train_finetuning_run(tmp_path, tmp_path / "runs" / "finetuning", config)
