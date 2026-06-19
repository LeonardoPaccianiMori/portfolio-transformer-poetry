import csv
from pathlib import Path

import pytest
import torch

from sonnet_corpus.dataset_text import (
    build_text_stream,
    dataset_include_column,
    dataset_split_column,
    encode_text_stream,
    load_encoded_splits,
    load_poem_text,
    load_poem_texts,
    load_split_text_stream,
    read_manifest_rows,
    select_manifest_rows,
)

from sonnet_corpus.tokenizer import CharTokenizer


def write_test_manifest(path: Path) -> None:
    rows = [
        {
            "poem_id": "poem_train",
            "clean_text_path": "data/processed/poems/poem_train.txt",
            "include_in_core_pre_petrarch": "True",
            "include_in_expanded_with_petrarch": "True",
            "split_core_pre_petrarch": "train",
            "split_expanded_with_petrarch": "train",
        },
        {
            "poem_id": "poem_validation",
            "clean_text_path": "data/processed/poems/poem_validation.txt",
            "include_in_core_pre_petrarch": "True",
            "include_in_expanded_with_petrarch": "True",
            "split_core_pre_petrarch": "validation",
            "split_expanded_with_petrarch": "validation",
        },
        {
            "poem_id": "poem_excluded",
            "clean_text_path": "",
            "include_in_core_pre_petrarch": "False",
            "include_in_expanded_with_petrarch": "False",
            "split_core_pre_petrarch": "excluded",
            "split_expanded_with_petrarch": "excluded",
        },
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_dataset_column_helpers_return_manifest_column_names():
    assert (
        dataset_include_column("expanded_with_petrarch")
        == "include_in_expanded_with_petrarch"
    )
    assert (
        dataset_split_column("expanded_with_petrarch")
        == "split_expanded_with_petrarch"
    )


def test_read_manifest_rows_loads_csv_rows(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    write_test_manifest(manifest_path)

    rows = read_manifest_rows(manifest_path)

    assert len(rows) == 3
    assert rows[0]["poem_id"] == "poem_train"


def test_select_manifest_rows_filters_by_dataset_and_split(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    write_test_manifest(manifest_path)
    rows = read_manifest_rows(manifest_path)

    selected_rows = select_manifest_rows(
        rows=rows,
        dataset="expanded_with_petrarch",
        split="train",
    )

    assert len(selected_rows) == 1
    assert selected_rows[0]["poem_id"] == "poem_train"


def test_select_manifest_rows_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="unknown dataset"):
        select_manifest_rows(
            rows=[],
            dataset="unknown_dataset",
            split="train",
        )


def test_select_manifest_rows_rejects_unknown_split():
    with pytest.raises(ValueError, match="unknown split"):
        select_manifest_rows(
            rows=[],
            dataset="expanded_with_petrarch",
            split="dev",
        )


def test_load_poem_text_reads_processed_text(tmp_path):
    poem_path = tmp_path / "data" / "processed" / "poems" / "poem_train.txt"
    poem_path.parent.mkdir(parents=True)
    poem_path.write_text("Amor\n", encoding="utf-8")

    row = {
        "poem_id": "poem_train",
        "clean_text_path": "data/processed/poems/poem_train.txt",
    }

    text = load_poem_text(row, repo_root=tmp_path)

    assert text == "Amor\n"


def test_load_poem_texts_reads_multiple_rows(tmp_path):
    first_path = tmp_path / "data" / "processed" / "poems" / "first.txt"
    second_path = tmp_path / "data" / "processed" / "poems" / "second.txt"
    first_path.parent.mkdir(parents=True)
    first_path.write_text("First\n", encoding="utf-8")
    second_path.write_text("Second\n", encoding="utf-8")

    rows = [
        {
            "poem_id": "first",
            "clean_text_path": "data/processed/poems/first.txt",
        },
        {
            "poem_id": "second",
            "clean_text_path": "data/processed/poems/second.txt",
        },
    ]

    texts = load_poem_texts(rows, repo_root=tmp_path)

    assert texts == ["First\n", "Second\n"]


def test_load_poem_text_rejects_empty_clean_text_path(tmp_path):
    row = {
        "poem_id": "missing_path",
        "clean_text_path": "",
    }

    with pytest.raises(ValueError, match="no clean_text_path"):
        load_poem_text(row, repo_root=tmp_path)


def test_load_poem_text_rejects_missing_file(tmp_path):
    row = {
        "poem_id": "missing_file",
        "clean_text_path": "data/processed/poems/missing.txt",
    }

    with pytest.raises(FileNotFoundError, match="missing poem text file"):
        load_poem_text(row, repo_root=tmp_path)


def test_build_text_stream_joins_poems_with_default_separator():
    texts = ["First poem\n", "Second poem\n"]

    text_stream = build_text_stream(texts)

    assert text_stream == "First poem\n\n\n<|poem_end|>\n\nSecond poem\n"


def test_build_text_stream_accepts_custom_separator():
    texts = ["First poem\n", "Second poem\n"]

    text_stream = build_text_stream(
        texts,
        poem_separator="\n---\n",
    )

    assert text_stream == "First poem\n\n---\nSecond poem\n"


def test_build_text_stream_rejects_empty_text_list():
    with pytest.raises(ValueError, match="at least one poem"):
        build_text_stream([])


def test_real_expanded_train_split_loads_processed_poems():
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "data" / "metadata" / "poems_manifest.csv"

    rows = read_manifest_rows(manifest_path)
    selected_rows = select_manifest_rows(
        rows=rows,
        dataset="expanded_with_petrarch",
        split="train",
    )
    texts = load_poem_texts(selected_rows, repo_root=repo_root)
    text_stream = build_text_stream(texts)

    assert len(selected_rows) > 0
    assert len(texts) == len(selected_rows)
    assert all(text.strip() for text in texts)
    assert "<|poem_end|>" in text_stream


def test_encode_text_stream_returns_1d_long_tensor():
    text_stream = "Amor\n<|poem_end|>\n"
    tokenizer = CharTokenizer.from_texts([text_stream])

    encoded = encode_text_stream(text_stream, tokenizer)

    assert encoded.ndim == 1
    assert encoded.dtype == torch.long
    assert encoded.tolist() == tokenizer.encode(text_stream)


def test_encode_text_stream_round_trips_through_tokenizer():
    text_stream = "Amor\n<|poem_end|>\n"
    tokenizer = CharTokenizer.from_texts([text_stream])

    encoded = encode_text_stream(text_stream, tokenizer)

    decoded = tokenizer.decode(encoded.tolist())

    assert decoded == text_stream


def test_encode_text_stream_rejects_empty_stream():
    tokenizer = CharTokenizer.from_texts(["abc"])

    with pytest.raises(ValueError, match="must not be empty"):
        encode_text_stream("", tokenizer)


def test_load_split_text_stream_builds_stream_from_manifest_rows(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    write_test_manifest(manifest_path)

    poem_path = tmp_path / "data" / "processed" / "poems" / "poem_train.txt"
    poem_path.parent.mkdir(parents=True)
    poem_path.write_text("Amor\n", encoding="utf-8")

    text_stream = load_split_text_stream(
        manifest_path=manifest_path,
        repo_root=tmp_path,
        dataset="expanded_with_petrarch",
        split="train",
    )

    assert text_stream == "Amor\n"


def test_load_split_text_stream_uses_poem_separator_for_multiple_poems(tmp_path):
    manifest_path = tmp_path / "manifest.csv"

    rows = [
        {
            "poem_id": "first",
            "clean_text_path": "data/processed/poems/first.txt",
            "include_in_core_pre_petrarch": "True",
            "include_in_expanded_with_petrarch": "True",
            "split_core_pre_petrarch": "train",
            "split_expanded_with_petrarch": "train",
        },
        {
            "poem_id": "second",
            "clean_text_path": "data/processed/poems/second.txt",
            "include_in_core_pre_petrarch": "True",
            "include_in_expanded_with_petrarch": "True",
            "split_core_pre_petrarch": "train",
            "split_expanded_with_petrarch": "train",
        },
    ]

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    first_path = tmp_path / "data" / "processed" / "poems" / "first.txt"
    second_path = tmp_path / "data" / "processed" / "poems" / "second.txt"
    first_path.parent.mkdir(parents=True)
    first_path.write_text("First\n", encoding="utf-8")
    second_path.write_text("Second\n", encoding="utf-8")

    text_stream = load_split_text_stream(
        manifest_path=manifest_path,
        repo_root=tmp_path,
        dataset="expanded_with_petrarch",
        split="train",
        poem_separator="\n---\n",
    )

    assert text_stream == "First\n\n---\nSecond\n"


def write_split_manifest(path: Path) -> None:
    rows = [
        {
            "poem_id": "train",
            "clean_text_path": "data/processed/poems/train.txt",
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
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_load_encoded_splits_returns_split_tensors_and_shared_tokenizer(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    write_split_manifest(manifest_path)

    poems_dir = tmp_path / "data" / "processed" / "poems"
    poems_dir.mkdir(parents=True)
    (poems_dir / "train.txt").write_text("abc\n", encoding="utf-8")
    (poems_dir / "validation.txt").write_text("cabV\n", encoding="utf-8")
    (poems_dir / "test.txt").write_text("bcaT\n", encoding="utf-8")

    train_tokens, validation_tokens, test_tokens, tokenizer = load_encoded_splits(
        manifest_path=manifest_path,
        repo_root=tmp_path,
        dataset="expanded_with_petrarch",
    )

    assert train_tokens.dtype == torch.long
    assert validation_tokens.dtype == torch.long
    assert test_tokens.dtype == torch.long

    assert train_tokens.ndim == 1
    assert validation_tokens.ndim == 1
    assert test_tokens.ndim == 1

    assert tokenizer.decode(train_tokens.tolist()) == "abc\n"
    assert tokenizer.decode(validation_tokens.tolist()) == "cabV\n"
    assert tokenizer.decode(test_tokens.tolist()) == "bcaT\n"
