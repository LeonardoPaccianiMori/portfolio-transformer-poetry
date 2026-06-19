import csv
from pathlib import Path

import torch

from sonnet_corpus.tokenizer import CharTokenizer


VALID_DATASETS = {
    "core_pre_petrarch",
    "expanded_with_petrarch",
}

VALID_SPLITS = {
    "train",
    "validation",
    "test",
}


def dataset_include_column(dataset: str) -> str:
    if dataset not in VALID_DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")

    return f"include_in_{dataset}"


def dataset_split_column(dataset: str) -> str:
    if dataset not in VALID_DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")

    return f"split_{dataset}"


def read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def select_manifest_rows(
    rows: list[dict[str, str]],
    dataset: str,
    split: str,
) -> list[dict[str, str]]:
    if split not in VALID_SPLITS:
        raise ValueError(f"unknown split: {split}")

    include_column = dataset_include_column(dataset)
    split_column = dataset_split_column(dataset)

    selected_rows = []

    for row in rows:
        is_in_dataset = row[include_column] == "True"
        is_in_split = row[split_column] == split
        has_clean_text = row["clean_text_path"] != ""

        if is_in_dataset and is_in_split and has_clean_text:
            selected_rows.append(row)

    return selected_rows


def load_poem_text(row: dict[str, str], repo_root: Path) -> str:
    clean_text_path = row["clean_text_path"]
    poem_path = repo_root / clean_text_path

    if clean_text_path == "":
        raise ValueError(f"row has no clean_text_path: {row['poem_id']}")

    if not poem_path.is_file():
        raise FileNotFoundError(f"missing poem text file: {poem_path}")

    return poem_path.read_text(encoding="utf-8")


def load_poem_texts(
    rows: list[dict[str, str]],
    repo_root: Path,
) -> list[str]:
    return [
        load_poem_text(row, repo_root)
        for row in rows
    ]

DEFAULT_POEM_SEPARATOR = "\n\n<|poem_end|>\n\n"


def build_text_stream(
    texts: list[str],
    poem_separator: str = DEFAULT_POEM_SEPARATOR,
) -> str:
    if not texts:
        raise ValueError("texts must contain at least one poem")

    return poem_separator.join(texts)


def encode_text_stream(
    text_stream: str,
    tokenizer: CharTokenizer,
) -> torch.Tensor:
    if text_stream == "":
        raise ValueError("text_stream must not be empty")

    token_ids = tokenizer.encode(text_stream)

    return torch.tensor(
        token_ids,
        dtype=torch.long,
    )


def load_split_text_stream(
    manifest_path: Path,
    repo_root: Path,
    dataset: str,
    split: str,
    poem_separator: str = DEFAULT_POEM_SEPARATOR,
) -> str:
    rows = read_manifest_rows(manifest_path)
    selected_rows = select_manifest_rows(
        rows=rows,
        dataset=dataset,
        split=split,
    )
    texts = load_poem_texts(
        rows=selected_rows,
        repo_root=repo_root,
    )

    return build_text_stream(
        texts,
        poem_separator=poem_separator,
    )


def load_encoded_splits(
    manifest_path: Path,
    repo_root: Path,
    dataset: str,
    poem_separator: str = DEFAULT_POEM_SEPARATOR,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, CharTokenizer]:
    train_text = load_split_text_stream(
        manifest_path=manifest_path,
        repo_root=repo_root,
        dataset=dataset,
        split="train",
        poem_separator=poem_separator,
    )
    validation_text = load_split_text_stream(
        manifest_path=manifest_path,
        repo_root=repo_root,
        dataset=dataset,
        split="validation",
        poem_separator=poem_separator,
    )
    test_text = load_split_text_stream(
        manifest_path=manifest_path,
        repo_root=repo_root,
        dataset=dataset,
        split="test",
        poem_separator=poem_separator,
    )

    tokenizer = CharTokenizer.from_texts([
        train_text,
        validation_text,
        test_text,
    ])

    train_tokens = encode_text_stream(train_text, tokenizer)
    validation_tokens = encode_text_stream(validation_text, tokenizer)
    test_tokens = encode_text_stream(test_text, tokenizer)

    return train_tokens, validation_tokens, test_tokens, tokenizer
