import csv
from pathlib import Path

import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer
from sonnet_corpus.pretraining_tokenizer import encode_text_by_pretoken
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


def validate_manifest_rows(
    rows: list[dict[str, str]],
    dataset: str,
) -> None:
    """Reject empty or structurally incompatible sonnet manifests."""
    if not rows:
        raise ValueError("sonnet manifest must contain at least one data row")

    required_columns = {
        "poem_id",
        "clean_text_path",
        dataset_include_column(dataset),
        dataset_split_column(dataset),
    }
    missing_columns = sorted(required_columns - rows[0].keys())
    if missing_columns:
        raise ValueError(
            "sonnet manifest is missing columns: " + ", ".join(missing_columns)
        )


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
PRETRAINING_DOCUMENT_SEPARATOR = "<|endoftext|>"


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


def encode_bpe_text_stream(
    text_stream: str,
    tokenizer: BytePairEncodingTokenizer,
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


def load_bpe_encoded_splits(
    manifest_path: Path,
    repo_root: Path,
    dataset: str,
    tokenizer_path: Path,
    poem_separator: str = DEFAULT_POEM_SEPARATOR,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    BytePairEncodingTokenizer,
]:
    tokenizer = BytePairEncodingTokenizer.load(tokenizer_path)
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

    train_tokens = encode_bpe_text_stream(train_text, tokenizer)
    validation_tokens = encode_bpe_text_stream(validation_text, tokenizer)
    test_tokens = encode_bpe_text_stream(test_text, tokenizer)

    return train_tokens, validation_tokens, test_tokens, tokenizer


def encode_poem_texts_with_pretraining_tokenizer(
    texts: list[str],
    tokenizer: BytePairEncodingTokenizer,
    document_separator: str = PRETRAINING_DOCUMENT_SEPARATOR,
) -> torch.Tensor:
    """Encode one sonnet split with the tokenizer used by pretraining.

    The separator is appended between poems as one protected BPE token. This
    preserves poem boundaries while keeping IDs compatible with the parent
    transformer's token embedding and output projection.
    """
    if not texts:
        raise ValueError("texts must contain at least one poem")

    separator_ids = tokenizer.encode(document_separator)
    if len(separator_ids) != 1:
        raise ValueError(
            "document_separator must encode to exactly one token for fine-tuning"
        )

    token_ids: list[int] = []
    for index, text in enumerate(texts):
        if not text.strip():
            raise ValueError("poem text must not be empty")
        if index > 0:
            token_ids.extend(separator_ids)
        token_ids.extend(encode_text_by_pretoken(text, tokenizer))

    return torch.tensor(token_ids, dtype=torch.long)


def missing_tokenizer_characters(
    texts: list[str],
    tokenizer: BytePairEncodingTokenizer,
) -> list[str]:
    """Return sorted literal characters absent from a fixed BPE vocabulary."""
    return sorted({
        character
        for text in texts
        for character in text
        if character not in tokenizer.token_to_id
    })


def extend_tokenizer_for_character_coverage(
    tokenizer: BytePairEncodingTokenizer,
    texts: list[str],
) -> tuple[BytePairEncodingTokenizer, list[str]]:
    """Append raw character tokens needed to preserve fine-tuning text exactly.

    Existing token IDs and merge rules are unchanged. New characters receive
    IDs after the parent vocabulary, so a fine-tuning model can retain every
    parent embedding/output row and learn only the appended rows.
    """
    added_characters = missing_tokenizer_characters(texts, tokenizer)
    if not added_characters:
        return tokenizer, []

    token_to_id = dict(tokenizer.token_to_id)
    for character in added_characters:
        token_to_id[character] = len(token_to_id)

    return (
        BytePairEncodingTokenizer(
            token_to_id=token_to_id,
            merges=list(tokenizer.merges),
            special_tokens=list(tokenizer.special_tokens),
        ),
        added_characters,
    )


def load_pretraining_bpe_encoded_splits(
    manifest_path: Path,
    repo_root: Path,
    dataset: str,
    tokenizer_path: Path,
    document_separator: str = PRETRAINING_DOCUMENT_SEPARATOR,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    BytePairEncodingTokenizer,
]:
    """Load sonnet splits encoded by the fixed broader-pretraining tokenizer."""
    tokenizer = BytePairEncodingTokenizer.load(tokenizer_path)
    rows = read_manifest_rows(manifest_path)
    validate_manifest_rows(rows, dataset)

    split_texts: list[list[str]] = []
    for split in ("train", "validation", "test"):
        selected_rows = select_manifest_rows(
            rows=rows,
            dataset=dataset,
            split=split,
        )
        split_texts.append(load_poem_texts(selected_rows, repo_root=repo_root))

    tokenizer, _ = extend_tokenizer_for_character_coverage(
        tokenizer=tokenizer,
        texts=[text for texts in split_texts for text in texts],
    )
    encoded_splits = []
    for texts in split_texts:
        encoded_splits.append(
            encode_poem_texts_with_pretraining_tokenizer(
                texts=texts,
                tokenizer=tokenizer,
                document_separator=document_separator,
            )
        )

    train_tokens, validation_tokens, test_tokens = encoded_splits
    return train_tokens, validation_tokens, test_tokens, tokenizer
