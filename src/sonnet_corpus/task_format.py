"""Create masked opening-line continuation examples from 14-line sonnets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer
from sonnet_corpus.dataset_text import (
    PRETRAINING_DOCUMENT_SEPARATOR,
    extend_tokenizer_for_character_coverage,
    extend_tokenizer_for_special_tokens,
    load_poem_text,
    read_manifest_rows,
    select_manifest_rows,
    validate_manifest_rows,
)
from sonnet_corpus.pretraining_tokenizer import encode_text_by_pretoken


SONNET_OPENING_TOKEN = "<|sonnet_opening|>"
SONNET_CONTINUATION_TOKEN = "<|sonnet_continuation|>"
TASK_FORMAT_SPECIAL_TOKENS = (
    SONNET_OPENING_TOKEN,
    SONNET_CONTINUATION_TOKEN,
)
SONNET_LINE_COUNT = 14
IGNORE_INDEX = -100


@dataclass(frozen=True)
class SonnetContinuationExample:
    """One document-level example for first-line sonnet continuation."""

    poem_id: str
    split: str
    opening_line: str
    continuation_text: str


@dataclass(frozen=True)
class EncodedSonnetContinuationExample:
    """One task example with labels masked through the completion boundary."""

    example: SonnetContinuationExample
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    continuation_target_start: int


def split_sonnet_for_continuation(poem_text: str) -> tuple[str, str]:
    """Return the first line and the remaining 13 lines of one strict sonnet."""
    lines = poem_text.splitlines()
    if len(lines) != SONNET_LINE_COUNT:
        raise ValueError(
            f"sonnet must contain exactly {SONNET_LINE_COUNT} lines, got {len(lines)}"
        )
    if any(not line.strip() for line in lines):
        raise ValueError("sonnet must not contain empty or whitespace-only lines")

    return lines[0], "\n".join(lines[1:])


def build_task_prompt(opening_line: str) -> str:
    """Build the exact model prefix used for a user-supplied first line."""
    _validate_opening_line(opening_line)
    return (
        f"{SONNET_OPENING_TOKEN}{opening_line}\n"
        f"{SONNET_CONTINUATION_TOKEN}"
    )


def format_sonnet_continuation_example(
    opening_line: str,
    continuation_text: str,
) -> str:
    """Format one supervised example with an explicit completion boundary.

    The continuation begins immediately after its protected marker. This keeps
    tokenization of the prompt prefix stable: BPE cannot merge a prompt token
    with the first token of the predicted continuation.
    """
    _validate_opening_line(opening_line)
    _validate_continuation_text(continuation_text)
    return (
        f"{build_task_prompt(opening_line)}{continuation_text}\n"
        f"{PRETRAINING_DOCUMENT_SEPARATOR}"
    )


def build_sonnet_continuation_examples(
    manifest_path: Path,
    repo_root: Path,
    dataset: str,
    split: str,
) -> list[SonnetContinuationExample]:
    """Load one manifest split as opening-line continuation examples."""
    rows = read_manifest_rows(manifest_path)
    validate_manifest_rows(rows, dataset)
    selected_rows = select_manifest_rows(rows=rows, dataset=dataset, split=split)
    if not selected_rows:
        raise ValueError(f"no sonnets selected for dataset={dataset}, split={split}")

    examples = []
    for row in selected_rows:
        opening_line, continuation_text = split_sonnet_for_continuation(
            load_poem_text(row, repo_root=repo_root)
        )
        examples.append(
            SonnetContinuationExample(
                poem_id=row["poem_id"],
                split=split,
                opening_line=opening_line,
                continuation_text=continuation_text,
            )
        )

    return examples


def extend_tokenizer_for_task_format(
    tokenizer: BytePairEncodingTokenizer,
) -> tuple[BytePairEncodingTokenizer, list[str]]:
    """Add the fixed task-format control tokens to a sonnet tokenizer."""
    return extend_tokenizer_for_special_tokens(
        tokenizer=tokenizer,
        special_tokens=TASK_FORMAT_SPECIAL_TOKENS,
    )


def encode_sonnet_continuation_example(
    example: SonnetContinuationExample,
    tokenizer: BytePairEncodingTokenizer,
) -> EncodedSonnetContinuationExample:
    """Encode one task example and mask labels before its continuation body."""
    _require_task_format_tokens(tokenizer)
    prompt = build_task_prompt(example.opening_line)
    formatted_text = format_sonnet_continuation_example(
        opening_line=example.opening_line,
        continuation_text=example.continuation_text,
    )
    prompt_ids = encode_text_by_pretoken(prompt, tokenizer)
    token_ids = encode_text_by_pretoken(formatted_text, tokenizer)

    if token_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("task prompt must remain a token-prefix of its example")
    if len(token_ids) <= len(prompt_ids):
        raise ValueError("task example must contain continuation tokens")

    continuation_target_start = len(prompt_ids) - 1
    input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
    target_ids = torch.tensor(
        [
            *([IGNORE_INDEX] * continuation_target_start),
            *token_ids[len(prompt_ids):],
        ],
        dtype=torch.long,
    )

    if input_ids.shape != target_ids.shape:
        raise AssertionError("task-format inputs and targets must have matching shapes")

    return EncodedSonnetContinuationExample(
        example=example,
        input_ids=input_ids,
        target_ids=target_ids,
        continuation_target_start=continuation_target_start,
    )


def load_encoded_sonnet_continuation_splits(
    manifest_path: Path,
    repo_root: Path,
    dataset: str,
    tokenizer_path: Path,
) -> tuple[
    list[EncodedSonnetContinuationExample],
    list[EncodedSonnetContinuationExample],
    list[EncodedSonnetContinuationExample],
    BytePairEncodingTokenizer,
]:
    """Load all document-disjoint splits for masked task-format training."""
    split_examples = {
        split: build_sonnet_continuation_examples(
            manifest_path=manifest_path,
            repo_root=repo_root,
            dataset=dataset,
            split=split,
        )
        for split in ("train", "validation", "test")
    }
    tokenizer = BytePairEncodingTokenizer.load(tokenizer_path)
    tokenizer, _ = extend_tokenizer_for_character_coverage(
        tokenizer=tokenizer,
        texts=[
            text
            for examples in split_examples.values()
            for example in examples
            for text in (example.opening_line, example.continuation_text)
        ],
    )
    tokenizer, _ = extend_tokenizer_for_task_format(tokenizer)

    encoded_splits = tuple(
        [
            encode_sonnet_continuation_example(example, tokenizer)
            for example in split_examples[split]
        ]
        for split in ("train", "validation", "test")
    )

    return (*encoded_splits, tokenizer)


def _validate_opening_line(opening_line: str) -> None:
    if not opening_line.strip():
        raise ValueError("opening_line must not be empty")
    if "\n" in opening_line or "\r" in opening_line:
        raise ValueError("opening_line must contain exactly one line")


def _validate_continuation_text(continuation_text: str) -> None:
    lines = continuation_text.splitlines()
    expected_line_count = SONNET_LINE_COUNT - 1
    if len(lines) != expected_line_count:
        raise ValueError(
            "continuation_text must contain exactly "
            f"{expected_line_count} lines, got {len(lines)}"
        )
    if any(not line.strip() for line in lines):
        raise ValueError(
            "continuation_text must not contain empty or whitespace-only lines"
        )


def _require_task_format_tokens(tokenizer: BytePairEncodingTokenizer) -> None:
    missing_tokens = [
        token
        for token in TASK_FORMAT_SPECIAL_TOKENS
        if token not in tokenizer.special_tokens
    ]
    if missing_tokens:
        raise ValueError(
            "tokenizer is missing task-format special tokens: "
            + ", ".join(missing_tokens)
        )
