from pathlib import Path

import pytest
import torch

from sonnet_corpus.bpe import train_bpe_tokenizer
from sonnet_corpus.dataset_text import extend_tokenizer_for_special_tokens
from sonnet_corpus.pretraining_tokenizer import encode_text_by_pretoken
from sonnet_corpus.task_format import (
    IGNORE_INDEX,
    SONNET_CONTINUATION_TOKEN,
    SONNET_OPENING_TOKEN,
    SonnetContinuationExample,
    build_task_prompt,
    encode_sonnet_continuation_example,
    extend_tokenizer_for_task_format,
    format_sonnet_continuation_example,
    load_encoded_sonnet_continuation_splits,
    split_sonnet_for_continuation,
)


def make_sonnet(prefix: str = "line") -> str:
    return "\n".join(f"{prefix} {number}" for number in range(1, 15))


def write_manifest(path: Path) -> None:
    path.write_text(
        "poem_id,clean_text_path,include_in_expanded_with_petrarch,split_expanded_with_petrarch\n"
        "train_poem,data/processed/train.txt,True,train\n"
        "validation_poem,data/processed/validation.txt,True,validation\n"
        "test_poem,data/processed/test.txt,True,test\n",
        encoding="utf-8",
    )


def write_split_poems(repo_root: Path) -> None:
    processed_dir = repo_root / "data" / "processed"
    processed_dir.mkdir(parents=True)
    (processed_dir / "train.txt").write_text(make_sonnet("train"), encoding="utf-8")
    (processed_dir / "validation.txt").write_text(
        make_sonnet("validation"),
        encoding="utf-8",
    )
    (processed_dir / "test.txt").write_text(make_sonnet("test"), encoding="utf-8")


def test_split_sonnet_for_continuation_preserves_all_lines():
    opening_line, continuation_text = split_sonnet_for_continuation(make_sonnet())

    assert opening_line == "line 1"
    assert continuation_text == "\n".join(f"line {number}" for number in range(2, 15))


@pytest.mark.parametrize(
    "poem_text, message",
    [
        ("\n".join(f"line {number}" for number in range(1, 14)), "exactly 14"),
        ("\n".join(["line 1", "", *[f"line {number}" for number in range(3, 15)]]), "empty"),
    ],
)
def test_split_sonnet_for_continuation_rejects_invalid_lines(poem_text, message):
    with pytest.raises(ValueError, match=message):
        split_sonnet_for_continuation(poem_text)


def test_task_format_uses_explicit_prompt_and_completion_boundary():
    opening_line, continuation_text = split_sonnet_for_continuation(make_sonnet())

    assert build_task_prompt(opening_line) == (
        f"{SONNET_OPENING_TOKEN}line 1\n{SONNET_CONTINUATION_TOKEN}"
    )
    assert format_sonnet_continuation_example(opening_line, continuation_text) == (
        f"{SONNET_OPENING_TOKEN}line 1\n{SONNET_CONTINUATION_TOKEN}"
        + "\n".join(f"line {number}" for number in range(2, 15))
        + "\n<|endoftext|>"
    )


def test_special_token_extension_preserves_existing_vocabulary_and_merges():
    tokenizer = train_bpe_tokenizer(
        texts=["Amor\n<|endoftext|>"],
        vocab_size=12,
        special_tokens=["<|endoftext|>"],
    )
    original_ids = dict(tokenizer.token_to_id)
    original_merges = list(tokenizer.merges)

    extended, added_tokens = extend_tokenizer_for_special_tokens(
        tokenizer,
        [SONNET_OPENING_TOKEN, SONNET_CONTINUATION_TOKEN],
    )

    assert added_tokens == [SONNET_OPENING_TOKEN, SONNET_CONTINUATION_TOKEN]
    assert extended.merges == original_merges
    assert all(extended.token_to_id[token] == token_id for token, token_id in original_ids.items())
    assert extended.special_tokens == [
        "<|endoftext|>",
        SONNET_OPENING_TOKEN,
        SONNET_CONTINUATION_TOKEN,
    ]
    assert extended.decode(extended.encode(SONNET_OPENING_TOKEN)) == SONNET_OPENING_TOKEN


def test_encoded_task_example_masks_prompt_and_scores_only_continuation():
    opening_line, continuation_text = split_sonnet_for_continuation(make_sonnet())
    tokenizer = train_bpe_tokenizer(
        texts=[make_sonnet(), "<|endoftext|>"],
        vocab_size=32,
        special_tokens=["<|endoftext|>"],
    )
    tokenizer, added_tokens = extend_tokenizer_for_task_format(tokenizer)
    assert added_tokens == [SONNET_OPENING_TOKEN, SONNET_CONTINUATION_TOKEN]

    encoded = encode_sonnet_continuation_example(
        SonnetContinuationExample(
            poem_id="poem",
            split="train",
            opening_line=opening_line,
            continuation_text=continuation_text,
        ),
        tokenizer,
    )

    assert encoded.input_ids.dtype == torch.long
    assert encoded.target_ids.dtype == torch.long
    assert encoded.input_ids.shape == encoded.target_ids.shape
    assert torch.all(encoded.target_ids[:encoded.continuation_target_start] == IGNORE_INDEX)
    assert encoded.input_ids[:encoded.continuation_target_start + 1].tolist() == (
        encode_text_by_pretoken(build_task_prompt(opening_line), tokenizer)
    )
    assert tokenizer.decode(encoded.target_ids[encoded.continuation_target_start:].tolist()) == (
        "\n".join(f"line {number}" for number in range(2, 15))
        + "\n<|endoftext|>"
    )


def test_task_format_loader_keeps_manifest_splits_document_disjoint(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path)
    write_split_poems(tmp_path)
    tokenizer = train_bpe_tokenizer(
        texts=[make_sonnet("train"), "<|endoftext|>"],
        vocab_size=48,
        special_tokens=["<|endoftext|>"],
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    train_examples, validation_examples, test_examples, task_tokenizer = (
        load_encoded_sonnet_continuation_splits(
            manifest_path=manifest_path,
            repo_root=tmp_path,
            dataset="expanded_with_petrarch",
            tokenizer_path=tokenizer_path,
        )
    )

    assert [item.example.poem_id for item in train_examples] == ["train_poem"]
    assert [item.example.poem_id for item in validation_examples] == ["validation_poem"]
    assert [item.example.poem_id for item in test_examples] == ["test_poem"]
    assert task_tokenizer.special_tokens[-2:] == [
        SONNET_OPENING_TOKEN,
        SONNET_CONTINUATION_TOKEN,
    ]
