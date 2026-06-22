from pathlib import Path

import pytest

from sonnet_corpus.bpe import (
    BytePairEncodingTokenizer,
    apply_merges,
    build_base_vocabulary,
    choose_best_pair,
    get_pair_counts,
    merge_token_pair,
    split_text_into_initial_tokens,
    train_bpe_tokenizer,
)
from sonnet_corpus.dataset_text import load_split_text_stream


def test_split_text_into_initial_tokens_keeps_special_token_atomic():
    tokens = split_text_into_initial_tokens(
        text="Amor\n<|poem_end|>\n",
        special_tokens=["<|poem_end|>"],
    )

    assert tokens == ["A", "m", "o", "r", "\n", "<|poem_end|>", "\n"]


def test_get_pair_counts_counts_adjacent_pairs():
    pair_counts = get_pair_counts([
        ["a", "b", "a"],
        ["a", "b"],
    ])

    assert pair_counts[("a", "b")] == 2
    assert pair_counts[("b", "a")] == 1


def test_get_pair_counts_does_not_cross_special_tokens():
    pair_counts = get_pair_counts(
        [["a", "<|poem_end|>", "b"]],
        special_tokens=["<|poem_end|>"],
    )

    assert pair_counts == {}


def test_merge_token_pair_replaces_non_overlapping_pairs():
    merged = merge_token_pair(
        token_sequence=["a", "a", "a"],
        pair=("a", "a"),
        merged_token="aa",
    )

    assert merged == ["aa", "a"]


def test_apply_merges_uses_merge_order():
    tokens = apply_merges(
        token_sequence=["a", "b", "c"],
        merges=[("a", "b"), ("ab", "c")],
    )

    assert tokens == ["abc"]


def test_build_base_vocabulary_keeps_special_tokens_first_and_chars_sorted():
    vocabulary = build_base_vocabulary(
        texts=["ba", "c"],
        special_tokens=["<|poem_end|>"],
    )

    assert vocabulary == ["<|poem_end|>", "a", "b", "c"]


def test_choose_best_pair_breaks_ties_deterministically():
    pair = choose_best_pair({
        ("b", "a"): 2,
        ("a", "b"): 2,
    })

    assert pair == ("a", "b")


def test_train_bpe_tokenizer_round_trips_text():
    text = "abab"

    tokenizer = train_bpe_tokenizer(
        texts=[text],
        vocab_size=5,
    )

    token_ids = tokenizer.encode(text)

    assert tokenizer.decode(token_ids) == text
    assert tokenizer.merges == [("a", "b"), ("ab", "ab")]


def test_train_bpe_tokenizer_is_deterministic():
    first = train_bpe_tokenizer(
        texts=["abab", "baba"],
        vocab_size=6,
    )
    second = train_bpe_tokenizer(
        texts=["abab", "baba"],
        vocab_size=6,
    )

    assert first.token_to_id == second.token_to_id
    assert first.merges == second.merges


def test_bpe_tokenizer_preserves_accents_and_newlines():
    text = "più\nïo"

    tokenizer = train_bpe_tokenizer(
        texts=[text],
        vocab_size=10,
    )

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_bpe_tokenizer_preserves_special_tokens():
    text = "Amor\n<|poem_end|>\n"

    tokenizer = train_bpe_tokenizer(
        texts=[text],
        vocab_size=20,
        special_tokens=["<|poem_end|>"],
    )

    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)

    assert decoded == text
    assert "<|poem_end|>" in tokenizer.token_to_id
    assert tokenizer.token_to_id["<|poem_end|>"] in token_ids


def test_bpe_tokenizer_tokenize_texts_encodes_multiple_texts():
    tokenizer = train_bpe_tokenizer(
        texts=["amor", "donna"],
        vocab_size=10,
    )

    encoded_texts = tokenizer.tokenize_texts(["amor", "donna"])

    assert len(encoded_texts) == 2
    assert tokenizer.decode(encoded_texts[0]) == "amor"
    assert tokenizer.decode(encoded_texts[1]) == "donna"


def test_bpe_tokenizer_to_dict_and_from_dict_round_trip():
    tokenizer = train_bpe_tokenizer(
        texts=["abab"],
        vocab_size=5,
        special_tokens=["<|poem_end|>"],
    )

    restored = BytePairEncodingTokenizer.from_dict(tokenizer.to_dict())

    assert restored.token_to_id == tokenizer.token_to_id
    assert restored.merges == tokenizer.merges
    assert restored.special_tokens == tokenizer.special_tokens
    assert restored.decode(restored.encode("abab")) == "abab"


def test_bpe_tokenizer_save_and_load_round_trip(tmp_path):
    tokenizer = train_bpe_tokenizer(
        texts=["più\npiù"],
        vocab_size=8,
        special_tokens=["<|poem_end|>"],
    )
    tokenizer_path = tmp_path / "tokenizer.json"

    tokenizer.save(tokenizer_path)
    restored = BytePairEncodingTokenizer.load(tokenizer_path)

    assert tokenizer_path.is_file()
    assert restored.token_to_id == tokenizer.token_to_id
    assert restored.merges == tokenizer.merges
    assert restored.decode(restored.encode("più\npiù")) == "più\npiù"


def test_bpe_tokenizer_from_dict_rejects_unknown_type():
    with pytest.raises(ValueError, match="unicode_bpe"):
        BytePairEncodingTokenizer.from_dict({
            "type": "other",
            "token_to_id": {},
            "merges": [],
            "special_tokens": [],
        })


def test_train_bpe_tokenizer_can_use_base_texts_for_character_coverage():
    tokenizer = train_bpe_tokenizer(
        texts=["amor"],
        base_texts=["amor", "più"],
        vocab_size=8,
    )

    assert tokenizer.decode(tokenizer.encode("più")) == "più"


def test_train_bpe_tokenizer_rejects_too_small_vocab_size():
    with pytest.raises(ValueError, match="base vocabulary"):
        train_bpe_tokenizer(
            texts=["abc"],
            vocab_size=2,
        )


def test_bpe_tokenizer_rejects_unknown_character():
    tokenizer = train_bpe_tokenizer(
        texts=["amor"],
        vocab_size=4,
    )

    with pytest.raises(KeyError):
        tokenizer.encode("amore")


def test_bpe_tokenizer_rejects_duplicate_token_ids():
    with pytest.raises(ValueError, match="duplicate IDs"):
        BytePairEncodingTokenizer(
            token_to_id={
                "a": 0,
                "b": 0,
            },
            merges=[],
            special_tokens=[],
        )


def test_real_corpus_bpe_tokenizer_round_trips_validation_text():
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "data" / "metadata" / "poems_manifest.csv"
    train_text = load_split_text_stream(
        manifest_path=manifest_path,
        repo_root=repo_root,
        dataset="expanded_with_petrarch",
        split="train",
    )
    validation_text = load_split_text_stream(
        manifest_path=manifest_path,
        repo_root=repo_root,
        dataset="expanded_with_petrarch",
        split="validation",
    )
    test_text = load_split_text_stream(
        manifest_path=manifest_path,
        repo_root=repo_root,
        dataset="expanded_with_petrarch",
        split="test",
    )

    tokenizer = train_bpe_tokenizer(
        texts=[train_text],
        base_texts=[
            train_text,
            validation_text,
            test_text,
        ],
        vocab_size=200,
        special_tokens=["<|poem_end|>"],
    )
    validation_excerpt = validation_text[:1000]

    assert tokenizer.decode(tokenizer.encode(validation_excerpt)) == validation_excerpt
    assert len(tokenizer.merges) > 0
