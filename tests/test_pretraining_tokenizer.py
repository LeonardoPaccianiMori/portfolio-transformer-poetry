import json
from pathlib import Path

import pytest

from sonnet_corpus.pretraining_tokenizer import (
    PretrainingTokenizerConfig,
    count_bpe_tokens_by_pretoken,
    inspect_build_report_boundaries,
    train_pretraining_bpe_tokenizer,
    train_weighted_pretoken_bpe_tokenizer,
)


def test_weighted_pretoken_bpe_tokenizer_round_trips_text():
    text = "amor amor\nvirtute antica\n"

    tokenizer = train_weighted_pretoken_bpe_tokenizer(
        training_text=text,
        base_text=text,
        vocab_size=30,
        special_tokens=["<|endoftext|>"],
    )

    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.token_to_id["<|endoftext|>"] == 0
    assert len(tokenizer.merges) > 0


def test_count_bpe_tokens_by_pretoken_matches_regular_encoding():
    text = "amor amor\nvirtute\n"
    tokenizer = train_weighted_pretoken_bpe_tokenizer(
        training_text=text,
        base_text=text,
        vocab_size=24,
        special_tokens=["<|endoftext|>"],
    )

    token_count = count_bpe_tokens_by_pretoken(text, tokenizer)

    assert token_count == len(tokenizer.encode(text))


def test_train_pretraining_bpe_tokenizer_writes_tokenizer_and_report(tmp_path: Path):
    corpus_path = tmp_path / "corpus.txt"
    tokenizer_path = tmp_path / "tokenizer.json"
    report_path = tmp_path / "report.json"
    build_report_path = tmp_path / "build_report.json"
    corpus_text = (
        "Questa e una prosa antica con parole ripetute.\n"
        "Questa prosa serve per addestrare un tokenizzatore.\n"
        "Virtute, amore, memoria, cronica, novella.\n"
    ) * 20
    corpus_path.write_text(corpus_text, encoding="utf-8")
    build_report_path.write_text(
        json.dumps({
            "sources": [
                {
                    "source_id": "pg_test",
                    "first_characters": "Indice generale e nota.",
                    "last_characters": "Fine del testo.",
                }
            ]
        }),
        encoding="utf-8",
    )

    report = train_pretraining_bpe_tokenizer(
        PretrainingTokenizerConfig(
            corpus_path=corpus_path,
            tokenizer_path=tokenizer_path,
            report_path=report_path,
            build_report_path=build_report_path,
            vocab_size=80,
            special_tokens=("<|endoftext|>",),
            training_character_limit=500,
        )
    )

    assert tokenizer_path.is_file()
    assert report_path.is_file()
    assert report["actual_vocab_size"] == 80
    assert report["token_count"] > 0
    assert report["characters_per_token"] > 1
    assert report["round_trip_samples"][0]["round_trip_ok"] is True
    assert report["boundary_warnings"][0]["source_id"] == "pg_test"

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["tokenizer_path"].endswith("tokenizer.json")
    assert saved["special_tokens"] == ["<|endoftext|>"]


def test_train_pretraining_bpe_tokenizer_rejects_missing_corpus(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="corpus file does not exist"):
        train_pretraining_bpe_tokenizer(
            PretrainingTokenizerConfig(
                corpus_path=tmp_path / "missing.txt",
                tokenizer_path=tmp_path / "tokenizer.json",
                report_path=tmp_path / "report.json",
                build_report_path=tmp_path / "build_report.json",
                vocab_size=20,
            )
        )


def test_train_pretraining_bpe_tokenizer_rejects_empty_corpus(tmp_path: Path):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(" \n\t", encoding="utf-8")

    with pytest.raises(ValueError, match="corpus file is empty"):
        train_pretraining_bpe_tokenizer(
            PretrainingTokenizerConfig(
                corpus_path=corpus_path,
                tokenizer_path=tmp_path / "tokenizer.json",
                report_path=tmp_path / "report.json",
                build_report_path=tmp_path / "build_report.json",
                vocab_size=20,
            )
        )


def test_inspect_build_report_boundaries_returns_empty_list_when_report_is_missing(
    tmp_path: Path,
):
    warnings = inspect_build_report_boundaries(tmp_path / "missing.json")

    assert warnings == []
