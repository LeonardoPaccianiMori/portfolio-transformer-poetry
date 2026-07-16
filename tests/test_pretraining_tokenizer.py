import json
from pathlib import Path

import pytest

from sonnet_corpus.pretraining_tokenizer import (
    PretrainingTokenizerConfig,
    count_bpe_tokens_by_pretoken,
    encode_text_by_pretoken,
    inspect_build_report_boundaries,
    train_pretraining_bpe_tokenizer,
    train_weighted_pretoken_bpe_tokenizer,
)
from sonnet_corpus.pretraining_manifest import (
    PretrainingSourceRow,
    write_pretraining_manifest,
)


def make_pretraining_row(**overrides) -> PretrainingSourceRow:
    values = {
        "source_id": "source_a",
        "title": "Work A",
        "author": "Author A",
        "source_archive": "Project Gutenberg",
        "source_collection": "Project Gutenberg Italian",
        "landing_page_url": "https://example.test/a",
        "download_url": "",
        "ebook_id": "1",
        "language": "Italian",
        "period_bucket": "tier_a_pre_1375",
        "approx_date": "XIV secolo",
        "genre": "prose",
        "text_kind": "prose",
        "inclusion_status": "include_probe",
        "public_domain_status": "public domain",
        "license_notes": "test",
        "edition_notes": "",
        "source_release_date": "",
        "source_last_updated": "",
        "expected_clean_text_path": "",
        "token_count_report_path": "",
        "split": "",
        "boilerplate_strategy": "strip Project Gutenberg header and footer",
        "mixed_text_strategy": "",
        "cleaning_notes": "",
        "audit_notes": "",
    }
    values.update(overrides)
    return PretrainingSourceRow(**values)


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


def test_encode_text_by_pretoken_matches_regular_encoding():
    text = "amor amor\nvirtute\n"
    tokenizer = train_weighted_pretoken_bpe_tokenizer(
        training_text=text,
        base_text=text,
        vocab_size=24,
        special_tokens=["<|endoftext|>"],
    )

    token_ids = encode_text_by_pretoken(text, tokenizer)

    assert token_ids == tokenizer.encode(text)
    assert tokenizer.decode(token_ids) == text


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


def test_train_pretraining_bpe_tokenizer_stratifies_a_source_sample(tmp_path: Path):
    source_dir = tmp_path / "processed" / "sources"
    source_dir.mkdir(parents=True)
    source_a = ("amor virtute memoria ") * 20
    source_b = ("cronica novella istoria ") * 20
    (source_dir / "source_a.txt").write_text(source_a, encoding="utf-8")
    (source_dir / "source_b.txt").write_text(source_b, encoding="utf-8")
    corpus_path = tmp_path / "processed" / "corpus.txt"
    corpus_path.write_text(f"{source_a}\n{source_b}", encoding="utf-8")
    manifest_path = tmp_path / "manifest.csv"
    write_pretraining_manifest(
        [
            make_pretraining_row(),
            make_pretraining_row(
                source_id="source_b",
                title="Work B",
                author="Author B",
                landing_page_url="https://example.test/b",
                ebook_id="2",
            ),
        ],
        manifest_path,
    )
    messages: list[str] = []

    report = train_pretraining_bpe_tokenizer(
        PretrainingTokenizerConfig(
            corpus_path=corpus_path,
            tokenizer_path=tmp_path / "tokenizer.json",
            report_path=tmp_path / "report.json",
            build_report_path=tmp_path / "build_report.json",
            vocab_size=80,
            training_character_limit=120,
            manifest_path=manifest_path,
            source_dir=source_dir,
            minimum_source_characters=40,
            merge_progress_interval=10,
        ),
        progress=messages.append,
    )

    assert report["sampling_strategy"] == "stratified_sources"
    assert [item["source_id"] for item in report["sample_sources"]] == [
        "source_a",
        "source_b",
    ]
    assert sum(item["allocated_character_count"] for item in report["sample_sources"]) == 120
    assert all(item["sampled_character_count"] > 0 for item in report["sample_sources"])
    assert any(message.startswith("BPE merges:") for message in messages)
    assert any(message.startswith("token count cache entries:") for message in messages)


def test_train_pretraining_bpe_tokenizer_resumes_merge_checkpoints(tmp_path: Path):
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(("amor virtute memoria cronica novella\n") * 20, encoding="utf-8")
    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path = tmp_path / "training_state.json"
    config = PretrainingTokenizerConfig(
        corpus_path=corpus_path,
        tokenizer_path=tokenizer_path,
        report_path=tmp_path / "report.json",
        build_report_path=tmp_path / "build_report.json",
        vocab_size=40,
        training_character_limit=400,
        training_checkpoint_path=checkpoint_path,
        max_merges_per_run=2,
    )

    report = train_pretraining_bpe_tokenizer(config)
    assert report["status"] == "incomplete"
    assert checkpoint_path.is_file()
    assert not tokenizer_path.exists()

    while report["status"] == "incomplete":
        report = train_pretraining_bpe_tokenizer(config)

    assert report["status"] == "complete"
    assert report["actual_vocab_size"] == 40
    assert tokenizer_path.is_file()
    assert not checkpoint_path.exists()


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
