import json
from pathlib import Path

import pytest
import torch

from sonnet_corpus.pretraining_dataset import (
    PretrainingDatasetConfig,
    build_pretraining_token_dataset,
    list_processed_source_paths,
    load_pretraining_token_splits,
    split_source_token_ids,
    validate_processed_source_manifest,
)
from sonnet_corpus.pretraining_manifest import PretrainingSourceRow, write_pretraining_manifest
from sonnet_corpus.pretraining_tokenizer import train_weighted_pretoken_bpe_tokenizer


def write_tiny_tokenizer(path: Path) -> None:
    text = (
        "amor antico memoria cronica\n"
        "virtute novella lingua storia\n"
        "donna cortesia ragione tempo\n"
    )
    tokenizer = train_weighted_pretoken_bpe_tokenizer(
        training_text=text,
        base_text=text,
        vocab_size=60,
        special_tokens=["<|endoftext|>"],
    )
    tokenizer.save(path)


def make_pretraining_row(**overrides) -> PretrainingSourceRow:
    values = {
        "source_id": "a_source",
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


def test_list_processed_source_paths_returns_sorted_text_files(tmp_path: Path):
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "b.txt").write_text("b", encoding="utf-8")
    (sources_dir / "a.txt").write_text("a", encoding="utf-8")
    (sources_dir / "notes.md").write_text("ignored", encoding="utf-8")

    paths = list_processed_source_paths(sources_dir)

    assert [path.name for path in paths] == ["a.txt", "b.txt"]


def test_list_processed_source_paths_rejects_missing_directory(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        list_processed_source_paths(tmp_path / "missing")


def test_list_processed_source_paths_rejects_directory_without_text_files(
    tmp_path: Path,
):
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()

    with pytest.raises(ValueError, match="no .txt files"):
        list_processed_source_paths(sources_dir)


def test_split_source_token_ids_uses_final_fraction_for_validation():
    train_ids, validation_ids = split_source_token_ids(
        list(range(10)),
        validation_fraction=0.2,
        source_id="source",
    )

    assert train_ids == list(range(8))
    assert validation_ids == [8, 9]


def test_split_source_token_ids_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="validation_fraction"):
        split_source_token_ids(
            [1, 2, 3],
            validation_fraction=1,
            source_id="source",
        )


def test_validate_processed_source_manifest_rejects_stale_source_files(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    write_pretraining_manifest([make_pretraining_row()], manifest_path)
    source_paths = [tmp_path / "stale_source.txt"]

    with pytest.raises(ValueError, match="missing=a_source; unexpected=stale_source"):
        validate_processed_source_manifest(source_paths, manifest_path)


def test_build_pretraining_token_dataset_writes_tensors_and_report(tmp_path: Path):
    sources_dir = tmp_path / "sources"
    tokenizer_path = tmp_path / "tokenizer.json"
    output_dir = tmp_path / "encoded"
    report_path = output_dir / "report.json"
    sources_dir.mkdir()
    (sources_dir / "a_source.txt").write_text(
        "amor antico memoria cronica\n" * 4,
        encoding="utf-8",
    )
    (sources_dir / "b_source.txt").write_text(
        "virtute novella lingua storia\n" * 4,
        encoding="utf-8",
    )
    write_tiny_tokenizer(tokenizer_path)

    report = build_pretraining_token_dataset(
        PretrainingDatasetConfig(
            processed_sources_dir=sources_dir,
            tokenizer_path=tokenizer_path,
            output_dir=output_dir,
            report_path=report_path,
            validation_fraction=0.25,
            document_separator="<|endoftext|>",
        )
    )

    train_path = output_dir / "bpe_8000_train.pt"
    validation_path = output_dir / "bpe_8000_validation.pt"
    assert train_path.is_file()
    assert validation_path.is_file()
    assert report_path.is_file()

    train_tokens, validation_tokens = load_pretraining_token_splits(output_dir)
    assert train_tokens.ndim == 1
    assert validation_tokens.ndim == 1
    assert train_tokens.dtype == torch.long
    assert validation_tokens.dtype == torch.long
    assert train_tokens.numel() == report["train_tokens"]
    assert validation_tokens.numel() == report["validation_tokens"]

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["source_count"] == 2
    assert saved_report["split_policy"] == "final_token_fraction_per_source"
    assert saved_report["document_separator"] == "<|endoftext|>"
    assert saved_report["sources"][0]["source_id"] == "a_source"
    assert saved_report["sources"][1]["source_id"] == "b_source"


def test_build_pretraining_token_dataset_rejects_separator_without_single_token(
    tmp_path: Path,
):
    sources_dir = tmp_path / "sources"
    tokenizer_path = tmp_path / "tokenizer.json"
    sources_dir.mkdir()
    (sources_dir / "source.txt").write_text(
        "amor antico memoria cronica\n",
        encoding="utf-8",
    )
    write_tiny_tokenizer(tokenizer_path)

    with pytest.raises(ValueError, match="exactly one token"):
        build_pretraining_token_dataset(
            PretrainingDatasetConfig(
                processed_sources_dir=sources_dir,
                tokenizer_path=tokenizer_path,
                output_dir=tmp_path / "encoded",
                report_path=tmp_path / "encoded" / "report.json",
                validation_fraction=0.25,
                document_separator="amor antico",
            )
        )


def test_load_pretraining_token_splits_rejects_wrong_dtype(tmp_path: Path):
    output_dir = tmp_path / "encoded"
    output_dir.mkdir()
    torch.save(torch.tensor([1.0, 2.0]), output_dir / "bpe_8000_train.pt")
    torch.save(
        torch.tensor([1, 2], dtype=torch.long),
        output_dir / "bpe_8000_validation.pt",
    )

    with pytest.raises(ValueError, match="torch.long"):
        load_pretraining_token_splits(output_dir)
