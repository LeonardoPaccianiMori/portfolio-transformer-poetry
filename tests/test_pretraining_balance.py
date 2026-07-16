import json
from pathlib import Path

import pytest

from sonnet_corpus.pretraining_balance import (
    PretrainingBalanceConfig,
    audit_pretraining_corpus_balance,
)
from sonnet_corpus.pretraining_manifest import PretrainingSourceRow, write_pretraining_manifest


def make_row(**overrides) -> PretrainingSourceRow:
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


def make_config(tmp_path: Path) -> PretrainingBalanceConfig:
    return PretrainingBalanceConfig(
        manifest_path=tmp_path / "manifest.csv",
        processed_dir=tmp_path / "processed",
        json_report_path=tmp_path / "reports" / "balance.json",
        markdown_report_path=tmp_path / "reports" / "balance.md",
        max_source_share=0.40,
        max_author_share=0.60,
    )


def write_sources(processed_dir: Path, texts: dict[str, str]) -> None:
    source_dir = processed_dir / "sources"
    source_dir.mkdir(parents=True)
    for source_id, text in texts.items():
        (source_dir / f"{source_id}.txt").write_text(text, encoding="utf-8")


def test_audit_reports_source_author_shares_and_cap_violations(tmp_path: Path):
    config = make_config(tmp_path)
    write_pretraining_manifest(
        [
            make_row(source_id="source_a", author="Author A"),
            make_row(source_id="source_b", title="Work B", author="Author A", ebook_id="2"),
            make_row(source_id="source_c", title="Work C", author="Author B", ebook_id="3"),
        ],
        config.manifest_path,
    )
    write_sources(
        config.processed_dir,
        {
            "source_a": "aaaaa",
            "source_b": "bb",
            "source_c": "ccc",
        },
    )

    report = audit_pretraining_corpus_balance(config)

    assert report.selected_source_count == 3
    assert report.total_cleaned_characters == 10
    assert report.total_cleaned_words == 3
    assert [entry.name for entry in report.source_entries] == [
        "source_a",
        "source_c",
        "source_b",
    ]
    assert report.source_character_cap_violations == ["source_a"]
    assert report.source_word_cap_violations == []
    assert report.author_character_cap_violations == ["Author A"]
    assert report.author_word_cap_violations == ["Author A"]

    saved = json.loads(config.json_report_path.read_text(encoding="utf-8"))
    assert saved["source_entries"][0]["character_share"] == 0.5
    markdown = config.markdown_report_path.read_text(encoding="utf-8")
    assert "- Work character cap: source_a" in markdown
    assert "- Author word cap: Author A" in markdown


def test_audit_rejects_missing_or_unexpected_processed_source_files(tmp_path: Path):
    config = make_config(tmp_path)
    write_pretraining_manifest([make_row()], config.manifest_path)
    write_sources(config.processed_dir, {"unexpected": "text"})

    with pytest.raises(ValueError, match="missing=source_a; unexpected=unexpected"):
        audit_pretraining_corpus_balance(config)


def test_audit_rejects_invalid_share_thresholds(tmp_path: Path):
    config = make_config(tmp_path)
    invalid_config = PretrainingBalanceConfig(
        manifest_path=config.manifest_path,
        processed_dir=config.processed_dir,
        json_report_path=config.json_report_path,
        markdown_report_path=config.markdown_report_path,
        max_source_share=0,
    )

    with pytest.raises(ValueError, match="max_source_share"):
        audit_pretraining_corpus_balance(invalid_config)
