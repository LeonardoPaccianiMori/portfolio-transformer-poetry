import json
from pathlib import Path

import pytest

from sonnet_corpus.bpe import BytePairEncodingTokenizer
from sonnet_corpus.gutenberg import FetchedGutenbergText
from sonnet_corpus.pretraining_manifest import (
    PretrainingSourceRow,
    write_pretraining_manifest,
)
from sonnet_corpus.pretraining_probe import (
    count_whitespace_words,
    probe_gutenberg_sources,
    select_gutenberg_probe_rows,
)


def make_row(**overrides):
    values = {
        "source_id": "pg_sidrac_44549",
        "title": "Il libro di Sidrach",
        "author": "Sidrac / anonymous tradition",
        "source_archive": "Project Gutenberg",
        "source_collection": "Project Gutenberg Italian",
        "landing_page_url": "https://www.gutenberg.org/ebooks/44549",
        "download_url": "",
        "ebook_id": "44549",
        "language": "Italian",
        "period_bucket": "tier_a_b_borderline",
        "approx_date": "XIV secolo",
        "genre": "encyclopedic / philosophical prose",
        "text_kind": "prose",
        "inclusion_status": "include_probe",
        "public_domain_status": "Public domain in the USA",
        "license_notes": "Project Gutenberg public-domain status on landing page.",
        "edition_notes": "",
        "source_release_date": "",
        "source_last_updated": "",
        "expected_clean_text_path": "data/processed/pretraining/pg_sidrac_44549.txt",
        "token_count_report_path": "",
        "split": "",
        "boilerplate_strategy": "strip Project Gutenberg header and footer",
        "mixed_text_strategy": "",
        "cleaning_notes": "Preserve source spelling.",
        "audit_notes": "",
    }
    values.update(overrides)
    return PretrainingSourceRow(**values)


def test_select_gutenberg_probe_rows_skips_conditional_and_non_gutenberg_rows():
    rows = [
        make_row(source_id="active"),
        make_row(
            source_id="mixed",
            text_kind="mixed",
            inclusion_status="conditional_extract_prose",
            mixed_text_strategy="extract prose only",
        ),
        make_row(
            source_id="liberliber",
            source_archive="Liber Liber",
            landing_page_url="https://liberliber.it/example",
            ebook_id="",
            public_domain_status="underlying work out of copyright",
            license_notes="Liber Liber license layer recorded separately.",
            edition_notes="Liber Liber license layer applies to edition material.",
        ),
    ]

    selected = select_gutenberg_probe_rows(rows)

    assert [row.source_id for row in selected] == ["active"]


def test_select_gutenberg_probe_rows_filters_to_requested_active_sources():
    rows = [make_row(source_id="one"), make_row(source_id="two")]

    selected = select_gutenberg_probe_rows(rows, source_ids={"two"})

    assert [row.source_id for row in selected] == ["two"]

    with pytest.raises(ValueError, match="not active prose rows"):
        select_gutenberg_probe_rows(rows, source_ids={"missing"})


def test_count_whitespace_words_counts_non_empty_units():
    assert count_whitespace_words(" uno  due\ntre ") == 3


def test_probe_gutenberg_sources_writes_report(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    report_path = tmp_path / "report.json"
    write_pretraining_manifest(
        [
            make_row(source_id="pg_sidrac_44549", ebook_id="44549"),
            make_row(
                source_id="pg_vita_nuova_71218",
                title="La vita nuova",
                ebook_id="71218",
                text_kind="mixed",
                inclusion_status="conditional_extract_prose",
                mixed_text_strategy="extract prose only",
            ),
        ],
        manifest_path,
    )

    def fake_fetch_text(ebook_id, session=None):
        text = "\n".join(
            [
                "header",
                "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***",
                f"Corpo del libro {ebook_id}.",
                "Seconda riga.",
                "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***",
                "footer",
            ]
        )
        return FetchedGutenbergText(
            ebook_id=ebook_id,
            url=f"https://example.test/{ebook_id}.txt",
            text=text,
        )

    report = probe_gutenberg_sources(
        manifest_path=manifest_path,
        report_path=report_path,
        request_delay=0,
        fetch_text=fake_fetch_text,
    )

    assert report["selected_rows"] == 1
    assert report["skipped_rows"] == 1
    assert report["total_cleaned_words"] == 6
    assert report["total_bpe_tokens"] is None
    assert report["bpe_tokenized_rows"] == 0
    assert report["bpe_tokenization_error_rows"] == 0

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["results"][0]["source_id"] == "pg_sidrac_44549"
    assert saved["results"][0]["status"] == "ok"
    assert saved["results"][0]["fetched_url"] == "https://example.test/44549.txt"
    assert saved["results"][0]["cleaned_word_count"] == 6


def test_probe_gutenberg_sources_reports_per_source_progress(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    report_path = tmp_path / "report.json"
    write_pretraining_manifest([make_row()], manifest_path)
    progress_messages = []

    def fake_fetch_text(ebook_id, session=None):
        return FetchedGutenbergText(
            ebook_id=ebook_id,
            url="https://example.test/book.txt",
            text="text",
        )

    probe_gutenberg_sources(
        manifest_path=manifest_path,
        report_path=report_path,
        request_delay=0,
        fetch_text=fake_fetch_text,
        progress=progress_messages.append,
    )

    assert progress_messages == ["probing source 1/1: pg_sidrac_44549"]


def test_probe_gutenberg_sources_records_fetch_errors(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    report_path = tmp_path / "report.json"
    write_pretraining_manifest([make_row()], manifest_path)

    def fake_fetch_text(ebook_id, session=None):
        raise ConnectionError("network unavailable")

    report = probe_gutenberg_sources(
        manifest_path=manifest_path,
        report_path=report_path,
        request_delay=0,
        fetch_text=fake_fetch_text,
    )

    result = report["results"][0]
    assert result["status"] == "error"
    assert result["error"] == "network unavailable"
    assert result["cleaned_character_count"] == 0

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["results"][0]["status"] == "error"


def test_probe_records_incompatible_tokenizer_without_losing_text_counts(
    tmp_path: Path,
):
    manifest_path = tmp_path / "manifest.csv"
    report_path = tmp_path / "report.json"
    tokenizer_path = tmp_path / "tokenizer.json"
    write_pretraining_manifest([make_row()], manifest_path)
    BytePairEncodingTokenizer(
        token_to_id={"a": 0},
        merges=[],
        special_tokens=[],
    ).save(tokenizer_path)

    def fake_fetch_text(ebook_id, session=None):
        return FetchedGutenbergText(
            ebook_id=ebook_id,
            url="https://example.test/book.txt",
            text="ab",
        )

    report = probe_gutenberg_sources(
        manifest_path=manifest_path,
        report_path=report_path,
        tokenizer_path=tokenizer_path,
        request_delay=0,
        fetch_text=fake_fetch_text,
    )

    result = report["results"][0]
    assert result["status"] == "ok"
    assert result["cleaned_character_count"] == 3
    assert result["bpe_token_count"] is None
    assert result["bpe_tokenization_error"] == (
        "tokenizer vocabulary does not contain 'b'"
    )
    assert report["bpe_tokenized_rows"] == 0
    assert report["bpe_tokenization_error_rows"] == 1
    assert report["total_bpe_tokens"] is None
