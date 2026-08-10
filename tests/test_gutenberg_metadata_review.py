import csv
import hashlib
import json
from pathlib import Path

import pytest

from sonnet_corpus.gutenberg import FetchedGutenbergText
from sonnet_corpus.gutenberg_metadata_review import (
    GutenbergMetadataReviewConfig,
    extract_gutenberg_date_evidence,
    resolve_gutenberg_metadata_row,
    run_gutenberg_metadata_review,
)
from sonnet_corpus.gutenberg_metadata_review_queue import QUEUE_FIELDS


def _row(
    ebook_id: str,
    status: str,
    *,
    title: str = "Libro di prova",
    role: str = "date_and_role_review",
    births: str = "1840",
) -> dict[str, str]:
    row = {field: "" for field in QUEUE_FIELDS}
    row.update(
        {
            "ebook_id": ebook_id,
            "title": title,
            "authors": "Autore, Prova",
            "author_birth_years": births,
            "author_death_years": "1910",
            "languages": "it",
            "subjects": "Italian literature",
            "preliminary_role": role,
            "period_bucket": "author_died_after_1900_review",
            "inventory_status": status,
            "landing_page_url": f"https://example.test/{ebook_id}",
            "plain_text_url": f"https://example.test/{ebook_id}.txt",
            "resolution_evidence_required": "test evidence",
        }
    )
    return row


def _text(body: str, *, original_publication: str = "") -> str:
    original = (
        f"Original publication: {original_publication}\n"
        if original_publication
        else ""
    )
    return (
        "The Project Gutenberg eBook of Test\n"
        "Release date: August 3, 2026 [eBook #1]\n"
        "Most recently updated: August 4, 2026\n"
        + original
        + "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        + body
        + "\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
    )


def _write_queue(path: Path, rows: list[dict[str, str]]) -> str:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_extract_date_evidence_ignores_release_date_and_reads_original_publication():
    evidence, release_years = extract_gutenberg_date_evidence(
        _text("ROMA\nEDITORE TEST\n1897\nTesto", original_publication="Roma, 1897"),
        title="Libro di prova",
    )

    assert release_years == [2026]
    assert any(
        item["kind"] == "gutenberg_original_publication"
        and item["year_start"] == 1897
        for item in evidence
    )
    assert all(item["year_start"] != 2026 for item in evidence)


def test_title_period_overrides_modern_edition_year_for_historical_work():
    row = _row(
        "2",
        "review_missing_period_evidence",
        title="Il libro della cucina del sec. XIV",
        births="",
    )
    result = resolve_gutenberg_metadata_row(
        row,
        _text("BOLOGNA\nPRESSO GAETANO ROMAGNOLI\n1863\nTesto italiano"),
    )

    assert result["resolution_status"] == "automatic_resolved"
    assert result["resolved_period_bucket"] == "origins_through_1800"
    assert result["resolved_role"] == "historical_general_candidate"
    assert result["selected_evidence_kind"] == "title_work_period"


@pytest.mark.parametrize(
    "title",
    [
        "La vita italiana nel Trecento: Conferenze tenute a Firenze nel 1891",
        "La pergamena distrutta: Romanzo del secolo XVI",
    ],
)
def test_subject_period_in_secondary_work_title_is_not_composition_evidence(title):
    evidence, _ = extract_gutenberg_date_evidence(
        _text("ROMA\nEDITORE TEST\n1891\nTesto italiano"),
        title=title,
    )

    assert not any(item["kind"] == "title_work_period" for item in evidence)


def test_translation_after_1900_and_language_variety_remain_manual():
    translation = _row(
        "3",
        "review_translation_edition_date",
        role="nineteenth_century_bridge_candidate",
    )
    translation_result = resolve_gutenberg_metadata_row(
        translation,
        _text(
            "PRIMA VERSIONE ITALIANA\nSECONDA EDIZIONE\n"
            "Copyright 1920 by Editore\nTesto italiano"
        ),
    )
    language = _row(
        "4",
        "review_language_variety_before_download",
        title="Sonetti romaneschi",
        role="language_variety_review_required",
    )
    language_result = resolve_gutenberg_metadata_row(
        language,
        _text("Questi sonetti sono scritti in dialetto romanesco. " * 20),
    )

    assert translation_result["resolution_status"] == "manual_review"
    assert "post_1900" in translation_result["manual_review_reasons"]
    assert language_result["resolution_status"] == "manual_review"
    markers = json.loads(language_result["language_variety_evidence_json"])
    assert markers["marker_counts"]["romanesco"] > 0


def test_unscoped_first_edition_reference_is_recorded_but_not_decisive():
    row = _row("5", "review_work_publication_date")
    result = resolve_gutenberg_metadata_row(
        row,
        _text(
            "DELLO STESSO AUTORE\n"
            "ALTRO LIBRO--Prima edizione, 1878.\n"
            "Testo italiano"
        ),
    )

    assert result["resolution_status"] == "manual_review"
    assert result["manual_review_reasons"] == "no_direct_period_evidence"
    evidence = json.loads(result["date_evidence_json"])
    assert any(item["kind"] == "explicit_first_edition" for item in evidence)
    assert result["selected_evidence_kind"] == ""


def test_run_review_retries_fetches_and_accounts_for_every_row(tmp_path):
    queue_path = tmp_path / "queue.csv"
    rows = [
        _row("1", "review_work_publication_date"),
        _row("2", "review_missing_period_evidence", title="Testo del Trecento", births=""),
        _row("3", "review_translation_edition_date"),
        _row(
            "4",
            "review_language_variety_before_download",
            role="language_variety_review_required",
        ),
    ]
    queue_sha = _write_queue(queue_path, rows)
    status_counts = {
        "review_work_publication_date": 1,
        "review_missing_period_evidence": 1,
        "review_translation_edition_date": 1,
        "review_language_variety_before_download": 1,
    }
    config = GutenbergMetadataReviewConfig(
        repo_root=tmp_path,
        queue_csv_path=queue_path,
        cache_dir=tmp_path / "cache",
        output_csv_path=tmp_path / "resolution.csv",
        manual_review_csv_path=tmp_path / "manual.csv",
        json_report_path=tmp_path / "report.json",
        markdown_report_path=tmp_path / "report.md",
        expected_queue_sha256=queue_sha,
        expected_record_count=4,
        expected_status_counts=status_counts,
        request_delay_seconds=0,
        fetch_attempts=2,
    )
    attempts: dict[str, int] = {}

    def fetch(ebook_id, **kwargs):
        attempts[ebook_id] = attempts.get(ebook_id, 0) + 1
        if ebook_id == "1" and attempts[ebook_id] == 1:
            raise ConnectionError("temporary")
        publications = {
            "1": "Milano, 1897",
            "2": "",
            "3": "Roma, 1888",
            "4": "",
        }
        return FetchedGutenbergText(
            ebook_id=ebook_id,
            url=f"https://example.test/{ebook_id}.txt",
            text=_text(
                "ROMA\nEDITORE TEST\n1897\nTesto italiano in lingua.",
                original_publication=publications[ebook_id],
            ),
        )

    report = run_gutenberg_metadata_review(config, fetch_text=fetch)

    assert attempts["1"] == 2
    assert report["record_count"] == 4
    assert report["automatic_resolved_count"] == 3
    assert report["manual_review_count"] == 1
    assert sum(report["automatic_decision_counts"].values()) == 3
    assert "manual_authoritative_review_required" not in report[
        "automatic_decision_counts"
    ]
    assert report["policy"]["activation_authorized"] is False
    assert len(list(config.cache_dir.glob("pg*.txt"))) == 4
    with config.manual_review_csv_path.open(encoding="utf-8", newline="") as handle:
        manual = list(csv.DictReader(handle))
    assert [row["ebook_id"] for row in manual] == ["4"]


def test_run_review_rejects_queue_hash_mismatch_before_fetch(tmp_path):
    queue_path = tmp_path / "queue.csv"
    _write_queue(queue_path, [_row("1", "review_work_publication_date")])
    config = GutenbergMetadataReviewConfig(
        repo_root=tmp_path,
        queue_csv_path=queue_path,
        cache_dir=tmp_path / "cache",
        output_csv_path=tmp_path / "resolution.csv",
        manual_review_csv_path=tmp_path / "manual.csv",
        json_report_path=tmp_path / "report.json",
        markdown_report_path=tmp_path / "report.md",
        expected_queue_sha256="0" * 64,
        expected_record_count=1,
        expected_status_counts={"review_work_publication_date": 1},
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_gutenberg_metadata_review(config)
