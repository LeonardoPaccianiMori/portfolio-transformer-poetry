import json
from pathlib import Path

import pytest

from sonnet_corpus.liber_liber import FetchedLiberLiberText
from sonnet_corpus.liber_liber_probe import (
    probe_liber_liber_source,
    probe_liber_liber_sources,
    select_liber_liber_candidate_probe_row,
    select_liber_liber_probe_rows,
    write_liber_liber_attribution,
)
from sonnet_corpus.pretraining_manifest import (
    PretrainingSourceRow,
    write_pretraining_manifest,
)


def make_row(**overrides):
    values = {
        "source_id": "ll_novellino",
        "title": "Il Novellino",
        "author": "Anonymous / multiple traditions",
        "source_archive": "Liber Liber",
        "source_collection": "Liber Liber / Progetto Manuzio",
        "landing_page_url": "https://liberliber.it/novellino/",
        "download_url": "",
        "ebook_id": "",
        "language": "Italian",
        "period_bucket": "tier_a_pre_1375",
        "approx_date": "late XIII secolo",
        "genre": "short prose narratives",
        "text_kind": "prose",
        "inclusion_status": "include_probe",
        "public_domain_status": (
            "Underlying medieval work is out of copyright; "
            "digital edition is Creative Commons licensed."
        ),
        "license_notes": (
            "Liber Liber digital edition license: CC BY-NC-SA 4.0; "
            "https://creativecommons.org/licenses/by-nc-sa/4.0/"
        ),
        "edition_notes": "Rizzoli edition; digitization Marina Pianu.",
        "source_release_date": "2007-04-18",
        "source_last_updated": "",
        "expected_clean_text_path": "data/processed/pretraining/ll_novellino.txt",
        "token_count_report_path": "",
        "split": "",
        "boilerplate_strategy": "extract primary text from TXT archive",
        "mixed_text_strategy": "",
        "cleaning_notes": "Remove modern introduction.",
        "audit_notes": "",
    }
    values.update(overrides)
    return PretrainingSourceRow(**values)


def test_select_liber_liber_probe_rows_skips_conditional_mixed_work():
    rows = [
        make_row(source_id="active"),
        make_row(
            source_id="sercambi",
            text_kind="mixed",
            inclusion_status="conditional_extract_prose",
            mixed_text_strategy="remove poetic interludes",
        ),
        make_row(
            source_id="gutenberg",
            source_archive="Project Gutenberg",
        ),
    ]

    selected = select_liber_liber_probe_rows(rows)

    assert [row.source_id for row in selected] == ["active"]


def test_select_liber_liber_candidate_probe_row_requires_an_audit_only_prose_row():
    candidate = select_liber_liber_candidate_probe_row(
        [make_row(source_id="candidate", inclusion_status="audit_then_include")],
        "candidate",
    )

    assert candidate.source_id == "candidate"

    with pytest.raises(ValueError, match="not audit-only"):
        select_liber_liber_candidate_probe_row([make_row()], "ll_novellino")


def test_probe_writes_measurements_and_attribution(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    report_path = tmp_path / "report.json"
    attribution_path = tmp_path / "attribution.md"
    write_pretraining_manifest([make_row()], manifest_path)

    def fake_fetch_text(landing_page_url, title, session=None):
        return FetchedLiberLiberText(
            landing_page_url=landing_page_url,
            download_page_url="https://liberliber.it/download/",
            archive_url="https://media.test/novellino.zip",
            archive_format="txt_zip",
            raw_byte_count=123,
            text="Il Novellino\nCorpo del testo.\n",
        )

    report = probe_liber_liber_sources(
        manifest_path=manifest_path,
        report_path=report_path,
        attribution_path=attribution_path,
        request_delay=0,
        fetch_text=fake_fetch_text,
    )

    assert report["selected_rows"] == 1
    assert report["successful_rows"] == 1
    assert report["total_cleaned_words"] == 5
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["results"][0]["archive_format"] == "txt_zip"

    attribution = attribution_path.read_text(encoding="utf-8")
    assert "## Il Novellino" in attribution
    assert "CC BY-NC-SA 4.0" in attribution
    assert "digitization Marina Pianu" in attribution


def test_probe_records_fetch_error_without_stopping_report(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    report_path = tmp_path / "report.json"
    attribution_path = tmp_path / "attribution.md"
    write_pretraining_manifest([make_row()], manifest_path)

    def fake_fetch_text(landing_page_url, title, session=None):
        raise ConnectionError("source unavailable")

    report = probe_liber_liber_sources(
        manifest_path=manifest_path,
        report_path=report_path,
        attribution_path=attribution_path,
        request_delay=0,
        fetch_text=fake_fetch_text,
    )

    assert report["successful_rows"] == 0
    assert report["error_rows"] == 1
    assert report["results"][0]["error"] == "source unavailable"


def test_candidate_probe_writes_cleaned_samples_without_full_text(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    report_path = tmp_path / "candidate.json"
    write_pretraining_manifest(
        [make_row(source_id="candidate", inclusion_status="audit_then_include")],
        manifest_path,
    )
    progress_messages = []

    def fake_fetch_text(landing_page_url, title, session=None):
        return FetchedLiberLiberText(
            landing_page_url=landing_page_url,
            download_page_url="https://liberliber.it/download/",
            archive_url="https://media.test/candidate.zip",
            archive_format="txt_zip",
            raw_byte_count=123,
            text="Il Novellino\nCorpo del testo.\n",
        )

    report = probe_liber_liber_source(
        manifest_path=manifest_path,
        source_id="candidate",
        report_path=report_path,
        fetch_text=fake_fetch_text,
        progress=progress_messages.append,
    )

    assert report["activation_status"] == "audit_then_include"
    assert report["result"]["status"] == "ok"
    assert report["result"]["first_characters"] == "Il Novellino\nCorpo del testo.\n"
    assert "probing source: candidate" in progress_messages
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["result"]["archive_url"] == "https://media.test/candidate.zip"
    assert "text" not in saved["result"]


def test_write_attribution_uses_only_rows_passed_by_caller(tmp_path: Path):
    path = tmp_path / "attribution.md"

    write_liber_liber_attribution([make_row()], path)

    text = path.read_text(encoding="utf-8")
    assert text.count("## Il Novellino") == 1
    assert "non-commercial" in text
