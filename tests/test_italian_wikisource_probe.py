import json
from pathlib import Path

import pytest

from sonnet_corpus.italian_wikisource import (
    FetchedItalianWikisourceWork,
    WikisourcePageRevision,
)
from sonnet_corpus.italian_wikisource_probe import (
    WORK_BOUNDARIES,
    audit_italian_wikisource_editorial_markers,
    find_editorial_markers,
    probe_italian_wikisource_sources,
    probe_italian_wikisource_source,
    select_italian_wikisource_probe_row,
)
from sonnet_corpus.pretraining_manifest import (
    PretrainingSourceRow,
    write_pretraining_manifest,
)


def make_row(**overrides):
    values = {
        "source_id": "ws_galileo_saggiatore",
        "title": "Il Saggiatore",
        "author": "Galileo Galilei",
        "source_archive": "Italian Wikisource",
        "source_collection": "Italian Wikisource",
        "landing_page_url": "https://it.wikisource.org/wiki/Il_Saggiatore",
        "download_url": "",
        "ebook_id": "",
        "language": "Italian",
        "period_bucket": "tier_d_post_1600",
        "approx_date": "1623",
        "genre": "scientific essay",
        "text_kind": "prose",
        "inclusion_status": "audit_then_include",
        "public_domain_status": "Underlying work is public domain.",
        "license_notes": "Retain Wikisource attribution and license metadata.",
        "edition_notes": "Edition requires confirmation during extraction.",
        "source_release_date": "",
        "source_last_updated": "",
        "expected_clean_text_path": "data/local/pretraining/processed/sources/ws_galileo_saggiatore.txt",
        "token_count_report_path": "",
        "split": "",
        "boilerplate_strategy": "strip navigation wrappers",
        "mixed_text_strategy": "",
        "cleaning_notes": "Preserve spelling.",
        "audit_notes": "Probe only.",
    }
    values.update(overrides)
    return PretrainingSourceRow(**values)


def test_select_probe_row_requires_audit_only_italian_wikisource_prose():
    selected = select_italian_wikisource_probe_row([make_row()], "ws_galileo_saggiatore")

    assert selected.title == "Il Saggiatore"

    with pytest.raises(ValueError, match="not audit-only"):
        select_italian_wikisource_probe_row(
            [make_row(inclusion_status="include_probe")],
            "ws_galileo_saggiatore",
        )


def test_probe_writes_revision_provenance_without_writing_full_text(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    report_path = tmp_path / "probe.json"
    write_pretraining_manifest([make_row()], manifest_path)
    progress_messages = []

    def fake_fetch_work(*args, **kwargs):
        assert kwargs["expected_first_subpage"] == "Il Saggiatore/Dedica"
        assert kwargs["expected_last_subpage"] == "Il Saggiatore/53"
        return FetchedItalianWikisourceWork(
            landing_page_url=args[0],
            title="Il Saggiatore",
            root_revision=WikisourcePageRevision(
                title="Il Saggiatore",
                revision_id=100,
                revision_timestamp="2026-07-15T10:00:00Z",
            ),
            page_revisions=[
                WikisourcePageRevision(
                    title="Il Saggiatore/Dedica",
                    revision_id=101,
                    revision_timestamp="2026-07-15T10:01:00Z",
                )
            ],
            text="## Il Saggiatore/Dedica\n\nCorpo primario.\n",
            raw_html_character_count=500,
        )

    report = probe_italian_wikisource_source(
        manifest_path=manifest_path,
        source_id="ws_galileo_saggiatore",
        report_path=report_path,
        request_delay=0,
        fetch_work=fake_fetch_work,
        progress=progress_messages.append,
    )

    assert report["activation_status"] == "audit_then_include"
    assert report["result"]["status"] == "ok"
    assert report["result"]["root_revision_id"] == 100
    assert report["result"]["page_count"] == 1
    assert report["result"]["cleaned_character_count"] == 41
    assert "probing source: ws_galileo_saggiatore" in progress_messages
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["result"]["page_revisions"][0]["revision_id"] == 101
    assert "Corpo primario." in saved["result"]["first_characters"]
    assert "text" not in saved["result"]


def test_probe_records_fetch_error_and_keeps_source_audit_only(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    report_path = tmp_path / "probe.json"
    write_pretraining_manifest([make_row()], manifest_path)

    def failing_fetch_work(*args, **kwargs):
        raise ConnectionError("network unavailable")

    report = probe_italian_wikisource_source(
        manifest_path=manifest_path,
        source_id="ws_galileo_saggiatore",
        report_path=report_path,
        request_delay=0,
        fetch_work=failing_fetch_work,
    )

    assert report["activation_status"] == "audit_then_include"
    assert report["result"]["status"] == "error"
    assert report["result"]["error"] == "network unavailable"
    assert report["result"]["page_count"] == 0


def test_batch_probe_continues_after_one_source_error_and_reports_markers(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    report_path = tmp_path / "batch.json"
    second_row = make_row(
        source_id="ws_galileo_dialogo",
        title="Dialogo sopra i due massimi sistemi del mondo tolemaico e copernicano",
    )
    write_pretraining_manifest([make_row(), second_row], manifest_path)

    def fake_fetch_work(*args, **kwargs):
        if kwargs["expected_title"] == "Il Saggiatore":
            raise ConnectionError("network unavailable")
        return FetchedItalianWikisourceWork(
            landing_page_url=args[0],
            title=kwargs["expected_title"],
            root_revision=WikisourcePageRevision(
                title=kwargs["expected_title"],
                revision_id=100,
                revision_timestamp="2026-07-23T10:00:00Z",
            ),
            page_revisions=[
                WikisourcePageRevision(
                    title="Dialogo sopra i due massimi sistemi del mondo tolemaico e copernicano/Dedica",
                    revision_id=101,
                    revision_timestamp="2026-07-23T10:01:00Z",
                )
            ],
            text=(
                "## Dialogo sopra i due massimi sistemi del mondo tolemaico e copernicano/Dedica\n\n"
                "Testo [nota].\nSi veda p. 3.\n"
            ),
            raw_html_character_count=500,
        )

    report = probe_italian_wikisource_sources(
        manifest_path=manifest_path,
        source_ids=["ws_galileo_saggiatore", "ws_galileo_dialogo"],
        report_path=report_path,
        request_delay=0,
        fetch_work=fake_fetch_work,
    )

    assert report["activation_status"] == "audit_then_include"
    assert report["successful_sources"] == 1
    assert report["error_sources"] == 1
    assert report["results"][1]["marker_summary"]["counts"] == {
        "bracketed_text": 1,
        "si_veda_reference": 1,
    }


def test_vico_probe_uses_exclusions_and_a_dynamic_final_boundary():
    boundaries = WORK_BOUNDARIES["ws_vico_scienza_nuova"]

    assert boundaries.root_page_title == "La scienza nuova - Volume I"
    assert boundaries.first_subpage == "La scienza nuova - Volume I/Titolo"
    assert boundaries.last_subpage == ""
    assert boundaries.excluded_subpage_prefixes == (
        "La scienza nuova - Volume I/Dedica dell'editore",
        "La scienza nuova - Volume I/Introduzione dell'editore",
        "La scienza nuova - Volume I/Illustrazione",
    )


def test_beccaria_probe_uses_the_primary_work_hierarchy():
    boundaries = WORK_BOUNDARIES["ws_beccaria_delitti_pene"]

    assert boundaries.first_subpage == "Dei delitti e delle pene/A chi legge"
    assert boundaries.last_subpage == ""


def test_vico_probe_uses_its_explicit_root_page_title(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    report_path = tmp_path / "probe.json"
    row = make_row(
        source_id="ws_vico_scienza_nuova",
        title="La scienza nuova",
        author="Giambattista Vico",
        landing_page_url="https://it.wikisource.org/wiki/La_scienza_nuova_-_Volume_I",
    )
    write_pretraining_manifest([row], manifest_path)

    def fake_fetch_work(*args, **kwargs):
        assert kwargs["expected_title"] == "La scienza nuova - Volume I"
        assert kwargs["expected_first_subpage"] == "La scienza nuova - Volume I/Titolo"
        return FetchedItalianWikisourceWork(
            landing_page_url=args[0],
            title=kwargs["expected_title"],
            root_revision=WikisourcePageRevision(
                title=kwargs["expected_title"],
                revision_id=100,
                revision_timestamp="2026-07-16T10:00:00Z",
            ),
            page_revisions=[
                WikisourcePageRevision(
                    title="La scienza nuova - Volume I/Titolo",
                    revision_id=101,
                    revision_timestamp="2026-07-16T10:01:00Z",
                )
            ],
            text="## La scienza nuova - Volume I/Titolo\n\nCorpo primario.\n",
            raw_html_character_count=500,
        )

    report = probe_italian_wikisource_source(
        manifest_path=manifest_path,
        source_id="ws_vico_scienza_nuova",
        report_path=report_path,
        request_delay=0,
        fetch_work=fake_fetch_work,
    )

    assert report["result"]["status"] == "ok"


def test_find_editorial_markers_counts_candidate_patterns_by_page():
    text = (
        "## La scienza nuova - Volume I/Libro I\n\n"
        "Testo principale. [Nota redazionale.]\n"
        "2 Si veda p. 300, nota 3.\n"
    )

    summary = find_editorial_markers(text, max_samples_per_marker=2)

    assert summary["counts"] == {
        "bracketed_text": 1,
        "si_veda_reference": 1,
    }
    assert summary["samples"] == [
        {
            "marker_type": "bracketed_text",
            "page_title": "La scienza nuova - Volume I/Libro I",
            "matched_text": "[Nota redazionale.]",
            "context": "Testo principale. [Nota redazionale.] 2 Si veda p. 300, nota 3.",
        },
        {
            "marker_type": "si_veda_reference",
            "page_title": "La scienza nuova - Volume I/Libro I",
            "matched_text": "2 Si veda p. 300, nota 3.",
            "context": "Testo principale. [Nota redazionale.] 2 Si veda p. 300, nota 3.",
        },
    ]


def test_editorial_marker_audit_writes_bounded_contexts_without_full_text(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    report_path = tmp_path / "marker_audit.json"
    write_pretraining_manifest([make_row()], manifest_path)

    def fake_fetch_work(*args, **kwargs):
        return FetchedItalianWikisourceWork(
            landing_page_url=args[0],
            title="Il Saggiatore",
            root_revision=WikisourcePageRevision(
                title="Il Saggiatore",
                revision_id=100,
                revision_timestamp="2026-07-16T10:00:00Z",
            ),
            page_revisions=[
                WikisourcePageRevision(
                    title="Il Saggiatore/Dedica",
                    revision_id=101,
                    revision_timestamp="2026-07-16T10:01:00Z",
                )
            ],
            text="## Il Saggiatore/Dedica\n\nTesto [nota].\nSi veda p. 3.\n",
            raw_html_character_count=500,
        )

    report = audit_italian_wikisource_editorial_markers(
        manifest_path=manifest_path,
        source_id="ws_galileo_saggiatore",
        report_path=report_path,
        request_delay=0,
        max_samples_per_marker=1,
        fetch_work=fake_fetch_work,
    )

    assert report["result"]["status"] == "ok"
    assert report["result"]["marker_summary"]["counts"] == {
        "bracketed_text": 1,
        "si_veda_reference": 1,
    }
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["result"]["marker_summary"]["samples"][0]["page_title"] == "Il Saggiatore/Dedica"
    assert "text" not in saved["result"]
