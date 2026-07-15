import json
from pathlib import Path

import pytest

from sonnet_corpus.italian_wikisource import (
    FetchedItalianWikisourceWork,
    WikisourcePageRevision,
)
from sonnet_corpus.italian_wikisource_probe import (
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
