import json
from pathlib import Path

import pytest

from sonnet_corpus.gutenberg import FetchedGutenbergText
from sonnet_corpus.liber_liber import FetchedLiberLiberText
from sonnet_corpus.italian_wikisource import (
    FetchedItalianWikisourceWork,
    WikisourcePageRevision,
)
from sonnet_corpus.pretraining_build import (
    PretrainingBuildConfig,
    build_pretraining_corpus,
    select_pretraining_build_rows,
)
from sonnet_corpus.pretraining_manifest import (
    PretrainingSourceRow,
    write_pretraining_manifest,
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


def make_liber_liber_row(**overrides):
    values = {
        "source_id": "ll_novellino",
        "title": "Il Novellino",
        "author": "Anonymous",
        "source_archive": "Liber Liber",
        "source_collection": "Liber Liber / Progetto Manuzio",
        "landing_page_url": "https://liberliber.it/example/novellino/",
        "download_url": "",
        "ebook_id": "",
        "language": "Italian",
        "period_bucket": "tier_a_pre_1375",
        "approx_date": "XIII secolo",
        "genre": "novella prose",
        "text_kind": "prose",
        "inclusion_status": "include_probe",
        "public_domain_status": "underlying work out of copyright",
        "license_notes": "Liber Liber digital edition license: CC BY-NC-SA 4.0.",
        "edition_notes": "Liber Liber license applies to the digital edition.",
        "source_release_date": "",
        "source_last_updated": "",
        "expected_clean_text_path": "data/processed/pretraining/ll_novellino.txt",
        "token_count_report_path": "",
        "split": "",
        "boilerplate_strategy": "remove Liber Liber wrapper text",
        "mixed_text_strategy": "",
        "cleaning_notes": "Preserve source spelling.",
        "audit_notes": "",
    }
    values.update(overrides)
    return PretrainingSourceRow(**values)


def make_wikisource_row(**overrides):
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
        "inclusion_status": "include_probe",
        "public_domain_status": "Underlying work is public domain.",
        "license_notes": "Retain Wikisource attribution and license metadata.",
        "edition_notes": "Favaro edition.",
        "source_release_date": "",
        "source_last_updated": "",
        "expected_clean_text_path": "data/local/pretraining/expanded/sources/ws_galileo_saggiatore.txt",
        "token_count_report_path": "",
        "split": "",
        "boilerplate_strategy": "render pinned revisions",
        "mixed_text_strategy": "",
        "cleaning_notes": "Preserve spelling.",
        "audit_notes": "Activated only for expanded corpus.",
    }
    values.update(overrides)
    return PretrainingSourceRow(**values)


def test_select_pretraining_build_rows_keeps_only_active_supported_prose():
    rows = [
        make_row(source_id="active_pg"),
        make_liber_liber_row(source_id="active_ll"),
        make_wikisource_row(source_id="active_ws"),
        make_row(
            source_id="mixed",
            text_kind="mixed",
            inclusion_status="conditional_extract_prose",
            mixed_text_strategy="extract prose only",
        ),
        make_row(source_id="deferred", inclusion_status="defer", split="excluded"),
    ]

    selected = select_pretraining_build_rows(rows)

    assert [row.source_id for row in selected] == ["active_pg", "active_ll", "active_ws"]


def test_build_pretraining_corpus_uses_committed_wikisource_snapshot(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    snapshot_dir.joinpath("ws_galileo_saggiatore.json").write_text(
        json.dumps(
            {
                "source_id": "ws_galileo_saggiatore",
                "landing_page_url": "https://it.wikisource.org/wiki/Il_Saggiatore",
                "title": "Il Saggiatore",
                "root_revision": {
                    "title": "Il Saggiatore",
                    "revision_id": 100,
                    "revision_timestamp": "2026-07-15T10:00:00Z",
                },
                "page_revisions": [
                    {
                        "title": "Il Saggiatore/1",
                        "revision_id": 101,
                        "revision_timestamp": "2026-07-15T10:01:00Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    write_pretraining_manifest([make_wikisource_row()], manifest_path)

    def fake_fetch_wikisource(snapshot, request_delay, session=None, progress=None):
        assert snapshot.root_revision.revision_id == 100
        return FetchedItalianWikisourceWork(
            landing_page_url=snapshot.landing_page_url,
            title=snapshot.title,
            root_revision=snapshot.root_revision,
            page_revisions=snapshot.page_revisions,
            text="Corpo Wikisource verificato. " * 20,
            raw_html_character_count=1000,
        )

    processed_dir = tmp_path / "processed"
    report = build_pretraining_corpus(
        PretrainingBuildConfig(
            manifest_path=manifest_path,
            processed_dir=processed_dir,
            report_path=tmp_path / "report.json",
            temp_dir=tmp_path / "temp",
            wikisource_snapshot_dir=snapshot_dir,
            request_delay_seconds=0,
            min_character_count=20,
        ),
        fetch_italian_wikisource=fake_fetch_wikisource,
    )

    assert report.selected_rows == 1
    assert "Corpo Wikisource" in (processed_dir / "sources/ws_galileo_saggiatore.txt").read_text(encoding="utf-8")


def test_build_pretraining_corpus_writes_processed_files_report_and_cleans_temp(
    tmp_path: Path,
):
    manifest_path = tmp_path / "manifest.csv"
    processed_dir = tmp_path / "processed"
    report_path = tmp_path / "build_report.json"
    temp_dir = tmp_path / "temp"
    write_pretraining_manifest(
        [
            make_row(source_id="pg_sidrac_44549", ebook_id="44549"),
            make_liber_liber_row(source_id="ll_novellino"),
        ],
        manifest_path,
    )

    def fake_fetch_gutenberg(ebook_id, session=None):
        text = "\n".join(
            [
                "header",
                "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***",
                f"Corpo Project Gutenberg {ebook_id}. " * 20,
                "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***",
                "footer",
            ]
        )
        return FetchedGutenbergText(
            ebook_id=ebook_id,
            url=f"https://example.test/{ebook_id}.txt",
            text=text,
        )

    def fake_fetch_liber_liber(landing_page_url, title, session=None):
        return FetchedLiberLiberText(
            landing_page_url=landing_page_url,
            download_page_url=f"{landing_page_url}?download=1",
            archive_url=f"{landing_page_url}archive.zip",
            archive_format="txt_zip",
            raw_byte_count=123,
            text=("Corpo Liber Liber con prosa antica. " * 20),
        )

    report = build_pretraining_corpus(
        PretrainingBuildConfig(
            manifest_path=manifest_path,
            processed_dir=processed_dir,
            report_path=report_path,
            temp_dir=temp_dir,
            corpus_version="test_v1",
            request_delay_seconds=0,
            min_character_count=20,
        ),
        fetch_gutenberg=fake_fetch_gutenberg,
        fetch_liber_liber=fake_fetch_liber_liber,
    )

    pg_path = processed_dir / "sources" / "pg_sidrac_44549.txt"
    ll_path = processed_dir / "sources" / "ll_novellino.txt"
    combined_path = processed_dir / "corpus.txt"
    assert pg_path.exists()
    assert ll_path.exists()
    assert combined_path.exists()
    assert not temp_dir.exists()
    assert "PROJECT GUTENBERG EBOOK" not in pg_path.read_text(encoding="utf-8")
    assert "Corpo Liber Liber" in ll_path.read_text(encoding="utf-8")
    assert report.selected_rows == 2
    assert report.skipped_rows == 0

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["corpus_version"] == "test_v1"
    assert saved_report["selected_rows"] == 2
    assert set(saved_report["source_archive_shares"]) == {
        "Liber Liber",
        "Project Gutenberg",
    }
    assert saved_report["sources"][0]["processed_path"].endswith(
        "processed/sources/pg_sidrac_44549.txt"
    )


def test_build_pretraining_corpus_fails_without_publishing_partial_processed_files(
    tmp_path: Path,
):
    manifest_path = tmp_path / "manifest.csv"
    processed_dir = tmp_path / "processed"
    report_path = tmp_path / "build_report.json"
    temp_dir = tmp_path / "temp"
    write_pretraining_manifest([make_row()], manifest_path)

    def failing_fetch_gutenberg(ebook_id, session=None):
        raise ConnectionError("network unavailable")

    with pytest.raises(ConnectionError, match="network unavailable"):
        build_pretraining_corpus(
            PretrainingBuildConfig(
                manifest_path=manifest_path,
                processed_dir=processed_dir,
                report_path=report_path,
                temp_dir=temp_dir,
                request_delay_seconds=0,
                min_character_count=20,
            ),
            fetch_gutenberg=failing_fetch_gutenberg,
        )

    assert not processed_dir.exists()
    assert not report_path.exists()
    assert temp_dir.exists()


def test_build_pretraining_corpus_rejects_manifest_with_no_selected_rows(
    tmp_path: Path,
):
    manifest_path = tmp_path / "manifest.csv"
    write_pretraining_manifest(
        [make_row(source_id="deferred", inclusion_status="defer", split="excluded")],
        manifest_path,
    )

    with pytest.raises(ValueError, match="no active"):
        build_pretraining_corpus(
            PretrainingBuildConfig(
                manifest_path=manifest_path,
                processed_dir=tmp_path / "processed",
                report_path=tmp_path / "build_report.json",
                temp_dir=tmp_path / "temp",
                request_delay_seconds=0,
            )
        )


def test_build_pretraining_corpus_rejects_dangerous_deletable_paths(
    tmp_path: Path,
):
    manifest_path = tmp_path / "manifest.csv"
    write_pretraining_manifest([make_row()], manifest_path)

    with pytest.raises(ValueError, match="too broad"):
        build_pretraining_corpus(
            PretrainingBuildConfig(
                manifest_path=manifest_path,
                processed_dir=Path.cwd(),
                report_path=tmp_path / "build_report.json",
                temp_dir=tmp_path / "temp",
                request_delay_seconds=0,
            )
        )
