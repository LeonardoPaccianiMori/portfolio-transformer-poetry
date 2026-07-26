import json
from pathlib import Path

import pytest

from sonnet_corpus.italian_wikisource import read_wikisource_work_snapshot
from sonnet_corpus.pretraining_manifest import PretrainingSourceRow, write_pretraining_manifest
from sonnet_corpus.pretraining_wikisource_snapshot import (
    PretrainingWikisourceSnapshotSelection,
    create_pretraining_wikisource_snapshots,
)


def make_row(source_id: str, title: str) -> PretrainingSourceRow:
    return PretrainingSourceRow(
        source_id=source_id,
        title=title,
        author="Test author",
        source_archive="Italian Wikisource",
        source_collection="Italian Wikisource",
        landing_page_url=f"https://it.wikisource.org/wiki/{source_id}",
        download_url="",
        ebook_id="",
        language="Italian",
        period_bucket="tier_d_post_1600",
        approx_date="1700",
        genre="prose",
        text_kind="prose",
        inclusion_status="audit_then_include",
        public_domain_status="Public domain.",
        license_notes="Retain Wikisource attribution.",
        edition_notes="",
        source_release_date="",
        source_last_updated="",
        expected_clean_text_path="",
        token_count_report_path="",
        split="",
        boilerplate_strategy="",
        mixed_text_strategy="",
        cleaning_notes="",
        audit_notes="",
    )


def test_create_snapshots_pins_a_successful_audit_and_excludes_reviewed_index(tmp_path: Path):
    source_id = "ws_sarpi_istoria_concilio"
    title = "Istoria del Concilio tridentino"
    manifest_path = tmp_path / "manifest.csv"
    report_path = tmp_path / "audit.json"
    snapshot_dir = tmp_path / "snapshots"
    write_pretraining_manifest([make_row(source_id, title)], manifest_path)
    report_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "source_id": source_id,
                        "status": "ok",
                        "root_revision_id": 100,
                        "root_revision_timestamp": "2026-07-26T10:00:00Z",
                        "page_revisions": [
                            {
                                "title": "Istoria del Concilio tridentino/Indice del primo volume",
                                "revision_id": 100,
                                "revision_timestamp": "2026-07-26T10:00:00Z",
                            },
                            {
                                "title": "Istoria del Concilio tridentino/Indice del terzo volume",
                                "revision_id": 101,
                                "revision_timestamp": "2026-07-26T10:01:00Z",
                            },
                            {
                                "title": "Istoria del Concilio tridentino/Indice dei nomi/A",
                                "revision_id": 102,
                                "revision_timestamp": "2026-07-26T10:02:00Z",
                            },
                            {
                                "title": "Istoria del Concilio tridentino/Libro primo/Capitolo I",
                                "revision_id": 103,
                                "revision_timestamp": "2026-07-26T10:03:00Z",
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    snapshots = create_pretraining_wikisource_snapshots(
        manifest_path=manifest_path,
        audit_report_path=report_path,
        snapshot_dir=snapshot_dir,
        selections=(
            PretrainingWikisourceSnapshotSelection(
                source_id=source_id,
                scope="recursive_leaf_pages",
                excluded_page_titles=(
                    "Istoria del Concilio tridentino/Indice del primo volume",
                    "Istoria del Concilio tridentino/Indice del terzo volume",
                    "Istoria del Concilio tridentino/Indice dei nomi/A",
                ),
            ),
        ),
    )

    assert len(snapshots) == 1
    snapshot = read_wikisource_work_snapshot(snapshot_dir / f"{source_id}.json")
    assert snapshot.scope == "recursive_leaf_pages"
    assert [page.title for page in snapshot.page_revisions] == [
        "Istoria del Concilio tridentino/Libro primo/Capitolo I"
    ]


def test_create_snapshots_rejects_unsuccessful_audit_results(tmp_path: Path):
    source_id = "ws_sarpi_istoria_concilio"
    manifest_path = tmp_path / "manifest.csv"
    report_path = tmp_path / "audit.json"
    write_pretraining_manifest(
        [make_row(source_id, "Istoria del Concilio tridentino")], manifest_path
    )
    report_path.write_text(
        json.dumps({"results": [{"source_id": source_id, "status": "error"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="successful reviewed audit"):
        create_pretraining_wikisource_snapshots(
            manifest_path=manifest_path,
            audit_report_path=report_path,
            snapshot_dir=tmp_path / "snapshots",
            selections=(
                PretrainingWikisourceSnapshotSelection(
                    source_id=source_id,
                    scope="recursive_leaf_pages",
                ),
            ),
        )
