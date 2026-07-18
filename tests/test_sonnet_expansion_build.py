import json
from pathlib import Path

from sonnet_corpus.italian_wikisource import (
    FetchedItalianWikisourcePage,
    FetchedItalianWikisourcePageCollection,
    WikisourcePageRevision,
)
from sonnet_corpus.manifest import ManifestRow, write_manifest
from sonnet_corpus.sonnet_expansion_build import (
    build_sonnets_expanded,
    build_sonnets_expanded_v2,
    create_sonnet_source_snapshot,
    read_manifest_rows,
)


ROOT_TITLE = "Rime varie (Alfieri, 1912)"
PAGE_TITLE = f"{ROOT_TITLE}/II. Loda le bellezze di una signora"


def make_manifest_row(poem_id: str, *, include: bool, clean_text_path: str) -> ManifestRow:
    return ManifestRow(
        poem_id=poem_id,
        title_or_first_line=poem_id,
        author="Base author",
        displayed_author="Base author",
        source_archive="Italian Wikisource",
        source_collection="Base collection",
        source_subcollection="Sonetti",
        source_url="https://example.test/base",
        source_revision_id="1",
        source_revision_timestamp="2026-07-18T10:00:00Z",
        downloaded_at_utc="2026-07-18T10:00:00+00:00",
        source_edition="Base edition",
        license_notes="Base license",
        period="XIV secolo",
        form="sonnet",
        form_evidence="test",
        count_method="line_count_14",
        attribution_status="secure",
        line_count_raw=14,
        line_count_clean=14,
        raw_text_path="",
        clean_text_path=clean_text_path,
        include_in_core_pre_petrarch=include,
        include_in_expanded_with_petrarch=include,
        include_in_training=include,
        split_core_pre_petrarch="train" if include else "excluded",
        split_expanded_with_petrarch="train" if include else "excluded",
        editorial_brackets_removed=False,
        line_markers_removed=False,
        cleaning_notes="test",
        audit_notes="test",
    )


def write_audit_report(path: Path) -> None:
    payload = {
        "source": {
            "source_id": "ws_alfieri_rime_1912",
            "landing_page_url": "https://it.wikisource.org/wiki/Rime_varie_(Alfieri,_1912)",
        },
        "activation_status": "audit_then_include",
        "root_revision": {
            "title": ROOT_TITLE,
            "revision_id": 100,
            "revision_timestamp": "2026-07-18T10:00:00Z",
        },
        "page_count": 2,
        "started_at_utc": "2026-07-18T10:00:00+00:00",
        "finished_at_utc": "2026-07-18T10:01:00+00:00",
        "candidates": [
            {
                "page_title": PAGE_TITLE,
                "revision_id": 101,
                "revision_timestamp": "2026-07-18T10:01:00Z",
                "status": "eligible_14_lines",
                "line_count_clean": 14,
                "exact_active_duplicate_poem_ids": [],
            },
            {
                "page_title": f"{ROOT_TITLE}/V. Una canzone",
                "revision_id": 102,
                "revision_timestamp": "2026-07-18T10:02:00Z",
                "status": "not_14_cleaned_lines",
                "line_count_clean": 28,
                "exact_active_duplicate_poem_ids": [],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_collection() -> FetchedItalianWikisourcePageCollection:
    poem = "\n".join(f"Alfieri line {index}" for index in range(1, 15))
    return FetchedItalianWikisourcePageCollection(
        landing_page_url="https://it.wikisource.org/wiki/Rime_varie_(Alfieri,_1912)",
        title=ROOT_TITLE,
        root_revision=WikisourcePageRevision(ROOT_TITLE, 100, "2026-07-18T10:00:00Z"),
        root_html="<div class='mw-parser-output'></div>",
        pages=[
            FetchedItalianWikisourcePage(
                revision=WikisourcePageRevision(PAGE_TITLE, 101, "2026-07-18T10:01:00Z"),
                html=f"<div class='mw-parser-output'><div class='poem'>{poem}</div></div>",
            )
        ],
    )


def write_foscolo_audit_report(path: Path) -> None:
    payload = {
        "source": {
            "source_id": "ws_foscolo_sonetti",
            "landing_page_url": "https://it.wikisource.org/wiki/Sonetti_(Foscolo)",
        },
        "activation_status": "audit_then_include",
        "edition_page_title_suffix": "1835)",
        "root_revision": {
            "title": "Opera:Sonetti (Foscolo)",
            "revision_id": 200,
            "revision_timestamp": "2026-07-18T10:00:00Z",
        },
        "page_count": 1,
        "started_at_utc": "2026-07-18T10:00:00+00:00",
        "finished_at_utc": "2026-07-18T10:01:00+00:00",
        "candidates": [
            {
                "page_title": "Alla Sera (1835)",
                "revision_id": 202,
                "revision_timestamp": "2026-07-18T10:02:00Z",
                "status": "eligible_14_lines",
                "line_count_clean": 14,
                "exact_active_duplicate_poem_ids": [],
                "source_record": {
                    "page_title": "Opera:Alla Sera",
                    "revision_id": 201,
                    "revision_timestamp": "2026-07-18T10:01:00Z",
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_foscolo_collection() -> FetchedItalianWikisourcePageCollection:
    poem = "\n".join(f"Foscolo line {index}" for index in range(1, 15))
    record = WikisourcePageRevision("Opera:Alla Sera", 201, "2026-07-18T10:01:00Z")
    return FetchedItalianWikisourcePageCollection(
        landing_page_url="https://it.wikisource.org/wiki/Sonetti_(Foscolo)",
        title="Opera:Sonetti (Foscolo)",
        root_revision=WikisourcePageRevision(
            "Opera:Sonetti (Foscolo)", 200, "2026-07-18T10:00:00Z"
        ),
        root_html="<div class='mw-parser-output'></div>",
        pages=[
            FetchedItalianWikisourcePage(
                revision=WikisourcePageRevision(
                    "Alla Sera (1835)", 202, "2026-07-18T10:02:00Z"
                ),
                html=f"<div class='mw-parser-output'><div class='poem'>{poem}</div></div>",
                source_record_revision=record,
            )
        ],
    )


def test_snapshot_keeps_only_reviewed_eligible_non_duplicate_pages(tmp_path: Path):
    audit_path = tmp_path / "audit.json"
    snapshot_path = tmp_path / "snapshot.json"
    write_audit_report(audit_path)

    snapshot = create_sonnet_source_snapshot(
        audit_report_path=audit_path,
        snapshot_path=snapshot_path,
        source_id="ws_alfieri_rime_1912",
    )

    assert snapshot["scope"] == "explicit_subpages"
    assert [page["title"] for page in snapshot["page_revisions"]] == [PAGE_TITLE]
    assert snapshot["audit_report"]["eligible_sonnet_count"] == 1


def test_versioned_build_copies_only_active_base_poems_and_adds_pinned_alfieri(tmp_path: Path):
    base_manifest_path = tmp_path / "data/metadata/poems_manifest.csv"
    base_poem_path = tmp_path / "data/processed/poems/base.txt"
    base_poem_path.parent.mkdir(parents=True)
    base_poem_path.write_text("base line\n", encoding="utf-8")
    write_manifest(
        [
            make_manifest_row(
                "base_active", include=True, clean_text_path="data/processed/poems/base.txt"
            ),
            make_manifest_row("base_excluded", include=False, clean_text_path=""),
        ],
        base_manifest_path,
    )
    audit_path = tmp_path / "audit.json"
    snapshot_path = tmp_path / "data/metadata/wikisource_snapshots/ws_alfieri_rime_1912.json"
    write_audit_report(audit_path)
    create_sonnet_source_snapshot(
        audit_report_path=audit_path,
        snapshot_path=snapshot_path,
        source_id="ws_alfieri_rime_1912",
    )
    temporary_dir = tmp_path / "data/interim"
    temporary_dir.mkdir(parents=True)
    (temporary_dir / "temporary.txt").write_text("temporary", encoding="utf-8")

    report = build_sonnets_expanded_v2(
        repo_root=tmp_path,
        base_manifest_path=base_manifest_path,
        snapshot_path=snapshot_path,
        request_delay=0,
        fetch_collection=lambda *args, **kwargs: make_collection(),
    )

    output_manifest = tmp_path / "data/metadata/sonnets_expanded_v2_manifest.csv"
    rows = read_manifest_rows(output_manifest)
    assert report["base_poem_count"] == 1
    assert report["added_poem_count"] == 1
    assert report["total_poem_count"] == 2
    assert [row.poem_id for row in rows] == [
        "base_active",
        "alfieri_ii_loda_le_bellezze_di_una_signora",
    ]
    assert (tmp_path / rows[0].clean_text_path).read_text(encoding="utf-8") == "base line\n"
    assert (tmp_path / rows[1].clean_text_path).read_text(encoding="utf-8").count("\n") == 14
    assert rows[1].include_in_core_pre_petrarch is False
    assert rows[1].include_in_expanded_with_petrarch is True
    assert not temporary_dir.exists()


def test_versioned_build_supports_an_edition_page_snapshot_for_foscolo(tmp_path: Path):
    base_manifest_path = tmp_path / "data/metadata/sonnets_expanded_v2_manifest.csv"
    base_poem_path = tmp_path / "data/processed/sonnets_expanded_v2/poems/base.txt"
    base_poem_path.parent.mkdir(parents=True)
    base_poem_path.write_text("base line\n", encoding="utf-8")
    write_manifest(
        [
            make_manifest_row(
                "base_active",
                include=True,
                clean_text_path="data/processed/sonnets_expanded_v2/poems/base.txt",
            )
        ],
        base_manifest_path,
    )
    audit_path = tmp_path / "audit.json"
    snapshot_path = tmp_path / "data/metadata/wikisource_snapshots/ws_foscolo_sonetti.json"
    write_foscolo_audit_report(audit_path)
    snapshot = create_sonnet_source_snapshot(
        audit_report_path=audit_path,
        snapshot_path=snapshot_path,
        source_id="ws_foscolo_sonetti",
    )

    report = build_sonnets_expanded(
        repo_root=tmp_path,
        base_manifest_path=base_manifest_path,
        snapshot_path=snapshot_path,
        output_dataset_id="sonnets_expanded_v3",
        request_delay=0,
        fetch_collection=lambda *args, **kwargs: make_foscolo_collection(),
    )

    rows = read_manifest_rows(tmp_path / "data/metadata/sonnets_expanded_v3_manifest.csv")
    assert snapshot["scope"] == "explicit_edition_pages"
    assert snapshot["source_record_revisions"][0]["title"] == "Opera:Alla Sera"
    assert report["base_poem_count"] == 1
    assert report["added_poem_count"] == 1
    assert [row.poem_id for row in rows] == ["base_active", "foscolo_alla_sera_1835"]
    assert rows[1].source_edition.startswith("Opere scelte di Ugo Foscolo II")
    assert rows[1].source_revision_id == "202"
