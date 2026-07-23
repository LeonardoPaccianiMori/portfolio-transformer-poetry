import json
from pathlib import Path

from sonnet_corpus.italian_wikisource import (
    FetchedItalianWikisourcePage,
    FetchedItalianWikisourcePageCollection,
    WikisourcePageRevision,
)
from sonnet_corpus.manifest import ManifestRow, write_manifest
from sonnet_corpus.sonnet_expansion_build import (
    ACTIVATED_SOURCE_METADATA,
    build_sonnets_expanded,
    build_sonnets_expanded_from_snapshots,
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


def test_snapshot_pins_a_shared_page_once_but_keeps_both_poem_segments(
    tmp_path: Path,
):
    audit_path = tmp_path / "audit.json"
    snapshot_path = tmp_path / "snapshot.json"
    page_title = "Rime (Andreini)/Sonetti CLXXI-CLXXII"
    payload = {
        "source": {
            "source_id": "ws_andreini_rime_1601",
            "landing_page_url": "https://it.wikisource.org/wiki/Rime_(Andreini)",
        },
        "activation_status": "audit_then_include",
        "root_revision": {
            "title": "Rime (Andreini)",
            "revision_id": 300,
            "revision_timestamp": "2026-07-18T10:00:00Z",
        },
        "page_count": 1,
        "started_at_utc": "2026-07-18T10:00:00+00:00",
        "finished_at_utc": "2026-07-18T10:01:00+00:00",
        "candidates": [
            {
                "page_title": page_title,
                "revision_id": 301,
                "revision_timestamp": "2026-07-18T10:01:00Z",
                "segment_index": segment_index,
                "segment_label": f"SONETTO CLXX{segment_index}",
                "cleaned_text_sha256": f"hash-{segment_index}",
                "status": "eligible_14_lines",
                "line_count_clean": 14,
                "exact_active_duplicate_poem_ids": [],
            }
            for segment_index in (1, 2)
        ],
    }
    audit_path.write_text(json.dumps(payload), encoding="utf-8")

    snapshot = create_sonnet_source_snapshot(
        audit_report_path=audit_path,
        snapshot_path=snapshot_path,
        source_id="ws_andreini_rime_1601",
    )

    assert len(snapshot["page_revisions"]) == 1
    assert len(snapshot["eligible_candidates"]) == 2
    assert snapshot["audit_report"]["pinned_page_count"] == 1
    assert [
        candidate["segment_index"]
        for candidate in snapshot["eligible_candidates"]
    ] == [1, 2]


def test_varchi_activation_metadata_records_the_audited_edition_and_license():
    metadata = ACTIVATED_SOURCE_METADATA["ws_varchi_infermita"]

    assert metadata.author == "Benedetto Varchi"
    assert metadata.source_edition.endswith("original sonnets dated 1563.")
    assert "Firenze, per il Magheri, 1821" in metadata.source_edition
    assert metadata.license_notes.startswith("CC BY-SA 3.0 / GFDL")
    assert metadata.poem_id_prefix == "varchi"


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


def test_versioned_build_recovers_an_incomplete_prior_attempt(tmp_path: Path):
    base_manifest_path = tmp_path / "data/metadata/poems_manifest.csv"
    base_poem_path = tmp_path / "data/processed/poems/base.txt"
    base_poem_path.parent.mkdir(parents=True)
    base_poem_path.write_text("base line\n", encoding="utf-8")
    write_manifest(
        [
            make_manifest_row(
                "base_active",
                include=True,
                clean_text_path="data/processed/poems/base.txt",
            )
        ],
        base_manifest_path,
    )
    incomplete_path = tmp_path / "data/processed/sonnets_expanded_v2/poems/partial.txt"
    incomplete_path.parent.mkdir(parents=True)
    incomplete_path.write_text("partial", encoding="utf-8")
    audit_path = tmp_path / "audit.json"
    snapshot_path = tmp_path / "data/metadata/wikisource_snapshots/ws_alfieri_rime_1912.json"
    write_audit_report(audit_path)
    create_sonnet_source_snapshot(
        audit_report_path=audit_path,
        snapshot_path=snapshot_path,
        source_id="ws_alfieri_rime_1912",
    )

    build_sonnets_expanded_v2(
        repo_root=tmp_path,
        base_manifest_path=base_manifest_path,
        snapshot_path=snapshot_path,
        request_delay=0,
        fetch_collection=lambda *args, **kwargs: make_collection(),
    )

    assert not (tmp_path / "data/processed/sonnets_expanded_v2/poems/partial.txt").exists()
    rows = read_manifest_rows(tmp_path / "data/metadata/sonnets_expanded_v2_manifest.csv")
    assert all("sonnets_expanded_v2_build" not in row.clean_text_path for row in rows)
    assert all((tmp_path / row.clean_text_path).is_file() for row in rows)


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


def test_multi_source_build_splits_paired_pages_and_keeps_nested_ids_unique(
    tmp_path: Path,
):
    base_manifest_path = tmp_path / "data/metadata/sonnets_expanded_v4_manifest.csv"
    base_poem_path = tmp_path / "data/processed/sonnets_expanded_v4/poems/base.txt"
    base_poem_path.parent.mkdir(parents=True)
    base_poem_path.write_text("base line\n", encoding="utf-8")
    write_manifest(
        [
            make_manifest_row(
                "base_active",
                include=True,
                clean_text_path=(
                    "data/processed/sonnets_expanded_v4/poems/base.txt"
                ),
            )
        ],
        base_manifest_path,
    )

    snapshot_dir = tmp_path / "data/metadata/wikisource_snapshots"
    snapshot_dir.mkdir(parents=True)
    andreini_path = snapshot_dir / "ws_andreini_rime_1601.json"
    colonna_path = snapshot_dir / "ws_colonna_rime_1760.json"
    andreini_page = "Rime (Andreini)/Sonetti CLXXI-CLXXII"
    colonna_pages = [
        "Rime (Vittoria Colonna)/Sonetto I",
        "Rime (Vittoria Colonna)/Sonetti spirituali/Sonetto I",
    ]

    write_source_snapshot(
        andreini_path,
        source_id="ws_andreini_rime_1601",
        root_title="Rime (Andreini)",
        page_titles=[andreini_page],
        candidates=[
            {
                "page_title": andreini_page,
                "revision_id": 301,
                "revision_timestamp": "2026-07-18T10:01:00Z",
                "segment_index": 1,
                "segment_label": "SONETTO CLXXI",
                "cleaned_text_sha256": "",
            },
            {
                "page_title": andreini_page,
                "revision_id": 301,
                "revision_timestamp": "2026-07-18T10:01:00Z",
                "segment_index": 2,
                "segment_label": "SONETTO CLXXII",
                "cleaned_text_sha256": "",
            },
        ],
    )
    write_source_snapshot(
        colonna_path,
        source_id="ws_colonna_rime_1760",
        root_title="Rime (Vittoria Colonna)",
        page_titles=colonna_pages,
        candidates=[
            {
                "page_title": page_title,
                "revision_id": 301 + index,
                "revision_timestamp": "2026-07-18T10:01:00Z",
                "segment_index": 1,
                "segment_label": "",
                "cleaned_text_sha256": "",
            }
            for index, page_title in enumerate(colonna_pages)
        ],
    )

    collections = {
        "ws_andreini_rime_1601": make_andreini_paired_collection(
            andreini_page
        ),
        "ws_colonna_rime_1760": make_colonna_nested_collection(
            colonna_pages
        ),
    }
    report = build_sonnets_expanded_from_snapshots(
        repo_root=tmp_path,
        base_manifest_path=base_manifest_path,
        snapshot_paths=[andreini_path, colonna_path],
        output_dataset_id="sonnets_expanded_v5",
        request_delay=0,
        fetch_collection=lambda snapshot, **kwargs: collections[
            snapshot.source_id
        ],
    )

    rows = read_manifest_rows(
        tmp_path / "data/metadata/sonnets_expanded_v5_manifest.csv"
    )
    assert report["added_poem_count"] == 4
    assert [row.poem_id for row in rows] == [
        "base_active",
        "andreini_sonetti_clxxi_clxxii_sonetto_clxxi",
        "andreini_sonetti_clxxi_clxxii_sonetto_clxxii",
        "colonna_sonetto_i",
        "colonna_sonetti_spirituali_sonetto_i",
    ]
    assert [source["added_poem_count"] for source in report["sources"]] == [
        2,
        2,
    ]
    assert all((tmp_path / row.clean_text_path).is_file() for row in rows)


def write_source_snapshot(
    path: Path,
    *,
    source_id: str,
    root_title: str,
    page_titles: list[str],
    candidates: list[dict[str, object]],
) -> None:
    page_revisions = [
        {
            "title": page_title,
            "revision_id": 301 + index,
            "revision_timestamp": "2026-07-18T10:01:00Z",
        }
        for index, page_title in enumerate(page_titles)
    ]
    payload = {
        "source_id": source_id,
        "landing_page_url": f"https://example.test/{source_id}",
        "title": root_title,
        "scope": "explicit_subpages",
        "root_revision": {
            "title": root_title,
            "revision_id": 300,
            "revision_timestamp": "2026-07-18T10:00:00Z",
        },
        "page_revisions": page_revisions,
        "source_record_revisions": [],
        "edition_page_title_suffix": "",
        "source_metadata": {
            field: getattr(ACTIVATED_SOURCE_METADATA[source_id], field)
            for field in ACTIVATED_SOURCE_METADATA[source_id].__dataclass_fields__
        },
        "eligible_candidates": candidates,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_andreini_paired_collection(
    page_title: str,
) -> FetchedItalianWikisourcePageCollection:
    poem_one = "\n".join(f"Andreini one {index}" for index in range(1, 15))
    poem_two = "\n".join(f"Andreini two {index}" for index in range(1, 15))
    html = (
        "<div class='mw-parser-output'><div class='prp-pages-output'>"
        "<div class='centertext'>SONETTO CLXXI.</div>"
        f"<div class='poem'>{poem_one}</div>"
        "<div class='centertext'>SONETTO CLXXII.</div>"
        f"<div class='poem'>{poem_two}</div>"
        "</div></div>"
    )
    return FetchedItalianWikisourcePageCollection(
        landing_page_url="https://example.test/andreini",
        title="Rime (Andreini)",
        root_revision=WikisourcePageRevision(
            "Rime (Andreini)", 300, "2026-07-18T10:00:00Z"
        ),
        root_html="<div class='mw-parser-output'></div>",
        pages=[
            FetchedItalianWikisourcePage(
                revision=WikisourcePageRevision(
                    page_title, 301, "2026-07-18T10:01:00Z"
                ),
                html=html,
            )
        ],
    )


def make_colonna_nested_collection(
    page_titles: list[str],
) -> FetchedItalianWikisourcePageCollection:
    pages = []
    for index, page_title in enumerate(page_titles):
        poem = "\n".join(
            f"Colonna poem {index} line {line}" for line in range(1, 15)
        )
        pages.append(
            FetchedItalianWikisourcePage(
                revision=WikisourcePageRevision(
                    page_title,
                    301 + index,
                    "2026-07-18T10:01:00Z",
                ),
                html=(
                    "<div class='mw-parser-output'>"
                    f"<div class='poem'>{poem}</div></div>"
                ),
            )
        )
    return FetchedItalianWikisourcePageCollection(
        landing_page_url="https://example.test/colonna",
        title="Rime (Vittoria Colonna)",
        root_revision=WikisourcePageRevision(
            "Rime (Vittoria Colonna)", 300, "2026-07-18T10:00:00Z"
        ),
        root_html="<div class='mw-parser-output'></div>",
        pages=pages,
    )
