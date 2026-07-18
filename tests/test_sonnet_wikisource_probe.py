import csv
import json
from pathlib import Path

import pytest
import sonnet_corpus.sonnet_wikisource_probe as probe_module

from sonnet_corpus.italian_wikisource import (
    FetchedItalianWikisourcePage,
    FetchedItalianWikisourcePageCollection,
    WikisourcePageRevision,
)
from sonnet_corpus.sonnet_wikisource_probe import (
    SONNET_COLLECTION_EXPECTATIONS,
    candidate_status,
    normalize_poem_for_duplicate_check,
    probe_sonnet_wikisource_source,
    read_sonnet_source_manifest,
    select_sonnet_wikisource_source,
)


def write_source_manifest(path: Path, *, status: str = "audit_then_include") -> None:
    fields = [
        "source_id",
        "title",
        "author",
        "source_archive",
        "landing_page_url",
        "language_variety",
        "role",
        "status",
        "license_or_reuse_status",
        "attribution_required",
    ]
    row = {
        "source_id": "ws_alfieri_rime_1912",
        "title": "Rime varie (1912 edition)",
        "author": "Vittorio Alfieri",
        "source_archive": "Italian Wikisource",
        "landing_page_url": "https://it.wikisource.org/wiki/Rime_varie_(Alfieri,_1912)",
        "language_variety": "standard Italian",
        "role": "core_standard_italian",
        "status": status,
        "license_or_reuse_status": "Record source metadata.",
        "attribution_required": "yes",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def write_active_manifest(repo_root: Path, path: Path, poem_text: str) -> None:
    poem_path = repo_root / "data/processed/poems/existing.txt"
    poem_path.parent.mkdir(parents=True)
    poem_path.write_text(poem_text, encoding="utf-8")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["poem_id", "clean_text_path", "include_in_training"])
        writer.writeheader()
        writer.writerow(
            {
                "poem_id": "existing_poem",
                "clean_text_path": "data/processed/poems/existing.txt",
                "include_in_training": "True",
            }
        )


def make_collection() -> FetchedItalianWikisourcePageCollection:
    duplicate_lines = "\n".join(f"Duplicate line {index}" for index in range(1, 15))
    eligible_lines = "\n".join(f"Eligible line {index}" for index in range(1, 15))
    non_sonnet_lines = "\n".join(f"Other line {index}" for index in range(1, 13))
    return FetchedItalianWikisourcePageCollection(
        landing_page_url="https://it.wikisource.org/wiki/Rime_varie_(Alfieri,_1912)",
        title="Rime varie (Alfieri, 1912)",
        root_revision=WikisourcePageRevision(
            title="Rime varie (Alfieri, 1912)",
            revision_id=100,
            revision_timestamp="2026-07-18T10:00:00Z",
        ),
        root_html="<div class='mw-parser-output'></div>",
        pages=[
            FetchedItalianWikisourcePage(
                revision=WikisourcePageRevision(
                    title="Rime varie (Alfieri, 1912)/Duplicate",
                    revision_id=101,
                    revision_timestamp="2026-07-18T10:01:00Z",
                ),
                html=f"<div class='mw-parser-output'><div class='poem'>{duplicate_lines}</div></div>",
            ),
            FetchedItalianWikisourcePage(
                revision=WikisourcePageRevision(
                    title="Rime varie (Alfieri, 1912)/Eligible",
                    revision_id=102,
                    revision_timestamp="2026-07-18T10:02:00Z",
                ),
                html=f"<div class='mw-parser-output'><div class='poem'>{eligible_lines}</div></div>",
            ),
            FetchedItalianWikisourcePage(
                revision=WikisourcePageRevision(
                    title="Rime varie (Alfieri, 1912)/Other",
                    revision_id=103,
                    revision_timestamp="2026-07-18T10:03:00Z",
                ),
                html=f"<div class='mw-parser-output'><div class='poem'>{non_sonnet_lines}</div></div>",
            ),
        ],
    )


def test_select_source_requires_a_core_italian_wikisource_audit_candidate(tmp_path: Path):
    manifest_path = tmp_path / "sources.csv"
    write_source_manifest(manifest_path)

    source = select_sonnet_wikisource_source(
        read_sonnet_source_manifest(manifest_path), "ws_alfieri_rime_1912"
    )

    assert source.author == "Vittorio Alfieri"

    write_source_manifest(manifest_path, status="audit_only_auxiliary")
    with pytest.raises(ValueError, match="not awaiting audit"):
        select_sonnet_wikisource_source(
            read_sonnet_source_manifest(manifest_path), "ws_alfieri_rime_1912"
        )


def test_committed_source_manifest_preserves_the_activated_alfieri_record():
    manifest_path = Path("data/metadata/sonnet_expansion_sources_manifest.csv")

    source = next(
        row
        for row in read_sonnet_source_manifest(manifest_path)
        if row.source_id == "ws_alfieri_rime_1912"
    )

    assert source.role == "core_standard_italian"
    assert source.status == "activated"
    assert source.landing_page_url.endswith("Rime_varie_(Alfieri,_1912)")


def test_probe_records_line_counts_duplicates_and_bounded_samples_without_full_text(tmp_path: Path):
    source_manifest_path = tmp_path / "sources.csv"
    active_manifest_path = tmp_path / "active.csv"
    report_path = tmp_path / "audit/probe.json"
    write_source_manifest(source_manifest_path)
    duplicate_text = "\n".join(f"Duplicate line {index}" for index in range(1, 15)) + "\n"
    write_active_manifest(tmp_path, active_manifest_path, duplicate_text)
    progress = []

    report = probe_sonnet_wikisource_source(
        source_manifest_path=source_manifest_path,
        active_poems_manifest_path=active_manifest_path,
        repo_root=tmp_path,
        source_id="ws_alfieri_rime_1912",
        report_path=report_path,
        request_delay=0,
        fetch_collection=lambda *args, **kwargs: make_collection(),
        progress=progress.append,
    )

    assert report["page_count"] == 3
    assert report["candidate_status_counts"] == {
        "eligible_14_lines": 1,
        "exact_duplicate_active_corpus": 1,
        "not_14_cleaned_lines": 1,
    }
    duplicate = report["candidates"][0]
    assert duplicate["exact_active_duplicate_poem_ids"] == ["existing_poem"]
    assert duplicate["status"] == "exact_duplicate_active_corpus"
    assert report["candidates"][1]["status"] == "eligible_14_lines"
    assert report["candidates"][2]["line_count_clean"] == 12
    assert "loaded 1 active poems across 1 exact-text fingerprints" in progress[1]

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    serialized = json.dumps(saved)
    assert "Duplicate line 1" in serialized
    assert "Eligible line 14" in serialized
    assert '"cleaned_text"' not in serialized
    assert '"raw_text"' not in serialized


def test_duplicate_normalization_is_case_and_whitespace_insensitive():
    assert normalize_poem_for_duplicate_check("A  line\nB line\n") == "a line b line"


def test_foscolo_probe_uses_the_verified_collection_root_title():
    expectation = SONNET_COLLECTION_EXPECTATIONS["ws_foscolo_sonetti"]

    assert expectation.root_page_title == "Opera:Sonetti (Foscolo)"
    assert expectation.explicit_page_titles == (
        "Opera:Alla Sera",
        "Opera:Non son chi fui; perì di noi gran parte",
        "Opera:Te nudrice alle Muse, ospite e Dea",
        "Opera:Perchè taccia il rumor di mia catena",
        "Opera:Così gl'interi giorni in lungo, incerto",
        "Opera:Meritamente, però ch'io potei",
        "Opera:Solcata ho fronte, occhi incavati intenti",
        "Opera:E tu ne' carmi avrai perenne vita",
        "Opera:A Zacinto",
        "Opera:In morte del fratello Giovanni",
        "Opera:Alla Musa (Foscolo)",
        "Opera:Che stai? già il secol l'orma ultima lascia",
    )
    assert expectation.edition_page_title_suffix == "1835)"


def test_probe_records_bibliographic_record_provenance_for_an_edition_page(tmp_path: Path):
    source_manifest_path = tmp_path / "sources.csv"
    active_manifest_path = tmp_path / "active.csv"
    report_path = tmp_path / "probe.json"
    write_source_manifest(source_manifest_path)
    write_active_manifest(tmp_path, active_manifest_path, "existing line\n")
    edition_lines = "\n".join(f"Edition line {index}" for index in range(1, 15))
    record_revision = WikisourcePageRevision(
        title="Opera:Example",
        revision_id=101,
        revision_timestamp="2026-07-18T10:01:00Z",
    )
    collection = FetchedItalianWikisourcePageCollection(
        landing_page_url="https://example.test/root",
        title="Root",
        root_revision=WikisourcePageRevision("Root", 100, "2026-07-18T10:00:00Z"),
        root_html="<div class='mw-parser-output'></div>",
        pages=[
            FetchedItalianWikisourcePage(
                revision=WikisourcePageRevision(
                    title="Example (1835)",
                    revision_id=102,
                    revision_timestamp="2026-07-18T10:02:00Z",
                ),
                html=f"<div class='mw-parser-output'><div class='poem'>{edition_lines}</div></div>",
                source_record_revision=record_revision,
            )
        ],
    )

    report = probe_sonnet_wikisource_source(
        source_manifest_path=source_manifest_path,
        active_poems_manifest_path=active_manifest_path,
        repo_root=tmp_path,
        source_id="ws_alfieri_rime_1912",
        report_path=report_path,
        request_delay=0,
        fetch_collection=lambda *args, **kwargs: collection,
    )

    assert report["candidates"][0]["source_record"] == {
        "page_title": "Opera:Example",
        "page_url": "https://it.wikisource.org/wiki/Opera:Example",
        "revision_id": 101,
        "revision_timestamp": "2026-07-18T10:01:00Z",
    }


def test_candidate_status_prioritizes_empty_then_form_then_duplicates():
    assert candidate_status(raw_text="", line_count_clean=14, duplicate_ids=[]) == "empty_after_extraction"
    assert candidate_status(raw_text="text", line_count_clean=13, duplicate_ids=["old"]) == "not_14_cleaned_lines"
    assert candidate_status(raw_text="text", line_count_clean=14, duplicate_ids=["old"]) == "exact_duplicate_active_corpus"


def test_probe_records_the_start_time_before_network_fetching(tmp_path: Path, monkeypatch):
    source_manifest_path = tmp_path / "sources.csv"
    active_manifest_path = tmp_path / "active.csv"
    report_path = tmp_path / "probe.json"
    write_source_manifest(source_manifest_path)
    write_active_manifest(tmp_path, active_manifest_path, "existing line\n")
    timestamps = iter(["2026-07-18T10:00:00+00:00", "2026-07-18T10:05:00+00:00"])
    monkeypatch.setattr(probe_module, "utc_now", lambda: next(timestamps))

    report = probe_sonnet_wikisource_source(
        source_manifest_path=source_manifest_path,
        active_poems_manifest_path=active_manifest_path,
        repo_root=tmp_path,
        source_id="ws_alfieri_rime_1912",
        report_path=report_path,
        request_delay=0,
        fetch_collection=lambda *args, **kwargs: make_collection(),
    )

    assert report["started_at_utc"] == "2026-07-18T10:00:00+00:00"
    assert report["finished_at_utc"] == "2026-07-18T10:05:00+00:00"
