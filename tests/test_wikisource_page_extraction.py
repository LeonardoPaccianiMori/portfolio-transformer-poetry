import bz2
import csv
import hashlib
import json
from pathlib import Path

import pytest

from sonnet_corpus.wikisource_page_extraction import (
    DumpPage,
    PageCache,
    WikisourcePageExtractionConfig,
    clean_wikisource_wikitext,
    extract_section,
    iter_mediawiki_dump,
    parse_pages_transclusions,
    reconstruct_page,
    run_wikisource_page_extraction,
)


def _write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _xml_page(title: str, namespace: int, page_id: int, revision_id: int, text: str) -> str:
    import html

    return f"""
    <page>
      <title>{html.escape(title)}</title><ns>{namespace}</ns><id>{page_id}</id>
      <revision><id>{revision_id}</id><timestamp>2026-08-01T00:00:00Z</timestamp>
      <text xml:space="preserve">{html.escape(text)}</text></revision>
    </page>"""


def _write_dump(path: Path, pages: list[str]) -> str:
    payload = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">'
        + "".join(pages)
        + "</mediawiki>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with bz2.open(path, "wt", encoding="utf-8") as handle:
        handle.write(payload)
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, mismatch: bool = False, omit_page_two: bool = False):
    metadata = tmp_path / "data/metadata"
    processed = tmp_path / "data/processed"
    reports = tmp_path / "reports"
    resolution = metadata / "resolution.csv"
    resolution_fields = (
        "work_root_id",
        "root_title",
        "landing_page_url",
        "proposed_role",
        "period_bucket",
        "author_evidence",
        "hierarchy_page_count",
        "direct_scan_titles",
        "checkpoint_4b_decision",
    )
    _write_csv(
        resolution,
        resolution_fields,
        [
            {
                "work_root_id": "itws:1",
                "root_title": "Opera",
                "landing_page_url": "https://it.wikisource.org/wiki/Opera",
                "proposed_role": "historical_general",
                "period_bucket": "origins_through_1800",
                "author_evidence": "Autore Uno",
                "hierarchy_page_count": "2",
                "direct_scan_titles": "Scan.djvu",
                "checkpoint_4b_decision": "eligible_page_level_audit_queue",
            },
            {
                "work_root_id": "itws:3",
                "root_title": "Sonetto",
                "landing_page_url": "https://it.wikisource.org/wiki/Sonetto",
                "proposed_role": "standard_sonnets",
                "period_bucket": "origins_through_1800",
                "author_evidence": "Autore Due",
                "hierarchy_page_count": "1",
                "direct_scan_titles": "Scan2.djvu",
                "checkpoint_4b_decision": "eligible_page_level_audit_queue",
            },
            {
                "work_root_id": "itws:4",
                "root_title": "Dialetto",
                "landing_page_url": "https://it.wikisource.org/wiki/Dialetto",
                "proposed_role": "conditioned_language_variant",
                "period_bucket": "nineteenth_century",
                "author_evidence": "Autore Tre",
                "hierarchy_page_count": "1",
                "direct_scan_titles": "Dialect.djvu",
                "checkpoint_4b_decision": "hold_scan_language_conflict",
            },
        ],
    )
    hierarchy = metadata / "hierarchy.csv"
    hierarchy_fields = (
        "work_root_id",
        "page_id",
        "page_title",
        "relative_title",
        "hierarchy_depth",
        "is_redirect",
        "latest_revision_id",
    )
    _write_csv(
        hierarchy,
        hierarchy_fields,
        [
            {
                "work_root_id": "itws:1",
                "page_id": "1",
                "page_title": "Opera",
                "relative_title": "",
                "hierarchy_depth": "0",
                "is_redirect": "False",
                "latest_revision_id": "10",
            },
            {
                "work_root_id": "itws:1",
                "page_id": "2",
                "page_title": "Opera/Capitolo 1",
                "relative_title": "Capitolo 1",
                "hierarchy_depth": "1",
                "is_redirect": "False",
                "latest_revision_id": "11",
            },
            {
                "work_root_id": "itws:3",
                "page_id": "3",
                "page_title": "Sonetto",
                "relative_title": "",
                "hierarchy_depth": "0",
                "is_redirect": "False",
                "latest_revision_id": "20",
            },
            {
                "work_root_id": "itws:4",
                "page_id": "4",
                "page_title": "Dialetto",
                "relative_title": "",
                "hierarchy_depth": "0",
                "is_redirect": "False",
                "latest_revision_id": "30",
            },
        ],
    )
    pages = [
        _xml_page("Opera", 0, 1, 10, "Indice che non deve duplicare il capitolo."),
        _xml_page(
            "Opera/Capitolo 1",
            0,
            2,
            99 if mismatch else 11,
            '<pages index="Scan.djvu" from="1" to="2" fromsection="testo" tosection="fine"/>',
        ),
        _xml_page(
            "Sonetto",
            0,
            3,
            20,
            "Tanto gentile e tanto onesta pare\nla donna mia quand'ella altrui saluta\n"
            "e ogne lingua deven tremando muta\ne li occhi no l'ardiscon di guardare.\n"
            "Ella si va, sentendosi laudare,\nbenignamente d'umiltà vestuta;\n"
            "e par che sia una cosa venuta\nda cielo in terra a miracol mostrare.",
        ),
        _xml_page(
            "Pagina:Scan.djvu/1",
            104,
            101,
            1001,
            '<noinclude>{{RigaIntestazione|1|TESTA}}</noinclude><section begin="testo"/>'
            "Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura.",
        ),
    ]
    if not omit_page_two:
        pages.append(
            _xml_page(
                "Pagina:Scan.djvu/2",
                104,
                102,
                1002,
                "ché la diritta via era smarrita.<section end=\"fine\"/>"
                "<noinclude>PIEDE</noinclude>",
            )
        )
    pages.append(_xml_page("Dialetto", 0, 4, 30, "Testo in dialetto."))
    dump = tmp_path / "dump.xml.bz2"
    dump_sha1 = _write_dump(dump, pages)

    bibit = processed / "bibit.csv"
    _write_csv(
        bibit,
        ("object_id", "artifact_status", "shard_path", "byte_start", "byte_end"),
        [],
    )
    broader = metadata / "broader.csv"
    _write_csv(broader, ("source_id", "expected_clean_text_path"), [])
    previous_probe = metadata / "previous.csv"
    pass1b_probe = metadata / "pass1b.csv"
    _write_csv(previous_probe, ("ebook_id",), [])
    _write_csv(pass1b_probe, ("ebook_id",), [])
    gutenberg = processed / "gutenberg.csv"
    _write_csv(
        gutenberg,
        ("ebook_id", "artifact_status", "shard_path", "byte_start", "byte_end"),
        [],
    )
    protected_text = tmp_path / "protected.txt"
    protected_text.write_text("Questo testo protetto non appare nelle opere di prova.\n", encoding="utf-8")
    sonnets = metadata / "sonnets.csv"
    _write_csv(
        sonnets,
        ("poem_id", "split_expanded_with_petrarch", "clean_text_path"),
        [
            {
                "poem_id": "protected-1",
                "split_expanded_with_petrarch": "validation",
                "clean_text_path": protected_text.relative_to(tmp_path).as_posix(),
            }
        ],
    )
    config = WikisourcePageExtractionConfig(
        repo_root=tmp_path,
        dump_path=dump,
        resolution_path=resolution,
        hierarchy_path=hierarchy,
        extraction_path=metadata / "extraction.csv",
        boundaries_path=metadata / "boundaries.csv",
        review_path=metadata / "review.csv",
        json_report_path=reports / "report.json",
        markdown_report_path=reports / "report.md",
        local_cache_dir=tmp_path / "data/local/wikisource/page_extraction",
        bibit_record_manifest_path=bibit,
        broader_sources_manifest_path=broader,
        sonnet_manifest_path=sonnets,
        gutenberg_previous_probe_path=previous_probe,
        gutenberg_previous_cache_dir=tmp_path / "previous-cache",
        gutenberg_pass_1b_probe_path=pass1b_probe,
        gutenberg_pass_1b_cache_dir=tmp_path / "pass1b-cache",
        gutenberg_resolved_manifest_path=gutenberg,
        expected_dump_sha1=dump_sha1,
        expected_eligible_roots=2,
        progress_interval=2,
    )
    return config


def _read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_dump_parser_selects_titles_and_preserves_revision(tmp_path):
    dump = tmp_path / "fixture.xml.bz2"
    _write_dump(
        dump,
        [
            _xml_page("Uno", 0, 1, 10, "Primo"),
            _xml_page("Due", 0, 2, 20, "Secondo"),
        ],
    )
    pages = list(iter_mediawiki_dump(dump, selected_titles={"Due"}))
    assert pages == [DumpPage("Due", 0, 2, 20, "2026-08-01T00:00:00Z", "Secondo")]


def test_pages_parser_handles_range_include_exclude_and_sections():
    directives = parse_pages_transclusions(
        '<pages index="Libro.djvu" from="2" to="5" exclude="3" '
        'fromsection="inizio" tosection="fine"/>'
    )
    assert directives[0].page_titles == (
        "Pagina:Libro.djvu/2",
        "Pagina:Libro.djvu/4",
        "Pagina:Libro.djvu/5",
    )
    assert directives[0].from_section == "inizio"
    assert directives[0].to_section == "fine"
    assert parse_pages_transclusions(
        '<pages index="Libro.djvu" include="i,iv,7"/>'
    )[0].page_titles[-1] == "Pagina:Libro.djvu/7"


def test_section_and_page_reconstruction_preserve_body_order():
    pages = {
        "Pagina:Libro.djvu/1": DumpPage(
            "Pagina:Libro.djvu/1", 104, 1, 1, "", '<section begin="x"/>Prima<noinclude>testa</noinclude>'
        ),
        "Pagina:Libro.djvu/2": DumpPage(
            "Pagina:Libro.djvu/2", 104, 2, 2, "", 'Seconda<section end="y"/>coda'
        ),
    }
    text, evidence = reconstruct_page(
        '<pages index="Libro.djvu" from="1" to="2" fromsection="x" tosection="y"/>',
        pages,
    )
    assert text == "Prima\n\nSeconda"
    assert evidence["missing_titles"] == []
    assert extract_section('<section begin="a"/>testo<section end="b"/>', begin="a", end="b") == "testo"


def test_cleaner_quarantines_unknown_templates_and_discards_headers():
    text, flags = clean_wikisource_wikitext(
        "{{RigaIntestazione|x}}Testo {{TemplateSconosciuto|dato}}",
        page_namespace=True,
    )
    assert text == "Testo"
    assert "unresolved_template:templatesconosciuto" in flags


def test_cleaner_applies_inspected_navigation_formatting_and_gap_rules():
    text, flags = clean_wikisource_wikitext(
        "{{Intestazione|Titolo=Opera}}{{Ct|t=2|CAPITOLO}} "
        "{{Pt|mo-|monumenti}} {{Capolettera|[[File:Capo.jpg|50px|S]]}}i "
        "{{TestoAssente|parola}}"
    )
    assert text == "CAPITOLO monumenti Si parola"
    assert flags == ["transcription_gap_template"]


def test_empty_layout_templates_and_literal_section_templates_are_safe():
    text, flags = clean_wikisource_wikitext("{{Centrato}}Titolo {{§}} 4 {{Ni}}testo")
    assert text == "Titolo § 4 testo"
    assert flags == []


def test_complete_fixture_excludes_parent_and_conditioned_root_and_is_deterministic(tmp_path):
    config = _fixture(tmp_path)
    messages = []
    first = run_wikisource_page_extraction(config, progress=messages.append)
    first_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            config.extraction_path,
            config.boundaries_path,
            config.review_path,
            config.json_report_path,
            config.markdown_report_path,
        )
    }
    second = run_wikisource_page_extraction(config)
    second_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            config.extraction_path,
            config.boundaries_path,
            config.review_path,
            config.json_report_path,
            config.markdown_report_path,
        )
    }
    rows = _read_csv(config.extraction_path)
    boundaries = _read_csv(config.boundaries_path)
    assert first == second
    assert first_hashes == second_hashes
    assert first["eligible_root_count"] == 2
    assert {row["root_title"] for row in rows} == {"Opera", "Sonetto"}
    assert [row["page_title"] for row in boundaries if row["work_root_id"] == "itws:1"] == [
        "Opera/Capitolo 1"
    ]
    assert first["required_proofread_page_count"] == 2
    assert first["matched_proofread_page_count"] == 2
    assert any("dump-pass-1" in message for message in messages)


def test_revision_mismatch_and_missing_transcription_are_held(tmp_path):
    mismatch = _fixture(tmp_path / "mismatch", mismatch=True)
    run_wikisource_page_extraction(mismatch)
    rows = {row["root_title"]: row for row in _read_csv(mismatch.extraction_path)}
    assert rows["Opera"]["checkpoint_4c_decision"] == "hold_revision_mismatch"

    missing = _fixture(tmp_path / "missing", omit_page_two=True)
    run_wikisource_page_extraction(missing)
    rows = {row["root_title"]: row for row in _read_csv(missing.extraction_path)}
    assert rows["Opera"]["checkpoint_4c_decision"] == "hold_missing_transcription"


def test_page_cache_recovers_completed_records_after_reopen(tmp_path):
    path = tmp_path / "cache.sqlite3"
    first = DumpPage("Pagina:Libro/1", 104, 1, 10, "time", "uno")
    second = DumpPage("Pagina:Libro/2", 104, 2, 20, "time", "due")
    with PageCache(path) as cache:
        assert cache.put_many([first]) == 1
    with PageCache(path) as cache:
        assert cache.titles() == {"Pagina:Libro/1"}
        cache.put_many([second])
        assert cache.get_many([first.title, second.title]) == {
            first.title: first,
            second.title: second,
        }


def test_non_numeric_range_is_rejected():
    with pytest.raises(ValueError, match="non-numeric"):
        parse_pages_transclusions('<pages index="Libro.djvu" from="i" to="iv"/>')


def test_malformed_range_is_quarantined_by_page_reconstruction():
    with pytest.raises(ValueError, match="non-numeric"):
        reconstruct_page('<pages index="Libro.djvu" from="354" to=\'"385\'/>', {})
