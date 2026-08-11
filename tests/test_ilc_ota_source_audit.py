import io
import zipfile

import pytest

from sonnet_corpus.ilc_ota_source_audit import (
    _language_scores,
    extract_libretti_units,
    extract_record_units,
    extract_xml_units,
    ota_metadata_decision,
    parse_ota_catalog_handles,
    parse_ota_item_page,
)


def _ota_html(*, created="1596", rights="http://creativecommons.org/publicdomain/zero/1.0/", title="Rime italiane", language="Italian", source_type="Text", bitstream=True):
    citation = '<meta name="citation_pdf_url" content="https://example.test/work.xml">' if bitstream else ""
    return f"""<html><head>
    <meta name="DC.title" content="{title}">
    <meta name="DC.creator" content="Autore">
    <meta name="DCTERMS.created" content="{created}">
    <meta name="DC.language" content="{language}">
    <meta name="DC.type" content="{source_type}">
    <meta name="DC.rights" content="{rights}">
    <meta name="DC.format" content="text/xml">
    {citation}</head></html>""".encode()


def _tei(body, *, title="Titolo", author="Autore", xml_id=""):
    identifier = f' xml:id="{xml_id}"' if xml_id else ""
    return f"""<?xml version="1.0"?><!DOCTYPE TEI SYSTEM "remote.dtd">
    <TEI xmlns="http://www.tei-c.org/ns/1.0"{identifier}>
      <teiHeader><fileDesc><titleStmt><title>{title}</title><author>{author}</author></titleStmt></fileDesc></teiHeader>
      <text><body>{body}</body></text>
    </TEI>""".encode()


def test_parse_catalog_handles_is_unique_and_sorted():
    content = b'<a href="/llds/xmlui/handle/20.500.14106/B2">x</a><a href="/llds/xmlui/handle/20.500.14106/A1">y</a><a href="/llds/xmlui/handle/20.500.14106/A1">z</a>'
    assert parse_ota_catalog_handles(content) == ["A1", "B2"]


def test_parse_ota_item_page_preserves_rights_date_and_bitstream():
    row = parse_ota_item_page(_ota_html(), handle="A1", landing_page_url="https://example.test/A1")
    assert row["title"] == "Rime italiane"
    assert row["created_year"] == 1596
    assert row["license_url"].endswith("/zero/1.0/")
    assert row["bitstream_url"] == "https://example.test/work.xml"


def test_parse_ota_item_page_inventories_all_primary_bitstreams():
    content = _ota_html().replace(
        b"</head>",
        b'</head><body><a href="/llds/xmlui/bitstream/handle/20.500.14106/A1/one.txt?sequence=4">one</a>'
        b'<a href="/llds/xmlui/bitstream/handle/20.500.14106/A1/two.txt?sequence=5">two</a></body>',
    )
    row = parse_ota_item_page(content, handle="A1", landing_page_url="https://example.test/A1")
    assert row["bitstream_url"].split(";") == [
        "https://llds.ling-phil.ox.ac.uk/llds/xmlui/bitstream/handle/20.500.14106/A1/one.txt?sequence=4",
        "https://llds.ling-phil.ox.ac.uk/llds/xmlui/bitstream/handle/20.500.14106/A1/two.txt?sequence=5",
    ]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, "eligible_text_probe_inactive"),
        ({"title": "Seven dialect tales"}, "conditioned_language_excluded_inactive"),
        ({"language": "Latin"}, "excluded_non_italian_metadata"),
        ({"source_type": "Dataset"}, "excluded_not_primary_text"),
        ({"license_url": ""}, "excluded_terms_unresolved"),
        ({"created_year": ""}, "excluded_period_unresolved"),
        ({"created_year": 1923}, "excluded_post_1900"),
        ({"bitstream_url": ""}, "excluded_no_text_bitstream"),
    ],
)
def test_ota_metadata_gate_fails_closed(overrides, expected):
    row = parse_ota_item_page(_ota_html(), handle="A1", landing_page_url="https://example.test/A1")
    row.update(overrides)
    assert ota_metadata_decision(row)[0] == expected


def test_ota_role_routes_poetry_and_ottocento():
    row = parse_ota_item_page(_ota_html(), handle="A1", landing_page_url="https://example.test/A1")
    assert ota_metadata_decision(row)[2] == "historical_non_sonnet_poetry"
    row.update({"title": "Lettere", "created_year": 1840})
    assert ota_metadata_decision(row)[2] == "ottocento_bridge_capped"


def test_xml_extractor_removes_header_notes_and_preserves_original_choice():
    payload = _tei("<p>Testo <note>apparato</note><choice><orig>antico</orig><reg>moderno</reg></choice>.</p>")
    units = extract_xml_units(payload, record_id="sample", member_path="sample.xml")
    assert len(units) == 1
    assert units[0].title == "Titolo"
    assert units[0].author == "Autore"
    assert "Testo antico." in units[0].text
    assert "apparato" not in units[0].text
    assert "moderno" not in units[0].text


def test_zip_extractor_ignores_lists_and_readme():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("works/one.xml", _tei("<p>Questo è un testo italiano abbastanza lungo.</p>"))
        archive.writestr("works/lists/people.xml", _tei("<p>Apparato.</p>"))
        archive.writestr("README.txt", "metadata")
    row = {"record_id": "sample", "title": "", "creator": ""}
    units = extract_record_units(row, buffer.getvalue(), file_name="sample.zip")
    assert [unit.member_path for unit in units] == ["works/one.xml"]


def test_legacy_plain_text_tags_are_removed_without_treating_sgml_as_xml():
    row = {"record_id": "ota_legacy", "title": "Opera", "creator": "Autore"}
    units = extract_record_units(
        row, b"<P 1><L 1>QUESTO E UN TESTO\n<L 2>ITALIANO.", file_name="legacy.txt",
    )
    assert len(units) == 1
    assert "<P 1>" not in units[0].text
    assert "QUESTO E UN TESTO" in units[0].text


def test_binary_legacy_payload_is_fail_closed_without_a_text_unit():
    row = {"record_id": "ota_binary", "title": "Opera", "creator": "Autore"}
    assert extract_record_units(row, b"text\x00\x01\x02" * 100, file_name="binary.txt") == []


def test_epub_uses_xhtml_text_instead_of_generic_zip_filter():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("chapter.xhtml", "<html><body><p>Testo italiano.</p></body></html>")
    row = {"record_id": "ota_epub", "title": "Opera", "creator": "Autore"}
    units = extract_record_units(row, buffer.getvalue(), file_name="work.epub")
    assert len(units) == 1
    assert units[0].text == "Testo italiano."


def test_language_differential_separates_italian_english_and_latin():
    italian = "che della sono questa non per con una".split()
    english = "the and of to with this is not".split()
    latin = "et est cum ad quod sunt nec ut".split()
    assert max(range(3), key=_language_scores(italian).__getitem__) == 0
    assert max(range(3), key=_language_scores(english).__getitem__) == 1
    assert max(range(3), key=_language_scores(latin).__getitem__) == 2


def test_libretti_extractor_merges_split_work_and_excludes_fonti():
    bodies = ['<text><body><head>Fonti</head><p>apparato</p></body></text>']
    for index in range(1, 56):
        bodies.append(f'<text xml:id="L{index}"><body><head>Opera {index}</head><p>testo {index}</p></body></text>')
    bodies.extend([
        '<text xml:id="L56-1"><body><head>Opera 56 parte 1</head><p>prima</p></body></text>',
        '<text xml:id="L56-2"><body><head>Opera 56 parte 2</head><p>seconda</p></body></text>',
    ])
    payload = ('<teiCorpus xmlns="http://www.tei-c.org/ns/1.0">' + ''.join(bodies) + '</teiCorpus>').encode()
    units = extract_libretti_units(payload)
    assert len(units) == 56
    merged = next(unit for unit in units if unit.unit_id.endswith(":L56"))
    assert "prima" in merged.text and "seconda" in merged.text
    assert all("apparato" not in unit.text for unit in units)
