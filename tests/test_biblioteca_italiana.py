from pathlib import Path

import pytest

from sonnet_corpus.biblioteca_italiana import (
    BibItCatalogRecord,
    catalog_record_from_solr,
    parse_bibit_tei,
)


FIXTURE = Path("tests/fixtures/bibit_mixed_work.xml")


def test_parse_bibit_tei_separates_explicit_sonnets_and_provenance():
    parsed = parse_bibit_tei(FIXTURE.read_bytes(), object_id="bibit000001")

    assert parsed.provenance.digital_title == "Opera mista"
    assert parsed.provenance.digital_authors == ("Autore Esempio",)
    assert parsed.provenance.source_editors == ("Curatrice, Ada",)
    assert parsed.provenance.source_identifier == "TEST-001"
    assert "uso commerciale vietato" in parsed.provenance.availability
    assert parsed.provenance.languages == ("Italiano",)
    assert parsed.provenance.genres == ("Poesia",)
    assert len(parsed.sonnets) == 1
    assert parsed.sonnets[0].sonnet_type == "sonetto"
    assert parsed.sonnets[0].heading_path == ("RIME", "1")
    assert parsed.sonnets[0].line_count == 4
    assert "Verso secondo del sonetto\n\nVerso terzo" in parsed.sonnets[0].text


def test_parse_bibit_tei_preserves_structure_and_excludes_apparatus():
    parsed = parse_bibit_tei(FIXTURE.read_text(encoding="utf-8"))

    assert "PROSA\n\nQuesta è la prosa principale." in parsed.body_text
    assert "La lezione scelta conserva l'antica grafia." in parsed.body_text
    assert "nota editoriale" not in parsed.body_text
    assert "rifiutata" not in parsed.body_text
    assert "moderna" not in parsed.body_text
    assert "Commento moderno" not in parsed.body_text
    assert "Introduzione editoriale" not in parsed.body_text
    assert "Verso primo del sonetto" in parsed.body_text
    assert "Verso primo del sonetto" not in parsed.non_sonnet_text
    assert "Primo verso del canto\n\nSecondo verso del canto" not in parsed.non_sonnet_text
    assert "Primo verso del canto" in parsed.non_sonnet_text
    assert "Secondo verso del canto" in parsed.non_sonnet_text


def test_parse_bibit_tei_rejects_entity_declarations():
    xml = """<?xml version='1.0'?>
    <!DOCTYPE TEI.2 [<!ENTITY leaked SYSTEM 'file:///etc/passwd'>]>
    <TEI.2><teiHeader><fileDesc/></teiHeader><text><body>&leaked;</body></text></TEI.2>
    """

    with pytest.raises(ValueError, match="entity declarations"):
        parse_bibit_tei(xml)


def test_parse_bibit_tei_resolves_legacy_html_entities_without_dtd_access():
    xml = """<?xml version='1.0'?>
    <!DOCTYPE TEI.2 SYSTEM 'http://example.invalid/external.dtd'>
    <TEI.2>
      <teiHeader><fileDesc><titleStmt><title>Prova</title></titleStmt></fileDesc></teiHeader>
      <text><body><p>&laquo;testo&raquo;</p></body></text>
    </TEI.2>
    """

    parsed = parse_bibit_tei(xml)

    assert parsed.body_text == "«testo»\n"


def test_catalog_record_from_solr_normalizes_lists_and_urls():
    record = catalog_record_from_solr({
        "obj_id_s": "bibit000019",
        "ID": "135001",
        "post_title": "Commedia",
        "author_str": ["Alighieri, Dante"],
        "resource_genre_str": ["Poesia"],
        "resource_period_str": ["300"],
        "resource_language_str": ["ita"],
        "source_desc_publisher_str": ["Le Lettere"],
        "source_desc_pub_place_str": ["Firenze"],
        "source_desc_pub_date_str": ["1994"],
        "source_desc_author_str": ["Alighieri, Dante"],
        "source_desc_identifier_str": ["TEST"],
        "post_modified_gmt": "2019-01-01T00:00:00Z",
    })

    assert isinstance(record, BibItCatalogRecord)
    assert record.authors == ("Alighieri, Dante",)
    assert record.landing_page_url.endswith("/scheda/bibit000019")
    assert record.xml_url.endswith("/xml/bibit000019")
