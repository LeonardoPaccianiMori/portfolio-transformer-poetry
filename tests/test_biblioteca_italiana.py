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
    assert len(parsed.non_sonnet_verse) == 1
    assert parsed.structural_sonnet_candidates == ()
    assert parsed.non_sonnet_verse[0].verse_type == "ottava"
    assert parsed.non_sonnet_verse[0].heading_path == ("RIME", "Canto non sonetto")
    assert parsed.non_sonnet_verse[0].line_count == 2


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
    assert parsed.sonnet_candidate_safe_text == parsed.non_sonnet_text
    assert "Primo verso del canto\n\nSecondo verso del canto" not in parsed.non_sonnet_text
    assert "Primo verso del canto" in parsed.non_sonnet_text
    assert "Secondo verso del canto" in parsed.non_sonnet_text
    assert parsed.non_sonnet_verse[0].text == (
        "Primo verso del canto\nSecondo verso del canto\n"
    )
    assert "Questa è la prosa principale." in parsed.residual_text
    assert "Verso primo del sonetto" not in parsed.residual_text
    assert "Primo verso del canto" not in parsed.residual_text


def test_parse_bibit_tei_quarantines_untyped_fourteen_line_verse():
    lines = "".join(f"<l>Verso {index}</l>" for index in range(1, 15))
    xml = f"""<TEI.2>
      <teiHeader><fileDesc><titleStmt><title>Rime</title></titleStmt></fileDesc></teiHeader>
      <text><body><p>Prosa sicura.</p><lg type="poesia">{lines}</lg></body></text>
    </TEI.2>"""

    parsed = parse_bibit_tei(xml)

    assert parsed.sonnets == ()
    assert len(parsed.non_sonnet_verse) == 1
    assert parsed.non_sonnet_verse[0].line_count == 14
    assert len(parsed.structural_sonnet_candidates) == 1
    assert parsed.structural_sonnet_candidates[0].unit_id == "structural_0001"
    assert "Verso 1" in parsed.non_sonnet_text
    assert "Verso 1" not in parsed.sonnet_candidate_safe_text
    assert parsed.sonnet_candidate_safe_text == "Prosa sicura.\n"


def test_parse_bibit_tei_quarantines_nested_structural_sonnet_only_once():
    lines = "".join(f"<l>Verso {index}</l>" for index in range(1, 15))
    xml = f"""<TEI.2>
      <teiHeader><fileDesc><titleStmt><title>Rime</title></titleStmt></fileDesc></teiHeader>
      <text><body><lg type="raccolta">
        <lg type="poesia">{lines}</lg>
        <lg type="poesia"><l>Verso non sonetto uno</l><l>Verso non sonetto due</l></lg>
      </lg></body></text>
    </TEI.2>"""

    parsed = parse_bibit_tei(xml)

    assert len(parsed.non_sonnet_verse) == 1
    assert len(parsed.structural_sonnet_candidates) == 1
    assert parsed.structural_sonnet_candidates[0].line_count == 14
    assert "Verso 1" not in parsed.sonnet_candidate_safe_text
    assert "Verso non sonetto uno" in parsed.sonnet_candidate_safe_text


def test_parse_bibit_tei_quarantines_four_stanza_sonnet_container():
    stanzas = []
    next_line = 1
    for stanza_size in (4, 4, 3, 3):
        lines = "".join(
            f"<l>Verso {index}</l>"
            for index in range(next_line, next_line + stanza_size)
        )
        stanzas.append(f"<lg>{lines}</lg>")
        next_line += stanza_size
    xml = f"""<TEI.2>
      <teiHeader><fileDesc><titleStmt><title>Rime</title></titleStmt></fileDesc></teiHeader>
      <text><body>
        <div1 type="poesia"><head>I</head>{''.join(stanzas)}</div1>
        <div1 type="poesia"><head>II</head><lg><l>Verso non sonetto</l></lg></div1>
      </body></text>
    </TEI.2>"""

    parsed = parse_bibit_tei(xml)

    assert len(parsed.structural_sonnet_candidates) == 1
    assert parsed.structural_sonnet_candidates[0].line_count == 14
    assert "Verso 1" not in parsed.sonnet_candidate_safe_text
    assert "Verso non sonetto" in parsed.sonnet_candidate_safe_text


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


def test_parse_bibit_tei_removes_only_xml_forbidden_control_characters():
    xml = """<TEI.2>
      <teiHeader><fileDesc>
        <titleStmt><title>Prova</title></titleStmt>
        <sourceDesc><bibl><idno>ITICCU\x001234</idno></bibl></sourceDesc>
      </fileDesc></teiHeader>
      <text><body><p>Testo\x07 letterario.</p></body></text>
    </TEI.2>"""

    parsed = parse_bibit_tei(xml)

    assert parsed.provenance.source_identifier == "ITICCU1234"
    assert parsed.body_text == "Testo letterario.\n"


def test_parse_bibit_tei_resolves_only_pinned_legacy_greek_entities():
    xml = """<!DOCTYPE TEI.2 SYSTEM 'http://example.invalid/bibit.dtd'>
    <TEI.2>
      <teiHeader><fileDesc><titleStmt><title>Prova</title></titleStmt></fileDesc></teiHeader>
      <text><body><p lang="grc">&esmogr;&sgr;&khgr;&eegr;&mgr;&agr;</p></body></text>
    </TEI.2>"""

    parsed = parse_bibit_tei(xml)

    assert parsed.body_text == "ἐσχημα\n"

    with pytest.raises(ValueError, match="unsupported TEI named entity"):
        parse_bibit_tei(xml.replace("&esmogr;", "&unapproved;"))


def test_parse_bibit_tei_preserves_empty_language_identifier():
    xml = """<TEI.2>
      <teiHeader>
        <fileDesc><titleStmt><title>Rime</title></titleStmt></fileDesc>
        <profileDesc><langUsage>
          <language id="ita"/><language id="lat">Latino</language>
        </langUsage></profileDesc>
      </teiHeader>
      <text><body><p>Testo italiano.</p></body></text>
    </TEI.2>"""

    parsed = parse_bibit_tei(xml)

    assert parsed.provenance.languages == ("ita", "Latino")


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
