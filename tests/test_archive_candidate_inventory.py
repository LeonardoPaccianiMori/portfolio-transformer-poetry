import csv
import hashlib
import json
from pathlib import Path

from sonnet_corpus.archive_candidate_inventory import (
    ArchiveCandidateInventoryConfig,
    _ia_rights_status,
    _normalize_ia,
    build_archive_candidate_inventory,
    parse_beic_oai_page,
    parse_gallica_sru_page,
    parse_internet_culturale_collections,
    parse_midia_table,
)


def _config(tmp_path: Path) -> ArchiveCandidateInventoryConfig:
    return ArchiveCandidateInventoryConfig(
        repo_root=tmp_path,
        cache_dir=tmp_path / "data/local/archive_candidate_inventory_v1",
        inventory_path=tmp_path / "data/metadata/corpus_archive_candidate_inventory_v1.csv",
        summary_path=tmp_path / "data/metadata/corpus_archive_inventory_summary_v1.csv",
        json_report_path=tmp_path / "reports/corpus_archive_candidate_inventory_v1.json",
        markdown_report_path=tmp_path / "reports/corpus_archive_candidate_inventory_v1.md",
        request_delay_seconds=0,
        ia_rows_per_page=1000,
    )


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _ic_html() -> str:
    return """
    <span class="count-result">1 risultati trovati</span>
    <div class="module-row listing-height clearfix">
      <div class="block-tail-cover"><h1><a href="https://www.internetculturale.it/it/41/collezioni-digitali/123/opere-antiche">Opere antiche</a></h1></div>
      <div class="block-tail-text"><h2>Biblioteca Esempio</h2>
      <p>Libri e manoscritti di letteratura italiana dal Cinquecento.</p>
      <li class="bibDig"><a href="https://www.internetculturale.it/it/16/search?collection=123">Vedi biblioteca digitale</a></li></div>
    </div>
    """


def _beic_xml() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8"?>
    <OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
      <ListRecords>
        <record><header><identifier>oai:alma.39BEIC_INST:1</identifier><datestamp>2020-01-01</datestamp></header>
          <metadata><dc xmlns:dc="http://purl.org/dc/elements/1.1/">
            <dc:title>Rime e poesie</dc:title><dc:contributor>Autore Antico</dc:contributor>
            <dc:type>text</dc:type><dc:date>1880</dc:date><dc:language>ita</dc:language>
            <dc:rights>Creative Commons 4.0 Attribuzione Condividi allo stesso modo</dc:rights>
            <dc:identifier>https://example.test/delivery/1</dc:identifier>
          </dc></metadata></record>
        <record><header><identifier>oai:alma.39BEIC_INST:2</identifier><datestamp>2020-01-01</datestamp></header>
          <metadata><dc xmlns:dc="http://purl.org/dc/elements/1.1/">
            <dc:title>Modern English work</dc:title><dc:date>1950</dc:date><dc:language>eng</dc:language>
          </dc></metadata></record>
      </ListRecords>
    </OAI-PMH>"""


def _seed_complete_cache(config: ArchiveCandidateInventoryConfig) -> None:
    cache = config.cache_dir
    _write(
        cache / "eltec_italian/ELTeC-ita_metadata.tsv",
        "corpus-id\tfilename\txmlid\tauthor-name\ttitle\treference-year\tfirst-edition\tlanguage\tnumwords\n"
        "ELTeC-ita\tIT1880_Test\tIT1880\tRossi, Ada\tRomanzo antico\t1880\t1880\tita\t1000\n",
    )
    _write(cache / "eltec_italian/tree.json", json.dumps({"tree": [{"path": "level1/IT1880_Test.xml"}]}))
    ia_payload = {
        "total": 1,
        "count": 1,
        "items": [{
                "identifier": "old-poems", "title": "Poesie italiane", "creator": "Poeta",
                "year": 1750, "language": "ita", "subject": ["Poesia"],
                "licenseurl": "https://creativecommons.org/publicdomain/mark/1.0/",
                "format": ["DjVuTXT"], "collection": ["opensource"],
            }],
    }
    _write(cache / "internet_archive/cursor_0001.json", json.dumps(ia_payload))
    _write(cache / "gallica/sru_response.bin", b"Access Denied")
    _write(cache / "gallica/sru_response.json", json.dumps({
        "http_status": 403, "content_type": "text/plain", "content_sha256": "x", "url": "test",
    }))
    _write(cache / "internet_culturale/page_01.html", _ic_html())
    _write(cache / "beic/page_0001.xml", _beic_xml())
    _write(cache / "midia/opere-autori.pdf", b"fixture")
    _write(
        cache / "midia/opere-autori.txt",
        "ID                AUTORE                     GENERE       PERIODO   OPERA\n"
        "POE1_TEST_RIME    Autore Antico              Poesia      I         Rime\n"
        "LET5_TEST_PROS    Autore Moderno             Prosa Lett. V         Prosa moderna\n",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_internet_culturale_parser_preserves_collection_routes():
    rows = parse_internet_culturale_collections(_ic_html())

    assert rows == [{
        "collection_id": "123",
        "title": "Opere antiche",
        "institution": "Biblioteca Esempio",
        "description": "Libri e manoscritti di letteratura italiana dal Cinquecento.",
        "detail_url": "https://www.internetculturale.it/it/41/collezioni-digitali/123/opere-antiche",
        "digital_search_url": "https://www.internetculturale.it/it/16/search?collection=123",
    }]


def test_beic_oai_parser_preserves_repeated_dc_fields_and_token():
    content = _beic_xml().replace(
        b"</ListRecords>", b"<resumptionToken>next-token</resumptionToken></ListRecords>",
    )
    rows, token = parse_beic_oai_page(content)

    assert token == "next-token"
    assert len(rows) == 2
    assert rows[0]["title"] == ["Rime e poesie"]
    assert rows[0]["language"] == ["ita"]
    assert rows[0]["identifier"] == ["https://example.test/delivery/1"]


def test_gallica_sru_parser_preserves_rights_ocr_and_next_position():
    content = b"""<?xml version="1.0"?>
    <srw:searchRetrieveResponse xmlns:srw="http://www.loc.gov/zing/srw/" xmlns:dc="http://purl.org/dc/elements/1.1/">
      <srw:numberOfRecords>2</srw:numberOfRecords><srw:records><srw:record>
        <srw:recordData><dc:dc><dc:title>Rime</dc:title><dc:date>1880</dc:date>
        <dc:rights>public domain</dc:rights><dc:identifier>https://gallica.bnf.fr/ark:/12148/bpt-test</dc:identifier></dc:dc></srw:recordData>
        <srw:extraRecordData><uri>bpt-test</uri><nqamoyen>91.5</nqamoyen></srw:extraRecordData>
      </srw:record></srw:records><srw:nextRecordPosition>2</srw:nextRecordPosition>
    </srw:searchRetrieveResponse>"""
    rows, total, following = parse_gallica_sru_page(content)

    assert total == 2
    assert following == 2
    assert rows[0]["rights"] == ["public domain"]
    assert rows[0]["extra_uri"] == "bpt-test"
    assert rows[0]["extra_nqamoyen"] == "91.5"


def test_midia_parser_joins_author_and_work_continuations():
    content = """
ID                AUTORE                     GENERE       PERIODO   OPERA
POE1_FOLG_SON     Folgore di San             Poesia      I         Sonetti dei mesi
                  Gimignano                                         e della settimana
LET2_TEST_PRO     Autore Due                 Prosa Lett. II        Opera seconda
"""
    rows = parse_midia_table(content)

    assert rows[0] == {
        "id": "POE1_FOLG_SON", "author": "Folgore di San Gimignano",
        "genre": "Poesia", "period": "I",
        "work": "Sonetti dei mesi e della settimana",
        "source_id_occurrence": "1", "source_id_count": "1",
    }
    assert rows[1]["period"] == "II"


def test_internet_archive_rights_fail_closed_for_missing_and_no_derivatives():
    assert _ia_rights_status("", "") == "item_rights_missing_or_unresolved"
    assert _ia_rights_status(
        "https://creativecommons.org/licenses/by-nd/4.0/", "",
    ) == "incompatible_or_ambiguous_no_derivatives"
    assert _ia_rights_status(
        "https://creativecommons.org/publicdomain/mark/1.0/", "",
    ) == "explicit_reusable_item_rights"


def test_internet_archive_normalizer_requires_rights_format_and_literary_signal():
    base = {
        "identifier": "candidate", "title": "Rime antiche", "creator": "Poeta",
        "year": 1750, "language": "ita", "licenseurl": "https://creativecommons.org/publicdomain/mark/1.0/",
        "format": ["DjVuTXT"], "subject": ["Poesia"],
    }
    eligible = _normalize_ia(base)
    no_rights = _normalize_ia(base | {"licenseurl": ""})
    dialect = _normalize_ia(base | {"title": "Rime in dialetto veneziano"})

    assert eligible["inventory_decision"] == "eligible_item_text_probe_inactive"
    assert no_rights["inventory_decision"] == "hold_item_rights_unresolved"
    assert dialect["inventory_decision"] == "conditioned_language_metadata_hold"
    assert dialect["activation_status"] == "inactive_metadata_only"


def test_complete_cache_backed_build_reconciles_all_six_archives_and_is_deterministic(tmp_path):
    config = _config(tmp_path)
    _seed_complete_cache(config)

    first = build_archive_candidate_inventory(config)
    paths = (
        config.inventory_path, config.summary_path,
        config.json_report_path, config.markdown_report_path,
    )
    first_hashes = [_sha(path) for path in paths]
    second = build_archive_candidate_inventory(config)
    second_hashes = [_sha(path) for path in paths]
    rows = list(csv.DictReader(config.inventory_path.open(encoding="utf-8")))
    summaries = list(csv.DictReader(config.summary_path.open(encoding="utf-8")))

    assert first_hashes == second_hashes
    assert first == second
    assert {row["archive_id"] for row in summaries} == {
        "eltec_italian", "internet_archive", "gallica",
        "internet_culturale", "beic", "midia",
    }
    assert len(rows) == 7
    assert first["scope"]["corpus_text_acquired"] is False
    assert first["scope"]["text_activated"] is False
    assert first["scope"]["v7_created"] is False
    assert first["scope"]["gpu_work_started"] is False


def test_validation_rejects_invalid_runtime_and_page_settings(tmp_path):
    config = _config(tmp_path)
    bad = ArchiveCandidateInventoryConfig(**{
        **config.__dict__, "request_timeout_seconds": 0,
    })

    try:
        build_archive_candidate_inventory(bad)
    except ValueError as error:
        assert "timeout" in str(error)
    else:
        raise AssertionError("invalid timeout was accepted")
