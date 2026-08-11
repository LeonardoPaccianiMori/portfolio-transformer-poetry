import csv
import gzip
import hashlib
import json
from pathlib import Path

import sonnet_corpus.wikisource_archive_inventory as inventory_module
from sonnet_corpus.wikisource_archive_inventory import (
    DUMP_BASE_URL,
    DUMP_DATE,
    WikisourceArchiveInventoryConfig,
    _classify_metadata,
    build_wikisource_archive_inventory,
    iter_sql_insert_rows,
)


class FakeResponse:
    def __init__(self, payload, *, status_code=200, headers=None):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, *, rate_limited_requests=0):
        self.calls = []
        self.rate_limited_requests = rate_limited_requests

    def get(self, url, params=None, timeout=None, stream=False):
        self.calls.append((url, params, timeout, stream))
        if params and params.get("action") == "parse" and self.rate_limited_requests:
            self.rate_limited_requests -= 1
            return FakeResponse({}, status_code=429, headers={"Retry-After": "0"})
        revision_id = int(params["oldid"])
        text = (
            "Il testo italiano che non si perde e che con la sua voce racconta "
            "la storia di una città e della sua gente. "
        ) * 15
        return FakeResponse(
            {
                "parse": {
                    "revid": revision_id,
                    "displaytitle": f"Revision {revision_id}",
                    "text": f'<div class="mw-parser-output"><p>{text}</p></div>',
                }
            }
        )


def _write_gzip(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(text)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _page_row(
    page_id: int,
    title: str,
    *,
    latest: int,
    length: int,
    redirect: int = 0,
) -> str:
    return (
        f"({page_id},0,'{title}',{redirect},0,0.5,'20260801000000',"
        f"'20260801000000',{latest},{length},'wikitext',NULL)"
    )


def _fixture(tmp_path: Path, monkeypatch) -> WikisourceArchiveInventoryConfig:
    root = tmp_path
    cache = root / "data/local/wikisource/archive_inventory_v1"
    cache.mkdir(parents=True)
    page_rows = [
        _page_row(1, "Old_Work", latest=101, length=1000),
        _page_row(2, "Old_Work/Chapter", latest=102, length=5000),
        _page_row(3, "Bridge_Work", latest=103, length=2000),
        _page_row(4, "Dialect_Work", latest=104, length=2000),
        _page_row(5, "Modern_Work", latest=105, length=2000),
        _page_row(6, "Translation_Work", latest=106, length=2000),
        _page_row(7, "Unknown_Work", latest=107, length=2000),
        _page_row(8, "Existing_Work", latest=108, length=2000),
    ]
    page_text = "INSERT INTO `page` VALUES " + ",".join(page_rows) + ";\n"

    category_titles = [
        "Testi_di_Autore_Antico",
        "Testi_del_1700",
        "Poesie",
        "Testi_di_Autore_Ottocento",
        "Testi_del_1850",
        "Testi_di_Autore_Dialettale",
        "Testi_in_romanesco",
        "Testi_del_1800",
        "Testi_di_Autore_Moderno",
        "Testi_del_1920",
        "Testi_di_Traduttore",
        "Traduzioni_da_inglese",
        "Testi_del_1750",
        "Testi_di_Autore_Esistente",
        "Testi_del_1600",
    ]
    link_rows = [
        f"({index},14,'{title}')" for index, title in enumerate(category_titles, start=1)
    ]
    link_text = "INSERT INTO `linktarget` VALUES " + ",".join(link_rows) + ";\n"
    page_targets = {
        1: (1, 2, 3),
        3: (4, 5),
        4: (6, 7, 8),
        5: (9, 10),
        6: (11, 12, 13),
        8: (14, 15),
    }
    category_rows = []
    for page_id, target_ids in page_targets.items():
        for target_id in target_ids:
            category_rows.append(
                f"({page_id},'SORT','2026-08-01 00:00:00','',"
                f"'page',1,{target_id})"
            )
    category_text = (
        "INSERT INTO `categorylinks` VALUES " + ",".join(category_rows) + ";\n"
    )

    dump_payloads = {
        "page": page_text,
        "categorylinks": category_text,
        "linktarget": link_text,
    }
    dump_files = {}
    for label, text in dump_payloads.items():
        filename = f"fixture-{label}.sql.gz"
        path = cache / filename
        _write_gzip(path, text)
        dump_files[label] = (filename, hashlib.sha1(path.read_bytes()).hexdigest())
    monkeypatch.setattr(inventory_module, "DUMP_FILES", dump_files)

    (cache / "siteinfo_rights_v1.json").write_text(
        json.dumps(
            {
                "query": {
                    "rightsinfo": {
                        "text": "Creative Commons Attribution-Share Alike 4.0",
                        "url": "https://creativecommons.org/licenses/by-sa/4.0/deed.it",
                    },
                    "general": {"generator": "MediaWiki test"},
                }
            }
        ),
        encoding="utf-8",
    )
    metadata = root / "data/metadata"
    metadata.mkdir(parents=True)
    (metadata / "broader.csv").write_text(
        "source_id,source_archive,landing_page_url\n"
        "existing,Italian Wikisource,https://it.wikisource.org/wiki/Existing_Work\n",
        encoding="utf-8",
    )
    (metadata / "poems.csv").write_text(
        "poem_id,source_archive,source_url\n",
        encoding="utf-8",
    )
    snapshots = metadata / "wikisource_snapshots"
    snapshots.mkdir()

    return WikisourceArchiveInventoryConfig(
        repo_root=root,
        cache_dir=cache,
        inventory_path=metadata / "inventory.csv",
        page_hierarchy_path=metadata / "pages.csv",
        composition_gate_path=metadata / "gate.csv",
        inspection_sample_path=metadata / "sample.csv",
        json_report_path=root / "reports/report.json",
        markdown_report_path=root / "reports/report.md",
        broader_manifest_path=metadata / "broader.csv",
        poems_manifest_path=metadata / "poems.csv",
        snapshot_dir=snapshots,
        dump_date=DUMP_DATE,
        dump_base_url=DUMP_BASE_URL,
        sample_size=6,
        request_delay=0,
        progress_interval=2,
    )


def test_sql_parser_decodes_escaped_strings_and_null(tmp_path):
    path = tmp_path / "test.sql.gz"
    _write_gzip(
        path,
        "INSERT INTO `sample` VALUES "
        "(1,'L\\'opera','riga\\nnuova',NULL),(2,'slash\\\\finale','x',0);\n",
    )

    assert list(iter_sql_insert_rows(path, "sample")) == [
        ["1", "L'opera", "riga\nnuova", None],
        ["2", "slash\\finale", "x", "0"],
    ]


def test_sql_parser_tolerates_malformed_binary_sort_keys(tmp_path):
    path = tmp_path / "binary.sql.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(
            b"INSERT INTO `categorylinks` VALUES "
            b"(10,'malformed-\xc3','2026-08-01 00:00:00','',"
            b"'page',1,99);\n"
        )

    rows = list(iter_sql_insert_rows(path, "categorylinks"))

    assert rows[0][0] == "10"
    assert rows[0][6] == "99"
    assert "\ufffd" in rows[0][1]


def test_metadata_classifier_routes_period_language_translation_and_form():
    historical = _classify_metadata(
        "Rime antiche",
        [
            "Testi di Autore",
            "Testi di storia dell'arte",
            "Testi del 1700",
            "Poesie",
            "Sonetti",
        ],
    )
    dialect = _classify_metadata(
        "Versi",
        ["Testi del XIX secolo", "Testi in romanesco"],
    )
    translation = _classify_metadata(
        "Versione",
        ["Testi del 1750", "Traduzioni dall'inglese"],
    )

    assert historical["metadata_decision"] == "historical_core_metadata_candidate"
    assert historical["author_evidence"] == "Autore"
    assert historical["proposed_role"] == "historical_non_sonnet_poetry"
    assert historical["form_route"] == "sonnet_signal"
    assert dialect["metadata_decision"] == "conditioned_language_candidate"
    assert dialect["language_route"] == "conditioned_romanesco"
    assert translation["metadata_decision"] == "hold_translation_edition_review"


def test_inspection_sample_retries_a_rate_limited_request(tmp_path, monkeypatch):
    config = _fixture(tmp_path, monkeypatch)
    messages = []
    sleeps = []
    monkeypatch.setattr(inventory_module, "sleep", sleeps.append)

    report = build_wikisource_archive_inventory(
        config,
        session=FakeSession(rate_limited_requests=1),
        progress=messages.append,
    )

    assert report["bounded_inspection"]["sample_size"] == 6
    assert sleeps == [0.0]
    assert any("inspection-sample rate-limited retry=1/6" in item for item in messages)


def test_archive_inventory_is_complete_separated_and_deterministic(
    tmp_path, monkeypatch
):
    config = _fixture(tmp_path, monkeypatch)
    messages = []
    session = FakeSession()

    first = build_wikisource_archive_inventory(
        config, session=session, progress=messages.append
    )
    first_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            config.inventory_path,
            config.page_hierarchy_path,
            config.composition_gate_path,
            config.inspection_sample_path,
            config.json_report_path,
            config.markdown_report_path,
        )
    }
    second = build_wikisource_archive_inventory(config, session=FakeSession())

    assert first == second
    assert first["main_namespace_page_count"] == 8
    assert first["work_root_count"] == 7
    assert first["candidate_work_root_count"] == 2
    assert first["decision_counts"] == {
        "conditioned_language_candidate": 1,
        "exclude_post_1900_scope": 1,
        "existing_project_reference": 1,
        "historical_core_metadata_candidate": 1,
        "hold_period_or_work_identity": 1,
        "hold_translation_edition_review": 1,
        "nineteenth_century_bridge_metadata_candidate": 1,
    }
    assert first["bounded_inspection"]["sample_size"] == 6
    assert first["bounded_inspection"]["primary_text_signal_pass_count"] == 6
    assert first["dump"]["full_page_text_dump_downloaded"] is False
    assert first["policy"]["corpus_text_activated"] is False
    assert len(_read_csv(config.page_hierarchy_path)) == 8
    inventory = {row["root_title"]: row for row in _read_csv(config.inventory_path)}
    assert inventory["Dialect Work"]["proposed_role"] == "conditioned_language_variant"
    assert inventory["Existing Work"]["existing_reference_ids"] == "existing"
    assert inventory["Old Work"]["hierarchy_page_count"] == "2"
    assert "dump-cache-hit" in " ".join(messages)
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            config.inventory_path,
            config.page_hierarchy_path,
            config.composition_gate_path,
            config.inspection_sample_path,
            config.json_report_path,
            config.markdown_report_path,
        )
    } == first_hashes
