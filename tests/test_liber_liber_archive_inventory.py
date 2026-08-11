import csv
import hashlib
import json
from pathlib import Path

import requests

from sonnet_corpus.liber_liber_archive_inventory import (
    BOOK_LICENSE_PAGE_ID,
    LiberLiberArchiveInventoryConfig,
    build_liber_liber_archive_inventory,
    fetch_liber_liber_pages,
    parse_liber_liber_work_page,
)


class FakeResponse:
    def __init__(self, payload, *, total, total_pages):
        self._payload = payload
        self.headers = {"X-WP-Total": str(total), "X-WP-TotalPages": str(total_pages)}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FailingResponse(FakeResponse):
    def raise_for_status(self):
        raise requests.HTTPError("temporary server error")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append((url, params, timeout))
        return self.responses.pop(0)


def _content(
    *,
    title="Opera antica",
    author="Autore antico",
    dewey="Narrativa italiana (1375-1492)",
    license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
    marker="free",
    translator="",
    formats=("txt", "odt"),
    description="",
    bisac="",
    reference_edition="Edizione 1880",
    editor="",
):
    links = "".join(
        f'<a href="https://liberliber.it/opere/download/?op=10&type=opera_url_{value}">'
        f'<img class="ll_ebook_{value}_{marker}"></a>'
        for value in formats
    )
    translation = (
        f'<div class="ll_metadati_etichetta">traduzione:</div>'
        f'<div class="ll_metadati_dato">{translator}</div>'
        if translator else ""
    )
    return f"""
    <div class="ll_opera_download_scarica">Scarica gratis</div>{links}
    <div class="ll_metadati">
      <div class="ll_metadati_etichetta">titolo:</div><div class="ll_metadati_dato">{title}</div>
      <div class="ll_metadati_etichetta">autore:</div><div class="ll_metadati_dato"><a href="https://example.test/author/">{author}</a></div>
      <div class="ll_metadati_etichetta">opera di riferimento:</div><div class="ll_metadati_dato">{reference_edition}</div>
      <div class="ll_metadati_etichetta">cura:</div><div class="ll_metadati_dato">{editor}</div>
      {translation}
      <div class="ll_metadati_etichetta">descrizione breve:</div><div class="ll_metadati_dato">{description}</div>
      <div class="ll_metadati_etichetta">descrittore Dewey:</div><div class="ll_metadati_dato">{dewey}</div>
      <div class="ll_metadati_etichetta">soggetto BISAC:</div><div class="ll_metadati_dato">{bisac}</div>
      <div class="ll_metadati_etichetta">digitalizzazione:</div><div class="ll_metadati_dato">Ada Rossi, ada@example.test</div>
      <div class="ll_metadati_etichetta">licenza:</div><div class="ll_metadati_dato"><a href="{license_url}">CC BY-NC-SA 4.0</a></div>
    </div>
    """


def _page(page_id, content, *, link=None, parent=1, modified="2026-08-11T00:00:00"):
    return {
        "id": page_id,
        "parent": parent,
        "slug": f"page-{page_id}",
        "link": link or f"https://liberliber.it/autori/a/autore/opera-{page_id}/",
        "title": {"rendered": f"Page {page_id}"},
        "modified": modified,
        "content": {"rendered": content, "protected": False},
        "excerpt": {"rendered": "", "protected": False},
    }


def _license_page():
    return _page(
        BOOK_LICENSE_PAGE_ID,
        "<p>Creative Commons Attribuzione - Non commerciale - Condividi allo stesso modo 4.0. "
        "E-book liberi. Autori, traduttori e curatori deceduti da oltre 70 anni.</p>",
        link="https://liberliber.it/opere/libri/licenze/",
        parent=23,
        modified="2021-02-25T17:40:34",
    )


def _author_page(page_id=1, *, name="Autore antico", birth=1400, death=1470):
    content = (
        f"<p>{name} nacque nel {birth} e morì nel {death}.</p>"
        f'<div class="ll_metadati_etichetta">autore:</div>'
        f'<div class="ll_metadati_dato">{name}</div>'
    )
    return _page(
        page_id, content, link=f"https://liberliber.it/autori/a/{page_id}/", parent=687,
    )


def _config(tmp_path: Path) -> LiberLiberArchiveInventoryConfig:
    metadata = tmp_path / "metadata"
    processed = tmp_path / "processed"
    metadata.mkdir(); processed.mkdir()
    with (metadata / "sources.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source_id", "source_archive", "landing_page_url"))
        writer.writeheader()
        writer.writerow({
            "source_id": "existing", "source_archive": "Liber Liber",
            "landing_page_url": "https://liberliber.it/autori/a/autore/opera-30/",
        })
    (metadata / "prior.json").write_text(json.dumps({"results": [
        {"status": "ok", "cleaned_character_count": 1000},
        {"status": "ok", "cleaned_character_count": 3000},
    ]}), encoding="utf-8")
    (processed / "bibit.json").write_text(json.dumps({"record_characters_by_role": {"historical_general": 10000}}), encoding="utf-8")
    (processed / "gutenberg.json").write_text(json.dumps({"record_characters_by_role": {"historical_general": 20000, "conditioned_source_variants": 999}}), encoding="utf-8")
    (processed / "wikisource.json").write_text(json.dumps({"materialized_broader_character_count": 30000}), encoding="utf-8")
    return LiberLiberArchiveInventoryConfig(
        repo_root=tmp_path,
        local_cache_path=tmp_path / "local/pages.json",
        inventory_path=metadata / "inventory.csv",
        rights_path=metadata / "rights.csv",
        composition_gate_path=metadata / "gate.csv",
        json_report_path=tmp_path / "reports/report.json",
        markdown_report_path=tmp_path / "reports/report.md",
        broader_sources_manifest_path=metadata / "sources.csv",
        prior_probe_report_path=metadata / "prior.json",
        bibit_build_report_path=processed / "bibit.json",
        gutenberg_build_report_path=processed / "gutenberg.json",
        wikisource_build_report_path=processed / "wikisource.json",
        request_delay_seconds=0,
        per_page=2,
    )


def _snapshot(pages):
    return {
        "inventory_version": "liber_liber_archive_inventory_v1",
        "api_url": "https://liberliber.it/wp-json/wp/v2/pages",
        "api_fields": "test",
        "fetched_at_utc": "2026-08-11T00:00:00+00:00",
        "total_records": len(pages),
        "total_pages": 1,
        "pages": sorted(pages, key=lambda row: row["id"]),
    }


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fetches_complete_paginated_catalog_then_reuses_cache(tmp_path):
    config = _config(tmp_path)
    pages = [_license_page(), _page(10, _content()), _page(20, _content())]
    session = FakeSession([
        FakeResponse(pages[:2], total=3, total_pages=2),
        FakeResponse(pages[2:], total=3, total_pages=2),
    ])

    fetched = fetch_liber_liber_pages(config, session=session, sleep=lambda _value: None)
    cached = fetch_liber_liber_pages(config, session=FakeSession([]))

    assert fetched == cached
    assert [row["id"] for row in fetched["pages"]] == sorted(row["id"] for row in pages)
    assert len(session.calls) == 2


def test_fetch_resumes_an_atomic_partial_cache(tmp_path):
    config = _config(tmp_path)
    pages = [_license_page(), _page(10, _content()), _page(20, _content())]
    partial = _snapshot(pages[:2]) | {
        "total_records": 3, "total_pages": 2, "per_page": 2, "complete": False,
    }
    config.local_cache_path.parent.mkdir(parents=True)
    config.local_cache_path.write_text(json.dumps(partial), encoding="utf-8")
    session = FakeSession([FakeResponse(pages[2:], total=3, total_pages=2)])

    fetched = fetch_liber_liber_pages(config, session=session, sleep=lambda _value: None)

    assert fetched["complete"] is True
    assert [row["id"] for row in fetched["pages"]] == sorted(row["id"] for row in pages)
    assert session.calls[0][1]["page"] == 2


def test_fetch_splits_one_server_failing_page_and_preserves_accounting(tmp_path):
    config = _config(tmp_path)
    pages = [_license_page(), _page(10, _content()), _page(20, _content())]
    session = FakeSession([
        FailingResponse([], total=3, total_pages=2),
        FailingResponse([], total=3, total_pages=2),
        FakeResponse(pages[:1], total=3, total_pages=3),
        FakeResponse(pages[1:2], total=3, total_pages=3),
        FakeResponse(pages[2:], total=3, total_pages=2),
    ])

    fetched = fetch_liber_liber_pages(config, session=session, sleep=lambda _value: None)

    assert fetched["total_records"] == 3
    assert len(fetched["pages"]) == 3
    assert any("offset" in call[1] for call in session.calls)


def test_fetch_records_basic_metadata_when_one_content_page_always_fails(tmp_path):
    config = _config(tmp_path)
    config = LiberLiberArchiveInventoryConfig(
        **{**config.__dict__, "per_page": 1}
    )
    basic = _license_page()
    basic.pop("content"); basic.pop("excerpt")
    failures = [FailingResponse([], total=1, total_pages=1) for _ in range(4)]
    session = FakeSession([*failures, FakeResponse([basic], total=1, total_pages=1)])

    fetched = fetch_liber_liber_pages(config, session=session, sleep=lambda _value: None)

    assert fetched["pages"][0]["content_fetch_error"] == "HTTPError"
    assert fetched["pages"][0]["content"]["rendered"] == ""


def test_work_parser_preserves_provenance_and_removes_credit_email():
    row = parse_liber_liber_work_page(
        _page(10, _content(
            reference_edition="Edizione di Ada Rossi, ada@example.test",
            editor="Ada Rossi, ada@example.test",
        )),
        author_page=_author_page(),
    )

    assert row is not None
    assert row["title"] == "Opera antica"
    assert row["author"] == "Autore antico"
    assert row["site_copyright_route"] == "site_marked_free"
    assert row["supported_primary_text_formats"] == "odt;txt_zip"
    assert row["license_url"] == "https://creativecommons.org/licenses/by-nc-sa/4.0/"
    assert row["digitization_credit"] == "Ada Rossi"
    assert row["reference_edition"] == "Edizione di Ada Rossi"
    assert row["editor"] == "Ada Rossi"
    assert row["author_biography_years"] == "1400;1470"


def test_inventory_routes_eligible_protected_translation_and_existing_rows(tmp_path):
    config = _config(tmp_path)
    pages = [
        _license_page(),
        _author_page(),
        _author_page(2, name="Studioso moderno", birth=1866, death=1952),
        _page(10, _content()),
        _page(20, _content(marker="prot", dewey="Narrativa italiana (1900-1999)")),
        _page(30, _content()),
        _page(40, _content(translator="Traduttore moderno")),
        _page(50, _content(author="Studioso moderno", dewey="Poesia italiana (origini-1375)"), parent=2),
    ]
    config.local_cache_path.parent.mkdir(parents=True)
    config.local_cache_path.write_text(json.dumps(_snapshot(pages)), encoding="utf-8")

    report = build_liber_liber_archive_inventory(config)
    rows = list(csv.DictReader(config.inventory_path.open(encoding="utf-8")))
    decisions = {row["record_id"]: row["composition_decision"] for row in rows}

    assert report["book_work_count"] == 5
    assert decisions["ll:10"] == "eligible_fulltext_probe_inactive"
    assert decisions["ll:20"] == "exclude_personal_use_only_protected_text"
    assert decisions["ll:30"] == "existing_project_corpus_reference"
    assert decisions["ll:40"] == "hold_translation_edition_review"
    assert decisions["ll:50"] == "hold_work_period_review"
    assert report["projection"]["prior_probe_median_characters"] == 2000
    assert report["projection"]["current_frozen_archive_characters"] == 60000
    assert report["policy"]["fulltext_acquired"] is False


def test_inventory_holds_conflicting_prose_verse_and_drama_form_metadata(tmp_path):
    config = _config(tmp_path)
    pages = [
        _license_page(),
        _author_page(),
        _page(10, _content(
            title="Memorie del poeta",
            description="Biografia con riferimenti alle opere poetiche.",
            bisac="POESIA / Generale",
        )),
        _page(20, _content(
            title="Racconti umoristici",
            description="Racconti, dramma teatrale e versi liberi.",
            bisac="ARTI RAPPRESENTATIVE / Teatro",
        )),
    ]
    config.local_cache_path.parent.mkdir(parents=True)
    config.local_cache_path.write_text(json.dumps(_snapshot(pages)), encoding="utf-8")

    build_liber_liber_archive_inventory(config)
    rows = list(csv.DictReader(config.inventory_path.open(encoding="utf-8")))

    assert {row["genre_route"] for row in rows} == {"mixed_form_review"}
    assert {row["composition_decision"] for row in rows} == {
        "hold_drama_prose_verse_review"
    }


def test_cache_backed_inventory_outputs_are_deterministic(tmp_path):
    config = _config(tmp_path)
    pages = [_license_page(), _page(10, _content())]
    config.local_cache_path.parent.mkdir(parents=True)
    config.local_cache_path.write_text(json.dumps(_snapshot(pages)), encoding="utf-8")

    build_liber_liber_archive_inventory(config)
    first = [_sha(path) for path in (
        config.inventory_path, config.rights_path, config.composition_gate_path,
        config.json_report_path, config.markdown_report_path,
    )]
    build_liber_liber_archive_inventory(config)
    second = [_sha(path) for path in (
        config.inventory_path, config.rights_path, config.composition_gate_path,
        config.json_report_path, config.markdown_report_path,
    )]

    assert first == second
