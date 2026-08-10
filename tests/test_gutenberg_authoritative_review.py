import csv
import hashlib
import json
from pathlib import Path

import pytest

from sonnet_corpus.gutenberg_authoritative_review import (
    FINAL_FIELDS,
    GutenbergAuthoritativeReviewConfig,
    extract_sbn_detail,
    extract_wikidata_p577,
    load_or_fetch_authority_json,
    resolve_authoritative_row,
    run_gutenberg_authoritative_review,
    select_sbn_candidate,
    select_wikidata_candidate,
)
from sonnet_corpus.gutenberg_metadata_review import RESOLUTION_FIELDS


def _row(
    ebook_id: str = "1",
    *,
    title: str = "Libro di prova: Romanzo",
    authors: str = "Autore, Prova",
    inventory_status: str = "review_work_publication_date",
    resolution_status: str = "manual_review",
    decision: str = "manual_authoritative_review_required",
    period: str = "",
    role: str = "date_and_role_review",
) -> dict[str, str]:
    row = {field: "" for field in RESOLUTION_FIELDS}
    row.update(
        {
            "ebook_id": ebook_id,
            "title": title,
            "authors": authors,
            "preliminary_role": role,
            "inventory_status": inventory_status,
            "landing_page_url": f"https://www.gutenberg.org/ebooks/{ebook_id}",
            "plain_text_url": f"https://www.gutenberg.org/ebooks/{ebook_id}.txt.utf-8",
            "automatic_decision": decision,
            "resolution_status": resolution_status,
            "resolved_period_bucket": period,
            "resolved_role": (
                "historical_general_candidate"
                if period == "origins_through_1800"
                else ""
            ),
        }
    )
    return row


def _sbn_result(
    *,
    bid: str,
    title: str = "Libro di prova / Prova Autore",
    author: str = "Autore, Prova <1850-1910>",
    publication: str = "Milano : Editore, 1897",
    index: int = 1,
) -> dict:
    return {
        "id": f"ITICCU{bid}",
        "index": index,
        "title": {"text": author, "info": title},
        "infos": [publication, f"Testo - Monografia [IT\\ICCU\\{bid}]"],
    }


def _sbn_search(*results: dict) -> dict:
    return {
        "status": "success",
        "data": {"total": len(results), "results": list(results)},
    }


def _table_row(label: str, value: str) -> list[dict]:
    return [
        {"type": "table-title", "value": label},
        {"type": "table-text", "value": value},
    ]


def _sbn_detail(
    *,
    bid: str = "ABC0000001",
    title: str = "Libro di prova / Prova Autore",
    author: str = "Autore, Prova <1850-1910>",
    publication: str = "Milano : Editore, 1897",
) -> dict:
    return {
        "status": "success",
        "data": {
            "results": [
                {
                    "id": f"ITICCU{bid}",
                    "title": title,
                    "pretitle": author,
                    "contents": [
                        {
                            "type": "table",
                            "body": [
                                _table_row("Autore principale", author),
                                _table_row("Titolo", title),
                                _table_row("Pubblicazione", publication),
                            ],
                        }
                    ],
                }
            ]
        },
    }


def _evidence(
    method: str,
    *,
    source: str,
    year: int | None = None,
    record_id: str = "record",
) -> dict:
    return {
        "source": source,
        "record_id": record_id,
        "url": "https://example.test/evidence",
        "retrieved_at": "2026-08-10T00:00:00+00:00",
        "payload_sha256": "a" * 64,
        "method": method,
        "year_start": year,
        "year_end": year,
        "text": "direct evidence",
        "confidence": "high",
        "decisive": year is not None,
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> str:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESOLUTION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_authority_cache_is_request_pinned_and_payload_hashed(tmp_path):
    cache = tmp_path / "cache.json"
    calls = []

    def fetch(method, url, params, timeout):
        calls.append((method, url, params, timeout))
        return {"status": "success", "data": {"value": 1}}

    first = load_or_fetch_authority_json(
        cache,
        source="test",
        method="GET",
        url="https://example.test/api",
        params={"q": "one"},
        timeout_seconds=10,
        fetch_json=fetch,
        retrieved_at="2026-08-10T00:00:00+00:00",
    )
    second = load_or_fetch_authority_json(
        cache,
        source="test",
        method="GET",
        url="https://example.test/api",
        params={"q": "one"},
        timeout_seconds=10,
        fetch_json=fetch,
        retrieved_at="later",
    )

    assert first[1] == "downloaded"
    assert second[1] == "hit"
    assert len(calls) == 1
    with pytest.raises(ValueError, match="request pin mismatch"):
        load_or_fetch_authority_json(
            cache,
            source="test",
            method="GET",
            url="https://example.test/api",
            params={"q": "two"},
            timeout_seconds=10,
            fetch_json=fetch,
            retrieved_at="later",
        )

    envelope = json.loads(cache.read_text(encoding="utf-8"))
    envelope["payload"]["data"]["value"] = 2
    cache.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        load_or_fetch_authority_json(
            cache,
            source="test",
            method="GET",
            url="https://example.test/api",
            params={"q": "one"},
            timeout_seconds=10,
            fetch_json=fetch,
            retrieved_at="later",
        )


def test_sbn_selection_requires_title_and_author_and_uses_earliest_date():
    row = _row()
    payload = _sbn_search(
        _sbn_result(
            bid="BAD0000001",
            title="Altro libro / Altro Autore",
            author="Autore, Prova",
            publication="Roma : Editore, 1700",
            index=1,
        ),
        _sbn_result(
            bid="BAD0000002",
            title="Libro di prova / Persona Diversa",
            author="Persona, Diversa",
            publication="Roma : Editore, 1800",
            index=2,
        ),
        _sbn_result(
            bid="OK00000002",
            publication="Roma : Editore, 1920",
            index=3,
        ),
        _sbn_result(
            bid="OK00000001",
            publication="Milano : Editore, 1897",
            index=4,
        ),
    )

    selected, audit = select_sbn_candidate(row, payload)

    assert selected["bid"] == "OK00000001"
    assert selected["year_start"] == 1897
    assert audit["sbn_direct_match_count"] == 2


def test_sbn_undated_match_is_opened_for_record_level_metadata():
    row = _row()
    selected, _ = select_sbn_candidate(
        row,
        _sbn_search(
            _sbn_result(
                bid="ABC0000001",
                publication="Milano : Editore, [18..]",
            )
        ),
    )

    assert selected["bid"] == "ABC0000001"
    assert selected["year_start"] is None
    detail = extract_sbn_detail(_sbn_detail(publication="Milano : Editore, 1897"))
    assert detail["year_start"] == 1897
    assert detail["permalink"] == "https://opac.sbn.it/bid/ABC0000001"


def test_wikidata_match_rejects_an_edition_and_extracts_p577():
    row = _row(title="Sei personaggi in cerca d'autore", authors="Pirandello, Luigi")
    search = {
        "search": [
            {
                "id": "QEDITION",
                "label": "Sei personaggi in cerca d'autore",
                "description": "Italian edition of Luigi Pirandello play",
                "match": {"text": "Sei personaggi in cerca d'autore"},
            },
            {
                "id": "QWORK",
                "label": "Sei personaggi in cerca d'autore",
                "description": "opera teatrale di Luigi Pirandello",
                "match": {"text": "Sei personaggi in cerca d'autore"},
            },
        ]
    }

    candidate = select_wikidata_candidate(row, search)
    dates = extract_wikidata_p577(
        {
            "entities": {
                "QWORK": {
                    "claims": {
                        "P577": [
                            {
                                "mainsnak": {
                                    "datavalue": {"value": {"time": "+1921-01-01T00:00:00Z"}}
                                }
                            }
                        ]
                    }
                }
            }
        },
        "QWORK",
    )

    assert candidate["id"] == "QWORK"
    assert dates["claim_years"] == [1921]
    assert dates["claim_period_conflict"] is False


def test_conflicting_sbn_and_wikidata_dates_are_documented_exclusion():
    row = _row()
    evidence = [
        _evidence("sbn_edition_publication_upper_bound", source="sbn_iccu", year=1897),
        _evidence("wikidata_p577_title_author_match", source="wikidata", year=1921),
    ]

    result = resolve_authoritative_row(
        row,
        evidence=evidence,
        sbn_detail={"year_start": 1897, "year_end": 1897},
        wikidata_detail={
            "year_start": 1921,
            "year_end": 1921,
            "claim_years": [1921],
            "claim_period_conflict": False,
        },
        cache_sha256s=["b" * 64],
    )

    assert result["final_decision"] == "exclude_unresolved_authoritative_metadata"
    assert result["final_exclusion_reason"] == (
        "sbn_edition_predates_wikidata_work_date_conflict"
    )


def test_post_1900_one_year_catalog_disagreement_is_still_post_1900():
    row = _row()
    evidence = [
        _evidence("sbn_edition_publication_upper_bound", source="sbn_iccu", year=1909),
        _evidence("wikidata_p577_title_author_match", source="wikidata", year=1910),
    ]

    result = resolve_authoritative_row(
        row,
        evidence=evidence,
        sbn_detail={"year_start": 1909, "year_end": 1909},
        wikidata_detail={
            "year_start": 1910,
            "year_end": 1910,
            "claim_years": [1910],
            "claim_period_conflict": False,
        },
        cache_sha256s=[],
    )

    assert result["final_decision"] == "exclude_post_1900_original_text"
    assert result["final_exclusion_reason"] == "authoritative_date_after_1900"


def test_pre_1901_primary_text_title_page_is_conservative_bridge_evidence():
    row = _row()
    row["date_evidence_json"] = json.dumps(
        [
            {
                "kind": "title_page_edition_year",
                "year_start": 1896,
                "year_end": 1896,
                "text": "NAPOLI | EDITORE | 1896",
            }
        ]
    )

    result = resolve_authoritative_row(
        row,
        evidence=[],
        sbn_detail=None,
        wikidata_detail=None,
        cache_sha256s=[],
    )

    assert result["final_decision"] == "eligible_nineteenth_century_candidate"
    assert result["authoritative_method"] == (
        "project_gutenberg_title_page_edition_upper_bound"
    )


@pytest.mark.parametrize(
    ("ebook_id", "role", "expected_decision", "expected_method"),
    [
        (
            "22025",
            "date_and_role_review",
            "eligible_nineteenth_century_candidate",
            "project_gutenberg_dated_primary_documents",
        ),
        (
            "22502",
            "date_and_role_review",
            "eligible_historical_core_candidate",
            "project_gutenberg_dated_anthology_contents",
        ),
        (
            "28869",
            "date_and_role_review",
            "eligible_nineteenth_century_candidate",
            "project_gutenberg_dated_anthology_contents",
        ),
        (
            "30738",
            "sonnet_specialization_candidate",
            "eligible_historical_core_candidate",
            "project_gutenberg_primary_text_edition_note",
        ),
        (
            "31285",
            "historical_non_sonnet_poetry_candidate",
            "eligible_historical_core_candidate",
            "project_gutenberg_primary_text_edition_note",
        ),
        (
            "31818",
            "historical_non_sonnet_poetry_candidate",
            "eligible_historical_core_candidate",
            "project_gutenberg_primary_text_edition_note",
        ),
        (
            "32599",
            "date_and_role_review",
            "eligible_nineteenth_century_candidate",
            "project_gutenberg_primary_text_delivery_date",
        ),
        (
            "37776",
            "date_and_role_review",
            "eligible_nineteenth_century_candidate",
            "project_gutenberg_primary_text_title_page",
        ),
        (
            "37936",
            "date_and_role_review",
            "eligible_nineteenth_century_candidate",
            "project_gutenberg_primary_text_first_performance",
        ),
        (
            "38216",
            "date_and_role_review",
            "eligible_nineteenth_century_candidate",
            "project_gutenberg_primary_text_first_performance",
        ),
        (
            "38218",
            "date_and_role_review",
            "eligible_nineteenth_century_candidate",
            "project_gutenberg_primary_text_first_performance",
        ),
        (
            "39239",
            "historical_non_sonnet_poetry_candidate",
            "eligible_historical_core_candidate",
            "project_gutenberg_primary_text_collection_period",
        ),
        (
            "54070",
            "date_and_role_review",
            "eligible_historical_core_candidate",
            "project_gutenberg_primary_text_title_page",
        ),
        (
            "54167",
            "date_and_role_review",
            "eligible_historical_core_candidate",
            "project_gutenberg_italian_edition_title_page",
        ),
        (
            "60249",
            "date_and_role_review",
            "eligible_nineteenth_century_candidate",
            "project_gutenberg_primary_text_delivery_date",
        ),
        (
            "63106",
            "date_and_role_review",
            "eligible_nineteenth_century_candidate",
            "project_gutenberg_dated_anthology_contents",
        ),
        (
            "64393",
            "date_and_role_review",
            "eligible_historical_core_candidate",
            "project_gutenberg_primary_text_title_page",
        ),
    ],
)
def test_record_specific_primary_text_dates_resolve_clear_false_negatives(
    ebook_id, role, expected_decision, expected_method
):
    row = _row(ebook_id=ebook_id, role=role)

    result = resolve_authoritative_row(
        row,
        evidence=[],
        sbn_detail=None,
        wikidata_detail=None,
        cache_sha256s=[],
    )

    assert result["final_decision"] == expected_decision
    assert result["authoritative_method"] == expected_method
    evidence = json.loads(result["authoritative_evidence_json"])
    assert evidence[-1]["source"] == "project_gutenberg_primary_text"
    assert evidence[-1]["decisive"] is True


def test_sbn_publication_range_uses_earliest_date_as_existence_upper_bound():
    row = _row()
    evidence = [
        {
            **_evidence(
                "sbn_edition_publication_upper_bound",
                source="sbn_iccu",
                year=1891,
            ),
            "year_end": 1910,
        }
    ]

    result = resolve_authoritative_row(
        row,
        evidence=evidence,
        sbn_detail={"year_start": 1891, "year_end": 1910},
        wikidata_detail=None,
        cache_sha256s=[],
    )

    assert result["final_decision"] == "eligible_nineteenth_century_candidate"


def test_translation_uses_confirmed_italian_edition_date():
    row = _row(inventory_status="review_translation_edition_date")
    row["date_evidence_json"] = json.dumps(
        [
            {
                "kind": "title_page_edition_year",
                "year_start": 1911,
                "year_end": 1911,
            }
        ]
    )
    evidence = [
        _evidence("sbn_italian_edition_publication", source="sbn_iccu", year=1911)
    ]

    result = resolve_authoritative_row(
        row,
        evidence=evidence,
        sbn_detail={"year_start": 1911, "year_end": 1911},
        wikidata_detail=None,
        cache_sha256s=[],
    )

    assert result["final_decision"] == "exclude_post_1900_original_text"
    assert result["authoritative_method"] == (
        "italian_translation_title_page_confirmed_by_sbn"
    )
    assert result["authoritative_evidence_year_start"] == 1911


@pytest.mark.parametrize(
    ("ebook_id", "expected_decision", "expected_class"),
    [
        ("34734", "route_conditioned_romanesco_sonnets", "conditioned_probe"),
        (
            "48542",
            "route_conditioned_bolognese_prose_and_drama",
            "conditioned_probe",
        ),
        ("49523", "eligible_nineteenth_century_candidate", "eligible_probe"),
    ],
)
def test_language_variety_rows_follow_explicit_routes(
    ebook_id, expected_decision, expected_class
):
    row = _row(
        ebook_id=ebook_id,
        inventory_status="review_language_variety_before_download",
        role="language_variety_review_required",
    )

    result = resolve_authoritative_row(
        row,
        evidence=[],
        sbn_detail=None,
        wikidata_detail=None,
        cache_sha256s=[],
    )

    assert result["final_decision"] == expected_decision
    assert result["final_activation_class"] == expected_class
    evidence = json.loads(result["authoritative_evidence_json"])
    assert evidence[-1]["url"] == row["landing_page_url"]
    assert evidence[-1]["decisive"] is True


def test_run_reconciles_pass_1a_and_pass_1b_without_activation(tmp_path):
    resolved = _row(
        "1",
        title="Testo antico",
        resolution_status="automatic_resolved",
        decision="eligible_historical_core_candidate",
        period="origins_through_1800",
    )
    manual = _row("2")
    pass_1a_path = tmp_path / "pass1a.csv"
    manual_path = tmp_path / "manual.csv"
    pass_1a_sha = _write_csv(pass_1a_path, [resolved, manual])
    manual_sha = _write_csv(manual_path, [manual])
    config = GutenbergAuthoritativeReviewConfig(
        repo_root=tmp_path,
        pass_1a_csv_path=pass_1a_path,
        manual_csv_path=manual_path,
        cache_dir=tmp_path / "cache",
        final_csv_path=tmp_path / "final.csv",
        exclusion_csv_path=tmp_path / "exclusions.csv",
        json_report_path=tmp_path / "report.json",
        markdown_report_path=tmp_path / "report.md",
        expected_pass_1a_sha256=pass_1a_sha,
        expected_manual_sha256=manual_sha,
        expected_pass_1a_count=2,
        expected_manual_count=1,
        request_delay_seconds=0,
    )

    def fetch(method, url, params, timeout):
        if "titles-search" in url:
            return _sbn_search(_sbn_result(bid="ABC0000001"))
        if url.endswith("/title"):
            return _sbn_detail()
        if "w/api.php" in url:
            return {"search": []}
        raise AssertionError((method, url, params, timeout))

    report = run_gutenberg_authoritative_review(config, fetch_json=fetch)

    assert report["final_record_count"] == 2
    assert report["policy"]["activation_authorized"] is False
    with config.final_csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        assert tuple(rows[0]) == FINAL_FIELDS
    assert rows[0]["resolution_pass"] == "pass_1a"
    assert rows[1]["resolution_pass"] == "pass_1b"
    assert rows[1]["final_decision"] == "eligible_nineteenth_century_candidate"
    assert sum(report["final_decision_counts"].values()) == 2
