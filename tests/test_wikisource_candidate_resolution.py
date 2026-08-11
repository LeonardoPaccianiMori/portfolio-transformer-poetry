import csv
import gzip
import hashlib
import json
from pathlib import Path

from sonnet_corpus.wikisource_candidate_resolution import (
    DUMP_BASE_URL,
    DUMP_DATE,
    WikisourceCandidateResolutionConfig,
    build_wikisource_candidate_resolution,
    classify_language_evidence,
    scan_title_language_signals,
)


INVENTORY_FIELDS = (
    "work_root_id",
    "root_title",
    "landing_page_url",
    "metadata_decision",
    "proposed_role",
    "author_evidence",
    "period_bucket",
    "language_route",
    "language_evidence",
    "genre_route",
    "form_route",
    "projected_wikitext_bytes",
    "hierarchy_page_count",
)


def _write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_gzip(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _candidate(number: int, title: str, *, size: int = 1000) -> dict[str, str]:
    return {
        "work_root_id": f"itws:{number}",
        "root_title": title,
        "landing_page_url": f"https://it.wikisource.org/wiki/{title}",
        "metadata_decision": "historical_core_metadata_candidate",
        "proposed_role": "historical_general",
        "author_evidence": "Autore Storico",
        "period_bucket": "origins_through_1800",
        "language_route": "standard_italian_unmarked_pending_page_check",
        "language_evidence": "",
        "genre_route": "general_or_unresolved",
        "form_route": "no_explicit_sonnet_signal",
        "projected_wikitext_bytes": str(size),
        "hierarchy_page_count": "1",
    }


def _noncandidate(
    number: int,
    title: str,
    *,
    decision: str,
    language_evidence: str,
) -> dict[str, str]:
    row = _candidate(number, title)
    row["metadata_decision"] = decision
    row["proposed_role"] = "metadata_hold"
    row["language_evidence"] = language_evidence
    return row


def _page_row(
    page_id: int,
    namespace: int,
    title: str,
    *,
    revision: int,
    redirect: int = 0,
) -> str:
    return (
        f"({page_id},{namespace},'{title}',{redirect},0,0.5,'20260801000000',"
        f"'20260801000000',{revision},500,'proofread-index',NULL)"
    )


def _fixture(tmp_path: Path) -> WikisourceCandidateResolutionConfig:
    metadata = tmp_path / "data/metadata"
    cache = tmp_path / "data/local/wikisource/archive_inventory_v1"
    reports = tmp_path / "reports"
    candidates = [
        _candidate(1, "Safe_Work", size=1000),
        _candidate(2, "No_Scan", size=2000),
        _candidate(3, "Multiple_Scans", size=3000),
        _candidate(4, "Dialect_Title", size=4000),
        _candidate(5, "Shared_Language_Hazard", size=5000),
        _candidate(6, "Missing_Index", size=6000),
        _candidate(7, "Redirected_Index", size=7000),
        _candidate(8, "Citation_Shared", size=8000),
    ]
    conditioned = _noncandidate(
        20,
        "Conditioned_Root",
        decision="conditioned_language_candidate",
        language_evidence="romanesco",
    )
    citation = _noncandidate(
        21,
        "Citation_Index_Root",
        decision="hold_language_variety_review",
        language_evidence="cui è citato Dante Alighieri | italiano",
    )
    inventory_rows = [*candidates, conditioned, citation]
    inventory_path = metadata / "inventory.csv"
    _write_csv(inventory_path, INVENTORY_FIELDS, inventory_rows)

    hierarchy_path = metadata / "hierarchy.csv"
    _write_csv(
        hierarchy_path,
        ("work_root_id", "page_id"),
        [
            {"work_root_id": row["work_root_id"], "page_id": index}
            for index, row in enumerate(inventory_rows, start=1)
        ],
    )

    namespace_filename = "fixture-namespaces.json.gz"
    namespace_payload = {
        "query": {
            "namespaces": {
                "110": {"id": 110, "canonical": "Index", "*": "Indice"}
            }
        }
    }
    namespace_path = cache / namespace_filename
    namespace_sha1 = _write_gzip(
        namespace_path,
        json.dumps(namespace_payload, ensure_ascii=False),
    )

    index_titles = [
        (101, "Safe_Scan.djvu", 1001, 0),
        (102, "Scan_A.djvu", 1002, 0),
        (103, "Scan_B.djvu", 1003, 0),
        (104, "Sonetti_romaneschi.djvu", 1004, 0),
        (105, "Shared_Scan.djvu", 1005, 0),
        (106, "Redirect_Scan.djvu", 1006, 1),
        (107, "Citation_Scan.djvu", 1007, 0),
    ]
    page_filename = "fixture-page.sql.gz"
    page_path = cache / page_filename
    page_sha1 = _write_gzip(
        page_path,
        "INSERT INTO `page` VALUES "
        + ",".join(
            _page_row(page_id, 110, title, revision=revision, redirect=redirect)
            for page_id, title, revision, redirect in index_titles
        )
        + ";\n",
    )

    target_titles = [
        "Safe_Scan.djvu",
        "Scan_A.djvu",
        "Scan_B.djvu",
        "Sonetti_romaneschi.djvu",
        "Shared_Scan.djvu",
        "Missing_Scan.djvu",
        "Redirect_Scan.djvu",
        "Citation_Scan.djvu",
    ]
    linktarget_filename = "fixture-linktarget.sql.gz"
    linktarget_path = cache / linktarget_filename
    linktarget_sha1 = _write_gzip(
        linktarget_path,
        "INSERT INTO `linktarget` VALUES "
        + ",".join(
            f"({index},110,'{title}')"
            for index, title in enumerate(target_titles, start=1)
        )
        + ";\n",
    )

    # Candidate root pages use IDs 1..8. The conditioned and citation-only
    # roots use IDs 9 and 10 and share scans with candidates 5 and 8.
    pagelinks = [
        (1, 1),
        (3, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (9, 5),
        (6, 6),
        (7, 7),
        (8, 8),
        (10, 8),
    ]
    pagelinks_filename = "fixture-pagelinks.sql.gz"
    pagelinks_path = cache / pagelinks_filename
    pagelinks_sha1 = _write_gzip(
        pagelinks_path,
        "INSERT INTO `pagelinks` VALUES "
        + ",".join(
            f"({page_id},0,{target_id})" for page_id, target_id in pagelinks
        )
        + ";\n",
    )

    return WikisourceCandidateResolutionConfig(
        repo_root=tmp_path,
        cache_dir=cache,
        inventory_path=inventory_path,
        hierarchy_path=hierarchy_path,
        resolution_path=metadata / "resolution.csv",
        scan_links_path=metadata / "scan_links.csv",
        review_path=metadata / "review.csv",
        json_report_path=reports / "report.json",
        markdown_report_path=reports / "report.md",
        dump_date=DUMP_DATE,
        dump_base_url=DUMP_BASE_URL,
        page_filename=page_filename,
        page_sha1=page_sha1,
        linktarget_filename=linktarget_filename,
        linktarget_sha1=linktarget_sha1,
        pagelinks_filename=pagelinks_filename,
        pagelinks_sha1=pagelinks_sha1,
        namespaces_filename=namespace_filename,
        namespaces_sha1=namespace_sha1,
        expected_inventory_rows=len(inventory_rows),
        expected_hierarchy_rows=len(inventory_rows),
        expected_candidate_rows=len(candidates),
        progress_interval=2,
    )


def test_language_evidence_separates_citations_standard_and_nonstandard():
    assert (
        classify_language_evidence("cui è citato Dante Alighieri | italiano")
        == "standard_italian_explicit"
    )
    assert (
        classify_language_evidence("cui è citato Dante Alighieri")
        == "citation_only_or_unmarked"
    )
    assert (
        classify_language_evidence("cui è citato Dante Alighieri | romanesco")
        == "nonstandard_or_unknown_language_evidence"
    )


def test_scan_title_language_signals_are_specific():
    assert scan_title_language_signals("Sonetti romaneschi III.djvu") == [
        "romanesco"
    ]
    assert scan_title_language_signals("Storia di Venezia.djvu") == []
    assert scan_title_language_signals("Opere di Francesco Petrarca.djvu") == []


def test_candidate_resolution_is_complete_conservative_and_deterministic(tmp_path):
    config = _fixture(tmp_path)
    messages = []

    first = build_wikisource_candidate_resolution(config, progress=messages.append)
    first_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            config.resolution_path,
            config.scan_links_path,
            config.review_path,
            config.json_report_path,
            config.markdown_report_path,
        )
    }
    second = build_wikisource_candidate_resolution(config)

    assert first == second
    assert first["candidate_count"] == 8
    assert first["direct_scan_linked_candidate_count"] == 7
    assert first["distinct_candidate_scan_count"] == 8
    assert first["eligible_page_level_audit_count"] == 2
    assert first["decision_counts"] == {
        "eligible_page_level_audit_queue": 2,
        "hold_missing_index_page": 1,
        "hold_multiple_source_scans": 1,
        "hold_no_direct_scan_link": 1,
        "hold_redirected_index_page": 1,
        "hold_scan_language_conflict": 2,
    }
    assert first["language_evidence_audit"] == {
        "hold_language_variety_row_count": 1,
        "citation_only_or_standard_row_count": 1,
        "citation_only_or_unmarked_row_count": 0,
        "standard_italian_explicit_row_count": 1,
        "nonstandard_or_unknown_row_count": 0,
        "policy": (
            "citation-index categories do not propagate language hazards; held rows "
            "are not promoted into the 4B candidate queue"
        ),
    }
    decisions = {
        row["root_title"]: row for row in _read_csv(config.resolution_path)
    }
    assert (
        decisions["Citation_Shared"]["checkpoint_4b_decision"]
        == "eligible_page_level_audit_queue"
    )
    assert (
        decisions["Shared_Language_Hazard"]["checkpoint_4b_decision"]
        == "hold_scan_language_conflict"
    )
    assert decisions["Citation_Shared"]["scan_group_nonstandard_hold_count"] == "0"
    assert decisions["Shared_Language_Hazard"]["scan_group_conditioned_count"] == "1"
    assert len(_read_csv(config.scan_links_path)) == 8
    assert len(_read_csv(config.review_path)) == 6
    assert first["policy"]["eligible_queue_authorizes_extraction"] is False
    assert first["dump"]["full_page_text_dump_downloaded"] is False
    assert "dump-cache-hit" in " ".join(messages)
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            config.resolution_path,
            config.scan_links_path,
            config.review_path,
            config.json_report_path,
            config.markdown_report_path,
        )
    } == first_hashes


def test_candidate_resolution_rejects_a_tampered_cached_dump(tmp_path):
    config = _fixture(tmp_path)
    (config.cache_dir / config.pagelinks_filename).write_bytes(b"tampered")

    try:
        build_wikisource_candidate_resolution(config)
    except ValueError as error:
        assert "cached dump hash mismatch" in str(error)
    else:
        raise AssertionError("expected a cached-dump hash failure")
