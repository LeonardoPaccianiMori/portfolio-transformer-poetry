import csv
import hashlib
import json
from pathlib import Path

import pytest

from sonnet_corpus.archive_discovery import (
    DECISION_FIELDS,
    EVIDENCE_FIELDS,
    EVIDENCE_SPECS,
    QUERY_FIELDS,
    QUERY_SPECS,
    REGISTRY_FIELDS,
    ArchiveDiscoveryConfig,
    build_archive_discovery,
    parse_query_result_count,
)


def _config(tmp_path: Path) -> ArchiveDiscoveryConfig:
    metadata = tmp_path / "data/metadata"
    registry = metadata / "corpus_archive_expansion_registry.csv"
    metadata.mkdir(parents=True)
    with registry.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow({
            "archive_id": "existing", "archive_name": "Existing Archive",
            "landing_page": "https://example.test/", "corpus_roles": "general historical",
            "coverage": "complete", "license_or_reuse_status": "permitted",
            "bulk_access": "api", "status": "processed_build_complete",
            "next_action": "canonicalize", "notes": "preserve",
        })
    return ArchiveDiscoveryConfig(
        repo_root=tmp_path,
        registry_path=registry,
        cache_dir=tmp_path / "data/local/archive_discovery_v1",
        query_path=metadata / "corpus_archive_discovery_queries_v1.csv",
        evidence_path=metadata / "corpus_archive_discovery_evidence_v1.csv",
        decision_path=metadata / "corpus_archive_discovery_decisions_v1.csv",
        json_report_path=tmp_path / "reports/corpus_archive_discovery_v1.json",
        markdown_report_path=tmp_path / "reports/corpus_archive_discovery_v1.md",
        request_delay_seconds=0,
    )


def _query_rows() -> list[dict[str, str]]:
    return [{
        "query_id": spec.query_id,
        "surface_id": spec.surface_id,
        "authority": spec.authority,
        "query_text": spec.query_text,
        "endpoint_url": spec.endpoint_url,
        "result_boundary": spec.result_boundary,
        "response_format": spec.response_format,
        "purpose": spec.purpose,
        "retrieval_date": "2026-08-11",
        "http_status": "200",
        "content_type": "application/json",
        "content_sha256": hashlib.sha256(spec.query_id.encode()).hexdigest(),
        "result_count": "0",
        "verification_status": "official_or_curated_metadata_response_verified",
    } for spec in QUERY_SPECS]


def _evidence_rows() -> list[dict[str, str]]:
    return [{
        "evidence_id": spec.evidence_id,
        "candidate_id": spec.candidate_id,
        "evidence_type": spec.evidence_type,
        "authority": "official_first_party",
        "source_url": spec.url,
        "resolved_url": spec.url,
        "retrieval_date": "2026-08-11",
        "http_status": "200",
        "content_type": "application/json",
        "content_sha256": hashlib.sha256(spec.evidence_id.encode()).hexdigest(),
        "evidence_quote": spec.quote,
        "supports_decision": spec.supports,
        "limitation": spec.limitation,
        "verification_status": "content_needles_verified",
    } for spec in EVIDENCE_SPECS]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("parser", "content", "expected"),
    [
        ("re3data", b"<list><repository/><repository/></list>", 2),
        ("zenodo", json.dumps({"hits": {"total": {"value": 7}}}).encode(), 7),
        ("github", json.dumps({"total_count": 9}).encode(), 9),
        ("dspace7", json.dumps({"_embedded": {"searchResult": {"page": {"totalElements": 11}}}}).encode(), 11),
        ("solr", json.dumps({"response": {"numFound": 13}}).encode(), 13),
        ("ckan", json.dumps({"result": {"count": 15}}).encode(), 15),
        ("oai_sets", b'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListSets><set/><set/></ListSets></OAI-PMH>', 2),
        ("ota_handles", b'<a href="/llds/xmlui/handle/20.500.14106/A1">one</a><a href="/llds/xmlui/handle/20.500.14106/A1">again</a><a href="/llds/xmlui/handle/20.500.14106/A2">two</a>', 2),
        ("clarin_family", b'<meta name="description" content="Corpora in French and Swedish." />', 0),
    ],
)
def test_query_result_count_parsers(parser, content, expected):
    assert parse_query_result_count(parser, content) == expected


def test_build_reconciles_materiality_roles_and_inactive_scope(tmp_path):
    config = _config(tmp_path)
    report = build_archive_discovery(
        config, query_rows=_query_rows(), evidence_rows=_evidence_rows(),
    )
    decisions = _rows(config.decision_path)

    assert tuple(decisions[0]) == DECISION_FIELDS
    assert report["query_count"] == len(QUERY_SPECS)
    assert report["evidence_count"] == len(EVIDENCE_SPECS)
    assert report["candidate_decision_count"] == len(decisions) == 16
    assert report["eligible_standard_audit_count"] == 4
    assert report["conditioned_auxiliary_count"] == 1
    assert report["materiality_hold_count"] == 0
    assert report["registry_addition_count"] == 2
    assert report["activated_corpus_characters"] == 0
    assert report["corpus_text_acquired"] is False
    assert report["text_activated"] is False
    assert report["v7_created"] is False
    assert report["gpu_work_started"] is False
    assert all(row["activation_status"].startswith("inactive_") for row in decisions)
    assert {row["composition_decision"] for row in decisions} <= {"core_training", "auxiliary", "excluded"}


def test_build_preserves_existing_registry_and_adds_only_two_inactive_boundaries(tmp_path):
    config = _config(tmp_path)
    build_archive_discovery(
        config, query_rows=_query_rows(), evidence_rows=_evidence_rows(),
    )
    registry = {row["archive_id"]: row for row in _rows(config.registry_path)}

    assert set(registry) == {
        "existing", "ilc_cnr_historical_corpora", "oxford_text_archive",
    }
    assert registry["existing"]["status"] == "processed_build_complete"
    assert registry["ilc_cnr_historical_corpora"]["status"] == "discovered_material_bounded_audit_pending_inactive"
    assert registry["oxford_text_archive"]["status"] == "discovered_material_bounded_audit_pending_inactive"
    assert "No text acquired or activated" in registry["oxford_text_archive"]["notes"]


def test_second_build_is_byte_deterministic(tmp_path):
    config = _config(tmp_path)
    kwargs = {"query_rows": _query_rows(), "evidence_rows": _evidence_rows()}
    build_archive_discovery(config, **kwargs)
    paths = (
        config.registry_path, config.query_path, config.evidence_path,
        config.decision_path, config.json_report_path, config.markdown_report_path,
    )
    first = {path.name: _sha(path) for path in paths}
    build_archive_discovery(config, **kwargs)
    second = {path.name: _sha(path) for path in paths}

    assert first == second


def test_public_artifacts_have_frozen_schemas_and_no_cache_payload(tmp_path):
    config = _config(tmp_path)
    build_archive_discovery(
        config, query_rows=_query_rows(), evidence_rows=_evidence_rows(),
    )

    assert tuple(_rows(config.query_path)[0]) == QUERY_FIELDS
    assert tuple(_rows(config.evidence_path)[0]) == EVIDENCE_FIELDS
    assert tuple(_rows(config.decision_path)[0]) == DECISION_FIELDS
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (config.query_path, config.evidence_path, config.decision_path)
    )
    assert "body_base64" not in combined
    assert "data/local" not in combined


def test_missing_query_fails_closed(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="query accounting"):
        build_archive_discovery(
            config,
            query_rows=_query_rows()[:-1],
            evidence_rows=_evidence_rows(),
        )


def test_missing_evidence_fails_closed(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="evidence accounting"):
        build_archive_discovery(
            config,
            query_rows=_query_rows(),
            evidence_rows=_evidence_rows()[:-1],
        )


def test_markdown_states_stop_rule_and_forbidden_actions(tmp_path):
    config = _config(tmp_path)
    build_archive_discovery(
        config, query_rows=_query_rows(), evidence_rows=_evidence_rows(),
    )
    markdown = config.markdown_report_path.read_text(encoding="utf-8")

    assert "stop rule did **not** close directly into checkpoint 7" in markdown
    assert "No corpus text, V7 split, mixture weight, cache deletion, or GPU work" in markdown
    assert "Next checkpoint: 6D" in markdown
