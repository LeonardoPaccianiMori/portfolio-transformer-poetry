import csv
import hashlib
import json
from pathlib import Path

import pytest

from sonnet_corpus.archive_registry_resolution import (
    ARCHIVE_IDS,
    EVIDENCE_FIELDS,
    EVIDENCE_SPECS,
    ArchiveRegistryResolutionConfig,
    EvidenceSpec,
    build_archive_registry_resolution,
    fetch_official_evidence,
)


REGISTRY_FIELDS = (
    "archive_id", "archive_name", "landing_page", "corpus_roles", "coverage",
    "license_or_reuse_status", "bulk_access", "status", "next_action", "notes",
)


class FakeResponse:
    def __init__(self, body, *, url="https://official.test/terms", status=200, content_type="text/html"):
        self.content = body.encode() if isinstance(body, str) else body
        self.url = url
        self.status_code = status
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = []

    def get(self, url, *, timeout, allow_redirects):
        self.calls.append((url, timeout, allow_redirects))
        return self.responses.pop(0)


def _config(tmp_path: Path) -> ArchiveRegistryResolutionConfig:
    metadata = tmp_path / "data/metadata"
    reports = tmp_path / "reports"
    metadata.mkdir(parents=True)
    reports.mkdir()
    rows = [{
        "archive_id": "bibit_texts", "archive_name": "BibIt", "landing_page": "x",
        "corpus_roles": "core", "coverage": "done", "license_or_reuse_status": "pinned",
        "bulk_access": "tei", "status": "processed_build_complete", "next_action": "7", "notes": "done",
    }]
    frozen = {spec.archive_id: spec for spec in EVIDENCE_SPECS}
    input_status = {
        "bibit_scrittori_italia": "inventory_pending", "bibit_incunaboli": "inventory_pending",
        "eltec_italian": "terms_audit_pending", "internet_archive": "rights_and_quality_gate_required",
        "gallica": "terms_audit_pending", "internet_culturale": "terms_and_access_audit_pending",
        "beic": "terms_and_access_audit_pending", "hathitrust": "blocked_pending_permission",
        "google_books": "discovery_only", "ovi_tlio": "terms_and_access_audit_pending",
        "midia": "terms_and_access_audit_pending", "diacoris": "terms_and_access_audit_pending",
    }
    for archive_id in ARCHIVE_IDS:
        rows.append({
            "archive_id": archive_id, "archive_name": archive_id, "landing_page": frozen[archive_id].url,
            "corpus_roles": "pending", "coverage": "pending", "license_or_reuse_status": "pending",
            "bulk_access": "pending", "status": input_status[archive_id], "next_action": "audit", "notes": "pending",
        })
    with (metadata / "registry.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    return ArchiveRegistryResolutionConfig(
        repo_root=tmp_path,
        registry_path=metadata / "registry.csv",
        cache_dir=tmp_path / "data/local/cache",
        evidence_path=metadata / "evidence.csv",
        resolution_path=metadata / "resolution.csv",
        composition_gate_path=metadata / "gate.csv",
        json_report_path=reports / "report.json",
        markdown_report_path=reports / "report.md",
        request_delay_seconds=0,
    )


def _evidence_rows():
    digest = hashlib.sha256(b"official").hexdigest()
    return [{
        "evidence_id": spec.evidence_id, "archive_id": spec.archive_id,
        "evidence_type": spec.evidence_type, "authority": "official_first_party",
        "source_url": spec.url, "resolved_url": spec.url, "retrieval_date": "2026-08-11",
        "http_status": "200", "content_type": "text/plain", "content_sha256": digest,
        "evidence_quote": spec.quote or "Pinned metadata query returned 10 records.",
        "supports_decision": spec.supports, "limitation": spec.limitation,
        "verification_status": "verified_official_evidence",
    } for spec in EVIDENCE_SPECS]


def _rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_frozen_scope_has_exactly_twelve_unresolved_archives():
    assert len(ARCHIVE_IDS) == 12
    assert len(set(ARCHIVE_IDS)) == 12
    assert "bibit_texts" not in ARCHIVE_IDS


def test_fetches_official_page_atomically_then_reuses_hash_pinned_cache(tmp_path):
    config = _config(tmp_path)
    spec = EvidenceSpec(
        "test_terms", "midia", "terms", "https://official.test/terms",
        "Official quote.", ("Official quote",), "supports", "limit",
    )
    session = FakeSession([FakeResponse("<p>Official quote</p>")])
    first = fetch_official_evidence(config, session=session, specs=(spec,), sleep=lambda _: None)
    second = fetch_official_evidence(config, session=FakeSession([]), specs=(spec,), sleep=lambda _: None)
    assert first == second
    assert first[0]["verification_status"] == "verified_official_evidence"
    assert len(list(config.cache_dir.glob("*.body"))) == 1
    assert not list(config.cache_dir.glob("*.tmp"))


def test_fetch_fails_when_official_quote_changes(tmp_path):
    config = _config(tmp_path)
    spec = EvidenceSpec("changed", "midia", "terms", "https://official.test/changed", "Expected", ("Expected",), "s", "l")
    with pytest.raises(ValueError, match="missing expected text"):
        fetch_official_evidence(config, session=FakeSession([FakeResponse("Changed")]), specs=(spec,), sleep=lambda _: None)


def test_inaccessible_official_terms_are_recorded_not_interpreted(tmp_path):
    config = _config(tmp_path)
    spec = EvidenceSpec("blocked", "hathitrust", "terms", "https://official.test/blocked", "Official page inaccessible.", (), "s", "no permission", allow_inaccessible=True)
    rows = fetch_official_evidence(config, session=FakeSession([FakeResponse("challenge", status=403)]), specs=(spec,), sleep=lambda _: None)
    assert rows[0]["verification_status"] == "official_page_inaccessible_http_403"
    assert rows[0]["limitation"] == "no permission"


def test_json_count_evidence_uses_pinned_machine_value(tmp_path):
    config = _config(tmp_path)
    spec = EvidenceSpec("count", "internet_archive", "count", "https://official.test/count", "", ("language:ita",), "s", "l", json_count_path=("response", "numFound"))
    body = json.dumps({"query": "language:ita", "response": {"numFound": 1234}})
    rows = fetch_official_evidence(config, session=FakeSession([FakeResponse(body, content_type="application/json")]), specs=(spec,), sleep=lambda _: None)
    assert rows[0]["evidence_quote"] == "Pinned metadata query returned 1,234 records."


def test_build_reconciles_decisions_roles_and_inactive_constraints(tmp_path):
    config = _config(tmp_path)
    report = build_archive_registry_resolution(config, evidence_rows=_evidence_rows())
    resolutions = _rows(config.resolution_path)
    gates = _rows(config.composition_gate_path)
    assert len(resolutions) == len(gates) == report["archive_count"] == 12
    assert {row["archive_id"] for row in resolutions} == set(ARCHIVE_IDS)
    assert all(row["activation_status"] == "inactive_metadata_only" for row in resolutions + gates)
    assert report["eligible_bounded_inventory_count"] == 6
    assert report["activated_corpus_characters"] == 0
    assert report["text_downloads"] == 0
    assert report["v7_split_created"] is False
    assert report["gpu_work_started"] is False


def test_build_updates_only_frozen_registry_rows(tmp_path):
    config = _config(tmp_path)
    build_archive_registry_resolution(config, evidence_rows=_evidence_rows())
    registry = {row["archive_id"]: row for row in _rows(config.registry_path)}
    assert registry["bibit_texts"]["status"] == "processed_build_complete"
    assert registry["eltec_italian"]["status"] == "eligible_bounded_inventory_inactive"
    assert registry["hathitrust"]["status"] == "blocked_official_terms_and_bulk_access"
    assert registry["google_books"]["status"] == "discovery_only_closed"


def test_second_build_is_byte_deterministic(tmp_path):
    config = _config(tmp_path)
    build_archive_registry_resolution(config, evidence_rows=_evidence_rows())
    paths = (config.registry_path, config.evidence_path, config.resolution_path, config.composition_gate_path, config.json_report_path, config.markdown_report_path)
    first = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    build_archive_registry_resolution(config, evidence_rows=_evidence_rows())
    second = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert first == second


def test_public_evidence_has_no_cache_payload_or_email(tmp_path):
    config = _config(tmp_path)
    build_archive_registry_resolution(config, evidence_rows=_evidence_rows())
    header = tuple(_rows(config.evidence_path)[0])
    assert header == EVIDENCE_FIELDS
    text = config.evidence_path.read_text(encoding="utf-8")
    assert "data/local" not in text
    assert "@gmail.com" not in text
    assert "body_base64" not in text


def test_scope_drift_fails_closed(tmp_path):
    config = _config(tmp_path)
    rows = _rows(config.registry_path)
    rows = [row for row in rows if row["archive_id"] != "ovi_tlio"]
    with config.registry_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError, match="missing a frozen"):
        build_archive_registry_resolution(config, evidence_rows=_evidence_rows())


def test_markdown_states_bounded_inventory_is_not_text_authorization(tmp_path):
    config = _config(tmp_path)
    build_archive_registry_resolution(config, evidence_rows=_evidence_rows())
    markdown = config.markdown_report_path.read_text(encoding="utf-8")
    assert "does not authorize corpus-text acquisition" in markdown
    assert "open-ended final discovery pass remains checkpoint 6C" in markdown
