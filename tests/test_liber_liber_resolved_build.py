import csv
import hashlib
import json
from pathlib import Path

import pytest

import sonnet_corpus.liber_liber_resolved_build as build_module
from sonnet_corpus.liber_liber_resolved_build import (
    ATTRIBUTION_MANIFEST_FIELDS,
    CANONICAL_EDITION_FIELDS,
    RECORD_MANIFEST_FIELDS,
    SEGMENT_DECISION_FIELDS,
    SEGMENT_MANIFEST_FIELDS,
    SONNET_DECISION_FIELDS,
    SONNET_MANIFEST_FIELDS,
    SOURCE_DECISION_FIELDS,
    _discover_varaldo,
    _replace_verified_output,
)


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data/metadata"
OUTPUT = ROOT / "data/processed/liber_liber_resolved_v1"
REPORT = ROOT / "reports/liber_liber_resolved_v1_build.json"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_decision_ledgers_have_frozen_schemas_and_counts() -> None:
    paths = {
        METADATA / "liber_liber_extraction_decisions_v1.csv": (SOURCE_DECISION_FIELDS, 129),
        METADATA / "liber_liber_segment_decisions_v1.csv": (SEGMENT_DECISION_FIELDS, None),
        METADATA / "liber_liber_sonnet_candidates_v1.csv": (SONNET_DECISION_FIELDS, 64),
        METADATA / "liber_liber_canonical_editions_v1.csv": (CANONICAL_EDITION_FIELDS, 4),
    }
    for path, (fields, count) in paths.items():
        rows = _rows(path)
        assert tuple(rows[0]) == fields
        if count is not None:
            assert len(rows) == count


def test_processed_manifests_have_frozen_schemas() -> None:
    assert tuple(_rows(OUTPUT / "records_manifest.csv")[0]) == RECORD_MANIFEST_FIELDS
    assert tuple(_rows(OUTPUT / "segments_manifest.csv")[0]) == SEGMENT_MANIFEST_FIELDS
    assert tuple(_rows(OUTPUT / "sonnets_manifest.csv")[0]) == SONNET_MANIFEST_FIELDS
    assert tuple(_rows(OUTPUT / "attribution_manifest.csv")[0]) == ATTRIBUTION_MANIFEST_FIELDS


def test_source_accounting_and_inactive_materialization_are_exact() -> None:
    records = _rows(OUTPUT / "records_manifest.csv")
    assert len(records) == 129
    assert len({row["record_id"] for row in records}) == 129
    assert sum(row["artifact_status"] == "text_materialized_inactive" for row in records) == 91
    assert sum(row["final_decision"] == "exclude_canonical_cross_corpus_duplicate" for row in records) == 30
    assert sum(row["final_decision"] == "exclude_alternate_vita_nuova_edition" for row in records) == 3
    assert sum(row["final_decision"] == "exclude_composite_source_covered_by_existing_corpora" for row in records) == 4
    assert {row["activation_status"] for row in records} == {"inactive_pending_cross_archive_freeze"}


def test_sonnet_accounting_is_exact_and_no_artifact_is_active() -> None:
    sonnets = _rows(OUTPUT / "sonnets_manifest.csv")
    assert len(sonnets) == 64
    assert sum(row["artifact_status"] == "sonnet_materialized_inactive" for row in sonnets) == 40
    assert sum(row["candidate_decision"] == "exclude_existing_corpus_sonnet_duplicate" for row in sonnets) == 22
    assert sum(row["candidate_decision"] == "exclude_protected_v6_sonnet" for row in sonnets) == 2
    assert all(row["activation_status"] == "inactive_pending_cross_archive_freeze" for row in sonnets)


def test_materialized_sonnets_are_exactly_fourteen_lines_and_hash_pinned() -> None:
    cache: dict[Path, bytes] = {}
    for row in _rows(OUTPUT / "sonnets_manifest.csv"):
        if row["artifact_status"] != "sonnet_materialized_inactive":
            assert not row["shard_path"]
            continue
        path = ROOT / row["shard_path"]
        payload = cache.setdefault(path, path.read_bytes())
        text = payload[int(row["byte_start"]):int(row["byte_end"])].decode("utf-8")
        assert len(text.strip().splitlines()) == 14
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == row["cleaned_sha256"]


def test_record_and_segment_byte_ranges_are_recoverable() -> None:
    records = _rows(OUTPUT / "records_manifest.csv")
    segments = _rows(OUTPUT / "segments_manifest.csv")
    cache: dict[Path, bytes] = {}
    for row in records:
        if row["artifact_status"] != "text_materialized_inactive":
            assert not row["shard_path"]
            continue
        path = ROOT / row["shard_path"]
        payload = cache.setdefault(path, path.read_bytes())
        part = payload[int(row["byte_start"]):int(row["byte_end"])]
        assert hashlib.sha256(part).hexdigest() == row["cleaned_sha256"]
    for row in segments:
        if row["artifact_status"] != "materialized_in_inactive_record":
            continue
        path = ROOT / row["output_shard_path"]
        payload = cache.setdefault(path, path.read_bytes())
        part = payload[int(row["output_byte_start"]):int(row["output_byte_end"])]
        assert hashlib.sha256(part).hexdigest() == row["output_sha256"]


def test_vita_nuova_canonical_precedence_selects_one_clean_edition() -> None:
    rows = _rows(METADATA / "liber_liber_canonical_editions_v1.csv")
    selected = [row for row in rows if row["canonical_decision"] == "select_primary_canonical_edition"]
    assert [row["record_id"] for row in selected] == ["ll:2344213"]
    assert {row["selected_record_id"] for row in rows} == {"ll:2344213"}


def test_carducci_composite_canonicalization_has_two_exact_bibit_spans() -> None:
    source = next(
        row for row in _rows(METADATA / "liber_liber_extraction_decisions_v1.csv")
        if row["record_id"] == "ll:2344854"
    )
    assert source["final_decision"] == "exclude_composite_source_covered_by_existing_corpora"
    assert {"bibit:bibit000521", "bibit:bibit001121"}.issubset(
        set(source["canonical_reference_ids"].split(";"))
    )
    segments = [
        row for row in _rows(METADATA / "liber_liber_segment_decisions_v1.csv")
        if row["record_id"] == "ll:2344854"
    ]
    assert sum(int(row["character_count"]) for row in segments) == 53_433
    assert not any(row["segment_decision"] == "include_broader_text" for row in segments)


def test_cenni_source_is_not_misclassified_as_exact_fourteen_line_sonnets() -> None:
    source = next(
        row for row in _rows(METADATA / "liber_liber_extraction_decisions_v1.csv")
        if row["record_id"] == "ll:2344942"
    )
    assert source["final_broader_role"] == "historical_non_sonnet_poetry"
    assert source["sonnet_candidate_count"] == "0"
    review = next(
        row for row in _rows(METADATA / "liber_liber_sonnet_review_v1.csv")
        if row["record_id"] == "ll:2344942"
    )
    assert review["review_resolution"] == "retain_as_historical_non_sonnet_poetry"


def test_varaldo_signature_extractor_requires_and_returns_39_sonnets() -> None:
    parts = ["ALESSANDRO VARALDO\nMARIO MALFETTANI\nALESSANDRO GIRIBALDI\n"]
    authors = ("Alessandro Varaldo.", "Mario Malfettani", "Alessandro Giribaldi.")
    for index in range(39):
        parts.extend(f"poesia {index} verso {line}\n" for line in range(14))
        parts.append(authors[index % 3] + "\n")
    found = _discover_varaldo("".join(parts), {"record_id": "ll:2427167"})
    assert len(found) == 39
    assert all(len(row["cleaned"].strip().splitlines()) == 14 for row in found)
    assert {row["poem_author"] for row in found} == {
        "Alessandro Varaldo", "Mario Malfettani", "Alessandro Giribaldi"
    }


def test_attribution_is_complete_and_contains_no_email_addresses() -> None:
    rows = _rows(OUTPUT / "attribution_manifest.csv")
    assert len(rows) == 92
    assert all(row["landing_page_url"] and row["license_url"] and row["required_notice"] for row in rows)
    payload = (OUTPUT / "attribution_manifest.csv").read_text(encoding="utf-8")
    assert "@" not in payload
    assert "Non-commercial" in payload


def test_report_hashes_and_counts_match_installed_artifacts() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    installed = json.loads((OUTPUT / "build_report.json").read_text(encoding="utf-8"))
    assert report == installed
    assert report["materialized_record_count"] == 91
    assert report["materialized_broader_character_count"] == 26_515_621
    assert report["materialized_sonnet_count"] == 40
    for name, digest in report["manifest_sha256"].items():
        assert _sha_file(OUTPUT / name) == digest


def test_report_freezes_non_activation_boundaries() -> None:
    policy = json.loads(REPORT.read_text(encoding="utf-8"))["policy"]
    assert policy == {
        "cache_deleted": False,
        "conditioned_material_excluded": True,
        "dialogue_terminal_hyphens_preserved": True,
        "gpu_work_started": False,
        "mixture_assigned": False,
        "spelling_and_punctuation_preserved": True,
        "text_activated": False,
        "text_materialized_inactive": True,
        "v7_created": False,
    }


def test_every_shard_is_bounded_below_64_mib() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    for shards in report["shards"].values():
        for shard in shards:
            assert shard["byte_count"] <= 64 * 1024 * 1024
            assert _sha_file(ROOT / shard["path"]) == shard["sha256"]


def test_recoverable_replacement_restores_previous_output_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "resolved"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    temp = tmp_path / ".resolved.new"
    temp.mkdir()
    (temp / "new.txt").write_text("new", encoding="utf-8")
    real_replace = build_module.os.replace
    calls = 0

    def failing_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated installation failure")
        real_replace(source, destination)

    monkeypatch.setattr(build_module.os, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated installation failure"):
        _replace_verified_output(temp, output)
    assert (output / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (tmp_path / ".resolved.previous").exists()
