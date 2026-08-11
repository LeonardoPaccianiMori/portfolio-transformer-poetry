import csv
import hashlib
import json
from pathlib import Path

import pytest

import sonnet_corpus.wikisource_resolved_build as build_module
from sonnet_corpus.wikisource_resolved_build import (
    RECORD_MANIFEST_FIELDS,
    SEGMENT_MANIFEST_FIELDS,
    SONNET_MANIFEST_FIELDS,
    WikisourceResolvedBuildConfig,
    build_wikisource_resolved_corpus,
)
from sonnet_corpus.wikisource_review_resolution import RIGHTS_FIELDS


def _write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _sonnet() -> str:
    return "\n".join(f"verso poetico numero {index} amore" for index in range(1, 15))


def _fixture(tmp_path: Path) -> WikisourceResolvedBuildConfig:
    metadata = tmp_path / "metadata"
    cache = tmp_path / "cache"
    cache.mkdir()
    prose = "Testo storico unico e verificato."
    poem = _sonnet()
    (cache / "itws_1.txt").write_text(prose + "\n", encoding="utf-8")
    (cache / "itws_2.txt").write_text(poem + "\n", encoding="utf-8")
    root_fields = (
        "work_root_id", "root_title", "landing_page_url", "author_evidence", "period_bucket",
        "input_role", "final_broader_role", "direct_scan_title", "scan_rights_id",
        "checkpoint_4c_decision", "quality_flags", "source_cache_path", "source_sha256",
        "source_character_count", "canonical_reference_ids", "removed_reference_ids",
        "rights_decision", "final_decision", "resolution_reason",
        "retained_broader_character_count", "excluded_character_count", "sonnet_candidate_count",
        "activation_status",
    )
    roots = [
        dict.fromkeys(root_fields, "") | {
            "work_root_id": "itws:1", "root_title": "Prosa", "landing_page_url": "https://example/1",
            "author_evidence": "Autore", "period_bucket": "origins_through_1800",
            "input_role": "historical_general", "final_broader_role": "historical_general",
            "direct_scan_title": "Scan.djvu", "scan_rights_id": "itws-scan:1",
            "source_cache_path": "cache/itws_1.txt", "source_sha256": _sha(prose),
            "source_character_count": str(len(prose)), "rights_decision": "rights_pass",
            "final_decision": "eligible_inactive_processed_build", "resolution_reason": "pass",
            "retained_broader_character_count": str(len(prose)), "excluded_character_count": "0",
            "sonnet_candidate_count": "0", "activation_status": "inactive",
        },
        dict.fromkeys(root_fields, "") | {
            "work_root_id": "itws:2", "root_title": "Sonetto", "landing_page_url": "https://example/2",
            "author_evidence": "Poeta", "period_bucket": "origins_through_1800",
            "input_role": "standard_sonnets", "final_broader_role": "historical_non_sonnet_poetry",
            "direct_scan_title": "Scan.djvu", "scan_rights_id": "itws-scan:1",
            "source_cache_path": "cache/itws_2.txt", "source_sha256": _sha(poem),
            "source_character_count": str(len(poem)), "rights_decision": "rights_pass",
            "final_decision": "eligible_sonnets_only_inactive", "resolution_reason": "sonnet only",
            "retained_broader_character_count": "0", "excluded_character_count": str(len(poem)),
            "sonnet_candidate_count": "1", "activation_status": "inactive",
        },
    ]
    segments = [
        {field: "" for field in SEGMENT_MANIFEST_FIELDS[:12]} | {
            "segment_id": "itws:1:seg0001", "work_root_id": "itws:1", "source_sha256": _sha(prose),
            "character_start": "0", "character_end": str(len(prose)), "character_count": str(len(prose)),
            "segment_sha256": _sha(prose), "segment_decision": "include_broader_text",
            "final_role": "historical_general", "reason": "retained", "activation_status": "inactive",
        },
        {field: "" for field in SEGMENT_MANIFEST_FIELDS[:12]} | {
            "segment_id": "itws:2:seg0001", "work_root_id": "itws:2", "source_sha256": _sha(poem),
            "character_start": "0", "character_end": str(len(poem)), "character_count": str(len(poem)),
            "segment_sha256": _sha(poem), "segment_decision": "materialize_standard_sonnet_inactive",
            "final_role": "standard_sonnets", "reason": "verified", "reference_ids": "itws:2:sonnet0001",
            "activation_status": "inactive",
        },
    ]
    cleaned_poem = poem + "\n"
    sonnet_fields = SONNET_MANIFEST_FIELDS[:24]
    sonnets = [{field: "" for field in sonnet_fields} | {
        "candidate_id": "itws:2:sonnet0001", "work_root_id": "itws:2", "root_title": "Sonetto",
        "source_record_author": "Poeta", "poem_author": "Poeta", "poem_author_resolution": "root proxy",
        "period_bucket": "origins_through_1800", "source_url": "https://example/2",
        "source_scan_title": "Scan.djvu", "source_kind": "source_metadata_sonnet",
        "stanza_pattern": "14", "line_count": "14", "first_line": poem.splitlines()[0],
        "last_line": poem.splitlines()[-1], "character_start": "0", "character_end": str(len(poem)),
        "source_text_sha256": _sha(poem), "cleaned_text_sha256": _sha(cleaned_poem),
        "candidate_decision": "eligible_standard_sonnet_inactive_pending_v7",
        "final_role": "standard_sonnets", "activation_status": "inactive",
    }]
    rights = [{field: "" for field in RIGHTS_FIELDS} | {
        "scan_rights_id": "itws-scan:1", "scan_title": "Scan.djvu", "rights_decision": "rights_pass",
    }]
    roots_path = metadata / "roots.csv"; segments_path = metadata / "segments.csv"
    sonnets_path = metadata / "sonnets.csv"; rights_path = metadata / "rights.csv"
    _write_csv(roots_path, root_fields, roots)
    _write_csv(segments_path, SEGMENT_MANIFEST_FIELDS[:12], segments)
    _write_csv(sonnets_path, sonnet_fields, sonnets)
    _write_csv(rights_path, RIGHTS_FIELDS, rights)
    report_path = metadata / "review.json"
    report_path.write_text(json.dumps({"output_sha256": {
        "roots": _sha_file(roots_path), "segments": _sha_file(segments_path),
        "sonnets": _sha_file(sonnets_path), "scan_rights": _sha_file(rights_path),
    }}), encoding="utf-8")
    return WikisourceResolvedBuildConfig(
        repo_root=tmp_path, root_decisions_path=roots_path, segment_decisions_path=segments_path,
        sonnet_decisions_path=sonnets_path, scan_rights_path=rights_path,
        review_report_path=report_path, output_dir=tmp_path / "processed",
        markdown_report_path=tmp_path / "report.md", bibit_record_manifest_path=tmp_path / "unused1",
        broader_sources_manifest_path=tmp_path / "unused2", gutenberg_previous_probe_path=tmp_path / "unused3",
        gutenberg_previous_cache_dir=tmp_path / "unused4", gutenberg_pass_1b_probe_path=tmp_path / "unused5",
        gutenberg_pass_1b_cache_dir=tmp_path / "unused6", gutenberg_resolved_record_manifest_path=tmp_path / "unused7",
        protected_sonnet_manifest_path=tmp_path / "unused8", max_shard_bytes=1024,
    )


def test_build_materializes_inactive_records_sonnets_and_recoverable_ranges(tmp_path, monkeypatch):
    config = _fixture(tmp_path)
    monkeypatch.setattr(build_module, "_verify_final", lambda *_args, **_kwargs: {"all_clear": True})
    report = build_wikisource_resolved_corpus(config)
    assert report["materialized_record_count"] == 1
    assert report["materialized_sonnet_count"] == 1
    records = list(csv.DictReader((config.output_dir / "records_manifest.csv").open()))
    sonnets = list(csv.DictReader((config.output_dir / "sonnets_manifest.csv").open()))
    assert records[0]["artifact_status"] == "text_materialized_inactive"
    assert sonnets[0]["artifact_status"] == "sonnet_materialized_inactive"
    payload = (config.repo_root / sonnets[0]["shard_path"]).read_bytes()
    part = payload[int(sonnets[0]["byte_start"]):int(sonnets[0]["byte_end"])]
    assert hashlib.sha256(part).hexdigest() == sonnets[0]["cleaned_sha256"]


def test_build_rejects_stale_review_ledger(tmp_path, monkeypatch):
    config = _fixture(tmp_path)
    config.root_decisions_path.write_text(config.root_decisions_path.read_text() + "\n")
    monkeypatch.setattr(build_module, "_verify_final", lambda *_args, **_kwargs: {})
    with pytest.raises(ValueError, match="stale or modified"):
        build_wikisource_resolved_corpus(config)


def test_build_rejects_eligible_broader_root_that_materializes_no_text(tmp_path, monkeypatch):
    config = _fixture(tmp_path)
    roots = list(csv.DictReader(config.root_decisions_path.open(encoding="utf-8")))
    roots[0]["source_cache_path"] = "cache/whitespace.txt"
    roots[0]["source_sha256"] = _sha(" ")
    roots[0]["source_character_count"] = "1"
    roots[0]["retained_broader_character_count"] = "1"
    (tmp_path / "cache/whitespace.txt").write_text(" \n", encoding="utf-8")
    segments = list(csv.DictReader(config.segment_decisions_path.open(encoding="utf-8")))
    segments[0].update({
        "source_sha256": _sha(" "), "character_end": "1", "character_count": "1",
        "segment_sha256": _sha(" "),
    })
    _write_csv(config.root_decisions_path, roots[0].keys(), roots)
    _write_csv(config.segment_decisions_path, segments[0].keys(), segments)
    report = json.loads(config.review_report_path.read_text(encoding="utf-8"))
    report["output_sha256"]["roots"] = _sha_file(config.root_decisions_path)
    report["output_sha256"]["segments"] = _sha_file(config.segment_decisions_path)
    config.review_report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(build_module, "_verify_final", lambda *_args, **_kwargs: {})

    with pytest.raises(ValueError, match="resolved broader-root count"):
        build_wikisource_resolved_corpus(config)
