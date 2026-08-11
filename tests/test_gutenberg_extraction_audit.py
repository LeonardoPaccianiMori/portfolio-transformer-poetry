import csv
import hashlib
import json
from pathlib import Path

import pytest

from sonnet_corpus.gutenberg_extraction_audit import (
    GutenbergExtractionAuditConfig,
    audit_gutenberg_extraction,
    discover_gutenberg_sonnet_candidates,
    locate_reference_segments,
)


def _write_csv(path: Path, rows: list[dict[str, str]], fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or tuple(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _gutenberg(body: str) -> str:
    return (
        "Header\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        + body.strip()
        + "\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nFooter\n"
    )


def _sonnet(prefix: str) -> str:
    groups = (4, 4, 3, 3)
    lines = []
    number = 1
    for count in groups:
        lines.extend(f"    {prefix} verso poetico numero {number}" for _ in range(count))
        number += 1
        lines.append("")
    return "\n".join(lines).strip()


def _label(index: int) -> str:
    return chr(ord("a") + index // 26) + chr(ord("a") + index % 26)


def _fixture(tmp_path: Path) -> GutenbergExtractionAuditConfig:
    root = tmp_path
    prior_cache = root / "data/local/gutenberg/fulltext_gate_v1"
    pass1b_cache = root / "data/local/gutenberg/metadata_review_v1"
    prior_cache.mkdir(parents=True)
    pass1b_cache.mkdir(parents=True)

    embedded = " ".join(f"parolaembedded{_label(index)}" for index in range(40))
    heldout_lines = [f"Linea protetta distinta parola{_label(index)}" for index in range(14)]
    heldout = "\n".join(heldout_lines) + "\n"
    eligible = _sonnet("Nuovo")
    structural = "\n".join(f"    Strutturale verso distinto {index}" for index in range(1, 15))
    bodies = {
        "1": f"Introduzione unica.\n\nSONETTO\n\n{eligible}\n\nCongedo unico.",
        "2": f"Introduzione unica.\n\n{embedded}\n\nCongedo unico.",
        "3": f"Romanzo prima.\n\n{heldout}\nRomanzo dopo.",
        "4": "Testo interamente già canonico.",
        "5": "Testo dialettale non attivato.",
        "17440": """English dedication.

Personaggi.

AMADIGI.

  Io sono qui con te e non ti lascio.
  La gloria e il mio amore sono vivi.

  I am here with you and will not leave.
  My glory and my love are still alive.

Italian:
Transcriber notes.
""",
        "17834": """NOTICE SUR LA ZAFFETTA

Texte français.

Poi ch'ogni bestia in volgar e in latino,
Primo verso della prima edizione italiana.
Secondo verso della prima edizione italiana.
IL FINE.

Poich'ogni bestia in volgare e in latino,
Testo della seconda edizione.
IL FINE.
""",
        "10": structural,
        "11": "Materiale romanesco condizionato.",
    }
    for ebook_id in ("1", "2", "3", "4", "5", "17440", "17834"):
        (prior_cache / f"pg{ebook_id}.txt").write_text(_gutenberg(bodies[ebook_id]), encoding="utf-8")
    for ebook_id in ("10", "11"):
        (pass1b_cache / f"pg{ebook_id}.txt").write_text(_gutenberg(bodies[ebook_id]), encoding="utf-8")

    base = {
        "authors": "Autore",
        "period_bucket": "origins_through_1800",
        "inventory_status": "audit_then_deduplicate",
        "landing_page_url": "https://example.test",
        "bibit_overlap_metrics": "",
        "current_corpus_overlap_metrics": "",
        "heldout_sonnet_overlap_metrics": "",
    }
    prior_rows = [
        {**base, "ebook_id": "1", "title": "Sonetti nuovi", "preliminary_role": "sonnet_specialization_candidate", "probe_decision": "quality_pass_pending_editorial_activation_review"},
        {**base, "ebook_id": "2", "title": "Raccolta parziale", "preliminary_role": "historical_general_candidate", "probe_decision": "quarantine_embedded_duplicate_segments_before_activation", "bibit_overlap_metrics": "bibit:bibit000001|candidate_containment=0.4|reference_containment=1.0"},
        {**base, "ebook_id": "3", "title": "Romanzo", "preliminary_role": "nineteenth_century_bridge_candidate", "probe_decision": "quarantine_heldout_sonnet_segment_before_activation", "heldout_sonnet_overlap_metrics": "heldout_test|containment=1.0"},
        {**base, "ebook_id": "4", "title": "Duplicato", "preliminary_role": "historical_general_candidate", "probe_decision": "exclude_cross_corpus_duplicate_candidate", "bibit_overlap_metrics": "bibit:bibit000002|candidate_containment=1.0|reference_containment=1.0"},
        {**base, "ebook_id": "5", "title": "Dialetto", "preliminary_role": "historical_non_sonnet_poetry_candidate", "probe_decision": "exclude_standard_italian_core_language_composition"},
        {**base, "ebook_id": "17440", "title": "Amadigi", "preliminary_role": "historical_non_sonnet_poetry_candidate", "probe_decision": "source_specific_language_extraction_before_activation"},
        {**base, "ebook_id": "17834", "title": "Zaffetta", "preliminary_role": "historical_non_sonnet_poetry_candidate", "probe_decision": "source_specific_language_extraction_before_activation"},
    ]
    prior_probe = root / "prior.csv"
    _write_csv(prior_probe, prior_rows)

    pass1b_rows = [
        {
            **base,
            "ebook_id": "10",
            "title": "Poesie strutturali",
            "preliminary_role": "historical_non_sonnet_poetry_candidate",
            "resolution_pass": "pass_1b",
            "final_period_bucket": "origins_through_1800",
            "final_role": "historical_non_sonnet_poetry_candidate",
            "final_activation_class": "eligible_probe",
            "probe_decision": "quality_pass_pending_editorial_activation_review",
        }
    ]
    pass1b_probe = root / "pass1b.csv"
    _write_csv(pass1b_probe, pass1b_rows)

    final_resolution = root / "final.csv"
    _write_csv(
        final_resolution,
        [{
            "ebook_id": "11",
            "title": "Sonetti romaneschi",
            "authors": "Autore",
            "period_bucket": "language_conditioned_period_unresolved",
            "final_period_bucket": "language_conditioned_period_unresolved",
            "preliminary_role": "language_variety_review",
            "final_role": "conditioned_romanesco_sonnet_candidate",
            "final_activation_class": "conditioned_probe",
            "landing_page_url": "https://example.test/11",
        }],
    )

    shard = root / "data/processed/bibit_resolved_v1/historical_general/part-0001.txt"
    shard.parent.mkdir(parents=True)
    shard.write_text(embedded + "\nTesto interamente già canonico.", encoding="utf-8")
    embedded_bytes = embedded.encode("utf-8")
    record_manifest = root / "data/processed/bibit_resolved_v1/records_manifest.csv"
    _write_csv(
        record_manifest,
        [
            {"object_id": "bibit000001", "artifact_status": "text_materialized", "shard_path": shard.relative_to(root).as_posix(), "byte_start": "0", "byte_end": str(len(embedded_bytes))},
            {"object_id": "bibit000002", "artifact_status": "text_materialized", "shard_path": shard.relative_to(root).as_posix(), "byte_start": str(len(embedded_bytes) + 1), "byte_end": str(shard.stat().st_size)},
        ],
    )
    broader_manifest = root / "data/metadata/broader_prose_sources_manifest.csv"
    _write_csv(broader_manifest, [], fields=("source_id", "expected_clean_text_path"))

    heldout_path = root / "data/processed/sonnets_expanded_v6/poems/heldout.txt"
    heldout_path.parent.mkdir(parents=True)
    heldout_path.write_text(heldout, encoding="utf-8")
    sonnet_manifest = root / "data/metadata/sonnets_expanded_v6_manifest.csv"
    _write_csv(
        sonnet_manifest,
        [{"poem_id": "heldout_test", "clean_text_path": heldout_path.relative_to(root).as_posix(), "split_expanded_with_petrarch": "validation"}],
    )

    bibit_sonnet_text = "\n".join(f"Altro sonetto bibit verso {index}" for index in range(1, 15)) + "\n"
    bibit_sonnet_shard = root / "data/processed/bibit_resolved_v1/standard_sonnets/part-0001.txt"
    bibit_sonnet_shard.parent.mkdir(parents=True)
    bibit_sonnet_shard.write_text(bibit_sonnet_text, encoding="utf-8")
    bibit_sonnet_manifest = root / "data/processed/bibit_resolved_v1/sonnets_manifest.csv"
    _write_csv(
        bibit_sonnet_manifest,
        [{"candidate_id": "bibit000001:test", "shard_path": bibit_sonnet_shard.relative_to(root).as_posix(), "byte_start": "0", "byte_end": str(bibit_sonnet_shard.stat().st_size)}],
    )

    return GutenbergExtractionAuditConfig(
        repo_root=root,
        prior_probe_csv_path=prior_probe,
        pass1b_probe_csv_path=pass1b_probe,
        final_resolution_csv_path=final_resolution,
        prior_cache_dir=prior_cache,
        pass1b_cache_dir=pass1b_cache,
        bibit_record_manifest_path=record_manifest,
        broader_sources_manifest_path=broader_manifest,
        sonnet_manifest_path=sonnet_manifest,
        bibit_sonnet_manifest_path=bibit_sonnet_manifest,
        source_csv_path=root / "data/metadata/project_gutenberg_extraction_decisions_v1.csv",
        segment_csv_path=root / "data/metadata/project_gutenberg_segment_decisions_v1.csv",
        sonnet_csv_path=root / "data/metadata/project_gutenberg_sonnet_candidates_v1.csv",
        review_csv_path=root / "data/metadata/project_gutenberg_sonnet_review_v1.csv",
        json_report_path=root / "reports/project_gutenberg_extraction_audit_v1.json",
        markdown_report_path=root / "reports/project_gutenberg_extraction_audit_v1.md",
        expected_prior_count=7,
        expected_pass1b_count=1,
        expected_conditioned_count=1,
        progress_interval=1,
    )


def test_locate_reference_segments_preserves_surrounding_unique_text():
    reference = " ".join(f"parola{_label(index)}" for index in range(40))
    candidate = f"Introduzione unica.\n{reference}\nCongedo unico.\n"

    spans = locate_reference_segments(candidate, reference)

    assert len(spans) == 1
    start, end, anchors = spans[0]
    assert anchors >= 20
    assert "parolaaa" in candidate[start:end]
    assert f"parola{_label(39)}" in candidate[start:end]
    assert "Introduzione" not in candidate[start:end]
    assert "Congedo" not in candidate[start:end]


def test_discover_gutenberg_sonnet_candidates_recognizes_explicit_stanzas():
    poem = _sonnet("Verso")
    text = f"Prefazione\n\nSONETTO\n\n{poem}\n\nCongedo\n"

    candidates = discover_gutenberg_sonnet_candidates(
        text,
        ebook_id="1",
        title="Opera",
        authors="Autore",
        role="historical_general",
        allowed_ranges=[(0, len(text))],
    )

    assert len(candidates) == 1
    assert candidates[0].source_kind == "explicit_sonetto_heading"
    assert candidates[0].stanza_pattern == "4-4-3-3"
    assert len(candidates[0].cleaned_text.strip().splitlines()) == 14


def test_audit_writes_complete_partition_and_bounded_review(tmp_path):
    config = _fixture(tmp_path)
    messages = []

    report = audit_gutenberg_extraction(config, progress=messages.append)

    assert report["source_count"] == 9
    assert report["unresolved_sonnet_review_count"] == 1
    assert report["policy"]["processed_text_materialized"] is False
    assert "source-audit 9/9" in " ".join(messages)
    with config.source_csv_path.open(encoding="utf-8", newline="") as handle:
        sources = {row["ebook_id"]: row for row in csv.DictReader(handle)}
    assert sources["4"]["source_decision"] == "exclude_canonical_cross_corpus_duplicate"
    assert sources["5"]["source_decision"] == "conditioned_candidate_not_activated"
    assert sources["11"]["source_decision"] == "conditioned_candidate_not_activated"
    assert sources["3"]["residual_heldout_overlap_ids"] == ""
    with config.segment_csv_path.open(encoding="utf-8", newline="") as handle:
        segments = list(csv.DictReader(handle))
    assert any(row["ebook_id"] == "2" and row["segment_decision"] == "exclude_embedded_canonical_text" for row in segments)
    assert any(row["ebook_id"] == "3" and row["segment_decision"] == "exclude_protected_v6_sonnet" for row in segments)
    assert any(row["ebook_id"] == "17440" and row["segment_decision"] == "exclude_parallel_english_translation" for row in segments)
    assert any(row["ebook_id"] == "17834" and row["segment_decision"] == "exclude_duplicate_modernized_edition" for row in segments)


def test_audit_applies_hash_pinned_structural_review(tmp_path):
    config = _fixture(tmp_path)
    first = audit_gutenberg_extraction(config)
    assert first["unresolved_sonnet_review_count"] == 1
    with config.review_csv_path.open(encoding="utf-8", newline="") as handle:
        reviews = list(csv.DictReader(handle))
    reviews[0]["review_resolution"] = "accept_structurally_verified_standard_sonnet"
    reviews[0]["review_rationale"] = "Fourteen distinct verse lines form one bounded poem."
    _write_csv(config.review_csv_path, reviews)

    second = audit_gutenberg_extraction(config)

    assert second["unresolved_sonnet_review_count"] == 0
    assert second["eligible_standard_sonnet_count"] == 2
    assert json.loads(config.json_report_path.read_text(encoding="utf-8"))["source_count"] == 9


def test_audit_retains_manual_non_sonnet_false_positive_in_broader_text(tmp_path):
    config = _fixture(tmp_path)
    audit_gutenberg_extraction(config)
    with config.review_csv_path.open(encoding="utf-8", newline="") as handle:
        reviews = list(csv.DictReader(handle))
    candidate_id = reviews[0]["candidate_id"]
    reviews[0]["review_resolution"] = "exclude_not_sonnet"
    reviews[0]["review_rationale"] = "Fourteen indented lines are not a sonnet."
    _write_csv(config.review_csv_path, reviews)

    report = audit_gutenberg_extraction(config)

    assert report["unresolved_sonnet_review_count"] == 0
    assert report["sonnet_decision_counts"]["exclude_manual_not_sonnet"] == 1
    with config.segment_csv_path.open(encoding="utf-8", newline="") as handle:
        segments = list(csv.DictReader(handle))
    assert not any(
        row["ebook_id"] == "10"
        and row["segment_decision"] == "quarantine_sonnet_candidate"
        for row in segments
    )
    with config.sonnet_csv_path.open(encoding="utf-8", newline="") as handle:
        candidates = {row["candidate_id"]: row for row in csv.DictReader(handle)}
    assert candidates[candidate_id]["candidate_decision"] == "exclude_manual_not_sonnet"


def test_audit_routes_nonstandard_language_sonnet_outside_standard_queue(tmp_path):
    config = _fixture(tmp_path)
    audit_gutenberg_extraction(config)
    with config.review_csv_path.open(encoding="utf-8", newline="") as handle:
        reviews = list(csv.DictReader(handle))
    reviews[0]["review_resolution"] = "exclude_nonstandard_language_sonnet"
    reviews[0]["review_rationale"] = "Verified sonnet, but not standard Italian."
    _write_csv(config.review_csv_path, reviews)

    report = audit_gutenberg_extraction(config)

    assert report["sonnet_decision_counts"]["conditioned_sonnet_candidate_not_activated"] == 1
    assert report["manual_review_resolution_counts"]["exclude_nonstandard_language_sonnet"] == 1
    with config.segment_csv_path.open(encoding="utf-8", newline="") as handle:
        segments = list(csv.DictReader(handle))
    assert any(
        row["ebook_id"] == "10"
        and row["segment_decision"] == "quarantine_conditioned_sonnet_candidate"
        and row["final_role"] == "conditioned_language_variant"
        for row in segments
    )


def test_audit_rejects_stale_manual_review_hash(tmp_path):
    config = _fixture(tmp_path)
    audit_gutenberg_extraction(config)
    with config.review_csv_path.open(encoding="utf-8", newline="") as handle:
        reviews = list(csv.DictReader(handle))
    reviews[0]["source_text_sha256"] = hashlib.sha256(b"changed").hexdigest()
    reviews[0]["review_resolution"] = "exclude_not_sonnet"
    reviews[0]["review_rationale"] = "Changed evidence must not be accepted."
    _write_csv(config.review_csv_path, reviews)

    with pytest.raises(ValueError, match="stale sonnet review evidence"):
        audit_gutenberg_extraction(config)
