import csv
import hashlib
import json
from pathlib import Path

import pytest

from sonnet_corpus.sonnet_v8_correction import (
    _assign_v8_splits,
    _render_4_4_3_3,
    _verify_pinned_tei_payload,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v8_rendering_is_exact_and_named_4_4_3_3():
    text = _render_4_4_3_3([f"line {number}" for number in range(1, 15)])

    assert text.split("\n\n") == [
        "line 1\nline 2\nline 3\nline 4",
        "line 5\nline 6\nline 7\nline 8",
        "line 9\nline 10\nline 11",
        "line 12\nline 13\nline 14",
    ]


def test_v8_split_quarantines_all_author_and_work_collisions():
    rows = [
        _split_row("heldout", "validation", "author:a", "work:held"),
        _split_row("author-collision", "train", "author:a", "work:new"),
        _split_row("work-collision", "train", "author:b", "work:held"),
        _split_row("clean", "train", "author:c", "work:clean"),
        _split_row("unresolved", "train", "author:d", "work:unresolved"),
    ]
    ledger = [
        {
            "unit_id": "unresolved",
            "resolution_status": "quarantine_unresolved_abbreviation",
        }
    ]

    _assign_v8_splits(rows, ledger)

    by_id = {row["unit_id"]: row for row in rows}
    assert by_id["heldout"]["v8_split"] == "validation"
    assert by_id["author-collision"]["v8_split"] == "quarantine"
    assert by_id["work-collision"]["v8_split"] == "quarantine"
    assert by_id["unresolved"]["v8_split"] == "quarantine"
    assert by_id["clean"]["v8_split"] == "train"
    assert by_id["clean"]["v8_training_eligible"] == "true"


def test_v8_ledger_exactly_covers_active_anthology_sonnets():
    with (ROOT / "data/metadata/sonnets_expanded_v8_corrections_v1.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        ledger = list(csv.DictReader(handle))
    with (ROOT / "data/metadata/sonnets_expanded_v7_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        active = {
            row["unit_id"]
            for row in csv.DictReader(handle)
            if row["include_in_v7"] == "true"
            and row["source_group"] == "bibit"
            and row["source_id"].split(":", 1)[0]
            in {"bibit000118", "bibit001444"}
        }

    assert len(active) == 69
    assert {row["unit_id"] for row in ledger} == active
    assert sum(row["source_record_id"] == "bibit000118" for row in ledger) == 48
    assert sum(row["source_record_id"] == "bibit001444" for row in ledger) == 21
    assert sum(row["resolution_status"].startswith("resolved_") for row in ledger) == 67
    assert sum(
        row["resolution_status"] == "quarantine_unresolved_abbreviation"
        for row in ledger
    ) == 2
    by_id = {row["unit_id"]: row for row in ledger}
    assert by_id["bibit:sonnet:bibit000118:sonnet_0016"]["corrected_author"] == "Torquato Tasso"
    assert by_id["bibit:sonnet:bibit000118:sonnet_0016"]["resolution_status"] == "resolved_curated_alias"


def test_v8_tei_source_hashes_are_pinned_and_reject_changed_payloads():
    payload = b"pinned TEI source"
    expected_sha256 = hashlib.sha256(payload).hexdigest()

    assert _verify_pinned_tei_payload("source", payload, expected_sha256) == expected_sha256
    with pytest.raises(ValueError, match="pinned TEI hash mismatch"):
        _verify_pinned_tei_payload("source", payload + b"changed", expected_sha256)


def test_generated_v8_view_enforces_correction_and_split_gates():
    output = ROOT / "data/processed/sonnets_expanded_v8"
    with (output / "records_manifest.csv").open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    with (output / "sonnets_manifest.csv").open(encoding="utf-8", newline="") as handle:
        sonnets = list(csv.DictReader(handle))
    with (output / "exclusions_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        exclusions = list(csv.DictReader(handle))
    with (output / "attribution_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        attributions = list(csv.DictReader(handle))
    overlap = json.loads(
        (ROOT / "reports/sonnets_expanded_v8_broader_heldout_overlap_v1.json").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads(
        (ROOT / "reports/sonnets_expanded_v8_correction_v1.json").read_text(
            encoding="utf-8"
        )
    )
    policy = json.loads(
        (ROOT / "data/metadata/sonnets_expanded_v8_policy_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert {"bibit:record:bibit000118", "bibit:record:bibit001444"}.isdisjoint(
        row["unit_id"] for row in records
    )
    assert all(row["line_count"] == "14" for row in sonnets)
    assert all(row["v8_rendering_policy"] == "derived_source_lines_4+4+3+3_v1" for row in sonnets)
    assert all(row["storage_kind"] == "v8_derived_4+4+3+3" for row in sonnets)
    assert all(
        row["source_storage_path"]
        and row["source_byte_start"]
        and row["source_byte_end"]
        and row["source_logical_sha256"]
        for row in (*records, *sonnets)
    )
    assert "ſ" not in (output / "derived_sonnets/part-0001.txt").read_text(encoding="utf-8")
    assert "ſ" not in (output / "derived_broader_long_s/part-0001.txt").read_text(encoding="utf-8")
    assert all(int(row["long_s_replacement_count"]) == 0 for row in records if row["storage_kind"] != "v8_derived_long_s_normalization")
    assert all(row["activation_status"] == "v8_future_training_view" for row in attributions)
    assert all(
        "replaces U+017F long s with ASCII s" in row["modification_notice"]
        for row in attributions
    )
    train_authors = {
        row["author_group_id"] for row in sonnets
        if row["v8_training_eligible"] == "true" and row["author_group_id"]
    }
    heldout_authors = {
        row["author_group_id"] for row in sonnets
        if row["v8_split"] in {"validation", "test"} and row["author_group_id"]
    }
    train_works = {
        row["work_group_id"] for row in sonnets if row["v8_training_eligible"] == "true"
    }
    heldout_works = {
        row["work_group_id"] for row in sonnets if row["v8_split"] in {"validation", "test"}
    }
    assert train_authors.isdisjoint(heldout_authors)
    assert train_works.isdisjoint(heldout_works)
    by_id = {row["unit_id"]: row for row in sonnets}
    assert {
        row["source_tei_sha256"]
        for row in sonnets
        if row["source_tei_sha256"]
    } == set(policy["tei_sources"].values())
    assert by_id["bibit:sonnet:bibit000118:sonnet_0012"]["supersedes_logical_sha256"]
    assert by_id["bibit:sonnet:bibit001444:structural_0009"]["supersedes_logical_sha256"]
    initial = overlap["pre_quarantine_audit"]
    final = overlap["post_quarantine_audit"]
    quarantined_ids = set(overlap["quarantined_broader_unit_ids"])
    assert initial["meaningful_pair_count"] == 69
    assert initial["maximum_heldout_containment"] == 0.79166667
    assert len(quarantined_ids) == overlap["quarantined_broader_unit_count"] == 41
    assert quarantined_ids == {
        row["broader_unit_id"] for row in initial["meaningful_pairs"]
    }
    assert quarantined_ids.isdisjoint(row["unit_id"] for row in records)
    assert quarantined_ids == {
        row["unit_id"]
        for row in exclusions
        if row["decision"] == "quarantine_initial_broader_heldout_overlap"
    }
    assert final["broader_training_unit_count"] == len(records) == 4501
    assert final["meaningful_pair_count"] == 0
    assert final["meaningful_pairs"] == []
    assert report["reviewed_authorship_count"] == 69
    assert report["resolved_authorship_count"] == 67
    assert report["unresolved_authorship_count"] == 2
    assert report["curated_alias_row_count"] == 45
    assert report["curated_alias_identity_count"] == 19
    assert report["gates"]["pinned_tei_sources_verified"] is True
    assert report["gates"]["exact_alias_evidence_verified"] is True
    assert report["gates"]["v8_attribution_notices_applied"] is True
    assert report["excluded_broader_record_count"] == len(exclusions) == 43


def _split_row(unit_id: str, split: str, author_group: str, work_group: str) -> dict[str, str]:
    return {
        "unit_id": unit_id,
        "v7_split": split,
        "training_eligible": "true" if split == "train" else "false",
        "canonical_author_key": author_group.removeprefix("author:"),
        "author_group_id": author_group,
        "work_group_id": work_group,
        "split_group_id": f"old:{unit_id}",
    }
