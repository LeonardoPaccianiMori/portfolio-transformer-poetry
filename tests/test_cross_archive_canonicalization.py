import hashlib
from pathlib import Path

import pytest

from sonnet_corpus.cross_archive_canonicalization import (
    CanonicalUnit,
    _add_unit,
    _discover_pairs,
    _normalize_role,
    _overlap_row,
    _resolve_decisions,
    _verified_text,
)
from sonnet_corpus.gutenberg_fulltext_probe import TextReference, fingerprint_text


def _unit(tmp_path: Path, unit_id: str, text: str, *, priority: int, kind="broader", status="audited_candidate") -> CanonicalUnit:
    path = tmp_path / f"{unit_id.replace(':', '_')}.txt"
    path.write_text(text, encoding="utf-8")
    unit = CanonicalUnit(
        unit_id=unit_id,
        source_group="test",
        source_id=unit_id,
        unit_kind=kind,
        assigned_role="standard_sonnets" if kind == "standard_sonnet" else "historical_general",
        priority=priority,
        canonical_status=status,
        original_split="test" if status == "protected_v6_locked" else "",
        title=unit_id,
        author="Autore",
        source_archive="Test",
        source_url="https://example.test",
        reference=TextReference(unit_id, "test", path),
        expected_character_count=len(text),
        expected_sha256=hashlib.sha256(text.encode()).hexdigest(),
        input_artifact_status="test",
    )
    unit.fingerprint, _ = fingerprint_text(text, anchor_mask=0)
    return unit


def test_normalizes_ottocento_role_without_changing_other_roles():
    assert _normalize_role("ottocento_bridge_capped") == "nineteenth_century_bridge"
    assert _normalize_role("historical_general") == "historical_general"


def test_candidate_discovery_finds_exact_text(tmp_path):
    text = "uno due tre quattro cinque sei sette otto nove dieci undici dodici"
    left = _unit(tmp_path, "left", text, priority=1)
    right = _unit(tmp_path, "right", text, priority=2)
    assert ("left", "right") in _discover_pairs({
        "left": left.fingerprint,
        "right": right.fingerprint,
    })


def test_mutual_duplicate_excludes_lower_priority(tmp_path):
    text = "uno due tre quattro cinque sei sette otto nove dieci undici dodici"
    left = _unit(tmp_path, "left", text, priority=1)
    right = _unit(tmp_path, "right", text, priority=2)
    overlap = _overlap_row(left, right, "same_role", {
        "left_containment": 1.0,
        "right_containment": 1.0,
        "matching_shingles": 5,
    })
    decisions, reviews = _resolve_decisions({"left": left, "right": right}, [overlap])
    by_id = {row["unit_id"]: row for row in decisions}
    assert by_id["left"]["final_decision"] == "retain_canonical_candidate_7b"
    assert by_id["right"]["final_decision"] == "exclude_fully_covered_by_preferred_canonical"
    assert by_id["right"]["canonical_reference_ids"] == "left"
    assert reviews == []


def test_larger_lower_priority_text_keeps_only_unique_material(tmp_path):
    canonical = _unit(tmp_path, "canonical", "uno due tre quattro cinque sei sette otto nove", priority=1)
    composite = _unit(
        tmp_path,
        "composite",
        "prefazione nuova uno due tre quattro cinque sei sette otto nove coda unica",
        priority=4,
    )
    overlap = _overlap_row(canonical, composite, "same_role", {
        "left_containment": 1.0,
        "right_containment": 0.6,
        "matching_shingles": 2,
    })
    decisions, reviews = _resolve_decisions(
        {"canonical": canonical, "composite": composite}, [overlap],
    )
    by_id = {row["unit_id"]: row for row in decisions}
    assert by_id["canonical"]["final_decision"] == "retain_canonical_candidate_7b"
    assert by_id["composite"]["final_decision"] == "retain_unique_after_canonical_segment_quarantine_7b"
    assert by_id["composite"]["segment_quarantine_ids"] == "canonical"
    assert reviews[0]["resolution"] == "remove_exact_matched_span_in_checkpoint_7b"


def test_protected_sonnet_in_broader_text_is_quarantined(tmp_path):
    broader = _unit(tmp_path, "broader", "introduzione lunga e testo unico", priority=3)
    protected = _unit(
        tmp_path, "v6", "verso uno due tre quattro cinque sei sette otto",
        priority=0, kind="standard_sonnet", status="protected_v6_locked",
    )
    overlap = _overlap_row(broader, protected, "protected_cross_role", {
        "left_containment": 0.1,
        "right_containment": 1.0,
        "matching_shingles": 1,
    })
    decisions, reviews = _resolve_decisions({"broader": broader, "v6": protected}, [overlap])
    by_id = {row["unit_id"]: row for row in decisions}
    assert by_id["broader"]["final_decision"] == "retain_unique_after_protected_segment_quarantine_7b"
    assert by_id["broader"]["protected_v6_ids"] == "v6"
    assert by_id["v6"]["final_decision"] == "retain_protected_v6_split_locked"
    assert reviews[0]["review_type"] == "protected_v6_segment_quarantine"


def test_decision_rows_are_stably_sorted(tmp_path):
    units = {
        "z": _unit(tmp_path, "z", "uno due tre quattro cinque sei sette otto", priority=2),
        "a": _unit(tmp_path, "a", "nove dieci undici dodici tredici quattordici quindici sedici", priority=1),
    }
    decisions, _ = _resolve_decisions(units, [])
    assert [row["unit_id"] for row in decisions] == ["a", "z"]


def test_text_hash_and_character_count_are_fail_closed(tmp_path):
    unit = _unit(tmp_path, "sample", "uno due tre", priority=1)
    unit.expected_sha256 = "0" * 64
    with pytest.raises(ValueError, match="text hash mismatch"):
        _verified_text(unit)


def test_duplicate_unit_ids_are_rejected(tmp_path):
    unit = _unit(tmp_path, "same", "uno due tre quattro cinque sei sette otto", priority=1)
    units = {}
    _add_unit(units, unit)
    with pytest.raises(ValueError, match="duplicate canonical unit ID"):
        _add_unit(units, unit)
