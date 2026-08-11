"""Global checkpoint-7A overlap and canonical-decision freeze.

This module indexes only text whose source-specific rights, extraction, and
quality audits are already complete.  It records decisions for checkpoint 7B;
it never materializes a new corpus or activates text.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from time import monotonic
from typing import Any

from sonnet_corpus.gutenberg_fulltext_probe import (
    TextFingerprint,
    TextReference,
    _normalized_words,
    _rolling_shingle_hashes,
    fingerprint_text,
    measure_word_shingle_containment,
)


AUDIT_VERSION = "cross_archive_canonicalization_v1"
AUDIT_DATE = "2026-08-11"
CONTAINMENT_THRESHOLD = 0.8

UNIT_FIELDS = (
    "unit_id", "source_group", "source_id", "unit_kind", "assigned_role",
    "canonical_priority", "canonical_status", "original_split", "title",
    "author", "source_archive", "source_url", "text_path", "byte_start",
    "byte_end", "cleaned_character_count", "cleaned_word_count",
    "cleaned_sha256", "normalized_word_sha256", "input_artifact_status",
)
OVERLAP_FIELDS = (
    "left_unit_id", "right_unit_id", "pair_scope", "left_containment",
    "right_containment", "matching_shingles", "exact_normalized_text",
    "preferred_unit_id", "decision_effect",
)
DECISION_FIELDS = (
    "unit_id", "unit_kind", "assigned_role", "canonical_priority",
    "overlap_ids", "canonical_reference_ids", "segment_quarantine_ids",
    "protected_v6_ids", "role_mismatch_ids", "final_decision",
    "next_action", "activation_status",
)
REVIEW_FIELDS = (
    "review_id", "unit_id", "reference_id", "review_type", "evidence",
    "resolution", "next_checkpoint",
)

Progress = Callable[[str], None]


@dataclass(frozen=True)
class CrossArchiveCanonicalizationConfig:
    repo_root: Path
    existing_historical_reports: tuple[Path, ...]
    v6_manifest_path: Path
    bibit_record_manifest_path: Path
    bibit_sonnet_manifest_path: Path
    gutenberg_record_manifest_path: Path
    gutenberg_sonnet_manifest_path: Path
    wikisource_record_manifest_path: Path
    wikisource_sonnet_manifest_path: Path
    liber_liber_record_manifest_path: Path
    liber_liber_sonnet_manifest_path: Path
    ilc_ota_unit_path: Path
    unit_index_path: Path
    overlap_path: Path
    decision_path: Path
    review_path: Path
    json_report_path: Path
    markdown_report_path: Path
    containment_threshold: float = CONTAINMENT_THRESHOLD
    sketch_size: int = 256
    anchor_mask: int = 1023
    audit_date: str = AUDIT_DATE


@dataclass
class CanonicalUnit:
    unit_id: str
    source_group: str
    source_id: str
    unit_kind: str
    assigned_role: str
    priority: int
    canonical_status: str
    original_split: str
    title: str
    author: str
    source_archive: str
    source_url: str
    reference: TextReference
    expected_character_count: int | None
    expected_sha256: str
    input_artifact_status: str
    text: str | None = field(default=None, repr=False)
    fingerprint: TextFingerprint | None = field(default=None, repr=False)


@dataclass
class _DecisionState:
    overlaps: set[str] = field(default_factory=set)
    covered_by: set[str] = field(default_factory=set)
    segment_quarantine: set[str] = field(default_factory=set)
    protected_v6: set[str] = field(default_factory=set)
    role_mismatch: set[str] = field(default_factory=set)


def run_cross_archive_canonicalization(
    config: CrossArchiveCanonicalizationConfig,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Build the deterministic decision-only checkpoint-7A artifacts."""

    _validate_config(config)
    started = monotonic()
    units = _load_units(config)
    _report(progress, f"inventory units={len(units)}")
    _fingerprint_units(units, config, progress)
    overlap_rows = _measure_overlaps(units, config, progress)
    decision_rows, review_rows = _resolve_decisions(units, overlap_rows)
    unit_rows = [
        _unit_row(unit, config.repo_root)
        for unit in sorted(units.values(), key=lambda item: item.unit_id)
    ]

    _write_csv(config.unit_index_path, UNIT_FIELDS, unit_rows)
    _write_csv(config.overlap_path, OVERLAP_FIELDS, overlap_rows)
    _write_csv(config.decision_path, DECISION_FIELDS, decision_rows)
    _write_csv(config.review_path, REVIEW_FIELDS, review_rows)

    report = _build_report(config, units, overlap_rows, decision_rows, review_rows)
    config.json_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.json_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config.markdown_report_path.write_text(_markdown_report(report), encoding="utf-8")
    _report(
        progress,
        f"complete elapsed={_duration(monotonic() - started)} units={len(units)} "
        f"overlaps={len(overlap_rows)} activated=0",
    )
    return report


def _load_units(config: CrossArchiveCanonicalizationConfig) -> dict[str, CanonicalUnit]:
    units: dict[str, CanonicalUnit] = {}
    for path in config.existing_historical_reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        for row in report["sources"]:
            source_id = row["source_id"]
            _add_unit(
                units,
                CanonicalUnit(
                    unit_id=f"existing:record:{source_id}",
                    source_group="existing_historical",
                    source_id=source_id,
                    unit_kind="broader",
                    assigned_role="historical_general",
                    priority=0,
                    canonical_status="existing_corpus_locked",
                    original_split="",
                    title=row["title"],
                    author=row["author"],
                    source_archive=row["source_archive"],
                    source_url=row["landing_page_url"],
                    reference=_whole_file_reference(
                        config.repo_root, f"existing:record:{source_id}", row["processed_path"],
                    ),
                    expected_character_count=int(row["cleaned_character_count"]),
                    expected_sha256="",
                    input_artifact_status="existing_materialized_corpus",
                ),
            )

    for row in _read_csv(config.v6_manifest_path):
        if row["include_in_expanded_with_petrarch"].casefold() != "true":
            continue
        split = row["split_expanded_with_petrarch"]
        poem_id = row["poem_id"]
        status = "protected_v6_locked" if split in {"validation", "test"} else "existing_v6_train_locked"
        _add_unit(
            units,
            CanonicalUnit(
                unit_id=f"v6:sonnet:{poem_id}", source_group="v6_sonnets",
                source_id=poem_id, unit_kind="standard_sonnet",
                assigned_role="standard_sonnets", priority=0,
                canonical_status=status, original_split=split,
                title=row["title_or_first_line"], author=row["author"],
                source_archive=row["source_archive"], source_url=row["source_url"],
                reference=_whole_file_reference(
                    config.repo_root, f"v6:sonnet:{poem_id}", row["clean_text_path"],
                ),
                expected_character_count=None, expected_sha256="",
                input_artifact_status="v6_split_locked",
            ),
        )

    _load_bibit(units, config)
    _load_gutenberg(units, config)
    _load_wikisource(units, config)
    _load_liber_liber(units, config)
    _load_ilc_ota(units, config)
    return units


def _load_bibit(units: dict[str, CanonicalUnit], config: CrossArchiveCanonicalizationConfig) -> None:
    for row in _read_csv(config.bibit_record_manifest_path):
        if row["artifact_status"] != "text_materialized":
            continue
        _add_manifest_unit(
            units, config, row, unit_id=f"bibit:record:{row['object_id']}",
            source_group="bibit", source_id=row["object_id"], unit_kind="broader",
            role=row["final_role"], priority=1, status="audited_candidate",
            title=row["title"], author=row["authors"], archive=row["source_archive"],
            url=row["source_url"], artifact_status=row["artifact_status"],
        )
    for row in _read_csv(config.bibit_sonnet_manifest_path):
        if row["final_role"] not in {"sonnet_core_standard_14_line", "sonnet_core_inferred_14_line"}:
            continue
        _add_manifest_unit(
            units, config, row, unit_id=f"bibit:sonnet:{row['candidate_id']}",
            source_group="bibit", source_id=row["candidate_id"], unit_kind="standard_sonnet",
            role="standard_sonnets", priority=1, status="audited_candidate",
            title=row["title"], author=row["author"], archive=row["source_archive"],
            url=row["source_url"], artifact_status=row["final_role"],
        )


def _load_gutenberg(units: dict[str, CanonicalUnit], config: CrossArchiveCanonicalizationConfig) -> None:
    for row in _read_csv(config.gutenberg_record_manifest_path):
        if row["artifact_status"] != "text_materialized_pending_v7":
            continue
        _add_manifest_unit(
            units, config, row, unit_id=f"gutenberg:record:pg{row['ebook_id']}",
            source_group="gutenberg", source_id=f"pg{row['ebook_id']}", unit_kind="broader",
            role=row["final_role"], priority=2, status="audited_candidate",
            title=row["title"], author=row["authors"], archive=row["source_archive"],
            url=row["source_url"], artifact_status=row["artifact_status"],
        )
    for row in _read_csv(config.gutenberg_sonnet_manifest_path):
        if row["artifact_status"] != "standard_sonnet_materialized_pending_v7":
            continue
        _add_manifest_unit(
            units, config, row, unit_id=f"gutenberg:sonnet:{row['candidate_id']}",
            source_group="gutenberg", source_id=row["candidate_id"], unit_kind="standard_sonnet",
            role="standard_sonnets", priority=2, status="audited_candidate",
            title=row["title"], author=row["poem_author"], archive=row["source_archive"],
            url=row["source_url"], artifact_status=row["artifact_status"],
        )


def _load_wikisource(units: dict[str, CanonicalUnit], config: CrossArchiveCanonicalizationConfig) -> None:
    for row in _read_csv(config.wikisource_record_manifest_path):
        if row["artifact_status"] != "text_materialized_inactive":
            continue
        _add_manifest_unit(
            units, config, row, unit_id=f"wikisource:record:{row['work_root_id']}",
            source_group="wikisource", source_id=row["work_root_id"], unit_kind="broader",
            role=row["final_role"], priority=3, status="audited_candidate",
            title=row["root_title"], author=row["author_evidence"], archive=row["source_archive"],
            url=row["source_url"], artifact_status=row["artifact_status"],
        )
    for row in _read_csv(config.wikisource_sonnet_manifest_path):
        if row["artifact_status"] != "sonnet_materialized_inactive":
            continue
        _add_manifest_unit(
            units, config, row, unit_id=f"wikisource:sonnet:{row['candidate_id']}",
            source_group="wikisource", source_id=row["candidate_id"], unit_kind="standard_sonnet",
            role="standard_sonnets", priority=3, status="audited_candidate",
            title=row["root_title"], author=row["poem_author"], archive="Italian Wikisource",
            url=row["source_url"], artifact_status=row["artifact_status"],
        )


def _load_liber_liber(units: dict[str, CanonicalUnit], config: CrossArchiveCanonicalizationConfig) -> None:
    for row in _read_csv(config.liber_liber_record_manifest_path):
        if row["artifact_status"] != "text_materialized_inactive":
            continue
        _add_manifest_unit(
            units, config, row, unit_id=f"liber_liber:record:{row['record_id']}",
            source_group="liber_liber", source_id=row["record_id"], unit_kind="broader",
            role=row["final_broader_role"], priority=4, status="audited_candidate",
            title=row["title"], author=row["author"], archive="Liber Liber",
            url=row["landing_page_url"], artifact_status=row["artifact_status"],
        )
    for row in _read_csv(config.liber_liber_sonnet_manifest_path):
        if row["artifact_status"] != "sonnet_materialized_inactive":
            continue
        _add_manifest_unit(
            units, config, row, unit_id=f"liber_liber:sonnet:{row['candidate_id']}",
            source_group="liber_liber", source_id=row["candidate_id"], unit_kind="standard_sonnet",
            role="standard_sonnets", priority=4, status="audited_candidate",
            title=row["source_title"], author=row["poem_author"], archive="Liber Liber",
            url=row["source_url"], artifact_status=row["artifact_status"],
        )


def _load_ilc_ota(units: dict[str, CanonicalUnit], config: CrossArchiveCanonicalizationConfig) -> None:
    for row in _read_csv(config.ilc_ota_unit_path):
        if row["probe_decision"] != "eligible_checkpoint_7_canonicalization_inactive":
            continue
        unit_id = f"ilc_ota:record:{row['unit_id']}"
        path = config.repo_root / row["cleaned_cache_path"]
        _add_unit(
            units,
            CanonicalUnit(
                unit_id=unit_id, source_group="ilc_ota", source_id=row["unit_id"],
                unit_kind="broader", assigned_role=_normalize_role(row["assigned_role"]),
                priority=5, canonical_status="audited_candidate", original_split="",
                title=row["title"], author=row["author"], source_archive=row["archive_id"],
                source_url="", reference=TextReference(unit_id, "ilc_ota", path),
                expected_character_count=int(row["cleaned_character_count"]),
                expected_sha256=row["cleaned_sha256"],
                input_artifact_status=row["probe_decision"],
            ),
        )


def _add_manifest_unit(
    units: dict[str, CanonicalUnit], config: CrossArchiveCanonicalizationConfig,
    row: dict[str, str], *, unit_id: str, source_group: str, source_id: str,
    unit_kind: str, role: str, priority: int, status: str, title: str,
    author: str, archive: str, url: str, artifact_status: str,
) -> None:
    reference = TextReference(
        unit_id, source_group, config.repo_root / row["shard_path"],
        int(row["byte_start"]), int(row["byte_end"]),
    )
    _add_unit(
        units,
        CanonicalUnit(
            unit_id=unit_id, source_group=source_group, source_id=source_id,
            unit_kind=unit_kind, assigned_role=_normalize_role(role), priority=priority,
            canonical_status=status, original_split="", title=title, author=author,
            source_archive=archive, source_url=url, reference=reference,
            expected_character_count=int(row["cleaned_character_count"]),
            expected_sha256=row["cleaned_sha256"], input_artifact_status=artifact_status,
        ),
    )


def _fingerprint_units(
    units: dict[str, CanonicalUnit], config: CrossArchiveCanonicalizationConfig,
    progress: Progress | None,
) -> None:
    sonnets = [unit for unit in units.values() if unit.unit_kind == "standard_sonnet"]
    broader = [unit for unit in units.values() if unit.unit_kind == "broader"]
    watched: dict[int, list[str]] = defaultdict(list)
    started = monotonic()
    for index, unit in enumerate(sorted(sonnets, key=lambda item: item.unit_id), 1):
        text = _verified_text(unit)
        unit.fingerprint, _ = fingerprint_text(
            text, sketch_size=config.sketch_size, anchor_mask=config.anchor_mask,
        )
        for value in set(_rolling_shingle_hashes(_normalized_words(text))):
            watched[value].append(unit.unit_id)
        unit.text = None
        if index == 1 or index == len(sonnets) or index % 1000 == 0:
            _report(progress, f"sonnet-index={index}/{len(sonnets)} elapsed={_duration(monotonic() - started)}")

    watched_frozen = {value: tuple(ids) for value, ids in watched.items()}
    started = monotonic()
    for index, unit in enumerate(sorted(broader, key=lambda item: item.unit_id), 1):
        text = _verified_text(unit)
        unit.fingerprint, hits = fingerprint_text(
            text, sketch_size=config.sketch_size, anchor_mask=config.anchor_mask,
            watched_shingles=watched_frozen,
        )
        unit.text = json.dumps(
            {key: sorted(values) for key, values in sorted(hits.items())},
            separators=(",", ":"),
        )
        if index == 1 or index == len(broader) or index % 100 == 0:
            _report(progress, f"broader-index={index}/{len(broader)} elapsed={_duration(monotonic() - started)}")


def _measure_overlaps(
    units: dict[str, CanonicalUnit], config: CrossArchiveCanonicalizationConfig,
    progress: Progress | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind in ("broader", "standard_sonnet"):
        fingerprints = {
            unit.unit_id: unit.fingerprint
            for unit in units.values()
            if unit.unit_kind == kind and unit.fingerprint is not None
        }
        pairs = _discover_pairs(fingerprints)
        exact_groups: dict[str, list[str]] = defaultdict(list)
        for unit_id, fingerprint in fingerprints.items():
            exact_groups[fingerprint.normalized_word_sha256].append(unit_id)
        for ids in exact_groups.values():
            pairs.update(combinations(sorted(ids), 2))
        _report(progress, f"pair-candidates kind={kind} count={len(pairs)}")
        for index, (left_id, right_id) in enumerate(sorted(pairs), 1):
            left_text = _verified_text(units[left_id])
            right_text = _verified_text(units[right_id])
            metrics = measure_word_shingle_containment(left_text, right_text)
            if metrics["containment"] >= config.containment_threshold:
                rows.append(_overlap_row(units[left_id], units[right_id], "same_role", metrics))
            if index % 250 == 0:
                _report(progress, f"pair-measure kind={kind} completed={index}/{len(pairs)}")

    sonnet_denominators = {
        unit.unit_id: len(set(_rolling_shingle_hashes(_normalized_words(_verified_text(unit)))))
        for unit in units.values() if unit.unit_kind == "standard_sonnet"
    }
    broader_units = sorted(
        (unit for unit in units.values() if unit.unit_kind == "broader"),
        key=lambda item: item.unit_id,
    )
    for index, unit in enumerate(broader_units, 1):
        hits = json.loads(unit.text or "{}")
        broader_text: str | None = None
        for sonnet_id, values in sorted(hits.items()):
            denominator = sonnet_denominators[sonnet_id]
            right_containment = len(values) / denominator if denominator else 0.0
            if right_containment < config.containment_threshold:
                continue
            sonnet = units[sonnet_id]
            if broader_text is None:
                broader_text = _verified_text(unit)
            metrics = measure_word_shingle_containment(
                broader_text, _verified_text(sonnet),
            )
            scope = "protected_cross_role" if sonnet.canonical_status == "protected_v6_locked" else "sonnet_cross_role"
            rows.append(_overlap_row(unit, sonnet, scope, metrics))
        unit.text = None
        if index == 1 or index == len(broader_units) or index % 250 == 0:
            _report(progress, f"cross-role={index}/{len(broader_units)}")
    return sorted(rows, key=lambda row: (row["pair_scope"], row["left_unit_id"], row["right_unit_id"]))


def _overlap_row(
    left: CanonicalUnit, right: CanonicalUnit, scope: str, metrics: dict[str, Any],
) -> dict[str, Any]:
    preferred = min((left, right), key=lambda item: (item.priority, item.unit_id))
    left_containment = float(metrics["left_containment"])
    right_containment = float(metrics["right_containment"])
    exact = bool(
        left.fingerprint and right.fingerprint
        and left.fingerprint.normalized_word_sha256 == right.fingerprint.normalized_word_sha256
    )
    if scope != "same_role":
        effect = "exclude_role_mismatch" if left_containment >= CONTAINMENT_THRESHOLD else "segment_quarantine_7b"
        preferred = right
    elif left_containment >= CONTAINMENT_THRESHOLD and right_containment >= CONTAINMENT_THRESHOLD:
        effect = f"exclude_lower_priority:{right.unit_id if preferred is left else left.unit_id}"
    elif left_containment >= CONTAINMENT_THRESHOLD:
        effect = "exclude_left" if preferred is right else "quarantine_left_span_from_right"
    else:
        effect = "exclude_right" if preferred is left else "quarantine_right_span_from_left"
    return {
        "left_unit_id": left.unit_id,
        "right_unit_id": right.unit_id,
        "pair_scope": scope,
        "left_containment": f"{left_containment:.6f}",
        "right_containment": f"{right_containment:.6f}",
        "matching_shingles": int(metrics["matching_shingles"]),
        "exact_normalized_text": str(exact).lower(),
        "preferred_unit_id": preferred.unit_id,
        "decision_effect": effect,
    }


def _resolve_decisions(
    units: dict[str, CanonicalUnit], overlaps: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    states = {unit_id: _DecisionState() for unit_id in units}
    reviews: list[dict[str, Any]] = []
    for row in overlaps:
        left_id, right_id = row["left_unit_id"], row["right_unit_id"]
        left, right = units[left_id], units[right_id]
        states[left_id].overlaps.add(right_id)
        states[right_id].overlaps.add(left_id)
        left_containment = float(row["left_containment"])
        right_containment = float(row["right_containment"])
        preferred_id = row["preferred_unit_id"]
        if row["pair_scope"] != "same_role":
            states[left_id].segment_quarantine.add(right_id)
            if right.canonical_status == "protected_v6_locked":
                states[left_id].protected_v6.add(right_id)
            review_type = "protected_v6_segment_quarantine" if row["pair_scope"] == "protected_cross_role" else "sonnet_segment_quarantine"
            resolution = "exclude_role_mismatch_from_broader" if left_containment >= CONTAINMENT_THRESHOLD else "remove_exact_matched_span_in_checkpoint_7b"
            if left_containment >= CONTAINMENT_THRESHOLD:
                states[left_id].covered_by.add(right_id)
                states[left_id].role_mismatch.add(right_id)
            reviews.append(_review_row(left_id, right_id, review_type, row, resolution))
            continue

        if left_containment >= CONTAINMENT_THRESHOLD and right_containment >= CONTAINMENT_THRESHOLD:
            loser_id = right_id if preferred_id == left_id else left_id
            states[loser_id].covered_by.add(preferred_id)
        elif left_containment >= CONTAINMENT_THRESHOLD:
            if preferred_id == right_id:
                states[left_id].covered_by.add(right_id)
            else:
                states[right_id].segment_quarantine.add(left_id)
                reviews.append(_review_row(right_id, left_id, "embedded_canonical_span", row, "remove_exact_matched_span_in_checkpoint_7b"))
        elif right_containment >= CONTAINMENT_THRESHOLD:
            if preferred_id == left_id:
                states[right_id].covered_by.add(left_id)
            else:
                states[left_id].segment_quarantine.add(right_id)
                reviews.append(_review_row(left_id, right_id, "embedded_canonical_span", row, "remove_exact_matched_span_in_checkpoint_7b"))

    decisions = []
    for unit in sorted(units.values(), key=lambda item: item.unit_id):
        state = states[unit.unit_id]
        if unit.canonical_status == "protected_v6_locked":
            decision = "retain_protected_v6_split_locked"
            action = "preserve identity and split; exclude from all training stages"
        elif unit.canonical_status in {"existing_v6_train_locked", "existing_corpus_locked"}:
            decision = "retain_existing_canonical_locked"
            action = "carry unchanged into checkpoint 7B"
        elif state.role_mismatch:
            decision = "exclude_broader_unit_misrouted_as_sonnet"
            action = "exclude broader unit; evaluate sonnet identity only in checkpoint 7B"
        elif state.covered_by:
            decision = "exclude_fully_covered_by_preferred_canonical"
            action = "retain provenance row; materialize no duplicate text"
        elif state.protected_v6:
            decision = "retain_unique_after_protected_segment_quarantine_7b"
            action = "remove protected span and verify zero residual overlap in checkpoint 7B"
        elif state.segment_quarantine:
            decision = "retain_unique_after_canonical_segment_quarantine_7b"
            action = "remove recorded canonical spans before checkpoint-7B build"
        else:
            decision = "retain_canonical_candidate_7b"
            action = "carry unchanged into checkpoint-7B role-specific build"
        decisions.append({
            "unit_id": unit.unit_id,
            "unit_kind": unit.unit_kind,
            "assigned_role": unit.assigned_role,
            "canonical_priority": unit.priority,
            "overlap_ids": ";".join(sorted(state.overlaps)),
            "canonical_reference_ids": ";".join(sorted(state.covered_by)),
            "segment_quarantine_ids": ";".join(sorted(state.segment_quarantine)),
            "protected_v6_ids": ";".join(sorted(state.protected_v6)),
            "role_mismatch_ids": ";".join(sorted(state.role_mismatch)),
            "final_decision": decision,
            "next_action": action,
            "activation_status": "inactive_decision_only",
        })
    return decisions, sorted(reviews, key=lambda row: row["review_id"])


def _review_row(
    unit_id: str, reference_id: str, review_type: str, overlap: dict[str, Any], resolution: str,
) -> dict[str, Any]:
    evidence = (
        f"left={overlap['left_containment']};right={overlap['right_containment']};"
        f"matching_shingles={overlap['matching_shingles']};preferred={overlap['preferred_unit_id']}"
    )
    return {
        "review_id": f"review:{unit_id}:{reference_id}",
        "unit_id": unit_id,
        "reference_id": reference_id,
        "review_type": review_type,
        "evidence": evidence,
        "resolution": resolution,
        "next_checkpoint": "7B",
    }


def _build_report(
    config: CrossArchiveCanonicalizationConfig, units: dict[str, CanonicalUnit],
    overlaps: list[dict[str, Any]], decisions: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    decision_counts = Counter(row["final_decision"] for row in decisions)
    decisions_by_id = {row["unit_id"]: row for row in decisions}
    role_counts = Counter(unit.assigned_role for unit in units.values())
    role_characters = Counter()
    retained_role_characters = Counter()
    training_role_characters = Counter()
    group_counts = Counter()
    group_characters = Counter()
    for unit in units.values():
        count = int(unit.expected_character_count or len(_verified_text(unit)))
        role_characters[unit.assigned_role] += count
        group_counts[unit.source_group] += 1
        group_characters[unit.source_group] += count
        decision = decisions_by_id[unit.unit_id]["final_decision"]
        if not decision.startswith("exclude_"):
            retained_role_characters[unit.assigned_role] += count
            if unit.canonical_status != "protected_v6_locked":
                training_role_characters[unit.assigned_role] += count
    input_paths = list(config.existing_historical_reports) + [
        config.v6_manifest_path, config.bibit_record_manifest_path,
        config.bibit_sonnet_manifest_path, config.gutenberg_record_manifest_path,
        config.gutenberg_sonnet_manifest_path, config.wikisource_record_manifest_path,
        config.wikisource_sonnet_manifest_path, config.liber_liber_record_manifest_path,
        config.liber_liber_sonnet_manifest_path, config.ilc_ota_unit_path,
    ]
    artifact_paths = [config.unit_index_path, config.overlap_path, config.decision_path, config.review_path]
    return {
        "report_version": AUDIT_VERSION,
        "audit_date": config.audit_date,
        "containment_method": "normalized_directional_8_word_shingles",
        "containment_threshold": config.containment_threshold,
        "canonical_precedence": [
            "protected_and_existing_v6_or_existing_historical",
            "bibit", "project_gutenberg", "italian_wikisource", "liber_liber", "ilc_ota",
            "stable_unit_id_tiebreak",
        ],
        "unit_count": len(units),
        "broader_unit_count": sum(unit.unit_kind == "broader" for unit in units.values()),
        "standard_sonnet_unit_count": sum(unit.unit_kind == "standard_sonnet" for unit in units.values()),
        "protected_v6_unit_count": sum(unit.canonical_status == "protected_v6_locked" for unit in units.values()),
        "unit_counts_by_source_group": dict(sorted(group_counts.items())),
        "characters_by_source_group": dict(sorted(group_characters.items())),
        "unit_counts_by_role": dict(sorted(role_counts.items())),
        "characters_by_role": dict(sorted(role_characters.items())),
        "whole_unit_retained_characters_by_role_before_segment_removal": dict(
            sorted(retained_role_characters.items())
        ),
        "whole_unit_training_characters_by_role_before_segment_removal": dict(
            sorted(training_role_characters.items())
        ),
        "whole_unit_character_count": sum(role_characters.values()),
        "whole_unit_retained_character_count_before_segment_removal": sum(
            retained_role_characters.values()
        ),
        "whole_unit_training_character_count_before_segment_removal": sum(
            training_role_characters.values()
        ),
        "segment_removal_character_count_frozen": False,
        "overlap_pair_count": len(overlaps),
        "overlap_counts_by_scope": dict(sorted(Counter(row["pair_scope"] for row in overlaps).items())),
        "decision_status_counts": dict(sorted(decision_counts.items())),
        "review_row_count": len(reviews),
        "input_sha256": {_relative(config.repo_root, path): _sha256(path) for path in input_paths},
        "artifact_sha256": {_relative(config.repo_root, path): _sha256(path) for path in artifact_paths},
        "conditioned_material_included": False,
        "metadata_only_archives_included": False,
        "text_activated": False,
        "final_corpus_materialized": False,
        "v7_created": False,
        "mixture_weights_assigned": False,
        "cache_deleted": False,
        "gpu_work_started": False,
        "next_checkpoint": "7B role-specific extraction, segment quarantine, and inactive final builds",
    }


def _markdown_report(report: dict[str, Any]) -> str:
    groups = "\n".join(
        f"| `{name}` | {count:,} | {report['characters_by_source_group'][name]:,} |"
        for name, count in report["unit_counts_by_source_group"].items()
    )
    decisions = "\n".join(
        f"- `{name}`: {count:,}"
        for name, count in report["decision_status_counts"].items()
    )
    scopes = "\n".join(
        f"- `{name}`: {count:,}" for name, count in report["overlap_counts_by_scope"].items()
    ) or "- No threshold overlaps."
    return f"""# Checkpoint 7A: Global Cross-Archive Canonicalization

Audit date: `{report['audit_date']}`

## Outcome

The decision-only index contains {report['unit_count']:,} audited units:
{report['broader_unit_count']:,} broader units and
{report['standard_sonnet_unit_count']:,} standard-sonnet units. It uses exact
normalized hashes plus directional normalized eight-word-shingle containment at
`{report['containment_threshold']}`. It records {report['overlap_pair_count']:,}
threshold overlap pairs and {report['review_row_count']:,} bounded checkpoint-7B
segment decisions.

The indexed units contain {report['whole_unit_character_count']:,} characters
before global exclusions. Removing only fully covered or role-misrouted whole
units leaves {report['whole_unit_retained_character_count_before_segment_removal']:,}
characters, including protected evaluation sonnets. The corresponding
pre-segment-removal training projection is
{report['whole_unit_training_character_count_before_segment_removal']:,}
characters. These are ceilings, not final corpus totals: checkpoint 7B must
remove the hash-pinned embedded spans before final character counts are frozen.

## Frozen input universe

| Source group | Units | Characters |
| --- | ---: | ---: |
{groups}

Only completed text-level audits are included. Conditioned material and the
metadata-only archive inventories remain inactive and outside this index.

## Overlap scopes

{scopes}

## Canonical decisions

{decisions}

Canonical precedence is protected/existing V6 and existing historical text,
then BibIt, Project Gutenberg, Italian Wikisource, Liber Liber, ILC/OTA, and a
stable unit-ID tie-break. A larger lower-priority unit containing a preferred
canonical unit is not silently discarded: its matched span is quarantined for
checkpoint 7B so unique material can be retained.

## Safety boundary

- Protected V6 validation/test identities remain split-locked.
- No conditioned text or metadata-only archive candidate is included.
- No corpus text is activated or newly materialized.
- No V7 split, mixture weight, cache deletion, or GPU work occurs.
- Next checkpoint: {report['next_checkpoint']}.
"""


def _unit_row(unit: CanonicalUnit, repo_root: Path) -> dict[str, Any]:
    fingerprint = unit.fingerprint
    if fingerprint is None:
        raise ValueError(f"missing fingerprint for {unit.unit_id}")
    text = _verified_text(unit)
    return {
        "unit_id": unit.unit_id,
        "source_group": unit.source_group,
        "source_id": unit.source_id,
        "unit_kind": unit.unit_kind,
        "assigned_role": unit.assigned_role,
        "canonical_priority": unit.priority,
        "canonical_status": unit.canonical_status,
        "original_split": unit.original_split,
        "title": unit.title,
        "author": unit.author,
        "source_archive": unit.source_archive,
        "source_url": unit.source_url,
        "text_path": _relative(repo_root, unit.reference.path),
        "byte_start": unit.reference.byte_start,
        "byte_end": "" if unit.reference.byte_end is None else unit.reference.byte_end,
        "cleaned_character_count": len(text),
        "cleaned_word_count": fingerprint.word_count,
        "cleaned_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "normalized_word_sha256": fingerprint.normalized_word_sha256,
        "input_artifact_status": unit.input_artifact_status,
    }


def _verified_text(unit: CanonicalUnit) -> str:
    if unit.text is not None and not unit.text.startswith("{"):
        return unit.text
    text = unit.reference.read_text()
    if unit.expected_character_count is not None and len(text) != unit.expected_character_count:
        raise ValueError(
            f"character-count mismatch for {unit.unit_id}: "
            f"{len(text)} != {unit.expected_character_count}"
        )
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if unit.expected_sha256 and digest != unit.expected_sha256:
        raise ValueError(f"text hash mismatch for {unit.unit_id}")
    return text


def _discover_pairs(fingerprints: dict[str, TextFingerprint]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for attribute in ("anchors", "sketch"):
        postings: dict[int, list[str]] = defaultdict(list)
        for unit_id, fingerprint in fingerprints.items():
            for value in getattr(fingerprint, attribute):
                postings[value].append(unit_id)
        collisions: Counter[tuple[str, str]] = Counter()
        for ids in postings.values():
            if 1 < len(ids) <= 40:
                collisions.update(combinations(sorted(ids), 2))
        for pair, count in collisions.items():
            denominator = min(
                len(getattr(fingerprints[pair[0]], attribute)),
                len(getattr(fingerprints[pair[1]], attribute)),
            )
            if denominator and count >= 2 and count / denominator >= 0.4:
                pairs.add(pair)
    return pairs


def _whole_file_reference(repo_root: Path, unit_id: str, relative_path: str) -> TextReference:
    return TextReference(unit_id, "existing", repo_root / relative_path)


def _normalize_role(role: str) -> str:
    return "nineteenth_century_bridge" if role == "ottocento_bridge_capped" else role


def _add_unit(units: dict[str, CanonicalUnit], unit: CanonicalUnit) -> None:
    if unit.unit_id in units:
        raise ValueError(f"duplicate canonical unit ID: {unit.unit_id}")
    units[unit.unit_id] = unit


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _report(progress: Progress | None, message: str) -> None:
    if progress:
        progress(message)


def _duration(seconds: float) -> str:
    value = int(seconds)
    return f"{value // 3600:02d}:{value % 3600 // 60:02d}:{value % 60:02d}"


def _validate_config(config: CrossArchiveCanonicalizationConfig) -> None:
    if not 0 < config.containment_threshold <= 1:
        raise ValueError("containment threshold must be in (0, 1]")
    if config.sketch_size <= 0 or config.anchor_mask < 0:
        raise ValueError("fingerprint settings must be non-negative")
    required = list(config.existing_historical_reports) + [
        config.v6_manifest_path, config.bibit_record_manifest_path,
        config.bibit_sonnet_manifest_path, config.gutenberg_record_manifest_path,
        config.gutenberg_sonnet_manifest_path, config.wikisource_record_manifest_path,
        config.wikisource_sonnet_manifest_path, config.liber_liber_record_manifest_path,
        config.liber_liber_sonnet_manifest_path, config.ilc_ota_unit_path,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing canonicalization inputs: {missing}")
