"""Build the inactive checkpoint-7B logical Italian corpora.

The build deliberately separates *logical* corpus membership from physical
storage.  Unchanged audited text remains a hash-pinned slice of its existing
committed shard.  Only ILC/OTA text and text rewritten by a quarantine decision
is written to a bounded delta shard.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import os
import re
import shutil
import statistics
import tempfile
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import monotonic
from typing import Any, BinaryIO

from .gutenberg_fulltext_probe import (
    _normalized_words,
    _rolling_shingle_hashes,
    measure_word_shingle_containment,
)


BUILD_VERSION = "canonical_italian_corpora_v1"
BUILD_DATE = "2026-08-11"
CONTAINMENT_THRESHOLD = 0.8
SHINGLE_SIZE = 8

SEGMENT_RANGE_FIELDS = (
    "range_id", "review_id", "unit_id", "reference_id", "review_type",
    "source_sha256", "reference_sha256", "localization_method",
    "reference_token_count", "matched_reference_token_count", "reference_coverage",
    "source_token_start", "source_token_end", "character_start", "character_end",
    "removed_character_count", "removed_sha256", "line_expanded", "range_decision",
    "verification_status",
)
SONNET_ROUTING_FIELDS = (
    "routing_id", "source_unit_id", "reference_unit_id", "candidate_id", "source_group",
    "source_title", "poem_author", "character_start", "character_end", "line_count",
    "source_text_sha256", "cleaned_character_count", "cleaned_sha256", "form_evidence", "language_evidence",
    "overlap_reference_ids", "routing_decision", "final_role", "activation_status",
)
RECORD_FIELDS = (
    "unit_id", "source_group", "source_id", "title", "author", "source_archive",
    "source_url", "epoch_bucket", "final_role", "canonical_priority", "final_decision",
    "quarantine_status", "attribution_id", "logical_character_count", "logical_byte_count",
    "logical_sha256", "storage_kind", "storage_path", "byte_start", "byte_end",
    "activation_status", "training_eligible",
)
SONNET_FIELDS = (
    "unit_id", "source_group", "source_id", "title", "author", "source_archive",
    "source_url", "epoch_bucket", "original_split", "final_decision", "routing_status",
    "attribution_id", "line_count", "logical_character_count", "logical_byte_count",
    "logical_sha256", "storage_kind", "storage_path", "byte_start", "byte_end",
    "activation_status", "training_eligible",
)
STORAGE_FIELDS = (
    "storage_id", "unit_id", "unit_kind", "final_role", "storage_kind", "storage_path",
    "byte_start", "byte_end", "logical_character_count", "logical_byte_count",
    "logical_sha256", "physical_file_sha256", "public_repository_status",
)
ATTRIBUTION_FIELDS = (
    "attribution_id", "source_group", "origin_id", "source_archive", "title", "author",
    "source_url", "rights_label", "license_url", "required_notice", "modification_notice",
    "upstream_attribution_artifact", "activation_status",
)

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_ROMAN = re.compile(r"^[IVXLCDM]+[.)]?$", re.IGNORECASE)
_PRINTED_LINE_NUMBER = re.compile(r"^(?:5|10)\s+")
_TRAILING_SOURCE_NUMBER = re.compile(r"\s+(?:\d+|I)\s*$")
_SONNET_COLLECTION = re.compile(r"\b(?:rime|amori|sonett[io])\b", re.IGNORECASE)
_CORPUS_ROLES = {
    "historical_general", "historical_non_sonnet_poetry",
    "nineteenth_century_bridge", "standard_sonnets",
}

Progress = Callable[[str], None]


@dataclass(frozen=True)
class PositionedToken:
    value: str
    start: int
    end: int


@dataclass(frozen=True)
class LocalizedRange:
    source_token_start: int
    source_token_end: int
    character_start: int
    character_end: int
    matched_reference_tokens: int
    reference_token_count: int
    coverage: float
    line_expanded: bool
    localization_method: str = "contiguous_shingle_window_alignment"


@dataclass(frozen=True)
class LogicalLocation:
    storage_kind: str
    path: str
    byte_start: int
    byte_end: int
    physical_file_sha256: str


@dataclass(frozen=True)
class CanonicalCorpusBuildConfig:
    repo_root: Path
    unit_path: Path
    decision_path: Path
    review_path: Path
    overlap_path: Path
    ilc_ota_unit_path: Path
    ilc_ota_inventory_path: Path
    v6_manifest_path: Path
    bibit_period_path: Path
    gutenberg_period_path: Path
    gutenberg_attribution_path: Path
    wikisource_period_path: Path
    wikisource_sonnet_path: Path
    wikisource_attribution_path: Path
    liber_liber_period_path: Path
    liber_liber_sonnet_path: Path
    liber_liber_attribution_path: Path
    existing_historical_reports: tuple[Path, ...]
    segment_range_path: Path
    sonnet_routing_path: Path
    output_dir: Path
    json_report_path: Path
    markdown_report_path: Path
    containment_threshold: float = CONTAINMENT_THRESHOLD
    max_shard_bytes: int = 64 * 1024 * 1024
    progress_interval: int = 250
    build_date: str = BUILD_DATE


def positioned_tokens(text: str) -> list[PositionedToken]:
    """Return overlap-normalized alphabetic tokens with original character bounds."""

    tokens: list[PositionedToken] = []
    for match in _WORD.finditer(text):
        folded = unicodedata.normalize("NFKD", match.group(0).casefold())
        value = "".join(char for char in folded if not unicodedata.combining(char))
        if value:
            tokens.append(PositionedToken(value, match.start(), match.end()))
    return tokens


def locate_reference_ranges(
    source_text: str,
    reference_text: str,
    *,
    threshold: float = CONTAINMENT_THRESHOLD,
    expand_to_lines: bool = False,
) -> list[LocalizedRange]:
    """Locate every unambiguous occurrence covering at least ``threshold`` of a reference.

    Shared eight-token shingles create bounded candidate windows.  A local
    ``SequenceMatcher`` then aligns spelling variants without normalizing or
    rewriting the source text.  Distinct verified occurrences are all returned;
    overlapping candidate occurrences fail closed.
    """

    source = positioned_tokens(source_text)
    reference = positioned_tokens(reference_text)
    if len(reference) < SHINGLE_SIZE or len(source) < SHINGLE_SIZE:
        return []
    reference_shingles: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index in range(len(reference) - SHINGLE_SIZE + 1):
        reference_shingles[tuple(token.value for token in reference[index:index + SHINGLE_SIZE])].append(index)

    hits: list[tuple[int, int]] = []
    for source_index in range(len(source) - SHINGLE_SIZE + 1):
        key = tuple(token.value for token in source[source_index:source_index + SHINGLE_SIZE])
        for reference_index in reference_shingles.get(key, ()):
            hits.append((source_index, reference_index))
    if not hits:
        return []

    # Separate distant occurrences.  Within one occurrence, insertions may move
    # the diagonal, so source proximity is a safer cluster key than exact offset.
    gap = max(48, len(reference) * 2)
    diagonal_tolerance = max(3, len(reference) // 10)
    diagonal_groups: list[list[tuple[int, int]]] = []
    for hit in sorted(set(hits), key=lambda item: (item[0] - item[1], item[0], item[1])):
        diagonal = hit[0] - hit[1]
        if not diagonal_groups:
            diagonal_groups.append([hit])
            continue
        previous_diagonal = diagonal_groups[-1][-1][0] - diagonal_groups[-1][-1][1]
        if diagonal - previous_diagonal > diagonal_tolerance:
            diagonal_groups.append([hit])
        else:
            diagonal_groups[-1].append(hit)
    clusters: list[list[tuple[int, int]]] = []
    for diagonal_group in diagonal_groups:
        group_clusters: list[list[tuple[int, int]]] = []
        for hit in sorted(diagonal_group):
            if not group_clusters or hit[0] - group_clusters[-1][-1][0] > gap:
                group_clusters.append([hit])
            else:
                group_clusters[-1].append(hit)
        clusters.extend(group_clusters)

    localized: list[LocalizedRange] = []
    partial: list[tuple[LocalizedRange, set[int]]] = []
    ref_values = [token.value for token in reference]
    margin = max(8, len(reference) // 5)
    for cluster in clusters:
        diagonals = sorted(item[0] - item[1] for item in cluster)
        expected_start = diagonals[len(diagonals) // 2]
        window_start = max(0, expected_start - margin)
        window_end = min(len(source), expected_start + len(reference) + margin)
        window_values = [token.value for token in source[window_start:window_end]]
        matcher = difflib.SequenceMatcher(None, ref_values, window_values, autojunk=False)
        blocks = [block for block in matcher.get_matching_blocks() if block.size]
        matched = sum(block.size for block in blocks)
        coverage = matched / len(reference)
        if not blocks:
            continue
        matched_source_start = window_start + min(block.b for block in blocks)
        matched_source_end = window_start + max(block.b + block.size for block in blocks)
        char_start = source[matched_source_start].start
        char_end = source[matched_source_end - 1].end
        if expand_to_lines:
            char_start, char_end = _line_bounds(source_text, char_start, char_end)
        candidate = LocalizedRange(
            source_token_start=matched_source_start,
            source_token_end=matched_source_end,
            character_start=char_start,
            character_end=char_end,
            matched_reference_tokens=matched,
            reference_token_count=len(reference),
            coverage=coverage,
            line_expanded=expand_to_lines,
        )
        reference_positions = {
            position
            for block in blocks
            for position in range(block.a, block.a + block.size)
        }
        partial.append((candidate, reference_positions))
        if coverage >= threshold:
            localized.append(candidate)

    if not localized and partial:
        covered_reference_positions: set[int] = set()
        for candidate, positions in partial:
            if candidate.matched_reference_tokens >= SHINGLE_SIZE:
                covered_reference_positions.update(positions)
        aggregate_coverage = len(covered_reference_positions) / len(reference)
        if aggregate_coverage >= threshold:
            localized = [
                LocalizedRange(
                    **{
                        **candidate.__dict__,
                        "localization_method": "fragmented_multi_span_shingle_alignment",
                    }
                )
                for candidate, _positions in partial
                if candidate.matched_reference_tokens >= SHINGLE_SIZE
            ]
    if not localized:
        shingle_ranges: list[tuple[LocalizedRange, set[int]]] = []
        all_reference_positions: set[int] = set()
        for group in diagonal_groups:
            source_start = min(item[0] for item in group)
            source_end = max(item[0] + SHINGLE_SIZE for item in group)
            reference_positions = {
                position
                for _source_index, reference_index in group
                for position in range(reference_index, reference_index + SHINGLE_SIZE)
            }
            all_reference_positions.update(reference_positions)
            char_start = source[source_start].start
            char_end = source[source_end - 1].end
            if expand_to_lines:
                char_start, char_end = _line_bounds(source_text, char_start, char_end)
            shingle_ranges.append((LocalizedRange(
                source_token_start=source_start, source_token_end=source_end,
                character_start=char_start, character_end=char_end,
                matched_reference_tokens=len(reference_positions),
                reference_token_count=len(reference),
                coverage=len(reference_positions) / len(reference),
                line_expanded=expand_to_lines,
                localization_method="fragmented_shingle_union_alignment",
            ), reference_positions))
        if len(all_reference_positions) / len(reference) >= threshold:
            localized = [candidate for candidate, _positions in shingle_ranges]

    localized.sort(key=lambda item: (item.character_start, item.character_end))
    for previous, current in zip(localized, localized[1:]):
        if current.character_start < previous.character_end:
            raise ValueError("ambiguous overlapping localized occurrences")
    return localized


def merge_character_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(set(ranges)):
        if start < 0 or end <= start:
            raise ValueError(f"invalid character range: {start}:{end}")
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def remove_character_ranges(text: str, ranges: Sequence[tuple[int, int]]) -> str:
    """Remove verified source spans without changing retained characters."""

    cursor = 0
    parts: list[str] = []
    for start, end in merge_character_ranges(ranges):
        if end > len(text):
            raise ValueError("character range exceeds source text")
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def run_canonical_corpus_build(
    config: CanonicalCorpusBuildConfig,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Resolve checkpoint 7B and install a deterministic inactive logical build."""

    _validate_config(config)
    started = monotonic()
    units = _unique(_read_csv(config.unit_path), "unit_id")
    decisions = _unique(_read_csv(config.decision_path), "unit_id")
    reviews = _read_csv(config.review_path)
    review_by_unit: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in reviews:
        review_by_unit[row["unit_id"]].append(row)
    if set(units) != set(decisions):
        raise ValueError("unit and decision universes differ")

    _report(progress, f"inventory units={len(units)} reviews={len(reviews)}")
    rewritten: dict[str, str] = {}
    range_rows: list[dict[str, Any]] = []
    held_units: dict[str, str] = {}
    for index, unit_id in enumerate(sorted(review_by_unit), 1):
        decision = decisions[unit_id]["final_decision"]
        unit_reviews = sorted(review_by_unit[unit_id], key=lambda row: row["review_id"])
        if decision.startswith("exclude_"):
            for review in unit_reviews:
                range_rows.append(_excluded_review_range(review, units[unit_id]))
            continue
        source_text = _read_verified_unit(config.repo_root, units[unit_id])
        source_hash = _sha(source_text)
        removal: list[tuple[int, int]] = []
        unit_range_rows: list[dict[str, Any]] = []
        try:
            for review in unit_reviews:
                reference = units[review["reference_id"]]
                reference_text = _read_verified_unit(config.repo_root, reference)
                occurrences = locate_reference_ranges(
                    source_text,
                    reference_text,
                    threshold=config.containment_threshold,
                    expand_to_lines=reference["unit_kind"] == "standard_sonnet",
                )
                if not occurrences:
                    raise ValueError(f"no verified occurrence for {review['reference_id']}")
                for occurrence_index, occurrence in enumerate(occurrences, 1):
                    removed = source_text[occurrence.character_start:occurrence.character_end]
                    removal.append((occurrence.character_start, occurrence.character_end))
                    unit_range_rows.append(_localized_range_row(
                        review, source_hash, _sha(reference_text), occurrence,
                        removed, occurrence_index,
                    ))
            rewritten_text = remove_character_ranges(source_text, removal)
            for review in unit_reviews:
                residual = measure_word_shingle_containment(
                    rewritten_text,
                    _read_verified_unit(config.repo_root, units[review["reference_id"]]),
                )
                if float(residual["right_containment"]) >= config.containment_threshold:
                    raise ValueError(f"residual reference containment for {review['reference_id']}")
            if not rewritten_text.strip():
                raise ValueError("quarantine removed the entire unit")
            rewritten[unit_id] = rewritten_text
            range_rows.extend(unit_range_rows)
        except ValueError as error:
            held_units[unit_id] = str(error)
            range_rows.extend(_held_review_ranges(unit_reviews, units[unit_id], source_hash, str(error)))
        if progress and (index == 1 or index % 25 == 0 or index == len(review_by_unit)):
            _report(progress, f"localize={index}/{len(review_by_unit)} held={len(held_units)}")

    routing_rows, new_sonnets = _route_sonnets(config, units, decisions, rewritten, progress)
    routed_removals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in routing_rows:
        if row["routing_decision"] in {
            "retain_new_verified_standard_sonnet_inactive",
            "exclude_existing_canonical_sonnet_overlap",
            "exclude_new_intra_ilc_ota_duplicate",
        } and row["character_start"] != "":
            routed_removals[row["source_unit_id"]].append(
                (int(row["character_start"]), int(row["character_end"]))
            )
    for unit_id, removals in routed_removals.items():
        source = rewritten.get(unit_id) or _read_verified_unit(config.repo_root, units[unit_id])
        rewritten[unit_id] = remove_character_ranges(source, removals)
    for candidate_id, candidate in new_sonnets.items():
        residual = measure_word_shingle_containment(
            rewritten[candidate["source_unit_id"]], candidate["text"],
        )
        if float(residual["right_containment"]) >= config.containment_threshold:
            raise ValueError(f"routed sonnet remains in broader text: {candidate_id}")

    protected_shingles = {
        unit_id: set(_rolling_shingle_hashes(_normalized_words(_read_verified_unit(config.repo_root, unit))))
        for unit_id, unit in units.items()
        if decisions[unit_id]["final_decision"] == "retain_protected_v6_split_locked"
    }
    for unit_id, text in sorted(rewritten.items()):
        source_shingles = set(_rolling_shingle_hashes(_normalized_words(text)))
        for protected_id, reference_shingles in protected_shingles.items():
            containment = (
                len(source_shingles & reference_shingles) / len(reference_shingles)
                if reference_shingles else 0.0
            )
            if containment >= config.containment_threshold:
                raise ValueError(
                    f"modified broader unit overlaps protected V6 sonnet: {unit_id} -> {protected_id}"
                )
    if held_units:
        details = "; ".join(f"{key}: {value}" for key, value in sorted(held_units.items()))
        raise ValueError(f"fail-closed span localization holds ({len(held_units)}): {details}")

    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{config.output_dir.name}.", dir=config.output_dir.parent))
    final_prefix = config.output_dir.relative_to(config.repo_root).as_posix()
    writers = {
        role: _DeltaShardWriter(temp_dir / role, f"{final_prefix}/{role}", config.max_shard_bytes)
        for role in sorted(_CORPUS_ROLES)
    }
    record_rows: list[dict[str, Any]] = []
    sonnet_rows: list[dict[str, Any]] = []
    storage_rows: list[dict[str, Any]] = []
    attribution_rows: dict[str, dict[str, Any]] = {}
    try:
        metadata = _MetadataIndex(config)
        retained_modified: dict[str, str] = dict(rewritten)
        retained_modified.update({candidate_id: row["text"] for candidate_id, row in new_sonnets.items()})
        for index, unit_id in enumerate(sorted(units), 1):
            unit = units[unit_id]
            decision = decisions[unit_id]
            attribution = metadata.attribution(unit)
            attribution_rows.setdefault(attribution["attribution_id"], attribution)
            if unit["unit_kind"] == "broader":
                row, storage = _record_manifest_row(
                    config, unit, decision, rewritten.get(unit_id), writers,
                    metadata, attribution["attribution_id"],
                )
                record_rows.append(row)
            else:
                row, storage = _sonnet_manifest_row(
                    config, unit, decision, writers, metadata,
                    attribution["attribution_id"],
                )
                sonnet_rows.append(row)
            if storage is not None:
                storage_rows.append(storage)
            if progress and (index == 1 or index % config.progress_interval == 0 or index == len(units)):
                _report(progress, f"manifest={index}/{len(units)}")

        for candidate_id, candidate in sorted(new_sonnets.items()):
            source_unit = units[candidate["source_unit_id"]]
            attribution = metadata.attribution(source_unit)
            attribution_rows.setdefault(attribution["attribution_id"], attribution)
            location = writers["standard_sonnets"].add(candidate_id, candidate["text"])
            stats = _stats(candidate["text"])
            sonnet_rows.append({
                "unit_id": candidate_id, "source_group": source_unit["source_group"],
                "source_id": source_unit["source_id"], "title": candidate["first_line"],
                "author": candidate["author"], "source_archive": source_unit["source_archive"],
                "source_url": metadata.source_url(source_unit),
                "epoch_bucket": metadata.epoch(source_unit), "original_split": "",
                "final_decision": "retain_new_verified_standard_sonnet_inactive",
                "routing_status": "new_unique_exact_14_line",
                "attribution_id": attribution["attribution_id"], "line_count": 14,
                **stats, **_location_fields(location), "activation_status": "inactive_pending_v7",
                "training_eligible": "true",
            })
            storage_rows.append(_storage_row(candidate_id, "standard_sonnet", "standard_sonnets", location, stats))

        shard_reports = [report for writer in writers.values() for report in writer.close()]
        shard_hashes = {row["path"]: row["sha256"] for row in shard_reports}
        for row in storage_rows:
            if row["storage_kind"] == "checkpoint_7b_delta_slice":
                row["physical_file_sha256"] = shard_hashes[row["storage_path"]]
        _write_csv(temp_dir / "records_manifest.csv", RECORD_FIELDS, record_rows)
        _write_csv(temp_dir / "sonnets_manifest.csv", SONNET_FIELDS, sonnet_rows)
        _write_csv(temp_dir / "storage_manifest.csv", STORAGE_FIELDS, storage_rows)
        _write_csv(temp_dir / "attribution_manifest.csv", ATTRIBUTION_FIELDS, attribution_rows.values())
        report = _build_report(
            config, record_rows, sonnet_rows, storage_rows, range_rows, routing_rows,
            attribution_rows, shard_reports, len(rewritten), monotonic() - started,
        )
        _write_json(temp_dir / "build_report.json", report)
        _validate_temp_build(config, temp_dir, record_rows, sonnet_rows, storage_rows)
        _replace_verified_output(temp_dir, config.output_dir)
    except BaseException:
        for writer in writers.values():
            writer.abort()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    _write_csv(config.segment_range_path, SEGMENT_RANGE_FIELDS, range_rows)
    _write_csv(config.sonnet_routing_path, SONNET_ROUTING_FIELDS, routing_rows)
    _write_json(config.json_report_path, report)
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(_render_markdown(report), encoding="utf-8")
    _report(progress, f"complete elapsed={_duration(monotonic()-started)} logical_chars={report['logical_character_count']} activated=0")
    return report


def _line_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    line_start = text.rfind("\n", 0, start) + 1
    newline = text.find("\n", end)
    line_end = len(text) if newline < 0 else newline + 1
    return line_start, line_end


def _localized_range_row(
    review: dict[str, str], source_sha: str, reference_sha: str,
    occurrence: LocalizedRange, removed: str, occurrence_index: int,
) -> dict[str, Any]:
    return {
        "range_id": f"{review['review_id']}:occurrence_{occurrence_index:02d}",
        "review_id": review["review_id"], "unit_id": review["unit_id"],
        "reference_id": review["reference_id"], "review_type": review["review_type"],
        "source_sha256": source_sha, "reference_sha256": reference_sha,
        "localization_method": occurrence.localization_method,
        "reference_token_count": occurrence.reference_token_count,
        "matched_reference_token_count": occurrence.matched_reference_tokens,
        "reference_coverage": f"{occurrence.coverage:.6f}",
        "source_token_start": occurrence.source_token_start,
        "source_token_end": occurrence.source_token_end,
        "character_start": occurrence.character_start, "character_end": occurrence.character_end,
        "removed_character_count": len(removed), "removed_sha256": _sha(removed),
        "line_expanded": str(occurrence.line_expanded).lower(),
        "range_decision": "remove_verified_occurrence",
        "verification_status": "residual_reference_below_threshold",
    }


def _excluded_review_range(review: dict[str, str], unit: dict[str, str]) -> dict[str, Any]:
    return {
        "range_id": f"{review['review_id']}:whole_unit_excluded", "review_id": review["review_id"],
        "unit_id": review["unit_id"], "reference_id": review["reference_id"],
        "review_type": review["review_type"], "source_sha256": unit["cleaned_sha256"],
        "reference_sha256": "", "localization_method": "not_required_whole_unit_excluded",
        "reference_token_count": "", "matched_reference_token_count": "", "reference_coverage": "",
        "source_token_start": "", "source_token_end": "", "character_start": "",
        "character_end": "", "removed_character_count": unit["cleaned_character_count"],
        "removed_sha256": unit["cleaned_sha256"], "line_expanded": "false",
        "range_decision": "exclude_whole_unit_by_7a_decision",
        "verification_status": "duplicate_representation_not_materialized",
    }


def _held_review_ranges(
    reviews: Sequence[dict[str, str]], unit: dict[str, str], source_sha: str, reason: str,
) -> list[dict[str, Any]]:
    rows = []
    for review in reviews:
        row = _excluded_review_range(review, unit)
        row.update({
            "range_id": f"{review['review_id']}:hold", "source_sha256": source_sha,
            "localization_method": "fail_closed", "removed_character_count": 0,
            "removed_sha256": "", "range_decision": "hold_whole_unit_unmaterialized",
            "verification_status": reason,
        })
        rows.append(row)
    return rows


def _route_sonnets(
    config: CanonicalCorpusBuildConfig,
    units: dict[str, dict[str, str]],
    decisions: dict[str, dict[str, str]],
    rewritten: dict[str, str],
    progress: Progress | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    new: dict[str, dict[str, Any]] = {}
    canonical_sonnets = {
        unit_id: _read_verified_unit(config.repo_root, unit)
        for unit_id, unit in units.items()
        if unit["unit_kind"] == "standard_sonnet"
        and not decisions[unit_id]["final_decision"].startswith("exclude_")
    }
    shingle_index: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for unit_id, text in canonical_sonnets.items():
        values = [token.value for token in positioned_tokens(text)]
        for index in range(max(0, len(values) - SHINGLE_SIZE + 1)):
            shingle_index[tuple(values[index:index + SHINGLE_SIZE])].add(unit_id)

    for unit_id, decision in sorted(decisions.items()):
        if decision["final_decision"] != "exclude_broader_unit_misrouted_as_sonnet":
            continue
        references = sorted(set(filter(None, decision["role_mismatch_ids"].split(";"))))
        source = units[unit_id]
        rows.append({
            "routing_id": f"route:{unit_id}", "source_unit_id": unit_id,
            "reference_unit_id": ";".join(references), "candidate_id": "",
            "source_group": source["source_group"], "source_title": source["title"],
            "poem_author": source["author"], "character_start": "", "character_end": "",
            "line_count": "", "cleaned_character_count": source["cleaned_character_count"],
            "source_text_sha256": source["cleaned_sha256"],
            "cleaned_sha256": source["cleaned_sha256"],
            "form_evidence": "7A broader root is covered by canonical standard-sonnet identity",
            "language_evidence": "source-specific standard queue already passed",
            "overlap_reference_ids": ";".join(references),
            "routing_decision": "exclude_duplicate_broader_representation_reference_canonical_sonnet",
            "final_role": "standard_sonnets", "activation_status": "inactive_pending_v7",
        })

    ilc_units = [
        unit for unit_id, unit in units.items()
        if unit["source_group"] == "ilc_ota"
        and not decisions[unit_id]["final_decision"].startswith("exclude_")
    ]
    for unit_index, unit in enumerate(sorted(ilc_units, key=lambda row: row["unit_id"]), 1):
        unit_id = unit["unit_id"]
        text = rewritten.get(unit_id) or _read_verified_unit(config.repo_root, unit)
        for block_index, (start, end, raw_lines) in enumerate(_blank_separated_blocks(text), 1):
            if len(raw_lines) != 14:
                continue
            cleaned_lines = [_clean_verse_line(line) for line in raw_lines]
            cleaned_lines = [line for line in cleaned_lines if line]
            candidate_text = "\n".join(cleaned_lines) + "\n"
            candidate_id = f"ilc_ota:sonnet:{unit['source_id']}:block_{block_index:04d}"
            values = [token.value for token in positioned_tokens(candidate_text)] if len(cleaned_lines) == 14 else []
            possible: set[str] = set()
            for index in range(max(0, len(values) - SHINGLE_SIZE + 1)):
                possible.update(shingle_index.get(tuple(values[index:index + SHINGLE_SIZE]), ()))
            overlaps = []
            for reference_id in sorted(possible):
                metrics = measure_word_shingle_containment(candidate_text, canonical_sonnets[reference_id])
                if float(metrics["containment"]) >= config.containment_threshold:
                    overlaps.append(reference_id)
            structurally_supported = (
                unit["source_archive"] == "oxford_text_archive"
                and bool(_SONNET_COLLECTION.search(unit["title"]))
                and unit["author"].strip()
                and unit["author"] != "ILC-CNR digital edition"
            )
            standard_layout = _is_standard_sonnet_layout(cleaned_lines)
            if len(cleaned_lines) != 14:
                route = "hold_14_line_raw_block_contains_structural_number_or_page_artifact"
            elif overlaps:
                route = "exclude_existing_canonical_sonnet_overlap"
            elif not structurally_supported:
                route = "hold_14_line_block_insufficient_sonnet_or_attribution_evidence"
            elif any(_ROMAN.fullmatch(line.strip()) for line in cleaned_lines):
                route = "hold_14_line_block_contains_heading_or_number_artifact"
            elif not standard_layout:
                route = "hold_14_line_block_not_full_length_sonnet_verse_layout"
            else:
                route = "retain_new_verified_standard_sonnet_inactive"
                new[candidate_id] = {
                    "source_unit_id": unit_id, "text": candidate_text,
                    "author": unit["author"], "first_line": cleaned_lines[0],
                    "character_start": start, "character_end": end,
                }
            rows.append({
                "routing_id": f"route:{candidate_id}", "source_unit_id": unit_id,
                "reference_unit_id": "", "candidate_id": candidate_id,
                "source_group": unit["source_group"], "source_title": unit["title"],
                "poem_author": unit["author"], "character_start": start, "character_end": end,
                "line_count": len(cleaned_lines), "source_text_sha256": _sha(text),
                "cleaned_character_count": len(candidate_text),
                "cleaned_sha256": _sha(candidate_text),
                "form_evidence": (
                    "exact 14-line blank-separated source block with 14 full-length verse lines"
                    if standard_layout else "raw 14-line blank-separated block; full-length sonnet layout failed"
                ),
                "language_evidence": "checkpoint-6D standard-Italian unit gate",
                "overlap_reference_ids": ";".join(overlaps), "routing_decision": route,
                "final_role": "standard_sonnets" if route.startswith("retain_") else "conditioned_or_inactive_hold",
                "activation_status": "inactive_pending_v7" if route.startswith("retain_") else "inactive_hold",
            })
        if progress and (unit_index == 1 or unit_index % 50 == 0 or unit_index == len(ilc_units)):
            _report(progress, f"sonnet-block-audit={unit_index}/{len(ilc_units)} candidates={len(rows)} new={len(new)}")
    # Resolve duplicates among newly discovered candidates with stable source-ID
    # precedence.  Duplicate poems are still removed from their broader source,
    # but only the first canonical candidate is materialized as a sonnet.
    accepted_ids = sorted(new)
    for left_index, left_id in enumerate(accepted_ids):
        if left_id not in new:
            continue
        for right_id in accepted_ids[left_index + 1:]:
            if right_id not in new:
                continue
            metrics = measure_word_shingle_containment(new[left_id]["text"], new[right_id]["text"])
            if float(metrics["containment"]) < config.containment_threshold:
                continue
            del new[right_id]
            for row in rows:
                if row["candidate_id"] == right_id:
                    row["overlap_reference_ids"] = left_id
                    row["routing_decision"] = "exclude_new_intra_ilc_ota_duplicate"
                    row["final_role"] = "standard_sonnets"
                    row["activation_status"] = "inactive_duplicate_not_materialized"
                    break
    return sorted(rows, key=lambda row: row["routing_id"]), new


def _blank_separated_blocks(text: str) -> Iterable[tuple[int, int, list[str]]]:
    position = 0
    block_start: int | None = None
    lines: list[str] = []
    block_end = 0
    for physical in text.splitlines(keepends=True):
        value = physical.rstrip("\r\n")
        if value.strip():
            if block_start is None:
                block_start = position
            lines.append(value)
            block_end = position + len(physical)
        elif block_start is not None:
            yield block_start, block_end, lines
            block_start, lines = None, []
        position += len(physical)
    if block_start is not None:
        yield block_start, block_end, lines


def _clean_verse_line(line: str) -> str:
    value = line.strip()
    if value.startswith("$"):
        value = value[1:]
    value = _PRINTED_LINE_NUMBER.sub("", value)
    value = _TRAILING_SOURCE_NUMBER.sub("", value)
    return value.strip()


def _is_standard_sonnet_layout(lines: Sequence[str]) -> bool:
    """Require 14 source-bounded, full-length verse lines rather than short forms/prose."""

    first_alpha = next((char for char in lines[0] if char.isalpha()), "") if lines else ""
    if len(lines) != 14 or not first_alpha.isupper():
        return False
    alphabetic_lengths = [sum(char.isalpha() for char in line) for line in lines]
    return min(alphabetic_lengths) >= 22 and 25 <= statistics.mean(alphabetic_lengths) <= 38


class _MetadataIndex:
    def __init__(self, config: CanonicalCorpusBuildConfig) -> None:
        self.config = config
        self.v6 = _unique(_read_csv(config.v6_manifest_path), "poem_id")
        self.bibit_period = _unique(_read_csv(config.bibit_period_path), "object_id")
        self.gutenberg_period = _unique(_read_csv(config.gutenberg_period_path), "ebook_id")
        self.gutenberg_attr = _unique(_read_csv(config.gutenberg_attribution_path), "ebook_id")
        self.wikisource_period = _unique(_read_csv(config.wikisource_period_path), "work_root_id")
        self.wikisource_sonnets = _unique(_read_csv(config.wikisource_sonnet_path), "candidate_id")
        self.wikisource_attr = _unique(_read_csv(config.wikisource_attribution_path), "work_root_id")
        self.liber_period = _unique(_read_csv(config.liber_liber_period_path), "record_id")
        self.liber_sonnets = _unique(_read_csv(config.liber_liber_sonnet_path), "candidate_id")
        self.liber_attr = _unique(_read_csv(config.liber_liber_attribution_path), "record_id")
        self.ilc_units = _unique(_read_csv(config.ilc_ota_unit_path), "unit_id")
        self.ilc_inventory = _unique(_read_csv(config.ilc_ota_inventory_path), "record_id")
        self.existing: dict[str, dict[str, Any]] = {}
        for path in config.existing_historical_reports:
            for row in json.loads(path.read_text(encoding="utf-8"))["sources"]:
                self.existing[row["source_id"]] = row

    def epoch(self, unit: dict[str, str]) -> str:
        group, source_id = unit["source_group"], unit["source_id"]
        if group == "v6_sonnets":
            return self.v6[source_id]["period"]
        if group == "bibit":
            object_id = source_id.split(":", 1)[0]
            return self.bibit_period.get(object_id, {}).get("periods", "unresolved") or "unresolved"
        if group == "gutenberg":
            ebook = source_id.removeprefix("pg").split(":", 1)[0]
            return self.gutenberg_period.get(ebook, {}).get("period_bucket", "unresolved") or "unresolved"
        if group == "wikisource":
            root = source_id if unit["unit_kind"] == "broader" else self.wikisource_sonnets[source_id]["work_root_id"]
            return self.wikisource_period.get(root, {}).get("period_bucket", "unresolved") or "unresolved"
        if group == "liber_liber":
            record = source_id if unit["unit_kind"] == "broader" else self.liber_sonnets[source_id]["record_id"]
            return self.liber_period.get(record, {}).get("period_bucket", "unresolved") or "unresolved"
        if group == "ilc_ota":
            record = self.ilc_units[source_id]["record_id"]
            return self.ilc_inventory[record].get("period_bucket", "unresolved") or "unresolved"
        if group == "existing_historical":
            return "origins_through_1800"
        return "unresolved"

    def source_url(self, unit: dict[str, str]) -> str:
        if unit["source_url"]:
            return unit["source_url"]
        if unit["source_group"] == "ilc_ota":
            return self.ilc_inventory[self.ilc_units[unit["source_id"]]["record_id"]]["landing_page_url"]
        return ""

    def attribution(self, unit: dict[str, str]) -> dict[str, Any]:
        group, source_id = unit["source_group"], unit["source_id"]
        origin_id, rights, license_url, notice, upstream = source_id, "", "", "", ""
        if group == "bibit":
            origin_id = source_id.split(":", 1)[0]
            rights = "personal/scientific non-commercial use; public reuse requires citation"
            license_url = "http://www.bibliotecaitaliana.it/"
            notice = "Credit Biblioteca Italiana and cite the work-level TEI source; commercial reuse is prohibited."
            upstream = "data/metadata/bibit_historical_attribution.md"
        elif group == "gutenberg":
            ebook = source_id.removeprefix("pg").split(":", 1)[0]
            attr = self.gutenberg_attr[ebook]
            origin_id, rights, license_url = f"pg{ebook}", attr["rights_basis"], attr["terms_url"]
            notice, upstream = attr["required_notice"], "data/processed/project_gutenberg_resolved_v1/attribution_manifest.csv"
        elif group == "wikisource":
            origin_id = source_id if unit["unit_kind"] == "broader" else self.wikisource_sonnets[source_id]["work_root_id"]
            attr = self.wikisource_attr[origin_id]
            rights, license_url, notice = attr["site_license"], attr["site_license_url"], attr["required_notice"]
            upstream = "data/processed/italian_wikisource_resolved_v1/attribution_manifest.csv"
        elif group == "liber_liber":
            origin_id = source_id if unit["unit_kind"] == "broader" else self.liber_sonnets[source_id]["record_id"]
            attr = self.liber_attr[origin_id]
            rights, license_url, notice = attr["license_label"], attr["license_url"], attr["required_notice"]
            upstream = "data/processed/liber_liber_resolved_v1/attribution_manifest.csv"
        elif group == "ilc_ota":
            origin_id = self.ilc_units[source_id]["record_id"]
            attr = self.ilc_inventory[origin_id]
            rights, license_url = attr["rights_text"], attr["license_url"]
            notice = f"Credit {attr['archive_id']} and the item creator; retain the stable item and terms links."
            upstream = "data/metadata/ilc_ota_source_inventory_v1.csv"
        elif group == "v6_sonnets":
            attr = self.v6[source_id]
            rights, notice = attr["license_notes"], "Retain the poem source URL, edition, and recorded source terms."
            upstream = "data/metadata/sonnets_expanded_v6_manifest.csv"
        elif group == "existing_historical":
            attr = self.existing[source_id]
            rights = attr.get("license_notes", "recorded upstream terms")
            notice = "Retain the existing processed-corpus attribution and source URL."
            upstream = "existing historical processed build report"
        attribution_id = f"attr:{group}:{origin_id}"
        return {
            "attribution_id": attribution_id, "source_group": group, "origin_id": origin_id,
            "source_archive": unit["source_archive"], "title": unit["title"], "author": unit["author"],
            "source_url": self.source_url(unit), "rights_label": rights, "license_url": license_url,
            "required_notice": notice,
            "modification_notice": "Canonical duplicate and verified embedded-sonnet spans may be removed; retained spelling and punctuation are preserved.",
            "upstream_attribution_artifact": upstream, "activation_status": "inactive_pending_v7",
        }


def _record_manifest_row(
    config: CanonicalCorpusBuildConfig, unit: dict[str, str], decision: dict[str, str],
    rewritten: str | None, writers: dict[str, "_DeltaShardWriter"], metadata: _MetadataIndex,
    attribution_id: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    included = not decision["final_decision"].startswith("exclude_")
    storage: dict[str, Any] | None = None
    if included:
        text = rewritten if rewritten is not None else _read_verified_unit(config.repo_root, unit)
        if rewritten is not None or unit["source_group"] == "ilc_ota":
            text = _canonical_delta_text(text)
            location = writers[unit["assigned_role"]].add(unit["unit_id"], text)
        else:
            location = _existing_location(config.repo_root, unit)
        stats = _stats(text)
        storage = _storage_row(unit["unit_id"], "broader", unit["assigned_role"], location, stats)
    else:
        stats = _empty_stats()
        location = None
    row = {
        "unit_id": unit["unit_id"], "source_group": unit["source_group"],
        "source_id": unit["source_id"], "title": unit["title"], "author": unit["author"],
        "source_archive": unit["source_archive"], "source_url": metadata.source_url(unit),
        "epoch_bucket": metadata.epoch(unit), "final_role": unit["assigned_role"],
        "canonical_priority": unit["canonical_priority"], "final_decision": decision["final_decision"],
        "quarantine_status": "rewritten_verified" if rewritten is not None else ("not_required" if included else "excluded"),
        "attribution_id": attribution_id, **stats, **_location_fields(location),
        "activation_status": "inactive_pending_v7" if included else "inactive_excluded",
        "training_eligible": str(included).lower(),
    }
    return row, storage


def _sonnet_manifest_row(
    config: CanonicalCorpusBuildConfig, unit: dict[str, str], decision: dict[str, str],
    writers: dict[str, "_DeltaShardWriter"], metadata: _MetadataIndex, attribution_id: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    included = not decision["final_decision"].startswith("exclude_")
    protected = decision["final_decision"] == "retain_protected_v6_split_locked"
    storage = None
    if included:
        text = _read_verified_unit(config.repo_root, unit)
        location = _existing_location(config.repo_root, unit)
        stats = _stats(text)
        storage = _storage_row(unit["unit_id"], "standard_sonnet", "standard_sonnets", location, stats)
        line_count = len([line for line in text.splitlines() if line.strip()])
    else:
        stats, location, line_count = _empty_stats(), None, ""
    row = {
        "unit_id": unit["unit_id"], "source_group": unit["source_group"],
        "source_id": unit["source_id"], "title": unit["title"], "author": unit["author"],
        "source_archive": unit["source_archive"], "source_url": metadata.source_url(unit),
        "epoch_bucket": metadata.epoch(unit), "original_split": unit["original_split"],
        "final_decision": decision["final_decision"], "routing_status": "canonical_identity",
        "attribution_id": attribution_id, "line_count": line_count, **stats,
        **_location_fields(location),
        "activation_status": "protected_v6_validation_test" if protected else ("inactive_pending_v7" if included else "inactive_excluded"),
        "training_eligible": str(included and not protected).lower(),
    }
    return row, storage


def _existing_location(repo_root: Path, unit: dict[str, str]) -> LogicalLocation:
    path = repo_root / unit["text_path"]
    payload, physical_sha256 = _read_file(path)
    start = int(unit["byte_start"] or 0)
    end = int(unit["byte_end"] or len(payload))
    return LogicalLocation(
        storage_kind="existing_committed_slice", path=unit["text_path"],
        byte_start=start, byte_end=end, physical_file_sha256=physical_sha256,
    )


def _storage_row(
    unit_id: str, unit_kind: str, role: str, location: LogicalLocation,
    stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "storage_id": f"storage:{unit_id}", "unit_id": unit_id, "unit_kind": unit_kind,
        "final_role": role, "storage_kind": location.storage_kind, "storage_path": location.path,
        "byte_start": location.byte_start, "byte_end": location.byte_end,
        **stats, "physical_file_sha256": location.physical_file_sha256,
        "public_repository_status": "committed_or_checkpoint_delta",
    }


def _location_fields(location: LogicalLocation | None) -> dict[str, Any]:
    if location is None:
        return {"storage_kind": "none", "storage_path": "", "byte_start": "", "byte_end": ""}
    return {
        "storage_kind": location.storage_kind, "storage_path": location.path,
        "byte_start": location.byte_start, "byte_end": location.byte_end,
    }


class _DeltaShardWriter:
    def __init__(self, directory: Path, portable: str, maximum: int) -> None:
        self.directory, self.portable, self.maximum = directory, portable, maximum
        self.handle: BinaryIO | None = None
        self.path: Path | None = None
        self.bytes = self.items = 0
        self.hasher = hashlib.sha256()
        self.reports: list[dict[str, Any]] = []

    def add(self, item_id: str, text: str) -> LogicalLocation:
        payload = text.encode("utf-8")
        if not payload.strip() or len(payload) > self.maximum:
            raise ValueError(f"invalid delta item {item_id}: {len(payload):,} bytes")
        separator = b"\n" if self.items else b""
        if self.handle is None or self.bytes + len(separator) + len(payload) > self.maximum:
            self._finish(); self._start(); separator = b""
        self._write(separator)
        start = self.bytes
        self._write(payload)
        self.items += 1
        return LogicalLocation(
            storage_kind="checkpoint_7b_delta_slice",
            path=f"{self.portable}/{self.path.name}", byte_start=start, byte_end=self.bytes,
            physical_file_sha256="pending_manifest_close",
        )

    def close(self) -> list[dict[str, Any]]:
        self._finish()
        return self.reports

    def abort(self) -> None:
        if self.handle is not None:
            self.handle.close(); self.handle = None

    def _start(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"delta-{len(self.reports)+1:04d}.txt"
        self.handle = self.path.open("wb")
        self.bytes = self.items = 0
        self.hasher = hashlib.sha256()

    def _write(self, payload: bytes) -> None:
        assert self.handle is not None
        self.handle.write(payload); self.hasher.update(payload); self.bytes += len(payload)

    def _finish(self) -> None:
        if self.handle is None or self.path is None:
            return
        self.handle.close()
        self.reports.append({
            "path": f"{self.portable}/{self.path.name}", "item_count": self.items,
            "byte_count": self.bytes, "sha256": self.hasher.hexdigest(),
        })
        self.handle = None; self.path = None


def _build_report(
    config: CanonicalCorpusBuildConfig, records: list[dict[str, Any]],
    sonnets: list[dict[str, Any]], storage: list[dict[str, Any]],
    ranges: list[dict[str, Any]], routes: list[dict[str, Any]],
    attribution: dict[str, dict[str, Any]], shards: list[dict[str, Any]],
    modified_broader_count: int, elapsed: float,
) -> dict[str, Any]:
    retained_records = [row for row in records if row["training_eligible"] == "true"]
    retained_sonnets = [row for row in sonnets if row["training_eligible"] == "true"]
    protected = [row for row in sonnets if row["activation_status"] == "protected_v6_validation_test"]
    role_chars = Counter()
    source_chars = Counter()
    author_chars = Counter()
    epoch_chars = Counter()
    for row in retained_records:
        count = int(row["logical_character_count"])
        role_chars[row["final_role"]] += count
        source_chars[row["source_group"]] += count
        author_chars[row["author"] or "unresolved"] += count
        epoch_chars[row["epoch_bucket"]] += count
    for row in retained_sonnets:
        count = int(row["logical_character_count"])
        role_chars["standard_sonnets"] += count
        source_chars[row["source_group"]] += count
        author_chars[row["author"] or "unresolved"] += count
        epoch_chars[row["epoch_bucket"]] += count
    logical = sum(role_chars.values())
    delta = [row for row in storage if row["storage_kind"] == "checkpoint_7b_delta_slice"]
    return {
        "build_version": BUILD_VERSION, "build_date": config.build_date,
        "activation_status": "inactive_pending_v7", "record_universe_count": len(records),
        "token_count_status": "not_measured_pending_checkpoint_8_minerva_tokenization",
        "sonnet_universe_count": len(sonnets), "training_record_count": len(retained_records),
        "training_sonnet_count": len(retained_sonnets), "protected_v6_sonnet_count": len(protected),
        "excluded_record_count": sum(row["training_eligible"] == "false" for row in records),
        "excluded_sonnet_count": sum(row["storage_kind"] == "none" for row in sonnets),
        "segment_review_count": len({row["review_id"] for row in ranges}),
        "segment_range_count": sum(row["range_decision"] == "remove_verified_occurrence" for row in ranges),
        "routing_row_count": len(routes),
        "new_standard_sonnet_count": sum(row["routing_decision"] == "retain_new_verified_standard_sonnet_inactive" for row in routes),
        "logical_character_count": logical,
        "logical_role_characters": dict(sorted(role_chars.items())),
        "source_group_characters": dict(sorted(source_chars.items())),
        "author_character_shares": _shares(author_chars),
        "epoch_character_shares": _shares(epoch_chars),
        "existing_slice_count": sum(row["storage_kind"] == "existing_committed_slice" for row in storage),
        "delta_slice_count": len(delta), "delta_logical_character_count": sum(int(row["logical_character_count"]) for row in delta),
        "delta_shard_count": len(shards), "delta_shard_byte_count": sum(int(row["byte_count"]) for row in shards),
        "attribution_count": len(attribution),
        "modified_broader_protected_recheck_count": modified_broader_count,
        "protected_pair_recheck_count": modified_broader_count * len(protected),
        "verification": {
            "all_264_reviews_accounted": len({row["review_id"] for row in ranges}) == 264,
            "protected_v6_count_preserved": len(protected) == 387,
            "conditioned_material_included": False, "v7_created": False,
            "mixture_weights_assigned": False, "gpu_work_started": False,
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    roles = "\n".join(f"- `{role}`: {count:,} characters." for role, count in report["logical_role_characters"].items())
    return (
        "# Canonical Italian Corpora v1: Inactive Logical Build\n\n"
        "## Result\n\n"
        f"Checkpoint 7B freezes {report['training_record_count']:,} broader training records and "
        f"{report['training_sonnet_count']:,} training-eligible standard sonnets as an inactive, manifest-backed logical corpus. "
        f"The logical training total is {report['logical_character_count']:,} characters.\n\n"
        "An exact Minerva token count is deliberately not estimated here; checkpoint 8 will tokenize the frozen "
        "V7 training mixtures with the pinned Minerva tokenizer.\n\n"
        f"{roles}\n\n"
        "## Storage\n\n"
        f"The build references {report['existing_slice_count']:,} unchanged committed slices and writes only "
        f"{report['delta_slice_count']:,} new or rewritten slices ({report['delta_logical_character_count']:,} logical characters) "
        f"to {report['delta_shard_count']:,} bounded delta shards. This avoids copying the unchanged corpus.\n\n"
        "## Isolation And Boundary\n\n"
        f"All {report['segment_review_count']:,} checkpoint-7A review decisions are accounted, and all "
        f"{report['protected_v6_sonnet_count']:,} protected V6 validation/test identities remain excluded from training. "
        "Conditioned language/form variants are absent. The build is inactive: it creates no V7 split, assigns no mixture "
        "weight, starts no GPU work, and deletes no reusable cache.\n"
    )


def _validate_temp_build(
    config: CanonicalCorpusBuildConfig, temp_dir: Path, records: list[dict[str, Any]],
    sonnets: list[dict[str, Any]], storage: list[dict[str, Any]],
) -> None:
    final_prefix = config.output_dir.relative_to(config.repo_root).as_posix()
    by_unit = _unique(storage, "unit_id")
    for row in records + sonnets:
        if row["storage_kind"] == "none":
            continue
        stored = by_unit[row["unit_id"]]
        portable = stored["storage_path"]
        path = temp_dir / Path(portable).relative_to(final_prefix) if portable.startswith(final_prefix + "/") else config.repo_root / portable
        payload = path.read_bytes()[int(stored["byte_start"]):int(stored["byte_end"])]
        text = payload.decode("utf-8")
        if len(text) != int(stored["logical_character_count"]) or _sha(text) != stored["logical_sha256"]:
            raise ValueError(f"logical storage mismatch: {row['unit_id']}")


def _validate_config(config: CanonicalCorpusBuildConfig) -> None:
    for value in config.__dict__.values():
        if isinstance(value, Path) and value in {config.segment_range_path, config.sonnet_routing_path, config.output_dir, config.json_report_path, config.markdown_report_path}:
            continue
        if isinstance(value, Path) and not value.exists():
            raise FileNotFoundError(value)
    if not 0 < config.containment_threshold <= 1:
        raise ValueError("containment threshold must be in (0, 1]")
    if config.max_shard_bytes <= 0 or config.progress_interval <= 0:
        raise ValueError("shard and progress limits must be positive")


def _read_verified_unit(repo_root: Path, row: dict[str, str]) -> str:
    path = repo_root / row["text_path"]
    try:
        path.resolve().relative_to(repo_root.resolve())
    except ValueError as error:
        raise ValueError(f"unit path escapes repository: {row['unit_id']}") from error
    payload, _physical_sha256 = _read_file(path)
    start = int(row["byte_start"] or 0)
    end = int(row["byte_end"] or len(payload))
    if start < 0 or end <= start or end > len(payload):
        raise ValueError(f"invalid unit slice: {row['unit_id']}")
    text = payload[start:end].decode("utf-8")
    if len(text) != int(row["cleaned_character_count"]) or _sha(text) != row["cleaned_sha256"]:
        raise ValueError(f"unit text mismatch: {row['unit_id']}")
    return text


@lru_cache(maxsize=4)
def _read_file(path: Path) -> tuple[bytes, str]:
    payload = path.read_bytes()
    return payload, hashlib.sha256(payload).hexdigest()


def _stats(text: str) -> dict[str, Any]:
    payload = text.encode("utf-8")
    return {
        "logical_character_count": len(text), "logical_byte_count": len(payload),
        "logical_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _canonical_delta_text(text: str) -> str:
    """Use one terminal newline without altering retained spelling or punctuation."""

    return text.rstrip("\r\n") + "\n"


def _empty_stats() -> dict[str, Any]:
    return {"logical_character_count": 0, "logical_byte_count": 0, "logical_sha256": ""}


def _shares(counts: Counter[str]) -> list[dict[str, Any]]:
    total = sum(counts.values())
    return [
        {"label": key, "character_count": value, "share": round(value / total, 8) if total else 0.0}
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _unique(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise ValueError(f"duplicate {key}: {value}")
        result[value] = row
    return result


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: tuple(str(row.get(field, "")) for field in fields[:3])))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _replace_verified_output(temp_dir: Path, output_dir: Path) -> None:
    backup: Path | None = None
    try:
        if output_dir.exists():
            backup = output_dir.with_name(f".{output_dir.name}.backup")
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(output_dir, backup)
        os.replace(temp_dir, output_dir)
        if backup is not None:
            shutil.rmtree(backup)
    except BaseException:
        if output_dir.exists() and backup is not None:
            shutil.rmtree(output_dir, ignore_errors=True)
        if backup is not None and backup.exists():
            os.replace(backup, output_dir)
        raise


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _report(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
