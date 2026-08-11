"""Build deterministic role-specific shards from frozen Gutenberg audit ledgers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from time import monotonic
from typing import Any, BinaryIO

from .gutenberg import strip_gutenberg_boilerplate
from .gutenberg_fulltext_probe import (
    TextFingerprint,
    TextReference,
    fingerprint_text,
    measure_word_shingle_containment,
)


RECORD_MANIFEST_FIELDS = (
    "ebook_id",
    "title",
    "authors",
    "source_pool",
    "source_archive",
    "source_url",
    "period_bucket",
    "input_role",
    "final_role",
    "language_route",
    "source_decision",
    "extraction_policy",
    "canonical_reference_ids",
    "activation_status",
    "artifact_status",
    "cache_path",
    "cache_sha256",
    "source_cleaned_sha256",
    "source_cleaned_character_count",
    "retained_source_character_count",
    "excluded_source_character_count",
    "shard_path",
    "byte_start",
    "byte_end",
    "cleaned_character_count",
    "cleaned_byte_count",
    "cleaned_sha256",
)

SEGMENT_MANIFEST_FIELDS = (
    "segment_id",
    "ebook_id",
    "source_cleaned_sha256",
    "character_start",
    "character_end",
    "character_count",
    "segment_sha256",
    "segment_decision",
    "final_role",
    "reason",
    "reference_ids",
    "start_anchor",
    "end_anchor",
    "artifact_status",
    "artifact_kind",
    "output_shard_path",
    "output_byte_start",
    "output_byte_end",
    "output_sha256",
)

SONNET_MANIFEST_FIELDS = (
    "candidate_id",
    "ebook_id",
    "title",
    "source_record_authors",
    "poem_author",
    "poem_author_resolution",
    "period_bucket",
    "source_archive",
    "source_url",
    "source_kind",
    "stanza_pattern",
    "line_count",
    "first_line",
    "last_line",
    "character_start",
    "character_end",
    "source_text_sha256",
    "cleaned_source_text_sha256",
    "candidate_decision",
    "manual_review_resolution",
    "manual_review_rationale",
    "exact_reference_ids",
    "near_reference_ids",
    "heldout_reference_ids",
    "duplicate_gutenberg_candidate_ids",
    "final_role",
    "activation_status",
    "artifact_status",
    "shard_path",
    "byte_start",
    "byte_end",
    "cleaned_character_count",
    "cleaned_byte_count",
    "cleaned_line_count",
    "cleaned_sha256",
)

ATTRIBUTION_MANIFEST_FIELDS = (
    "ebook_id",
    "title",
    "authors",
    "source_archive",
    "source_url",
    "plain_text_url",
    "catalog_snapshot_version",
    "catalog_copyright",
    "media_type",
    "rights_basis",
    "reuse_scope",
    "terms_url",
    "required_notice",
    "acquired_date",
    "acquisition_evidence",
    "text_modifications",
    "downstream_note",
    "artifact_status",
)

_RECORD_ROLES = {
    "historical_general",
    "historical_non_sonnet_poetry",
    "nineteenth_century_bridge",
}
_ELIGIBLE_SOURCE_DECISIONS = {
    "eligible_standard_core_pending_processed_build",
    "eligible_after_source_specific_extraction_pending_build",
}
_CONDITIONED_SOURCE_DECISION = "conditioned_candidate_not_activated"
_EXCLUDED_SOURCE_DECISION = "exclude_canonical_cross_corpus_duplicate"
_STANDARD_SONNET_DECISION = "eligible_standard_sonnet_pending_processed_build"
_CONDITIONED_SONNET_DECISION = "conditioned_sonnet_candidate_not_activated"
_EXCLUDED_SONNET_DECISIONS = {
    "exclude_existing_corpus_sonnet_duplicate",
    "exclude_intra_gutenberg_sonnet_duplicate",
    "exclude_manual_not_sonnet",
}
_TERMS_URL = "https://www.gutenberg.org/policy/permission.html"


@dataclass(frozen=True)
class GutenbergResolvedBuildConfig:
    """Pinned audit inputs and bounded output settings for checkpoint 3B."""

    repo_root: Path
    source_decisions_path: Path
    segment_decisions_path: Path
    sonnet_decisions_path: Path
    sonnet_review_path: Path
    audit_report_path: Path
    inventory_path: Path
    bibit_record_manifest_path: Path
    broader_sources_manifest_path: Path
    output_dir: Path
    markdown_report_path: Path
    max_shard_bytes: int = 64 * 1024 * 1024
    near_duplicate_threshold: float = 0.8
    expected_source_count: int = 587
    expected_eligible_source_count: int = 566
    expected_conditioned_source_count: int = 6
    expected_standard_sonnet_count: int = 499
    expected_conditioned_sonnet_count: int = 2
    progress_interval: int = 25


Progress = Callable[[str], None]


def build_gutenberg_resolved_corpus(
    config: GutenbergResolvedBuildConfig,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Materialize the frozen 3A decisions without assigning V7 or mixtures."""

    _validate_config(config)
    started = monotonic()
    sources = _read_csv(config.source_decisions_path)
    segments = _read_csv(config.segment_decisions_path)
    candidates = _read_csv(config.sonnet_decisions_path)
    inventory = _read_csv(config.inventory_path)
    audit_report = json.loads(config.audit_report_path.read_text(encoding="utf-8"))
    _validate_audit_hashes(config, audit_report)
    _validate_input_counts(config, sources, candidates)

    source_by_id = _unique_by(sources, "ebook_id", "source decisions")
    inventory_by_id = _unique_by(inventory, "ebook_id", "Gutenberg inventory")
    segments_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in segments:
        segments_by_source[row["ebook_id"]].append(row)
    candidates_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        candidates_by_source[row["ebook_id"]].append(row)
    _validate_partitions(source_by_id, segments_by_source)

    output_parent = config.output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{config.output_dir.name}.", dir=output_parent)
    )
    final_prefix = _portable(config.output_dir, config.repo_root)
    writers = {
        role: _ShardWriter(
            temp_dir / directory,
            f"{final_prefix}/{directory}",
            max_shard_bytes=config.max_shard_bytes,
        )
        for role, directory in (
            ("historical_general", "historical_general"),
            ("historical_non_sonnet_poetry", "historical_non_sonnet_poetry"),
            ("nineteenth_century_bridge", "nineteenth_century_bridge"),
            ("standard_sonnets", "standard_sonnets"),
            ("conditioned_source_variants", "conditioned_source_variants"),
            ("conditioned_sonnet_variants", "conditioned_sonnet_variants"),
        )
    }
    record_manifest: list[dict[str, Any]] = []
    segment_manifest: list[dict[str, Any]] = []
    sonnet_manifest: list[dict[str, Any]] = []
    attribution_manifest: list[dict[str, Any]] = []
    standard_fingerprints: dict[str, TextFingerprint] = {}

    try:
        for index, source in enumerate(
            sorted(sources, key=lambda row: int(row["ebook_id"])), start=1
        ):
            ebook_id = source["ebook_id"]
            text = _read_and_verify_source(config, source)
            source_segments = sorted(
                segments_by_source[ebook_id],
                key=lambda row: int(row["character_start"]),
            )
            segment_output_rows = [_base_segment_manifest(row) for row in source_segments]
            segment_output_by_bounds = {
                (int(row["character_start"]), int(row["character_end"])): row
                for row in segment_output_rows
            }

            selected_decisions, writer_role, activation_status, artifact_status = (
                _record_materialization_route(source)
            )
            selected_segments = [
                row for row in source_segments if row["segment_decision"] in selected_decisions
            ]
            retained_source_characters = sum(
                int(row["character_count"]) for row in selected_segments
            )
            record_text, relative_ranges = _compose_segments(text, selected_segments)
            if record_text:
                canonical_record_text = _canonical_text(record_text)
                location = writers[writer_role].add(f"pg{ebook_id}", canonical_record_text)
                for segment, relative_start, relative_end in relative_ranges:
                    output_row = segment_output_by_bounds[
                        (int(segment["character_start"]), int(segment["character_end"]))
                    ]
                    output_row.update(
                        {
                            "artifact_status": "materialized_in_record_artifact",
                            "artifact_kind": writer_role,
                            "output_shard_path": location["shard_path"],
                            "output_byte_start": int(location["byte_start"]) + relative_start,
                            "output_byte_end": int(location["byte_start"]) + relative_end,
                            "output_sha256": segment["segment_sha256"],
                        }
                    )
                if source["source_decision"] in _ELIGIBLE_SOURCE_DECISIONS:
                    fingerprint, _ = fingerprint_text(canonical_record_text)
                    standard_fingerprints[ebook_id] = fingerprint
            else:
                canonical_record_text = ""
                location = {"shard_path": "", "byte_start": "", "byte_end": ""}

            for candidate in sorted(
                candidates_by_source.get(ebook_id, []),
                key=lambda row: int(row["character_start"]),
            ):
                sonnet_row, sonnet_location = _materialize_candidate(
                    candidate,
                    source=source,
                    text=text,
                    writers=writers,
                )
                sonnet_manifest.append(sonnet_row)
                bounds = (
                    int(candidate["character_start"]),
                    int(candidate["character_end"]),
                )
                segment_row = segment_output_by_bounds.get(bounds)
                if candidate["candidate_decision"] == "exclude_manual_not_sonnet":
                    if segment_row is not None and segment_row["segment_decision"].startswith(
                        "quarantine_"
                    ):
                        raise ValueError(
                            f"manual non-sonnet remained quarantined: {candidate['candidate_id']}"
                        )
                    continue
                if segment_row is None:
                    raise ValueError(
                        f"sonnet quarantine segment missing: {candidate['candidate_id']}"
                    )
                _attach_sonnet_segment_artifact(
                    segment_row,
                    candidate=candidate,
                    sonnet_row=sonnet_row,
                    sonnet_location=sonnet_location,
                )

            segment_manifest.extend(segment_output_rows)
            stats = _text_stats(canonical_record_text)
            record_manifest.append(
                {
                    "ebook_id": ebook_id,
                    "title": source["title"],
                    "authors": source["authors"],
                    "source_pool": source["source_pool"],
                    "source_archive": source["source_archive"],
                    "source_url": source["source_url"],
                    "period_bucket": source["period_bucket"],
                    "input_role": source["input_role"],
                    "final_role": source["final_role"],
                    "language_route": source["language_route"],
                    "source_decision": source["source_decision"],
                    "extraction_policy": source["extraction_policy"],
                    "canonical_reference_ids": source["canonical_reference_ids"],
                    "activation_status": activation_status,
                    "artifact_status": artifact_status,
                    "cache_path": source["cache_path"],
                    "cache_sha256": source["cache_sha256"],
                    "source_cleaned_sha256": source["cleaned_sha256"],
                    "source_cleaned_character_count": source["cleaned_character_count"],
                    "retained_source_character_count": retained_source_characters,
                    "excluded_source_character_count": (
                        int(source["cleaned_character_count"])
                        - retained_source_characters
                    ),
                    **location,
                    **stats,
                }
            )
            attribution_manifest.append(
                _attribution_row(source, inventory_by_id.get(ebook_id))
            )
            if index == 1 or index % config.progress_interval == 0 or index == len(sources):
                _progress(
                    progress,
                    "source-build",
                    index,
                    len(sources),
                    started,
                    f"pg{ebook_id} sonnets={len(sonnet_manifest):,}",
                )

        if set(source_by_id) != set(segments_by_source):
            raise ValueError("source and segment ledgers do not cover the same ebook IDs")
        if len(sonnet_manifest) != len(candidates):
            raise ValueError("not every sonnet candidate reached the processed manifest")

        shard_reports = {role: writer.close() for role, writer in writers.items()}
        _validate_materialized_counts(config, record_manifest, sonnet_manifest)
        deduplication = _verify_final_deduplication(
            config,
            temp_dir=temp_dir,
            final_prefix=final_prefix,
            record_manifest=record_manifest,
            standard_fingerprints=standard_fingerprints,
            progress=progress,
        )
        _validate_segment_artifacts(segment_manifest)

        record_manifest_path = temp_dir / "records_manifest.csv"
        segment_manifest_path = temp_dir / "segments_manifest.csv"
        sonnet_manifest_path = temp_dir / "sonnets_manifest.csv"
        attribution_manifest_path = temp_dir / "attribution_manifest.csv"
        _write_csv(record_manifest_path, RECORD_MANIFEST_FIELDS, record_manifest)
        _write_csv(segment_manifest_path, SEGMENT_MANIFEST_FIELDS, segment_manifest)
        _write_csv(sonnet_manifest_path, SONNET_MANIFEST_FIELDS, sonnet_manifest)
        _write_csv(
            attribution_manifest_path,
            ATTRIBUTION_MANIFEST_FIELDS,
            attribution_manifest,
        )
        report = _build_report(
            config,
            record_manifest=record_manifest,
            segment_manifest=segment_manifest,
            sonnet_manifest=sonnet_manifest,
            shard_reports=shard_reports,
            deduplication=deduplication,
            manifest_paths={
                "record": record_manifest_path,
                "segment": segment_manifest_path,
                "sonnet": sonnet_manifest_path,
                "attribution": attribution_manifest_path,
            },
        )
        _write_json(temp_dir / "build_report.json", report)
        _validate_output_artifacts(
            temp_dir,
            final_prefix,
            record_manifest,
            segment_manifest,
            sonnet_manifest,
        )
        _validate_shard_sizes(temp_dir, config.max_shard_bytes)

        _replace_verified_output(temp_dir, config.output_dir)
    except Exception:
        for writer in writers.values():
            writer.abort()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(
        render_gutenberg_resolved_build_markdown(report),
        encoding="utf-8",
    )
    return report


class _ShardWriter:
    def __init__(
        self,
        directory: Path,
        portable_directory: str,
        *,
        max_shard_bytes: int,
    ) -> None:
        self.directory = directory
        self.portable_directory = portable_directory
        self.max_shard_bytes = max_shard_bytes
        self._handle: BinaryIO | None = None
        self._path: Path | None = None
        self._portable_path = ""
        self._bytes = 0
        self._items = 0
        self._hasher = hashlib.sha256()
        self._reports: list[dict[str, Any]] = []

    def add(self, item_id: str, text: str) -> dict[str, Any]:
        if not text.strip():
            raise ValueError(f"cannot shard empty text: {item_id}")
        canonical_text = _canonical_text(text)
        payload = canonical_text.encode("utf-8")
        if len(payload) > self.max_shard_bytes:
            raise ValueError(
                f"single item exceeds max shard bytes: {item_id} ({len(payload):,})"
            )
        separator_bytes = 1 if self._items else 0
        if (
            self._handle is None
            or self._bytes + separator_bytes + len(payload) > self.max_shard_bytes
        ):
            self._finish_shard()
            self._start_shard()
            separator_bytes = 0
        assert self._handle is not None
        if separator_bytes:
            self._write(b"\n")
        byte_start = self._bytes
        self._write(payload)
        self._items += 1
        return {
            "shard_path": self._portable_path,
            "byte_start": byte_start,
            "byte_end": self._bytes,
        }

    def close(self) -> list[dict[str, Any]]:
        self._finish_shard()
        return list(self._reports)

    def abort(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _start_shard(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        name = f"part-{len(self._reports) + 1:04d}.txt"
        self._path = self.directory / name
        self._portable_path = f"{self.portable_directory}/{name}"
        self._handle = self._path.open("wb")
        self._bytes = 0
        self._items = 0
        self._hasher = hashlib.sha256()

    def _write(self, payload: bytes) -> None:
        assert self._handle is not None
        self._handle.write(payload)
        self._hasher.update(payload)
        self._bytes += len(payload)

    def _finish_shard(self) -> None:
        if self._handle is None or self._path is None:
            return
        self._handle.close()
        self._reports.append(
            {
                "path": self._portable_path,
                "item_count": self._items,
                "byte_count": self._bytes,
                "sha256": self._hasher.hexdigest(),
            }
        )
        self._handle = None
        self._path = None


def _read_and_verify_source(
    config: GutenbergResolvedBuildConfig,
    source: dict[str, str],
) -> str:
    cache_path = config.repo_root / source["cache_path"]
    resolved_root = config.repo_root.resolve()
    try:
        cache_path.resolve().relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"cache path escapes repository: {cache_path}") from error
    raw = cache_path.read_text(encoding="utf-8")
    if _sha256_text(raw) != source["cache_sha256"]:
        raise ValueError(f"cached primary-text hash mismatch: pg{source['ebook_id']}")
    text = strip_gutenberg_boilerplate(raw)
    if _sha256_text(text) != source["cleaned_sha256"]:
        raise ValueError(f"cleaned primary-text hash mismatch: pg{source['ebook_id']}")
    if len(text) != int(source["cleaned_character_count"]):
        raise ValueError(f"cleaned primary-text length mismatch: pg{source['ebook_id']}")
    return text


def _record_materialization_route(
    source: dict[str, str],
) -> tuple[set[str], str, str, str]:
    decision = source["source_decision"]
    if decision in _ELIGIBLE_SOURCE_DECISIONS:
        role = source["final_role"]
        if role not in _RECORD_ROLES:
            raise ValueError(f"unsupported standard record role: {role}")
        return (
            {"include_record_text"},
            role,
            "eligible_pending_v7_and_mixture_freeze",
            "text_materialized_pending_v7",
        )
    if decision == _CONDITIONED_SOURCE_DECISION:
        return (
            {"conditioned_not_activated"},
            "conditioned_source_variants",
            "inactive_conditioned_experiment_not_authorized",
            "conditioned_text_materialized_inactive",
        )
    if decision == _EXCLUDED_SOURCE_DECISION:
        return (
            set(),
            "",
            "excluded_canonical_duplicate",
            "not_materialized_canonical_duplicate",
        )
    raise ValueError(f"unsupported source decision: {decision}")


def _compose_segments(
    text: str,
    selected_segments: list[dict[str, str]],
) -> tuple[str, list[tuple[dict[str, str], int, int]]]:
    payload = bytearray()
    ranges: list[tuple[dict[str, str], int, int]] = []
    character_parts: list[str] = []
    for index, segment in enumerate(selected_segments):
        start = int(segment["character_start"])
        end = int(segment["character_end"])
        part = text[start:end]
        if _sha256_text(part) != segment["segment_sha256"]:
            raise ValueError(f"source segment hash mismatch: {segment['segment_id']}")
        if index:
            payload.extend(b"\n")
            character_parts.append("\n")
        relative_start = len(payload)
        encoded = part.encode("utf-8")
        payload.extend(encoded)
        relative_end = len(payload)
        character_parts.append(part)
        ranges.append((segment, relative_start, relative_end))
    result = "".join(character_parts)
    if result.encode("utf-8") != bytes(payload):
        raise AssertionError("segment byte composition changed")
    return result, ranges


def _materialize_candidate(
    candidate: dict[str, str],
    *,
    source: dict[str, str],
    text: str,
    writers: dict[str, _ShardWriter],
) -> tuple[dict[str, Any], dict[str, Any]]:
    start = int(candidate["character_start"])
    end = int(candidate["character_end"])
    raw = text[start:end]
    if _sha256_text(raw) != candidate["source_text_sha256"]:
        raise ValueError(f"sonnet source hash mismatch: {candidate['candidate_id']}")
    cleaned = _clean_candidate_text(raw)
    if _sha256_text(cleaned) != candidate["cleaned_text_sha256"]:
        raise ValueError(f"sonnet cleaned hash mismatch: {candidate['candidate_id']}")

    decision = candidate["candidate_decision"]
    if decision == _STANDARD_SONNET_DECISION:
        final_role = "standard_sonnets"
        activation_status = "eligible_pending_v7_and_poem_attribution"
        artifact_status = "standard_sonnet_materialized_pending_v7"
        location = writers["standard_sonnets"].add(candidate["candidate_id"], cleaned)
    elif decision == _CONDITIONED_SONNET_DECISION:
        final_role = "conditioned_sonnet_variants"
        activation_status = "inactive_conditioned_experiment_not_authorized"
        artifact_status = "conditioned_sonnet_materialized_inactive"
        location = writers["conditioned_sonnet_variants"].add(
            candidate["candidate_id"], cleaned
        )
    elif decision in _EXCLUDED_SONNET_DECISIONS:
        final_role = "not_standard_sonnet_artifact"
        activation_status = "excluded_from_sonnet_corpus"
        artifact_status = f"not_materialized_{decision}"
        location = {"shard_path": "", "byte_start": "", "byte_end": ""}
    else:
        raise ValueError(f"unsupported sonnet decision: {decision}")

    stats = (
        _text_stats(cleaned, include_lines=True)
        if location["shard_path"]
        else _text_stats("", include_lines=True)
    )
    if location["shard_path"] and stats["cleaned_line_count"] != 14:
        raise ValueError(f"materialized sonnet is not fourteen lines: {candidate['candidate_id']}")
    row = {
        "candidate_id": candidate["candidate_id"],
        "ebook_id": candidate["ebook_id"],
        "title": candidate["title"],
        "source_record_authors": candidate["authors"],
        "poem_author": "",
        "poem_author_resolution": "pending_candidate_level_attribution_audit",
        "period_bucket": source["period_bucket"],
        "source_archive": source["source_archive"],
        "source_url": source["source_url"],
        "source_kind": candidate["source_kind"],
        "stanza_pattern": candidate["stanza_pattern"],
        "line_count": candidate["line_count"],
        "first_line": candidate["first_line"],
        "last_line": candidate["last_line"],
        "character_start": candidate["character_start"],
        "character_end": candidate["character_end"],
        "source_text_sha256": candidate["source_text_sha256"],
        "cleaned_source_text_sha256": candidate["cleaned_text_sha256"],
        "candidate_decision": decision,
        "manual_review_resolution": candidate["manual_review_resolution"],
        "manual_review_rationale": candidate["manual_review_rationale"],
        "exact_reference_ids": candidate["exact_reference_ids"],
        "near_reference_ids": candidate["near_reference_ids"],
        "heldout_reference_ids": candidate["heldout_reference_ids"],
        "duplicate_gutenberg_candidate_ids": candidate[
            "duplicate_gutenberg_candidate_ids"
        ],
        "final_role": final_role,
        "activation_status": activation_status,
        "artifact_status": artifact_status,
        **location,
        **stats,
    }
    return row, location


def _attach_sonnet_segment_artifact(
    segment: dict[str, Any],
    *,
    candidate: dict[str, str],
    sonnet_row: dict[str, Any],
    sonnet_location: dict[str, Any],
) -> None:
    if sonnet_location["shard_path"]:
        segment.update(
            {
                "artifact_status": "materialized_in_sonnet_artifact",
                "artifact_kind": sonnet_row["final_role"],
                "output_shard_path": sonnet_location["shard_path"],
                "output_byte_start": sonnet_location["byte_start"],
                "output_byte_end": sonnet_location["byte_end"],
                "output_sha256": sonnet_row["cleaned_sha256"],
            }
        )
    else:
        segment.update(
            {
                "artifact_status": sonnet_row["artifact_status"],
                "artifact_kind": "excluded_sonnet_candidate",
            }
        )


def _base_segment_manifest(row: dict[str, str]) -> dict[str, Any]:
    return {
        **row,
        "artifact_status": "not_materialized_by_segment_decision",
        "artifact_kind": "excluded_or_quarantined_source_segment",
        "output_shard_path": "",
        "output_byte_start": "",
        "output_byte_end": "",
        "output_sha256": "",
    }


def _attribution_row(
    source: dict[str, str],
    inventory: dict[str, str] | None,
) -> dict[str, Any]:
    if inventory is None:
        raise ValueError(f"source absent from Gutenberg inventory: pg{source['ebook_id']}")
    if inventory["copyright"] != "False" or inventory["media_type"] != "Text":
        raise ValueError(f"incompatible Gutenberg rights/media metadata: pg{source['ebook_id']}")
    acquired_date = (
        "2026-08-09"
        if source["source_pool"] == "initial_eligible_pool"
        else "2026-08-10"
    )
    acquisition_evidence = {
        "initial_eligible_pool": "reports/project_gutenberg_fulltext_probe_v1.json",
        "pass_1b_eligible_pool": "reports/project_gutenberg_fulltext_probe_pass_1b_v1.json",
        "conditioned_metadata_pool": "reports/project_gutenberg_metadata_resolution_v1.json",
    }[source["source_pool"]]
    return {
        "ebook_id": source["ebook_id"],
        "title": source["title"],
        "authors": source["authors"],
        "source_archive": "Project Gutenberg",
        "source_url": source["source_url"],
        "plain_text_url": inventory["plain_text_url"],
        "catalog_snapshot_version": "project_gutenberg_italian_inventory_v1",
        "catalog_copyright": inventory["copyright"],
        "media_type": inventory["media_type"],
        "rights_basis": (
            "Gutendex copyright=False and record-level Project Gutenberg "
            "public-domain statement"
        ),
        "reuse_scope": (
            "underlying ebook marked public domain in the USA; jurisdiction-specific "
            "status must be checked"
        ),
        "terms_url": _TERMS_URL,
        "required_notice": (
            "retain eBook landing-page provenance and do not imply Project "
            "Gutenberg endorsement"
        ),
        "acquired_date": acquired_date,
        "acquisition_evidence": acquisition_evidence,
        "text_modifications": (
            "Gutenberg boilerplate removed; audited source/segment exclusions "
            "applied; spelling and punctuation preserved"
        ),
        "downstream_note": (
            "Project Gutenberg trademark and redistribution terms remain applicable "
            "to Gutenberg-branded distribution"
        ),
        "artifact_status": source["source_decision"],
    }


def _verify_final_deduplication(
    config: GutenbergResolvedBuildConfig,
    *,
    temp_dir: Path,
    final_prefix: str,
    record_manifest: list[dict[str, Any]],
    standard_fingerprints: dict[str, TextFingerprint],
    progress: Progress | None,
) -> dict[str, Any]:
    manifest_by_id = {row["ebook_id"]: row for row in record_manifest}
    exact_groups: dict[str, list[str]] = defaultdict(list)
    for ebook_id, fingerprint in standard_fingerprints.items():
        exact_groups[fingerprint.normalized_word_sha256].append(ebook_id)
    duplicates = [sorted(ids, key=int) for ids in exact_groups.values() if len(ids) > 1]
    if duplicates:
        raise ValueError(
            f"final Gutenberg records contain normalized exact duplicates: {duplicates}"
        )

    internal_pairs = _discover_candidate_pairs(standard_fingerprints)
    internal_near = []
    for left_id, right_id in sorted(internal_pairs, key=lambda pair: (int(pair[0]), int(pair[1]))):
        metric = measure_word_shingle_containment(
            _read_temp_slice(temp_dir, final_prefix, manifest_by_id[left_id]),
            _read_temp_slice(temp_dir, final_prefix, manifest_by_id[right_id]),
        )
        if metric["containment"] >= config.near_duplicate_threshold:
            internal_near.append(
                {
                    "left_id": left_id,
                    "right_id": right_id,
                    "containment": round(metric["containment"], 6),
                }
            )
    if internal_near:
        raise ValueError(f"final Gutenberg records contain near duplicates: {internal_near}")

    references = _load_cross_references(config)
    reference_fingerprints: dict[str, TextFingerprint] = {}
    reference_started = monotonic()
    for index, (reference_id, reference) in enumerate(sorted(references.items()), start=1):
        reference_fingerprints[reference_id], _ = fingerprint_text(reference.read_text())
        if index == 1 or index % config.progress_interval == 0 or index == len(references):
            _progress(
                progress,
                "cross-reference-fingerprint",
                index,
                len(references),
                reference_started,
                reference_id,
            )
    cross_pairs = _discover_cross_candidates(standard_fingerprints, reference_fingerprints)
    cross_near = []
    cross_started = monotonic()
    ordered_cross = sorted(cross_pairs, key=lambda pair: (int(pair[0]), pair[1]))
    for index, (ebook_id, reference_id) in enumerate(ordered_cross, start=1):
        metric = measure_word_shingle_containment(
            _read_temp_slice(temp_dir, final_prefix, manifest_by_id[ebook_id]),
            references[reference_id].read_text(),
        )
        if metric["containment"] >= config.near_duplicate_threshold:
            cross_near.append(
                {
                    "ebook_id": ebook_id,
                    "reference_id": reference_id,
                    "containment": round(metric["containment"], 6),
                    "candidate_containment": round(metric["left_containment"], 6),
                    "reference_containment": round(metric["right_containment"], 6),
                }
            )
        if ordered_cross and (
            index == 1 or index % config.progress_interval == 0 or index == len(ordered_cross)
        ):
            _progress(
                progress,
                "cross-dedup",
                index,
                len(ordered_cross),
                cross_started,
                f"pg{ebook_id}:{reference_id}",
            )
    if cross_near:
        raise ValueError(
            f"final Gutenberg records retain cross-corpus near duplicates: {cross_near}"
        )
    return {
        "threshold": config.near_duplicate_threshold,
        "standard_record_count": len(standard_fingerprints),
        "normalized_exact_duplicate_group_count": 0,
        "internal_candidate_pair_count": len(internal_pairs),
        "internal_near_duplicate_pair_count": 0,
        "cross_reference_count": len(references),
        "cross_candidate_pair_count": len(cross_pairs),
        "cross_near_duplicate_pair_count": 0,
        "protected_v6_verification": (
            "hash-pinned checkpoint-3A reconstruction with zero residual source "
            "overlap and no materialized held-out candidate"
        ),
    }


def _load_cross_references(
    config: GutenbergResolvedBuildConfig,
) -> dict[str, TextReference]:
    references: dict[str, TextReference] = {}
    for row in _read_csv(config.bibit_record_manifest_path, allow_empty=True):
        if row.get("artifact_status") != "text_materialized" or not row.get("shard_path"):
            continue
        reference_id = f"bibit:{row['object_id']}"
        references[reference_id] = TextReference(
            reference_id=reference_id,
            source_kind="bibit",
            path=config.repo_root / row["shard_path"],
            byte_start=int(row["byte_start"]),
            byte_end=int(row["byte_end"]),
        )
    for row in _read_csv(config.broader_sources_manifest_path, allow_empty=True):
        path_value = row.get("expected_clean_text_path", "")
        if not path_value:
            continue
        path = config.repo_root / path_value
        if not path.is_file():
            continue
        reference_id = f"current:{row['source_id']}"
        references[reference_id] = TextReference(
            reference_id=reference_id,
            source_kind="current_corpus",
            path=path,
        )
    return references


def _discover_candidate_pairs(
    fingerprints: dict[str, TextFingerprint],
) -> set[tuple[str, str]]:
    candidates: set[tuple[str, str]] = set()
    for attribute in ("anchors", "sketch"):
        postings: dict[int, list[str]] = defaultdict(list)
        for document_id, fingerprint in fingerprints.items():
            for value in getattr(fingerprint, attribute):
                postings[value].append(document_id)
        collisions: Counter[tuple[str, str]] = Counter()
        for ids in postings.values():
            if 1 < len(ids) <= 40:
                collisions.update(
                    tuple(sorted(pair, key=int)) for pair in combinations(ids, 2)
                )
        for pair, count in collisions.items():
            denominator = min(
                len(getattr(fingerprints[pair[0]], attribute)),
                len(getattr(fingerprints[pair[1]], attribute)),
            )
            if denominator and count >= 2 and count / denominator >= 0.4:
                candidates.add(pair)
    return candidates


def _discover_cross_candidates(
    candidates: dict[str, TextFingerprint],
    references: dict[str, TextFingerprint],
) -> set[tuple[str, str]]:
    discovered: set[tuple[str, str]] = set()
    for attribute in ("anchors", "sketch"):
        postings: dict[int, list[str]] = defaultdict(list)
        for reference_id, fingerprint in references.items():
            for value in getattr(fingerprint, attribute):
                postings[value].append(reference_id)
        for ebook_id, fingerprint in candidates.items():
            collisions: Counter[str] = Counter()
            for value in getattr(fingerprint, attribute):
                reference_ids = postings.get(value, [])
                if len(reference_ids) <= 40:
                    collisions.update(reference_ids)
            for reference_id, count in collisions.items():
                denominator = min(
                    len(getattr(fingerprint, attribute)),
                    len(getattr(references[reference_id], attribute)),
                )
                if denominator and count >= 2 and count / denominator >= 0.4:
                    discovered.add((ebook_id, reference_id))
    return discovered


def _build_report(
    config: GutenbergResolvedBuildConfig,
    *,
    record_manifest: list[dict[str, Any]],
    segment_manifest: list[dict[str, Any]],
    sonnet_manifest: list[dict[str, Any]],
    shard_reports: dict[str, list[dict[str, Any]]],
    deduplication: dict[str, Any],
    manifest_paths: dict[str, Path],
) -> dict[str, Any]:
    materialized_records = [row for row in record_manifest if row["shard_path"]]
    materialized_sonnets = [row for row in sonnet_manifest if row["shard_path"]]
    record_counts = Counter(row["artifact_status"] for row in record_manifest)
    record_characters: Counter[str] = Counter()
    retained_source_characters: Counter[str] = Counter()
    for row in materialized_records:
        role = (
            "conditioned_source_variants"
            if row["artifact_status"] == "conditioned_text_materialized_inactive"
            else row["final_role"]
        )
        record_characters[role] += int(row["cleaned_character_count"])
        retained_source_characters[role] += int(
            row["retained_source_character_count"]
        )
    sonnet_counts = Counter(row["artifact_status"] for row in sonnet_manifest)
    sonnet_characters: Counter[str] = Counter()
    for row in materialized_sonnets:
        sonnet_characters[row["final_role"]] += int(row["cleaned_character_count"])
    prefix = _portable(config.output_dir, config.repo_root)
    return {
        "build_version": "project_gutenberg_resolved_v1",
        "inputs": {
            "source_decisions_path": _portable(config.source_decisions_path, config.repo_root),
            "source_decisions_sha256": _sha256_file(config.source_decisions_path),
            "segment_decisions_path": _portable(config.segment_decisions_path, config.repo_root),
            "segment_decisions_sha256": _sha256_file(config.segment_decisions_path),
            "sonnet_decisions_path": _portable(config.sonnet_decisions_path, config.repo_root),
            "sonnet_decisions_sha256": _sha256_file(config.sonnet_decisions_path),
            "sonnet_review_path": _portable(config.sonnet_review_path, config.repo_root),
            "sonnet_review_sha256": _sha256_file(config.sonnet_review_path),
            "audit_report_path": _portable(config.audit_report_path, config.repo_root),
            "audit_report_sha256": _sha256_file(config.audit_report_path),
            "inventory_path": _portable(config.inventory_path, config.repo_root),
            "inventory_sha256": _sha256_file(config.inventory_path),
            "bibit_record_manifest_sha256": _sha256_file(config.bibit_record_manifest_path),
            "broader_sources_manifest_sha256": _sha256_file(config.broader_sources_manifest_path),
        },
        "outputs": {
            "output_dir": prefix,
            "record_manifest_path": f"{prefix}/records_manifest.csv",
            "record_manifest_sha256": _sha256_file(manifest_paths["record"]),
            "segment_manifest_path": f"{prefix}/segments_manifest.csv",
            "segment_manifest_sha256": _sha256_file(manifest_paths["segment"]),
            "sonnet_manifest_path": f"{prefix}/sonnets_manifest.csv",
            "sonnet_manifest_sha256": _sha256_file(manifest_paths["sonnet"]),
            "attribution_manifest_path": f"{prefix}/attribution_manifest.csv",
            "attribution_manifest_sha256": _sha256_file(manifest_paths["attribution"]),
            "markdown_report_path": _portable(config.markdown_report_path, config.repo_root),
        },
        "max_shard_bytes": config.max_shard_bytes,
        "source_count": len(record_manifest),
        "materialized_source_count": len(materialized_records),
        "record_artifact_status_counts": dict(sorted(record_counts.items())),
        "record_characters_by_role": dict(sorted(record_characters.items())),
        "retained_source_characters_by_role": dict(
            sorted(retained_source_characters.items())
        ),
        "segment_count": len(segment_manifest),
        "segment_artifact_status_counts": dict(
            sorted(Counter(row["artifact_status"] for row in segment_manifest).items())
        ),
        "sonnet_candidate_count": len(sonnet_manifest),
        "materialized_sonnet_count": len(materialized_sonnets),
        "sonnet_artifact_status_counts": dict(sorted(sonnet_counts.items())),
        "sonnet_characters_by_role": dict(sorted(sonnet_characters.items())),
        "pending_poem_attribution_count": sum(
            row["poem_author_resolution"] == "pending_candidate_level_attribution_audit"
            and bool(row["shard_path"])
            for row in sonnet_manifest
        ),
        "deduplication": deduplication,
        "shards": shard_reports,
        "policy": {
            "text_storage": "bounded_plain_utf8_shards_with_exact_manifest_byte_ranges",
            "source_spelling_and_punctuation_preserved": True,
            "confirmed_sonnets_absent_from_broader_text": True,
            "manual_non_sonnet_false_positives_retained_in_broader_text": True,
            "conditioned_artifacts_materialized_but_inactive": True,
            "conditioned_experiment_authorized": False,
            "nineteenth_century_bridge_cap_applied": False,
            "nineteenth_century_bridge_cap_deferred_to_checkpoint_8": True,
            "poem_author_inferred_from_source_record": False,
            "v7_split_assigned": False,
            "training_mixture_weight_assigned": False,
            "local_primary_text_caches_deleted": False,
            "gpu_work_started": False,
        },
    }


def render_gutenberg_resolved_build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Project Gutenberg Resolved Corpus Build",
        "",
        "## Result",
        "",
        (
            f"Materialized {report['materialized_source_count']:,} of "
            f"{report['source_count']:,} audited source records and "
            f"{report['materialized_sonnet_count']:,} of "
            f"{report['sonnet_candidate_count']:,} sonnet candidates."
        ),
        "",
        "Every artifact is stored once in bounded UTF-8 shards. The manifests retain",
        "exact byte ranges and SHA-256 values for independent recovery and verification.",
        "",
        "## Record Roles",
        "",
        "| Role | Retained source characters | Materialized shard characters |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(
        f"| `{role}` | {report['retained_source_characters_by_role'][role]:,} | {characters:,} |"
        for role, characters in report["record_characters_by_role"].items()
    )
    lines.extend(
        [
            "",
            "## Sonnet Artifacts",
            "",
            "| Status | Candidates |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        f"| `{status}` | {count:,} |"
        for status, count in report["sonnet_artifact_status_counts"].items()
    )
    lines.extend(
        [
            "",
            "## Deduplication",
            "",
            (
                "- Standard broader-text records checked: "
                f"{report['deduplication']['standard_record_count']:,}."
            ),
            (
                "- Cross-corpus references checked: "
                f"{report['deduplication']['cross_reference_count']:,}."
            ),
            "- Final normalized exact duplicate groups: 0.",
            "- Final internal near-duplicate pairs at the frozen threshold: 0.",
            "- Final cross-corpus near-duplicate pairs at the frozen threshold: 0.",
            "- Protected V6 validation/test evidence remains excluded through the "
            "hash-pinned 3A reconstruction.",
            "",
            "## Boundaries",
            "",
            "- Standard sonnets are absent from historical-general, "
            "non-sonnet-poetry, and bridge shards.",
            "- Conditioned source and sonnet shards are materialized separately and "
            "remain inactive.",
            "- The full Ottocento candidate pool is materialized; no 10% exposure "
            "cap is applied yet.",
            "- Candidate-level poem authors are not guessed from source-record metadata.",
            "- No V7 split, training-mixture weight, cache deletion, or GPU work "
            "occurs in this build.",
            "",
            "## Artifacts",
            "",
            f"- Record manifest: `{report['outputs']['record_manifest_path']}`",
            f"- Segment manifest: `{report['outputs']['segment_manifest_path']}`",
            f"- Sonnet manifest: `{report['outputs']['sonnet_manifest_path']}`",
            f"- Attribution manifest: `{report['outputs']['attribution_manifest_path']}`",
            f"- Machine-readable report: `{report['outputs']['output_dir']}/build_report.json`",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_config(config: GutenbergResolvedBuildConfig) -> None:
    for path in (
        config.source_decisions_path,
        config.segment_decisions_path,
        config.sonnet_decisions_path,
        config.sonnet_review_path,
        config.audit_report_path,
        config.inventory_path,
        config.bibit_record_manifest_path,
        config.broader_sources_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if config.progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    if config.max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    if not 0 < config.near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")


def _validate_audit_hashes(
    config: GutenbergResolvedBuildConfig,
    report: dict[str, Any],
) -> None:
    if report.get("audit_version") != "project_gutenberg_extraction_audit_v1":
        raise ValueError("unsupported Gutenberg extraction audit version")
    expected = {
        "source_csv_sha256": config.source_decisions_path,
        "segment_csv_sha256": config.segment_decisions_path,
        "sonnet_csv_sha256": config.sonnet_decisions_path,
        "review_csv_sha256": config.sonnet_review_path,
    }
    for field, path in expected.items():
        if report.get("outputs", {}).get(field) != _sha256_file(path):
            raise ValueError(f"checkpoint-3A audit hash mismatch: {field}")
    if report.get("unresolved_sonnet_review_count") != 0:
        raise ValueError("checkpoint-3A sonnet review is unresolved")


def _validate_input_counts(
    config: GutenbergResolvedBuildConfig,
    sources: list[dict[str, str]],
    candidates: list[dict[str, str]],
) -> None:
    if len(sources) != config.expected_source_count:
        raise ValueError(
            f"expected {config.expected_source_count} source rows, found {len(sources)}"
        )
    source_counts = Counter(row["source_decision"] for row in sources)
    eligible = sum(source_counts[value] for value in _ELIGIBLE_SOURCE_DECISIONS)
    if eligible != config.expected_eligible_source_count:
        raise ValueError(
            f"expected {config.expected_eligible_source_count} eligible sources, found {eligible}"
        )
    if source_counts[_CONDITIONED_SOURCE_DECISION] != config.expected_conditioned_source_count:
        raise ValueError("conditioned source count changed")
    candidate_counts = Counter(row["candidate_decision"] for row in candidates)
    if candidate_counts[_STANDARD_SONNET_DECISION] != config.expected_standard_sonnet_count:
        raise ValueError("standard sonnet count changed")
    if candidate_counts[_CONDITIONED_SONNET_DECISION] != config.expected_conditioned_sonnet_count:
        raise ValueError("conditioned sonnet count changed")
    materializable_decisions = {
        _STANDARD_SONNET_DECISION,
        _CONDITIONED_SONNET_DECISION,
    }
    if any(
        row["heldout_reference_ids"]
        for row in candidates
        if row["candidate_decision"] in materializable_decisions
    ):
        raise ValueError("materializable sonnet contains protected held-out reference")
    if any(row["residual_heldout_overlap_ids"] for row in sources):
        raise ValueError("checkpoint-3A source decision retains protected V6 overlap")


def _validate_partitions(
    sources: dict[str, dict[str, str]],
    segments: dict[str, list[dict[str, str]]],
) -> None:
    for ebook_id, source in sources.items():
        rows = sorted(segments.get(ebook_id, []), key=lambda row: int(row["character_start"]))
        if not rows:
            raise ValueError(f"source has no segment partition: pg{ebook_id}")
        cursor = 0
        for row in rows:
            if row["source_cleaned_sha256"] != source["cleaned_sha256"]:
                raise ValueError(f"segment source hash mismatch: {row['segment_id']}")
            if int(row["character_start"]) != cursor:
                raise ValueError(f"segment partition gap: pg{ebook_id}")
            if int(row["character_end"]) - int(row["character_start"]) != int(
                row["character_count"]
            ):
                raise ValueError(f"segment length mismatch: {row['segment_id']}")
            cursor = int(row["character_end"])
        if cursor != int(source["cleaned_character_count"]):
            raise ValueError(f"segment partition does not cover source: pg{ebook_id}")


def _validate_materialized_counts(
    config: GutenbergResolvedBuildConfig,
    records: list[dict[str, Any]],
    sonnets: list[dict[str, Any]],
) -> None:
    standard_records = sum(
        row["artifact_status"] == "text_materialized_pending_v7" for row in records
    )
    conditioned_records = sum(
        row["artifact_status"] == "conditioned_text_materialized_inactive"
        for row in records
    )
    if standard_records != config.expected_eligible_source_count:
        raise ValueError("not every eligible standard source was materialized")
    if conditioned_records != config.expected_conditioned_source_count:
        raise ValueError("not every conditioned source was materialized separately")
    standard_sonnets = sum(
        row["artifact_status"] == "standard_sonnet_materialized_pending_v7"
        for row in sonnets
    )
    conditioned_sonnets = sum(
        row["artifact_status"] == "conditioned_sonnet_materialized_inactive"
        for row in sonnets
    )
    if standard_sonnets != config.expected_standard_sonnet_count:
        raise ValueError("not every eligible standard sonnet was materialized")
    if conditioned_sonnets != config.expected_conditioned_sonnet_count:
        raise ValueError("not every conditioned sonnet was materialized separately")


def _validate_segment_artifacts(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        materialized = row["artifact_status"].startswith("materialized_")
        has_path = bool(row["output_shard_path"])
        if materialized != has_path:
            raise ValueError(f"inconsistent segment artifact location: {row['segment_id']}")


def _validate_output_artifacts(
    temp_dir: Path,
    final_prefix: str,
    records: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    sonnets: list[dict[str, Any]],
) -> None:
    for row in (*records, *sonnets):
        if not row["shard_path"]:
            continue
        payload = _read_temp_slice(temp_dir, final_prefix, row)
        if _sha256_text(payload) != row["cleaned_sha256"]:
            identifier = row.get("candidate_id") or f"pg{row['ebook_id']}"
            raise ValueError(f"output slice hash mismatch: {identifier}")
    for row in segments:
        if not row["output_shard_path"]:
            continue
        payload = _read_temp_range(
            temp_dir,
            final_prefix,
            shard_path=row["output_shard_path"],
            byte_start=int(row["output_byte_start"]),
            byte_end=int(row["output_byte_end"]),
        )
        if _sha256_text(payload) != row["output_sha256"]:
            raise ValueError(f"output segment hash mismatch: {row['segment_id']}")


def _validate_shard_sizes(output_dir: Path, max_shard_bytes: int) -> None:
    oversized = [
        path for path in output_dir.rglob("part-*.txt") if path.stat().st_size > max_shard_bytes
    ]
    if oversized:
        raise ValueError(f"build created oversized shards: {oversized}")


def _replace_verified_output(temp_dir: Path, output_dir: Path) -> None:
    """Install a verified build while keeping the previous directory recoverable."""

    previous_dir = temp_dir.with_name(f"{temp_dir.name}.previous")
    had_previous = output_dir.exists()
    if had_previous:
        os.replace(output_dir, previous_dir)
    try:
        os.replace(temp_dir, output_dir)
    except Exception:
        if had_previous and previous_dir.exists():
            os.replace(previous_dir, output_dir)
        raise
    if had_previous:
        shutil.rmtree(previous_dir)


def _read_temp_slice(
    temp_dir: Path,
    final_prefix: str,
    row: dict[str, Any],
) -> str:
    return _read_temp_range(
        temp_dir,
        final_prefix,
        shard_path=row["shard_path"],
        byte_start=int(row["byte_start"]),
        byte_end=int(row["byte_end"]),
    )


def _read_temp_range(
    temp_dir: Path,
    final_prefix: str,
    *,
    shard_path: str,
    byte_start: int,
    byte_end: int,
) -> str:
    relative = Path(shard_path).relative_to(final_prefix)
    path = temp_dir / relative
    with path.open("rb") as handle:
        handle.seek(byte_start)
        payload = handle.read(byte_end - byte_start)
    return payload.decode("utf-8")


def _clean_candidate_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines).strip() + "\n"


def _canonical_text(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def _text_stats(text: str, *, include_lines: bool = False) -> dict[str, Any]:
    payload = text.encode("utf-8")
    stats: dict[str, Any] = {
        "cleaned_character_count": len(text),
        "cleaned_byte_count": len(payload),
        "cleaned_sha256": _sha256_bytes(payload),
    }
    if include_lines:
        stats["cleaned_line_count"] = sum(bool(line.strip()) for line in text.splitlines())
    return stats


def _read_csv(path: Path, *, allow_empty: bool = False) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows and not allow_empty:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _unique_by(
    rows: list[dict[str, str]],
    field: str,
    label: str,
) -> dict[str, dict[str, str]]:
    result = {row[field]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{label} contains duplicate {field} values")
    return result


def _write_csv(
    path: Path,
    fields: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _portable(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _progress(
    progress: Progress | None,
    phase: str,
    index: int,
    total: int,
    started: float,
    detail: str,
) -> None:
    if progress is None:
        return
    elapsed = monotonic() - started
    eta = elapsed / index * (total - index) if index else 0.0
    progress(
        f"{phase} {index:,}/{total:,} ({index / total:.1%}) {detail} "
        f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}"
    )


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"
