"""Resolve and materialize the bounded Liber Liber checkpoint-5C corpus."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, BinaryIO

from .gutenberg_extraction_audit import (
    GutenbergSonnetCandidate,
    TextSpan,
    _SonnetReference,
    _build_sonnet_reference_index,
    _candidate_reference_matches,
    _partition_spans,
    discover_gutenberg_sonnet_candidates,
    locate_reference_segments,
)
from .gutenberg_fulltext_probe import (
    _normalized_words,
    _rolling_shingle_hashes,
    fingerprint_text,
    measure_word_shingle_containment,
)
from .liber_liber_archive_probe import (
    LiberLiberArchiveProbeConfig,
    _load_references,
)
from .wikisource_page_extraction import _discover_cross_pairs, _discover_pairs


SOURCE_DECISION_FIELDS = (
    "record_id", "wordpress_page_id", "title", "author", "landing_page_url",
    "reference_edition", "period_bucket", "input_role", "final_broader_role",
    "probe_decision", "final_decision", "resolution_reason", "extraction_policy",
    "canonical_reference_ids", "removed_reference_ids", "license_label", "license_url",
    "source_cache_path", "source_sha256", "source_character_count",
    "retained_broader_character_count", "excluded_source_character_count",
    "sonnet_candidate_count", "eligible_sonnet_count", "protected_sonnet_count",
    "activation_status",
)
SEGMENT_DECISION_FIELDS = (
    "segment_id", "record_id", "source_sha256", "character_start", "character_end",
    "character_count", "segment_sha256", "segment_decision", "final_role", "reason",
    "reference_ids", "start_anchor", "end_anchor",
)
SONNET_DECISION_FIELDS = (
    "candidate_id", "record_id", "source_title", "source_record_author", "poem_author",
    "poem_author_resolution", "period_bucket", "source_url", "source_kind",
    "stanza_pattern", "line_count", "character_start", "character_end",
    "source_text_sha256", "cleaned_text_sha256", "first_line", "last_line",
    "exact_reference_ids", "near_reference_ids", "protected_v6_reference_ids",
    "candidate_decision", "final_role", "activation_status",
)
SONNET_REVIEW_FIELDS = (
    "review_id", "record_id", "candidate_id", "review_type", "evidence_sha256",
    "review_resolution", "review_rationale",
)
CANONICAL_EDITION_FIELDS = (
    "canonical_group_id", "record_id", "title", "reference_edition",
    "canonical_decision", "selected_record_id", "decision_evidence",
)

RECORD_MANIFEST_FIELDS = SOURCE_DECISION_FIELDS + (
    "artifact_status", "shard_path", "byte_start", "byte_end",
    "cleaned_character_count", "cleaned_byte_count", "cleaned_sha256",
)
SEGMENT_MANIFEST_FIELDS = SEGMENT_DECISION_FIELDS + (
    "activation_status", "artifact_status", "output_shard_path",
    "output_byte_start", "output_byte_end", "output_sha256",
)
SONNET_MANIFEST_FIELDS = SONNET_DECISION_FIELDS + (
    "artifact_status", "shard_path", "byte_start", "byte_end",
    "cleaned_character_count", "cleaned_byte_count", "cleaned_sha256",
)
ATTRIBUTION_MANIFEST_FIELDS = (
    "record_id", "title", "author", "landing_page_url", "reference_edition",
    "editor", "translator", "digitization_credit", "layout_credit",
    "publication_credit", "revision_credit", "license_label", "license_url",
    "book_license_terms_url", "required_notice", "modification_notice",
    "downstream_note", "activation_status",
)

_BROADER_ROLES = {
    "historical_general", "historical_non_sonnet_poetry", "nineteenth_century_bridge",
}
_FULL_DUPLICATE = "exclude_cross_corpus_duplicate_candidate"
_VITA_SELECTED = "ll:2344213"
_VITA_ALTERNATES = {"ll:2344214", "ll:2344215", "ll:2428858"}
_COMPOSITE_CANONICAL = {"ll:2346840", "ll:2428878", "ll:2433191"}
_CARDUCCI = "ll:2344854"
_CENNI = "ll:2344942"
_VARALDO = "ll:2427167"
_VARALDO_AUTHORS = re.compile(
    r"^(Alessandro Varaldo|Mario Malfettani|Alessandro Giribaldi)\.?$", re.I
)
Progress = Callable[[str], None]


@dataclass(frozen=True)
class LiberLiberResolvedBuildConfig:
    repo_root: Path
    probe_path: Path
    probe_review_path: Path
    probe_report_path: Path
    archive_inventory_path: Path
    source_rights_path: Path
    source_decisions_path: Path
    segment_decisions_path: Path
    sonnet_decisions_path: Path
    sonnet_review_path: Path
    canonical_editions_path: Path
    output_dir: Path
    json_report_path: Path
    markdown_report_path: Path
    attribution_notice_path: Path
    bibit_record_manifest_path: Path
    bibit_sonnet_manifest_path: Path
    gutenberg_previous_probe_path: Path
    gutenberg_previous_cache_dir: Path
    gutenberg_pass_1b_probe_path: Path
    gutenberg_pass_1b_cache_dir: Path
    gutenberg_resolved_record_manifest_path: Path
    gutenberg_resolved_sonnet_manifest_path: Path
    wikisource_resolved_record_manifest_path: Path
    wikisource_resolved_sonnet_manifest_path: Path
    broader_sources_manifest_path: Path
    protected_v6_sonnet_manifest_path: Path
    cache_dir: Path
    expected_record_count: int = 129
    expected_conditioned_count: int = 151
    max_shard_bytes: int = 64 * 1024 * 1024
    near_duplicate_threshold: float = 0.8
    sonnet_near_overlap_threshold: float = 0.72
    sonnet_near_sequence_threshold: float = 0.86
    progress_interval: int = 25


def build_liber_liber_resolved_corpus(
    config: LiberLiberResolvedBuildConfig,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Freeze checkpoint-5C ledgers and install inactive processed shards."""

    _validate_inputs(config)
    started = monotonic()
    probe_rows = _read_csv(config.probe_path)
    inventory = _unique(_read_csv(config.archive_inventory_path), "record_id")
    rights = _unique(_read_csv(config.source_rights_path), "record_id")
    texts = {row["record_id"]: _read_source(config, row) for row in probe_rows}
    references = _load_sonnet_references(config)
    reference_index = _build_sonnet_reference_index(references)
    reference_by_id = {row.reference_id: row for row in references}

    raw_candidates = _discover_sonnets(probe_rows, texts)
    sonnets = _resolve_sonnets(
        raw_candidates,
        probe_rows=probe_rows,
        reference_by_id=reference_by_id,
        references=references,
        reference_index=reference_index,
        config=config,
    )
    sonnets_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sonnets:
        sonnets_by_record[row["record_id"]].append(row)

    cross_references = _load_references(_probe_config(config))
    sources: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    reviews = _review_rows(texts, sonnets)
    canonical = _canonical_rows(inventory)
    for index, probe in enumerate(sorted(probe_rows, key=_record_sort_key), start=1):
        record_id = probe["record_id"]
        source, record_segments = _resolve_source(
            probe,
            inventory[record_id],
            rights[record_id],
            texts[record_id],
            sonnets_by_record.get(record_id, []),
            cross_references,
        )
        sources.append(source)
        segments.extend(record_segments)
        if progress and (index == 1 or index % config.progress_interval == 0 or index == len(probe_rows)):
            _emit(progress, "resolve", index, len(probe_rows), started)

    _validate_resolution(config, sources, segments, sonnets, reviews, canonical)
    _write_csv(config.source_decisions_path, SOURCE_DECISION_FIELDS, sources)
    _write_csv(config.segment_decisions_path, SEGMENT_DECISION_FIELDS, segments)
    _write_csv(config.sonnet_decisions_path, SONNET_DECISION_FIELDS, sonnets)
    _write_csv(config.sonnet_review_path, SONNET_REVIEW_FIELDS, reviews)
    _write_csv(config.canonical_editions_path, CANONICAL_EDITION_FIELDS, canonical)

    report = _materialize(
        config,
        sources=sources,
        segments=segments,
        sonnets=sonnets,
        inventory=inventory,
        rights=rights,
        texts=texts,
        progress=progress,
    )
    _write_json(config.json_report_path, report)
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(render_markdown(report), encoding="utf-8")
    config.attribution_notice_path.parent.mkdir(parents=True, exist_ok=True)
    config.attribution_notice_path.write_text(render_attribution_notice(report), encoding="utf-8")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    return (
        "# Liber Liber Resolved Corpus Build v1\n\n"
        "## Result\n\n"
        f"The deterministic inactive build materializes {report['materialized_record_count']:,} "
        f"broader records and {report['materialized_sonnet_count']:,} unique fourteen-line sonnets.\n\n"
        f"- Retained broader characters: {report['materialized_broader_character_count']:,}.\n"
        f"- Standard-sonnet characters: {report['materialized_sonnet_character_count']:,}.\n"
        f"- Excluded full/composite/alternate sources: {report['excluded_source_count']:,}.\n"
        f"- Attribution rows: {report['attribution_count']:,}.\n"
        f"- Bounded shards: {report['shard_count']:,}.\n"
        "- Final normalized exact, near-duplicate, cross-corpus, and protected-V6 gates pass.\n\n"
        "## Boundary\n\n"
        "All text remains inactive pending checkpoint 7 cross-archive canonicalization and checkpoint 8 "
        "V7/split/mixture freeze. No conditioned material, V7 split, mixture weight, cache deletion, "
        "corpus activation, or GPU work is included.\n"
    )


def render_attribution_notice(report: dict[str, Any]) -> str:
    return (
        "# Liber Liber Resolved Corpus v1: Attribution And Reuse\n\n"
        "## Source And Terms\n\n"
        "The processed text comes from [Liber Liber / Progetto Manuzio](https://liberliber.it/). "
        "Each contributing edition is recorded in `data/processed/liber_liber_resolved_v1/"
        "attribution_manifest.csv` with its stable item URL, edition statement, named contributors, "
        "and item-specific credit. The edition layer is licensed under [CC BY-NC-SA 4.0]"
        "(https://creativecommons.org/licenses/by-nc-sa/4.0/); non-commercial, attribution, "
        "and ShareAlike obligations remain attached to redistributed edition-derived text.\n\n"
        "## Audited Scope\n\n"
        f"Checkpoint 5C accounts for all {report['source_count']:,} probed records. The inactive build "
        f"contains {report['materialized_record_count']:,} broader records and "
        f"{report['materialized_sonnet_count']:,} unique sonnets. Fully covered editions, alternate "
        "editions, embedded canonical text, protected V6 sonnets, and sonnets already represented "
        "by existing corpora remain excluded with exact source-span decisions.\n\n"
        "## Required Credit And Modification Notice\n\n"
        "Redistribution must credit Liber Liber and every named contributor in the attribution "
        "manifest, retain the stable item and license links, identify this project as the modifier, "
        "and state that Liber Liber wrappers and canonical/protected/sonnet spans were removed while "
        "retained spelling, punctuation, and dialogue-terminal hyphens were preserved. The source "
        "edition limitation for `ll:2344098` is retained explicitly.\n\n"
        "These artifacts are materialized but inactive. This checkpoint creates no V7 split or "
        "mixture weight and authorizes no model training.\n"
    )


def _discover_sonnets(
    probe_rows: list[dict[str, str]], texts: dict[str, str]
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in probe_rows:
        record_id = row["record_id"]
        if row["probe_decision"] == _FULL_DUPLICATE or record_id in _VITA_ALTERNATES | _COMPOSITE_CANONICAL:
            continue
        text = texts[record_id]
        role = row["preliminary_role"]
        if record_id == _VITA_SELECTED:
            role = "sonnet_specialization"
        discovered = discover_gutenberg_sonnet_candidates(
            text,
            ebook_id=row["wordpress_page_id"],
            title=row["title"],
            authors=row["author"],
            role="sonnet_specialization" if role == "standard_sonnets" else role,
            allowed_ranges=[(0, len(text))],
        )
        for candidate in discovered:
            candidates.append(_candidate_dict(record_id, candidate, row["author"], "source_collection_author"))
        if record_id == _VARALDO:
            candidates.extend(_discover_varaldo(text, row))
    return sorted(candidates, key=lambda item: (_record_number(item["record_id"]), item["start"]))


def _discover_varaldo(text: str, row: dict[str, str]) -> list[dict[str, Any]]:
    physical: list[tuple[int, int, str]] = []
    position = 0
    for raw in text.splitlines(keepends=True):
        physical.append((position, position + len(raw), raw.rstrip("\r\n")))
        position += len(raw)
    found = []
    for index, (_start, _end, value) in enumerate(physical):
        match = _VARALDO_AUTHORS.fullmatch(value.strip())
        if not match or index < 17 or index < 14:
            continue
        selected = physical[index - 14:index]
        if any(not line.strip() for _a, _b, line in selected):
            continue
        raw_start, raw_end = selected[0][0], selected[-1][1]
        raw = text[raw_start:raw_end]
        cleaned = "\n".join(line.strip() for _a, _b, line in selected) + "\n"
        if len(cleaned.strip().splitlines()) != 14:
            raise ValueError("Varaldo signature extraction changed")
        found.append({
            "record_id": row["record_id"], "start": raw_start, "end": raw_end,
            "raw": raw, "cleaned": cleaned, "source_kind": "explicit_author_signature",
            "stanza_pattern": "14", "poem_author": match.group(1),
            "poem_author_resolution": "explicit_post_poem_signature",
        })
    if len(found) != 39:
        raise ValueError(f"Varaldo expected 39 signed sonnets, found {len(found)}")
    return found


def _candidate_dict(
    record_id: str,
    candidate: GutenbergSonnetCandidate,
    author: str,
    author_resolution: str,
) -> dict[str, Any]:
    return {
        "record_id": record_id, "start": candidate.start, "end": candidate.end,
        "raw": candidate.text, "cleaned": candidate.cleaned_text,
        "source_kind": candidate.source_kind, "stanza_pattern": candidate.stanza_pattern,
        "poem_author": author, "poem_author_resolution": author_resolution,
    }


def _resolve_sonnets(
    candidates: list[dict[str, Any]],
    *,
    probe_rows: list[dict[str, str]],
    references: list[_SonnetReference],
    reference_by_id: dict[str, _SonnetReference],
    reference_index: dict[str, Any],
    config: LiberLiberResolvedBuildConfig,
) -> list[dict[str, Any]]:
    probe_by_id = _unique(probe_rows, "record_id")
    rows = []
    for candidate in candidates:
        exact, near = _candidate_reference_matches(
            candidate["cleaned"], references=references, reference_by_id=reference_by_id,
            index=reference_index,
            near_overlap_threshold=config.sonnet_near_overlap_threshold,
            near_sequence_threshold=config.sonnet_near_sequence_threshold,
        )
        matched = exact or near
        protected = sorted(
            reference_id for reference_id in matched
            if reference_id.startswith("v6:")
            and reference_by_id[reference_id].split in {"validation", "test"}
        )
        if protected:
            decision = "exclude_protected_v6_sonnet"
        elif matched:
            decision = "exclude_existing_corpus_sonnet_duplicate"
        else:
            decision = "eligible_standard_sonnet_inactive_pending_checkpoint7"
        source = probe_by_id[candidate["record_id"]]
        lines = candidate["cleaned"].strip().splitlines()
        candidate_id = f"{candidate['record_id']}:char{candidate['start']}-{candidate['end']}"
        rows.append({
            "candidate_id": candidate_id, "record_id": candidate["record_id"],
            "source_title": source["title"], "source_record_author": source["author"],
            "poem_author": candidate["poem_author"],
            "poem_author_resolution": candidate["poem_author_resolution"],
            "period_bucket": source["period_bucket"], "source_url": source["landing_page_url"],
            "source_kind": candidate["source_kind"], "stanza_pattern": candidate["stanza_pattern"],
            "line_count": len(lines), "character_start": candidate["start"],
            "character_end": candidate["end"], "source_text_sha256": _sha(candidate["raw"]),
            "cleaned_text_sha256": _sha(candidate["cleaned"]), "first_line": lines[0],
            "last_line": lines[-1], "exact_reference_ids": ";".join(exact),
            "near_reference_ids": ";".join(near),
            "protected_v6_reference_ids": ";".join(protected),
            "candidate_decision": decision, "final_role": "standard_sonnets",
            "activation_status": "inactive_pending_cross_archive_freeze",
        })
    return rows


def _resolve_source(
    probe: dict[str, str],
    inventory: dict[str, str],
    rights: dict[str, str],
    text: str,
    sonnets: list[dict[str, Any]],
    references: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    record_id = probe["record_id"]
    canonical_ids = _metric_reference_ids(probe["cross_corpus_overlap_metrics"])
    removals: list[TextSpan] = []
    final_role = probe["preliminary_role"]
    final_decision = "eligible_inactive_processed_build"
    reason = "quality_and_rights_gates_pass"
    policy = "preserve_probe_cleaned_primary_text"
    forced_spans: list[TextSpan] | None = None

    if probe["probe_decision"] == _FULL_DUPLICATE:
        final_decision = "exclude_canonical_cross_corpus_duplicate"
        reason = "existing_corpus_has_canonical_complete_text"
        policy = "exclude_complete_source"
        removals.append(TextSpan(0, len(text), "exclude_full_source_duplicate", "excluded", reason, tuple(canonical_ids)))
    elif record_id in _VITA_ALTERNATES:
        final_decision = "exclude_alternate_vita_nuova_edition"
        reason = f"canonical_liber_liber_edition_is_{_VITA_SELECTED}"
        policy = "exclude_alternate_source_edition"
        removals.append(TextSpan(0, len(text), "exclude_alternate_canonical_edition", "excluded", reason, (_VITA_SELECTED,)))
    elif record_id in _COMPOSITE_CANONICAL:
        final_decision = "exclude_composite_source_covered_by_existing_corpora"
        reason = "combined_canonical_sources_cover_primary_work"
        policy = "exclude_complete_composite_source"
        ids = canonical_ids
        if record_id == "ll:2346840":
            ids = sorted(set(ids) | {"v6:complete_petrarch_sonnet_sequence"})
        removals.append(TextSpan(0, len(text), "exclude_composite_canonical_source", "excluded", reason, tuple(ids)))
    elif record_id == _CARDUCCI:
        embedded_ids = _embedded_reference_ids(probe["cross_corpus_overlap_metrics"])
        if embedded_ids != ["bibit:bibit000521"]:
            raise ValueError("Carducci embedded-reference evidence changed")
        all_ids = ["bibit:bibit000521", "bibit:bibit001121"]
        expected = {
            "bibit:bibit000521": [(32, 48311, 6181)],
            "bibit:bibit001121": [(48458, 53399, 770)],
        }
        for reference_id in all_ids:
            matches = locate_reference_segments(text, references[reference_id].read_text())
            if matches != expected[reference_id]:
                raise ValueError(f"Carducci canonical span changed for {reference_id}: {matches}")
            start, end, _anchors = matches[0]
            removals.append(TextSpan(
                start, end, "exclude_embedded_canonical_text", "excluded",
                "embedded_text_exists_in_bibit", (reference_id,),
            ))
        partition = _partition_spans(
            text, removals, default_role="excluded",
            default_decision="exclude_no_substantial_unique_material_after_canonicalization",
        )
        forced_spans = partition
        final_decision = "exclude_composite_source_covered_by_existing_corpora"
        reason = "two_bibit_records_cover_primary_collection_and_remaining_fragments_are_headings_or_three_line_congedo"
        policy = "exclude_complete_composite_source_after_two_segment_canonicalization"
        canonical_ids = sorted(set(canonical_ids) | set(all_ids))
    elif record_id == _VARALDO:
        final_role = "standard_sonnets"
        final_decision = "eligible_sonnet_only_inactive_processed_build"
        reason = "39_explicitly_signed_fourteen_line_sonnets_isolated"
        policy = "materialize_verified_sonnets_only_exclude_preface_and_headings"
        removals.append(TextSpan(0, len(text), "exclude_sonnet_collection_wrapper_from_broader_text", "excluded", reason))
    elif record_id == _CENNI:
        final_role = "historical_non_sonnet_poetry"
        reason = "source_title_sonetti_but_units_are_17_or_20_line_caudate_riddles"
        policy = "retain_as_historical_non_sonnet_poetry_no_target_form_activation"

    if final_decision == "eligible_inactive_processed_build":
        for sonnet in sonnets:
            decision = sonnet["candidate_decision"]
            segment_decision = (
                "exclude_protected_v6_sonnet" if decision == "exclude_protected_v6_sonnet"
                else "exclude_isolated_sonnet_from_broader_text"
            )
            reference_ids = tuple(filter(None, (
                sonnet["protected_v6_reference_ids"] or sonnet["exact_reference_ids"] or sonnet["near_reference_ids"]
            ).split(";")))
            removals.append(TextSpan(
                int(sonnet["character_start"]), int(sonnet["character_end"]),
                segment_decision, "excluded", "sonnet_isolated_from_broader_stage", reference_ids,
            ))

    if forced_spans is not None:
        spans = forced_spans
    elif final_decision == "eligible_inactive_processed_build":
        spans = _partition_spans(text, removals, default_role=final_role, default_decision="include_broader_text")
    else:
        spans = removals
    retained = sum(span.end - span.start for span in spans if span.decision == "include_broader_text")
    removed_ids = sorted({reference for span in spans if span.decision != "include_broader_text" for reference in span.reference_ids})
    row = {
        "record_id": record_id, "wordpress_page_id": probe["wordpress_page_id"],
        "title": probe["title"], "author": probe["author"],
        "landing_page_url": probe["landing_page_url"],
        "reference_edition": inventory["reference_edition"], "period_bucket": probe["period_bucket"],
        "input_role": probe["preliminary_role"], "final_broader_role": final_role,
        "probe_decision": probe["probe_decision"], "final_decision": final_decision,
        "resolution_reason": reason, "extraction_policy": policy,
        "canonical_reference_ids": ";".join(canonical_ids),
        "removed_reference_ids": ";".join(removed_ids),
        "license_label": rights["license_label"], "license_url": rights["license_url"],
        "source_cache_path": probe["cleaned_cache_path"], "source_sha256": _sha(text),
        "source_character_count": len(text), "retained_broader_character_count": retained,
        "excluded_source_character_count": len(text) - retained,
        "sonnet_candidate_count": len(sonnets),
        "eligible_sonnet_count": sum(row["candidate_decision"] == "eligible_standard_sonnet_inactive_pending_checkpoint7" for row in sonnets),
        "protected_sonnet_count": sum(row["candidate_decision"] == "exclude_protected_v6_sonnet" for row in sonnets),
        "activation_status": "inactive_pending_cross_archive_freeze",
    }
    return row, _segment_rows(record_id, text, spans)


def _segment_rows(record_id: str, text: str, spans: list[TextSpan]) -> list[dict[str, Any]]:
    rows = []
    for index, span in enumerate(spans, start=1):
        payload = text[span.start:span.end]
        rows.append({
            "segment_id": f"{record_id}:seg{index:04d}", "record_id": record_id,
            "source_sha256": _sha(text), "character_start": span.start,
            "character_end": span.end, "character_count": len(payload),
            "segment_sha256": _sha(payload), "segment_decision": span.decision,
            "final_role": span.role, "reason": span.reason,
            "reference_ids": ";".join(span.reference_ids),
            "start_anchor": " ".join(payload[:120].split()),
            "end_anchor": " ".join(payload[-120:].split()),
        })
    return rows


def _review_rows(texts: dict[str, str], sonnets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    abati = next(row for row in sonnets if row["record_id"] == "ll:2344098")
    return [
        {
            "review_id": "ll-review:2344098:structural-sonnet", "record_id": "ll:2344098",
            "candidate_id": abati["candidate_id"], "review_type": "structural_sonnet",
            "evidence_sha256": abati["source_text_sha256"],
            "review_resolution": "accept_structurally_verified_standard_sonnet",
            "review_rationale": "The bounded 4-4-3-3 window is exactly fourteen verse lines and is a complete sonnet embedded in the authorial poetry collection.",
        },
        {
            "review_id": "ll-review:2344942:source-form", "record_id": "ll:2344942",
            "candidate_id": "", "review_type": "source_form",
            "evidence_sha256": _sha(texts["ll:2344942"]),
            "review_resolution": "retain_as_historical_non_sonnet_poetry",
            "review_rationale": "The source title says Sonetti, but its labeled riddle units contain 17 or 20 lines with codas; none qualifies for the exact fourteen-line target-sonnet artifact.",
        },
    ]


def _canonical_rows(inventory: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    result = []
    for record_id in [_VITA_SELECTED, "ll:2344214", "ll:2344215", "ll:2428858"]:
        selected = record_id == _VITA_SELECTED
        result.append({
            "canonical_group_id": "liber-liber:vita-nuova", "record_id": record_id,
            "title": inventory[record_id]["title"],
            "reference_edition": inventory[record_id]["reference_edition"],
            "canonical_decision": "select_primary_canonical_edition" if selected else "exclude_alternate_edition",
            "selected_record_id": _VITA_SELECTED,
            "decision_evidence": (
                "Clean complete TXT edition with minimal front matter; retain prose and non-sonnet poetry after sonnet isolation."
                if selected else
                "Alternate edition adds no required unique primary work; canonical precedence selects ll:2344213 over editorial/modernized alternatives."
            ),
        })
    return result


def _materialize(
    config: LiberLiberResolvedBuildConfig,
    *,
    sources: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    sonnets: list[dict[str, Any]],
    inventory: dict[str, dict[str, str]],
    rights: dict[str, dict[str, str]],
    texts: dict[str, str],
    progress: Progress | None,
) -> dict[str, Any]:
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{config.output_dir.name}.", dir=config.output_dir.parent))
    prefix = config.output_dir.relative_to(config.repo_root).as_posix()
    writers = {
        role: _ShardWriter(temp / role, f"{prefix}/{role}", config.max_shard_bytes)
        for role in (*sorted(_BROADER_ROLES), "standard_sonnets")
    }
    segments_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sonnets_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in segments:
        segments_by_record[row["record_id"]].append(row)
    for row in sonnets:
        sonnets_by_record[row["record_id"]].append(row)
    record_manifest, segment_manifest, sonnet_manifest, attribution = [], [], [], []
    materialized_texts: dict[str, str] = {}
    materialized_sonnets: dict[str, str] = {}
    started = monotonic()
    try:
        for index, source in enumerate(sorted(sources, key=_record_sort_key), start=1):
            record_id = source["record_id"]
            source_text = texts[record_id]
            output_segments = [dict(row) for row in sorted(segments_by_record[record_id], key=lambda row: int(row["character_start"]))]
            included = [row for row in output_segments if row["segment_decision"] == "include_broader_text"]
            record_text, ranges = _compose_segments(source_text, included)
            location = _empty_location()
            artifact_status = "not_materialized_excluded_or_sonnet_only"
            if source["final_decision"] == "eligible_inactive_processed_build" and record_text.strip():
                role = source["final_broader_role"]
                if role not in _BROADER_ROLES:
                    raise ValueError(f"unsupported Liber Liber broader role: {role}")
                record_text = _canonical(record_text)
                location = writers[role].add(record_id, record_text)
                artifact_status = "text_materialized_inactive"
                materialized_texts[record_id] = record_text
                for segment, relative_start, relative_end in ranges:
                    segment.update({
                        "artifact_status": "materialized_in_inactive_record",
                        "output_shard_path": location["shard_path"],
                        "output_byte_start": int(location["byte_start"]) + relative_start,
                        "output_byte_end": int(location["byte_start"]) + relative_end,
                        "output_sha256": segment["segment_sha256"],
                    })
            for segment in output_segments:
                segment.setdefault("activation_status", "inactive_pending_cross_archive_freeze")
                segment.setdefault("artifact_status", "not_materialized")
                segment.setdefault("output_shard_path", "")
                segment.setdefault("output_byte_start", "")
                segment.setdefault("output_byte_end", "")
                segment.setdefault("output_sha256", "")
            segment_manifest.extend(output_segments)
            record_manifest.append({**source, "artifact_status": artifact_status, **location, **_stats(record_text)})

            any_sonnet = False
            for sonnet in sorted(sonnets_by_record.get(record_id, []), key=lambda row: int(row["character_start"])):
                built, cleaned = _materialize_sonnet(sonnet, source_text, writers["standard_sonnets"])
                sonnet_manifest.append(built)
                if cleaned:
                    any_sonnet = True
                    materialized_sonnets[sonnet["candidate_id"]] = cleaned
            if artifact_status == "text_materialized_inactive" or any_sonnet:
                attribution.append(_attribution_row(record_id, inventory[record_id], rights[record_id]))
            if progress and (index == 1 or index % config.progress_interval == 0 or index == len(sources)):
                _emit(progress, "materialize", index, len(sources), started)

        shard_reports = {role: writer.close() for role, writer in writers.items()}
        verification = _verify_final(config, materialized_texts, materialized_sonnets, progress)
        _write_csv(temp / "records_manifest.csv", RECORD_MANIFEST_FIELDS, record_manifest)
        _write_csv(temp / "segments_manifest.csv", SEGMENT_MANIFEST_FIELDS, segment_manifest)
        _write_csv(temp / "sonnets_manifest.csv", SONNET_MANIFEST_FIELDS, sonnet_manifest)
        _write_csv(temp / "attribution_manifest.csv", ATTRIBUTION_MANIFEST_FIELDS, attribution)
        report = _build_report(
            config, record_manifest, segment_manifest, sonnet_manifest, attribution,
            shard_reports, verification, temp,
        )
        _write_json(temp / "build_report.json", report)
        _validate_artifacts(temp, prefix, record_manifest, segment_manifest, sonnet_manifest)
        _replace_verified_output(temp, config.output_dir)
    except BaseException:
        for writer in writers.values():
            writer.abort()
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return report


def _materialize_sonnet(
    row: dict[str, Any], source: str, writer: "_ShardWriter"
) -> tuple[dict[str, Any], str]:
    start, end = int(row["character_start"]), int(row["character_end"])
    raw = source[start:end]
    if _sha(raw) != row["source_text_sha256"]:
        raise ValueError(f"sonnet source hash mismatch: {row['candidate_id']}")
    cleaned = "\n".join(line.strip() for line in raw.splitlines() if line.strip()).strip() + "\n"
    if _sha(cleaned) != row["cleaned_text_sha256"] or len(cleaned.strip().splitlines()) != 14:
        raise ValueError(f"sonnet cleaning mismatch: {row['candidate_id']}")
    result = dict(row)
    if row["candidate_decision"] == "eligible_standard_sonnet_inactive_pending_checkpoint7":
        location = writer.add(row["candidate_id"], cleaned)
        result.update({"artifact_status": "sonnet_materialized_inactive", **location, **_stats(cleaned)})
        return result, cleaned
    result.update({"artifact_status": "not_materialized_duplicate_or_protected", **_empty_location(), **_stats("")})
    return result, ""


def _verify_final(
    config: LiberLiberResolvedBuildConfig,
    records: dict[str, str],
    sonnets: dict[str, str],
    progress: Progress | None,
) -> dict[str, Any]:
    combined = {f"record:{key}": value for key, value in records.items()} | {
        f"sonnet:{key}": value for key, value in sonnets.items()
    }
    fingerprints = {key: fingerprint_text(value)[0] for key, value in combined.items()}
    exact = Counter(fp.normalized_word_sha256 for fp in fingerprints.values() if fp.word_count)
    if any(value > 1 for value in exact.values()):
        raise ValueError("final Liber Liber artifacts contain normalized exact duplicates")
    internal_checked = 0
    for left, right in _discover_pairs(fingerprints):
        internal_checked += 1
        if measure_word_shingle_containment(combined[left], combined[right])["containment"] >= config.near_duplicate_threshold:
            raise ValueError(f"final internal near duplicate: {left} / {right}")

    refs = _load_references(_probe_config(config))
    for row in _read_csv(config.protected_v6_sonnet_manifest_path):
        reference_id = f"v6:{row['poem_id']}"
        if reference_id not in refs:
            refs[reference_id] = _PathReference(config.repo_root / row["clean_text_path"])
    ref_fingerprints = {key: fingerprint_text(value.read_text())[0] for key, value in refs.items()}
    cross_checked = 0
    for candidate_id, reference_id in _discover_cross_pairs(fingerprints, ref_fingerprints):
        cross_checked += 1
        if measure_word_shingle_containment(combined[candidate_id], refs[reference_id].read_text())["containment"] >= config.near_duplicate_threshold:
            raise ValueError(f"final cross-corpus near duplicate: {candidate_id} / {reference_id}")

    watch: dict[int, list[str]] = defaultdict(list)
    denominators = {}
    for row in _read_csv(config.protected_v6_sonnet_manifest_path):
        if row["split_expanded_with_petrarch"] not in {"validation", "test"}:
            continue
        text = (config.repo_root / row["clean_text_path"]).read_text(encoding="utf-8")
        values = set(_rolling_shingle_hashes(_normalized_words(text)))
        denominators[row["poem_id"]] = len(values)
        for value in values:
            watch[value].append(row["poem_id"])
    frozen_watch = {value: tuple(ids) for value, ids in watch.items()}
    protected_checked = 0
    started = monotonic()
    for index, (artifact_id, text) in enumerate(sorted(combined.items()), start=1):
        _fingerprint, hits = fingerprint_text(text, watched_shingles=frozen_watch)
        protected_checked += len(hits)
        for poem_id, values in hits.items():
            if len(values) / max(1, denominators[poem_id]) >= config.near_duplicate_threshold:
                raise ValueError(f"protected V6 overlap remains: {artifact_id} / {poem_id}")
        if progress and (index == 1 or index % config.progress_interval == 0 or index == len(combined)):
            _emit(progress, "protected-verification", index, len(combined), started)
    return {
        "internal_candidate_pairs_checked": internal_checked,
        "cross_candidate_pairs_checked": cross_checked,
        "protected_candidate_pairs_checked": protected_checked,
        "normalized_exact_duplicate_count": 0, "near_duplicate_count": 0,
        "cross_duplicate_count": 0, "protected_overlap_count": 0,
    }


def _load_sonnet_references(config: LiberLiberResolvedBuildConfig) -> list[_SonnetReference]:
    result = []
    for row in _read_csv(config.protected_v6_sonnet_manifest_path):
        text = (config.repo_root / row["clean_text_path"]).read_text(encoding="utf-8")
        result.append(_SonnetReference(f"v6:{row['poem_id']}", "sonnets_v6", row["split_expanded_with_petrarch"], text))
    for prefix, manifest, id_field, status in (
        ("bibit_sonnet", config.bibit_sonnet_manifest_path, "candidate_id", None),
        ("gutenberg_sonnet", config.gutenberg_resolved_sonnet_manifest_path, "candidate_id", "standard_sonnet_materialized_pending_v7"),
        ("wikisource_sonnet", config.wikisource_resolved_sonnet_manifest_path, "candidate_id", "sonnet_materialized_inactive"),
    ):
        cache: dict[Path, bytes] = {}
        for row in _read_csv(manifest):
            if status and row.get("artifact_status") != status:
                continue
            if not row.get("shard_path"):
                continue
            path = config.repo_root / row["shard_path"]
            payload = cache.setdefault(path, path.read_bytes())
            text = payload[int(row["byte_start"]):int(row["byte_end"])].decode("utf-8")
            result.append(_SonnetReference(f"{prefix}:{row[id_field]}", prefix, "", text))
    return result


def _build_report(
    config: LiberLiberResolvedBuildConfig,
    records: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    sonnets: list[dict[str, Any]],
    attribution: list[dict[str, Any]],
    shards: dict[str, list[dict[str, Any]]],
    verification: dict[str, Any],
    temp: Path,
) -> dict[str, Any]:
    materialized = [row for row in records if row["artifact_status"] == "text_materialized_inactive"]
    poems = [row for row in sonnets if row["artifact_status"] == "sonnet_materialized_inactive"]
    manifests = ("records_manifest.csv", "segments_manifest.csv", "sonnets_manifest.csv", "attribution_manifest.csv")
    return {
        "checkpoint": "5C-resolved-build", "source_count": len(records),
        "materialized_record_count": len(materialized),
        "excluded_source_count": sum(row["final_decision"].startswith("exclude_") for row in records),
        "sonnet_candidate_count": len(sonnets), "materialized_sonnet_count": len(poems),
        "attribution_count": len(attribution),
        "materialized_broader_character_count": sum(int(row["cleaned_character_count"]) for row in materialized),
        "retained_source_span_character_count": sum(int(row["retained_broader_character_count"]) for row in materialized),
        "materialized_sonnet_character_count": sum(int(row["cleaned_character_count"]) for row in poems),
        "materialized_role_counts": dict(sorted(Counter(row["final_broader_role"] for row in materialized).items())),
        "source_decision_counts": dict(sorted(Counter(row["final_decision"] for row in records).items())),
        "sonnet_decision_counts": dict(sorted(Counter(row["candidate_decision"] for row in sonnets).items())),
        "shard_count": sum(len(value) for value in shards.values()), "shards": shards,
        "verification": verification,
        "inputs": {
            "probe_sha256": _sha_file(config.probe_path),
            "probe_review_sha256": _sha_file(config.probe_review_path),
            "archive_inventory_sha256": _sha_file(config.archive_inventory_path),
            "source_rights_sha256": _sha_file(config.source_rights_path),
        },
        "decision_sha256": {
            "sources": _sha_file(config.source_decisions_path),
            "segments": _sha_file(config.segment_decisions_path),
            "sonnets": _sha_file(config.sonnet_decisions_path),
            "sonnet_review": _sha_file(config.sonnet_review_path),
            "canonical_editions": _sha_file(config.canonical_editions_path),
        },
        "manifest_sha256": {name: _sha_file(temp / name) for name in manifests},
        "policy": {
            "text_materialized_inactive": True, "text_activated": False,
            "conditioned_material_excluded": True, "v7_created": False,
            "mixture_assigned": False, "cache_deleted": False, "gpu_work_started": False,
            "spelling_and_punctuation_preserved": True,
            "dialogue_terminal_hyphens_preserved": True,
        },
    }


def _validate_resolution(
    config: LiberLiberResolvedBuildConfig,
    sources: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    sonnets: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    canonical: list[dict[str, Any]],
) -> None:
    if len(sources) != config.expected_record_count or len(_unique(sources, "record_id")) != len(sources):
        raise ValueError("Liber Liber source accounting is not exactly 129 unique records")
    if len(canonical) != 4 or sum(row["canonical_decision"].startswith("select_") for row in canonical) != 1:
        raise ValueError("Vita nuova canonical ledger must select exactly one of four editions")
    if len(reviews) != 2 or any(not row["review_resolution"] for row in reviews):
        raise ValueError("bounded 5C reviews are incomplete")
    if len(sonnets) != 64:
        raise ValueError(f"expected 64 Liber Liber sonnet candidates, found {len(sonnets)}")
    if sum(row["candidate_decision"] == "eligible_standard_sonnet_inactive_pending_checkpoint7" for row in sonnets) != 40:
        raise ValueError("expected exactly 40 unique inactive Liber Liber sonnets")
    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in segments:
        by_record[row["record_id"]].append(row)
    for source in sources:
        rows = sorted(by_record[source["record_id"]], key=lambda row: int(row["character_start"]))
        if sum(int(row["character_count"]) for row in rows) != int(source["source_character_count"]):
            raise ValueError(f"segment accounting mismatch: {source['record_id']}")
        cursor = 0
        for row in rows:
            if int(row["character_start"]) != cursor:
                raise ValueError(f"segment gap or overlap: {source['record_id']}")
            cursor = int(row["character_end"])


def _validate_inputs(config: LiberLiberResolvedBuildConfig) -> None:
    required = (
        config.probe_path, config.probe_review_path, config.probe_report_path,
        config.archive_inventory_path, config.source_rights_path,
        config.bibit_record_manifest_path, config.bibit_sonnet_manifest_path,
        config.gutenberg_previous_probe_path, config.gutenberg_pass_1b_probe_path,
        config.gutenberg_resolved_record_manifest_path, config.gutenberg_resolved_sonnet_manifest_path,
        config.wikisource_resolved_record_manifest_path, config.wikisource_resolved_sonnet_manifest_path,
        config.broader_sources_manifest_path, config.protected_v6_sonnet_manifest_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoint-5C inputs: {missing}")
    report = json.loads(config.probe_report_path.read_text(encoding="utf-8"))
    if _sha_file(config.probe_path) != report["outputs"]["probe_csv_sha256"]:
        raise ValueError("stale Liber Liber probe CSV")
    if _sha_file(config.probe_review_path) != report["outputs"]["review_csv_sha256"]:
        raise ValueError("stale Liber Liber probe review CSV")
    if report["candidate_count"] != config.expected_record_count or report["conditioned_excluded_count"] != config.expected_conditioned_count:
        raise ValueError("checkpoint-5B scope changed")
    if config.max_shard_bytes <= 0 or config.progress_interval <= 0:
        raise ValueError("shard and progress intervals must be positive")


def _probe_config(config: LiberLiberResolvedBuildConfig) -> LiberLiberArchiveProbeConfig:
    return LiberLiberArchiveProbeConfig(
        repo_root=config.repo_root, inventory_path=config.archive_inventory_path,
        cache_dir=config.cache_dir, output_csv_path=config.probe_path,
        review_csv_path=config.probe_review_path, json_report_path=config.probe_report_path,
        markdown_report_path=config.markdown_report_path,
        bibit_record_manifest_path=config.bibit_record_manifest_path,
        bibit_sonnet_manifest_path=config.bibit_sonnet_manifest_path,
        gutenberg_previous_probe_path=config.gutenberg_previous_probe_path,
        gutenberg_previous_cache_dir=config.gutenberg_previous_cache_dir,
        gutenberg_pass_1b_probe_path=config.gutenberg_pass_1b_probe_path,
        gutenberg_pass_1b_cache_dir=config.gutenberg_pass_1b_cache_dir,
        gutenberg_resolved_record_manifest_path=config.gutenberg_resolved_record_manifest_path,
        gutenberg_resolved_sonnet_manifest_path=config.gutenberg_resolved_sonnet_manifest_path,
        wikisource_resolved_record_manifest_path=config.wikisource_resolved_record_manifest_path,
        wikisource_resolved_sonnet_manifest_path=config.wikisource_resolved_sonnet_manifest_path,
        broader_sources_manifest_path=config.broader_sources_manifest_path,
        protected_v6_sonnet_manifest_path=config.protected_v6_sonnet_manifest_path,
        expected_candidate_count=config.expected_record_count,
        expected_conditioned_count=config.expected_conditioned_count,
    )


def _read_source(config: LiberLiberResolvedBuildConfig, row: dict[str, str]) -> str:
    path = config.repo_root / row["cleaned_cache_path"]
    text = path.read_text(encoding="utf-8")
    if _sha(text) != row["cleaned_sha256"] or len(text) != int(row["cleaned_character_count"]):
        raise ValueError(f"cached source mismatch: {row['record_id']}")
    return text


def _compose_segments(text: str, rows: list[dict[str, Any]]) -> tuple[str, list[tuple[dict[str, Any], int, int]]]:
    payload = bytearray()
    parts, ranges = [], []
    for index, row in enumerate(rows):
        start, end = int(row["character_start"]), int(row["character_end"])
        part = text[start:end]
        if _sha(part) != row["segment_sha256"]:
            raise ValueError(f"segment hash mismatch: {row['segment_id']}")
        if index:
            payload.extend(b"\n"); parts.append("\n")
        relative_start = len(payload)
        encoded = part.encode("utf-8"); payload.extend(encoded); parts.append(part)
        ranges.append((row, relative_start, len(payload)))
    result = "".join(parts)
    if result.encode("utf-8") != payload:
        raise AssertionError("segment byte composition changed")
    return result, ranges


def _attribution_row(record_id: str, inventory: dict[str, str], rights: dict[str, str]) -> dict[str, Any]:
    return {
        "record_id": record_id, "title": inventory["title"], "author": inventory["author"],
        "landing_page_url": inventory["landing_page_url"],
        "reference_edition": inventory["reference_edition"], "editor": inventory["editor"],
        "translator": inventory["translator"], "digitization_credit": inventory["digitization_credit"],
        "layout_credit": inventory["layout_credit"], "publication_credit": inventory["publication_credit"],
        "revision_credit": inventory["revision_credit"], "license_label": rights["license_label"],
        "license_url": rights["license_url"], "book_license_terms_url": rights["book_license_terms_url"],
        "required_notice": rights["required_notice"],
        "modification_notice": "Liber Liber wrappers removed; canonical, protected, and isolated-sonnet spans removed where ledgered; retained spelling and punctuation preserved.",
        "downstream_note": rights["downstream_note"],
        "activation_status": "inactive_pending_cross_archive_freeze",
    }


class _ShardWriter:
    def __init__(self, directory: Path, portable: str, maximum: int) -> None:
        self.directory, self.portable, self.maximum = directory, portable, maximum
        self.handle: BinaryIO | None = None
        self.path: Path | None = None
        self.bytes = self.items = 0
        self.hasher = hashlib.sha256()
        self.reports: list[dict[str, Any]] = []

    def add(self, item_id: str, text: str) -> dict[str, Any]:
        payload = _canonical(text).encode("utf-8")
        if not payload.strip() or len(payload) > self.maximum:
            raise ValueError(f"invalid shard item {item_id}: {len(payload):,} bytes")
        separator = 1 if self.items else 0
        if self.handle is None or self.bytes + separator + len(payload) > self.maximum:
            self._finish(); self._start(); separator = 0
        if separator:
            self._write(b"\n")
        start = self.bytes
        self._write(payload); self.items += 1
        return {"shard_path": f"{self.portable}/{self.path.name}", "byte_start": start, "byte_end": self.bytes}

    def close(self) -> list[dict[str, Any]]:
        self._finish(); return self.reports

    def abort(self) -> None:
        if self.handle:
            self.handle.close(); self.handle = None

    def _start(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"part-{len(self.reports)+1:04d}.txt"
        self.handle = self.path.open("wb"); self.bytes = self.items = 0
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


class _PathReference:
    def __init__(self, path: Path) -> None:
        self.path = path

    def read_text(self) -> str:
        return self.path.read_text(encoding="utf-8")


def _validate_artifacts(
    temp: Path, prefix: str, records: list[dict[str, Any]],
    segments: list[dict[str, Any]], sonnets: list[dict[str, Any]],
) -> None:
    cache: dict[Path, bytes] = {}
    for row in [*records, *sonnets]:
        if not row["shard_path"]:
            continue
        relative = Path(row["shard_path"]).relative_to(prefix)
        payload = cache.setdefault(relative, (temp / relative).read_bytes())
        part = payload[int(row["byte_start"]):int(row["byte_end"])]
        if hashlib.sha256(part).hexdigest() != row["cleaned_sha256"]:
            raise ValueError(f"manifest artifact mismatch: {row.get('record_id') or row.get('candidate_id')}")
    for row in segments:
        if row["artifact_status"] != "materialized_in_inactive_record":
            continue
        relative = Path(row["output_shard_path"]).relative_to(prefix)
        payload = cache.setdefault(relative, (temp / relative).read_bytes())
        part = payload[int(row["output_byte_start"]):int(row["output_byte_end"])]
        if hashlib.sha256(part).hexdigest() != row["output_sha256"]:
            raise ValueError(f"segment artifact mismatch: {row['segment_id']}")


def _replace_verified_output(temp: Path, output: Path) -> None:
    backup = output.parent / f".{output.name}.previous"
    if backup.exists():
        raise FileExistsError(f"stale build backup exists: {backup}")
    moved = False
    try:
        if output.exists():
            os.replace(output, backup); moved = True
        os.replace(temp, output)
    except Exception:
        if moved and not output.exists() and backup.exists():
            os.replace(backup, output)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _metric_reference_ids(value: str) -> list[str]:
    return sorted({item.split("|", 1)[0] for item in value.split(";") if item})


def _embedded_reference_ids(value: str) -> list[str]:
    result = []
    for item in value.split(";"):
        if not item:
            continue
        pieces = item.split("|")
        metrics = dict(piece.split("=", 1) for piece in pieces[1:])
        if float(metrics.get("candidate", 1)) < 0.8 and float(metrics.get("reference", 0)) >= 0.8:
            result.append(pieces[0])
    return sorted(set(result))


def _canonical(text: str) -> str:
    return text.rstrip() + "\n" if text.strip() else ""


def _stats(text: str) -> dict[str, Any]:
    payload = text.encode("utf-8")
    return {
        "cleaned_character_count": len(text), "cleaned_byte_count": len(payload),
        "cleaned_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _empty_location() -> dict[str, str]:
    return {"shard_path": "", "byte_start": "", "byte_end": ""}


def _record_number(row: str | dict[str, Any]) -> int:
    value = row if isinstance(row, str) else row["record_id"]
    return int(value.split(":")[-1])


def _record_sort_key(row: dict[str, Any]) -> int:
    return _record_number(row)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _unique(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        if row[key] in result:
            raise ValueError(f"duplicate {key}: {row[key]}")
        result[row[key]] = row
    return result


def _emit(progress: Progress, phase: str, completed: int, total: int, started: float) -> None:
    elapsed = monotonic() - started
    rate = completed / elapsed if elapsed else 0.0
    eta = (total - completed) / rate if rate else 0.0
    progress(
        f"{phase} completed={completed:,}/{total:,} percent={completed/max(1,total):.1%} "
        f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
    )
