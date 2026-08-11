"""Freeze Project Gutenberg extraction, canonicalization, and sonnet evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from time import monotonic
from typing import Any

from .gutenberg import strip_gutenberg_boilerplate
from .gutenberg_fulltext_probe import fingerprint_text


SOURCE_FIELDS = (
    "ebook_id",
    "title",
    "authors",
    "source_pool",
    "source_archive",
    "source_url",
    "period_bucket",
    "input_role",
    "final_role",
    "probe_decision",
    "source_decision",
    "extraction_policy",
    "canonical_reference_ids",
    "language_route",
    "cache_path",
    "cache_sha256",
    "cleaned_sha256",
    "cleaned_character_count",
    "included_record_character_count",
    "excluded_character_count",
    "sonnet_candidate_count",
    "eligible_standard_sonnet_count",
    "unresolved_sonnet_review_count",
    "residual_heldout_overlap_ids",
)

SEGMENT_FIELDS = (
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
)

SONNET_FIELDS = (
    "candidate_id",
    "ebook_id",
    "title",
    "authors",
    "source_kind",
    "stanza_pattern",
    "line_count",
    "character_start",
    "character_end",
    "source_text_sha256",
    "cleaned_text_sha256",
    "first_line",
    "last_line",
    "exact_reference_ids",
    "near_reference_ids",
    "heldout_reference_ids",
    "duplicate_gutenberg_candidate_ids",
    "manual_review_resolution",
    "manual_review_rationale",
    "candidate_decision",
)

REVIEW_FIELDS = (
    "candidate_id",
    "ebook_id",
    "title",
    "source_kind",
    "stanza_pattern",
    "first_line",
    "last_line",
    "source_text_sha256",
    "review_resolution",
    "review_rationale",
)

FULL_DUPLICATE_DECISION = "exclude_cross_corpus_duplicate_candidate"
NONSTANDARD_DECISION = "exclude_standard_italian_core_language_composition"
EMBEDDED_DECISION = "quarantine_embedded_duplicate_segments_before_activation"
HELDOUT_DECISION = "quarantine_heldout_sonnet_segment_before_activation"
LANGUAGE_EXTRACTION_DECISION = "source_specific_language_extraction_before_activation"

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_SONETTO_HEADING = re.compile(r"^\s*sonett(?:o|i)(?:\s+[^\n]{0,80})?[.:]?\s*$", re.I)
_FORM_HEADING = re.compile(
    r"^\s*(?:strambott\w*|capitol\w*|epistol\w*|disperata|canzon\w*|"
    r"madrigal\w*|ballat\w*|od[ea]|inn[oi]|elegi\w*|sestin\w*|"
    r"traged\w*|commedi\w*)[^\n]{0,80}$",
    re.I,
)
_TRAILING_NOTE = re.compile(
    r"(?im)^\s*(?:note? del trascrittore|nota del trascrittore|"
    r"transcriber['’]s notes?)\s*[:.]?\s*$"
)
_ITALIAN_WORDS = {
    "che", "chi", "con", "da", "del", "della", "di", "e", "gli", "il",
    "in", "io", "la", "le", "lo", "ma", "mi", "non", "per", "se", "si",
    "sono", "tu", "un", "una", "voi",
}
_ENGLISH_WORDS = {
    "a", "and", "are", "as", "be", "but", "by", "for", "from", "he", "her",
    "his", "i", "in", "is", "it", "my", "not", "of", "on", "she", "that",
    "the", "their", "this", "to", "was", "we", "what", "who", "with", "you",
}
_REVIEW_RESOLUTIONS = {
    "accept_structurally_verified_standard_sonnet",
    "exclude_nonstandard_language_sonnet",
    "exclude_not_sonnet",
}
_SONNET_PATTERNS = ((4, 4, 3, 3), (8, 6), (4, 4, 6), (14,))


@dataclass(frozen=True)
class GutenbergExtractionAuditConfig:
    """Frozen inputs and public outputs for checkpoint 3A."""

    repo_root: Path
    prior_probe_csv_path: Path
    pass1b_probe_csv_path: Path
    final_resolution_csv_path: Path
    prior_cache_dir: Path
    pass1b_cache_dir: Path
    bibit_record_manifest_path: Path
    broader_sources_manifest_path: Path
    sonnet_manifest_path: Path
    bibit_sonnet_manifest_path: Path
    source_csv_path: Path
    segment_csv_path: Path
    sonnet_csv_path: Path
    review_csv_path: Path
    json_report_path: Path
    markdown_report_path: Path
    expected_prior_count: int = 416
    expected_pass1b_count: int = 167
    expected_conditioned_count: int = 4
    near_sonnet_overlap_threshold: float = 0.72
    near_sonnet_sequence_threshold: float = 0.86
    heldout_containment_threshold: float = 0.8
    progress_interval: int = 25


@dataclass(frozen=True)
class TextSpan:
    start: int
    end: int
    decision: str
    role: str
    reason: str
    reference_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GutenbergSonnetCandidate:
    ebook_id: str
    title: str
    authors: str
    source_kind: str
    stanza_pattern: str
    start: int
    end: int
    text: str
    cleaned_text: str

    @property
    def candidate_id(self) -> str:
        return f"pg{self.ebook_id}:char{self.start}-{self.end}"


@dataclass(frozen=True)
class _Line:
    start: int
    end: int
    raw: str


@dataclass(frozen=True)
class _Block:
    start: int
    end: int
    lines: tuple[_Line, ...]


@dataclass(frozen=True)
class _SonnetReference:
    reference_id: str
    corpus: str
    split: str
    text: str


Progress = Callable[[str], None]


def audit_gutenberg_extraction(
    config: GutenbergExtractionAuditConfig,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Audit all frozen Gutenberg inputs without materializing corpus text."""

    _validate_config(config)
    started = monotonic()
    prior_rows = _read_csv(config.prior_probe_csv_path)
    pass1b_rows = _read_csv(config.pass1b_probe_csv_path)
    conditioned_rows = [
        row
        for row in _read_csv(config.final_resolution_csv_path)
        if row.get("final_activation_class") == "conditioned_probe"
    ]
    _validate_queue(prior_rows, pass1b_rows, conditioned_rows, config)
    reviews = _load_review_resolutions(config.review_csv_path)
    required_cross_references = {
        reference_id
        for row in prior_rows
        for reference_id in _metric_reference_ids(row)
    }
    required_cross_references.update(
        f"sonnet:{reference_id}"
        for row in prior_rows
        for reference_id in _heldout_metric_ids(row)
    )
    cross_references = _load_cross_references(config, required_cross_references)
    sonnet_references = _load_sonnet_references(config)
    reference_index = _build_sonnet_reference_index(sonnet_references)
    heldout_watch, heldout_denominators = _build_heldout_watch(sonnet_references)

    inputs: list[tuple[dict[str, str], str, Path]] = []
    inputs.extend((row, "initial_eligible_pool", config.prior_cache_dir) for row in prior_rows)
    inputs.extend((row, "pass_1b_eligible_pool", config.pass1b_cache_dir) for row in pass1b_rows)
    inputs.extend((row, "conditioned_metadata_pool", config.pass1b_cache_dir) for row in conditioned_rows)
    inputs.sort(key=lambda item: int(item[0]["ebook_id"]))

    source_rows: list[dict[str, Any]] = []
    source_texts: dict[str, str] = {}
    base_spans: dict[str, list[TextSpan]] = {}
    candidates: list[GutenbergSonnetCandidate] = []

    for index, (row, source_pool, cache_dir) in enumerate(inputs, start=1):
        ebook_id = row["ebook_id"]
        cache_path = cache_dir / f"pg{ebook_id}.txt"
        raw = cache_path.read_text(encoding="utf-8")
        cleaned = strip_gutenberg_boilerplate(raw)
        source_texts[ebook_id] = cleaned
        role = _source_role(row)
        decision = row.get("probe_decision", "")
        if source_pool == "conditioned_metadata_pool":
            decision = "conditioned_probe"
        spans, source_decision, extraction_policy, canonical_ids, language_route = (
            _initial_source_spans(
                row,
                source_pool=source_pool,
                text=cleaned,
                role=role,
                cross_references=cross_references,
            )
        )
        base_spans[ebook_id] = spans
        allowed = [(span.start, span.end) for span in spans if span.decision == "include_record_text"]
        if allowed:
            candidates.extend(
                discover_gutenberg_sonnet_candidates(
                    cleaned,
                    ebook_id=ebook_id,
                    title=row["title"],
                    authors=row["authors"],
                    role=role,
                    allowed_ranges=allowed,
                )
            )
        source_rows.append(
            {
                "ebook_id": ebook_id,
                "title": row["title"],
                "authors": row["authors"],
                "source_pool": source_pool,
                "source_archive": "Project Gutenberg",
                "source_url": row.get("landing_page_url", f"https://www.gutenberg.org/ebooks/{ebook_id}"),
                "period_bucket": row.get("final_period_bucket") or row.get("period_bucket", ""),
                "input_role": role,
                "final_role": _record_output_role(role),
                "probe_decision": decision,
                "source_decision": source_decision,
                "extraction_policy": extraction_policy,
                "canonical_reference_ids": ";".join(canonical_ids),
                "language_route": language_route,
                "cache_path": _portable(cache_path, config.repo_root),
                "cache_sha256": _sha256_bytes(raw.encode("utf-8")),
                "cleaned_sha256": _sha256_text(cleaned),
                "cleaned_character_count": len(cleaned),
            }
        )
        if index == 1 or index % config.progress_interval == 0 or index == len(inputs):
            _progress(progress, "source-audit", index, len(inputs), started, f"pg{ebook_id}")

    candidate_rows, review_rows = _resolve_sonnet_candidates(
        candidates,
        reviews=reviews,
        references=sonnet_references,
        reference_index=reference_index,
        near_overlap_threshold=config.near_sonnet_overlap_threshold,
        near_sequence_threshold=config.near_sonnet_sequence_threshold,
    )
    _attach_intra_gutenberg_sonnet_duplicates(
        candidate_rows,
        candidates,
        near_overlap_threshold=config.near_sonnet_overlap_threshold,
        near_sequence_threshold=config.near_sonnet_sequence_threshold,
    )

    candidates_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    for row in candidate_rows:
        candidates_by_source[row["ebook_id"]].append(row)

    segment_rows: list[dict[str, Any]] = []
    sources_by_id = {row["ebook_id"]: row for row in source_rows}
    for ebook_id, source in sorted(sources_by_id.items(), key=lambda item: int(item[0])):
        text = source_texts[ebook_id]
        removal_spans = [span for span in base_spans[ebook_id] if span.decision != "include_record_text"]
        for candidate_row in candidates_by_source.get(ebook_id, []):
            candidate = candidate_by_id[candidate_row["candidate_id"]]
            candidate_decision = candidate_row["candidate_decision"]
            if candidate_decision == "exclude_manual_not_sonnet":
                continue
            segment_decision = "quarantine_sonnet_candidate"
            segment_role = "standard_sonnets"
            segment_reason = "confirmed_or_unresolved_sonnet_units_are_quarantined_from_broader_text"
            if candidate_decision == "conditioned_sonnet_candidate_not_activated":
                segment_decision = "quarantine_conditioned_sonnet_candidate"
                segment_role = "conditioned_language_variant"
                segment_reason = "nonstandard_language_sonnet_not_authorized_for_activation"
            removal_spans.append(
                TextSpan(
                    start=candidate.start,
                    end=candidate.end,
                    decision=segment_decision,
                    role=segment_role,
                    reason=segment_reason,
                    reference_ids=tuple(
                        value
                        for field in (
                            "exact_reference_ids",
                            "near_reference_ids",
                            "heldout_reference_ids",
                            "duplicate_gutenberg_candidate_ids",
                        )
                        for value in candidate_row[field].split(";")
                        if value
                    ),
                )
            )
        partition = _partition_spans(
            text,
            removal_spans,
            default_role=source["final_role"],
            default_decision="include_record_text",
        )
        included_text = "\n".join(
            text[span.start : span.end]
            for span in partition
            if span.decision == "include_record_text"
        )
        _, heldout_hits = fingerprint_text(included_text, watched_shingles=heldout_watch)
        residual_ids = _heldout_ids_above_threshold(
            heldout_hits,
            heldout_denominators,
            config.heldout_containment_threshold,
        )
        source["included_record_character_count"] = sum(
            span.end - span.start for span in partition if span.decision == "include_record_text"
        )
        source["excluded_character_count"] = len(text) - source["included_record_character_count"]
        source["sonnet_candidate_count"] = len(candidates_by_source.get(ebook_id, []))
        source["eligible_standard_sonnet_count"] = sum(
            row["candidate_decision"] == "eligible_standard_sonnet_pending_processed_build"
            for row in candidates_by_source.get(ebook_id, [])
        )
        source["unresolved_sonnet_review_count"] = sum(
            row["candidate_decision"] == "review_structural_sonnet_candidate"
            for row in candidates_by_source.get(ebook_id, [])
        )
        source["residual_heldout_overlap_ids"] = ";".join(residual_ids)
        segment_rows.extend(_segment_rows(ebook_id, text, partition))

    _validate_results(source_rows, segment_rows, candidate_rows, config)
    _write_csv(config.source_csv_path, SOURCE_FIELDS, source_rows)
    _write_csv(config.segment_csv_path, SEGMENT_FIELDS, segment_rows)
    _write_csv(config.sonnet_csv_path, SONNET_FIELDS, candidate_rows)
    _write_csv(config.review_csv_path, REVIEW_FIELDS, review_rows)
    report = _build_report(config, source_rows, segment_rows, candidate_rows, review_rows)
    _write_json(config.json_report_path, report)
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(render_extraction_audit_markdown(report), encoding="utf-8")
    return report


def locate_reference_segments(
    candidate_text: str,
    reference_text: str,
    *,
    minimum_anchor_count: int = 3,
) -> list[tuple[int, int, int]]:
    """Locate a reference inside a larger edition using unique ordered 8-word anchors."""

    candidate_words = _word_locations(candidate_text)
    reference_words = _word_locations(reference_text)
    if len(candidate_words) < 8 or len(reference_words) < 8:
        return []
    candidate_anchors = _unique_shingle_positions(candidate_words)
    reference_anchors = _unique_shingle_positions(reference_words)
    pairs = sorted(
        (reference_index, candidate_anchors[shingle])
        for shingle, reference_index in reference_anchors.items()
        if shingle in candidate_anchors
    )
    chain = _longest_increasing_pairs(pairs)
    if len(chain) < minimum_anchor_count:
        return []
    first_reference, first_candidate = chain[0]
    last_reference, last_candidate = chain[-1]
    start = candidate_words[first_candidate][1]
    end = candidate_words[last_candidate + 7][2]
    start = _line_start(candidate_text, start)
    end = _line_end(candidate_text, end)
    reference_coverage = len(chain) / max(1, len(reference_anchors))
    if reference_coverage < 0.05 and len(chain) < 20:
        return []
    return [(start, end, len(chain))]


def discover_gutenberg_sonnet_candidates(
    text: str,
    *,
    ebook_id: str,
    title: str,
    authors: str,
    role: str,
    allowed_ranges: Iterable[tuple[int, int]],
) -> list[GutenbergSonnetCandidate]:
    """Find explicit or structurally plausible 14-line sonnets in allowed text."""

    found: list[GutenbergSonnetCandidate] = []
    seen: set[tuple[int, int]] = set()
    for range_start, range_end in allowed_ranges:
        blocks = _blocks(text, range_start, range_end)
        section_active = False
        index = 0
        while index < len(blocks):
            block = blocks[index]
            heading = _heading_text(block)
            explicit_single = bool(heading and _SONETTO_HEADING.fullmatch(heading))
            if explicit_single and "sonetti" in heading.casefold():
                section_active = True
                index += 1
                continue
            if heading and _FORM_HEADING.fullmatch(heading):
                section_active = False
                index += 1
                continue
            previous_heading = _heading_text(blocks[index - 1]) if index else ""
            preceding_explicit = bool(
                previous_heading and _SONETTO_HEADING.fullmatch(previous_heading)
            )
            match = _match_sonnet_blocks(blocks, index)
            if match is None:
                index += 1
                continue
            count, stanza_pattern, lines = match
            source_kind = ""
            if preceding_explicit:
                source_kind = "explicit_sonetto_heading"
            elif section_active:
                source_kind = "explicit_sonetti_section"
            elif role == "sonnet_specialization":
                source_kind = "source_metadata_sonnet_collection"
            elif role == "historical_non_sonnet_poetry":
                source_kind = "structural_14_line"
            if not source_kind or not _verse_like(
                lines,
                require_indentation=source_kind == "structural_14_line",
            ):
                index += 1
                continue
            start = lines[0].start
            end = lines[-1].end
            if (start, end) in seen:
                index += count
                continue
            seen.add((start, end))
            raw = text[start:end]
            cleaned = "\n".join(line.raw.strip() for line in lines).strip() + "\n"
            found.append(
                GutenbergSonnetCandidate(
                    ebook_id=ebook_id,
                    title=title,
                    authors=authors,
                    source_kind=source_kind,
                    stanza_pattern=stanza_pattern,
                    start=start,
                    end=end,
                    text=raw,
                    cleaned_text=cleaned,
                )
            )
            index += count
    return found


def render_extraction_audit_markdown(report: dict[str, Any]) -> str:
    """Render the public checkpoint-3A report."""

    lines = [
        "# Project Gutenberg Extraction And Canonicalization Audit",
        "",
        "## Result",
        "",
        f"Accounted for {report['source_count']:,} cached Project Gutenberg records without materializing processed corpus text.",
        "",
        "## Source Decisions",
        "",
        "| Decision | Records |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{key}` | {value:,} |"
        for key, value in report["source_decision_counts"].items()
    )
    lines.extend(
        [
            "",
            "## Measured Roles",
            "",
            "| Role | Records | Retained characters |",
            "| --- | ---: | ---: |",
        ]
    )
    for role, values in report["role_summary"].items():
        lines.append(f"| `{role}` | {values['record_count']:,} | {values['included_record_characters']:,} |")
    lines.extend(
        [
            "",
            "## Sonnets",
            "",
            f"- Audited candidates: {report['sonnet_candidate_count']:,}.",
            f"- Eligible standard candidates pending the processed build: {report['eligible_standard_sonnet_count']:,}.",
            f"- Unresolved structural reviews: {report['unresolved_sonnet_review_count']:,}.",
            f"- Held-out conflicts excluded: {report['heldout_sonnet_conflict_count']:,}.",
            "",
            "### Candidate Decisions",
            "",
            "| Decision | Candidates |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        f"| `{key}` | {value:,} |"
        for key, value in report["sonnet_decision_counts"].items()
    )
    lines.extend(
        [
            "",
            "### Structural Review",
            "",
            "| Resolution | Candidates |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        f"| `{key}` | {value:,} |"
        for key, value in report["manual_review_resolution_counts"].items()
    )
    lines.extend(
        [
            "",
            "False-positive fourteen-line windows remain in their broader-text role; only confirmed or unresolved sonnet units are quarantined from broader stages.",
            "",
            "## Boundaries",
            "",
            "- The fifteen fully covered Gutenberg editions remain excluded in favor of their existing canonical references.",
            "- Unique material is retained from the six partial-overlap sources.",
            "- The Cino validation sonnet is quarantined from eBook 35321.",
            "- Six conditioned source records and two embedded non-standard-language sonnets remain outside the standard corpus.",
            "- No processed text, V7 split, training-mixture weight, or GPU job is created by this audit.",
            "",
            "## Reproduction",
            "",
            "Run `python3 scripts/audit_project_gutenberg_extraction.py` with both preserved local Gutenberg caches. The JSON report pins every public input and tabular output with SHA-256.",
        ]
    )
    return "\n".join(lines) + "\n"


def _initial_source_spans(
    row: dict[str, str],
    *,
    source_pool: str,
    text: str,
    role: str,
    cross_references: dict[str, str],
) -> tuple[list[TextSpan], str, str, tuple[str, ...], str]:
    decision = row.get("probe_decision", "")
    if source_pool == "conditioned_metadata_pool" or decision == NONSTANDARD_DECISION:
        return (
            [TextSpan(0, len(text), "conditioned_not_activated", "conditioned_language_variant", "conditioned_language_experiment_not_authorized")],
            "conditioned_candidate_not_activated",
            "no_standard_core_extraction",
            (),
            "conditioned_separate",
        )
    references = tuple(_metric_reference_ids(row))
    if decision == FULL_DUPLICATE_DECISION:
        return (
            [TextSpan(0, len(text), "exclude_full_source_duplicate", "excluded", "canonical_text_already_exists", references)],
            "exclude_canonical_cross_corpus_duplicate",
            "exclude_complete_source",
            references,
            "standard_italian",
        )
    if row["ebook_id"] == "17440":
        spans = _extract_amadigi_italian_spans(text, role)
        return spans, "eligible_after_source_specific_extraction_pending_build", "extract_italian_parallel_blocks", (), "standard_italian_extracted"
    if row["ebook_id"] == "17834":
        spans = _extract_zaffetta_first_edition(text, role)
        return spans, "eligible_after_source_specific_extraction_pending_build", "extract_first_italian_primary_edition", (), "standard_italian_extracted"

    removals: list[TextSpan] = []
    if decision == EMBEDDED_DECISION:
        for reference_id in references:
            reference = cross_references.get(reference_id)
            if not reference:
                raise ValueError(f"missing embedded reference {reference_id} for pg{row['ebook_id']}")
            matches = locate_reference_segments(text, reference)
            if not matches:
                raise ValueError(f"could not locate embedded reference {reference_id} in pg{row['ebook_id']}")
            for start, end, _ in matches:
                removals.append(TextSpan(start, end, "exclude_embedded_canonical_text", "excluded", "embedded_text_exists_in_canonical_reference", (reference_id,)))
    if decision == HELDOUT_DECISION:
        heldout_ids = tuple(_heldout_metric_ids(row))
        for reference_id in heldout_ids:
            reference = cross_references.get(f"sonnet:{reference_id}")
            if not reference:
                raise ValueError(f"missing held-out sonnet reference {reference_id}")
            matches = locate_reference_segments(text, reference)
            if not matches:
                raise ValueError(f"could not locate held-out sonnet {reference_id} in pg{row['ebook_id']}")
            for start, end, _ in matches:
                removals.append(TextSpan(start, end, "exclude_protected_v6_sonnet", "excluded", "protected_v6_validation_or_test_identity", (reference_id,)))
    trailing = _trailing_transcriber_note(text)
    if trailing:
        removals.append(TextSpan(trailing, len(text), "exclude_terminal_transcriber_note", "excluded", "modern_terminal_transcriber_note"))
    spans = _partition_spans(text, removals, default_role=_record_output_role(role), default_decision="include_record_text")
    policy = "preserve_boilerplate_stripped_primary_text"
    if decision == EMBEDDED_DECISION:
        policy = "remove_embedded_canonical_segments_preserve_unique_text"
    elif decision == HELDOUT_DECISION:
        policy = "remove_protected_v6_sonnet_preserve_source_text"
    return spans, "eligible_standard_core_pending_processed_build", policy, references if removals else (), "standard_italian"


def _extract_amadigi_italian_spans(text: str, role: str) -> list[TextSpan]:
    start_match = re.search(r"(?m)^Personaggi\.\s*$", text)
    end_match = re.search(r"(?m)^Italian:\s*$", text)
    if not start_match or not end_match or end_match.start() <= start_match.start():
        raise ValueError("Amadigi bilingual boundaries changed")
    removals = [
        TextSpan(0, start_match.start(), "exclude_non_primary_or_english_text", "excluded", "english_dedication_and_front_matter"),
        TextSpan(end_match.start(), len(text), "exclude_non_primary_or_english_text", "excluded", "transcriber_notes_and_gutenberg_tail"),
    ]
    for block in _blocks(text, start_match.start(), end_match.start()):
        payload = "\n".join(line.raw for line in block.lines)
        if _block_is_english(payload):
            removals.append(TextSpan(block.start, block.end, "exclude_parallel_english_translation", "excluded", "parallel_english_block"))
    return _partition_spans(text, removals, default_role=_record_output_role(role), default_decision="include_record_text")


def _extract_zaffetta_first_edition(text: str, role: str) -> list[TextSpan]:
    start_match = re.search(r"(?m)^Poi ch'ogni bestia in volgar e in latino,\s*$", text)
    if not start_match:
        raise ValueError("Zaffetta first-edition start changed")
    end_match = re.search(r"(?m)^IL FINE\.\s*$", text[start_match.start() :])
    if not end_match:
        raise ValueError("Zaffetta first-edition end changed")
    end = start_match.start() + end_match.start()
    removals = [
        TextSpan(0, start_match.start(), "exclude_french_notice_and_front_matter", "excluded", "French_bibliographic_notice"),
        TextSpan(end, len(text), "exclude_duplicate_modernized_edition", "excluded", "retain_first_Italian_primary_edition_only"),
    ]
    return _partition_spans(text, removals, default_role=_record_output_role(role), default_decision="include_record_text")


def _resolve_sonnet_candidates(
    candidates: list[GutenbergSonnetCandidate],
    *,
    reviews: dict[str, dict[str, str]],
    references: list[_SonnetReference],
    reference_index: dict[str, Any],
    near_overlap_threshold: float,
    near_sequence_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    reference_by_id = {reference.reference_id: reference for reference in references}
    for candidate in candidates:
        exact_ids, near_ids = _candidate_reference_matches(
            candidate.cleaned_text,
            references=references,
            reference_by_id=reference_by_id,
            index=reference_index,
            near_overlap_threshold=near_overlap_threshold,
            near_sequence_threshold=near_sequence_threshold,
        )
        heldout_ids = sorted(
            reference_id
            for reference_id in set(exact_ids) | set(near_ids)
            if reference_by_id[reference_id].split in {"validation", "test"}
        )
        review = reviews.get(candidate.candidate_id, {})
        if review and review.get("source_text_sha256") != _sha256_text(candidate.text):
            raise ValueError(f"stale sonnet review evidence: {candidate.candidate_id}")
        resolution = review.get("review_resolution", "")
        rationale = review.get("review_rationale", "")
        if heldout_ids:
            decision = "exclude_protected_v6_sonnet"
        elif exact_ids or near_ids:
            decision = "exclude_existing_corpus_sonnet_duplicate"
        elif candidate.source_kind == "structural_14_line" and not resolution:
            decision = "review_structural_sonnet_candidate"
        elif resolution == "exclude_not_sonnet":
            decision = "exclude_manual_not_sonnet"
        elif resolution == "exclude_nonstandard_language_sonnet":
            decision = "conditioned_sonnet_candidate_not_activated"
        else:
            decision = "eligible_standard_sonnet_pending_processed_build"
        lines = candidate.cleaned_text.strip().splitlines()
        row = {
            "candidate_id": candidate.candidate_id,
            "ebook_id": candidate.ebook_id,
            "title": candidate.title,
            "authors": candidate.authors,
            "source_kind": candidate.source_kind,
            "stanza_pattern": candidate.stanza_pattern,
            "line_count": len(lines),
            "character_start": candidate.start,
            "character_end": candidate.end,
            "source_text_sha256": _sha256_text(candidate.text),
            "cleaned_text_sha256": _sha256_text(candidate.cleaned_text),
            "first_line": lines[0],
            "last_line": lines[-1],
            "exact_reference_ids": ";".join(sorted(exact_ids)),
            "near_reference_ids": ";".join(sorted(near_ids)),
            "heldout_reference_ids": ";".join(heldout_ids),
            "duplicate_gutenberg_candidate_ids": "",
            "manual_review_resolution": resolution,
            "manual_review_rationale": rationale,
            "candidate_decision": decision,
        }
        rows.append(row)
        if candidate.source_kind == "structural_14_line":
            review_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "ebook_id": candidate.ebook_id,
                    "title": candidate.title,
                    "source_kind": candidate.source_kind,
                    "stanza_pattern": candidate.stanza_pattern,
                    "first_line": lines[0],
                    "last_line": lines[-1],
                    "source_text_sha256": row["source_text_sha256"],
                    "review_resolution": resolution,
                    "review_rationale": rationale,
                }
            )
    return rows, review_rows


def _attach_intra_gutenberg_sonnet_duplicates(
    rows: list[dict[str, Any]],
    candidates: list[GutenbergSonnetCandidate],
    *,
    near_overlap_threshold: float,
    near_sequence_threshold: float,
) -> None:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    inverted: dict[str, set[str]] = defaultdict(set)
    grams: dict[str, set[str]] = {}
    normalized: dict[str, str] = {}
    duplicates: dict[str, set[str]] = defaultdict(set)
    for row in sorted(rows, key=lambda item: (int(item["ebook_id"]), int(item["character_start"]))):
        candidate_id = row["candidate_id"]
        loose = _normalize_loose(by_id[candidate_id].cleaned_text)
        current_grams = _word_ngrams(loose)
        possible = Counter(other for gram in current_grams for other in inverted.get(gram, ()))
        for other_id, shared in possible.most_common(20):
            containment = shared / max(1, min(len(current_grams), len(grams[other_id])))
            if containment < near_overlap_threshold:
                continue
            ratio = SequenceMatcher(None, loose, normalized[other_id], autojunk=False).ratio()
            if ratio >= near_sequence_threshold:
                duplicates[candidate_id].add(other_id)
                duplicates[other_id].add(candidate_id)
        normalized[candidate_id] = loose
        grams[candidate_id] = current_grams
        for gram in current_grams:
            inverted[gram].add(candidate_id)
    row_by_id = {row["candidate_id"]: row for row in rows}
    for candidate_id, others in duplicates.items():
        row_by_id[candidate_id]["duplicate_gutenberg_candidate_ids"] = ";".join(sorted(others))
    for candidate_id, others in duplicates.items():
        canonical = min({candidate_id, *others}, key=_candidate_sort_key)
        if candidate_id != canonical and row_by_id[candidate_id]["candidate_decision"] == "eligible_standard_sonnet_pending_processed_build":
            row_by_id[candidate_id]["candidate_decision"] = "exclude_intra_gutenberg_sonnet_duplicate"


def _load_cross_references(
    config: GutenbergExtractionAuditConfig,
    required_ids: set[str],
) -> dict[str, str]:
    references: dict[str, str] = {}
    for row in _read_csv(config.bibit_record_manifest_path):
        reference_id = f"bibit:{row['object_id']}"
        if reference_id not in required_ids:
            continue
        if row.get("artifact_status") != "text_materialized" or not row.get("shard_path"):
            continue
        path = config.repo_root / row["shard_path"]
        with path.open("rb") as handle:
            handle.seek(int(row["byte_start"]))
            payload = handle.read(int(row["byte_end"]) - int(row["byte_start"]))
        references[reference_id] = payload.decode("utf-8")
    for row in _read_csv(config.broader_sources_manifest_path):
        reference_id = f"current:{row['source_id']}"
        if reference_id not in required_ids:
            continue
        path_value = row.get("expected_clean_text_path", "")
        if path_value and (config.repo_root / path_value).is_file():
            references[reference_id] = (config.repo_root / path_value).read_text(encoding="utf-8")
    for row in _read_csv(config.sonnet_manifest_path):
        reference_id = f"sonnet:{row['poem_id']}"
        if reference_id not in required_ids:
            continue
        if row.get("split_expanded_with_petrarch") not in {"validation", "test"}:
            continue
        path = config.repo_root / row["clean_text_path"]
        references[reference_id] = path.read_text(encoding="utf-8")
    missing = required_ids - set(references)
    if missing:
        raise ValueError(f"missing cross-corpus references: {sorted(missing)}")
    return references


def _load_sonnet_references(config: GutenbergExtractionAuditConfig) -> list[_SonnetReference]:
    references: list[_SonnetReference] = []
    for row in _read_csv(config.sonnet_manifest_path):
        path = config.repo_root / row["clean_text_path"]
        references.append(_SonnetReference(f"v6:{row['poem_id']}", "sonnets_v6", row["split_expanded_with_petrarch"], path.read_text(encoding="utf-8")))
    shard_cache: dict[Path, bytes] = {}
    for row in _read_csv(config.bibit_sonnet_manifest_path):
        path = config.repo_root / row["shard_path"]
        payload = shard_cache.setdefault(path, path.read_bytes())
        text = payload[int(row["byte_start"]) : int(row["byte_end"])].decode("utf-8")
        references.append(_SonnetReference(f"bibit_sonnet:{row['candidate_id']}", "bibit", "", text))
    return references


def _build_sonnet_reference_index(references: list[_SonnetReference]) -> dict[str, Any]:
    exact: dict[str, list[str]] = defaultdict(list)
    grams: dict[str, set[str]] = {}
    inverted: dict[str, set[str]] = defaultdict(set)
    normalized: dict[str, str] = {}
    for reference in references:
        exact[_normalize_exact(reference.text)].append(reference.reference_id)
        loose = _normalize_loose(reference.text)
        normalized[reference.reference_id] = loose
        ref_grams = _word_ngrams(loose)
        grams[reference.reference_id] = ref_grams
        for gram in ref_grams:
            inverted[gram].add(reference.reference_id)
    return {"exact": exact, "grams": grams, "inverted": inverted, "normalized": normalized}


def _candidate_reference_matches(
    text: str,
    *,
    references: list[_SonnetReference],
    reference_by_id: dict[str, _SonnetReference],
    index: dict[str, Any],
    near_overlap_threshold: float,
    near_sequence_threshold: float,
) -> tuple[list[str], list[str]]:
    exact = sorted(index["exact"].get(_normalize_exact(text), ()))
    if exact:
        return exact, []
    loose = _normalize_loose(text)
    candidate_grams = _word_ngrams(loose)
    possible = Counter(reference_id for gram in candidate_grams for reference_id in index["inverted"].get(gram, ()))
    near: list[str] = []
    for reference_id, shared in possible.most_common(20):
        reference_grams = index["grams"][reference_id]
        containment = shared / max(1, min(len(candidate_grams), len(reference_grams)))
        if containment < near_overlap_threshold:
            continue
        ratio = SequenceMatcher(None, loose, index["normalized"][reference_id], autojunk=False).ratio()
        if ratio >= near_sequence_threshold:
            near.append(reference_id)
    return [], sorted(near)


def _build_heldout_watch(references: list[_SonnetReference]) -> tuple[dict[int, tuple[str, ...]], dict[str, int]]:
    watch: dict[int, list[str]] = defaultdict(list)
    denominators: dict[str, int] = {}
    for reference in references:
        if reference.corpus != "sonnets_v6" or reference.split not in {"validation", "test"}:
            continue
        hashes = _stable_8gram_hashes(reference.text)
        denominators[reference.reference_id] = len(hashes)
        for value in hashes:
            watch[value].append(reference.reference_id)
    return {value: tuple(ids) for value, ids in watch.items()}, denominators


def _stable_8gram_hashes(text: str) -> set[int]:
    # Keep this aligned with gutenberg_fulltext_probe's stable rolling hash.
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    words = _WORD.findall(without_marks)
    if len(words) < 8:
        return set()
    mask = (1 << 64) - 1
    base = 1_000_003
    values = []
    for word in words:
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
        values.append(int.from_bytes(digest, "big"))
    factor = pow(base, 7, 1 << 64)
    current = 0
    for value in values[:8]:
        current = ((current * base) + value) & mask
    result = {current}
    for index in range(8, len(values)):
        current = (current - ((values[index - 8] * factor) & mask)) & mask
        current = ((current * base) + values[index]) & mask
        result.add(current)
    return result


def _heldout_ids_above_threshold(hits: dict[str, set[int]], denominators: dict[str, int], threshold: float) -> list[str]:
    return sorted(reference_id for reference_id, values in hits.items() if len(values) / denominators[reference_id] >= threshold)


def _partition_spans(text: str, removals: list[TextSpan], *, default_role: str, default_decision: str) -> list[TextSpan]:
    normalized = _merge_removal_spans(removals)
    result: list[TextSpan] = []
    cursor = 0
    for span in normalized:
        if span.start > cursor:
            result.append(TextSpan(cursor, span.start, default_decision, default_role, "retained_primary_text"))
        result.append(span)
        cursor = span.end
    if cursor < len(text):
        result.append(TextSpan(cursor, len(text), default_decision, default_role, "retained_primary_text"))
    if not result and text:
        result.append(TextSpan(0, len(text), default_decision, default_role, "retained_primary_text"))
    return [span for span in result if span.end > span.start]


def _merge_removal_spans(spans: list[TextSpan]) -> list[TextSpan]:
    spans = sorted((span for span in spans if span.end > span.start), key=lambda span: (span.start, span.end))
    merged: list[TextSpan] = []
    for span in spans:
        if not merged or span.start >= merged[-1].end:
            merged.append(span)
            continue
        prior = merged[-1]
        merged[-1] = TextSpan(
            prior.start,
            max(prior.end, span.end),
            prior.decision if prior.decision == span.decision else "exclude_overlapping_audited_segments",
            "excluded",
            ";".join(sorted({prior.reason, span.reason})),
            tuple(sorted(set(prior.reference_ids) | set(span.reference_ids))),
        )
    return merged


def _segment_rows(ebook_id: str, text: str, spans: list[TextSpan]) -> list[dict[str, Any]]:
    source_hash = _sha256_text(text)
    rows = []
    for index, span in enumerate(spans, start=1):
        payload = text[span.start : span.end]
        rows.append(
            {
                "segment_id": f"pg{ebook_id}:seg{index:04d}",
                "ebook_id": ebook_id,
                "source_cleaned_sha256": source_hash,
                "character_start": span.start,
                "character_end": span.end,
                "character_count": len(payload),
                "segment_sha256": _sha256_text(payload),
                "segment_decision": span.decision,
                "final_role": span.role,
                "reason": span.reason,
                "reference_ids": ";".join(span.reference_ids),
                "start_anchor": " ".join(payload[:120].split()),
                "end_anchor": " ".join(payload[-120:].split()),
            }
        )
    return rows


def _blocks(text: str, start: int, end: int) -> list[_Block]:
    blocks: list[_Block] = []
    lines: list[_Line] = []
    position = start
    for raw in text[start:end].splitlines(keepends=True):
        line_end = position + len(raw)
        if raw.strip():
            lines.append(_Line(position, line_end, raw.rstrip("\r\n")))
        elif lines:
            blocks.append(_Block(lines[0].start, lines[-1].end, tuple(lines)))
            lines = []
        position = line_end
    if lines:
        blocks.append(_Block(lines[0].start, lines[-1].end, tuple(lines)))
    return blocks


def _match_sonnet_blocks(blocks: list[_Block], index: int) -> tuple[int, str, tuple[_Line, ...]] | None:
    for pattern in _SONNET_PATTERNS:
        selected = blocks[index : index + len(pattern)]
        if len(selected) != len(pattern) or tuple(len(block.lines) for block in selected) != pattern:
            continue
        lines = tuple(line for block in selected for line in block.lines)
        return len(pattern), "-".join(map(str, pattern)), lines
    return None


def _verse_like(
    lines: tuple[_Line, ...],
    *,
    require_indentation: bool = False,
) -> bool:
    if len(lines) != 14:
        return False
    stripped = [line.raw.strip() for line in lines]
    if any(not value or len(value) > 150 for value in stripped):
        return False
    word_counts = [len(_WORD.findall(value)) for value in stripped]
    if sum(2 <= count <= 20 for count in word_counts) < 12:
        return False
    indented = sum(bool(line.raw[:1].isspace()) for line in lines)
    if require_indentation:
        return indented >= 8
    return indented >= 8 or sum(len(value) <= 100 for value in stripped) >= 12


def _heading_text(block: _Block) -> str:
    if len(block.lines) > 3:
        return ""
    text = " ".join(line.raw.strip() for line in block.lines)
    return text if len(text) <= 120 else ""


def _block_is_english(text: str) -> bool:
    words = [word.casefold() for word in _WORD.findall(text)]
    italian = sum(word in _ITALIAN_WORDS for word in words)
    english = sum(word in _ENGLISH_WORDS for word in words)
    if re.search(r"\b(?:ACT|SCENE|alone|Enter|Exit|sung by|The Palace|The Garden)\b", text, re.I):
        english += 3
    if re.search(r"\b(?:ATTO|SCENA|solo|sola|entra|esce|palazzo|giardino)\b", text, re.I):
        italian += 3
    return english > italian and english >= 2


def _trailing_transcriber_note(text: str) -> int | None:
    matches = list(_TRAILING_NOTE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    return match.start() if match.start() >= int(len(text) * 0.7) else None


def _word_locations(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0).casefold(), match.start(), match.end()) for match in _WORD.finditer(text)]


def _unique_shingle_positions(words: list[tuple[str, int, int]]) -> dict[tuple[str, ...], int]:
    positions: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index in range(len(words) - 7):
        positions[tuple(word for word, _, _ in words[index : index + 8])].append(index)
    return {shingle: values[0] for shingle, values in positions.items() if len(values) == 1}


def _longest_increasing_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not pairs:
        return []
    tails: list[int] = []
    tail_pair_indices: list[int] = []
    previous = [-1] * len(pairs)
    for pair_index, (_, candidate_index) in enumerate(pairs):
        position = bisect_left(tails, candidate_index)
        if position == len(tails):
            tails.append(candidate_index)
            tail_pair_indices.append(pair_index)
        else:
            tails[position] = candidate_index
            tail_pair_indices[position] = pair_index
        if position:
            previous[pair_index] = tail_pair_indices[position - 1]
    chain_indices = []
    current = tail_pair_indices[-1]
    while current >= 0:
        chain_indices.append(current)
        current = previous[current]
    return [pairs[index] for index in reversed(chain_indices)]


def _line_start(text: str, position: int) -> int:
    boundary = text.rfind("\n", 0, position)
    return boundary + 1


def _line_end(text: str, position: int) -> int:
    boundary = text.find("\n", position)
    return len(text) if boundary < 0 else boundary + 1


def _metric_reference_ids(row: dict[str, str]) -> list[str]:
    ids = []
    for field in ("bibit_overlap_metrics", "current_corpus_overlap_metrics"):
        for metric in row.get(field, "").split(";"):
            reference_id = metric.split("|", maxsplit=1)[0]
            if reference_id:
                ids.append(reference_id)
    return sorted(set(ids))


def _heldout_metric_ids(row: dict[str, str]) -> list[str]:
    return sorted(metric.split("|", maxsplit=1)[0] for metric in row.get("heldout_sonnet_overlap_metrics", "").split(";") if metric)


def _source_role(row: dict[str, str]) -> str:
    return (row.get("final_role") or row.get("preliminary_role", "")).removesuffix("_candidate")


def _record_output_role(role: str) -> str:
    return "historical_non_sonnet_poetry" if role == "sonnet_specialization" else role


def _normalize_exact(text: str) -> str:
    return "".join(character for character in unicodedata.normalize("NFKC", text).casefold() if character.isalnum())


def _normalize_loose(text: str) -> str:
    return " ".join(word.casefold() for word in _WORD.findall(unicodedata.normalize("NFKC", text)))


def _word_ngrams(normalized: str, size: int = 3) -> set[str]:
    words = normalized.split()
    return {" ".join(words[index : index + size]) for index in range(len(words) - size + 1)}


def _candidate_sort_key(candidate_id: str) -> tuple[int, int]:
    match = re.fullmatch(r"pg(\d+):char(\d+)-(\d+)", candidate_id)
    if not match:
        return (10**12, 10**12)
    return int(match.group(1)), int(match.group(2))


def _load_review_resolutions(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    rows = _read_csv(path)
    result = {}
    for row in rows:
        resolution = row.get("review_resolution", "")
        if resolution and resolution not in _REVIEW_RESOLUTIONS:
            raise ValueError(f"unsupported sonnet review resolution: {resolution}")
        result[row["candidate_id"]] = row
    return result


def _validate_queue(prior: list[dict[str, str]], pass1b: list[dict[str, str]], conditioned: list[dict[str, str]], config: GutenbergExtractionAuditConfig) -> None:
    if len(prior) != config.expected_prior_count:
        raise ValueError(f"expected {config.expected_prior_count} prior records, found {len(prior)}")
    if len(pass1b) != config.expected_pass1b_count:
        raise ValueError(f"expected {config.expected_pass1b_count} pass-1B records, found {len(pass1b)}")
    if len(conditioned) != config.expected_conditioned_count:
        raise ValueError(f"expected {config.expected_conditioned_count} conditioned records, found {len(conditioned)}")
    ids = [row["ebook_id"] for row in prior + pass1b + conditioned]
    if len(ids) != len(set(ids)):
        raise ValueError("Gutenberg extraction inputs contain duplicate ebook IDs")
    if any(row.get("resolution_pass") != "pass_1b" or row.get("final_activation_class") != "eligible_probe" for row in pass1b):
        raise ValueError("pass-1B probe input is not the frozen eligible queue")


def _validate_results(source_rows: list[dict[str, Any]], segment_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], config: GutenbergExtractionAuditConfig) -> None:
    expected = config.expected_prior_count + config.expected_pass1b_count + config.expected_conditioned_count
    if len(source_rows) != expected:
        raise ValueError(f"expected {expected} source decisions, found {len(source_rows)}")
    if any(row["residual_heldout_overlap_ids"] for row in source_rows):
        raise ValueError("protected V6 sonnet overlap remains in retained record text")
    if any(int(row["line_count"]) != 14 for row in candidate_rows):
        raise ValueError("non-14-line poem entered the Gutenberg sonnet ledger")
    segments_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in segment_rows:
        segments_by_source[row["ebook_id"]].append(row)
    for source in source_rows:
        rows = sorted(segments_by_source[source["ebook_id"]], key=lambda row: int(row["character_start"]))
        cursor = 0
        for row in rows:
            if int(row["character_start"]) != cursor:
                raise ValueError(f"segment partition gap for pg{source['ebook_id']}")
            cursor = int(row["character_end"])
        if cursor != int(source["cleaned_character_count"]):
            raise ValueError(f"segment partition length mismatch for pg{source['ebook_id']}")


def _build_report(config: GutenbergExtractionAuditConfig, sources: list[dict[str, Any]], segments: list[dict[str, Any]], sonnets: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> dict[str, Any]:
    roles: dict[str, dict[str, int]] = defaultdict(lambda: {"record_count": 0, "included_record_characters": 0})
    for row in sources:
        if row["source_decision"].startswith("eligible_"):
            roles[row["final_role"]]["record_count"] += 1
            roles[row["final_role"]]["included_record_characters"] += int(row["included_record_character_count"])
    return {
        "audit_version": "project_gutenberg_extraction_audit_v1",
        "created_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "source_count": len(sources),
        "source_decision_counts": dict(sorted(Counter(row["source_decision"] for row in sources).items())),
        "segment_decision_counts": dict(sorted(Counter(row["segment_decision"] for row in segments).items())),
        "role_summary": dict(sorted(roles.items())),
        "sonnet_candidate_count": len(sonnets),
        "sonnet_decision_counts": dict(sorted(Counter(row["candidate_decision"] for row in sonnets).items())),
        "eligible_standard_sonnet_count": sum(row["candidate_decision"] == "eligible_standard_sonnet_pending_processed_build" for row in sonnets),
        "unresolved_sonnet_review_count": sum(row["candidate_decision"] == "review_structural_sonnet_candidate" for row in sonnets),
        "heldout_sonnet_conflict_count": sum(bool(row["heldout_reference_ids"]) for row in sonnets),
        "manual_review_count": len(reviews),
        "manual_review_resolution_counts": dict(sorted(Counter(row["review_resolution"] or "unresolved" for row in reviews).items())),
        "outputs": {
            "source_csv_path": _portable(config.source_csv_path, config.repo_root),
            "source_csv_sha256": _sha256_file(config.source_csv_path),
            "segment_csv_path": _portable(config.segment_csv_path, config.repo_root),
            "segment_csv_sha256": _sha256_file(config.segment_csv_path),
            "sonnet_csv_path": _portable(config.sonnet_csv_path, config.repo_root),
            "sonnet_csv_sha256": _sha256_file(config.sonnet_csv_path),
            "review_csv_path": _portable(config.review_csv_path, config.repo_root),
            "review_csv_sha256": _sha256_file(config.review_csv_path),
            "json_report_path": _portable(config.json_report_path, config.repo_root),
            "markdown_report_path": _portable(config.markdown_report_path, config.repo_root),
        },
        "inputs": {
            "prior_probe_csv_sha256": _sha256_file(config.prior_probe_csv_path),
            "pass1b_probe_csv_sha256": _sha256_file(config.pass1b_probe_csv_path),
            "final_resolution_csv_sha256": _sha256_file(config.final_resolution_csv_path),
            "bibit_record_manifest_sha256": _sha256_file(config.bibit_record_manifest_path),
            "broader_sources_manifest_sha256": _sha256_file(config.broader_sources_manifest_path),
            "sonnet_manifest_sha256": _sha256_file(config.sonnet_manifest_path),
            "bibit_sonnet_manifest_sha256": _sha256_file(config.bibit_sonnet_manifest_path),
        },
        "policy": {
            "processed_text_materialized": False,
            "v7_split_assigned": False,
            "training_mixture_weight_assigned": False,
            "conditioned_experiment_authorized": False,
            "source_spelling_and_punctuation_preserved": True,
            "confirmed_and_unresolved_sonnet_units_quarantined_from_broader_text": True,
            "manual_non_sonnet_false_positives_retained_in_broader_text": True,
            "heldout_containment_threshold": config.heldout_containment_threshold,
            "near_sonnet_overlap_threshold": config.near_sonnet_overlap_threshold,
            "near_sonnet_sequence_threshold": config.near_sonnet_sequence_threshold,
        },
    }


def _validate_config(config: GutenbergExtractionAuditConfig) -> None:
    for path in (
        config.prior_probe_csv_path,
        config.pass1b_probe_csv_path,
        config.final_resolution_csv_path,
        config.bibit_record_manifest_path,
        config.broader_sources_manifest_path,
        config.sonnet_manifest_path,
        config.bibit_sonnet_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (config.prior_cache_dir, config.pass1b_cache_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if config.progress_interval <= 0:
        raise ValueError("progress_interval must be positive")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _portable(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _progress(progress: Progress | None, phase: str, index: int, total: int, started: float, detail: str) -> None:
    if progress is None:
        return
    elapsed = monotonic() - started
    eta = elapsed / index * (total - index) if index else 0.0
    progress(f"{phase} {index:,}/{total:,} ({index / total:.1%}) {detail} elapsed={elapsed:.1f}s eta={eta:.1f}s")
