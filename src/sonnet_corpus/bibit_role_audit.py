"""Resumable role-specific TEI audit for the Biblioteca Italiana catalog."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from time import monotonic, sleep as default_sleep
from typing import Any

import requests

from .biblioteca_italiana import (
    BibItSonnetUnit,
    BibItVerseUnit,
    fetch_bibit_tei,
    parse_bibit_tei,
)
from .bibit_composition_audit import (
    ROLE_BRIDGE,
    ROLE_EXCLUDED,
    ROLE_HISTORICAL_GENERAL,
    ROLE_HISTORICAL_POETRY,
    ROLE_SONNET_ONLY,
)


ACTIVE_ROLES = {
    ROLE_HISTORICAL_GENERAL,
    ROLE_HISTORICAL_POETRY,
    ROLE_SONNET_ONLY,
    ROLE_BRIDGE,
}
HELD_OUT_SPLITS = {"validation", "test"}
_DIALECT_METADATA = re.compile(
    r"\b(?:dialett\w*|romanes\w*)\b",
    re.IGNORECASE,
)
_KNOWN_DIALECT_WORK = re.compile(
    r"\b(?:lo cunto de li cunti|le muse napolitane|comedia di malpratico|"
    r"legenda de misier sento alban|testi veneziani|rainaldo e lesengrino|"
    r"rimatori bolognesi|commento di jacopo di giovanni dalla lana bolognese)\b",
    re.IGNORECASE,
)
_BRACKETED_TEXT = re.compile(r"\[[^\]\n]{1,160}\]")
_EDITORIAL_REFERENCE = re.compile(
    r"\b(?:si veda|cfr\.?|nota dell['’]editore|nota del curatore)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"\w+", re.UNICODE)

RECORD_FIELDS = (
    "object_id",
    "title",
    "authors",
    "periods",
    "genres",
    "assigned_role",
    "route",
    "audit_status",
    "audit_flags",
    "duplicate_of_object_id",
    "cache_status",
    "tei_sha256",
    "body_characters",
    "candidate_safe_characters",
    "routed_training_characters",
    "residual_characters",
    "non_sonnet_verse_characters",
    "explicit_sonnet_count",
    "structural_14_line_candidate_count",
    "eligible_new_sonnet_count",
    "held_out_text_hits",
    "bracketed_text_markers",
    "editorial_reference_markers",
    "digital_title",
    "digital_authors",
    "source_titles",
    "source_editors",
    "source_publisher",
    "source_publication_place",
    "source_publication_date",
    "source_identifier",
    "availability",
    "tei_languages",
    "tei_genres",
    "revision_count",
    "landing_page_url",
    "xml_url",
    "error",
)

SONNET_FIELDS = (
    "candidate_id",
    "object_id",
    "title",
    "authors",
    "periods",
    "source_kind",
    "tei_type",
    "heading_path",
    "line_count",
    "character_count",
    "status",
    "exact_active_duplicate_poem_ids",
    "near_active_duplicate_poem_ids",
    "held_out_duplicate_poem_ids",
    "duplicate_bibit_candidate_id",
    "text_sha256",
    "normalized_sha256",
    "first_line",
    "last_line",
    "landing_page_url",
)


@dataclass(frozen=True)
class BibItRoleAuditConfig:
    """Paths and bounded network settings for the full BibIt TEI audit."""

    repo_root: Path
    decision_csv_path: Path
    sonnet_manifest_path: Path
    tei_cache_dir: Path
    checkpoint_path: Path
    record_csv_path: Path
    sonnet_csv_path: Path
    json_report_path: Path
    markdown_report_path: Path
    request_delay_seconds: float = 0.25
    request_timeout_seconds: float = 180.0
    max_retries: int = 3
    progress_interval: int = 10
    checkpoint_interval: int = 25
    min_training_characters: int = 200
    near_sonnet_overlap_threshold: float = 0.72
    near_sonnet_sequence_threshold: float = 0.86
    near_document_sequence_threshold: float = 0.94
    limit: int = 0


@dataclass(frozen=True)
class ReferenceSonnet:
    poem_id: str
    split: str
    exact_text: str
    loose_text: str
    word_ngrams: frozenset[str]


FetchTEI = Callable[..., bytes]
Progress = Callable[[str], None]
Sleep = Callable[[float], None]


def audit_bibit_tei_roles(
    config: BibItRoleAuditConfig,
    *,
    fetch_tei: FetchTEI = fetch_bibit_tei,
    session: requests.Session | None = None,
    sleep: Sleep = default_sleep,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Audit every canonical BibIt TEI and write resumable public evidence."""

    _validate_config(config)
    started_at = _utc_now()
    started = monotonic()
    decisions = read_active_bibit_decisions(config.decision_csv_path)
    if config.limit:
        decisions = decisions[: config.limit]
    references = load_reference_sonnets(
        config.sonnet_manifest_path,
        config.repo_root,
    )
    reference_ngram_index = _build_reference_ngram_index(references)
    held_out_references = [
        reference for reference in references if reference.split in HELD_OUT_SPLITS
    ]
    config.tei_cache_dir.mkdir(parents=True, exist_ok=True)
    _report(
        progress,
        f"loaded records={len(decisions):,} active_sonnets={len(references):,} "
        f"held_out_sonnets={len(held_out_references):,}",
    )

    records: list[dict[str, Any]] = []
    sonnet_rows: list[dict[str, Any]] = []
    exact_document_groups: dict[str, list[str]] = defaultdict(list)
    document_samples: dict[str, str] = {}
    exact_bibit_sonnets: dict[str, str] = {}
    next_request_at = 0.0
    client = session or requests.Session()

    for index, decision in enumerate(decisions, start=1):
        object_id = decision["object_id"]
        cache_path = config.tei_cache_dir / f"{object_id}.xml"
        cache_status = "hit" if cache_path.is_file() else "miss"
        error = ""
        try:
            if cache_path.is_file():
                xml = cache_path.read_bytes()
            else:
                wait_seconds = max(0.0, next_request_at - monotonic())
                if wait_seconds:
                    sleep(wait_seconds)
                xml = _fetch_with_retries(
                    object_id,
                    fetch_tei=fetch_tei,
                    session=client,
                    timeout=config.request_timeout_seconds,
                    max_retries=config.max_retries,
                    sleep=sleep,
                    progress=progress,
                )
                next_request_at = monotonic() + config.request_delay_seconds
                _atomic_write_bytes(cache_path, xml)
                cache_status = "downloaded"
            parsed = parse_bibit_tei(xml, object_id=object_id)
        except Exception as caught:  # Preserve all other archive outcomes.
            error = f"{type(caught).__name__}: {caught}"
            record = _error_record(decision, cache_status, error)
            records.append(record)
            _report(progress, f"record error {object_id}: {error}")
        else:
            training_text, route = _route_training_text(decision["role"], parsed)
            held_out_hits = _find_held_out_text_hits(training_text, held_out_references)
            flags = _record_flags(
                decision,
                parsed=parsed,
                training_text=training_text,
                held_out_hits=held_out_hits,
                min_training_characters=config.min_training_characters,
            )
            document_fingerprint = _sha256_text(normalize_exact_text(training_text))
            if training_text.strip():
                exact_document_groups[document_fingerprint].append(object_id)
                document_samples[object_id] = _bounded_document_sample(training_text)

            candidate_rows = _audit_record_sonnet_candidates(
                decision,
                parsed_sonnets=parsed.sonnets,
                structural_candidates=parsed.structural_sonnet_candidates,
                references=references,
                reference_ngram_index=reference_ngram_index,
                exact_bibit_sonnets=exact_bibit_sonnets,
                source_blocked=bool(_candidate_source_blocking_flags(flags)),
                near_overlap_threshold=config.near_sonnet_overlap_threshold,
                near_sequence_threshold=config.near_sonnet_sequence_threshold,
            )
            sonnet_rows.extend(candidate_rows)
            record = _record_result(
                decision,
                parsed=parsed,
                xml=xml,
                training_text=training_text,
                route=route,
                flags=flags,
                held_out_hits=held_out_hits,
                candidate_rows=candidate_rows,
                cache_status=cache_status,
            )
            records.append(record)

        if index % config.checkpoint_interval == 0 or index == len(decisions):
            _write_checkpoint(
                config.checkpoint_path,
                started_at=started_at,
                total_records=len(decisions),
                records=records,
                sonnet_rows=sonnet_rows,
            )
        if (
            index == 1
            or index % config.progress_interval == 0
            or index == len(decisions)
        ):
            elapsed = monotonic() - started
            eta = elapsed / index * (len(decisions) - index) if index else 0.0
            _report(
                progress,
                f"record {index:,}/{len(decisions):,} ({index / len(decisions):.1%}) "
                f"id={object_id} cache={cache_status} errors="
                f"{sum(bool(row['error']) for row in records):,} "
                f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
            )

    _apply_exact_document_duplicates(records, exact_document_groups)
    _apply_near_document_duplicates(
        records,
        document_samples,
        threshold=config.near_document_sequence_threshold,
    )
    _refresh_record_statuses(records)
    _apply_final_record_blocks_to_sonnets(records, sonnet_rows)
    _refresh_eligible_sonnet_counts(records, sonnet_rows)
    report = build_bibit_role_report(
        config=config,
        records=records,
        sonnet_rows=sonnet_rows,
        started_at=started_at,
        finished_at=_utc_now(),
        elapsed_seconds=monotonic() - started,
        reference_count=len(references),
        held_out_reference_count=len(held_out_references),
    )
    write_csv(config.record_csv_path, RECORD_FIELDS, records)
    write_csv(config.sonnet_csv_path, SONNET_FIELDS, sonnet_rows)
    _write_json(config.json_report_path, report)
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(
        render_bibit_role_markdown(report),
        encoding="utf-8",
    )
    _write_checkpoint(
        config.checkpoint_path,
        started_at=started_at,
        total_records=len(decisions),
        records=records,
        sonnet_rows=sonnet_rows,
        complete=True,
    )
    return report


def read_active_bibit_decisions(path: Path) -> list[dict[str, str]]:
    """Read exactly the canonical, role-assigned records from the composition gate."""

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"object_id", "canonical_status", "role", "title", "authors"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("BibIt decision CSV is empty or missing required fields")
    selected = [
        row
        for row in rows
        if row["canonical_status"] == "selected" and row["role"] in ACTIVE_ROLES
    ]
    object_ids = [row["object_id"] for row in selected]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("BibIt decision CSV contains duplicate selected object IDs")
    if any(row["role"] == ROLE_EXCLUDED for row in selected):
        raise ValueError("excluded BibIt records cannot enter the TEI audit")
    return selected


def load_reference_sonnets(path: Path, repo_root: Path) -> list[ReferenceSonnet]:
    """Load active V6 sonnets and their fixed train/validation/test identities."""

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    references: list[ReferenceSonnet] = []
    for row in rows:
        if row.get("include_in_training") != "True":
            continue
        split = row.get("split_expanded_with_petrarch", "")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"active sonnet has invalid fixed split: {row.get('poem_id', '')}")
        text = (repo_root / row["clean_text_path"]).read_text(encoding="utf-8")
        loose_text = normalize_loose_text(text)
        references.append(
            ReferenceSonnet(
                poem_id=row["poem_id"],
                split=split,
                exact_text=normalize_exact_text(text),
                loose_text=loose_text,
                word_ngrams=frozenset(_word_ngrams(loose_text)),
            )
        )
    if not references:
        raise ValueError("sonnet reference manifest has no active poems")
    return references


def normalize_exact_text(text: str) -> str:
    """Normalize only whitespace and case for conservative exact identity checks."""

    return re.sub(r"\s+", " ", text).strip().casefold()


def normalize_loose_text(text: str) -> str:
    """Normalize accents and punctuation for edition-level near-duplicate checks."""

    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(_WORD.findall(without_marks))


def _route_training_text(role: str, parsed: Any) -> tuple[str, str]:
    if role in {ROLE_HISTORICAL_GENERAL, ROLE_HISTORICAL_POETRY, ROLE_BRIDGE}:
        return parsed.sonnet_candidate_safe_text, role
    if role == ROLE_SONNET_ONLY:
        return parsed.sonnet_candidate_safe_text, ROLE_HISTORICAL_POETRY
    raise ValueError(f"unsupported BibIt role: {role}")


def _record_flags(
    decision: dict[str, str],
    *,
    parsed: Any,
    training_text: str,
    held_out_hits: list[str],
    min_training_characters: int,
) -> list[str]:
    flags: list[str] = []
    if requires_language_variety_review(
        decision.get("title", ""),
        parsed.provenance.languages,
        parsed.provenance.genres,
    ):
        flags.append("review_language_variety")
    normalized_languages = [normalize_loose_text(value) for value in parsed.provenance.languages]
    has_italian = any(
        value == "ita" or "italian" in value
        for value in normalized_languages
    )
    if normalized_languages and not has_italian:
        flags.append("review_non_italian_language")
    elif len(normalized_languages) > 1:
        flags.append("note_multilingual_metadata")
    if not parsed.provenance.digital_title:
        flags.append("review_missing_digital_title")
    if not parsed.provenance.availability:
        flags.append("review_missing_availability")
    if not parsed.provenance.source_titles and not parsed.provenance.source_identifier:
        flags.append("review_missing_source_edition")
    if held_out_hits:
        flags.append("held_out_sonnet_text_in_earlier_stage")
    if decision["role"] != ROLE_SONNET_ONLY and len(training_text.strip()) < min_training_characters:
        flags.append("empty_or_too_short_after_sonnet_quarantine")
    if decision["role"] == ROLE_SONNET_ONLY and not parsed.sonnets and not any(
        parsed.structural_sonnet_candidates
    ):
        flags.append("review_no_sonnet_candidates")
    if len(_BRACKETED_TEXT.findall(training_text)) >= 20:
        flags.append("review_editorial_brackets")
    if len(_EDITORIAL_REFERENCE.findall(training_text)) >= 10:
        flags.append("review_editorial_references")
    return flags


def requires_language_variety_review(
    title: str,
    languages: tuple[str, ...],
    genres: tuple[str, ...],
) -> bool:
    """Flag explicit dialect evidence without matching geographic or personal names."""

    metadata_text = " ".join((title, *languages, *genres))
    return bool(
        _DIALECT_METADATA.search(metadata_text)
        or _KNOWN_DIALECT_WORK.search(metadata_text)
    )


def _blocking_record_flags(flags: list[str]) -> list[str]:
    return [
        flag
        for flag in flags
        if flag.startswith("review_")
        or flag.startswith("held_out_")
        or flag.startswith("duplicate_")
    ]


def _candidate_source_blocking_flags(flags: list[str]) -> list[str]:
    blocking_prefixes = (
        "review_language_variety",
        "review_non_italian_language",
        "review_missing_availability",
        "review_missing_source_edition",
    )
    return [flag for flag in flags if flag.startswith(blocking_prefixes)]


def _audit_record_sonnet_candidates(
    decision: dict[str, str],
    *,
    parsed_sonnets: tuple[BibItSonnetUnit, ...],
    structural_candidates: tuple[BibItVerseUnit, ...],
    references: list[ReferenceSonnet],
    reference_ngram_index: dict[str, set[int]],
    exact_bibit_sonnets: dict[str, str],
    source_blocked: bool,
    near_overlap_threshold: float,
    near_sequence_threshold: float,
) -> list[dict[str, Any]]:
    candidates: list[tuple[str, str, str, tuple[str, ...], str, int]] = []
    candidates.extend(
        (unit.unit_id, "explicit_tei_sonnet", unit.sonnet_type, unit.heading_path, unit.text, unit.line_count)
        for unit in parsed_sonnets
    )
    candidates.extend(
        (unit.unit_id, "structural_14_line", unit.verse_type, unit.heading_path, unit.text, unit.line_count)
        for unit in structural_candidates
    )
    rows: list[dict[str, Any]] = []
    for unit_id, source_kind, tei_type, heading_path, text, line_count in candidates:
        candidate_id = f"{decision['object_id']}:{unit_id}"
        exact = normalize_exact_text(text)
        loose = normalize_loose_text(text)
        exact_matches = [reference.poem_id for reference in references if reference.exact_text == exact]
        near_matches = _near_reference_matches(
            loose,
            references=references,
            ngram_index=reference_ngram_index,
            overlap_threshold=near_overlap_threshold,
            sequence_threshold=near_sequence_threshold,
            excluded_poem_ids=set(exact_matches),
        )
        reference_by_id = {reference.poem_id: reference for reference in references}
        held_out_matches = sorted(
            poem_id
            for poem_id in set(exact_matches) | set(near_matches)
            if reference_by_id[poem_id].split in HELD_OUT_SPLITS
        )
        prior_bibit = exact_bibit_sonnets.get(exact)
        if exact and prior_bibit is None:
            exact_bibit_sonnets[exact] = candidate_id

        missing_author = normalize_loose_text(decision.get("authors", "")) in {
            "",
            "non definito",
        }
        candidate_editorial_markers = bool(
            _BRACKETED_TEXT.search(text) or _EDITORIAL_REFERENCE.search(text)
        )
        if not text.strip():
            status = "excluded_empty"
        elif line_count != 14:
            status = "excluded_not_14_lines"
        elif held_out_matches:
            status = "excluded_held_out_identity_conflict"
        elif exact_matches:
            status = "excluded_exact_active_duplicate"
        elif near_matches:
            status = "review_near_active_duplicate"
        elif prior_bibit:
            status = "excluded_exact_bibit_duplicate"
        elif missing_author:
            status = "review_missing_author_attribution"
        elif candidate_editorial_markers:
            status = "review_candidate_editorial_markers"
        elif source_blocked:
            status = "review_source_blocked"
        elif source_kind == "structural_14_line":
            status = "review_structural_form"
        else:
            status = "eligible_explicit_nonduplicate"
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        rows.append(
            {
                "candidate_id": candidate_id,
                "object_id": decision["object_id"],
                "title": decision["title"],
                "authors": decision["authors"],
                "periods": decision.get("periods", ""),
                "source_kind": source_kind,
                "tei_type": tei_type,
                "heading_path": " > ".join(heading_path),
                "line_count": line_count,
                "character_count": len(text),
                "status": status,
                "exact_active_duplicate_poem_ids": ";".join(sorted(exact_matches)),
                "near_active_duplicate_poem_ids": ";".join(sorted(near_matches)),
                "held_out_duplicate_poem_ids": ";".join(held_out_matches),
                "duplicate_bibit_candidate_id": prior_bibit or "",
                "text_sha256": _sha256_text(text),
                "normalized_sha256": _sha256_text(loose),
                "first_line": lines[0] if lines else "",
                "last_line": lines[-1] if lines else "",
                "landing_page_url": decision.get("landing_page_url", ""),
            }
        )
    return rows


def _near_reference_matches(
    loose_text: str,
    *,
    references: list[ReferenceSonnet],
    ngram_index: dict[str, set[int]],
    overlap_threshold: float,
    sequence_threshold: float,
    excluded_poem_ids: set[str],
) -> list[str]:
    grams = set(_word_ngrams(loose_text))
    candidate_indices: set[int] = set()
    for gram in grams:
        candidate_indices.update(ngram_index.get(gram, ()))
    matches: list[str] = []
    for index in candidate_indices:
        reference = references[index]
        if reference.poem_id in excluded_poem_ids:
            continue
        denominator = min(len(grams), len(reference.word_ngrams))
        overlap = len(grams & reference.word_ngrams) / denominator if denominator else 0.0
        if overlap >= overlap_threshold:
            matches.append(reference.poem_id)
            continue
        if overlap < 0.20:
            continue
        ratio = SequenceMatcher(None, loose_text, reference.loose_text, autojunk=False).ratio()
        if ratio >= sequence_threshold:
            matches.append(reference.poem_id)
    return sorted(matches)


def _build_reference_ngram_index(references: list[ReferenceSonnet]) -> dict[str, set[int]]:
    index: dict[str, set[int]] = defaultdict(set)
    for reference_index, reference in enumerate(references):
        for gram in reference.word_ngrams:
            index[gram].add(reference_index)
    return index


def _word_ngrams(text: str, size: int = 3) -> list[str]:
    words = text.split()
    if len(words) < size:
        return [" ".join(words)] if words else []
    return [" ".join(words[index : index + size]) for index in range(len(words) - size + 1)]


def _find_held_out_text_hits(
    training_text: str,
    held_out_references: list[ReferenceSonnet],
) -> list[str]:
    if not training_text.strip():
        return []
    exact_document = normalize_exact_text(training_text)
    loose_document = normalize_loose_text(training_text)
    return sorted(
        reference.poem_id
        for reference in held_out_references
        if (reference.exact_text and reference.exact_text in exact_document)
        or (reference.loose_text and reference.loose_text in loose_document)
    )


def _record_result(
    decision: dict[str, str],
    *,
    parsed: Any,
    xml: bytes,
    training_text: str,
    route: str,
    flags: list[str],
    held_out_hits: list[str],
    candidate_rows: list[dict[str, Any]],
    cache_status: str,
) -> dict[str, Any]:
    safe_verse_characters = sum(
        len(unit.text) for unit in parsed.non_sonnet_verse if unit.line_count != 14
    )
    eligible_sonnets = sum(
        row["status"] == "eligible_explicit_nonduplicate" for row in candidate_rows
    )
    provenance = parsed.provenance
    return {
        "object_id": decision["object_id"],
        "title": decision["title"],
        "authors": decision["authors"],
        "periods": decision.get("periods", ""),
        "genres": decision.get("genres", ""),
        "assigned_role": decision["role"],
        "route": route,
        "audit_status": "review_required" if _blocking_record_flags(flags) else "activation_candidate",
        "audit_flags": ";".join(flags),
        "duplicate_of_object_id": "",
        "cache_status": cache_status,
        "tei_sha256": hashlib.sha256(xml).hexdigest(),
        "body_characters": len(parsed.body_text),
        "candidate_safe_characters": len(parsed.sonnet_candidate_safe_text),
        "routed_training_characters": len(training_text),
        "residual_characters": len(parsed.residual_text),
        "non_sonnet_verse_characters": safe_verse_characters,
        "explicit_sonnet_count": len(parsed.sonnets),
        "structural_14_line_candidate_count": sum(
            1 for _ in parsed.structural_sonnet_candidates
        ),
        "eligible_new_sonnet_count": eligible_sonnets,
        "held_out_text_hits": ";".join(held_out_hits),
        "bracketed_text_markers": len(_BRACKETED_TEXT.findall(training_text)),
        "editorial_reference_markers": len(_EDITORIAL_REFERENCE.findall(training_text)),
        "digital_title": provenance.digital_title,
        "digital_authors": ";".join(provenance.digital_authors),
        "source_titles": ";".join(provenance.source_titles),
        "source_editors": ";".join(provenance.source_editors),
        "source_publisher": provenance.source_publisher,
        "source_publication_place": provenance.source_publication_place,
        "source_publication_date": provenance.source_publication_date,
        "source_identifier": provenance.source_identifier,
        "availability": provenance.availability,
        "tei_languages": ";".join(provenance.languages),
        "tei_genres": ";".join(provenance.genres),
        "revision_count": len(provenance.revisions),
        "landing_page_url": decision.get("landing_page_url", ""),
        "xml_url": decision.get("xml_url", ""),
        "error": "",
    }


def _error_record(decision: dict[str, str], cache_status: str, error: str) -> dict[str, Any]:
    row = {field: "" for field in RECORD_FIELDS}
    row.update(
        {
            "object_id": decision["object_id"],
            "title": decision["title"],
            "authors": decision["authors"],
            "periods": decision.get("periods", ""),
            "genres": decision.get("genres", ""),
            "assigned_role": decision["role"],
            "audit_status": "error",
            "audit_flags": "fetch_or_parse_error",
            "cache_status": cache_status,
            "landing_page_url": decision.get("landing_page_url", ""),
            "xml_url": decision.get("xml_url", ""),
            "error": error,
        }
    )
    return row


def _apply_exact_document_duplicates(
    records: list[dict[str, Any]],
    exact_groups: dict[str, list[str]],
) -> None:
    by_id = {row["object_id"]: row for row in records}
    for object_ids in exact_groups.values():
        if len(object_ids) < 2:
            continue
        roles = {by_id[object_id]["route"] for object_id in object_ids}
        if len(roles) > 1:
            for object_id in object_ids:
                _append_flag(by_id[object_id], "review_cross_role_exact_duplicate")
            continue
        selected = max(
            object_ids,
            key=lambda object_id: (
                _provenance_score(by_id[object_id]),
                int(by_id[object_id]["routed_training_characters"] or 0),
                object_id,
            ),
        )
        for object_id in object_ids:
            if object_id == selected:
                continue
            row = by_id[object_id]
            _append_flag(row, "duplicate_exact_bibit_document")
            row["duplicate_of_object_id"] = selected


def _apply_near_document_duplicates(
    records: list[dict[str, Any]],
    samples: dict[str, str],
    *,
    threshold: float,
) -> None:
    by_author: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row["error"] or not row["routed_training_characters"]:
            continue
        by_author[normalize_loose_text(str(row["authors"]))].append(row)
    for author_rows in by_author.values():
        for left_index, left in enumerate(author_rows):
            left_count = int(left["routed_training_characters"])
            if not left_count:
                continue
            for right in author_rows[left_index + 1 :]:
                right_count = int(right["routed_training_characters"])
                length_ratio = min(left_count, right_count) / max(left_count, right_count)
                if length_ratio < 0.85:
                    continue
                if (
                    left.get("duplicate_of_object_id")
                    or right.get("duplicate_of_object_id")
                ):
                    continue
                ratio = SequenceMatcher(
                    None,
                    samples[left["object_id"]].split(),
                    samples[right["object_id"]].split(),
                    autojunk=True,
                ).ratio()
                if ratio >= threshold:
                    _append_flag(left, f"review_near_duplicate:{right['object_id']}")
                    _append_flag(right, f"review_near_duplicate:{left['object_id']}")


def _append_flag(row: dict[str, Any], flag: str) -> None:
    flags = [value for value in str(row.get("audit_flags", "")).split(";") if value]
    if flag not in flags:
        flags.append(flag)
    row["audit_flags"] = ";".join(flags)


def _refresh_record_statuses(records: list[dict[str, Any]]) -> None:
    for row in records:
        if row["error"]:
            row["audit_status"] = "error"
        elif _blocking_record_flags(
            [value for value in str(row["audit_flags"]).split(";") if value]
        ):
            row["audit_status"] = "review_required"
        else:
            row["audit_status"] = "activation_candidate"


def _apply_final_record_blocks_to_sonnets(
    records: list[dict[str, Any]],
    sonnet_rows: list[dict[str, Any]],
) -> None:
    blocked_ids = {
        row["object_id"]
        for row in records
        if _record_blocks_sonnet_candidates(row)
    }
    for row in sonnet_rows:
        if (
            row["object_id"] in blocked_ids
            and row["status"] == "eligible_explicit_nonduplicate"
        ):
            row["status"] = "review_source_blocked_post_dedup"


def _record_blocks_sonnet_candidates(row: dict[str, Any]) -> bool:
    if row["error"]:
        return True
    flags = [value for value in str(row["audit_flags"]).split(";") if value]
    return bool(
        _candidate_source_blocking_flags(flags)
        or any(
            flag.startswith(
                (
                    "duplicate_",
                    "review_cross_role_exact_duplicate",
                    "review_near_duplicate",
                )
            )
            for flag in flags
        )
    )


def _refresh_eligible_sonnet_counts(
    records: list[dict[str, Any]],
    sonnet_rows: list[dict[str, Any]],
) -> None:
    counts = Counter(
        row["object_id"]
        for row in sonnet_rows
        if row["status"] == "eligible_explicit_nonduplicate"
    )
    for row in records:
        row["eligible_new_sonnet_count"] = counts[row["object_id"]]


def build_bibit_role_report(
    *,
    config: BibItRoleAuditConfig,
    records: list[dict[str, Any]],
    sonnet_rows: list[dict[str, Any]],
    started_at: str,
    finished_at: str,
    elapsed_seconds: float,
    reference_count: int,
    held_out_reference_count: int,
) -> dict[str, Any]:
    record_statuses = Counter(row["audit_status"] for row in records)
    sonnet_statuses = Counter(row["status"] for row in sonnet_rows)
    flags = Counter(
        flag.split(":", maxsplit=1)[0]
        for row in records
        for flag in str(row["audit_flags"]).split(";")
        if flag
    )
    routes: dict[str, dict[str, int]] = {}
    for route in sorted({str(row["route"]) for row in records if row["route"]}):
        route_rows = [row for row in records if row["route"] == route]
        candidate_rows = [row for row in route_rows if row["audit_status"] == "activation_candidate"]
        routes[route] = {
            "record_count": len(route_rows),
            "activation_candidate_count": len(candidate_rows),
            "review_or_error_count": len(route_rows) - len(candidate_rows),
            "all_routed_characters": sum(int(row["routed_training_characters"] or 0) for row in route_rows),
            "candidate_routed_characters": sum(
                int(row["routed_training_characters"] or 0) for row in candidate_rows
            ),
        }
    author_characters = Counter()
    for row in records:
        if row["audit_status"] == "activation_candidate":
            author_characters[row["authors"] or "(missing)"] += int(
                row["routed_training_characters"] or 0
            )
    report = {
        "audit_version": "bibit_role_specific_tei_audit_v1",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "inputs": {
            "decision_csv_path": _portable(config.decision_csv_path, config.repo_root),
            "sonnet_manifest_path": _portable(config.sonnet_manifest_path, config.repo_root),
            "reference_sonnet_count": reference_count,
            "held_out_reference_sonnet_count": held_out_reference_count,
        },
        "outputs": {
            "record_csv_path": _portable(config.record_csv_path, config.repo_root),
            "sonnet_csv_path": _portable(config.sonnet_csv_path, config.repo_root),
            "json_report_path": _portable(config.json_report_path, config.repo_root),
            "markdown_report_path": _portable(config.markdown_report_path, config.repo_root),
            "local_tei_cache_path": _portable(config.tei_cache_dir, config.repo_root),
        },
        "record_count": len(records),
        "record_status_counts": dict(sorted(record_statuses.items())),
        "record_flag_counts": dict(sorted(flags.items())),
        "route_summary": routes,
        "body_character_count": sum(int(row["body_characters"] or 0) for row in records),
        "routed_training_character_count": sum(
            int(row["routed_training_characters"] or 0) for row in records
        ),
        "sonnet_candidate_count": len(sonnet_rows),
        "sonnet_status_counts": dict(sorted(sonnet_statuses.items())),
        "explicit_sonnet_candidate_count": sum(
            row["source_kind"] == "explicit_tei_sonnet" for row in sonnet_rows
        ),
        "structural_sonnet_candidate_count": sum(
            row["source_kind"] == "structural_14_line" for row in sonnet_rows
        ),
        "held_out_identity_conflict_count": sum(
            bool(row["held_out_duplicate_poem_ids"]) for row in sonnet_rows
        ),
        "top_candidate_authors_by_characters": [
            {"author": author, "characters": characters}
            for author, characters in author_characters.most_common(20)
        ],
        "corpus_activation_status": (
            "audit_complete_manual_review_required"
            if record_statuses.get("review_required", 0) or record_statuses.get("error", 0)
            else "audit_complete_ready_for_build"
        ),
        "policy": {
            "explicit_sonnets_removed_from_earlier_stages": True,
            "structural_14_line_units_quarantined": True,
            "v6_validation_test_text_blocked_from_earlier_stages": True,
            "dialect_heavy_material_requires_separate_conditioned_role": True,
            "raw_tei_cache_is_machine_local": True,
        },
    }
    return report


def render_bibit_role_markdown(report: dict[str, Any]) -> str:
    """Render the compact public decision report for the full TEI audit."""

    lines = [
        "# Biblioteca Italiana Role-Specific TEI Audit",
        "",
        "## Decision",
        "",
        (
            f"Audited {report['record_count']:,} canonical TEI records and "
            f"{report['sonnet_candidate_count']:,} sonnet candidates. Status: "
            f"`{report['corpus_activation_status']}`."
        ),
        "",
        "This is an activation gate, not a blind concatenation. Explicit sonnets and",
        "unverified 14-line verse units are absent from every earlier-stage text route.",
        "V6 validation/test identities are checked before any candidate can enter training.",
        "",
        "## Record Outcomes",
        "",
        "| Status | Records |",
        "| --- | ---: |",
    ]
    for status, count in report["record_status_counts"].items():
        lines.append(f"| `{status}` | {count:,} |")
    lines.extend(
        [
            "",
            "## Routed Corpus",
            "",
            "| Route | Records | Automatic candidates | Candidate characters | Review/error |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for route, values in report["route_summary"].items():
        lines.append(
            f"| `{route}` | {values['record_count']:,} | "
            f"{values['activation_candidate_count']:,} | "
            f"{values['candidate_routed_characters']:,} | "
            f"{values['review_or_error_count']:,} |"
        )
    lines.extend(
        [
            "",
            "## Sonnet Candidates",
            "",
            f"- Explicit TEI sonnets: {report['explicit_sonnet_candidate_count']:,}.",
            f"- Unverified structural 14-line candidates: {report['structural_sonnet_candidate_count']:,}.",
            f"- Held-out identity conflicts: {report['held_out_identity_conflict_count']:,}.",
            "",
            "| Status | Candidates |",
            "| --- | ---: |",
        ]
    )
    for status, count in report["sonnet_status_counts"].items():
        lines.append(f"| `{status}` | {count:,} |")
    lines.extend(
        [
            "",
            "## Review Queue",
            "",
            "| Flag | Records |",
            "| --- | ---: |",
        ]
    )
    for flag, count in report["record_flag_counts"].items():
        lines.append(f"| `{flag}` | {count:,} |")
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- Per-record decisions: `{report['outputs']['record_csv_path']}`",
            f"- Per-sonnet decisions: `{report['outputs']['sonnet_csv_path']}`",
            f"- Machine-readable summary: `{report['outputs']['json_report_path']}`",
            "- Raw TEI is cached only under ignored `data/local/`; every public record retains its TEI SHA-256.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _fetch_with_retries(
    object_id: str,
    *,
    fetch_tei: FetchTEI,
    session: requests.Session,
    timeout: float,
    max_retries: int,
    sleep: Sleep,
    progress: Progress | None,
) -> bytes:
    for attempt in range(1, max_retries + 1):
        try:
            return fetch_tei(object_id, session=session, timeout=timeout)
        except Exception:
            if attempt == max_retries:
                raise
            delay = float(2 ** (attempt - 1))
            _report(progress, f"retry {attempt}/{max_retries - 1} for {object_id} after {delay:.0f}s")
            sleep(delay)
    raise AssertionError("unreachable")


def _validate_config(config: BibItRoleAuditConfig) -> None:
    if config.request_delay_seconds < 0:
        raise ValueError("request_delay_seconds cannot be negative")
    if config.request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive")
    if config.max_retries <= 0:
        raise ValueError("max_retries must be positive")
    if config.progress_interval <= 0 or config.checkpoint_interval <= 0:
        raise ValueError("progress and checkpoint intervals must be positive")
    if config.min_training_characters < 0 or config.limit < 0:
        raise ValueError("minimum characters and limit cannot be negative")


def _write_checkpoint(
    path: Path,
    *,
    started_at: str,
    total_records: int,
    records: list[dict[str, Any]],
    sonnet_rows: list[dict[str, Any]],
    complete: bool = False,
) -> None:
    payload = {
        "audit_version": "bibit_role_specific_tei_audit_v1",
        "started_at_utc": started_at,
        "updated_at_utc": _utc_now(),
        "complete": complete,
        "total_records": total_records,
        "completed_records": len(records),
        "completed_object_ids": [row["object_id"] for row in records],
        "error_count": sum(bool(row["error"]) for row in records),
        "sonnet_candidate_count": len(sonnet_rows),
    }
    _write_json(path, payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _bounded_document_sample(text: str, limit: int = 12_000) -> str:
    normalized = normalize_loose_text(text)
    if len(normalized) <= limit:
        return normalized
    section = limit // 3
    middle = len(normalized) // 2
    return normalized[:section] + normalized[middle - section // 2 : middle + section // 2] + normalized[-section:]


def _provenance_score(row: dict[str, Any]) -> int:
    return sum(
        bool(row.get(field))
        for field in (
            "digital_title",
            "source_titles",
            "source_editors",
            "source_publisher",
            "source_publication_date",
            "source_identifier",
            "availability",
        )
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _report(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)
