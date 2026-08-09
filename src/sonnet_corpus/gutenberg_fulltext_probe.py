"""Archive-scale quality and duplicate probe for Italian Project Gutenberg text."""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from time import monotonic, sleep as default_sleep
from typing import Any

import requests

from .gutenberg import FetchedGutenbergText, fetch_gutenberg_text, strip_gutenberg_boilerplate
from .gutenberg_fulltext_gate import PROBE_STATUSES


PROBE_FIELDS = (
    "ebook_id",
    "title",
    "authors",
    "preliminary_role",
    "period_bucket",
    "inventory_status",
    "landing_page_url",
    "fetched_url",
    "cache_status",
    "probe_status",
    "error",
    "raw_character_count",
    "cleaned_character_count",
    "cleaned_word_count",
    "nonempty_line_count",
    "replacement_character_count",
    "italian_function_word_ratio",
    "alphabetic_character_ratio",
    "digit_character_ratio",
    "editorial_marker_count",
    "editorial_markers",
    "quality_review_flags",
    "language_variety_flags",
    "normalized_word_sha256",
    "fingerprint_anchor_count",
    "possible_existing_work_matches",
    "metadata_intra_gutenberg_duplicate_ids",
    "intra_gutenberg_exact_duplicate_ids",
    "intra_gutenberg_near_duplicate_metrics",
    "bibit_overlap_metrics",
    "current_corpus_overlap_metrics",
    "cross_corpus_duplicate_scope",
    "heldout_sonnet_overlap_metrics",
    "manual_review_resolution",
    "manual_review_rationale",
    "probe_decision",
)

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_PAGE_MARKER = re.compile(r"(?im)^\s*(?:pag(?:ina)?\.?|page)\s*\d+\s*$")
_DIALECT_MARKER = re.compile(
    r"\b(?:dialetto|vernacolo)\b|"
    r"\b(?:in|lingua|poesia)\s+(?:romanesco|milanese|napoletano|veneziano)\b",
    re.IGNORECASE,
)
_EDITORIAL_PATTERNS = {
    "editor_note": re.compile(
        r"\b(?:nota del(?:l['’])?editore|nota del trascrittore|transcriber['’]s note)\b",
        re.IGNORECASE,
    ),
    "errata": re.compile(r"\berrata(?:\s+corrige)?\b", re.IGNORECASE),
    "editor_preface": re.compile(
        r"\b(?:prefazione|introduzione) dell['’]editore\b", re.IGNORECASE
    ),
    "alphabetical_index": re.compile(r"\bindice alfabetico\b", re.IGNORECASE),
}
_ITALIAN_FUNCTION_WORDS = {
    "a",
    "che",
    "con",
    "da",
    "del",
    "della",
    "di",
    "e",
    "gli",
    "il",
    "in",
    "la",
    "le",
    "lo",
    "ma",
    "nel",
    "non",
    "per",
    "si",
    "un",
    "una",
}
_MASK_64 = (1 << 64) - 1
_ROLLING_BASE = 1_000_003
_SHINGLE_SIZE = 8

_MANUAL_REVIEW_RESOLUTIONS = {
    "15136": (
        "accept_standard_italian_context",
        "The phrase dialetto dell'Arno praises standard Tuscan in editorial prose; the poem is not dialectal.",
    ),
    "17440": (
        "extract_italian_parallel_text_only",
        "The libretto alternates Italian primary text with an English translation; retain only the Italian side.",
    ),
    "17834": (
        "extract_one_italian_primary_edition_only",
        "Remove the French bibliographic notice and retain one of the two duplicated Italian poem editions.",
    ),
    "33982": (
        "exclude_standard_italian_core_macaronic",
        "The primary work deliberately mixes Latin morphology with Italian and dialect vocabulary.",
    ),
    "46295": (
        "accept_standard_italian_context",
        "The dialect marker occurs in prefatory theatre history rather than identifying the play's language.",
    ),
    "46352": (
        "accept_standard_italian_context",
        "The dialect marker describes an actress in prefatory prose rather than the work's language.",
    ),
    "46898": (
        "accept_standardized_tuscan_text",
        "The editor explicitly says vernacular wording was rendered in an aulic form; retain the review evidence.",
    ),
    "47480": (
        "accept_dense_historical_dates",
        "The elevated digit ratio comes from legitimate chronology and dates, not OCR corruption.",
    ),
    "57549": (
        "accept_dense_historical_dates",
        "The elevated digit ratio comes from annal dates and day references, not OCR corruption.",
    ),
    "60641": (
        "accept_standard_italian_context",
        "The dialect marker refers to a character speaking Béarn dialect inside an Italian translation.",
    ),
    "60789": (
        "accept_dense_historical_dates_and_notes",
        "The elevated digit ratio comes from dates and numbered scholarly notes; note removal remains required.",
    ),
    "71399": (
        "accept_standard_tuscan_with_vocabulary_notes",
        "The preface mentions occasional obsolete or Tuscan vernacular vocabulary; the collection is core-compatible.",
    ),
}


@dataclass(frozen=True)
class GutenbergFullTextProbeConfig:
    repo_root: Path
    inventory_csv_path: Path
    cache_dir: Path
    output_csv_path: Path
    json_report_path: Path
    markdown_report_path: Path
    bibit_record_manifest_path: Path
    broader_sources_manifest_path: Path
    sonnet_manifest_path: Path
    request_delay_seconds: float = 1.0
    request_timeout_seconds: float = 60.0
    min_cleaned_characters: int = 1_000
    min_italian_function_word_ratio: float = 0.02
    sketch_size: int = 256
    anchor_mask: int = 1023
    near_duplicate_containment: float = 0.8
    heldout_containment: float = 0.8


@dataclass(frozen=True)
class TextFingerprint:
    normalized_word_sha256: str
    word_count: int
    shingle_count: int
    sketch: tuple[int, ...]
    anchors: tuple[int, ...]


@dataclass(frozen=True)
class TextReference:
    reference_id: str
    source_kind: str
    path: Path
    byte_start: int = 0
    byte_end: int | None = None

    def read_text(self) -> str:
        if self.byte_end is None:
            return self.path.read_text(encoding="utf-8")
        with self.path.open("rb") as handle:
            handle.seek(self.byte_start)
            payload = handle.read(self.byte_end - self.byte_start)
        return payload.decode("utf-8")


FetchText = Callable[..., FetchedGutenbergText]
Progress = Callable[[str], None]
Sleep = Callable[[float], None]


def fingerprint_text(
    text: str,
    *,
    sketch_size: int = 256,
    anchor_mask: int = 1023,
    watched_shingles: dict[int, tuple[str, ...]] | None = None,
) -> tuple[TextFingerprint, dict[str, set[int]]]:
    """Build a stable bounded fingerprint and optional contained-text hit sets."""

    words = _normalized_words(text)
    digest = hashlib.sha256()
    for word in words:
        digest.update(word.encode("utf-8"))
        digest.update(b"\0")

    heap: list[int] = []
    sketch_values: set[int] = set()
    anchors: set[int] = set()
    watched_hits: dict[str, set[int]] = defaultdict(set)
    shingle_count = max(0, len(words) - _SHINGLE_SIZE + 1)
    for shingle_hash in _rolling_shingle_hashes(words):
        if shingle_hash & anchor_mask == 0:
            anchors.add(shingle_hash)
        if watched_shingles and shingle_hash in watched_shingles:
            for reference_id in watched_shingles[shingle_hash]:
                watched_hits[reference_id].add(shingle_hash)
        if shingle_hash in sketch_values:
            continue
        if len(heap) < sketch_size:
            heapq.heappush(heap, -shingle_hash)
            sketch_values.add(shingle_hash)
        elif shingle_hash < -heap[0]:
            removed = -heapq.heapreplace(heap, -shingle_hash)
            sketch_values.remove(removed)
            sketch_values.add(shingle_hash)

    return (
        TextFingerprint(
            normalized_word_sha256=digest.hexdigest(),
            word_count=len(words),
            shingle_count=shingle_count,
            sketch=tuple(sorted(sketch_values)),
            anchors=tuple(sorted(anchors)),
        ),
        dict(watched_hits),
    )


def measure_word_shingle_containment(left: str, right: str) -> dict[str, Any]:
    """Measure directional and maximum normalized 8-word-shingle containment."""

    left_words = _normalized_words(left)
    right_words = _normalized_words(right)
    left_hashes = set(_rolling_shingle_hashes(left_words))
    right_hashes = set(_rolling_shingle_hashes(right_words))
    matching_count = len(left_hashes & right_hashes)
    left_containment = matching_count / len(left_hashes) if left_hashes else 0.0
    right_containment = matching_count / len(right_hashes) if right_hashes else 0.0
    return {
        "containment": max(left_containment, right_containment),
        "left_containment": left_containment,
        "right_containment": right_containment,
        "matching_shingles": matching_count,
        "left_unique_shingles": len(left_hashes),
        "right_unique_shingles": len(right_hashes),
        "denominator": min(len(left_hashes), len(right_hashes)),
    }


def run_gutenberg_fulltext_probe(
    config: GutenbergFullTextProbeConfig,
    *,
    fetch_text: FetchText = fetch_gutenberg_text,
    session: requests.Session | None = None,
    progress: Progress | None = None,
    sleep: Sleep = default_sleep,
) -> dict[str, Any]:
    """Probe every eligible record and produce non-activating deduplication evidence."""

    _validate_config(config)
    inventory = _read_csv(config.inventory_csv_path)
    candidates = sorted(
        (row for row in inventory if row["inventory_status"] in PROBE_STATUSES),
        key=lambda row: int(row["ebook_id"]),
    )
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    heldout_watch, heldout_denominators = _load_heldout_sonnet_watch(config)
    started = monotonic()
    last_download_at: float | None = None
    results: list[dict[str, Any]] = []
    fingerprints: dict[str, TextFingerprint] = {}
    heldout_hits: dict[str, dict[str, set[int]]] = {}

    for index, row in enumerate(candidates, start=1):
        cache_path = config.cache_dir / f"pg{row['ebook_id']}.txt"
        try:
            if cache_path.is_file():
                raw_text = cache_path.read_text(encoding="utf-8")
                fetched_url = row["plain_text_url"]
                cache_status = "hit"
            else:
                if last_download_at is not None and config.request_delay_seconds:
                    remaining = config.request_delay_seconds - (monotonic() - last_download_at)
                    if remaining > 0:
                        sleep(remaining)
                fetched = fetch_text(
                    row["ebook_id"],
                    session=session,
                    timeout=int(config.request_timeout_seconds),
                )
                last_download_at = monotonic()
                raw_text = fetched.text
                fetched_url = fetched.url
                cache_path.write_text(raw_text, encoding="utf-8")
                cache_status = "downloaded"
            cleaned = strip_gutenberg_boilerplate(raw_text)
            fingerprint, watched_hits = fingerprint_text(
                cleaned,
                sketch_size=config.sketch_size,
                anchor_mask=config.anchor_mask,
                watched_shingles=heldout_watch,
            )
            result = _inspect_candidate(row, raw_text, cleaned, fingerprint, config)
            result["fetched_url"] = fetched_url
            result["cache_status"] = cache_status
            fingerprints[row["ebook_id"]] = fingerprint
            heldout_hits[row["ebook_id"]] = watched_hits
        except Exception as error:
            result = _error_result(row, error)
            result["cache_status"] = "error"
        results.append(result)
        _progress_item(progress, "source", index, len(candidates), started, result)

    successful = {row["ebook_id"]: row for row in results if row["probe_status"] != "error"}
    _attach_heldout_overlaps(
        successful,
        heldout_hits,
        heldout_denominators,
        threshold=config.heldout_containment,
    )
    intra_pairs = _attach_intra_gutenberg_duplicates(
        config,
        successful,
        fingerprints,
        progress=progress,
    )

    references = _load_cross_corpus_references(config)
    reference_fingerprints = _fingerprint_references(
        references,
        config,
        progress=progress,
    )
    cross_pairs = _attach_cross_corpus_duplicates(
        config,
        successful,
        fingerprints,
        references,
        reference_fingerprints,
        progress=progress,
    )
    _finalize_probe_decisions(results)
    _write_csv(config.output_csv_path, PROBE_FIELDS, results)
    report = _build_report(
        config,
        results=results,
        intra_pairs=intra_pairs,
        cross_pairs=cross_pairs,
        reference_count=len(references),
        heldout_count=len(heldout_denominators),
    )
    _write_json(config.json_report_path, report)
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(
        render_gutenberg_fulltext_probe_markdown(report),
        encoding="utf-8",
    )
    return report


def _inspect_candidate(
    row: dict[str, str],
    raw_text: str,
    cleaned: str,
    fingerprint: TextFingerprint,
    config: GutenbergFullTextProbeConfig,
) -> dict[str, Any]:
    words = _WORD.findall(cleaned.casefold())
    function_ratio = (
        sum(word in _ITALIAN_FUNCTION_WORDS for word in words) / len(words)
        if words
        else 0.0
    )
    alpha_ratio = sum(character.isalpha() for character in cleaned) / max(1, len(cleaned))
    digit_ratio = sum(character.isdigit() for character in cleaned) / max(1, len(cleaned))
    replacement_count = cleaned.count("\ufffd")
    lines = [line for line in cleaned.splitlines() if line.strip()]
    hyphenated_line_ends = sum(line.rstrip().endswith("-") for line in lines)
    page_markers = len(_PAGE_MARKER.findall(cleaned))
    editorial_counts = {
        name: len(pattern.findall(cleaned))
        for name, pattern in _EDITORIAL_PATTERNS.items()
    }
    editorial_counts = {name: count for name, count in editorial_counts.items() if count}
    ocr_flags = []
    if len(cleaned) < config.min_cleaned_characters:
        ocr_flags.append("too_short")
    if function_ratio < config.min_italian_function_word_ratio:
        ocr_flags.append("low_italian_function_word_ratio")
    if alpha_ratio < 0.55:
        ocr_flags.append("low_alphabetic_character_ratio")
    if replacement_count:
        ocr_flags.append("replacement_characters")
    if digit_ratio > 0.03:
        ocr_flags.append("high_digit_character_ratio")
    if page_markers > max(10, len(lines) * 0.01):
        ocr_flags.append("page_markers")
    if hyphenated_line_ends > max(100, len(lines) * 0.08):
        ocr_flags.append("possible_ocr_line_hyphenation")
    language_flags = []
    if _DIALECT_MARKER.search(" ".join((row["title"], cleaned[:5_000]))):
        language_flags.append("review_language_variety_marker")
    return {
        "ebook_id": row["ebook_id"],
        "title": row["title"],
        "authors": row["authors"],
        "preliminary_role": row["preliminary_role"],
        "period_bucket": row["period_bucket"],
        "inventory_status": row["inventory_status"],
        "landing_page_url": row["landing_page_url"],
        "fetched_url": "",
        "cache_status": "",
        "probe_status": "quality_pass" if not ocr_flags else "review_quality",
        "error": "",
        "raw_character_count": len(raw_text),
        "cleaned_character_count": len(cleaned),
        "cleaned_word_count": len(words),
        "nonempty_line_count": len(lines),
        "replacement_character_count": replacement_count,
        "italian_function_word_ratio": round(function_ratio, 6),
        "alphabetic_character_ratio": round(alpha_ratio, 6),
        "digit_character_ratio": round(digit_ratio, 6),
        "editorial_marker_count": sum(editorial_counts.values()),
        "editorial_markers": ";".join(
            f"{name}:{count}" for name, count in sorted(editorial_counts.items())
        ),
        "quality_review_flags": ";".join(ocr_flags),
        "language_variety_flags": ";".join(language_flags),
        "normalized_word_sha256": fingerprint.normalized_word_sha256,
        "fingerprint_anchor_count": len(fingerprint.anchors),
        "possible_existing_work_matches": row["possible_existing_work_matches"],
        "metadata_intra_gutenberg_duplicate_ids": row[
            "intra_gutenberg_duplicate_ids"
        ],
        "intra_gutenberg_exact_duplicate_ids": "",
        "intra_gutenberg_near_duplicate_metrics": "",
        "bibit_overlap_metrics": "",
        "current_corpus_overlap_metrics": "",
        "cross_corpus_duplicate_scope": "",
        "heldout_sonnet_overlap_metrics": "",
        "manual_review_resolution": "",
        "manual_review_rationale": "",
        "probe_decision": "",
    }


def _error_result(row: dict[str, str], error: Exception) -> dict[str, Any]:
    result = {field: "" for field in PROBE_FIELDS}
    result.update(
        {
            "ebook_id": row["ebook_id"],
            "title": row["title"],
            "authors": row["authors"],
            "preliminary_role": row["preliminary_role"],
            "period_bucket": row["period_bucket"],
            "inventory_status": row["inventory_status"],
            "landing_page_url": row["landing_page_url"],
            "probe_status": "error",
            "error": f"{type(error).__name__}: {error}",
            "metadata_intra_gutenberg_duplicate_ids": row[
                "intra_gutenberg_duplicate_ids"
            ],
            "probe_decision": "blocked_fetch_or_parse_error",
        }
    )
    return result


def _load_heldout_sonnet_watch(
    config: GutenbergFullTextProbeConfig,
) -> tuple[dict[int, tuple[str, ...]], dict[str, int]]:
    watch: dict[int, list[str]] = defaultdict(list)
    denominators: dict[str, int] = {}
    for row in _read_csv(config.sonnet_manifest_path):
        if row["split_expanded_with_petrarch"] not in {"validation", "test"}:
            continue
        text_path = config.repo_root / row["clean_text_path"]
        hashes = set(_rolling_shingle_hashes(_normalized_words(text_path.read_text(encoding="utf-8"))))
        if not hashes:
            continue
        poem_id = row["poem_id"]
        denominators[poem_id] = len(hashes)
        for value in hashes:
            watch[value].append(poem_id)
    return {value: tuple(ids) for value, ids in watch.items()}, denominators


def _attach_heldout_overlaps(
    results: dict[str, dict[str, Any]],
    all_hits: dict[str, dict[str, set[int]]],
    denominators: dict[str, int],
    *,
    threshold: float,
) -> None:
    for ebook_id, hits in all_hits.items():
        metrics = []
        for poem_id, values in hits.items():
            containment = len(values) / denominators[poem_id]
            if containment >= threshold:
                metrics.append(f"{poem_id}|containment={containment:.6f}")
        results[ebook_id]["heldout_sonnet_overlap_metrics"] = ";".join(sorted(metrics))


def _attach_intra_gutenberg_duplicates(
    config: GutenbergFullTextProbeConfig,
    results: dict[str, dict[str, Any]],
    fingerprints: dict[str, TextFingerprint],
    *,
    progress: Progress | None,
) -> list[dict[str, Any]]:
    exact_groups: dict[str, list[str]] = defaultdict(list)
    for ebook_id, fingerprint in fingerprints.items():
        exact_groups[fingerprint.normalized_word_sha256].append(ebook_id)
    exact_pairs: set[tuple[str, str]] = set()
    for ids in exact_groups.values():
        if len(ids) < 2:
            continue
        ordered = sorted(ids, key=int)
        for ebook_id in ordered:
            others = [value for value in ordered if value != ebook_id]
            results[ebook_id]["intra_gutenberg_exact_duplicate_ids"] = ";".join(others)
        exact_pairs.update(tuple(sorted(pair, key=int)) for pair in combinations(ordered, 2))

    candidates = _discover_candidate_pairs(fingerprints)
    for row in results.values():
        for other_id in row["metadata_intra_gutenberg_duplicate_ids"].split(";"):
            if other_id in results and other_id != row["ebook_id"]:
                candidates.add(tuple(sorted((row["ebook_id"], other_id), key=int)))
    candidates.difference_update(exact_pairs)
    pairs = []
    started = monotonic()
    for index, (left_id, right_id) in enumerate(sorted(candidates, key=_numeric_pair), start=1):
        metric = measure_word_shingle_containment(
            _read_cleaned_candidate(config, left_id),
            _read_cleaned_candidate(config, right_id),
        )
        containment = metric["containment"]
        if containment >= config.near_duplicate_containment:
            record = {
                "left_id": left_id,
                "right_id": right_id,
                "containment": round(containment, 6),
                "matching_shingles": metric["matching_shingles"],
                "denominator": metric["denominator"],
            }
            pairs.append(record)
            _append_metric(
                results[left_id],
                "intra_gutenberg_near_duplicate_metrics",
                f"pg{right_id}|containment={containment:.6f}",
            )
            _append_metric(
                results[right_id],
                "intra_gutenberg_near_duplicate_metrics",
                f"pg{left_id}|containment={containment:.6f}",
            )
        _progress_phase(progress, "intra-dedup", index, len(candidates), started)
    return pairs


def _load_cross_corpus_references(
    config: GutenbergFullTextProbeConfig,
) -> dict[str, TextReference]:
    references: dict[str, TextReference] = {}
    for row in _read_csv(config.bibit_record_manifest_path):
        if row["artifact_status"] != "text_materialized" or not row["shard_path"]:
            continue
        reference_id = f"bibit:{row['object_id']}"
        references[reference_id] = TextReference(
            reference_id=reference_id,
            source_kind="bibit",
            path=config.repo_root / row["shard_path"],
            byte_start=int(row["byte_start"]),
            byte_end=int(row["byte_end"]),
        )
    for row in _read_csv(config.broader_sources_manifest_path):
        relative_path = row["expected_clean_text_path"]
        if not relative_path:
            continue
        path = config.repo_root / relative_path
        if not path.is_file():
            continue
        reference_id = f"current:{row['source_id']}"
        references[reference_id] = TextReference(
            reference_id=reference_id,
            source_kind="current_corpus",
            path=path,
        )
    return references


def _fingerprint_references(
    references: dict[str, TextReference],
    config: GutenbergFullTextProbeConfig,
    *,
    progress: Progress | None,
) -> dict[str, TextFingerprint]:
    fingerprints = {}
    started = monotonic()
    items = sorted(references.items())
    for index, (reference_id, reference) in enumerate(items, start=1):
        fingerprints[reference_id], _ = fingerprint_text(
            reference.read_text(),
            sketch_size=config.sketch_size,
            anchor_mask=config.anchor_mask,
        )
        if index == 1 or index == len(items) or index % 50 == 0:
            _progress_phase(progress, "reference-index", index, len(items), started)
    return fingerprints


def _attach_cross_corpus_duplicates(
    config: GutenbergFullTextProbeConfig,
    results: dict[str, dict[str, Any]],
    fingerprints: dict[str, TextFingerprint],
    references: dict[str, TextReference],
    reference_fingerprints: dict[str, TextFingerprint],
    *,
    progress: Progress | None,
) -> list[dict[str, Any]]:
    candidates = _discover_cross_candidates(fingerprints, reference_fingerprints)
    for ebook_id, row in results.items():
        for match in row["possible_existing_work_matches"].split(";"):
            if match in references:
                candidates.add((ebook_id, match))
    exact_map: dict[str, list[str]] = defaultdict(list)
    for reference_id, fingerprint in reference_fingerprints.items():
        exact_map[fingerprint.normalized_word_sha256].append(reference_id)
    for ebook_id, fingerprint in fingerprints.items():
        for reference_id in exact_map.get(fingerprint.normalized_word_sha256, []):
            candidates.add((ebook_id, reference_id))

    pairs = []
    started = monotonic()
    ordered = sorted(candidates, key=lambda pair: (int(pair[0]), pair[1]))
    for index, (ebook_id, reference_id) in enumerate(ordered, start=1):
        metric = measure_word_shingle_containment(
            _read_cleaned_candidate(config, ebook_id),
            references[reference_id].read_text(),
        )
        containment = metric["containment"]
        if containment >= config.near_duplicate_containment:
            candidate_containment = metric["left_containment"]
            reference_containment = metric["right_containment"]
            scope = (
                "candidate_covered"
                if candidate_containment >= config.near_duplicate_containment
                else "embedded_reference_only"
            )
            exact = (
                fingerprints[ebook_id].normalized_word_sha256
                == reference_fingerprints[reference_id].normalized_word_sha256
            )
            record = {
                "ebook_id": ebook_id,
                "reference_id": reference_id,
                "source_kind": references[reference_id].source_kind,
                "containment": round(containment, 6),
                "candidate_containment": round(candidate_containment, 6),
                "reference_containment": round(reference_containment, 6),
                "duplicate_scope": scope,
                "exact_normalized_text": exact,
                "matching_shingles": metric["matching_shingles"],
                "denominator": metric["denominator"],
            }
            pairs.append(record)
            field = (
                "bibit_overlap_metrics"
                if references[reference_id].source_kind == "bibit"
                else "current_corpus_overlap_metrics"
            )
            _append_metric(
                results[ebook_id],
                field,
                f"{reference_id}|candidate_containment={candidate_containment:.6f}|"
                f"reference_containment={reference_containment:.6f}|"
                f"exact={str(exact).lower()}",
            )
            scopes = set(results[ebook_id]["cross_corpus_duplicate_scope"].split(";"))
            scopes.discard("")
            scopes.add(scope)
            results[ebook_id]["cross_corpus_duplicate_scope"] = ";".join(sorted(scopes))
        _progress_phase(progress, "cross-dedup", index, len(ordered), started)
    return pairs


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
                collisions.update(tuple(sorted(pair, key=int)) for pair in combinations(ids, 2))
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


def _finalize_probe_decisions(results: list[dict[str, Any]]) -> None:
    for row in results:
        if row["probe_status"] == "error":
            continue
        manual_resolution, rationale = _MANUAL_REVIEW_RESOLUTIONS.get(
            row["ebook_id"],
            ("", ""),
        )
        row["manual_review_resolution"] = manual_resolution
        row["manual_review_rationale"] = rationale
        if "candidate_covered" in row["cross_corpus_duplicate_scope"]:
            decision = "exclude_cross_corpus_duplicate_candidate"
        elif row["heldout_sonnet_overlap_metrics"]:
            decision = "quarantine_heldout_sonnet_segment_before_activation"
        elif "embedded_reference_only" in row["cross_corpus_duplicate_scope"]:
            decision = "quarantine_embedded_duplicate_segments_before_activation"
        elif manual_resolution == "exclude_standard_italian_core_macaronic":
            decision = "exclude_standard_italian_core_language_composition"
        elif manual_resolution in {
            "extract_italian_parallel_text_only",
            "extract_one_italian_primary_edition_only",
        }:
            decision = "source_specific_language_extraction_before_activation"
        elif manual_resolution:
            decision = "quality_pass_pending_editorial_activation_review"
        elif row["language_variety_flags"]:
            decision = "review_language_variety"
        elif row["probe_status"] == "review_quality":
            decision = "review_text_quality"
        elif row["intra_gutenberg_exact_duplicate_ids"]:
            decision = "resolve_intra_gutenberg_exact_duplicate"
        elif row["intra_gutenberg_near_duplicate_metrics"]:
            decision = "resolve_intra_gutenberg_near_duplicate"
        else:
            decision = "quality_pass_pending_editorial_activation_review"
        row["probe_decision"] = decision


def _build_report(
    config: GutenbergFullTextProbeConfig,
    *,
    results: list[dict[str, Any]],
    intra_pairs: list[dict[str, Any]],
    cross_pairs: list[dict[str, Any]],
    reference_count: int,
    heldout_count: int,
) -> dict[str, Any]:
    status_counts = Counter(row["probe_status"] for row in results)
    decision_counts = Counter(row["probe_decision"] for row in results)
    quality_flags = Counter(
        flag
        for row in results
        for flag in str(row["quality_review_flags"]).split(";")
        if flag
    )
    manual_resolutions = Counter(
        row["manual_review_resolution"]
        for row in results
        if row["manual_review_resolution"]
    )
    role_summary = {}
    exact_groups = sorted(
        {
            tuple(
                sorted(
                    (row["ebook_id"], *row["intra_gutenberg_exact_duplicate_ids"].split(";")),
                    key=int,
                )
            )
            for row in results
            if row["intra_gutenberg_exact_duplicate_ids"]
        },
        key=lambda group: tuple(map(int, group)),
    )
    for role in sorted({row["preliminary_role"] for row in results}):
        rows = [row for row in results if row["preliminary_role"] == role]
        role_summary[role] = {
            "record_count": len(rows),
            "successful_record_count": sum(row["probe_status"] != "error" for row in rows),
            "cleaned_character_count": sum(int(row["cleaned_character_count"] or 0) for row in rows),
            "quality_pass_count": sum(row["probe_status"] == "quality_pass" for row in rows),
        }
    return {
        "probe_version": "project_gutenberg_fulltext_probe_v1",
        "created_at_utc": _utc_now(),
        "candidate_count": len(results),
        "probe_status_counts": dict(sorted(status_counts.items())),
        "probe_decision_counts": dict(sorted(decision_counts.items())),
        "quality_flag_counts": dict(sorted(quality_flags.items())),
        "manual_review_resolution_counts": dict(sorted(manual_resolutions.items())),
        "cleaned_character_count": sum(int(row["cleaned_character_count"] or 0) for row in results),
        "cleaned_word_count": sum(int(row["cleaned_word_count"] or 0) for row in results),
        "role_summary": role_summary,
        "intra_gutenberg_exact_duplicate_groups": [list(group) for group in exact_groups],
        "intra_gutenberg_near_duplicate_pairs": intra_pairs,
        "cross_corpus_duplicate_pairs": cross_pairs,
        "cross_corpus_reference_count": reference_count,
        "heldout_sonnet_reference_count": heldout_count,
        "outputs": {
            "probe_csv_path": _portable(config.output_csv_path, config.repo_root),
            "probe_csv_sha256": _sha256_file(config.output_csv_path),
            "json_report_path": _portable(config.json_report_path, config.repo_root),
            "markdown_report_path": _portable(config.markdown_report_path, config.repo_root),
            "local_cache_path": _portable(config.cache_dir, config.repo_root),
        },
        "policy": {
            "activation_authorized": False,
            "v7_split_authorized": False,
            "mixture_weight_authorized": False,
            "normalized_8word_shingle_near_duplicate_threshold": config.near_duplicate_containment,
            "heldout_sonnet_containment_threshold": config.heldout_containment,
            "raw_text_cache_is_machine_local": True,
        },
    }


def render_gutenberg_fulltext_probe_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Complete Italian Project Gutenberg Full-Text Probe",
        "",
        "## Result",
        "",
        f"Probed {report['candidate_count']:,} metadata-eligible Italian records.",
        "",
        "| Probe status | Records |",
        "| --- | ---: |",
    ]
    for status, count in report["probe_status_counts"].items():
        lines.append(f"| `{status}` | {count:,} |")
    lines.extend(["", "## Decisions", "", "| Decision | Records |", "| --- | ---: |"])
    for decision, count in report["probe_decision_counts"].items():
        lines.append(f"| `{decision}` | {count:,} |")
    lines.extend(
        [
            "",
            "## Manual Review Resolutions",
            "",
            "| Resolution | Records |",
            "| --- | ---: |",
        ]
    )
    for resolution, count in report["manual_review_resolution_counts"].items():
        lines.append(f"| `{resolution}` | {count:,} |")
    lines.extend(
        [
            "",
            "## Role Volume",
            "",
            "| Role | Records | Successful | Quality pass | Cleaned chars |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for role, row in report["role_summary"].items():
        lines.append(
            f"| `{role}` | {row['record_count']:,} | {row['successful_record_count']:,} | "
            f"{row['quality_pass_count']:,} | {row['cleaned_character_count']:,} |"
        )
    lines.extend(
        [
            "",
            f"Total cleaned volume: {report['cleaned_character_count']:,} characters and "
            f"{report['cleaned_word_count']:,} words.",
            "",
            "## Duplicate Evidence",
            "",
            f"- Intra-Gutenberg normalized exact-duplicate groups: {len(report['intra_gutenberg_exact_duplicate_groups']):,}.",
            f"- Intra-Gutenberg near-duplicate pairs: {len(report['intra_gutenberg_near_duplicate_pairs']):,}.",
            f"- Cross-corpus duplicate pairs: {len(report['cross_corpus_duplicate_pairs']):,}.",
            f"- Indexed existing-corpus references: {report['cross_corpus_reference_count']:,}.",
            f"- Protected V6 validation/test sonnets: {report['heldout_sonnet_reference_count']:,}.",
            "",
            "## Boundaries",
            "",
            "- This probe records evidence; it activates no Gutenberg source.",
            "- Raw downloaded texts remain machine-local.",
            "- Editorial decisions, canonical-edition selection, V7 splits, and mixture weights remain pending.",
            "",
        ]
    )
    return "\n".join(lines)


def _rolling_shingle_hashes(words: list[str]) -> Iterable[int]:
    if len(words) < _SHINGLE_SIZE:
        return
    values = [_stable_word_hash(word) for word in words]
    power = pow(_ROLLING_BASE, _SHINGLE_SIZE - 1, 1 << 64)
    rolling = 0
    for value in values[:_SHINGLE_SIZE]:
        rolling = (rolling * _ROLLING_BASE + value) & _MASK_64
    yield rolling
    for index in range(_SHINGLE_SIZE, len(values)):
        rolling = (rolling - values[index - _SHINGLE_SIZE] * power) & _MASK_64
        rolling = (rolling * _ROLLING_BASE + values[index]) & _MASK_64
        yield rolling


@lru_cache(maxsize=200_000)
def _stable_word_hash(word: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest(),
        "big",
    )


def _normalized_words(text: str) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return _WORD.findall(without_marks)


def _read_cleaned_candidate(config: GutenbergFullTextProbeConfig, ebook_id: str) -> str:
    raw = (config.cache_dir / f"pg{ebook_id}.txt").read_text(encoding="utf-8")
    return strip_gutenberg_boilerplate(raw)


def _append_metric(row: dict[str, Any], field: str, metric: str) -> None:
    row[field] = ";".join(value for value in (row[field], metric) if value)


def _numeric_pair(pair: tuple[str, str]) -> tuple[int, int]:
    return int(pair[0]), int(pair[1])


def _progress_item(
    progress: Progress | None,
    label: str,
    index: int,
    total: int,
    started: float,
    result: dict[str, Any],
) -> None:
    elapsed = monotonic() - started
    eta = elapsed / index * (total - index)
    _report(
        progress,
        f"{label} {index:,}/{total:,} ({index / total:.1%}) id={result['ebook_id']} "
        f"status={result['probe_status']} cache={result['cache_status']} "
        f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
    )


def _progress_phase(
    progress: Progress | None,
    label: str,
    index: int,
    total: int,
    started: float,
) -> None:
    if not total or (index != 1 and index != total and index % 25):
        return
    elapsed = monotonic() - started
    eta = elapsed / index * (total - index)
    _report(
        progress,
        f"{label} {index:,}/{total:,} ({index / total:.1%}) "
        f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
    )


def _validate_config(config: GutenbergFullTextProbeConfig) -> None:
    if config.request_delay_seconds < 0:
        raise ValueError("request_delay_seconds cannot be negative")
    if config.request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive")
    if config.sketch_size <= 0:
        raise ValueError("sketch_size must be positive")
    if config.anchor_mask <= 0 or config.anchor_mask & (config.anchor_mask + 1):
        raise ValueError("anchor_mask must be a positive 2^n - 1 value")
    for value in (config.near_duplicate_containment, config.heldout_containment):
        if not 0 <= value <= 1:
            raise ValueError("containment thresholds must be between zero and one")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _report(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)
