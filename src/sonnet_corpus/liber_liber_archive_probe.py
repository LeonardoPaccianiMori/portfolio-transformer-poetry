"""Bounded full-text probe for checkpoint-5A Liber Liber candidates."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from time import monotonic, sleep as default_sleep
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

from .gutenberg_fulltext_probe import (
    TextFingerprint,
    TextReference,
    _normalized_words,
    _rolling_shingle_hashes,
    fingerprint_text,
    measure_word_shingle_containment,
)
from .liber_liber import (
    LIBER_LIBER_USER_AGENT,
    discover_archive_url,
    extract_odt_text,
    extract_txt_zip_text,
    strip_liber_liber_boilerplate,
)


PROBE_FIELDS = (
    "record_id", "wordpress_page_id", "title", "author", "landing_page_url",
    "period_bucket", "preliminary_role", "metadata_decision", "license_url",
    "archive_format", "download_page_url", "archive_url", "cache_status",
    "archive_cache_path", "cleaned_cache_path", "archive_sha256", "cleaned_sha256",
    "raw_byte_count", "cleaned_character_count", "cleaned_word_count",
    "nonempty_line_count", "replacement_character_count",
    "italian_function_word_ratio", "alphabetic_character_ratio",
    "digit_character_ratio", "editorial_marker_count", "editorial_markers",
    "quality_review_flags", "language_variety_flags", "normalized_word_sha256",
    "fingerprint_anchor_count", "internal_exact_duplicate_ids",
    "internal_near_duplicate_metrics", "cross_corpus_overlap_metrics",
    "cross_corpus_duplicate_scope", "protected_v6_overlap_metrics",
    "manual_review_resolution", "manual_review_rationale", "probe_status",
    "probe_decision", "activation_status", "error",
)

REVIEW_FIELDS = (
    "record_id", "title", "quality_review_flags", "language_variety_flags",
    "manual_review_resolution", "manual_review_rationale",
)

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_PAGE_MARKER = re.compile(r"(?im)^\s*(?:pag(?:ina)?\.?|page)\s*\d+\s*$")
_DIALECT_MARKER = re.compile(
    r"\b(?:dialetto|vernacolo)\b|"
    r"\b(?:in|lingua|poesia|versi)\s+(?:romanesco|milanese|napoletano|"
    r"veneziano|veneto|siciliano|sardo|bolognese|piemontese)\b",
    re.IGNORECASE,
)
_EDITORIAL_PATTERNS = {
    "editor_note": re.compile(
        r"\b(?:nota del(?:l['’])?editore|nota del trascrittore|"
        r"transcriber['’]s note)\b", re.IGNORECASE,
    ),
    "errata": re.compile(r"\berrata(?:\s+corrige)?\b", re.IGNORECASE),
    "editor_preface": re.compile(
        r"\b(?:prefazione|introduzione) dell['’]editore\b", re.IGNORECASE,
    ),
    "alphabetical_index": re.compile(r"\bindice alfabetico\b", re.IGNORECASE),
    "edition_notes": re.compile(r"\bnote? di edizione\b", re.IGNORECASE),
    "liber_liber_wrapper": re.compile(
        r"\b(?:progetto manuzio|liberliber\.it|liber liber)\b", re.IGNORECASE,
    ),
}
_ITALIAN_FUNCTION_WORDS = {
    "a", "che", "con", "da", "del", "della", "di", "e", "gli", "il",
    "in", "la", "le", "lo", "ma", "nel", "non", "per", "si", "un", "una",
}

Progress = Callable[[str], None]
Sleep = Callable[[float], None]


@dataclass(frozen=True)
class LiberLiberArchiveProbeConfig:
    repo_root: Path
    inventory_path: Path
    cache_dir: Path
    output_csv_path: Path
    review_csv_path: Path
    json_report_path: Path
    markdown_report_path: Path
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
    expected_candidate_count: int = 129
    expected_conditioned_count: int = 151
    request_delay_seconds: float = 0.25
    request_timeout_seconds: float = 60.0
    min_cleaned_characters: int = 1_000
    min_italian_function_word_ratio: float = 0.02
    sketch_size: int = 256
    anchor_mask: int = 1023
    near_duplicate_containment: float = 0.8
    protected_containment: float = 0.8
    require_review_resolutions: bool = True


class _Acquirer:
    def __init__(
        self,
        config: LiberLiberArchiveProbeConfig,
        *,
        session: requests.Session | None,
        sleep: Sleep,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": LIBER_LIBER_USER_AGENT})
        self.sleep = sleep
        self.last_request_at: float | None = None

    def acquire(self, row: dict[str, str]) -> tuple[str, dict[str, Any]]:
        key = row["record_id"].replace(":", "_")
        metadata_path = self.config.cache_dir / "metadata" / f"{key}.json"
        text_path = self.config.cache_dir / "texts" / f"{key}.txt"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("record_id") != row["record_id"]:
                raise ValueError(f"cache record mismatch for {row['record_id']}")
            archive_path = self.config.repo_root / metadata["archive_cache_path"]
            if not archive_path.is_file() or _sha256_file(archive_path) != metadata["archive_sha256"]:
                raise ValueError(f"cached archive hash mismatch for {row['record_id']}")
            if metadata.get("cleaning_version") == 2 and text_path.is_file():
                text = text_path.read_text(encoding="utf-8")
                if _sha256_bytes(text.encode("utf-8")) != metadata["cleaned_sha256"]:
                    raise ValueError(f"cached cleaned-text hash mismatch for {row['record_id']}")
                return text, metadata | {"cache_status": "hit"}
            payload = archive_path.read_bytes()
            extracted = (
                extract_txt_zip_text(payload)
                if metadata["archive_format"] == "txt_zip"
                else extract_odt_text(payload)
            )
            cleaned = strip_liber_liber_boilerplate(extracted, title=row["title"])
            _write_text_atomic(text_path, cleaned)
            metadata.update({
                "cleaning_version": 2,
                "cleaned_sha256": _sha256_bytes(cleaned.encode("utf-8")),
            })
            _write_json_atomic(metadata_path, metadata)
            return cleaned, metadata | {"cache_status": "recleaned"}

        errors = []
        for archive_format, download_url in _download_candidates(row):
            try:
                download_response = self._get(download_url)
                archive_url = discover_archive_url(
                    download_response.text,
                    base_url=getattr(download_response, "url", download_url),
                    archive_format=archive_format,
                )
                archive_response = self._get(archive_url)
                payload = archive_response.content
                extracted = (
                    extract_txt_zip_text(payload)
                    if archive_format == "txt_zip"
                    else extract_odt_text(payload)
                )
                cleaned = strip_liber_liber_boilerplate(extracted, title=row["title"])
                if not cleaned.strip():
                    raise ValueError("extracted text is empty after wrapper removal")
            except (OSError, ValueError, requests.RequestException) as error:
                errors.append(f"{archive_format}: {type(error).__name__}: {error}")
                continue

            suffix = ".zip" if archive_format == "txt_zip" else ".odt"
            archive_path = self.config.cache_dir / "archives" / f"{key}{suffix}"
            _write_bytes_atomic(archive_path, payload)
            _write_text_atomic(text_path, cleaned)
            metadata = {
                "cache_version": "liber_liber_fulltext_probe_v1",
                "cleaning_version": 2,
                "record_id": row["record_id"],
                "fetched_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
                "archive_format": archive_format,
                "download_page_url": download_url,
                "archive_url": archive_url,
                "archive_cache_path": _portable(archive_path, self.config.repo_root),
                "cleaned_cache_path": _portable(text_path, self.config.repo_root),
                "raw_byte_count": len(payload),
                "archive_sha256": _sha256_bytes(payload),
                "cleaned_sha256": _sha256_bytes(cleaned.encode("utf-8")),
            }
            _write_json_atomic(metadata_path, metadata)
            return cleaned, metadata | {"cache_status": "downloaded"}
        raise FileNotFoundError(
            f"no supported archive succeeded for {row['record_id']}: " + "; ".join(errors)
        )

    def _get(self, url: str) -> requests.Response:
        if self.last_request_at is not None and self.config.request_delay_seconds:
            remaining = self.config.request_delay_seconds - (monotonic() - self.last_request_at)
            if remaining > 0:
                self.sleep(remaining)
        response = self.session.get(url, timeout=self.config.request_timeout_seconds)
        self.last_request_at = monotonic()
        response.raise_for_status()
        return response


def run_liber_liber_archive_probe(
    config: LiberLiberArchiveProbeConfig,
    *,
    session: requests.Session | None = None,
    progress: Progress | None = None,
    sleep: Sleep = default_sleep,
) -> dict[str, Any]:
    """Acquire and probe exactly the checkpoint-5A inactive eligible queue."""

    _validate_config(config)
    inventory = _read_csv(config.inventory_path)
    candidates = [
        row for row in inventory
        if row["composition_decision"] == "eligible_fulltext_probe_inactive"
    ]
    conditioned = [
        row for row in inventory
        if row["composition_decision"] == "conditioned_language_candidate_inactive"
    ]
    if len(candidates) != config.expected_candidate_count:
        raise ValueError(
            f"expected {config.expected_candidate_count} eligible records, found {len(candidates)}"
        )
    if len(conditioned) != config.expected_conditioned_count:
        raise ValueError(
            f"expected {config.expected_conditioned_count} conditioned records, found {len(conditioned)}"
        )
    candidate_ids = {row["record_id"] for row in candidates}
    if candidate_ids & {row["record_id"] for row in conditioned}:
        raise ValueError("conditioned records leaked into the standard probe queue")

    config.cache_dir.mkdir(parents=True, exist_ok=True)
    acquirer = _Acquirer(config, session=session, sleep=sleep)
    protected_watch, protected_denominators = _load_protected_watch(config)
    results: list[dict[str, Any]] = []
    fingerprints: dict[str, TextFingerprint] = {}
    text_paths: dict[str, Path] = {}
    protected_hits: dict[str, dict[str, set[int]]] = {}
    started = monotonic()
    for index, row in enumerate(sorted(candidates, key=lambda item: int(item["wordpress_page_id"])), 1):
        try:
            text, cache = acquirer.acquire(row)
            fingerprint, hits = fingerprint_text(
                text,
                sketch_size=config.sketch_size,
                anchor_mask=config.anchor_mask,
                watched_shingles=protected_watch,
            )
            result = _inspect(row, text, cache, fingerprint, config)
            fingerprints[row["record_id"]] = fingerprint
            text_paths[row["record_id"]] = config.repo_root / cache["cleaned_cache_path"]
            protected_hits[row["record_id"]] = hits
        except Exception as error:
            result = _error_row(row, error)
        results.append(result)
        _emit_item(progress, index, len(candidates), started, result)

    by_id = {row["record_id"]: row for row in results if row["probe_status"] != "error"}
    _attach_protected(by_id, protected_hits, protected_denominators, config.protected_containment)
    internal_pairs = _attach_internal(by_id, fingerprints, text_paths, config, progress)
    references = _load_references(config)
    reference_fingerprints = _fingerprint_references(references, config, progress)
    cross_pairs = _attach_cross(
        by_id, fingerprints, text_paths, references, reference_fingerprints, config, progress
    )
    review_rows = _apply_reviews(config, results)
    _finalize_decisions(results)
    _write_csv(config.output_csv_path, PROBE_FIELDS, results)
    report = _build_report(
        config, results, conditioned, references, internal_pairs, cross_pairs, review_rows
    )
    _write_json(config.json_report_path, report)
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _inspect(
    row: dict[str, str],
    text: str,
    cache: dict[str, Any],
    fingerprint: TextFingerprint,
    config: LiberLiberArchiveProbeConfig,
) -> dict[str, Any]:
    words = _WORD.findall(text.casefold())
    function_ratio = sum(word in _ITALIAN_FUNCTION_WORDS for word in words) / max(1, len(words))
    alpha_ratio = sum(character.isalpha() for character in text) / max(1, len(text))
    digit_ratio = sum(character.isdigit() for character in text) / max(1, len(text))
    lines = [line for line in text.splitlines() if line.strip()]
    replacement_count = text.count("\ufffd")
    page_markers = len(_PAGE_MARKER.findall(text))
    hyphenated = sum(line.rstrip().endswith("-") for line in lines)
    editorial = {
        name: len(pattern.findall(text))
        for name, pattern in _EDITORIAL_PATTERNS.items()
    }
    editorial = {name: count for name, count in editorial.items() if count}
    flags = []
    if len(text) < config.min_cleaned_characters:
        flags.append("too_short")
    if function_ratio < config.min_italian_function_word_ratio:
        flags.append("low_italian_function_word_ratio")
    if alpha_ratio < 0.55:
        flags.append("low_alphabetic_character_ratio")
    if replacement_count:
        flags.append("replacement_characters")
    if digit_ratio > 0.03:
        flags.append("high_digit_character_ratio")
    if page_markers > max(10, len(lines) * 0.01):
        flags.append("page_markers")
    if hyphenated > max(100, len(lines) * 0.08):
        flags.append("possible_ocr_line_hyphenation")
    if editorial.get("liber_liber_wrapper"):
        flags.append("residual_liber_liber_wrapper")
    if editorial.get("edition_notes"):
        flags.append("edition_notes_present")
    language_flags = []
    if _DIALECT_MARKER.search(" ".join((row["title"], text[:10_000]))):
        language_flags.append("review_language_variety_marker")

    result = {field: "" for field in PROBE_FIELDS}
    result.update({
        "record_id": row["record_id"],
        "wordpress_page_id": row["wordpress_page_id"],
        "title": row["title"],
        "author": row["author"],
        "landing_page_url": row["landing_page_url"],
        "period_bucket": row["period_bucket"],
        "preliminary_role": row["preliminary_role"],
        "metadata_decision": row["composition_decision"],
        "license_url": row["license_url"],
        "archive_format": cache["archive_format"],
        "download_page_url": cache["download_page_url"],
        "archive_url": cache["archive_url"],
        "cache_status": cache["cache_status"],
        "archive_cache_path": cache["archive_cache_path"],
        "cleaned_cache_path": cache["cleaned_cache_path"],
        "archive_sha256": cache["archive_sha256"],
        "cleaned_sha256": cache["cleaned_sha256"],
        "raw_byte_count": cache["raw_byte_count"],
        "cleaned_character_count": len(text),
        "cleaned_word_count": len(words),
        "nonempty_line_count": len(lines),
        "replacement_character_count": replacement_count,
        "italian_function_word_ratio": f"{function_ratio:.6f}",
        "alphabetic_character_ratio": f"{alpha_ratio:.6f}",
        "digit_character_ratio": f"{digit_ratio:.6f}",
        "editorial_marker_count": sum(editorial.values()),
        "editorial_markers": ";".join(f"{key}:{value}" for key, value in sorted(editorial.items())),
        "quality_review_flags": ";".join(flags),
        "language_variety_flags": ";".join(language_flags),
        "normalized_word_sha256": fingerprint.normalized_word_sha256,
        "fingerprint_anchor_count": len(fingerprint.anchors),
        "probe_status": "quality_pass" if not flags and not language_flags else "review",
        "activation_status": "inactive_probe_only",
    })
    return result


def _error_row(row: dict[str, str], error: Exception) -> dict[str, Any]:
    result = {field: "" for field in PROBE_FIELDS}
    result.update({
        "record_id": row["record_id"], "wordpress_page_id": row["wordpress_page_id"],
        "title": row["title"], "author": row["author"],
        "landing_page_url": row["landing_page_url"], "period_bucket": row["period_bucket"],
        "preliminary_role": row["preliminary_role"],
        "metadata_decision": row["composition_decision"], "license_url": row["license_url"],
        "probe_status": "error", "probe_decision": "blocked_fetch_or_parse_error",
        "activation_status": "inactive_probe_only",
        "error": f"{type(error).__name__}: {error}",
    })
    return result


def _attach_internal(
    rows: dict[str, dict[str, Any]],
    fingerprints: dict[str, TextFingerprint],
    text_paths: dict[str, Path],
    config: LiberLiberArchiveProbeConfig,
    progress: Progress | None,
) -> list[dict[str, Any]]:
    exact: dict[str, list[str]] = defaultdict(list)
    for record_id, fingerprint in fingerprints.items():
        exact[fingerprint.normalized_word_sha256].append(record_id)
    exact_pairs = set()
    for ids in exact.values():
        if len(ids) < 2:
            continue
        ordered = sorted(ids, key=_record_number)
        for record_id in ordered:
            rows[record_id]["internal_exact_duplicate_ids"] = ";".join(
                value for value in ordered if value != record_id
            )
        exact_pairs.update(tuple(sorted(pair, key=_record_number)) for pair in combinations(ordered, 2))

    candidates = _discover_pairs(fingerprints) - exact_pairs
    result = []
    started = monotonic()
    for index, (left_id, right_id) in enumerate(sorted(candidates, key=lambda pair: tuple(map(_record_number, pair))), 1):
        metric = measure_word_shingle_containment(
            text_paths[left_id].read_text(encoding="utf-8"),
            text_paths[right_id].read_text(encoding="utf-8"),
        )
        if metric["containment"] >= config.near_duplicate_containment:
            item = {
                "left_id": left_id, "right_id": right_id,
                "containment": round(metric["containment"], 6),
                "left_containment": round(metric["left_containment"], 6),
                "right_containment": round(metric["right_containment"], 6),
                "matching_shingles": metric["matching_shingles"],
            }
            result.append(item)
            rows[left_id]["internal_near_duplicate_metrics"] = _append(
                rows[left_id]["internal_near_duplicate_metrics"],
                f"{right_id}|containment={metric['containment']:.6f}",
            )
            rows[right_id]["internal_near_duplicate_metrics"] = _append(
                rows[right_id]["internal_near_duplicate_metrics"],
                f"{left_id}|containment={metric['containment']:.6f}",
            )
        _emit_phase(progress, "internal-overlap", index, len(candidates), started)
    return result


def _attach_cross(
    rows: dict[str, dict[str, Any]],
    fingerprints: dict[str, TextFingerprint],
    text_paths: dict[str, Path],
    references: dict[str, TextReference],
    reference_fingerprints: dict[str, TextFingerprint],
    config: LiberLiberArchiveProbeConfig,
    progress: Progress | None,
) -> list[dict[str, Any]]:
    candidates = _discover_cross_pairs(fingerprints, reference_fingerprints)
    exact: dict[str, list[str]] = defaultdict(list)
    for reference_id, fingerprint in reference_fingerprints.items():
        exact[fingerprint.normalized_word_sha256].append(reference_id)
    for record_id, fingerprint in fingerprints.items():
        candidates.update((record_id, ref) for ref in exact.get(fingerprint.normalized_word_sha256, ()))

    result = []
    started = monotonic()
    for index, (record_id, reference_id) in enumerate(sorted(candidates), 1):
        metric = measure_word_shingle_containment(
            text_paths[record_id].read_text(encoding="utf-8"), references[reference_id].read_text()
        )
        if metric["containment"] >= config.near_duplicate_containment:
            scope = (
                "candidate_covered"
                if metric["left_containment"] >= config.near_duplicate_containment
                else "embedded_reference_only"
            )
            item = {
                "record_id": record_id, "reference_id": reference_id,
                "source_kind": references[reference_id].source_kind,
                "containment": round(metric["containment"], 6),
                "candidate_containment": round(metric["left_containment"], 6),
                "reference_containment": round(metric["right_containment"], 6),
                "duplicate_scope": scope,
                "exact_normalized_text": (
                    fingerprints[record_id].normalized_word_sha256
                    == reference_fingerprints[reference_id].normalized_word_sha256
                ),
                "matching_shingles": metric["matching_shingles"],
            }
            result.append(item)
            rows[record_id]["cross_corpus_overlap_metrics"] = _append(
                rows[record_id]["cross_corpus_overlap_metrics"],
                f"{reference_id}|candidate={metric['left_containment']:.6f}|"
                f"reference={metric['right_containment']:.6f}",
            )
            scopes = set(rows[record_id]["cross_corpus_duplicate_scope"].split(";")) - {""}
            scopes.add(scope)
            rows[record_id]["cross_corpus_duplicate_scope"] = ";".join(sorted(scopes))
        _emit_phase(progress, "cross-corpus-overlap", index, len(candidates), started)
    return result


def _load_references(config: LiberLiberArchiveProbeConfig) -> dict[str, TextReference]:
    result: dict[str, TextReference] = {}
    _add_range_manifest(
        result, config.repo_root, config.bibit_record_manifest_path, "bibit", "object_id",
        {"text_materialized"}, "bibit",
    )
    _add_range_manifest(
        result, config.repo_root, config.bibit_sonnet_manifest_path, "bibit_sonnet", "candidate_id",
        None, "bibit_sonnet",
    )
    _add_probe_manifest(
        result, config.gutenberg_previous_probe_path,
        config.gutenberg_previous_cache_dir, "gutenberg_previous_pool",
    )
    _add_probe_manifest(
        result, config.gutenberg_pass_1b_probe_path,
        config.gutenberg_pass_1b_cache_dir, "gutenberg_pass_1b",
    )
    _add_range_manifest(
        result, config.repo_root, config.gutenberg_resolved_record_manifest_path,
        "gutenberg_resolved", "ebook_id", {"text_materialized_pending_v7"},
        "gutenberg_resolved",
    )
    _add_range_manifest(
        result, config.repo_root, config.gutenberg_resolved_sonnet_manifest_path,
        "gutenberg_sonnet", "candidate_id",
        {"standard_sonnet_materialized_pending_v7"}, "gutenberg_sonnet",
    )
    _add_range_manifest(
        result, config.repo_root, config.wikisource_resolved_record_manifest_path,
        "wikisource_resolved", "work_root_id", {"text_materialized_inactive"},
        "wikisource_resolved",
    )
    _add_range_manifest(
        result, config.repo_root, config.wikisource_resolved_sonnet_manifest_path,
        "wikisource_sonnet", "candidate_id", {"sonnet_materialized_inactive"},
        "wikisource_sonnet",
    )
    for row in _read_csv(config.broader_sources_manifest_path):
        relative = row.get("expected_clean_text_path", "")
        path = config.repo_root / relative if relative else Path("/")
        if relative and path.is_file():
            reference_id = f"current:{row['source_id']}"
            result[reference_id] = TextReference(
                reference_id, "existing_project_corpus", path
            )
    return result


def _add_range_manifest(
    result: dict[str, TextReference],
    repo_root: Path,
    manifest: Path,
    prefix: str,
    id_field: str,
    accepted_statuses: set[str] | None,
    kind: str,
) -> None:
    for row in _read_csv(manifest):
        if accepted_statuses is not None and row.get("artifact_status", "") not in accepted_statuses:
            continue
        if not row.get("shard_path"):
            continue
        reference_id = f"{prefix}:{row[id_field]}"
        if reference_id in result:
            raise ValueError(f"duplicate reference ID: {reference_id}")
        result[reference_id] = TextReference(
            reference_id, kind, repo_root / row["shard_path"],
            int(row["byte_start"]), int(row["byte_end"]),
        )


def _add_probe_manifest(
    result: dict[str, TextReference],
    manifest: Path,
    cache_dir: Path,
    kind: str,
) -> None:
    for row in _read_csv(manifest):
        path = cache_dir / f"pg{row['ebook_id']}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen Gutenberg cache entry: {path}")
        reference_id = f"{kind}:pg{row['ebook_id']}"
        result[reference_id] = TextReference(
            reference_id, kind, path, cleaning="gutenberg_boilerplate"
        )


def _fingerprint_references(
    references: dict[str, TextReference],
    config: LiberLiberArchiveProbeConfig,
    progress: Progress | None,
) -> dict[str, TextFingerprint]:
    result = {}
    started = monotonic()
    for index, (reference_id, reference) in enumerate(sorted(references.items()), 1):
        result[reference_id], _ = fingerprint_text(
            reference.read_text(), sketch_size=config.sketch_size,
            anchor_mask=config.anchor_mask,
        )
        if index == 1 or index == len(references) or index % 250 == 0:
            _emit_phase(progress, "reference-index", index, len(references), started)
    return result


def _load_protected_watch(
    config: LiberLiberArchiveProbeConfig,
) -> tuple[dict[int, tuple[str, ...]], dict[str, int]]:
    watch: dict[int, list[str]] = defaultdict(list)
    denominators = {}
    for row in _read_csv(config.protected_v6_sonnet_manifest_path):
        if row["split_expanded_with_petrarch"] not in {"validation", "test"}:
            continue
        path = config.repo_root / row["clean_text_path"]
        # Watched hits use the complete shingle set. Protected sonnets are short,
        # so direct rolling hashes remain bounded and exact at the frozen gate.
        values = set(
            _rolling_shingle_hashes(
                _normalized_words(path.read_text(encoding="utf-8"))
            )
        )
        if not values:
            continue
        denominators[row["poem_id"]] = len(values)
        for value in values:
            watch[value].append(row["poem_id"])
    return {value: tuple(ids) for value, ids in watch.items()}, denominators


def _attach_protected(
    rows: dict[str, dict[str, Any]],
    hits: dict[str, dict[str, set[int]]],
    denominators: dict[str, int],
    threshold: float,
) -> None:
    for record_id, record_hits in hits.items():
        metrics = []
        for poem_id, values in record_hits.items():
            containment = len(values) / denominators[poem_id]
            if containment >= threshold:
                metrics.append(f"{poem_id}|containment={containment:.6f}")
        rows[record_id]["protected_v6_overlap_metrics"] = ";".join(sorted(metrics))


def _apply_reviews(
    config: LiberLiberArchiveProbeConfig,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    anomalies = [
        row for row in results
        if row["probe_status"] != "error"
        and (row["quality_review_flags"] or row["language_variety_flags"])
    ]
    decisions = {}
    if config.review_csv_path.is_file():
        existing = _read_csv(config.review_csv_path)
        fields = tuple(existing[0].keys()) if existing else REVIEW_FIELDS
        if fields != REVIEW_FIELDS:
            raise ValueError("Liber Liber review CSV schema does not match REVIEW_FIELDS")
        anomaly_ids = {row["record_id"] for row in anomalies}
        stale = [row for row in existing if row["record_id"] not in anomaly_ids]
        reviewed_stale = [
            row["record_id"] for row in stale
            if row["manual_review_resolution"].strip()
            or row["manual_review_rationale"].strip()
        ]
        if reviewed_stale:
            raise ValueError(
                "review ledger contains manually reviewed non-anomaly IDs: "
                + ";".join(sorted(reviewed_stale))
            )
        decisions = {
            row["record_id"]: (
                row["manual_review_resolution"].strip(),
                row["manual_review_rationale"].strip(),
            )
            for row in existing
            if row["record_id"] in anomaly_ids
        }
    output = []
    unresolved = []
    for row in anomalies:
        resolution, rationale = decisions.get(row["record_id"], ("", ""))
        if bool(resolution) != bool(rationale):
            raise ValueError(f"review decision and rationale must both be set for {row['record_id']}")
        row["manual_review_resolution"] = resolution
        row["manual_review_rationale"] = rationale
        if not resolution:
            unresolved.append(row["record_id"])
        output.append({field: row[field] for field in REVIEW_FIELDS})
    _write_csv(config.review_csv_path, REVIEW_FIELDS, sorted(output, key=lambda row: _record_number(row["record_id"])))
    if config.require_review_resolutions and unresolved:
        raise ValueError("manual reviews required for: " + ";".join(sorted(unresolved, key=_record_number)))
    return output


def _finalize_decisions(results: list[dict[str, Any]]) -> None:
    for row in results:
        if row["probe_status"] == "error":
            continue
        manual = row["manual_review_resolution"]
        if "candidate_covered" in row["cross_corpus_duplicate_scope"]:
            decision = "exclude_cross_corpus_duplicate_candidate"
        elif row["protected_v6_overlap_metrics"]:
            decision = "quarantine_protected_v6_segment_before_activation"
        elif "embedded_reference_only" in row["cross_corpus_duplicate_scope"]:
            decision = "quarantine_embedded_duplicate_segments_before_activation"
        elif row["internal_exact_duplicate_ids"]:
            decision = "resolve_internal_exact_canonical_edition"
        elif row["internal_near_duplicate_metrics"]:
            decision = "resolve_internal_near_duplicate"
        elif manual.startswith("exclude_"):
            decision = "exclude_after_bounded_review"
        elif manual.startswith("extract_"):
            decision = "source_specific_extraction_before_activation"
        elif manual:
            decision = "quality_pass_after_bounded_review"
        elif row["quality_review_flags"] or row["language_variety_flags"]:
            decision = "review_unresolved"
        else:
            decision = "quality_pass_pending_extraction_audit"
        row["probe_decision"] = decision


def _build_report(
    config: LiberLiberArchiveProbeConfig,
    results: list[dict[str, Any]],
    conditioned: list[dict[str, str]],
    references: dict[str, TextReference],
    internal_pairs: list[dict[str, Any]],
    cross_pairs: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    successful = [row for row in results if row["probe_status"] != "error"]
    fetched_times = []
    for row in successful:
        key = row["record_id"].replace(":", "_")
        metadata = json.loads(
            (config.cache_dir / "metadata" / f"{key}.json").read_text(encoding="utf-8")
        )
        fetched_times.append(metadata["fetched_at_utc"])
    report = {
        "checkpoint": "5B-liber-liber-bounded-fulltext-probe",
        "cache_snapshot_completed_at_utc": max(fetched_times, default=""),
        "candidate_count": len(results),
        "conditioned_excluded_count": len(conditioned),
        "conditioned_excluded_record_ids": [row["record_id"] for row in conditioned],
        "probe_error_count": sum(row["probe_status"] == "error" for row in results),
        "probe_status_counts": dict(sorted(Counter(row["probe_status"] for row in results).items())),
        "probe_decision_counts": dict(sorted(Counter(row["probe_decision"] for row in results).items())),
        "cache_status_counts": dict(sorted(Counter(row["cache_status"] for row in successful).items())),
        "archive_format_counts": dict(sorted(Counter(row["archive_format"] for row in successful).items())),
        "quality_flag_counts": dict(sorted(Counter(
            flag for row in successful for flag in row["quality_review_flags"].split(";") if flag
        ).items())),
        "language_flag_counts": dict(sorted(Counter(
            flag for row in successful for flag in row["language_variety_flags"].split(";") if flag
        ).items())),
        "manual_review_resolution_counts": dict(sorted(Counter(
            row["manual_review_resolution"] for row in successful if row["manual_review_resolution"]
        ).items())),
        "cleaned_character_count": sum(int(row["cleaned_character_count"] or 0) for row in results),
        "cleaned_word_count": sum(int(row["cleaned_word_count"] or 0) for row in results),
        "role_summary": {
            role: {
                "record_count": len(values),
                "cleaned_character_count": sum(int(row["cleaned_character_count"] or 0) for row in values),
            }
            for role in sorted({row["preliminary_role"] for row in results})
            for values in [[row for row in results if row["preliminary_role"] == role]]
        },
        "internal_near_duplicate_pairs": internal_pairs,
        "cross_corpus_duplicate_pairs": cross_pairs,
        "cross_corpus_covered_candidate_count": sum(
            "candidate_covered" in row["cross_corpus_duplicate_scope"]
            for row in successful
        ),
        "cross_corpus_embedded_reference_record_count": sum(
            "embedded_reference_only" in row["cross_corpus_duplicate_scope"]
            for row in successful
        ),
        "protected_v6_overlap_record_count": sum(
            bool(row["protected_v6_overlap_metrics"]) for row in successful
        ),
        "internal_exact_duplicate_group_count": len({
            tuple(sorted((row["record_id"], *row["internal_exact_duplicate_ids"].split(";"))))
            for row in successful if row["internal_exact_duplicate_ids"]
        }),
        "reference_kind_counts": dict(sorted(Counter(ref.source_kind for ref in references.values()).items())),
        "cross_corpus_reference_count": len(references),
        "protected_v6_reference_count": sum(
            row["split_expanded_with_petrarch"] in {"validation", "test"}
            for row in _read_csv(config.protected_v6_sonnet_manifest_path)
        ),
        "manual_review_anomaly_count": len(reviews),
        "manual_review_unresolved_count": sum(not row["manual_review_resolution"] for row in reviews),
        "inputs": {
            "inventory_path": _portable(config.inventory_path, config.repo_root),
            "inventory_sha256": _sha256_file(config.inventory_path),
        },
        "outputs": {
            "probe_csv_path": _portable(config.output_csv_path, config.repo_root),
            "probe_csv_sha256": _sha256_file(config.output_csv_path),
            "review_csv_path": _portable(config.review_csv_path, config.repo_root),
            "review_csv_sha256": _sha256_file(config.review_csv_path),
            "local_cache_path": _portable(config.cache_dir, config.repo_root),
        },
        "policy": {
            "txt_zip_preferred": True,
            "odt_fallback": True,
            "near_duplicate_containment_threshold": config.near_duplicate_containment,
            "protected_v6_containment_threshold": config.protected_containment,
            "text_activated": False, "v7_created": False,
            "mixture_assigned": False, "cache_deleted": False,
            "gpu_work_started": False,
        },
    }
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Liber Liber Bounded Full-Text Probe",
        "", "## Result", "",
        f"Checkpoint 5B probes exactly {report['candidate_count']:,} checkpoint-5A eligible records.",
        "",
        f"- Successful/failed status: `{report['probe_status_counts']}`.",
        f"- Cleaned volume: {report['cleaned_character_count']:,} characters and {report['cleaned_word_count']:,} words.",
        f"- Archive formats: `{report['archive_format_counts']}`.",
        f"- Cross-corpus references indexed: {report['cross_corpus_reference_count']:,}.",
        f"- Cross-corpus threshold pairs: {len(report['cross_corpus_duplicate_pairs']):,}.",
        f"- Fully covered candidates: {report['cross_corpus_covered_candidate_count']:,}; embedded-reference-only candidates: {report['cross_corpus_embedded_reference_record_count']:,}.",
        f"- Internal near-duplicate pairs: {len(report['internal_near_duplicate_pairs']):,}.",
        f"- Protected V6 validation/test sonnets: {report['protected_v6_reference_count']:,}; overlapping candidates: {report['protected_v6_overlap_record_count']:,}.",
        f"- Automated anomalies reviewed: {report['manual_review_anomaly_count']:,}; unresolved: {report['manual_review_unresolved_count']:,}.",
        f"- Conditioned language-variety records excluded from this queue: {report['conditioned_excluded_count']:,}.",
        "", "## Decisions", "", "| Decision | Records |", "| --- | ---: |",
    ]
    lines.extend(f"| `{key}` | {value:,} |" for key, value in report["probe_decision_counts"].items())
    lines.extend([
        "", "## Reference Coverage", "", "| Reference kind | Records |", "| --- | ---: |",
    ])
    lines.extend(f"| `{key}` | {value:,} |" for key, value in report["reference_kind_counts"].items())
    lines.extend([
        "", "## Boundary", "",
        "This probe records acquisition, quality, overlap, and protected-set evidence only. "
        "It activates no text, creates no V7 split, assigns no mixture weight, deletes no cache, "
        "and starts no GPU work.", "",
    ])
    return "\n".join(lines)


def _download_candidates(row: dict[str, str]) -> list[tuple[str, str]]:
    result = []
    for url in row["download_page_urls"].split(";"):
        if not url:
            continue
        value = parse_qs(urlparse(url).query).get("type", [""])[0]
        archive_format = {"opera_url_txt": "txt_zip", "opera_url_odt": "odt"}.get(value)
        if archive_format:
            result.append((archive_format, url))
    priority = {"txt_zip": 0, "odt": 1}
    return sorted(set(result), key=lambda item: (priority[item[0]], item[1]))


def _discover_pairs(fingerprints: dict[str, TextFingerprint]) -> set[tuple[str, str]]:
    pairs = set()
    for attribute in ("anchors", "sketch"):
        postings: dict[int, list[str]] = defaultdict(list)
        for document_id, fingerprint in fingerprints.items():
            for value in getattr(fingerprint, attribute):
                postings[value].append(document_id)
        collisions: Counter[tuple[str, str]] = Counter()
        for ids in postings.values():
            if 1 < len(ids) <= 40:
                collisions.update(tuple(sorted(pair)) for pair in combinations(ids, 2))
        for pair, count in collisions.items():
            denominator = min(
                len(getattr(fingerprints[pair[0]], attribute)),
                len(getattr(fingerprints[pair[1]], attribute)),
            )
            if denominator and count >= 2 and count / denominator >= 0.4:
                pairs.add(pair)
    return pairs


def _discover_cross_pairs(
    candidates: dict[str, TextFingerprint],
    references: dict[str, TextFingerprint],
) -> set[tuple[str, str]]:
    pairs = set()
    for attribute in ("anchors", "sketch"):
        postings: dict[int, list[str]] = defaultdict(list)
        for reference_id, fingerprint in references.items():
            for value in getattr(fingerprint, attribute):
                postings[value].append(reference_id)
        for record_id, fingerprint in candidates.items():
            collisions: Counter[str] = Counter()
            for value in getattr(fingerprint, attribute):
                values = postings.get(value, ())
                if len(values) <= 40:
                    collisions.update(values)
            for reference_id, count in collisions.items():
                denominator = min(
                    len(getattr(fingerprint, attribute)),
                    len(getattr(references[reference_id], attribute)),
                )
                if denominator and count >= 2 and count / denominator >= 0.4:
                    pairs.add((record_id, reference_id))
    return pairs


def _validate_config(config: LiberLiberArchiveProbeConfig) -> None:
    if config.request_delay_seconds < 0 or config.request_timeout_seconds <= 0:
        raise ValueError("request delay/timeout must be non-negative/positive")
    if not 0 < config.near_duplicate_containment <= 1:
        raise ValueError("near-duplicate threshold must be in (0, 1]")
    if not 0 < config.protected_containment <= 1:
        raise ValueError("protected threshold must be in (0, 1]")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary, payload)
    temporary.replace(path)


def _portable(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _append(current: str, value: str) -> str:
    return ";".join(item for item in (current, value) if item)


def _record_number(value: str) -> int:
    return int(value.split(":")[-1])


def _emit_item(
    progress: Progress | None,
    index: int,
    total: int,
    started: float,
    result: dict[str, Any],
) -> None:
    if progress is None:
        return
    elapsed = monotonic() - started
    eta = elapsed / index * (total - index)
    progress(
        f"record={index:,}/{total:,} percent={index / total:.1%} "
        f"id={result['record_id']} cache={result['cache_status'] or 'error'} "
        f"status={result['probe_status']} elapsed={elapsed:.1f}s eta={eta:.1f}s"
    )


def _emit_phase(
    progress: Progress | None,
    label: str,
    index: int,
    total: int,
    started: float,
) -> None:
    if progress is None or total == 0:
        return
    elapsed = monotonic() - started
    eta = elapsed / index * (total - index)
    progress(
        f"phase={label} item={index:,}/{total:,} percent={index / total:.1%} "
        f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
    )
