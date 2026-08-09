"""Stratified full-text value gate for Italian Project Gutenberg candidates."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep as default_sleep
from typing import Any

import requests

from .gutenberg import FetchedGutenbergText, fetch_gutenberg_text, strip_gutenberg_boilerplate


PROBE_STATUSES = {
    "audit_then_deduplicate",
    "deduplicate_before_full_text_audit",
    "deduplicate_intra_gutenberg_before_audit",
}
DEFAULT_ROLE_QUOTAS = {
    "historical_general_candidate": 10,
    "historical_non_sonnet_poetry_candidate": 10,
    "nineteenth_century_bridge_candidate": 16,
    "sonnet_specialization_candidate": 999,
}
SAMPLE_FIELDS = (
    "ebook_id",
    "title",
    "authors",
    "preliminary_role",
    "period_bucket",
    "selection_reasons",
    "status",
    "error",
    "fetched_url",
    "raw_character_count",
    "cleaned_character_count",
    "cleaned_word_count",
    "nonempty_line_count",
    "replacement_character_count",
    "italian_function_word_ratio",
    "alphabetic_character_ratio",
    "editorial_marker_count",
    "cleaned_sha256",
    "first_characters",
    "last_characters",
    "possible_existing_work_matches",
    "reference_overlap_metrics",
    "max_reference_8gram_containment",
    "content_duplicate_signal",
)

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
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
_EDITORIAL_MARKER = re.compile(
    r"\b(?:nota del trascrittore|transcriber['’]s note|errata corrige|"
    r"indice alfabetico|introduzione dell['’]editore|prefazione dell['’]editore)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GutenbergFullTextGateConfig:
    repo_root: Path
    inventory_csv_path: Path
    cache_dir: Path
    sample_csv_path: Path
    json_report_path: Path
    markdown_report_path: Path
    bibit_record_manifest_path: Path | None = None
    request_delay_seconds: float = 1.0
    request_timeout_seconds: float = 60.0
    sample_seed: int = 1337
    min_cleaned_characters: int = 1_000
    min_italian_function_word_ratio: float = 0.02


FetchText = Callable[..., FetchedGutenbergText]
Progress = Callable[[str], None]
Sleep = Callable[[float], None]


def select_gutenberg_fulltext_gate_sample(
    rows: list[dict[str, str]],
    *,
    sample_seed: int = 1337,
    role_quotas: dict[str, int] | None = None,
) -> list[dict[str, str]]:
    """Select a deterministic role-stratified sample plus every overlap signal."""

    quotas = role_quotas or DEFAULT_ROLE_QUOTAS
    candidates = [row for row in rows if row["inventory_status"] in PROBE_STATUSES]
    selected_reasons: dict[str, set[str]] = {}

    for row in candidates:
        reasons: set[str] = set()
        if row["possible_existing_work_matches"]:
            reasons.add("metadata_overlap_signal")
        if row["intra_gutenberg_duplicate_ids"]:
            reasons.add("intra_gutenberg_duplicate_signal")
        if row["preliminary_role"] == "sonnet_specialization_candidate":
            reasons.add("all_sonnet_candidates")
        if reasons:
            selected_reasons[row["ebook_id"]] = reasons

    for role, quota in quotas.items():
        if quota <= 0:
            continue
        role_rows = [row for row in candidates if row["preliminary_role"] == role]
        role_rows.sort(key=lambda row: _sample_key(row["ebook_id"], sample_seed))
        for row in role_rows[:quota]:
            selected_reasons.setdefault(row["ebook_id"], set()).add(
                f"stratified_role_sample:{role}"
            )

    by_id = {row["ebook_id"]: row for row in candidates}
    selected = []
    for ebook_id in sorted(selected_reasons, key=int):
        row = dict(by_id[ebook_id])
        row["selection_reasons"] = ";".join(sorted(selected_reasons[ebook_id]))
        selected.append(row)
    return selected


def run_gutenberg_fulltext_gate(
    config: GutenbergFullTextGateConfig,
    *,
    fetch_text: FetchText = fetch_gutenberg_text,
    session: requests.Session | None = None,
    progress: Progress | None = None,
    sleep: Sleep = default_sleep,
) -> dict[str, Any]:
    """Download and inspect the bounded sample, then project full-probe volume."""

    _validate_config(config)
    inventory_rows = _read_csv(config.inventory_csv_path)
    candidates = [
        row for row in inventory_rows if row["inventory_status"] in PROBE_STATUSES
    ]
    sample = select_gutenberg_fulltext_gate_sample(
        inventory_rows,
        sample_seed=config.sample_seed,
    )
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    reference_loader = (
        _BibItReferenceLoader(config.repo_root, config.bibit_record_manifest_path)
        if config.bibit_record_manifest_path is not None
        else None
    )
    started = monotonic()
    results: list[dict[str, Any]] = []

    for index, row in enumerate(sample, start=1):
        cache_path = config.cache_dir / f"pg{row['ebook_id']}.txt"
        try:
            if cache_path.is_file():
                raw_text = cache_path.read_text(encoding="utf-8")
                fetched_url = row["plain_text_url"]
                cache_status = "hit"
            else:
                if index > 1 and config.request_delay_seconds:
                    sleep(config.request_delay_seconds)
                fetched = fetch_text(
                    row["ebook_id"],
                    session=session,
                    timeout=int(config.request_timeout_seconds),
                )
                raw_text = fetched.text
                fetched_url = fetched.url
                cache_path.write_text(raw_text, encoding="utf-8")
                cache_status = "downloaded"
            cleaned = strip_gutenberg_boilerplate(raw_text)
            result = _inspect_text(row, raw_text, cleaned, config)
            result["fetched_url"] = fetched_url
            _attach_reference_overlap(result, row, cleaned, reference_loader)
        except Exception as error:
            result = _error_result(row, error)
            cache_status = "error"
        results.append(result)
        elapsed = monotonic() - started
        eta = elapsed / index * (len(sample) - index)
        _report(
            progress,
            f"source {index:,}/{len(sample):,} ({index / len(sample):.1%}) "
            f"id={row['ebook_id']} status={result['status']} cache={cache_status} "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
        )

    _write_csv(config.sample_csv_path, SAMPLE_FIELDS, results)
    report = _build_report(config, candidates=candidates, results=results)
    _write_json(config.json_report_path, report)
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(
        render_gutenberg_fulltext_gate_markdown(report),
        encoding="utf-8",
    )
    return report


def _inspect_text(
    row: dict[str, str],
    raw_text: str,
    cleaned: str,
    config: GutenbergFullTextGateConfig,
) -> dict[str, Any]:
    words = [word.casefold() for word in _WORD.findall(cleaned)]
    function_word_ratio = (
        sum(word in _ITALIAN_FUNCTION_WORDS for word in words) / len(words)
        if words
        else 0.0
    )
    alphabetic_ratio = (
        sum(character.isalpha() for character in cleaned) / len(cleaned)
        if cleaned
        else 0.0
    )
    replacement_characters = cleaned.count("\ufffd")
    editorial_markers = len(_EDITORIAL_MARKER.findall(cleaned))
    if len(cleaned) < config.min_cleaned_characters:
        status = "review_too_short"
    elif function_word_ratio < config.min_italian_function_word_ratio:
        status = "review_low_italian_signal"
    elif replacement_characters:
        status = "review_encoding_replacement_characters"
    elif alphabetic_ratio < 0.55:
        status = "review_low_alphabetic_ratio"
    else:
        status = "sample_quality_pass"
    return {
        "ebook_id": row["ebook_id"],
        "title": row["title"],
        "authors": row["authors"],
        "preliminary_role": row["preliminary_role"],
        "period_bucket": row["period_bucket"],
        "selection_reasons": row["selection_reasons"],
        "status": status,
        "error": "",
        "fetched_url": "",
        "raw_character_count": len(raw_text),
        "cleaned_character_count": len(cleaned),
        "cleaned_word_count": len(words),
        "nonempty_line_count": sum(bool(line.strip()) for line in cleaned.splitlines()),
        "replacement_character_count": replacement_characters,
        "italian_function_word_ratio": round(function_word_ratio, 6),
        "alphabetic_character_ratio": round(alphabetic_ratio, 6),
        "editorial_marker_count": editorial_markers,
        "cleaned_sha256": hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
        "first_characters": " ".join(cleaned[:240].split()),
        "last_characters": " ".join(cleaned[-240:].split()),
        "possible_existing_work_matches": row["possible_existing_work_matches"],
        "reference_overlap_metrics": "",
        "max_reference_8gram_containment": "",
        "content_duplicate_signal": False,
    }


def _error_result(row: dict[str, str], error: Exception) -> dict[str, Any]:
    result = {field: "" for field in SAMPLE_FIELDS}
    result.update(
        {
            "ebook_id": row["ebook_id"],
            "title": row["title"],
            "authors": row["authors"],
            "preliminary_role": row["preliminary_role"],
            "period_bucket": row["period_bucket"],
            "selection_reasons": row["selection_reasons"],
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "possible_existing_work_matches": row["possible_existing_work_matches"],
            "reference_overlap_metrics": "",
            "max_reference_8gram_containment": "",
            "content_duplicate_signal": False,
        }
    )
    return result


def _build_report(
    config: GutenbergFullTextGateConfig,
    *,
    candidates: list[dict[str, str]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_counts = Counter(row["preliminary_role"] for row in candidates)
    sample_counts = Counter(row["preliminary_role"] for row in results)
    status_counts = Counter(row["status"] for row in results)
    role_projections: dict[str, dict[str, Any]] = {}
    for role, candidate_count in sorted(candidate_counts.items()):
        lengths = [
            int(row["cleaned_character_count"])
            for row in results
            if row["preliminary_role"] == role
            and row["status"] != "error"
            and int(row["cleaned_character_count"])
        ]
        role_projections[role] = {
            "candidate_count": candidate_count,
            "sample_count": sample_counts.get(role, 0),
            "successful_sample_count": len(lengths),
            "mean_cleaned_characters": round(statistics.mean(lengths)) if lengths else 0,
            "median_cleaned_characters": round(statistics.median(lengths)) if lengths else 0,
            "projected_cleaned_characters": (
                round(statistics.mean(lengths) * candidate_count) if lengths else 0
            ),
        }
    projected_total = sum(
        row["projected_cleaned_characters"] for row in role_projections.values()
    )
    sample_total = sum(int(row["cleaned_character_count"] or 0) for row in results)
    reference_overlaps = [
        {
            "ebook_id": row["ebook_id"],
            "title": row["title"],
            "possible_existing_work_matches": row["possible_existing_work_matches"],
            "reference_overlap_metrics": row["reference_overlap_metrics"],
            "max_reference_8gram_containment": row[
                "max_reference_8gram_containment"
            ],
            "content_duplicate_signal": row["content_duplicate_signal"],
        }
        for row in results
        if row["possible_existing_work_matches"]
    ]
    return {
        "gate_version": "project_gutenberg_fulltext_gate_v1",
        "created_at_utc": _utc_now(),
        "inventory_csv_path": _portable(config.inventory_csv_path, config.repo_root),
        "eligible_probe_candidate_count": len(candidates),
        "sample_count": len(results),
        "sample_status_counts": dict(sorted(status_counts.items())),
        "sample_cleaned_character_count": sample_total,
        "sample_metadata_overlap_count": sum(
            bool(row["possible_existing_work_matches"]) for row in results
        ),
        "sample_content_duplicate_signal_count": sum(
            str(row["content_duplicate_signal"]).casefold() == "true"
            for row in results
        ),
        "sample_reference_overlaps": reference_overlaps,
        "role_projections": role_projections,
        "projected_total_cleaned_characters": projected_total,
        "full_probe_runtime_estimate": {
            "request_delay_seconds": config.request_delay_seconds,
            "minimum_request_delay_seconds": round(
                len(candidates) * config.request_delay_seconds,
                1,
            ),
            "range": "10m-90m network-dependent",
        },
        "outputs": {
            "sample_csv_path": _portable(config.sample_csv_path, config.repo_root),
            "sample_csv_sha256": _sha256_file(config.sample_csv_path),
            "json_report_path": _portable(config.json_report_path, config.repo_root),
            "markdown_report_path": _portable(config.markdown_report_path, config.repo_root),
            "local_cache_path": _portable(config.cache_dir, config.repo_root),
        },
        "policy": {
            "sample_is_deterministic_and_role_stratified": True,
            "all_metadata_overlap_signals_included": True,
            "all_sonnet_candidates_included": True,
            "sample_text_is_machine_local": True,
            "activation_authorized": False,
            "projected_volume_is_approximate": True,
            "content_duplicate_signal_minimum_8gram_containment": 0.8,
        },
    }


def render_gutenberg_fulltext_gate_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Project Gutenberg Full-Text Value Gate",
        "",
        "## Result",
        "",
        (
            f"Inspected {report['sample_count']:,} deterministic samples from "
            f"{report['eligible_probe_candidate_count']:,} metadata-compatible records."
        ),
        "",
        "| Sample status | Records |",
        "| --- | ---: |",
    ]
    for status, count in report["sample_status_counts"].items():
        lines.append(f"| `{status}` | {count:,} |")
    lines.extend(
        [
            "",
            "## Role Projections",
            "",
            "| Role | Candidates | Samples | Mean chars | Median chars | Projected chars |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for role, row in report["role_projections"].items():
        lines.append(
            f"| `{role}` | {row['candidate_count']:,} | "
            f"{row['successful_sample_count']:,} | {row['mean_cleaned_characters']:,} | "
            f"{row['median_cleaned_characters']:,} | "
            f"{row['projected_cleaned_characters']:,} |"
        )
    lines.extend(
        [
            "",
            f"Approximate projected total: {report['projected_total_cleaned_characters']:,} characters.",
            (
                f"Text-level duplicate signals among metadata overlaps: "
                f"{report['sample_content_duplicate_signal_count']:,}/"
                f"{report['sample_metadata_overlap_count']:,}."
            ),
            "",
            "## Cross-Archive Duplicate Evidence",
            "",
            "| Gutenberg ID | Existing match | 8-gram containment | Duplicate signal |",
            "| ---: | --- | ---: | --- |",
        ]
    )
    for row in report["sample_reference_overlaps"]:
        containment = row["max_reference_8gram_containment"]
        containment_text = f"{float(containment):.4f}" if containment != "" else "n/a"
        signal = "yes" if row["content_duplicate_signal"] else "no"
        lines.append(
            f"| {row['ebook_id']} | `{row['possible_existing_work_matches']}` | "
            f"{containment_text} | {signal} |"
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- The projection is a planning estimate, not an activated corpus size.",
            "- Sample text remains machine-local and is not committed.",
            "- Full-text editorial review and cross-corpus deduplication remain required.",
            "- No V7 split or training-mixture weight is assigned.",
            "",
            "## Next Gate",
            "",
            (
                "If sample availability and quality justify expansion, probe all eligible "
                f"records. Estimated runtime: {report['full_probe_runtime_estimate']['range']}."
            ),
            "",
        ]
    )
    return "\n".join(lines)


class _BibItReferenceLoader:
    def __init__(self, repo_root: Path, manifest_path: Path):
        self.repo_root = repo_root
        self.rows = {row["object_id"]: row for row in _read_csv(manifest_path)}
        self.shards: dict[str, bytes] = {}

    def text(self, object_id: str) -> str:
        row = self.rows.get(object_id)
        if row is None or not row["shard_path"]:
            raise ValueError(f"BibIt reference text is unavailable: {object_id}")
        shard_path = row["shard_path"]
        if shard_path not in self.shards:
            self.shards[shard_path] = (self.repo_root / shard_path).read_bytes()
        payload = self.shards[shard_path][
            int(row["byte_start"]) : int(row["byte_end"])
        ]
        return payload.decode("utf-8")


def _attach_reference_overlap(
    result: dict[str, Any],
    row: dict[str, str],
    cleaned: str,
    reference_loader: _BibItReferenceLoader | None,
) -> None:
    references = [
        value
        for value in row["possible_existing_work_matches"].split(";")
        if value
    ]
    if not references or reference_loader is None:
        return
    candidate_grams = _word_ngrams(cleaned, size=8)
    candidate_exact = _normalize_exact(cleaned)
    metrics = []
    max_containment = 0.0
    for reference in references:
        if not reference.startswith("bibit:"):
            continue
        reference_text = reference_loader.text(reference.split(":", maxsplit=1)[1])
        reference_grams = _word_ngrams(reference_text, size=8)
        intersection = len(candidate_grams & reference_grams)
        denominator = min(len(candidate_grams), len(reference_grams))
        union = len(candidate_grams | reference_grams)
        containment = intersection / denominator if denominator else 0.0
        jaccard = intersection / union if union else 0.0
        exact = candidate_exact == _normalize_exact(reference_text)
        max_containment = max(max_containment, containment)
        metrics.append(
            f"{reference}|containment={containment:.6f}|"
            f"jaccard={jaccard:.6f}|exact={str(exact).lower()}"
        )
    if metrics:
        result["reference_overlap_metrics"] = ";".join(metrics)
        result["max_reference_8gram_containment"] = round(max_containment, 6)
        result["content_duplicate_signal"] = max_containment >= 0.8


def _word_ngrams(text: str, *, size: int) -> set[tuple[str, ...]]:
    words = _normalized_words(text)
    return {
        tuple(words[index : index + size])
        for index in range(max(0, len(words) - size + 1))
    }


def _normalized_words(text: str) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return _WORD.findall(without_marks)


def _normalize_exact(text: str) -> str:
    return " ".join(text.casefold().split())


def _validate_config(config: GutenbergFullTextGateConfig) -> None:
    if config.request_delay_seconds < 0:
        raise ValueError("request_delay_seconds cannot be negative")
    if config.request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive")
    if config.min_cleaned_characters <= 0:
        raise ValueError("min_cleaned_characters must be positive")
    if not 0 <= config.min_italian_function_word_ratio <= 1:
        raise ValueError("min_italian_function_word_ratio must be between zero and one")


def _sample_key(ebook_id: str, sample_seed: int) -> bytes:
    return hashlib.sha256(f"{sample_seed}:{ebook_id}".encode("utf-8")).digest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, seconds = divmod(total, 60)
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _report(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)
