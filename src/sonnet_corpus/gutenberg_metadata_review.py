"""Evidence-preserving triage for unresolved Italian Gutenberg metadata."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep as default_sleep
from typing import Any

import requests

from .gutenberg import FetchedGutenbergText, fetch_gutenberg_text, strip_gutenberg_boilerplate
from .gutenberg_metadata_review_queue import QUEUE_FIELDS, REVIEW_STATUS_EVIDENCE


FROZEN_QUEUE_SHA256 = "6bbf0275d137ad14fff04c0811f1b265a3da3234e43d9709707577429ea25e9d"
FROZEN_QUEUE_COUNT = 673
FROZEN_STATUS_COUNTS = {
    "review_language_variety_before_download": 6,
    "review_missing_period_evidence": 79,
    "review_translation_edition_date": 25,
    "review_work_publication_date": 563,
}

RESOLUTION_FIELDS = (
    "ebook_id",
    "title",
    "authors",
    "author_birth_years",
    "author_death_years",
    "subjects",
    "bookshelves",
    "preliminary_role",
    "period_bucket",
    "inventory_status",
    "landing_page_url",
    "plain_text_url",
    "fetched_url",
    "cache_status",
    "fetch_error",
    "raw_character_count",
    "cleaned_character_count",
    "gutenberg_release_years",
    "date_evidence_json",
    "language_variety_evidence_json",
    "selected_evidence_kind",
    "selected_evidence_year_start",
    "selected_evidence_year_end",
    "selected_evidence_text",
    "evidence_confidence",
    "resolved_period_bucket",
    "resolved_role",
    "automatic_decision",
    "resolution_status",
    "manual_review_reasons",
)

_YEAR = re.compile(r"(?<!\d)(1[2-9]\d{2}|20\d{2})(?!\d)")
_ROMAN_YEAR = re.compile(r"^(?=[MDCLXVI]+$)[MDCLXVI]{4,}$", re.IGNORECASE)
_START_MARKER = re.compile(
    r"^\s*\*\*\*\s*START\s+OF\s+THE\s+PROJECT\s+GUTENBERG\b",
    re.IGNORECASE,
)
_RELEASE_FIELD = re.compile(r"^\s*(?:release date|most recently updated)\s*:", re.I)
_ORIGINAL_PUBLICATION_FIELD = re.compile(
    r"^\s*original publication\s*:\s*(.*)$", re.IGNORECASE
)
_FIRST_EDITION = re.compile(
    r"\b(?:prima\s+edizione|first\s+edition|prima\s+pubblicazione)\b",
    re.IGNORECASE,
)
_FIRST_ITALIAN_VERSION = re.compile(
    r"\b(?:prima\s+(?:versione|traduzione)\s+italiana)\b",
    re.IGNORECASE,
)
_LATER_EDITION = re.compile(
    r"\b(?:seconda|terza|quarta|quinta|nuova)\s+edizione\b",
    re.IGNORECASE,
)
_COPYRIGHT = re.compile(r"\b(?:copyright|propriet[aà]\s+letteraria)\b", re.I)
_PUBLISHER = re.compile(
    r"\b(?:editore|editori|tipografia|tipografico|presso|stampat[oa]|"
    r"pubblicat[oa])\b",
    re.IGNORECASE,
)
_CITY = re.compile(
    r"\b(?:roma|milano|torino|firenze|bologna|napoli|venezia|genova|"
    r"palermo|parma|pisa|siena|livorno|londra|parigi)\b",
    re.IGNORECASE,
)
_CENTURY_ROMAN = re.compile(
    r"\bsec(?:olo)?\.?\s*([IVXLCDM]+)\b", re.IGNORECASE
)
_NAMED_CENTURIES = {
    "duecento": 13,
    "trecento": 14,
    "quattrocento": 15,
    "cinquecento": 16,
    "seicento": 17,
    "settecento": 18,
    "ottocento": 19,
}
_PRIMARY_PERIOD_TITLE = re.compile(
    r"\b(?:testo|testi|libro|cantari|rimatori|rime|poesie|sonetti|"
    r"commedie|lettere|cronache|statuti)\b",
    re.IGNORECASE,
)
_SECONDARY_PERIOD_TITLE = re.compile(
    r"\b(?:romanzo|storia|studio|saggio|ricerche|conferenze|"
    r"vita\s+italiana|cronistoria|commentario)\b",
    re.IGNORECASE,
)
_LANGUAGE_VARIETY_PATTERNS = {
    "dialect": re.compile(r"\bdialett\w*\b", re.IGNORECASE),
    "vernacular": re.compile(r"\bvernacol\w*\b", re.IGNORECASE),
    "neapolitan": re.compile(r"\bnapoletan\w*\b", re.IGNORECASE),
    "romanesco": re.compile(r"\bromanesc\w*\b", re.IGNORECASE),
    "milanese": re.compile(r"\bmilanes\w*\b", re.IGNORECASE),
    "venetian": re.compile(r"\bvenezian\w*\b", re.IGNORECASE),
    "sicilian": re.compile(r"\bsicilian\w*\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class GutenbergMetadataReviewConfig:
    repo_root: Path
    queue_csv_path: Path
    cache_dir: Path
    output_csv_path: Path
    manual_review_csv_path: Path
    json_report_path: Path
    markdown_report_path: Path
    expected_queue_sha256: str = FROZEN_QUEUE_SHA256
    expected_record_count: int = FROZEN_QUEUE_COUNT
    expected_status_counts: dict[str, int] | None = None
    request_delay_seconds: float = 1.0
    request_timeout_seconds: float = 60.0
    fetch_attempts: int = 3


FetchText = Callable[..., FetchedGutenbergText]
Progress = Callable[[str], None]
Sleep = Callable[[float], None]


def extract_gutenberg_date_evidence(
    raw_text: str,
    *,
    title: str,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Extract dated primary-source evidence while isolating release metadata."""

    header, body = _split_header_and_body(raw_text)
    release_years = sorted(
        {
            year
            for line in header.splitlines()
            if _RELEASE_FIELD.search(line)
            for year in _years(line)
        }
    )
    evidence: list[dict[str, Any]] = []

    header_lines = header.splitlines()
    for index, line in enumerate(header_lines):
        match = _ORIGINAL_PUBLICATION_FIELD.match(line)
        if match is None:
            continue
        value_lines = [match.group(1).strip()]
        for continuation in header_lines[index + 1 : index + 4]:
            if not continuation.startswith((" ", "\t")) or ":" in continuation:
                break
            value_lines.append(continuation.strip())
        value = " ".join(part for part in value_lines if part)
        for year in _years(value):
            evidence.append(
                _evidence(
                    "gutenberg_original_publication",
                    year,
                    year,
                    f"Original publication: {value}",
                    "high",
                )
            )

    title_evidence = _extract_title_period_evidence(title)
    evidence.extend(title_evidence)

    lines = [" ".join(line.split()) for line in body.splitlines() if line.strip()]
    front_lines = lines[:160]
    for index, line in enumerate(front_lines):
        context = " | ".join(front_lines[max(0, index - 3) : index + 4])
        if _FIRST_EDITION.search(line):
            for year in _explicit_marker_years(front_lines, index):
                evidence.append(
                    _evidence("explicit_first_edition", year, year, context, "high")
                )
        if _FIRST_ITALIAN_VERSION.search(line):
            for year in _explicit_marker_years(front_lines, index):
                evidence.append(
                    _evidence(
                        "explicit_first_italian_version",
                        year,
                        year,
                        context,
                        "high",
                    )
                )
        if _COPYRIGHT.search(line):
            for year in _years(line):
                evidence.append(
                    _evidence("copyright_year", year, year, line, "medium")
                )
        standalone_year = _standalone_year(line)
        if standalone_year is not None:
            nearby = " ".join(front_lines[max(0, index - 5) : index + 1])
            if _PUBLISHER.search(nearby) or _CITY.search(nearby):
                evidence.append(
                    _evidence(
                        "title_page_edition_year",
                        standalone_year,
                        standalone_year,
                        context,
                        "medium",
                    )
                )
        roman_year = _standalone_roman_year(line)
        if roman_year is not None:
            nearby = " ".join(front_lines[max(0, index - 5) : index + 1])
            if _PUBLISHER.search(nearby) or _CITY.search(nearby):
                evidence.append(
                    _evidence(
                        "title_page_edition_year",
                        roman_year,
                        roman_year,
                        context,
                        "medium",
                    )
                )

    return _deduplicate_evidence(evidence), release_years


def resolve_gutenberg_metadata_row(
    row: dict[str, str],
    raw_text: str,
) -> dict[str, Any]:
    """Triage one queue row using only direct deterministic evidence."""

    evidence, release_years = extract_gutenberg_date_evidence(
        raw_text,
        title=row["title"],
    )
    cleaned = strip_gutenberg_boilerplate(raw_text)
    language_evidence = _language_variety_evidence(row, cleaned)
    result = _base_result(row)
    result.update(
        {
            "raw_character_count": len(raw_text),
            "cleaned_character_count": len(cleaned),
            "gutenberg_release_years": ";".join(map(str, release_years)),
            "date_evidence_json": json.dumps(
                evidence, ensure_ascii=False, separators=(",", ":")
            ),
            "language_variety_evidence_json": json.dumps(
                language_evidence, ensure_ascii=False, separators=(",", ":")
            ),
        }
    )

    if row["inventory_status"] == "review_language_variety_before_download":
        return _manual_result(
            result,
            "primary_text_language_variety_review_required",
        )

    title_period = _first_evidence(evidence, "title_work_period")
    if title_period is not None:
        return _resolved_result(result, row, title_period)

    original = _first_evidence(evidence, "gutenberg_original_publication")
    if original is not None:
        return _resolved_result(result, row, original)

    first_translation = _first_evidence(evidence, "explicit_first_italian_version")
    if first_translation is not None:
        return _resolved_result(result, row, first_translation)

    title_page = _first_evidence(evidence, "title_page_edition_year")
    copyright_evidence = _first_evidence(evidence, "copyright_year")
    candidate = title_page or copyright_evidence
    if candidate is None:
        reason = "no_direct_period_evidence"
        if row["inventory_status"] == "review_translation_edition_date":
            reason = "translation_edition_date_not_found"
        return _manual_result(result, reason)

    year = int(candidate["year_start"])
    if row["inventory_status"] == "review_translation_edition_date":
        if year <= 1900:
            return _resolved_result(result, row, candidate)
        return _manual_result(
            result,
            "post_1900_translation_edition_requires_authoritative_review",
        )

    births = _parse_year_list(row["author_birth_years"])
    if year <= 1900 and births and min(births) >= 1801:
        return _resolved_result(result, row, candidate, forced_period="nineteenth_century_bridge")
    if year <= 1800:
        return _resolved_result(result, row, candidate)
    if year <= 1900:
        return _manual_result(
            result,
            "edition_proves_pre_1901_eligibility_but_work_period_is_ambiguous",
        )
    return _manual_result(
        result,
        "post_1900_edition_does_not_prove_work_first_publication",
    )


def run_gutenberg_metadata_review(
    config: GutenbergMetadataReviewConfig,
    *,
    fetch_text: FetchText = fetch_gutenberg_text,
    session: requests.Session | None = None,
    progress: Progress | None = None,
    sleep: Sleep = default_sleep,
) -> dict[str, Any]:
    """Acquire and triage the complete frozen metadata-review queue."""

    _validate_config(config)
    queue_sha256 = _sha256_file(config.queue_csv_path)
    if queue_sha256 != config.expected_queue_sha256:
        raise ValueError(
            "Gutenberg review queue SHA-256 mismatch: "
            f"expected={config.expected_queue_sha256} actual={queue_sha256}"
        )
    rows = _read_queue(config.queue_csv_path)
    if len(rows) != config.expected_record_count:
        raise ValueError(
            "Gutenberg review queue count mismatch: "
            f"expected={config.expected_record_count} actual={len(rows)}"
        )
    expected_status_counts = config.expected_status_counts or FROZEN_STATUS_COUNTS
    actual_status_counts = Counter(row["inventory_status"] for row in rows)
    if dict(actual_status_counts) != expected_status_counts:
        raise ValueError(
            "Gutenberg review queue status counts do not match the frozen contract"
        )

    config.cache_dir.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    last_download_at: float | None = None
    results: list[dict[str, Any]] = []

    for index, row in enumerate(sorted(rows, key=lambda item: int(item["ebook_id"])), 1):
        cache_path = config.cache_dir / f"pg{row['ebook_id']}.txt"
        try:
            raw_text, fetched_url, cache_status, last_download_at = _load_or_fetch(
                row,
                cache_path,
                config,
                fetch_text=fetch_text,
                session=session,
                sleep=sleep,
                last_download_at=last_download_at,
            )
            result = resolve_gutenberg_metadata_row(row, raw_text)
            result["fetched_url"] = fetched_url
            result["cache_status"] = cache_status
        except Exception as error:
            result = _base_result(row)
            result.update(
                {
                    "cache_status": "error",
                    "fetch_error": f"{type(error).__name__}: {error}",
                    "resolution_status": "blocked_fetch_error",
                    "automatic_decision": "blocked_fetch_error",
                    "manual_review_reasons": "fetch_or_cache_error",
                }
            )
        results.append(result)
        elapsed = monotonic() - started
        eta = elapsed / index * (len(rows) - index)
        _report(
            progress,
            f"record {index:,}/{len(rows):,} ({index / len(rows):.1%}) "
            f"id={row['ebook_id']} status={result['resolution_status']} "
            f"cache={result['cache_status']} elapsed={_duration(elapsed)} "
            f"eta={_duration(eta)}",
        )

    manual_rows = [
        row for row in results if row["resolution_status"] != "automatic_resolved"
    ]
    _write_csv(config.output_csv_path, results)
    _write_csv(config.manual_review_csv_path, manual_rows)
    report = _build_report(
        config,
        results=results,
        queue_sha256=queue_sha256,
    )
    _write_json(config.json_report_path, report)
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(
        render_gutenberg_metadata_review_markdown(report),
        encoding="utf-8",
    )
    return report


def render_gutenberg_metadata_review_markdown(report: dict[str, Any]) -> str:
    """Render the public pass-1A evidence report."""

    lines = [
        "# Project Gutenberg Metadata Resolution Pass 1A",
        "",
        "## Result",
        "",
        (
            f"Acquired and triaged {report['record_count']:,} frozen review records. "
            f"Automatically resolved {report['automatic_resolved_count']:,}; "
            f"retained {report['manual_review_count']:,} for cited review."
        ),
        "",
        "## Resolution Status",
        "",
        "| Status | Records |",
        "| --- | ---: |",
    ]
    for status, count in report["resolution_status_counts"].items():
        lines.append(f"| `{status}` | {count:,} |")
    lines.extend(["", "## Automatic Decisions", "", "| Decision | Records |", "| --- | ---: |"])
    for decision, count in report["automatic_decision_counts"].items():
        lines.append(f"| `{decision}` | {count:,} |")
    lines.extend(["", "## Manual Review Reasons", "", "| Reason | Records |", "| --- | ---: |"])
    for reason, count in report["manual_review_reason_counts"].items():
        lines.append(f"| `{reason}` | {count:,} |")
    lines.extend(
        [
            "",
            "## Evidence Boundaries",
            "",
            "- Gutenberg release and update dates are recorded but never used as work-period evidence.",
            "- Original-publication metadata, explicit first-Italian-version evidence, and qualified title work periods may resolve a row automatically.",
            "- Generic first-edition mentions are recorded but remain non-decisive because front matter can describe another work or language edition.",
            "- A title-page edition year can prove that text existed by 1900, but it cannot silently backdate an ambiguous work.",
            "- Translation routing uses evidence for the Italian version or edition, not the source work's age.",
            "- Language-variety candidates remain manual and outside the standard core.",
            "- This pass activates no text, assigns no V7 split, and freezes no training weight.",
            "",
            "## Artifacts",
            "",
            f"- Complete evidence: `{report['outputs']['output_csv_path']}`",
            f"- Manual queue: `{report['outputs']['manual_review_csv_path']}`",
            f"- Machine report: `{report['outputs']['json_report_path']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _resolved_result(
    result: dict[str, Any],
    row: dict[str, str],
    evidence: dict[str, Any],
    *,
    forced_period: str | None = None,
) -> dict[str, Any]:
    period = forced_period or _period_from_evidence(evidence)
    role = _role_for_period(row["preliminary_role"], period)
    decision = "exclude_post_1900_original_text"
    if period == "origins_through_1800":
        decision = "eligible_historical_core_candidate"
    elif period == "nineteenth_century_bridge":
        decision = "eligible_nineteenth_century_candidate"
    result.update(
        {
            "selected_evidence_kind": evidence["kind"],
            "selected_evidence_year_start": evidence["year_start"],
            "selected_evidence_year_end": evidence["year_end"],
            "selected_evidence_text": evidence["text"],
            "evidence_confidence": evidence["confidence"],
            "resolved_period_bucket": period,
            "resolved_role": role,
            "automatic_decision": decision,
            "resolution_status": "automatic_resolved",
            "manual_review_reasons": "",
        }
    )
    return result


def _manual_result(result: dict[str, Any], reason: str) -> dict[str, Any]:
    result.update(
        {
            "automatic_decision": "manual_authoritative_review_required",
            "resolution_status": "manual_review",
            "manual_review_reasons": reason,
        }
    )
    return result


def _base_result(row: dict[str, str]) -> dict[str, Any]:
    result = {field: "" for field in RESOLUTION_FIELDS}
    for field in (
        "ebook_id",
        "title",
        "authors",
        "author_birth_years",
        "author_death_years",
        "subjects",
        "bookshelves",
        "preliminary_role",
        "period_bucket",
        "inventory_status",
        "landing_page_url",
        "plain_text_url",
    ):
        result[field] = row[field]
    return result


def _load_or_fetch(
    row: dict[str, str],
    cache_path: Path,
    config: GutenbergMetadataReviewConfig,
    *,
    fetch_text: FetchText,
    session: requests.Session | None,
    sleep: Sleep,
    last_download_at: float | None,
) -> tuple[str, str, str, float | None]:
    if cache_path.is_file():
        raw_text = cache_path.read_text(encoding="utf-8")
        if not raw_text.strip():
            raise ValueError(f"empty cached Gutenberg text: {cache_path}")
        return raw_text, row["plain_text_url"], "hit", last_download_at

    errors: list[str] = []
    for attempt in range(1, config.fetch_attempts + 1):
        if last_download_at is not None and config.request_delay_seconds:
            remaining = config.request_delay_seconds - (monotonic() - last_download_at)
            if remaining > 0:
                sleep(remaining)
        try:
            fetched = fetch_text(
                row["ebook_id"],
                session=session,
                timeout=int(config.request_timeout_seconds),
            )
            last_download_at = monotonic()
            if not fetched.text.strip():
                raise ValueError("downloaded Gutenberg text is empty")
            temporary = cache_path.with_suffix(".txt.tmp")
            temporary.write_text(fetched.text, encoding="utf-8")
            temporary.replace(cache_path)
            return fetched.text, fetched.url, "downloaded", last_download_at
        except Exception as error:
            last_download_at = monotonic()
            errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
    raise RuntimeError("; ".join(errors))


def _extract_title_period_evidence(title: str) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    if _SECONDARY_PERIOD_TITLE.search(title) or not _PRIMARY_PERIOD_TITLE.search(title):
        return evidence
    for match in _CENTURY_ROMAN.finditer(title):
        century = _roman_to_int(match.group(1))
        if 1 <= century <= 21:
            evidence.append(
                _evidence(
                    "title_work_period",
                    (century - 1) * 100 + 1,
                    century * 100,
                    match.group(0),
                    "high",
                )
            )
    lowered = title.casefold()
    for name, century in _NAMED_CENTURIES.items():
        if re.search(rf"\b{name}\b", lowered):
            evidence.append(
                _evidence(
                    "title_work_period",
                    (century - 1) * 100 + 1,
                    century * 100,
                    name,
                    "high",
                )
            )
    return evidence


def _language_variety_evidence(
    row: dict[str, str], cleaned: str
) -> dict[str, Any]:
    sample = " ".join((row["title"], row["subjects"], cleaned[:20_000]))
    counts = {
        name: len(pattern.findall(sample))
        for name, pattern in _LANGUAGE_VARIETY_PATTERNS.items()
    }
    return {
        "marker_counts": {name: count for name, count in counts.items() if count},
        "sample_character_count": min(len(cleaned), 20_000),
    }


def _explicit_marker_years(lines: list[str], index: int) -> list[int]:
    years: list[int] = []
    for offset, line in enumerate(lines[index : index + 7]):
        if offset and _LATER_EDITION.search(line):
            break
        years.extend(_years(line))
    return years


def _split_header_and_body(raw_text: str) -> tuple[str, str]:
    lines = raw_text.splitlines()
    for index, line in enumerate(lines):
        if _START_MARKER.search(line):
            return "\n".join(lines[:index]), "\n".join(lines[index + 1 :])
    return "", raw_text


def _evidence(
    kind: str,
    year_start: int,
    year_end: int,
    text: str,
    confidence: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "year_start": year_start,
        "year_end": year_end,
        "text": text[:500],
        "confidence": confidence,
    }


def _deduplicate_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in items:
        key = (item["kind"], item["year_start"], item["year_end"], item["text"])
        unique[key] = item
    return sorted(
        unique.values(),
        key=lambda item: (
            item["year_start"],
            item["year_end"],
            item["kind"],
            item["text"],
        ),
    )


def _first_evidence(
    evidence: list[dict[str, Any]], kind: str
) -> dict[str, Any] | None:
    matches = [item for item in evidence if item["kind"] == kind]
    return min(matches, key=lambda item: (item["year_start"], item["year_end"])) if matches else None


def _period_from_evidence(evidence: dict[str, Any]) -> str:
    year_end = int(evidence["year_end"])
    if year_end <= 1800:
        return "origins_through_1800"
    if year_end <= 1900:
        return "nineteenth_century_bridge"
    return "post_1900_excluded"


def _role_for_period(preliminary_role: str, period: str) -> str:
    if period == "post_1900_excluded":
        return "excluded_post_1900_original_text"
    if preliminary_role == "sonnet_specialization_candidate":
        return "sonnet_specialization_candidate"
    if preliminary_role == "historical_non_sonnet_poetry_candidate":
        return "historical_non_sonnet_poetry_candidate"
    if period == "nineteenth_century_bridge":
        return "nineteenth_century_bridge_candidate"
    return "historical_general_candidate"


def _years(value: str) -> list[int]:
    return [int(match.group(1)) for match in _YEAR.finditer(value)]


def _standalone_year(line: str) -> int | None:
    match = re.fullmatch(r"[_*\s]*(1[2-9]\d{2}|20\d{2})[.,;:_*\s]*", line)
    return int(match.group(1)) if match is not None else None


def _standalone_roman_year(line: str) -> int | None:
    candidate = line.strip(" .,_*-")
    if not _ROMAN_YEAR.fullmatch(candidate):
        return None
    value = _roman_to_int(candidate)
    return value if 1200 <= value <= 2099 else None


def _roman_to_int(value: str) -> int:
    numbers = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for character in reversed(value.upper()):
        current = numbers.get(character, 0)
        total += -current if current < previous else current
        previous = max(previous, current)
    return total


def _parse_year_list(value: str) -> list[int]:
    return [int(part) for part in value.split(";") if part.strip().isdigit()]


def _validate_config(config: GutenbergMetadataReviewConfig) -> None:
    if config.request_delay_seconds < 0:
        raise ValueError("request_delay_seconds cannot be negative")
    if config.request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive")
    if config.fetch_attempts <= 0:
        raise ValueError("fetch_attempts must be positive")


def _read_queue(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != QUEUE_FIELDS:
            raise ValueError("Gutenberg metadata-review queue schema mismatch")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Gutenberg metadata-review queue is empty: {path}")
    if len({row["ebook_id"] for row in rows}) != len(rows):
        raise ValueError("Gutenberg metadata-review queue contains duplicate IDs")
    if any(row["inventory_status"] not in REVIEW_STATUS_EVIDENCE for row in rows):
        raise ValueError("Gutenberg metadata-review queue has an unsupported status")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESOLUTION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _build_report(
    config: GutenbergMetadataReviewConfig,
    *,
    results: list[dict[str, Any]],
    queue_sha256: str,
) -> dict[str, Any]:
    resolution_counts = Counter(row["resolution_status"] for row in results)
    decision_counts = Counter(
        row["automatic_decision"]
        for row in results
        if row["resolution_status"] == "automatic_resolved"
    )
    manual_counts = Counter(
        row["manual_review_reasons"]
        for row in results
        if row["manual_review_reasons"]
    )
    evidence_counts = Counter(
        row["selected_evidence_kind"]
        for row in results
        if row["selected_evidence_kind"]
    )
    return {
        "resolution_version": "project_gutenberg_metadata_resolution_v1",
        "record_count": len(results),
        "automatic_resolved_count": resolution_counts["automatic_resolved"],
        "manual_review_count": len(results) - resolution_counts["automatic_resolved"],
        "resolution_status_counts": dict(sorted(resolution_counts.items())),
        "automatic_decision_counts": dict(sorted(decision_counts.items())),
        "manual_review_reason_counts": dict(sorted(manual_counts.items())),
        "selected_evidence_kind_counts": dict(sorted(evidence_counts.items())),
        "cache_status_counts": dict(
            sorted(Counter(row["cache_status"] for row in results).items())
        ),
        "resolved_role_counts": dict(
            sorted(
                Counter(
                    row["resolved_role"] for row in results if row["resolved_role"]
                ).items()
            )
        ),
        "outputs": {
            "queue_csv_path": _portable(config.queue_csv_path, config.repo_root),
            "queue_csv_sha256": queue_sha256,
            "output_csv_path": _portable(config.output_csv_path, config.repo_root),
            "output_csv_sha256": _sha256_file(config.output_csv_path),
            "manual_review_csv_path": _portable(
                config.manual_review_csv_path, config.repo_root
            ),
            "manual_review_csv_sha256": _sha256_file(config.manual_review_csv_path),
            "json_report_path": _portable(config.json_report_path, config.repo_root),
            "markdown_report_path": _portable(
                config.markdown_report_path, config.repo_root
            ),
        },
        "policy": {
            "activation_authorized": False,
            "release_dates_used_as_period_evidence": False,
            "ambiguous_rows_require_authoritative_review": True,
            "translations_use_italian_edition_period": True,
            "language_varieties_excluded_from_standard_core": True,
        },
    }


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


def _duration(seconds: float) -> str:
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
