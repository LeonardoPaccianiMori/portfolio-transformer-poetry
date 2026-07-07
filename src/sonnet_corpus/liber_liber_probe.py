"""Probe and attribution reporting for Liber Liber pretraining sources."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

from .liber_liber import (
    FetchedLiberLiberText,
    LiberLiberRateLimiter,
    fetch_liber_liber_text,
)
from .pretraining_manifest import PretrainingSourceRow, read_pretraining_manifest


FetchLiberLiberText = Callable[..., FetchedLiberLiberText]
CC_BY_NC_SA_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"


@dataclass(frozen=True)
class LiberLiberProbeResult:
    source_id: str
    title: str
    author: str
    landing_page_url: str
    status: str
    error: str
    archive_format: str
    archive_url: str
    raw_byte_count: int
    cleaned_character_count: int
    cleaned_word_count: int
    license_notes: str
    cleaning_notes: str


def probe_liber_liber_sources(
    *,
    manifest_path: Path,
    report_path: Path,
    attribution_path: Path,
    request_delay: float = 1.0,
    fetch_text: FetchLiberLiberText = fetch_liber_liber_text,
    session: requests.Session | None = None,
) -> dict[str, object]:
    """Probe active Liber Liber prose rows and write reports."""

    started_at = _utc_now()
    rows = read_pretraining_manifest(manifest_path)
    probe_rows = select_liber_liber_probe_rows(rows)
    rate_limiter = LiberLiberRateLimiter(request_delay=request_delay)
    results: list[LiberLiberProbeResult] = []

    for row in probe_rows:
        rate_limiter.wait()
        results.append(
            _probe_liber_liber_row(
                row=row,
                fetch_text=fetch_text,
                session=session,
            )
        )

    report = {
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "manifest_path": _portable_path(manifest_path),
        "selected_rows": len(probe_rows),
        "skipped_rows": len(rows) - len(probe_rows),
        "successful_rows": sum(result.status == "ok" for result in results),
        "error_rows": sum(result.status == "error" for result in results),
        "total_cleaned_characters": sum(
            result.cleaned_character_count for result in results
        ),
        "total_cleaned_words": sum(result.cleaned_word_count for result in results),
        "results": [asdict(result) for result in results],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_liber_liber_attribution(probe_rows, attribution_path)
    return report


def select_liber_liber_probe_rows(
    rows: list[PretrainingSourceRow],
) -> list[PretrainingSourceRow]:
    """Return active prose-only Liber Liber rows."""

    return [
        row
        for row in rows
        if row.source_archive == "Liber Liber"
        and row.inclusion_status == "include_probe"
        and row.text_kind == "prose"
    ]


def write_liber_liber_attribution(
    rows: list[PretrainingSourceRow],
    path: Path,
) -> None:
    """Write attribution for the approved Creative Commons source set."""

    lines = [
        "# Broader Prose Corpus Attribution",
        "",
        "These Liber Liber digital editions are approved source candidates for the",
        "historical Italian pretraining corpus. They are licensed under",
        f"[CC BY-NC-SA 4.0]({CC_BY_NC_SA_URL}). Attribution, non-commercial use,",
        "and share-alike conditions apply to this source track.",
        "",
        "Cleaning removes archive wrappers and modern paratext where separable;",
        "primary-text spelling and punctuation are otherwise preserved.",
        "",
    ]

    for row in rows:
        lines.extend(
            [
                f"## {row.title}",
                "",
                f"- Author: {row.author}",
                f"- Source: [Liber Liber]({row.landing_page_url})",
                f"- Period: {row.approx_date}",
                f"- Genre: {row.genre}",
                f"- License: {row.license_notes}",
                f"- Edition and contributors: {row.edition_notes}",
                f"- Planned changes: {row.cleaning_notes}",
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _probe_liber_liber_row(
    *,
    row: PretrainingSourceRow,
    fetch_text: FetchLiberLiberText,
    session: requests.Session | None,
) -> LiberLiberProbeResult:
    try:
        fetched = fetch_text(
            row.landing_page_url,
            title=row.title,
            session=session,
        )
    except Exception as exc:
        return LiberLiberProbeResult(
            source_id=row.source_id,
            title=row.title,
            author=row.author,
            landing_page_url=row.landing_page_url,
            status="error",
            error=str(exc),
            archive_format="",
            archive_url="",
            raw_byte_count=0,
            cleaned_character_count=0,
            cleaned_word_count=0,
            license_notes=row.license_notes,
            cleaning_notes=row.cleaning_notes,
        )

    return LiberLiberProbeResult(
        source_id=row.source_id,
        title=row.title,
        author=row.author,
        landing_page_url=row.landing_page_url,
        status="ok",
        error="",
        archive_format=fetched.archive_format,
        archive_url=fetched.archive_url,
        raw_byte_count=fetched.raw_byte_count,
        cleaned_character_count=len(fetched.text),
        cleaned_word_count=len(re.findall(r"\S+", fetched.text)),
        license_notes=row.license_notes,
        cleaning_notes=row.cleaning_notes,
    )


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)
