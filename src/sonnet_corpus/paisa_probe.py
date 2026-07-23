"""Inspect PAISÀ corpus metadata without downloading the corpus itself."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup


PAISA_DESCRIPTION_URL = "https://www.corpusitaliano.it/en/contents/description.html"


@dataclass(frozen=True)
class PaisaMetadataProbeResult:
    """License and provenance facts required before a PAISÀ corpus decision."""

    source_url: str
    status: str
    error: str
    document_count: int | None
    website_count: int | None
    reported_word_count: int | None
    corpus_license: str
    source_license_families: list[str]
    document_provenance_fields: list[str]
    download_page_url: str
    citation: str


def probe_paisa_metadata(
    *,
    report_path: Path,
    source_url: str = PAISA_DESCRIPTION_URL,
    session: requests.Session | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Write a small metadata report without retrieving PAISÀ text files."""

    started_at = _utc_now()
    _write_progress(progress, f"fetching metadata page: {source_url}")
    try:
        http = session or requests.Session()
        response = http.get(source_url, timeout=30)
        response.raise_for_status()
        result = _parse_metadata_page(source_url, response.text)
    except Exception as exc:
        result = PaisaMetadataProbeResult(
            source_url=source_url,
            status="error",
            error=str(exc),
            document_count=None,
            website_count=None,
            reported_word_count=None,
            corpus_license="",
            source_license_families=[],
            document_provenance_fields=[],
            download_page_url="",
            citation="",
        )

    report = {
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "scope": "metadata_only_no_corpus_download",
        "activation_status": "auxiliary_experiment_not_activated",
        "result": asdict(result),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_progress(progress, f"wrote metadata report: {report_path}")
    return report


def _parse_metadata_page(source_url: str, html: str) -> PaisaMetadataProbeResult:
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    document_count = _extract_number(
        text,
        r"approximately\s+([\d,]+)\s+documents",
    )
    website_count = _extract_number(
        text,
        r"about\s+([\d,]+)\s+different\s+websites",
    )
    reported_word_count = _extract_number(
        text,
        r"about\s+([\d,]+)\s+million\s+words",
    )
    if reported_word_count is not None:
        reported_word_count *= 1_000_000

    if "Attribution-Noncommercial-ShareAlike" not in text:
        raise ValueError("PAISÀ metadata page does not state the corpus license")
    if "Attribution-ShareAlike" not in text:
        raise ValueError("PAISÀ metadata page does not state source license families")
    if '"text" tag with "id" and "url" attributes' not in text:
        raise ValueError("PAISÀ metadata page does not state document provenance fields")

    soup = BeautifulSoup(html, "html.parser")
    download_page_url = ""
    for link in soup.select("a[href]"):
        if "download" in link.get_text(" ", strip=True).lower():
            download_page_url = link["href"]
            break
    if not download_page_url:
        raise ValueError("PAISÀ metadata page does not link a download route")

    citation_match = re.search(r"For citing the corpus:\s*(.+?)\s*\[", text)
    citation = citation_match.group(1).strip() if citation_match else ""
    return PaisaMetadataProbeResult(
        source_url=source_url,
        status="ok",
        error="",
        document_count=document_count,
        website_count=website_count,
        reported_word_count=reported_word_count,
        corpus_license="CC BY-NC-SA",
        source_license_families=["CC BY-SA", "CC BY-NC-SA"],
        document_provenance_fields=["id", "url"],
        download_page_url=download_page_url,
        citation=citation,
    )


def _extract_number(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1).replace(",", ""))


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
