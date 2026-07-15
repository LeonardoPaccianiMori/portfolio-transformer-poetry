"""Probe audited Italian Wikisource work collections without activating them."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

from .italian_wikisource import FetchedItalianWikisourceWork, fetch_italian_wikisource_work
from .pretraining_manifest import PretrainingSourceRow, read_pretraining_manifest


FetchItalianWikisourceWork = Callable[..., FetchedItalianWikisourceWork]


@dataclass(frozen=True)
class WorkBoundaries:
    """Recorded first and last subpages expected for one audited work."""

    first_subpage: str
    last_subpage: str
    selected_subpage_titles: tuple[str, ...] = ()


WORK_BOUNDARIES = {
    "ws_galileo_saggiatore": WorkBoundaries(
        first_subpage="Il Saggiatore/Dedica",
        last_subpage="Il Saggiatore/53",
    ),
    "ws_galileo_dialogo": WorkBoundaries(
        first_subpage="Dialogo sopra i due massimi sistemi del mondo tolemaico e copernicano/Dedica",
        last_subpage="Dialogo sopra i due massimi sistemi del mondo tolemaico e copernicano/Giornata quarta",
        selected_subpage_titles=(
            "Dialogo sopra i due massimi sistemi del mondo tolemaico e copernicano/Dedica",
            "Dialogo sopra i due massimi sistemi del mondo tolemaico e copernicano/Al discreto lettore",
            "Dialogo sopra i due massimi sistemi del mondo tolemaico e copernicano/Giornata prima",
            "Dialogo sopra i due massimi sistemi del mondo tolemaico e copernicano/Giornata seconda",
            "Dialogo sopra i due massimi sistemi del mondo tolemaico e copernicano/Giornata terza",
            "Dialogo sopra i due massimi sistemi del mondo tolemaico e copernicano/Giornata quarta",
        ),
    ),
}


@dataclass(frozen=True)
class ItalianWikisourceProbeResult:
    source_id: str
    title: str
    author: str
    landing_page_url: str
    status: str
    error: str
    root_revision_id: int | None
    root_revision_timestamp: str
    page_count: int
    page_revisions: list[dict[str, object]]
    raw_html_character_count: int
    cleaned_character_count: int
    cleaned_word_count: int
    first_characters: str
    last_characters: str
    license_notes: str
    cleaning_notes: str


def probe_italian_wikisource_source(
    *,
    manifest_path: Path,
    source_id: str,
    report_path: Path,
    request_delay: float = 6.0,
    fetch_work: FetchItalianWikisourceWork = fetch_italian_wikisource_work,
    session: requests.Session | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Fetch one audited work and write provenance for manual inspection."""

    started_at = _utc_now()
    rows = read_pretraining_manifest(manifest_path)
    row = select_italian_wikisource_probe_row(rows, source_id)
    boundaries = WORK_BOUNDARIES.get(row.source_id)
    if boundaries is None:
        raise ValueError(f"no recorded Wikisource boundaries for source: {row.source_id}")

    _write_progress(progress, f"probing source: {row.source_id}")
    result = _probe_row(
        row=row,
        boundaries=boundaries,
        request_delay=request_delay,
        fetch_work=fetch_work,
        session=session,
        progress=progress,
    )
    report = {
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "manifest_path": _portable_path(manifest_path),
        "source_id": source_id,
        "activation_status": "audit_then_include",
        "result": asdict(result),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_progress(progress, f"wrote inspection report: {report_path}")
    return report


def select_italian_wikisource_probe_row(
    rows: list[PretrainingSourceRow],
    source_id: str,
) -> PretrainingSourceRow:
    """Return one prose candidate that remains explicitly audit-only."""

    matching = [row for row in rows if row.source_id == source_id]
    if len(matching) != 1:
        raise ValueError(f"expected exactly one manifest row for source: {source_id}")
    row = matching[0]
    if row.source_archive != "Italian Wikisource":
        raise ValueError(f"source is not an Italian Wikisource row: {source_id}")
    if row.inclusion_status != "audit_then_include":
        raise ValueError(f"source is not audit-only: {source_id}")
    if row.text_kind != "prose":
        raise ValueError(f"source is not prose: {source_id}")
    return row


def _probe_row(
    *,
    row: PretrainingSourceRow,
    boundaries: WorkBoundaries,
    request_delay: float,
    fetch_work: FetchItalianWikisourceWork,
    session: requests.Session | None,
    progress: Callable[[str], None] | None,
) -> ItalianWikisourceProbeResult:
    try:
        fetched = fetch_work(
            row.landing_page_url,
            expected_title=row.title,
            expected_first_subpage=boundaries.first_subpage,
            expected_last_subpage=boundaries.last_subpage,
            selected_subpage_titles=list(boundaries.selected_subpage_titles) or None,
            request_delay=request_delay,
            session=session,
            progress=progress,
        )
    except Exception as exc:
        return ItalianWikisourceProbeResult(
            source_id=row.source_id,
            title=row.title,
            author=row.author,
            landing_page_url=row.landing_page_url,
            status="error",
            error=str(exc),
            root_revision_id=None,
            root_revision_timestamp="",
            page_count=0,
            page_revisions=[],
            raw_html_character_count=0,
            cleaned_character_count=0,
            cleaned_word_count=0,
            first_characters="",
            last_characters="",
            license_notes=row.license_notes,
            cleaning_notes=row.cleaning_notes,
        )

    return ItalianWikisourceProbeResult(
        source_id=row.source_id,
        title=row.title,
        author=row.author,
        landing_page_url=row.landing_page_url,
        status="ok",
        error="",
        root_revision_id=fetched.root_revision.revision_id,
        root_revision_timestamp=fetched.root_revision.revision_timestamp,
        page_count=len(fetched.page_revisions),
        page_revisions=[asdict(revision) for revision in fetched.page_revisions],
        raw_html_character_count=fetched.raw_html_character_count,
        cleaned_character_count=len(fetched.text),
        cleaned_word_count=len(re.findall(r"\S+", fetched.text)),
        first_characters=fetched.text[:240],
        last_characters=fetched.text[-240:],
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


def _write_progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
