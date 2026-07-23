"""Probe audited Italian Wikisource work collections without activating them."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

from .italian_wikisource import FetchedItalianWikisourceWork, fetch_italian_wikisource_work
from .pretraining_manifest import PretrainingSourceRow, read_pretraining_manifest


FetchItalianWikisourceWork = Callable[..., FetchedItalianWikisourceWork]

EDITORIAL_MARKER_PATTERNS = {
    "bracketed_text": re.compile(r"\[[^\]\n]{1,240}\]"),
    "si_veda_reference": re.compile(r"(?im)^(?:\d+\s+)?Si veda\b[^\n]*"),
}


@dataclass(frozen=True)
class WorkBoundaries:
    """Recorded stable boundaries and selection rules for one audited work."""

    first_subpage: str = ""
    last_subpage: str = ""
    selected_subpage_titles: tuple[str, ...] = ()
    excluded_subpage_prefixes: tuple[str, ...] = ()
    root_page_title: str = ""
    page_namespace_links: bool = False
    recursive_subpages: bool = False


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
    "ws_beccaria_delitti_pene": WorkBoundaries(
        first_subpage="Dei delitti e delle pene/A chi legge",
    ),
    "ws_vico_scienza_nuova": WorkBoundaries(
        first_subpage="La scienza nuova - Volume I/Titolo",
        excluded_subpage_prefixes=(
            "La scienza nuova - Volume I/Dedica dell'editore",
            "La scienza nuova - Volume I/Introduzione dell'editore",
            "La scienza nuova - Volume I/Illustrazione",
        ),
        root_page_title="La scienza nuova - Volume I",
    ),
    "ws_giannone_istoria_civile_vol1": WorkBoundaries(
        first_subpage="Pagina:Giannone - Istoria civile del regno di Napoli, 1770, Vol.1.djvu/i",
        last_subpage="Pagina:Giannone - Istoria civile del regno di Napoli, 1770, Vol.1.djvu/552",
        root_page_title="Indice:Giannone - Istoria civile del regno di Napoli, 1770, Vol.1.djvu",
        page_namespace_links=True,
    ),
    "ws_giannone_istoria_civile_vol2": WorkBoundaries(
        root_page_title="Indice:Giannone - Istoria civile del regno di Napoli, 1770, Vol.2.djvu",
        page_namespace_links=True,
    ),
    "ws_giannone_istoria_civile_vol3": WorkBoundaries(
        root_page_title="Indice:Giannone - Istoria civile del regno di Napoli, 1770, Vol.3.djvu",
        page_namespace_links=True,
    ),
    "ws_giannone_istoria_civile_vol4": WorkBoundaries(
        root_page_title="Indice:Giannone - Istoria civile del regno di Napoli, 1770, Vol.4.djvu",
        page_namespace_links=True,
    ),
    "ws_giannone_istoria_civile_vol5": WorkBoundaries(
        first_subpage="Pagina:Giannone - Istoria civile del regno di Napoli, 1770, Vol.5.djvu/i",
        last_subpage="Pagina:Giannone - Istoria civile del regno di Napoli, 1770, Vol.5.djvu/679",
        root_page_title="Indice:Giannone - Istoria civile del regno di Napoli, 1770, Vol.5.djvu",
        page_namespace_links=True,
    ),
    "ws_sarpi_istoria_concilio": WorkBoundaries(
        first_subpage="Istoria del Concilio tridentino/Libro primo",
        recursive_subpages=True,
    ),
    "ws_verri_storia_milano": WorkBoundaries(
        first_subpage="Storia di Milano/Dedicatoria",
    ),
    "ws_verri_osservazioni_tortura": WorkBoundaries(
        first_subpage="Osservazioni sulla tortura/I",
    ),
    "ws_verri_meditazioni_economia": WorkBoundaries(
        first_subpage="Meditazioni sulla economia politica/I",
    ),
    "ws_verri_discorso_piacere": WorkBoundaries(
        first_subpage="Discorso sull'indole del piacere e del dolore/I",
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
    marker_summary: dict[str, object]


def probe_italian_wikisource_source(
    *,
    manifest_path: Path,
    source_id: str,
    report_path: Path,
    request_delay: float = 6.0,
    max_samples_per_marker: int = 10,
    fetch_work: FetchItalianWikisourceWork = fetch_italian_wikisource_work,
    session: requests.Session | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Fetch one audited work and write provenance for manual inspection."""

    if max_samples_per_marker < 1:
        raise ValueError("max_samples_per_marker must be at least one")
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
        max_samples_per_marker=max_samples_per_marker,
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


def probe_italian_wikisource_sources(
    *,
    manifest_path: Path,
    source_ids: list[str],
    report_path: Path,
    request_delay: float = 6.0,
    max_samples_per_marker: int = 10,
    fetch_work: FetchItalianWikisourceWork = fetch_italian_wikisource_work,
    session: requests.Session | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Audit several declared Wikisource candidates without changing activation."""

    if not source_ids:
        raise ValueError("source_ids must not be empty")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source_ids must not contain duplicates")
    if max_samples_per_marker < 1:
        raise ValueError("max_samples_per_marker must be at least one")

    started_at = _utc_now()
    rows = read_pretraining_manifest(manifest_path)
    results: list[ItalianWikisourceProbeResult] = []
    for index, source_id in enumerate(source_ids, start=1):
        _write_progress(progress, f"auditing source {index}/{len(source_ids)}: {source_id}")
        row = select_italian_wikisource_probe_row(rows, source_id)
        boundaries = WORK_BOUNDARIES.get(row.source_id)
        if boundaries is None:
            raise ValueError(f"no recorded Wikisource boundaries for source: {row.source_id}")
        results.append(
            _probe_row(
                row=row,
                boundaries=boundaries,
                request_delay=request_delay,
                fetch_work=fetch_work,
                session=session,
                progress=progress,
                max_samples_per_marker=max_samples_per_marker,
            )
        )

    report = {
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "manifest_path": _portable_path(manifest_path),
        "source_ids": source_ids,
        "activation_status": "audit_then_include",
        "max_samples_per_marker": max_samples_per_marker,
        "successful_sources": sum(result.status == "ok" for result in results),
        "error_sources": sum(result.status == "error" for result in results),
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
    _write_progress(progress, f"wrote batch inspection report: {report_path}")
    return report


def audit_italian_wikisource_editorial_markers(
    *,
    manifest_path: Path,
    source_id: str,
    report_path: Path,
    request_delay: float = 6.0,
    max_samples_per_marker: int = 20,
    fetch_work: FetchItalianWikisourceWork = fetch_italian_wikisource_work,
    session: requests.Session | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Write a bounded, revision-pinned audit of candidate editorial markers."""

    if max_samples_per_marker < 1:
        raise ValueError("max_samples_per_marker must be at least one")

    started_at = _utc_now()
    rows = read_pretraining_manifest(manifest_path)
    row = select_italian_wikisource_probe_row(rows, source_id)
    boundaries = WORK_BOUNDARIES.get(row.source_id)
    if boundaries is None:
        raise ValueError(f"no recorded Wikisource boundaries for source: {row.source_id}")

    _write_progress(progress, f"auditing candidate editorial markers: {row.source_id}")
    try:
        fetched = _fetch_row_work(
            row=row,
            boundaries=boundaries,
            request_delay=request_delay,
            fetch_work=fetch_work,
            session=session,
            progress=progress,
        )
        result: dict[str, object] = {
            "status": "ok",
            "error": "",
            "root_revision": asdict(fetched.root_revision),
            "page_count": len(fetched.page_revisions),
            "page_revisions": [asdict(revision) for revision in fetched.page_revisions],
            "cleaned_character_count": len(fetched.text),
            "marker_summary": find_editorial_markers(
                fetched.text,
                max_samples_per_marker=max_samples_per_marker,
            ),
        }
    except Exception as exc:
        result = {
            "status": "error",
            "error": str(exc),
            "root_revision": None,
            "page_count": 0,
            "page_revisions": [],
            "cleaned_character_count": 0,
            "marker_summary": {"counts": {}, "samples": []},
        }

    report = {
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "manifest_path": _portable_path(manifest_path),
        "source_id": source_id,
        "activation_status": "audit_then_include",
        "max_samples_per_marker": max_samples_per_marker,
        "result": result,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_progress(progress, f"wrote marker audit report: {report_path}")
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
    max_samples_per_marker: int,
) -> ItalianWikisourceProbeResult:
    try:
        fetched = _fetch_row_work(
            row=row,
            boundaries=boundaries,
            request_delay=request_delay,
            fetch_work=fetch_work,
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
            marker_summary={"counts": {}, "samples": []},
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
        marker_summary=find_editorial_markers(
            fetched.text,
            max_samples_per_marker=max_samples_per_marker,
        ),
    )


def find_editorial_markers(
    text: str,
    *,
    max_samples_per_marker: int,
) -> dict[str, object]:
    """Count candidate editorial markers and retain bounded page-level samples."""

    counts: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    samples_per_marker: Counter[str] = Counter()
    for page_title, page_text in _split_work_text_by_page(text):
        for marker_type, pattern in EDITORIAL_MARKER_PATTERNS.items():
            for match in pattern.finditer(page_text):
                counts[marker_type] += 1
                if samples_per_marker[marker_type] >= max_samples_per_marker:
                    continue
                samples_per_marker[marker_type] += 1
                samples.append(
                    {
                        "marker_type": marker_type,
                        "page_title": page_title,
                        "matched_text": match.group(0),
                        "context": _marker_context(page_text, match.start(), match.end()),
                    }
                )

    return {"counts": dict(counts), "samples": samples}


def _fetch_row_work(
    *,
    row: PretrainingSourceRow,
    boundaries: WorkBoundaries,
    request_delay: float,
    fetch_work: FetchItalianWikisourceWork,
    session: requests.Session | None,
    progress: Callable[[str], None] | None,
) -> FetchedItalianWikisourceWork:
    return fetch_work(
        row.landing_page_url,
        expected_title=boundaries.root_page_title or row.title,
        expected_first_subpage=boundaries.first_subpage,
        expected_last_subpage=boundaries.last_subpage,
        selected_subpage_titles=list(boundaries.selected_subpage_titles) or None,
        excluded_subpage_prefixes=boundaries.excluded_subpage_prefixes,
        page_namespace_links=boundaries.page_namespace_links,
        recursive_subpages=boundaries.recursive_subpages,
        request_delay=request_delay,
        session=session,
        progress=progress,
    )


def _split_work_text_by_page(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"^## (.+)\n\n", text, flags=re.MULTILINE)
    if len(parts) < 3 or parts[0] != "":
        raise ValueError("Wikisource work text does not use expected page headings")
    return list(zip(parts[1::2], parts[2::2], strict=True))


def _marker_context(text: str, start: int, end: int) -> str:
    context_start = max(0, start - 120)
    context_end = min(len(text), end + 120)
    return re.sub(r"\s+", " ", text[context_start:context_end]).strip()


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
