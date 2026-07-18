"""Audit Italian Wikisource sonnet collections without changing the corpus."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .cleaning import clean_poem_text, count_poem_lines
from .italian_wikisource import (
    FetchedItalianWikisourcePageCollection,
    fetch_italian_wikisource_page_collection,
)
from .wikisource import extract_poem_text, url_from_title


FetchPageCollection = Callable[..., FetchedItalianWikisourcePageCollection]


@dataclass(frozen=True)
class SonnetWikisourceSource:
    """One source row approved for an audit-only sonnet probe."""

    source_id: str
    title: str
    author: str
    source_archive: str
    landing_page_url: str
    language_variety: str
    role: str
    status: str
    license_or_reuse_status: str
    attribution_required: str


@dataclass(frozen=True)
class SonnetCollectionExpectation:
    """Stable collection identity known before an initial page-level audit."""

    root_page_title: str
    expected_first_subpage: str = ""
    expected_last_subpage: str = ""
    explicit_page_titles: tuple[str, ...] = ()


SONNET_COLLECTION_EXPECTATIONS = {
    "ws_alfieri_rime_1912": SonnetCollectionExpectation(
        root_page_title="Rime varie (Alfieri, 1912)",
    ),
    "ws_foscolo_sonetti": SonnetCollectionExpectation(
        root_page_title="Opera:Sonetti (Foscolo)",
        explicit_page_titles=(
            "Opera:Alla Sera",
            "Opera:Non son chi fui; perì di noi gran parte",
            "Opera:Te nudrice alle Muse, ospite e Dea",
            "Opera:Perchè taccia il rumor di mia catena",
            "Opera:Così gl'interi giorni in lungo, incerto",
            "Opera:Meritamente, però ch'io potei",
            "Opera:Solcata ho fronte, occhi incavati intenti",
            "Opera:E tu ne' carmi avrai perenne vita",
            "Opera:A Zacinto",
            "Opera:In morte del fratello Giovanni",
            "Opera:Alla Musa (Foscolo)",
            "Opera:Che stai? già il secol l'orma ultima lascia",
        ),
    ),
}


def probe_sonnet_wikisource_source(
    *,
    source_manifest_path: Path,
    active_poems_manifest_path: Path,
    repo_root: Path,
    source_id: str,
    report_path: Path,
    request_delay: float = 1.0,
    fetch_collection: FetchPageCollection = fetch_italian_wikisource_page_collection,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Write a local, revision-pinned inspection report for one collection.

    The report contains page metadata, line-count decisions, exact duplicate
    identifiers, and bounded samples. It deliberately excludes full raw and
    cleaned text, so this audit cannot be mistaken for corpus activation.
    """

    source = select_sonnet_wikisource_source(read_sonnet_source_manifest(source_manifest_path), source_id)
    expectation = SONNET_COLLECTION_EXPECTATIONS.get(source.source_id)
    if expectation is None:
        raise ValueError(f"no sonnet collection expectation for source: {source.source_id}")

    started_at = utc_now()
    _write_progress(progress, f"probing source: {source.source_id}")
    active_texts = read_active_poem_texts(active_poems_manifest_path, repo_root)
    active_poem_count = sum(len(poem_ids) for poem_ids in active_texts.values())
    _write_progress(
        progress,
        "loaded "
        f"{active_poem_count} active poems across {len(active_texts)} "
        "exact-text fingerprints for duplicate checks",
    )
    collection = fetch_collection(
        source.landing_page_url,
        expected_title=expectation.root_page_title,
        expected_first_subpage=expectation.expected_first_subpage,
        expected_last_subpage=expectation.expected_last_subpage,
        explicit_page_titles=list(expectation.explicit_page_titles) or None,
        request_delay=request_delay,
        progress=progress,
    )

    candidates: list[dict[str, object]] = []
    for page in collection.pages:
        raw_text = extract_poem_text(page.html)
        cleaned_text = clean_poem_text(raw_text)
        duplicate_ids = active_texts.get(normalize_poem_for_duplicate_check(cleaned_text), [])
        line_count_clean = count_poem_lines(cleaned_text)
        status = candidate_status(
            raw_text=raw_text,
            line_count_clean=line_count_clean,
            duplicate_ids=duplicate_ids,
        )
        candidates.append(
            {
                "page_title": page.revision.title,
                "page_url": url_from_title(page.revision.title),
                "revision_id": page.revision.revision_id,
                "revision_timestamp": page.revision.revision_timestamp,
                "raw_character_count": len(raw_text),
                "cleaned_character_count": len(cleaned_text.strip()),
                "line_count_raw": count_poem_lines(raw_text),
                "line_count_clean": line_count_clean,
                "status": status,
                "exact_active_duplicate_poem_ids": duplicate_ids,
                "cleaned_text_sha256": hashlib.sha256(
                    cleaned_text.encode("utf-8")
                ).hexdigest(),
                "first_characters": bounded_sample(cleaned_text, from_end=False),
                "last_characters": bounded_sample(cleaned_text, from_end=True),
            }
        )

    status_counts = Counter(str(candidate["status"]) for candidate in candidates)
    report = {
        "started_at_utc": started_at,
        "source_manifest_path": portable_path(source_manifest_path, repo_root),
        "active_poems_manifest_path": portable_path(active_poems_manifest_path, repo_root),
        "source": asdict(source),
        "activation_status": "audit_then_include",
        "root_revision": asdict(collection.root_revision),
        "page_count": len(candidates),
        "candidate_status_counts": dict(sorted(status_counts.items())),
        "candidates": candidates,
        "finished_at_utc": utc_now(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_progress(progress, f"wrote inspection report: {report_path}")
    return report


def read_sonnet_source_manifest(path: Path) -> list[SonnetWikisourceSource]:
    """Read the public source-decision manifest used by the expansion audit."""

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "source_id",
        "title",
        "author",
        "source_archive",
        "landing_page_url",
        "language_variety",
        "role",
        "status",
        "license_or_reuse_status",
        "attribution_required",
    }
    if not rows or set(rows[0]) < required:
        raise ValueError("sonnet source manifest is missing required columns")
    return [SonnetWikisourceSource(**{field: row[field] for field in required}) for row in rows]


def select_sonnet_wikisource_source(
    rows: list[SonnetWikisourceSource], source_id: str
) -> SonnetWikisourceSource:
    """Select exactly one core Italian-Wikisource source still awaiting audit."""

    matches = [row for row in rows if row.source_id == source_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one sonnet source row: {source_id}")
    source = matches[0]
    if source.source_archive != "Italian Wikisource":
        raise ValueError(f"source is not Italian Wikisource: {source_id}")
    if source.role != "core_standard_italian":
        raise ValueError(f"source is not a core standard-Italian candidate: {source_id}")
    if source.status != "audit_then_include":
        raise ValueError(f"source is not awaiting audit: {source_id}")
    return source


def read_active_poem_texts(manifest_path: Path, repo_root: Path) -> dict[str, list[str]]:
    """Map normalized active corpus text to poem IDs for exact duplicate checks."""

    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    texts: dict[str, list[str]] = {}
    for row in rows:
        if row.get("include_in_training") != "True":
            continue
        clean_path = repo_root / row["clean_text_path"]
        text = clean_path.read_text(encoding="utf-8")
        normalized = normalize_poem_for_duplicate_check(text)
        texts.setdefault(normalized, []).append(row["poem_id"])
    return texts


def normalize_poem_for_duplicate_check(text: str) -> str:
    """Normalize only case and whitespace for conservative exact duplicate checks."""

    return re.sub(r"\s+", " ", text).strip().casefold()


def candidate_status(
    *, raw_text: str,
    line_count_clean: int,
    duplicate_ids: list[str],
) -> str:
    """Return the first activation gate that prevents a page from being eligible."""

    if not raw_text.strip():
        return "empty_after_extraction"
    if line_count_clean != 14:
        return "not_14_cleaned_lines"
    if duplicate_ids:
        return "exact_duplicate_active_corpus"
    return "eligible_14_lines"


def bounded_sample(text: str, *, from_end: bool, limit: int = 240) -> str:
    """Return a bounded audit sample without retaining an entire poem in the report."""

    compact = text.strip()
    if len(compact) <= limit:
        return compact
    if from_end:
        return compact[-limit:]
    return compact[:limit]


def portable_path(path: Path, repo_root: Path) -> str:
    """Use repository-relative paths when possible in a portable report."""

    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def utc_now() -> str:
    """Return a second-precision UTC timestamp for an inspection report."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
