"""Complete metadata inventory for the Italian Project Gutenberg catalog."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep as default_sleep
from typing import Any

import requests

from .gutenberg import GUTENBERG_USER_AGENT


GUTENDEX_ITALIAN_URL = "https://gutendex.com/books/?languages=it"

INVENTORY_FIELDS = (
    "ebook_id",
    "title",
    "authors",
    "author_birth_years",
    "author_death_years",
    "languages",
    "subjects",
    "bookshelves",
    "download_count",
    "copyright",
    "media_type",
    "landing_page_url",
    "plain_text_url",
    "preliminary_role",
    "period_bucket",
    "inventory_status",
    "genre_evidence",
    "period_evidence",
    "language_variety_evidence",
    "translation_evidence",
    "existing_project_source_ids",
    "possible_existing_work_matches",
    "intra_gutenberg_duplicate_ids",
    "notes",
)

_DIALECT = re.compile(
    r"\b(?:dialett\w*|vernacol\w*|romanesc\w*|franco[- ]italian\w*)\b",
    re.IGNORECASE,
)
_DIALECT_SUBJECT = re.compile(
    r"\b(?:italian language\s*--\s*dialects|dialect literature|"
    r"poetry\s*--\s*dialects|dialect poetry)\b",
    re.IGNORECASE,
)
_LANGUAGE_VARIETY_REVIEW_AUTHOR = re.compile(
    r"\b(?:pascarella, cesare|belli, giuseppe gioacchino|"
    r"salustri, carlo alberto|porta, carlo|di giacomo, salvatore)\b",
    re.IGNORECASE,
)
_SONNET = re.compile(r"\b(?:sonett\w*|sonnets?)\b", re.IGNORECASE)
_POETRY = re.compile(
    r"\b(?:poesia|poesie|poetry|poems?|poema|poemi|epic poetry|rime|canzonier\w*|"
    r"liric\w*|versi|verse|ballat\w*|canzon\w*)\b",
    re.IGNORECASE,
)
_TRANSLATION = re.compile(
    r"\b(?:translation\w* into italian|traduzion\w*|tradott\w*|versione italian\w*)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class GutenbergCatalogInventoryConfig:
    repo_root: Path
    snapshot_path: Path
    inventory_csv_path: Path
    json_report_path: Path
    markdown_report_path: Path
    bibit_record_manifest_path: Path
    broader_sources_manifest_path: Path
    sonnet_manifest_path: Path
    request_delay_seconds: float = 0.25
    request_timeout_seconds: float = 60.0


Progress = Callable[[str], None]
Sleep = Callable[[float], None]


def fetch_italian_gutenberg_catalog(
    *,
    session: requests.Session | None = None,
    request_delay_seconds: float = 0.25,
    request_timeout_seconds: float = 60.0,
    progress: Progress | None = None,
    sleep: Sleep = default_sleep,
) -> dict[str, Any]:
    """Fetch every page of the current Italian-language Gutendex result set."""

    if request_delay_seconds < 0:
        raise ValueError("request_delay_seconds cannot be negative")
    if request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive")
    client = session or requests.Session()
    client.headers.update({"User-Agent": GUTENBERG_USER_AGENT})
    url: str | None = GUTENDEX_ITALIAN_URL
    books: list[dict[str, Any]] = []
    expected_count: int | None = None
    expected_pages: int | None = None
    page = 0
    started = monotonic()

    while url:
        if page and request_delay_seconds:
            sleep(request_delay_seconds)
        response = client.get(url, timeout=request_timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError("Gutendex response is missing a results list")
        count = payload.get("count")
        if not isinstance(count, int) or count < 0:
            raise ValueError("Gutendex response has an invalid count")
        if expected_count is None:
            expected_count = count
            page_size = max(1, len(payload["results"]))
            expected_pages = math.ceil(count / page_size)
        elif count != expected_count:
            raise ValueError("Gutendex result count changed during pagination")
        page += 1
        for book in payload["results"]:
            if not isinstance(book, dict):
                raise ValueError("Gutendex result contains a non-object book")
            books.append(book)
        next_url = payload.get("next")
        if next_url is not None and (
            not isinstance(next_url, str)
            or not next_url.startswith("https://gutendex.com/books/")
        ):
            raise ValueError("Gutendex returned an unexpected pagination URL")
        url = next_url
        elapsed = monotonic() - started
        eta = (
            elapsed / page * max(0, expected_pages - page)
            if expected_pages is not None
            else 0.0
        )
        _report(
            progress,
            f"page {page:,}/{expected_pages or '?'} records={len(books):,}/{expected_count:,} "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
        )

    ebook_ids = [_ebook_id(book) for book in books]
    if expected_count != len(books):
        raise ValueError(
            f"Gutendex pagination returned {len(books)} records; expected {expected_count}"
        )
    if len(ebook_ids) != len(set(ebook_ids)):
        raise ValueError("Gutendex Italian catalog contains duplicate eBook IDs")
    return {
        "catalog_url": GUTENDEX_ITALIAN_URL,
        "record_count": len(books),
        "page_count": page,
        "books": books,
    }


def inventory_italian_gutenberg_catalog(
    config: GutenbergCatalogInventoryConfig,
    *,
    session: requests.Session | None = None,
    progress: Progress | None = None,
    sleep: Sleep = default_sleep,
) -> dict[str, Any]:
    """Fetch, classify, cross-reference, and report the complete Italian catalog."""

    started_at = _utc_now()
    catalog = fetch_italian_gutenberg_catalog(
        session=session,
        request_delay_seconds=config.request_delay_seconds,
        request_timeout_seconds=config.request_timeout_seconds,
        progress=progress,
        sleep=sleep,
    )
    references = _load_existing_references(config)
    rows = [classify_gutenberg_book(book) for book in catalog["books"]]
    _attach_existing_matches(rows, references)
    _attach_intra_gutenberg_duplicates(rows)
    snapshot = {
        "inventory_version": "project_gutenberg_italian_catalog_v1",
        "fetched_at_utc": started_at,
        **catalog,
    }
    _write_json(config.snapshot_path, snapshot)
    _write_csv(config.inventory_csv_path, INVENTORY_FIELDS, rows)
    report = _build_report(config, snapshot=snapshot, rows=rows)
    _write_json(config.json_report_path, report)
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(
        render_gutenberg_inventory_markdown(report),
        encoding="utf-8",
    )
    return report


def classify_gutenberg_book(book: dict[str, Any]) -> dict[str, Any]:
    """Assign conservative metadata-only routing evidence to one catalog book."""

    ebook_id = _ebook_id(book)
    title = str(book.get("title", "")).strip()
    if not title:
        raise ValueError(f"Gutendex book {ebook_id} has no title")
    authors = book.get("authors", [])
    if not isinstance(authors, list) or any(not isinstance(author, dict) for author in authors):
        raise ValueError(f"Gutendex book {ebook_id} has invalid authors")
    author_names = [str(author.get("name", "")).strip() for author in authors]
    author_names = [name for name in author_names if name]
    births = [_optional_year(author.get("birth_year")) for author in authors]
    deaths = [_optional_year(author.get("death_year")) for author in authors]
    subjects = _string_list(book.get("subjects"), ebook_id, "subjects")
    bookshelves = _string_list(book.get("bookshelves"), ebook_id, "bookshelves")
    languages = _string_list(book.get("languages"), ebook_id, "languages")
    metadata_text = " ".join((title, *subjects, *bookshelves))
    dialect_match = _DIALECT.search(metadata_text) or _DIALECT_SUBJECT.search(metadata_text)
    author_variety_match = _LANGUAGE_VARIETY_REVIEW_AUTHOR.search(
        " ".join(author_names)
    )
    translation_match = _TRANSLATION.search(metadata_text)
    period_bucket, period_evidence = _period_bucket(authors)
    genre_evidence = ""

    if dialect_match:
        role = "excluded_language_variety_metadata"
        status = "exclude_core_language_variety_metadata"
    elif author_variety_match:
        role = "language_variety_review_required"
        status = "review_language_variety_before_download"
    elif book.get("media_type") != "Text" or "it" not in languages:
        role = "excluded_non_text_or_non_italian"
        status = "exclude_metadata_scope"
    elif book.get("copyright") is True:
        role = "rights_review_required"
        status = "review_rights"
    elif _SONNET.search(metadata_text):
        role = "sonnet_specialization_candidate"
        genre_evidence = _SONNET.search(metadata_text).group(0)
    elif _POETRY.search(metadata_text):
        role = "historical_non_sonnet_poetry_candidate"
        genre_evidence = _POETRY.search(metadata_text).group(0)
    elif period_bucket == "nineteenth_century_bridge":
        role = "nineteenth_century_bridge_candidate"
    elif period_bucket == "origins_through_1800":
        role = "historical_general_candidate"
    else:
        role = "date_and_role_review"

    if not dialect_match and role not in {
        "excluded_non_text_or_non_italian",
        "rights_review_required",
        "language_variety_review_required",
    }:
        if translation_match:
            status = "review_translation_edition_date"
        elif period_bucket == "unknown":
            status = "review_missing_period_evidence"
        elif period_bucket == "author_died_after_1900_review":
            status = "review_work_publication_date"
        else:
            status = "audit_then_deduplicate"

    return {
        "ebook_id": ebook_id,
        "title": title,
        "authors": ";".join(author_names),
        "author_birth_years": ";".join("" if year is None else str(year) for year in births),
        "author_death_years": ";".join("" if year is None else str(year) for year in deaths),
        "languages": ";".join(languages),
        "subjects": ";".join(subjects),
        "bookshelves": ";".join(bookshelves),
        "download_count": int(book.get("download_count") or 0),
        "copyright": book.get("copyright"),
        "media_type": str(book.get("media_type", "")),
        "landing_page_url": f"https://www.gutenberg.org/ebooks/{ebook_id}",
        "plain_text_url": _plain_text_url(book.get("formats")),
        "preliminary_role": role,
        "period_bucket": period_bucket,
        "inventory_status": status,
        "genre_evidence": genre_evidence,
        "period_evidence": period_evidence,
        "language_variety_evidence": (
            dialect_match.group(0)
            if dialect_match
            else (
                f"known dialect-literature author: {author_variety_match.group(0)}"
                if author_variety_match
                else ""
            )
        ),
        "translation_evidence": translation_match.group(0) if translation_match else "",
        "existing_project_source_ids": "",
        "possible_existing_work_matches": "",
        "intra_gutenberg_duplicate_ids": "",
        "notes": "Metadata-only role; full-text and edition audit required before activation.",
    }


def _load_existing_references(
    config: GutenbergCatalogInventoryConfig,
) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    for row in _read_csv(config.bibit_record_manifest_path):
        references.append(
            {
                "reference_id": f"bibit:{row['object_id']}",
                "title": row["title"],
                "authors": row["authors"],
                "ebook_id": "",
            }
        )
    for row in _read_csv(config.broader_sources_manifest_path):
        references.append(
            {
                "reference_id": f"broader:{row['source_id']}",
                "title": row["title"],
                "authors": row["author"],
                "ebook_id": row.get("ebook_id", ""),
            }
        )
    for row in _read_csv(config.sonnet_manifest_path):
        references.append(
            {
                "reference_id": f"sonnet:{row['poem_id']}",
                "title": row["title_or_first_line"],
                "authors": row["author"],
                "ebook_id": "",
            }
        )
    return references


def _attach_existing_matches(
    rows: list[dict[str, Any]],
    references: list[dict[str, str]],
) -> None:
    by_ebook_id: dict[str, list[str]] = defaultdict(list)
    by_title: dict[str, list[dict[str, str]]] = defaultdict(list)
    for reference in references:
        if reference["ebook_id"]:
            by_ebook_id[reference["ebook_id"]].append(reference["reference_id"])
        title_key = _normalize_text(reference["title"])
        if title_key:
            by_title[title_key].append(reference)

    for row in rows:
        exact_ids = sorted(set(by_ebook_id.get(str(row["ebook_id"]), [])))
        title_matches = []
        row_author = _author_key(str(row["authors"]))
        for reference in by_title.get(_normalize_text(str(row["title"])), []):
            reference_author = _author_key(reference["authors"])
            if not row_author or not reference_author or row_author & reference_author:
                title_matches.append(reference["reference_id"])
        row["existing_project_source_ids"] = ";".join(exact_ids)
        row["possible_existing_work_matches"] = ";".join(sorted(set(title_matches)))
        if exact_ids:
            row["inventory_status"] = "already_registered_project_gutenberg_source"
        elif title_matches and str(row["inventory_status"]).startswith("audit_"):
            row["inventory_status"] = "deduplicate_before_full_text_audit"


def _attach_intra_gutenberg_duplicates(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
    for row in rows:
        key = (
            _normalize_text(str(row["title"])),
            tuple(sorted(_author_key(str(row["authors"])))),
        )
        groups[key].append(str(row["ebook_id"]))
    for row in rows:
        key = (
            _normalize_text(str(row["title"])),
            tuple(sorted(_author_key(str(row["authors"])))),
        )
        duplicates = sorted(
            (ebook_id for ebook_id in groups[key] if ebook_id != str(row["ebook_id"])),
            key=int,
        )
        row["intra_gutenberg_duplicate_ids"] = ";".join(duplicates)
        if duplicates and row["inventory_status"] == "audit_then_deduplicate":
            row["inventory_status"] = "deduplicate_intra_gutenberg_before_audit"


def _period_bucket(authors: list[dict[str, Any]]) -> tuple[str, str]:
    if not authors:
        return "unknown", "no author dates in catalog metadata"
    death_years = [_optional_year(author.get("death_year")) for author in authors]
    if any(year is None for year in death_years):
        return "unknown", "one or more authors have no death year"
    latest_death = max(year for year in death_years if year is not None)
    if latest_death <= 1800:
        return "origins_through_1800", f"latest catalog author death year={latest_death}"
    if latest_death <= 1900:
        return "nineteenth_century_bridge", f"latest catalog author death year={latest_death}"
    return "author_died_after_1900_review", f"latest catalog author death year={latest_death}"


def _plain_text_url(formats: Any) -> str:
    if not isinstance(formats, dict):
        return ""
    preferred = formats.get("text/plain; charset=utf-8")
    if isinstance(preferred, str):
        return preferred
    for media_type in sorted(formats):
        value = formats[media_type]
        if media_type.startswith("text/plain") and isinstance(value, str):
            return value
    return ""


def _build_report(
    config: GutenbergCatalogInventoryConfig,
    *,
    snapshot: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    role_counts = Counter(str(row["preliminary_role"]) for row in rows)
    period_counts = Counter(str(row["period_bucket"]) for row in rows)
    status_counts = Counter(str(row["inventory_status"]) for row in rows)
    return {
        "inventory_version": "project_gutenberg_italian_catalog_v1",
        "fetched_at_utc": snapshot["fetched_at_utc"],
        "catalog_url": snapshot["catalog_url"],
        "record_count": len(rows),
        "page_count": snapshot["page_count"],
        "preliminary_role_counts": dict(sorted(role_counts.items())),
        "period_bucket_counts": dict(sorted(period_counts.items())),
        "inventory_status_counts": dict(sorted(status_counts.items())),
        "records_with_plain_text_url": sum(bool(row["plain_text_url"]) for row in rows),
        "records_with_existing_project_source_id": sum(
            bool(row["existing_project_source_ids"]) for row in rows
        ),
        "records_with_possible_existing_work_match": sum(
            bool(row["possible_existing_work_matches"]) for row in rows
        ),
        "records_with_intra_gutenberg_duplicate": sum(
            bool(row["intra_gutenberg_duplicate_ids"]) for row in rows
        ),
        "outputs": {
            "snapshot_path": _portable(config.snapshot_path, config.repo_root),
            "snapshot_sha256": _sha256_file(config.snapshot_path),
            "inventory_csv_path": _portable(config.inventory_csv_path, config.repo_root),
            "inventory_csv_sha256": _sha256_file(config.inventory_csv_path),
            "json_report_path": _portable(config.json_report_path, config.repo_root),
            "markdown_report_path": _portable(config.markdown_report_path, config.repo_root),
        },
        "policy": {
            "metadata_only": True,
            "full_text_downloaded": False,
            "activation_authorized": False,
            "translations_require_edition_date_review": True,
            "dialect_metadata_excluded_from_unconditioned_core": True,
            "possible_duplicates_require_text_level_resolution": True,
        },
    }


def render_gutenberg_inventory_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Project Gutenberg Italian Catalog Inventory",
        "",
        "## Result",
        "",
        (
            f"Enumerated all {report['record_count']:,} Italian-language records exposed "
            f"by Gutendex across {report['page_count']:,} pages. This is a metadata gate, "
            "not corpus activation."
        ),
        "",
        "## Preliminary Roles",
        "",
        "| Role | Records |",
        "| --- | ---: |",
    ]
    for role, count in report["preliminary_role_counts"].items():
        lines.append(f"| `{role}` | {count:,} |")
    lines.extend(
        [
            "",
            "## Period Evidence",
            "",
            "| Bucket | Records |",
            "| --- | ---: |",
        ]
    )
    for bucket, count in report["period_bucket_counts"].items():
        lines.append(f"| `{bucket}` | {count:,} |")
    lines.extend(
        [
            "",
            "## Overlap Signals",
            "",
            f"- Existing registered Gutenberg IDs: {report['records_with_existing_project_source_id']:,}",
            f"- Possible existing work matches: {report['records_with_possible_existing_work_match']:,}",
            f"- Intra-Gutenberg duplicate-edition signals: {report['records_with_intra_gutenberg_duplicate']:,}",
            f"- Records with a catalog plain-text URL: {report['records_with_plain_text_url']:,}",
            "",
            "## Boundaries",
            "",
            "- No full text was downloaded by this inventory.",
            "- No record is activated from author dates, title, or subjects alone.",
            "- Translation editions require publication-date and primary-text review.",
            "- Dialect indicators exclude a record only from the unconditioned core.",
            "- Exact and near text deduplication remains required before activation.",
            "",
            "## Artifacts",
            "",
            f"- Raw catalog snapshot: `{report['outputs']['snapshot_path']}`",
            f"- Review inventory: `{report['outputs']['inventory_csv_path']}`",
            f"- Machine-readable report: `{report['outputs']['json_report_path']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _ebook_id(book: dict[str, Any]) -> str:
    value = book.get("id")
    if not isinstance(value, int) or value <= 0:
        raise ValueError("Gutendex result has an invalid eBook ID")
    return str(value)


def _optional_year(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError("Gutendex author year must be an integer or null")
    return value


def _string_list(value: Any, ebook_id: str, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Gutendex book {ebook_id} has invalid {field}")
    return [item.strip() for item in value if item.strip()]


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(_WORD.findall(without_marks))


def _author_key(value: str) -> set[str]:
    return set(_normalize_text(value).split())


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
