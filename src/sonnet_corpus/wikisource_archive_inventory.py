"""Metadata-first archive inventory for Italian Wikisource."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep
from typing import Any
from urllib.parse import quote, unquote, urlparse

import requests
from bs4 import BeautifulSoup


DUMP_DATE = "20260801"
DUMP_BASE_URL = f"https://dumps.wikimedia.org/itwikisource/{DUMP_DATE}"
SITEINFO_URL = "https://it.wikisource.org/w/api.php"
USER_AGENT = "portfolio-transformer-poetry-wikisource-inventory/0.1"
DUMP_FILES = {
    "page": (
        f"itwikisource-{DUMP_DATE}-page.sql.gz",
        "d346505404381269c65d933c2c1d65031693c615",
    ),
    "categorylinks": (
        f"itwikisource-{DUMP_DATE}-categorylinks.sql.gz",
        "016f85f2a180861a59a673fd17b1021da23f44c7",
    ),
    "linktarget": (
        f"itwikisource-{DUMP_DATE}-linktarget.sql.gz",
        "ba38f03d8c0127667d614e36a11f24ee40bb56f4",
    ),
}

INVENTORY_FIELDS = (
    "work_root_id",
    "root_title",
    "landing_page_url",
    "root_page_id",
    "root_revision_id",
    "root_touched_utc",
    "root_wikitext_bytes",
    "hierarchy_page_count",
    "hierarchy_nonredirect_page_count",
    "projected_wikitext_bytes",
    "category_count",
    "category_evidence",
    "author_evidence",
    "period_evidence",
    "exact_year",
    "period_bucket",
    "language_evidence",
    "language_route",
    "genre_evidence",
    "genre_route",
    "form_evidence",
    "form_route",
    "source_scan_status",
    "existing_reference_ids",
    "metadata_decision",
    "proposed_role",
    "review_reason",
    "site_license",
    "site_license_url",
    "activation_status",
)

PAGE_HIERARCHY_FIELDS = (
    "work_root_id",
    "root_title",
    "page_id",
    "page_title",
    "relative_title",
    "hierarchy_depth",
    "is_root_page",
    "is_redirect",
    "latest_revision_id",
    "touched_utc",
    "wikitext_bytes",
)

GATE_FIELDS = (
    "work_root_id",
    "root_title",
    "author_evidence",
    "period_bucket",
    "language_route",
    "genre_route",
    "form_route",
    "metadata_decision",
    "proposed_role",
    "hierarchy_page_count",
    "projected_wikitext_bytes",
    "projected_archive_share",
    "projected_role_share",
    "author_projected_share",
    "source_scan_status",
    "existing_reference_ids",
    "inspection_status",
    "next_action",
)

SAMPLE_FIELDS = (
    "sample_id",
    "work_root_id",
    "root_title",
    "metadata_decision",
    "proposed_role",
    "representative_page_id",
    "representative_page_title",
    "representative_revision_id",
    "page_wikitext_bytes",
    "rendered_html_characters",
    "visible_text_characters",
    "alphabetic_ratio",
    "italian_function_word_hits",
    "italian_function_word_ratio",
    "primary_text_signal",
    "inspection_decision",
    "source_url",
)

_EXACT_YEAR_RE = re.compile(r"^Testi del (\d{4})$")
_CENTURY_RE = re.compile(r"^Testi del ([IVXLCDM]+) secolo$")
_AUTHOR_PREFIX = "Testi di "
_LANGUAGE_PREFIX = "Testi in "
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
    "non",
    "per",
    "si",
    "un",
    "una",
}
_STANDARD_LANGUAGE_LABELS = {
    "italiano",
    "lingua italiana",
    "volgare italiano",
}
_CONDITIONED_LANGUAGE_TERMS = {
    "abruzzese",
    "bolognese",
    "calabrese",
    "friulano",
    "genovese",
    "lombardo",
    "milanese",
    "napoletano",
    "piemontese",
    "romanesco",
    "sardo",
    "siciliano",
    "toscano vernacolare",
    "veneto",
    "veneziano",
}
_NON_ITALIAN_LANGUAGE_TERMS = {
    "francese",
    "greco",
    "inglese",
    "latino",
    "occitano",
    "spagnolo",
    "tedesco",
}

Progress = Callable[[str], None]


@dataclass(frozen=True)
class WikisourceArchiveInventoryConfig:
    """Pinned metadata inputs and outputs for checkpoint 4A."""

    repo_root: Path
    cache_dir: Path
    inventory_path: Path
    page_hierarchy_path: Path
    composition_gate_path: Path
    inspection_sample_path: Path
    json_report_path: Path
    markdown_report_path: Path
    broader_manifest_path: Path
    poems_manifest_path: Path
    snapshot_dir: Path
    dump_date: str = DUMP_DATE
    dump_base_url: str = DUMP_BASE_URL
    sample_size: int = 30
    request_delay: float = 1.0
    api_retries: int = 6
    progress_interval: int = 25_000


@dataclass(frozen=True)
class WikiPage:
    page_id: int
    namespace: int
    title: str
    is_redirect: bool
    touched: str
    latest_revision_id: int
    wikitext_bytes: int


def build_wikisource_archive_inventory(
    config: WikisourceArchiveInventoryConfig,
    *,
    session: requests.Session | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Build a work-root inventory without acquiring the full page-text dump."""

    _validate_config(config)
    started = monotonic()
    http = session or requests.Session()
    if session is None:
        http.headers.update({"User-Agent": USER_AGENT})

    dump_paths = {
        label: _acquire_dump_file(
            config,
            filename=filename,
            expected_sha1=expected_sha1,
            session=http,
            progress=progress,
        )
        for label, (filename, expected_sha1) in DUMP_FILES.items()
    }
    siteinfo = _load_siteinfo(config, session=http, progress=progress)
    rights = siteinfo["query"]["rightsinfo"]

    pages = _load_main_pages(
        dump_paths["page"],
        progress_interval=config.progress_interval,
        progress=progress,
    )
    roots, page_rows = _group_pages(pages)
    root_page_ids = {
        group[0].page_id
        for group in roots.values()
        if group and group[0].title == _root_title(group[0].title)
    }
    category_targets = _load_category_targets(dump_paths["linktarget"])
    categories = _load_root_categories(
        dump_paths["categorylinks"],
        root_page_ids=root_page_ids,
        category_targets=category_targets,
        progress_interval=config.progress_interval,
        progress=progress,
    )
    existing_references = _load_existing_references(config)

    inventory_rows: list[dict[str, Any]] = []
    representative_pages: dict[str, WikiPage] = {}
    for index, root_title in enumerate(sorted(roots, key=str.casefold), start=1):
        group = roots[root_title]
        root_page = group[0] if group[0].title == root_title else None
        root_categories = sorted(categories.get(root_page.page_id, set())) if root_page else []
        row = _inventory_row(
            root_title,
            group,
            root_page=root_page,
            categories=root_categories,
            existing_reference_ids=existing_references.get(root_title, set()),
            site_license=rights["text"],
            site_license_url=rights["url"],
        )
        inventory_rows.append(row)
        representative_pages[row["work_root_id"]] = _representative_page(group)
        if index == 1 or index % config.progress_interval == 0 or index == len(roots):
            _progress(
                progress,
                "work-root-classification",
                index,
                len(roots),
                started,
                f"title={root_title!r}",
            )

    sample_rows = _inspect_representative_sample(
        config,
        inventory_rows=inventory_rows,
        representative_pages=representative_pages,
        session=http,
        progress=progress,
    )
    sample_status = {
        row["work_root_id"]: row["inspection_decision"] for row in sample_rows
    }
    gate_rows = _build_gate_rows(inventory_rows, sample_status=sample_status)

    _write_csv(config.inventory_path, INVENTORY_FIELDS, inventory_rows)
    _write_csv(config.page_hierarchy_path, PAGE_HIERARCHY_FIELDS, page_rows)
    _write_csv(config.composition_gate_path, GATE_FIELDS, gate_rows)
    _write_csv(config.inspection_sample_path, SAMPLE_FIELDS, sample_rows)
    report = _build_report(
        config,
        dump_paths=dump_paths,
        siteinfo=siteinfo,
        pages=pages,
        inventory_rows=inventory_rows,
        gate_rows=gate_rows,
        sample_rows=sample_rows,
    )
    _write_json(config.json_report_path, report)
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(
        render_wikisource_archive_inventory_markdown(report),
        encoding="utf-8",
    )
    return report


def iter_sql_insert_rows(path: Path, table: str) -> Iterator[list[Any]]:
    """Yield decoded tuples from one Wikimedia MariaDB SQL dump."""

    prefix = f"INSERT INTO `{table}` VALUES "
    # Wikimedia declares these varbinary dumps as UTF-8, but binary sort-key
    # fields can contain isolated malformed bytes. Those fields are irrelevant
    # here; replacement keeps tuple boundaries and numeric IDs intact.
    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as handle:
        for line in handle:
            if not line.startswith(prefix):
                continue
            payload = line[len(prefix) :].rstrip("\r\n;")
            yield from _parse_sql_tuples(payload)


def _parse_sql_tuples(payload: str) -> Iterator[list[Any]]:
    index = 0
    length = len(payload)
    while index < length:
        if payload[index] == ",":
            index += 1
            continue
        if payload[index] != "(":
            raise ValueError(f"expected SQL tuple at offset {index}")
        index += 1
        row: list[Any] = []
        while True:
            if index >= length:
                raise ValueError("unterminated SQL tuple")
            if payload[index] == "'":
                value, index = _parse_sql_string(payload, index + 1)
            else:
                end = index
                while end < length and payload[end] not in {",", ")"}:
                    end += 1
                token = payload[index:end]
                value = None if token == "NULL" else token
                index = end
            row.append(value)
            if index >= length:
                raise ValueError("unterminated SQL tuple")
            if payload[index] == ",":
                index += 1
                continue
            if payload[index] == ")":
                index += 1
                break
            raise ValueError(f"unexpected SQL delimiter at offset {index}")
        yield row


def _parse_sql_string(payload: str, index: int) -> tuple[str, int]:
    result: list[str] = []
    escapes = {
        "0": "\0",
        "b": "\b",
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "Z": "\x1a",
    }
    while index < len(payload):
        char = payload[index]
        if char == "'":
            return "".join(result), index + 1
        if char == "\\":
            index += 1
            if index >= len(payload):
                raise ValueError("unterminated SQL escape")
            escaped = payload[index]
            result.append(escapes.get(escaped, escaped))
            index += 1
            continue
        result.append(char)
        index += 1
    raise ValueError("unterminated SQL string")


def _acquire_dump_file(
    config: WikisourceArchiveInventoryConfig,
    *,
    filename: str,
    expected_sha1: str,
    session: requests.Session,
    progress: Progress | None,
) -> Path:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    path = config.cache_dir / filename
    if path.is_file() and _sha1_file(path) == expected_sha1:
        _emit(progress, f"dump-cache-hit {filename} bytes={path.stat().st_size:,}")
        return path
    if path.exists():
        raise ValueError(f"cached dump hash mismatch: {path}")

    url = f"{config.dump_base_url}/{filename}"
    _emit(progress, f"dump-download-start {filename} url={url}")
    response = session.get(url, stream=True, timeout=120)
    response.raise_for_status()
    temporary = path.with_suffix(path.suffix + ".part")
    hasher = hashlib.sha1()
    byte_count = 0
    next_progress = 8 * 1024 * 1024
    try:
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                hasher.update(chunk)
                byte_count += len(chunk)
                if byte_count >= next_progress:
                    _emit(
                        progress,
                        f"dump-download-progress {filename} bytes={byte_count:,}",
                    )
                    next_progress += 8 * 1024 * 1024
        if hasher.hexdigest() != expected_sha1:
            raise ValueError(f"downloaded dump hash mismatch: {filename}")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    _emit(progress, f"dump-download-complete {filename} bytes={byte_count:,}")
    return path


def _load_siteinfo(
    config: WikisourceArchiveInventoryConfig,
    *,
    session: requests.Session,
    progress: Progress | None,
) -> dict[str, Any]:
    path = config.cache_dir / "siteinfo_rights_v1.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    response = session.get(
        SITEINFO_URL,
        params={
            "action": "query",
            "meta": "siteinfo",
            "siprop": "rightsinfo|general",
            "format": "json",
            "formatversion": "2",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("query", {}).get("rightsinfo", {}).get("url"):
        raise ValueError("Wikisource siteinfo omitted rights metadata")
    _write_json(path, payload)
    _emit(progress, "siteinfo-cache-write rights metadata pinned")
    return payload


def _load_main_pages(
    path: Path,
    *,
    progress_interval: int,
    progress: Progress | None,
) -> list[WikiPage]:
    pages: list[WikiPage] = []
    parsed = 0
    for row in iter_sql_insert_rows(path, "page"):
        parsed += 1
        namespace = int(row[1])
        if namespace == 0:
            pages.append(
                WikiPage(
                    page_id=int(row[0]),
                    namespace=namespace,
                    title=_decode_title(str(row[2])),
                    is_redirect=row[3] == "1",
                    touched=_timestamp_utc(str(row[6])),
                    latest_revision_id=int(row[8]),
                    wikitext_bytes=int(row[9]),
                )
            )
        if parsed % progress_interval == 0:
            _emit(
                progress,
                f"page-metadata parsed={parsed:,} main_namespace={len(pages):,}",
            )
    if not pages:
        raise ValueError("page dump contains no main-namespace rows")
    return pages


def _group_pages(
    pages: list[WikiPage],
) -> tuple[dict[str, list[WikiPage]], list[dict[str, Any]]]:
    groups: dict[str, list[WikiPage]] = defaultdict(list)
    for page in pages:
        groups[_root_title(page.title)].append(page)

    rows: list[dict[str, Any]] = []
    for root_title, group in groups.items():
        group.sort(key=lambda page: (page.title != root_title, page.title.casefold()))
        root_id = _work_root_id(root_title, group)
        for page in group:
            relative = page.title.removeprefix(root_title).removeprefix("/")
            rows.append(
                {
                    "work_root_id": root_id,
                    "root_title": root_title,
                    "page_id": page.page_id,
                    "page_title": page.title,
                    "relative_title": relative,
                    "hierarchy_depth": page.title.count("/"),
                    "is_root_page": page.title == root_title,
                    "is_redirect": page.is_redirect,
                    "latest_revision_id": page.latest_revision_id,
                    "touched_utc": page.touched,
                    "wikitext_bytes": page.wikitext_bytes,
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["root_title"]).casefold(),
            str(row["page_title"]).casefold(),
        )
    )
    return dict(groups), rows


def _load_category_targets(path: Path) -> dict[int, str]:
    targets: dict[int, str] = {}
    for row in iter_sql_insert_rows(path, "linktarget"):
        if int(row[1]) == 14:
            targets[int(row[0])] = _decode_title(str(row[2]))
    if not targets:
        raise ValueError("linktarget dump contains no category targets")
    return targets


def _load_root_categories(
    path: Path,
    *,
    root_page_ids: set[int],
    category_targets: dict[int, str],
    progress_interval: int,
    progress: Progress | None,
) -> dict[int, set[str]]:
    categories: dict[int, set[str]] = defaultdict(set)
    parsed = 0
    for row in iter_sql_insert_rows(path, "categorylinks"):
        parsed += 1
        page_id = int(row[0])
        if page_id in root_page_ids:
            category = category_targets.get(int(row[6]))
            if category:
                categories[page_id].add(category)
        if parsed % progress_interval == 0:
            _emit(
                progress,
                f"category-metadata parsed={parsed:,} roots={len(categories):,}",
            )
    return categories


def _inventory_row(
    root_title: str,
    group: list[WikiPage],
    *,
    root_page: WikiPage | None,
    categories: list[str],
    existing_reference_ids: set[str],
    site_license: str,
    site_license_url: str,
) -> dict[str, Any]:
    evidence = _classify_metadata(root_title, categories)
    projected_bytes = sum(page.wikitext_bytes for page in group if not page.is_redirect)
    source_scan_status = (
        "scan_backed_category_signal_pending_exact_index_provenance"
        if any("versione cartacea a fronte" in value.casefold() for value in categories)
        else "source_scan_not_identified_metadata_only"
    )
    if existing_reference_ids:
        evidence["metadata_decision"] = "existing_project_reference"
        evidence["proposed_role"] = "cross_archive_reference_only"
        evidence["review_reason"] = (
            "Existing Wikisource lineage must be deduplicated, not silently reactivated."
        )
    return {
        "work_root_id": _work_root_id(root_title, group),
        "root_title": root_title,
        "landing_page_url": _url_from_title(root_title),
        "root_page_id": root_page.page_id if root_page else "",
        "root_revision_id": root_page.latest_revision_id if root_page else "",
        "root_touched_utc": root_page.touched if root_page else "",
        "root_wikitext_bytes": root_page.wikitext_bytes if root_page else 0,
        "hierarchy_page_count": len(group),
        "hierarchy_nonredirect_page_count": sum(not page.is_redirect for page in group),
        "projected_wikitext_bytes": projected_bytes,
        "category_count": len(categories),
        "category_evidence": " | ".join(categories),
        **evidence,
        "source_scan_status": source_scan_status,
        "existing_reference_ids": " | ".join(sorted(existing_reference_ids)),
        "site_license": site_license,
        "site_license_url": site_license_url,
        "activation_status": "metadata_only_not_activated",
    }


def _classify_metadata(root_title: str, categories: list[str]) -> dict[str, str]:
    authors = sorted(
        {
            value[len(_AUTHOR_PREFIX) :]
            for value in categories
            if value.startswith(_AUTHOR_PREFIX)
            and not value.startswith("Testi di autori")
            and value[len(_AUTHOR_PREFIX) :][:1].isupper()
        }
    )
    exact_years = sorted(
        {
            int(match.group(1))
            for value in categories
            if (match := _EXACT_YEAR_RE.match(value))
        }
    )
    centuries = sorted(
        {match.group(1) for value in categories if (match := _CENTURY_RE.match(value))}
    )
    languages = sorted(
        {
            value[len(_LANGUAGE_PREFIX) :]
            for value in categories
            if value.startswith(_LANGUAGE_PREFIX)
        }
    )
    period_bucket = _period_bucket(exact_years, centuries)
    language_route = _language_route(languages)
    genre_route, genre_evidence, form_route, form_evidence = _genre_and_form(
        root_title, categories
    )
    translation_signal = any("traduz" in value.casefold() for value in categories)

    if language_route == "explicit_non_italian":
        decision = "exclude_explicit_non_italian"
        role = "excluded"
        reason = "An explicit non-Italian language category prevents standard-core routing."
    elif language_route.startswith("conditioned_"):
        decision = "conditioned_language_candidate"
        role = "conditioned_language_variant"
        reason = "An explicit Italian language-variety category requires a separate experiment."
    elif language_route == "review_explicit_language_variety":
        decision = "hold_language_variety_review"
        role = "metadata_hold"
        reason = "The explicit language category is not safely mapped by metadata alone."
    elif translation_signal:
        decision = "hold_translation_edition_review"
        role = "metadata_hold"
        reason = "Translation routing requires source work and Italian edition dates."
    elif period_bucket == "origins_through_1800":
        decision = "historical_core_metadata_candidate"
        role = (
            "historical_non_sonnet_poetry"
            if genre_route == "poetry_or_mixed_collection"
            else "standard_sonnets"
            if form_route == "sonnet_signal"
            else "historical_general"
        )
        reason = "Metadata supports a pre-1801 candidate pending page and source-scan audit."
    elif period_bucket == "nineteenth_century":
        decision = "nineteenth_century_bridge_metadata_candidate"
        role = (
            "standard_sonnets"
            if form_route == "sonnet_signal"
            else "nineteenth_century_bridge"
        )
        reason = "Metadata supports an 1801-1900 bridge candidate pending the exposure cap."
    elif period_bucket == "post_1900":
        decision = "exclude_post_1900_scope"
        role = "excluded"
        reason = "The work falls outside the approved historical and bridge periods."
    else:
        decision = "hold_period_or_work_identity"
        role = "metadata_hold"
        reason = "Metadata does not establish a safe work date or primary-text identity."

    return {
        "author_evidence": " | ".join(authors),
        "period_evidence": " | ".join(
            [
                *(str(year) for year in exact_years),
                *(f"{value} secolo" for value in centuries),
            ]
        ),
        "exact_year": " | ".join(str(year) for year in exact_years),
        "period_bucket": period_bucket,
        "language_evidence": " | ".join(languages),
        "language_route": language_route,
        "genre_evidence": genre_evidence,
        "genre_route": genre_route,
        "form_evidence": form_evidence,
        "form_route": form_route,
        "metadata_decision": decision,
        "proposed_role": role,
        "review_reason": reason,
    }


def _period_bucket(exact_years: list[int], centuries: list[str]) -> str:
    if len(exact_years) == 1:
        year = exact_years[0]
        if year <= 1800:
            return "origins_through_1800"
        if year <= 1900:
            return "nineteenth_century"
        return "post_1900"
    if len(exact_years) > 1:
        return "conflicting_period_metadata"
    century_numbers = {_roman_to_int(value) for value in centuries}
    if len(century_numbers) != 1:
        return "period_unresolved"
    century = next(iter(century_numbers))
    if century <= 18:
        return "origins_through_1800"
    if century == 19:
        return "nineteenth_century"
    return "post_1900"


def _language_route(languages: list[str]) -> str:
    if not languages:
        return "standard_italian_unmarked_pending_page_check"
    normalized = {value.casefold() for value in languages}
    if normalized <= _STANDARD_LANGUAGE_LABELS:
        return "standard_italian_explicit"
    if normalized & _NON_ITALIAN_LANGUAGE_TERMS:
        return "explicit_non_italian"
    conditioned = normalized & _CONDITIONED_LANGUAGE_TERMS
    if conditioned:
        return "conditioned_" + "_".join(sorted(conditioned)).replace(" ", "_")
    return "review_explicit_language_variety"


def _genre_and_form(
    root_title: str, categories: list[str]
) -> tuple[str, str, str, str]:
    sonnet_evidence = sorted(
        value for value in categories if "sonett" in value.casefold()
    )
    if "sonett" in root_title.casefold():
        sonnet_evidence.append("title:sonett*")
    poetry_terms = ("poes", "poemi", "rime", "liric", "canzoni")
    poetry_evidence = sorted(
        value
        for value in categories
        if any(term in value.casefold() for term in poetry_terms)
    )
    if sonnet_evidence:
        form_route = "sonnet_signal"
        form_evidence = " | ".join(sorted(set(sonnet_evidence)))
    else:
        form_route = "no_explicit_sonnet_signal"
        form_evidence = ""
    if poetry_evidence or sonnet_evidence:
        genre_route = "poetry_or_mixed_collection"
        genre_evidence = " | ".join(sorted(set(poetry_evidence + sonnet_evidence)))
    else:
        genre_route = "general_or_unresolved"
        genre_evidence = ""
    return genre_route, genre_evidence, form_route, form_evidence


def _load_existing_references(
    config: WikisourceArchiveInventoryConfig,
) -> dict[str, set[str]]:
    references: dict[str, set[str]] = defaultdict(set)
    for path, url_field, id_field in (
        (config.broader_manifest_path, "landing_page_url", "source_id"),
        (config.poems_manifest_path, "source_url", "poem_id"),
    ):
        if not path.is_file():
            continue
        for row in _read_csv(path):
            if "wikisource" not in row.get("source_archive", "").casefold():
                continue
            title = _title_from_url(row.get(url_field, ""))
            if not title:
                continue
            root = _root_title(title)
            label = (
                row[id_field]
                if path == config.broader_manifest_path
                else f"{path.stem}:{root}"
            )
            references[root].add(label)
    for path in sorted(config.snapshot_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        title = payload.get("root_revision", {}).get("title") or payload.get("title", "")
        if title:
            references[_root_title(title)].add(payload.get("source_id", path.stem))
    return references


def _inspect_representative_sample(
    config: WikisourceArchiveInventoryConfig,
    *,
    inventory_rows: list[dict[str, Any]],
    representative_pages: dict[str, WikiPage],
    session: requests.Session,
    progress: Progress | None,
) -> list[dict[str, Any]]:
    selected = _select_stratified_sample(inventory_rows, config.sample_size)
    cache_dir = config.cache_dir / "inspection_sample"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    last_request_at = 0.0
    for index, inventory in enumerate(selected, start=1):
        page = representative_pages[inventory["work_root_id"]]
        cache_path = cache_dir / f"page_{page.page_id}_rev_{page.latest_revision_id}.json"
        if cache_path.is_file():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            elapsed = monotonic() - last_request_at
            if elapsed < config.request_delay:
                sleep(config.request_delay - elapsed)
            payload = _api_get_json(
                session,
                params={
                    "action": "parse",
                    "oldid": str(page.latest_revision_id),
                    "prop": "text|revid|displaytitle",
                    "format": "json",
                    "formatversion": "2",
                },
                retries=config.api_retries,
                request_delay=config.request_delay,
                progress=progress,
            )
            last_request_at = monotonic()
            _write_json(cache_path, payload)
        parsed = payload.get("parse", {})
        html = parsed.get("text", "")
        if not html:
            raise ValueError(f"sample revision did not render: {page.latest_revision_id}")
        visible = _visible_text(html)
        stats = _inspection_stats(visible)
        primary_signal = (
            len(visible) >= 400
            and stats["alphabetic_ratio"] >= 0.5
            and stats["italian_function_word_ratio"] >= 0.01
        )
        rows.append(
            {
                "sample_id": f"itws-sample-{index:03d}",
                "work_root_id": inventory["work_root_id"],
                "root_title": inventory["root_title"],
                "metadata_decision": inventory["metadata_decision"],
                "proposed_role": inventory["proposed_role"],
                "representative_page_id": page.page_id,
                "representative_page_title": page.title,
                "representative_revision_id": page.latest_revision_id,
                "page_wikitext_bytes": page.wikitext_bytes,
                "rendered_html_characters": len(html),
                "visible_text_characters": len(visible),
                **stats,
                "primary_text_signal": primary_signal,
                "inspection_decision": (
                    "sample_primary_text_signal_pass"
                    if primary_signal
                    else "sample_requires_page_level_review"
                ),
                "source_url": _url_from_title(page.title),
            }
        )
        _emit(
            progress,
            f"inspection-sample {index}/{len(selected)} pgid={page.page_id} "
            f"signal={'pass' if primary_signal else 'review'}",
        )
    return rows


def _api_get_json(
    session: requests.Session,
    *,
    params: dict[str, str],
    retries: int,
    request_delay: float,
    progress: Progress | None,
) -> dict[str, Any]:
    """Fetch one MediaWiki response with bounded 429 retry/backoff."""

    response = None
    for attempt in range(retries):
        response = session.get(SITEINFO_URL, params=params, timeout=30)
        if response.status_code != 429:
            break
        retry_after = response.headers.get("Retry-After", "")
        try:
            retry_seconds = float(retry_after)
        except (TypeError, ValueError):
            retry_seconds = max(request_delay, 1.0) * (2 ** (attempt + 1))
        retry_seconds = min(max(retry_seconds, 0.0), 60.0)
        _emit(
            progress,
            "inspection-sample rate-limited "
            f"retry={attempt + 1}/{retries} wait={retry_seconds:g}s",
        )
        sleep(retry_seconds)
    if response is None:
        raise RuntimeError("Wikisource API request was not attempted")
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Wikisource API returned a non-object payload")
    if "error" in payload:
        raise ValueError(f"Wikisource API error: {payload['error']}")
    return payload


def _select_stratified_sample(
    rows: list[dict[str, Any]], sample_size: int
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["metadata_decision"])].append(row)
    for values in buckets.values():
        values.sort(key=lambda row: _sample_key(str(row["work_root_id"])))

    preferred = (
        "historical_core_metadata_candidate",
        "nineteenth_century_bridge_metadata_candidate",
        "conditioned_language_candidate",
        "hold_translation_edition_review",
        "exclude_post_1900_scope",
        "hold_period_or_work_identity",
    )
    selected: list[dict[str, Any]] = []
    per_bucket = max(1, sample_size // len(preferred))
    for decision in preferred:
        selected.extend(buckets.get(decision, [])[:per_bucket])
    selected_ids = {row["work_root_id"] for row in selected}
    remaining = sorted(
        (row for row in rows if row["work_root_id"] not in selected_ids),
        key=lambda row: _sample_key(str(row["work_root_id"])),
    )
    selected.extend(remaining[: max(0, sample_size - len(selected))])
    return selected[:sample_size]


def _build_gate_rows(
    inventory_rows: list[dict[str, Any]], *, sample_status: dict[str, str]
) -> list[dict[str, Any]]:
    candidate_rows = [row for row in inventory_rows if row["proposed_role"] != "excluded"]
    total_bytes = sum(int(row["projected_wikitext_bytes"]) for row in candidate_rows)
    role_bytes = Counter()
    author_bytes = Counter()
    for row in candidate_rows:
        size = int(row["projected_wikitext_bytes"])
        role_bytes[row["proposed_role"]] += size
        for author in str(row["author_evidence"]).split(" | "):
            if author:
                author_bytes[author] += size

    gate_rows = []
    for row in inventory_rows:
        size = int(row["projected_wikitext_bytes"])
        authors = [value for value in str(row["author_evidence"]).split(" | ") if value]
        maximum_author_share = max(
            (author_bytes[author] / total_bytes for author in authors),
            default=0.0,
        )
        decision = str(row["metadata_decision"])
        gate_rows.append(
            {
                "work_root_id": row["work_root_id"],
                "root_title": row["root_title"],
                "author_evidence": row["author_evidence"],
                "period_bucket": row["period_bucket"],
                "language_route": row["language_route"],
                "genre_route": row["genre_route"],
                "form_route": row["form_route"],
                "metadata_decision": decision,
                "proposed_role": row["proposed_role"],
                "hierarchy_page_count": row["hierarchy_page_count"],
                "projected_wikitext_bytes": size,
                "projected_archive_share": _ratio(size, total_bytes),
                "projected_role_share": _ratio(size, role_bytes[row["proposed_role"]]),
                "author_projected_share": round(maximum_author_share, 8),
                "source_scan_status": row["source_scan_status"],
                "existing_reference_ids": row["existing_reference_ids"],
                "inspection_status": sample_status.get(
                    str(row["work_root_id"]), "not_in_bounded_sample"
                ),
                "next_action": _next_action(decision),
            }
        )
    return gate_rows


def _build_report(
    config: WikisourceArchiveInventoryConfig,
    *,
    dump_paths: dict[str, Path],
    siteinfo: dict[str, Any],
    pages: list[WikiPage],
    inventory_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    decision_counts = Counter(row["metadata_decision"] for row in inventory_rows)
    role_counts = Counter(row["proposed_role"] for row in inventory_rows)
    role_bytes = Counter()
    for row in inventory_rows:
        size = int(row["projected_wikitext_bytes"])
        role_bytes[row["proposed_role"]] += size
    rights = siteinfo["query"]["rightsinfo"]
    candidates = [
        row
        for row in inventory_rows
        if row["metadata_decision"].endswith("metadata_candidate")
    ]
    candidate_bytes = sum(int(row["projected_wikitext_bytes"]) for row in candidates)
    author_bytes = Counter()
    for row in candidates:
        size = int(row["projected_wikitext_bytes"])
        for author in str(row["author_evidence"]).split(" | "):
            if author:
                author_bytes[author] += size
    top_authors = [
        {
            "author": author,
            "projected_wikitext_bytes": size,
            "share": _ratio(size, candidate_bytes),
        }
        for author, size in author_bytes.most_common(20)
    ]
    largest_roots = [
        {
            "work_root_id": row["work_root_id"],
            "root_title": row["root_title"],
            "proposed_role": row["proposed_role"],
            "projected_wikitext_bytes": int(row["projected_wikitext_bytes"]),
        }
        for row in sorted(
            candidates,
            key=lambda row: int(row["projected_wikitext_bytes"]),
            reverse=True,
        )[:20]
    ]
    return {
        "inventory_version": "italian_wikisource_archive_inventory_v1",
        "dump": {
            "date": config.dump_date,
            "base_url": config.dump_base_url,
            "files": {
                label: {
                    "filename": path.name,
                    "byte_count": path.stat().st_size,
                    "sha1": _sha1_file(path),
                    "cache_path": _portable(path, config.repo_root),
                }
                for label, path in dump_paths.items()
            },
            "full_page_text_dump_downloaded": False,
        },
        "rights": {
            "site_license": rights["text"],
            "site_license_url": rights["url"],
            "underlying_work_and_scan_status": (
                "record-level public-domain or compatible-license and source-scan "
                "verification remains required before extraction"
            ),
        },
        "main_namespace_page_count": len(pages),
        "main_namespace_nonredirect_page_count": sum(not page.is_redirect for page in pages),
        "work_root_count": len(inventory_rows),
        "root_with_page_count": sum(bool(row["root_page_id"]) for row in inventory_rows),
        "hierarchy_only_root_count": sum(not bool(row["root_page_id"]) for row in inventory_rows),
        "projected_wikitext_bytes": sum(
            int(row["projected_wikitext_bytes"]) for row in inventory_rows
        ),
        "decision_counts": dict(sorted(decision_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "projected_wikitext_bytes_by_role": dict(sorted(role_bytes.items())),
        "candidate_work_root_count": len(candidates),
        "candidate_projected_wikitext_bytes": candidate_bytes,
        "candidate_projected_archive_share": _ratio(
            candidate_bytes,
            sum(int(row["projected_wikitext_bytes"]) for row in inventory_rows),
        ),
        "existing_project_reference_count": decision_counts["existing_project_reference"],
        "bounded_inspection": {
            "sample_size": len(sample_rows),
            "primary_text_signal_pass_count": sum(
                row["primary_text_signal"] for row in sample_rows
            ),
            "review_count": sum(not row["primary_text_signal"] for row in sample_rows),
            "sample_csv_path": _portable(config.inspection_sample_path, config.repo_root),
            "sample_csv_sha256": _sha256_file(config.inspection_sample_path),
        },
        "concentration": {
            "top_author_proxies": top_authors,
            "largest_candidate_roots": largest_roots,
            "warning": (
                "wikitext bytes are metadata projections, not cleaned characters or tokens; "
                "unknown and multi-author rows prevent final concentration claims"
            ),
        },
        "outputs": {
            "inventory_path": _portable(config.inventory_path, config.repo_root),
            "inventory_sha256": _sha256_file(config.inventory_path),
            "page_hierarchy_path": _portable(config.page_hierarchy_path, config.repo_root),
            "page_hierarchy_sha256": _sha256_file(config.page_hierarchy_path),
            "composition_gate_path": _portable(config.composition_gate_path, config.repo_root),
            "composition_gate_sha256": _sha256_file(config.composition_gate_path),
            "json_report_path": _portable(config.json_report_path, config.repo_root),
            "markdown_report_path": _portable(config.markdown_report_path, config.repo_root),
        },
        "policy": {
            "metadata_only": True,
            "source_text_extracted": False,
            "corpus_text_activated": False,
            "v7_split_assigned": False,
            "training_mixture_weight_assigned": False,
            "gpu_work_started": False,
            "conditioned_material_standard_core_eligible": False,
            "full_page_audit_requires_gate_decision": True,
        },
        "next_checkpoint": (
            "Resolve bounded metadata holds and approve only composition-compatible "
            "work roots for page-level extraction and cross-corpus deduplication."
        ),
    }


def render_wikisource_archive_inventory_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Italian Wikisource Archive Inventory And Composition Gate",
        "",
        "## Result",
        "",
        (
            f"The pinned `{report['dump']['date']}` metadata snapshot contains "
            f"{report['main_namespace_page_count']:,} main-namespace pages grouped "
            f"into {report['work_root_count']:,} structural work roots."
        ),
        "",
        (
            f"Metadata identifies {report['candidate_work_root_count']:,} historical "
            "or nineteenth-century candidates before page-level review, projecting "
            f"{report['candidate_projected_wikitext_bytes']:,} wikitext bytes "
            f"({report['candidate_projected_archive_share']:.1%} of the archive "
            "projection). No corpus text was downloaded, extracted, or activated."
        ),
        "",
        "## Decisions",
        "",
        "| Decision | Work roots |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{decision}` | {count:,} |"
        for decision, count in report["decision_counts"].items()
    )
    lines.extend(
        [
            "",
            "## Projected Roles",
            "",
            "| Role | Work roots | Wikitext-byte projection |",
            "| --- | ---: | ---: |",
        ]
    )
    lines.extend(
        (
            f"| `{role}` | {report['role_counts'][role]:,} | "
            f"{report['projected_wikitext_bytes_by_role'][role]:,} |"
        )
        for role in report["role_counts"]
    )
    top_authors = report["concentration"]["top_author_proxies"][:10]
    lines.extend(
        [
            "",
            "## Candidate Concentration",
            "",
            "| Author proxy | Wikitext-byte projection | Candidate share |",
            "| --- | ---: | ---: |",
        ]
    )
    lines.extend(
        (
            f"| {row['author']} | {row['projected_wikitext_bytes']:,} | "
            f"{row['share']:.1%} |"
        )
        for row in top_authors
    )
    lines.extend(
        [
            "",
            (
                "These are metadata projections rather than cleaned-text or token "
                "shares. Unknown and multi-author rows prevent a final concentration "
                "claim; later activation must recompute and cap dominance."
            ),
        ]
    )
    inspection = report["bounded_inspection"]
    lines.extend(
        [
            "",
            "## Bounded Inspection",
            "",
            (
                f"The stratified sample rendered {inspection['sample_size']:,} exact "
                "dump-pinned revisions."
            ),
            f"- Primary-text signal passes: {inspection['primary_text_signal_pass_count']:,}.",
            f"- Rows requiring page-level review: {inspection['review_count']:,}.",
            "- These signals validate inventory value; they do not authorize extraction.",
            "",
            "## Rights And Boundaries",
            "",
            (
                f"- Current site transcription terms: {report['rights']['site_license']} "
                f"({report['rights']['site_license_url']})."
            ),
            "- Underlying work and source-scan status still require record-level verification.",
            "- Explicit dialect/language varieties remain conditioned or held, never standard core.",
            "- Wikitext bytes are a metadata projection, not cleaned characters or Minerva tokens.",
            "- The full page-text dump was not downloaded.",
            "- No V7 split, mixture weight, cache deletion, or GPU work occurred.",
            "",
            "## Artifacts",
            "",
            f"- Work-root inventory: `{report['outputs']['inventory_path']}`",
            f"- Page hierarchy: `{report['outputs']['page_hierarchy_path']}`",
            f"- Composition gate: `{report['outputs']['composition_gate_path']}`",
            f"- Inspection sample: `{inspection['sample_csv_path']}`",
            f"- Machine-readable report: `{report['outputs']['json_report_path']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _representative_page(group: list[WikiPage]) -> WikiPage:
    nonredirect = [page for page in group if not page.is_redirect]
    if not nonredirect:
        return group[0]
    return max(nonredirect, key=lambda page: (page.wikitext_bytes, -page.page_id))


def _visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for node in soup.select(
        "script, style, table, .mw-editsection, .noprint, .ws-noexport, "
        ".licenseContainer, .navbox"
    ):
        node.decompose()
    return " ".join(soup.get_text(" ", strip=True).split())


def _inspection_stats(text: str) -> dict[str, Any]:
    words = re.findall(r"[^\W\d_]+", text.casefold(), flags=re.UNICODE)
    alphabetic = sum(char.isalpha() for char in text)
    hits = sum(word in _ITALIAN_FUNCTION_WORDS for word in words)
    return {
        "alphabetic_ratio": round(alphabetic / max(1, len(text)), 6),
        "italian_function_word_hits": hits,
        "italian_function_word_ratio": round(hits / max(1, len(words)), 6),
    }


def _next_action(decision: str) -> str:
    if decision == "existing_project_reference":
        return "Retain as an explicit cross-archive deduplication reference."
    if decision.endswith("metadata_candidate"):
        return "Verify source scan, authorship, page boundaries, quality, and duplicates."
    if decision == "conditioned_language_candidate":
        return "Define a separate conditioned experiment before any extraction."
    if decision.startswith("exclude_"):
        return "Retain the metadata exclusion; do not perform a page-level audit."
    return "Resolve the bounded metadata hold before page-level extraction."


def _work_root_id(root_title: str, group: list[WikiPage]) -> str:
    root = next((page for page in group if page.title == root_title), None)
    if root:
        return f"itws:{root.page_id}"
    digest = hashlib.sha1(root_title.encode("utf-8")).hexdigest()[:12]
    return f"itws:root:{digest}"


def _root_title(title: str) -> str:
    return title.split("/", 1)[0]


def _decode_title(value: str) -> str:
    return value.replace("_", " ")


def _timestamp_utc(value: str) -> str:
    parsed = datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    return parsed.isoformat().replace("+00:00", "Z")


def _url_from_title(title: str) -> str:
    encoded = quote(title.replace(" ", "_"), safe="()_',:")
    return f"https://it.wikisource.org/wiki/{encoded}"


def _title_from_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    if "wikisource.org" not in parsed.netloc:
        return ""
    if not parsed.path.startswith("/wiki/"):
        return ""
    return _decode_title(unquote(parsed.path.removeprefix("/wiki/")).strip("/"))


def _roman_to_int(value: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for char in reversed(value):
        current = values[char]
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total


def _sample_key(value: str) -> str:
    return hashlib.sha256(f"italian-wikisource-inventory-v1:{value}".encode()).hexdigest()


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


def _progress(
    progress: Progress | None,
    label: str,
    completed: int,
    total: int,
    started: float,
    detail: str,
) -> None:
    elapsed = monotonic() - started
    rate = completed / elapsed if elapsed else 0.0
    remaining = (total - completed) / rate if rate else 0.0
    _emit(
        progress,
        f"{label} {completed:,}/{total:,} ({completed / total:.1%}) "
        f"{detail} elapsed={_duration(elapsed)} eta={_duration(remaining)}",
    )


def _emit(progress: Progress | None, message: str) -> None:
    if progress:
        progress(message)


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _validate_config(config: WikisourceArchiveInventoryConfig) -> None:
    if config.dump_date != DUMP_DATE or config.dump_base_url != DUMP_BASE_URL:
        raise ValueError("checkpoint 4A requires the pinned 20260801 dump")
    if config.sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if config.request_delay < 0:
        raise ValueError("request_delay must be non-negative")
    if config.api_retries <= 0:
        raise ValueError("api_retries must be positive")
    if config.progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    for path in (config.broader_manifest_path, config.poems_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not config.snapshot_dir.is_dir():
        raise FileNotFoundError(config.snapshot_dir)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _sha1_file(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _portable(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()
