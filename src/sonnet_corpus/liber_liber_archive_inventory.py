"""Metadata-only archive inventory and composition gate for Liber Liber."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from statistics import median
from time import monotonic, sleep as default_sleep
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests


WORDPRESS_PAGES_URL = "https://liberliber.it/wp-json/wp/v2/pages"
BOOK_LICENSE_PAGE_ID = 2_013_151
BOOK_LICENSE_URL = "https://liberliber.it/opere/libri/licenze/"
CC_BY_NC_SA_4_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
USER_AGENT = "portfolio-transformer-poetry-liber-liber-inventory/1.0"
API_FIELDS = "id,parent,slug,link,title,modified,content,excerpt"
API_FIELDS_BASIC = "id,parent,slug,link,title,modified"

INVENTORY_FIELDS = (
    "record_id", "wordpress_page_id", "parent_page_id", "page_modified",
    "title", "sort_title", "author", "author_url", "landing_page_url",
    "author_biography_years", "author_period_evidence",
    "short_description", "reference_edition", "editor", "translator",
    "publication_date", "dewey_descriptor", "bisac_subject", "reliability",
    "digitization_credit", "layout_credit", "publication_credit",
    "revision_credit", "license_label", "license_url", "site_copyright_route",
    "download_formats", "supported_primary_text_formats", "download_page_urls",
    "period_bucket", "period_evidence", "genre_route", "genre_evidence",
    "language_route", "language_evidence", "translation_evidence",
    "preliminary_role", "composition_decision", "decision_reason",
    "existing_project_source_ids", "metadata_sha256", "activation_status",
)

RIGHTS_FIELDS = (
    "source_rights_id", "record_id", "title", "author", "landing_page_url",
    "reference_edition", "editor", "translator", "site_copyright_route",
    "license_label", "license_url", "book_license_terms_url",
    "book_license_page_modified", "underlying_work_evidence",
    "edition_layer_evidence", "rights_decision", "rights_reason",
    "required_notice", "downstream_note", "activation_status",
)

GATE_FIELDS = (
    "composition_decision", "preliminary_role", "record_count",
    "projected_cleaned_characters", "projected_share_of_resulting_corpus",
    "top_author", "top_author_record_count", "top_author_share",
    "fulltext_audit_value", "fulltext_audit_runtime_lower_minutes",
    "fulltext_audit_runtime_upper_minutes", "activation_status",
)

Progress = Callable[[str], None]
Sleep = Callable[[float], None]

_METADATA_PAIR = re.compile(
    r'<div class="ll_metadati_etichetta">(.*?)</div>\s*'
    r'<div class="ll_metadati_dato">(.*?)</div>',
    re.IGNORECASE | re.DOTALL,
)
_TAG = re.compile(r"<[^>]+>")
_BREAK = re.compile(r"<(?:br|/p|/li|/h\d)\b[^>]*>", re.IGNORECASE)
_SPACE = re.compile(r"\s+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_DIALECT = re.compile(
    r"\b(?:dialett\w*|vernacol\w*|romanesco|napoletan\w*|sicilian\w*|"
    r"venet\w*|milanes\w*|lombard\w*|piemontes\w*|sardo|friulan\w*|"
    r"salentin\w*|calabres\w*|genoves\w*|bolognes\w*)\b",
    re.IGNORECASE,
)
_LANGUAGE_VARIETY_AUTHOR = re.compile(
    r"\b(?:giorgio baffo|salvatore di giacomo|carlo porta|giuseppe gioacchino belli|"
    r"cesare pascarella|trilussa|carlo alberto salustri)\b",
    re.IGNORECASE,
)
_SONNET = re.compile(r"\bsonett\w*\b", re.IGNORECASE)
_POETRY = re.compile(
    r"\b(?:poesia|poesie|poetry|poemi?|liric\w*|versi|rime|canzonier\w*|"
    r"ballat\w*|canzon\w*)\b",
    re.IGNORECASE,
)
_DRAMA = re.compile(
    r"\b(?:teatro|drama|drammatic\w*|commedi\w*|tragedi\w*|melodramm\w*)\b",
    re.IGNORECASE,
)
_PROSE_FORM = re.compile(
    r"\b(?:biografi\w*|memorie|narrativa|novell\w*|raccont\w*|romanz\w*|"
    r"prosa|prose|sagg\w*|trattat\w*)\b",
    re.IGNORECASE,
)
_TRANSLATION = re.compile(
    r"\b(?:traduzion\w*|tradutt\w*|versione italian\w*|translated|translation)\b",
    re.IGNORECASE,
)
_FOREIGN_LITERATURE = re.compile(
    r"\b(?:letteratura|narrativa|poesia|teatro)\s+"
    r"(?:inglese|francese|tedesca|spagnola|russa|americana|greca|latina|"
    r"portoghese|polacca|norvegese|svedese|danese|olandese|giapponese|cinese)\b",
    re.IGNORECASE,
)
_HISTORICAL_PERIOD = re.compile(
    r"(?:fino al 1375|origini|medioev|1200\s*[-–]\s*1375|1375\s*[-–]\s*1492|"
    r"1492\s*[-–]\s*1542|1542\s*[-–]\s*1585|1585\s*[-–]\s*1748|"
    r"1748\s*[-–]\s*1814|1585\s*[-–]\s*1814|13\.?\s*sec|14\.?\s*sec|"
    r"15\.?\s*sec|16\.?\s*sec|17\.?\s*sec|18\.?\s*sec)",
    re.IGNORECASE,
)
_BRIDGE_PERIOD = re.compile(
    r"(?:1814\s*[-–]\s*1859|1815\s*[-–]\s*1860|1859\s*[-–]\s*1899|"
    r"1861\s*[-–]\s*1900|19\.?\s*sec|sec\.?\s*19|ottocento)",
    re.IGNORECASE,
)
_CROSS_CENTURY_PERIOD = re.compile(
    r"(?:1815\s*[-–]\s*1945|1800\s*[-–]\s*1999)",
    re.IGNORECASE,
)
_MODERN_PERIOD = re.compile(
    r"(?:1900\s*[-–]|1901\s*[-–]|1945\s*[-–]|20\.?\s*sec|21\.?\s*sec|"
    r"sec\.?\s*20|sec\.?\s*21|novecento|duemila)",
    re.IGNORECASE,
)
_YEAR = re.compile(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)")
_DEATH_WORD = re.compile(r"\b(?:mor[iì]|morto|morta|decedut\w*)\b", re.IGNORECASE)


@dataclass(frozen=True)
class LiberLiberArchiveInventoryConfig:
    repo_root: Path
    local_cache_path: Path
    inventory_path: Path
    rights_path: Path
    composition_gate_path: Path
    json_report_path: Path
    markdown_report_path: Path
    broader_sources_manifest_path: Path
    prior_probe_report_path: Path
    bibit_build_report_path: Path
    gutenberg_build_report_path: Path
    wikisource_build_report_path: Path
    request_delay_seconds: float = 0.10
    request_timeout_seconds: float = 60.0
    per_page: int = 100


class _LinkAndClassParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.classes: list[str] = []
        self._href = ""
        self._link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        class_name = values.get("class", "")
        if class_name:
            self.classes.extend(class_name.split())
        if tag == "a":
            self._href = values.get("href", "")
            self._link_text = []

    def handle_data(self, data: str) -> None:
        if self._href:
            self._link_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href:
            self.links.append((self._href, _SPACE.sub(" ", "".join(self._link_text)).strip()))
            self._href = ""
            self._link_text = []


class _CombinedPageResponse:
    def __init__(self, payload: list[dict[str, Any]], *, total: int, total_pages: int) -> None:
        self._payload = payload
        self.headers = {"X-WP-Total": str(total), "X-WP-TotalPages": str(total_pages)}

    def json(self) -> list[dict[str, Any]]:
        return self._payload


def fetch_liber_liber_pages(
    config: LiberLiberArchiveInventoryConfig,
    *,
    session: requests.Session | None = None,
    progress: Progress | None = None,
    sleep: Sleep = default_sleep,
) -> dict[str, Any]:
    """Fetch every public WordPress page once, or reuse the pinned local cache."""

    partial: dict[str, Any] | None = None
    if config.local_cache_path.is_file():
        partial = json.loads(config.local_cache_path.read_text(encoding="utf-8"))
        _validate_snapshot(partial, allow_incomplete=True)
        if partial.get("complete") is True:
            _report(progress, f"cache hit pages={partial['total_pages']:,} records={partial['total_records']:,}")
            return partial
        _report(
            progress,
            f"resuming partial cache records={len(partial['pages']):,}/{partial['total_records']:,}",
        )
    if not 1 <= config.per_page <= 100:
        raise ValueError("per_page must be between 1 and 100")
    if config.request_delay_seconds < 0 or config.request_timeout_seconds <= 0:
        raise ValueError("request delay/timeout must be non-negative/positive")

    client = session or requests.Session()
    client.headers.update({"User-Agent": USER_AGENT})
    pages: list[dict[str, Any]] = list(partial["pages"]) if partial else []
    expected_total = int(partial["total_records"]) if partial else None
    expected_pages = int(partial["total_pages"]) if partial else None
    cached_per_page = int(partial.get("per_page", config.per_page)) if partial else config.per_page
    if cached_per_page != config.per_page:
        raise ValueError("partial Liber Liber cache uses a different per_page value")
    page_number = len(pages) // config.per_page + 1
    fetched_at_utc = (
        str(partial["fetched_at_utc"])
        if partial
        else datetime.now(UTC).replace(microsecond=0).isoformat()
    )
    started = monotonic()
    while expected_pages is None or page_number <= expected_pages:
        if page_number > 1 and config.request_delay_seconds:
            sleep(config.request_delay_seconds)
        response = _get_page_with_retries(
            client,
            page_number=page_number,
            config=config,
            progress=progress,
            sleep=sleep,
        )
        batch = response.json()
        if not isinstance(batch, list) or any(not isinstance(row, dict) for row in batch):
            raise ValueError("Liber Liber WordPress page response is not a list of objects")
        total = _positive_header(response, "X-WP-Total")
        total_pages = _positive_header(response, "X-WP-TotalPages")
        if expected_total is None:
            expected_total, expected_pages = total, total_pages
        elif (total, total_pages) != (expected_total, expected_pages):
            raise ValueError("Liber Liber WordPress totals changed during pagination")
        pages.extend(batch)
        partial_snapshot = {
            "inventory_version": "liber_liber_archive_inventory_v1",
            "api_url": WORDPRESS_PAGES_URL,
            "api_fields": API_FIELDS,
            "fetched_at_utc": fetched_at_utc,
            "total_records": expected_total,
            "total_pages": expected_pages,
            "per_page": config.per_page,
            "complete": False,
            "pages": pages,
        }
        config.local_cache_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(config.local_cache_path, partial_snapshot)
        elapsed = monotonic() - started
        eta = elapsed / page_number * max(0, expected_pages - page_number)
        _report(
            progress,
            f"page={page_number:,}/{expected_pages:,} records={len(pages):,}/{expected_total:,} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
        )
        page_number += 1

    if len(pages) != expected_total:
        raise ValueError(f"WordPress pagination returned {len(pages)} pages; expected {expected_total}")
    ids = [int(row.get("id", 0)) for row in pages]
    if any(value <= 0 for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Liber Liber WordPress page IDs are missing or duplicated")
    snapshot = {
        "inventory_version": "liber_liber_archive_inventory_v1",
        "api_url": WORDPRESS_PAGES_URL,
        "api_fields": API_FIELDS,
        "fetched_at_utc": fetched_at_utc,
        "total_records": expected_total,
        "total_pages": expected_pages,
        "per_page": config.per_page,
        "complete": True,
        "pages": sorted(pages, key=lambda row: int(row["id"])),
    }
    _validate_snapshot(snapshot)
    _write_json_atomic(config.local_cache_path, snapshot)
    return snapshot


def _get_page_with_retries(
    client: requests.Session,
    *,
    page_number: int,
    config: LiberLiberArchiveInventoryConfig,
    progress: Progress | None,
    sleep: Sleep,
) -> Any:
    params = {
        "per_page": config.per_page,
        "page": page_number,
        "orderby": "id",
        "order": "asc",
        "_fields": API_FIELDS,
    }
    last_error: requests.RequestException | None = None
    for attempt in range(1, 3):
        try:
            response = client.get(
                WORDPRESS_PAGES_URL,
                params=params,
                timeout=config.request_timeout_seconds,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if attempt == 2:
                break
            delay = float(2 ** attempt)
            _report(
                progress,
                f"page={page_number:,} attempt={attempt}/2 failed={type(error).__name__} retry_in={delay:.0f}s",
            )
            sleep(delay)
    assert last_error is not None
    _report(progress, f"page={page_number:,} switching_to_two_50_record_offset_requests")
    return _get_split_page(
        client,
        page_number=page_number,
        config=config,
        progress=progress,
        sleep=sleep,
    )


def _get_split_page(
    client: requests.Session,
    *,
    page_number: int,
    config: LiberLiberArchiveInventoryConfig,
    progress: Progress | None,
    sleep: Sleep,
) -> _CombinedPageResponse:
    base_offset = (page_number - 1) * config.per_page
    batches: list[dict[str, Any]] = []
    total: int | None = None
    # Some catalog ranges contain enough rendered HTML to exceed the server's
    # response limit at 25-100 records. Ten-record offset slices remain bounded
    # while preserving the same stable ID ordering and complete accounting.
    split_size = min(10, max(1, config.per_page // 2))
    for offset in range(base_offset, base_offset + config.per_page, split_size):
        if total is not None and offset >= total:
            break
        requested = min(split_size, max(0, (total or base_offset + config.per_page) - offset))
        payload, response_total = _get_offset_chunk(
            client,
            offset=offset,
            size=requested,
            config=config,
            progress=progress,
            sleep=sleep,
        )
        if total is None:
            total = response_total
        elif response_total != total:
            raise ValueError("Liber Liber WordPress total changed during split-page fallback")
        batches.extend(payload)
    assert total is not None
    expected = min(config.per_page, total - base_offset)
    if len(batches) != expected:
        raise ValueError(f"split-page fallback returned {len(batches)} records; expected {expected}")
    return _CombinedPageResponse(
        batches,
        total=total,
        total_pages=math.ceil(total / config.per_page),
    )


def _get_offset_chunk(
    client: requests.Session,
    *,
    offset: int,
    size: int,
    config: LiberLiberArchiveInventoryConfig,
    progress: Progress | None,
    sleep: Sleep,
) -> tuple[list[dict[str, Any]], int]:
    if size <= 0:
        return [], 0
    params = {
        "per_page": size,
        "offset": offset,
        "orderby": "id",
        "order": "asc",
        "_fields": API_FIELDS,
    }
    last_error: requests.RequestException | None = None
    for attempt in range(1, 3):
        try:
            response = client.get(
                WORDPRESS_PAGES_URL,
                params=params,
                timeout=config.request_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
                raise ValueError("split WordPress page response is not a list of objects")
            return payload, _positive_header(response, "X-WP-Total")
        except requests.RequestException as error:
            last_error = error
            if attempt == 2:
                break
            delay = float(2 ** attempt)
            _report(
                progress,
                f"offset={offset:,} size={size} attempt={attempt}/2 failed={type(error).__name__} retry_in={delay:.0f}s",
            )
            sleep(delay)
    assert last_error is not None
    if size == 1:
        _report(progress, f"offset={offset:,} content_endpoint_failed_using_basic_metadata")
        response = client.get(
            WORDPRESS_PAGES_URL,
            params={
                "per_page": 1,
                "offset": offset,
                "orderby": "id",
                "order": "asc",
                "_fields": API_FIELDS_BASIC,
            },
            timeout=config.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
            raise ValueError("basic-metadata fallback did not return exactly one page")
        row = dict(payload[0])
        row["content_fetch_error"] = type(last_error).__name__
        row["content"] = {"rendered": "", "protected": False}
        row["excerpt"] = {"rendered": "", "protected": False}
        return [row], _positive_header(response, "X-WP-Total")
    left_size = size // 2
    right_size = size - left_size
    _report(progress, f"offset={offset:,} size={size} splitting={left_size}+{right_size}")
    left, left_total = _get_offset_chunk(
        client, offset=offset, size=left_size, config=config, progress=progress, sleep=sleep,
    )
    right, right_total = _get_offset_chunk(
        client, offset=offset + left_size, size=right_size,
        config=config, progress=progress, sleep=sleep,
    )
    if left_total != right_total:
        raise ValueError("Liber Liber WordPress total changed during adaptive split")
    return [*left, *right], left_total


def build_liber_liber_archive_inventory(
    config: LiberLiberArchiveInventoryConfig,
    *,
    session: requests.Session | None = None,
    progress: Progress | None = None,
    sleep: Sleep = default_sleep,
) -> dict[str, Any]:
    """Build a complete metadata-only book inventory and inactive composition gate."""

    snapshot = fetch_liber_liber_pages(
        config, session=session, progress=progress, sleep=sleep,
    )
    page_by_id = {int(row["id"]): row for row in snapshot["pages"]}
    license_page = page_by_id.get(BOOK_LICENSE_PAGE_ID)
    if license_page is None or license_page.get("link", "").rstrip("/") != BOOK_LICENSE_URL.rstrip("/"):
        raise ValueError("pinned Liber Liber book-license page is absent or moved")
    license_text = _plain_text(_rendered(license_page, "content"))
    if "Non commerciale" not in license_text or "70 anni" not in license_text:
        raise ValueError("Liber Liber book-license evidence no longer matches the audited policy")

    existing = _load_existing_urls(config.broader_sources_manifest_path)
    rows: list[dict[str, Any]] = []
    rights: list[dict[str, Any]] = []
    for page in snapshot["pages"]:
        row = parse_liber_liber_work_page(page, author_page=page_by_id.get(int(page.get("parent", 0))))
        if row is None:
            continue
        row["existing_project_source_ids"] = ";".join(
            existing.get(_normalize_url(row["landing_page_url"]), ())
        )
        _classify_row(row)
        rows.append(row)
        rights.append(_rights_row(row, license_page))
    rows.sort(key=lambda row: int(row["wordpress_page_id"]))
    rights.sort(key=lambda row: int(row["record_id"].split(":")[-1]))
    if not rows:
        raise ValueError("Liber Liber inventory found no book work pages")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("Liber Liber inventory contains duplicate record IDs")

    projection = _projection_inputs(config)
    gate_rows = _build_gate_rows(rows, projection)
    _write_csv(config.inventory_path, INVENTORY_FIELDS, rows)
    _write_csv(config.rights_path, RIGHTS_FIELDS, rights)
    _write_csv(config.composition_gate_path, GATE_FIELDS, gate_rows)
    report = _build_report(config, snapshot, rows, rights, gate_rows, projection, license_page)
    _write_json(config.json_report_path, report)
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(render_markdown(report), encoding="utf-8")
    return report


def parse_liber_liber_work_page(
    page: dict[str, Any],
    *,
    author_page: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Parse one WordPress book page; return None for navigation/author pages."""

    page_id = int(page.get("id", 0))
    content = _rendered(page, "content")
    metadata = _metadata_fields(content)
    if not metadata.get("titolo") or not metadata.get("autore"):
        return None
    parser = _LinkAndClassParser()
    parser.feed(content)
    formats, download_pages = _download_evidence(parser.links)
    supported = sorted(set(formats) & {"txt_zip", "odt"})
    free = any(value.startswith("ll_ebook_") and value.endswith("_free") for value in parser.classes)
    protected = any(value.startswith("ll_ebook_") and value.endswith("_prot") for value in parser.classes)
    if free and not protected:
        copyright_route = "site_marked_free"
    elif protected and not free:
        copyright_route = "site_marked_protected"
    else:
        copyright_route = "site_status_unclear"
    license_label, license_url = _license_evidence(content, metadata.get("licenza", ""), parser.links)
    author_url = next(
        (url for url, text in parser.links if text.strip() == metadata["autore"] and "/autori/" in url),
        "",
    )
    title = metadata["titolo"]
    biography_years, author_period_evidence = _author_period_evidence(author_page)
    metadata_payload = json.dumps(page, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "record_id": f"ll:{page_id}",
        "wordpress_page_id": page_id,
        "parent_page_id": int(page.get("parent", 0)),
        "page_modified": str(page.get("modified", "")),
        "title": title,
        "sort_title": metadata.get("titolo per ordinamento", ""),
        "author": metadata["autore"],
        "author_url": author_url,
        "landing_page_url": str(page.get("link", "")),
        "author_biography_years": ";".join(str(value) for value in biography_years),
        "author_period_evidence": author_period_evidence,
        "short_description": metadata.get("descrizione breve", ""),
        "reference_edition": metadata.get("opera di riferimento", ""),
        "editor": metadata.get("cura", ""),
        "translator": metadata.get("traduzione", metadata.get("traduttore", "")),
        "publication_date": metadata.get("data pubblicazione", ""),
        "dewey_descriptor": metadata.get("descrittore dewey", ""),
        "bisac_subject": metadata.get("soggetto bisac", ""),
        "reliability": metadata.get("affidabilità", metadata.get("affidabilita", "")),
        "digitization_credit": _clean_credit(metadata.get("digitalizzazione", "")),
        "layout_credit": _clean_credit(metadata.get("impaginazione", "")),
        "publication_credit": _clean_credit(metadata.get("pubblicazione", "")),
        "revision_credit": _clean_credit(metadata.get("revisione", "")),
        "license_label": license_label,
        "license_url": license_url,
        "site_copyright_route": copyright_route,
        "download_formats": ";".join(formats),
        "supported_primary_text_formats": ";".join(supported),
        "download_page_urls": ";".join(download_pages),
        "period_bucket": "",
        "period_evidence": "",
        "genre_route": "",
        "genre_evidence": "",
        "language_route": "",
        "language_evidence": "",
        "translation_evidence": "",
        "preliminary_role": "",
        "composition_decision": "",
        "decision_reason": "",
        "existing_project_source_ids": "",
        "metadata_sha256": hashlib.sha256(metadata_payload.encode()).hexdigest(),
        "activation_status": "inactive_metadata_only",
    }


def _classify_row(row: dict[str, Any]) -> None:
    metadata = " ".join(
        str(row[key])
        for key in ("title", "short_description", "dewey_descriptor", "bisac_subject")
    )
    work_genre_metadata = " ".join(
        str(row[key]) for key in ("title", "short_description", "bisac_subject")
    )
    dewey = str(row["dewey_descriptor"])
    author_years = [int(value) for value in str(row["author_biography_years"]).split(";") if value]
    modern_subject = _MODERN_PERIOD.search(dewey)
    cross_subject = _CROSS_CENTURY_PERIOD.search(dewey)
    bridge_subject = _BRIDGE_PERIOD.search(dewey)
    historical_subject = _HISTORICAL_PERIOD.search(dewey)
    if modern_subject:
        period = "post_1900"
        period_evidence = f"Dewey work-period signal={modern_subject.group(0)}."
    elif cross_subject:
        period = "period_review"
        period_evidence = f"Dewey range crosses the 1900 boundary={cross_subject.group(0)}."
    elif bridge_subject:
        if author_years and max(author_years) <= 1800:
            period = "period_review"
            period_evidence = (
                f"Dewey bridge signal={bridge_subject.group(0)} conflicts with "
                f"parent-author years={';'.join(str(value) for value in author_years)}."
            )
        else:
            period = "nineteenth_century"
            period_evidence = f"Dewey work-period signal={bridge_subject.group(0)}."
    elif historical_subject:
        if author_years and min(author_years) > 1900:
            period = "post_1900"
            period_evidence = (
                f"Historical Dewey subject={historical_subject.group(0)} conflicts with "
                f"parent-author years={';'.join(str(value) for value in author_years)}."
            )
        elif author_years and max(author_years) > 1800:
            period = "period_review"
            period_evidence = (
                f"Historical Dewey subject={historical_subject.group(0)} conflicts with "
                f"parent-author years={';'.join(str(value) for value in author_years)}."
            )
        elif author_years:
            period = "origins_through_1800"
            period_evidence = (
                f"Dewey work-period signal={historical_subject.group(0)}; "
                f"consistent parent-author years={';'.join(str(value) for value in author_years)}."
            )
        else:
            period = "unknown"
            period_evidence = (
                f"Dewey historical signal={historical_subject.group(0)} lacks parent-author chronology."
            )
    elif len(author_years) == 1 and "explicit_death_year" in str(row["author_period_evidence"]):
        death_year = author_years[0]
        period = (
            "origins_through_1800" if death_year <= 1800
            else "nineteenth_century" if death_year <= 1900
            else "post_1900"
        )
        period_evidence = str(row["author_period_evidence"])
    else:
        period = "unknown"
        period_evidence = "No decisive Dewey work-period range or explicit author death year."

    sonnet_match = _SONNET.search(work_genre_metadata)
    poetry_match = _POETRY.search(work_genre_metadata)
    drama_match = _DRAMA.search(work_genre_metadata)
    if str(row["title"]).casefold().strip() in {"la divina commedia", "divina commedia"}:
        drama_match = None
    prose_match = _PROSE_FORM.search(
        " ".join(str(row[key]) for key in ("title", "short_description"))
    )
    if sonnet_match:
        genre, genre_evidence = "standard_sonnet_review", sonnet_match.group(0)
    elif (poetry_match and prose_match) or (drama_match and (poetry_match or prose_match)):
        signals = [
            match.group(0)
            for match in (poetry_match, drama_match, prose_match)
            if match is not None
        ]
        genre, genre_evidence = "mixed_form_review", ";".join(signals)
    elif poetry_match:
        genre, genre_evidence = "non_sonnet_poetry_review", poetry_match.group(0)
    elif drama_match:
        genre, genre_evidence = "drama_form_review", drama_match.group(0)
    elif match := _POETRY.search(dewey):
        genre, genre_evidence = "poetry_subject_or_form_review", match.group(0)
    else:
        genre, genre_evidence = "general_text", "No poetry/sonnet/drama catalog signal."

    translation = str(row["translator"]).strip()
    foreign = _FOREIGN_LITERATURE.search(dewey)
    translation_match = _TRANSLATION.search(metadata)
    dialect = _DIALECT.search(metadata)
    variety_author = _LANGUAGE_VARIETY_AUTHOR.search(str(row["author"]))
    if dialect or variety_author:
        language, language_evidence = "conditioned_language_review", (
            dialect.group(0) if dialect else variety_author.group(0)
        )
    elif translation or foreign or translation_match:
        language = "translation_review"
        language_evidence = translation or (foreign.group(0) if foreign else translation_match.group(0))
    elif "italian" in dewey.casefold():
        language, language_evidence = "standard_italian_metadata", dewey
    else:
        language, language_evidence = "italian_unresolved", "Catalog page is Italian, but work-language evidence is not explicit."

    license_compatible = "creativecommons.org/licenses/by-nc-sa/4.0" in str(row["license_url"])
    supported = bool(row["supported_primary_text_formats"])
    existing = bool(row["existing_project_source_ids"])
    if existing:
        decision = "existing_project_corpus_reference"
        reason = "Exact landing page already appears in the project source manifest."
    elif row["site_copyright_route"] == "site_marked_protected":
        decision = "exclude_personal_use_only_protected_text"
        reason = "Liber Liber marks the downloadable edition protected; site terms permit personal use only."
    elif row["site_copyright_route"] != "site_marked_free" or not license_compatible:
        decision = "hold_rights_or_item_license_unclear"
        reason = "Exact item lacks both a site-free marker and the approved CC BY-NC-SA 4.0 edition license."
    elif not supported:
        decision = "hold_no_supported_primary_text_format"
        reason = "No TXT ZIP or ODT primary-text download is advertised."
    elif language == "conditioned_language_review":
        decision = "conditioned_language_candidate_inactive"
        reason = "Dialect or language-variety evidence keeps the work outside the standard-Italian queue."
    elif language == "translation_review":
        decision = "hold_translation_edition_review"
        reason = "Translation/source-language and Italian-edition chronology require explicit review."
    elif language == "italian_unresolved":
        decision = "hold_work_language_review"
        reason = "Catalog metadata does not explicitly establish standard-Italian primary text."
    elif period == "post_1900":
        decision = "exclude_post_1900_metadata"
        reason = "Catalog period evidence places the work after the approved Ottocento bridge."
    elif period in {"unknown", "period_review"}:
        decision = "hold_work_period_review"
        reason = "Work composition/publication period is missing or crosses the 1900 boundary."
    elif genre in {"drama_form_review", "mixed_form_review", "poetry_subject_or_form_review"}:
        decision = "hold_drama_prose_verse_review"
        reason = "Catalog subject metadata does not determine the primary text's prose/verse/form route."
    else:
        decision = "eligible_fulltext_probe_inactive"
        reason = "Rights, supported format, period, language, and role metadata pass the composition gate."

    if language == "conditioned_language_review":
        role = "conditioned_language_variants"
    elif genre == "standard_sonnet_review":
        role = "standard_sonnets"
    elif genre == "non_sonnet_poetry_review":
        role = "historical_non_sonnet_poetry"
    elif genre in {"drama_form_review", "mixed_form_review", "poetry_subject_or_form_review"}:
        role = "review_unassigned"
    elif period == "nineteenth_century":
        role = "nineteenth_century_bridge"
    elif period == "origins_through_1800":
        role = "historical_general"
    else:
        role = "review_unassigned"

    row.update({
        "period_bucket": period,
        "period_evidence": period_evidence,
        "genre_route": genre,
        "genre_evidence": genre_evidence,
        "language_route": language,
        "language_evidence": language_evidence,
        "translation_evidence": translation,
        "preliminary_role": role,
        "composition_decision": decision,
        "decision_reason": reason,
    })


def _rights_row(row: dict[str, Any], license_page: dict[str, Any]) -> dict[str, Any]:
    compatible = "creativecommons.org/licenses/by-nc-sa/4.0" in str(row["license_url"])
    if row["site_copyright_route"] == "site_marked_protected":
        decision = "rights_fail_personal_use_only"
        reason = "Item is marked protected; site terms prohibit redistribution to third parties."
    elif row["site_copyright_route"] == "site_marked_free" and compatible:
        decision = "rights_pass_metadata_gate"
        reason = "Item is site-marked free and pins the approved CC BY-NC-SA 4.0 edition layer."
    else:
        decision = "rights_hold_item_status_or_license_unclear"
        reason = "Item-level free/protected status or exact edition license is incomplete."
    edition_evidence = "; ".join(
        value for value in (
            f"reference edition={row['reference_edition']}" if row["reference_edition"] else "",
            f"editor={row['editor']}" if row["editor"] else "",
            f"translator={row['translator']}" if row["translator"] else "",
        ) if value
    )
    return {
        "source_rights_id": f"ll-rights:{row['wordpress_page_id']}",
        "record_id": row["record_id"],
        "title": row["title"],
        "author": row["author"],
        "landing_page_url": row["landing_page_url"],
        "reference_edition": row["reference_edition"],
        "editor": row["editor"],
        "translator": row["translator"],
        "site_copyright_route": row["site_copyright_route"],
        "license_label": row["license_label"],
        "license_url": row["license_url"],
        "book_license_terms_url": BOOK_LICENSE_URL,
        "book_license_page_modified": str(license_page.get("modified", "")),
        "underlying_work_evidence": "Liber Liber item free/protected marker; exact author/editor/translator term remains item evidence.",
        "edition_layer_evidence": edition_evidence,
        "rights_decision": decision,
        "rights_reason": reason,
        "required_notice": "Credit Liber Liber and named edition contributors; retain source link and CC BY-NC-SA 4.0 notice.",
        "downstream_note": "Non-commercial and ShareAlike obligations apply to redistributed edition-derived text and its lineage.",
        "activation_status": "inactive_metadata_only",
    }


def _projection_inputs(config: LiberLiberArchiveInventoryConfig) -> dict[str, Any]:
    prior = json.loads(config.prior_probe_report_path.read_text(encoding="utf-8"))
    values = [
        int(row["cleaned_character_count"])
        for row in prior.get("results", [])
        if row.get("status") == "ok" and row.get("cleaned_character_count") is not None
    ]
    if not values:
        raise ValueError("prior Liber Liber probe has no character measurements")
    bibit = json.loads(config.bibit_build_report_path.read_text(encoding="utf-8"))
    gutenberg = json.loads(config.gutenberg_build_report_path.read_text(encoding="utf-8"))
    wikisource = json.loads(config.wikisource_build_report_path.read_text(encoding="utf-8"))
    current = sum(int(value) for value in bibit["record_characters_by_role"].values())
    current += sum(
        int(value) for role, value in gutenberg["record_characters_by_role"].items()
        if role != "conditioned_source_variants"
    )
    current += int(wikisource["materialized_broader_character_count"])
    return {
        "prior_probe_record_count": len(values),
        "prior_probe_total_characters": sum(values),
        "prior_probe_mean_characters": round(sum(values) / len(values)),
        "prior_probe_median_characters": round(median(values)),
        "current_frozen_archive_characters": current,
    }


def _build_gate_rows(rows: list[dict[str, Any]], projection: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["composition_decision"], row["preliminary_role"]), []).append(row)
    result = []
    for (decision, role), values in sorted(groups.items()):
        eligible = decision == "eligible_fulltext_probe_inactive"
        projected = len(values) * int(projection["prior_probe_median_characters"]) if eligible else 0
        denominator = int(projection["current_frozen_archive_characters"]) + projected
        authors = Counter(row["author"] for row in values)
        top_author, top_count = authors.most_common(1)[0]
        lower = math.ceil(len(values) * 3 * 1.0 / 60) if eligible else 0
        upper = math.ceil(len(values) * 3 * 4.0 / 60) if eligible else 0
        result.append({
            "composition_decision": decision,
            "preliminary_role": role,
            "record_count": len(values),
            "projected_cleaned_characters": projected,
            "projected_share_of_resulting_corpus": f"{projected / denominator:.6f}" if denominator else "0.000000",
            "top_author": top_author,
            "top_author_record_count": top_count,
            "top_author_share": f"{top_count / len(values):.6f}",
            "fulltext_audit_value": "high_metadata_gate_pass" if eligible else "not_authorized_or_requires_review",
            "fulltext_audit_runtime_lower_minutes": lower,
            "fulltext_audit_runtime_upper_minutes": upper,
            "activation_status": "inactive_metadata_only",
        })
    return result


def _build_report(
    config: LiberLiberArchiveInventoryConfig,
    snapshot: dict[str, Any],
    rows: list[dict[str, Any]],
    rights: list[dict[str, Any]],
    gates: list[dict[str, Any]],
    projection: dict[str, Any],
    license_page: dict[str, Any],
) -> dict[str, Any]:
    eligible = [row for row in rows if row["composition_decision"] == "eligible_fulltext_probe_inactive"]
    authors = Counter(row["author"] for row in eligible)
    projected = len(eligible) * int(projection["prior_probe_median_characters"])
    resulting = int(projection["current_frozen_archive_characters"]) + projected
    top_authors = [
        {"author": author, "record_count": count, "record_share": count / max(1, len(eligible))}
        for author, count in authors.most_common(10)
    ]
    report = {
        "checkpoint": "5A-liber-liber-archive-inventory-composition-gate",
        "fetched_at_utc": snapshot["fetched_at_utc"],
        "wordpress_page_count": snapshot["total_records"],
        "wordpress_api_page_count": snapshot["total_pages"],
        "content_error_page_count": sum(bool(row.get("content_fetch_error")) for row in snapshot["pages"]),
        "book_work_count": len(rows),
        "records_with_supported_primary_text": sum(bool(row["supported_primary_text_formats"]) for row in rows),
        "existing_project_reference_count": sum(bool(row["existing_project_source_ids"]) for row in rows),
        "eligible_fulltext_probe_count": len(eligible),
        "decision_counts": dict(sorted(Counter(row["composition_decision"] for row in rows).items())),
        "role_counts": dict(sorted(Counter(row["preliminary_role"] for row in rows).items())),
        "period_counts": dict(sorted(Counter(row["period_bucket"] for row in rows).items())),
        "genre_counts": dict(sorted(Counter(row["genre_route"] for row in rows).items())),
        "language_counts": dict(sorted(Counter(row["language_route"] for row in rows).items())),
        "rights_decision_counts": dict(sorted(Counter(row["rights_decision"] for row in rights).items())),
        "eligible_top_authors": top_authors,
        "projection": {
            **projection,
            "eligible_projected_cleaned_characters": projected,
            "eligible_projected_share_of_resulting_corpus": projected / resulting if resulting else 0.0,
            "projection_basis": "Prior 23-record Liber Liber probe median; planning estimate, not acquired text or tokens.",
        },
        "runtime_estimate": {
            "fulltext_probe_lower_minutes": math.ceil(len(eligible) * 3 * 1.0 / 60),
            "fulltext_probe_upper_minutes": math.ceil(len(eligible) * 3 * 4.0 / 60),
            "assumption": "Three polite sequential requests per work at 1-4 seconds each, excluding manual review and overlap analysis.",
        },
        "license_policy": {
            "book_license_page_id": BOOK_LICENSE_PAGE_ID,
            "book_license_url": BOOK_LICENSE_URL,
            "book_license_page_modified": license_page["modified"],
            "approved_item_license": "CC BY-NC-SA 4.0",
            "personal_use_only_text_excluded": True,
            "item_level_status_required": True,
        },
        "outputs": {
            "inventory": _portable(config.inventory_path, config.repo_root),
            "rights": _portable(config.rights_path, config.repo_root),
            "composition_gate": _portable(config.composition_gate_path, config.repo_root),
        },
        "input_sha256": {
            "local_catalog_cache": _sha_file(config.local_cache_path),
            "broader_sources_manifest": _sha_file(config.broader_sources_manifest_path),
            "prior_probe_report": _sha_file(config.prior_probe_report_path),
        },
        "output_sha256": {
            "inventory": _sha_file(config.inventory_path),
            "rights": _sha_file(config.rights_path),
            "composition_gate": _sha_file(config.composition_gate_path),
        },
        "composition_gate_rows": gates,
        "policy": {
            "metadata_only": True,
            "fulltext_acquired": False,
            "text_activated": False,
            "v7_created": False,
            "mixture_assigned": False,
            "cache_deleted": False,
            "gpu_work_started": False,
        },
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    decisions = "\n".join(
        f"| `{key}` | {value:,} |" for key, value in report["decision_counts"].items()
    )
    roles = "\n".join(
        f"| `{key}` | {value:,} |" for key, value in report["role_counts"].items()
    )
    projection = report["projection"]
    runtime = report["runtime_estimate"]
    return (
        "# Liber Liber Archive Inventory And Composition Gate\n\n"
        "## Result\n\n"
        f"Checkpoint 5A inventories all {report['wordpress_page_count']:,} public WordPress pages and "
        f"identifies {report['book_work_count']:,} book work records without acquiring full text.\n\n"
        f"- Records advertising TXT ZIP or ODT: {report['records_with_supported_primary_text']:,}.\n"
        f"- WordPress pages requiring basic-metadata fallback: {report['content_error_page_count']:,}.\n"
        f"- Existing exact project source URLs: {report['existing_project_reference_count']:,}.\n"
        f"- Metadata-gated inactive full-text probes: {report['eligible_fulltext_probe_count']:,}.\n"
        f"- Planning projection: {projection['eligible_projected_cleaned_characters']:,} cleaned characters "
        f"({projection['eligible_projected_share_of_resulting_corpus']:.1%} of the resulting frozen-archive pool).\n"
        f"- Full-text probe runtime estimate: {runtime['fulltext_probe_lower_minutes']:,}-"
        f"{runtime['fulltext_probe_upper_minutes']:,} minutes before manual review and overlap analysis.\n\n"
        "The character estimate uses the median of the earlier 23-record Liber Liber probe. It is not "
        "downloaded text, a token count, or an activation decision.\n\n"
        "## Decisions\n\n| Decision | Records |\n| --- | ---: |\n"
        f"{decisions}\n\n"
        "## Preliminary Roles\n\n| Role | Records |\n| --- | ---: |\n"
        f"{roles}\n\n"
        "## Rights Boundary\n\n"
        "A work enters the inactive probe queue only when the item is marked free, pins the approved "
        "CC BY-NC-SA 4.0 edition license, advertises TXT ZIP or ODT, and has decisive standard-Italian "
        "historical/Ottocento metadata. Protected personal-use-only texts and unclear item licenses fail closed.\n\n"
        "## Checkpoint Boundary\n\n"
        "No archive full text was acquired. No source is activated, and no V7 split, mixture weight, "
        "cache deletion, or GPU work is authorized. Translation, dialect, drama-form, period, format, "
        "and rights holds require later bounded resolution.\n"
    )


def _metadata_fields(content: str) -> dict[str, str]:
    result: dict[str, list[str]] = {}
    for raw_label, raw_value in _METADATA_PAIR.findall(content):
        label = _plain_text(raw_label).casefold().rstrip(":").strip()
        value = _clean_credit(_plain_text(raw_value))
        if label and value:
            result.setdefault(label, []).append(value)
    return {key: "; ".join(dict.fromkeys(values)) for key, values in result.items()}


def _author_period_evidence(author_page: dict[str, Any] | None) -> tuple[list[int], str]:
    if author_page is None:
        return [], "Parent author page is unavailable."
    content = _rendered(author_page, "content")
    parent_metadata = _metadata_fields(content)
    if not parent_metadata.get("autore") or parent_metadata.get("titolo"):
        return [], "Parent page is not an unambiguous Liber Liber author record."
    paragraph_match = re.search(r"<p\b[^>]*>(.*?)</p>", content, re.IGNORECASE | re.DOTALL)
    if paragraph_match is None:
        return [], "Parent author page has no introductory biography paragraph."
    biography = _plain_text(paragraph_match.group(1))
    years = sorted({int(value) for value in _YEAR.findall(biography)})
    if len(years) >= 2:
        return years, (
            "Parent author biography first-paragraph observed years="
            f"{';'.join(str(value) for value in years)}; used only to test Dewey-period consistency."
        )
    if len(years) == 1 and _DEATH_WORD.search(biography):
        return years, f"Parent author biography explicit_death_year={years[0]}."
    return years, "Parent author biography lacks a decisive birth/death chronology."


def _download_evidence(links: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    formats = set()
    download_pages = set()
    mapping = {
        "opera_url_txt": "txt_zip", "opera_url_odt": "odt",
        "opera_url_pdf": "pdf", "opera_url_epub": "epub",
        "opera_url_rtf": "rtf_zip",
    }
    for url, label in links:
        query_type = parse_qs(urlparse(url).query).get("type", [""])[0]
        if query_type in mapping:
            formats.add(mapping[query_type]); download_pages.add(url)
        path = urlparse(url).path.casefold()
        label_folded = label.casefold()
        if path.endswith(".odt"):
            formats.add("odt")
        elif path.endswith(".epub"):
            formats.add("epub")
        elif path.endswith(".pdf"):
            formats.add("pdf")
        elif path.endswith(".zip") and "rtf" in label_folded:
            formats.add("rtf_zip")
        elif path.endswith(".zip") and "txt" in label_folded:
            formats.add("txt_zip")
    return sorted(formats), sorted(download_pages)


def _license_evidence(content: str, label: str, links: list[tuple[str, str]]) -> tuple[str, str]:
    urls = [url for url, _text in links if "creativecommons.org/licenses/" in url or "gnu.org/copyleft/fdl" in url]
    url = urls[-1] if urls else ""
    if "by-nc-sa/4.0" in url:
        return label or "Creative Commons Attribution-NonCommercial-ShareAlike 4.0", CC_BY_NC_SA_4_URL
    return label, url


def _load_existing_urls(path: Path) -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("source_archive") != "Liber Liber" or not row.get("landing_page_url"):
                continue
            result.setdefault(_normalize_url(row["landing_page_url"]), []).append(row["source_id"])
    return {key: tuple(sorted(values)) for key, values in result.items()}


def _validate_snapshot(payload: dict[str, Any], *, allow_incomplete: bool = False) -> None:
    if payload.get("inventory_version") != "liber_liber_archive_inventory_v1":
        raise ValueError("unexpected Liber Liber local cache version")
    pages = payload.get("pages")
    complete = payload.get("complete", True)
    if not isinstance(pages, list):
        raise ValueError("Liber Liber local cache pages are invalid")
    if complete and len(pages) != int(payload.get("total_records", -1)):
        raise ValueError("Liber Liber local cache count mismatch")
    if not complete and (
        not allow_incomplete
        or len(pages) >= int(payload.get("total_records", -1))
        or len(pages) % int(payload.get("per_page", 0) or 1)
    ):
        raise ValueError("Liber Liber partial cache accounting is invalid")
    ids = [int(row.get("id", 0)) for row in pages if isinstance(row, dict)]
    if len(ids) != len(pages) or any(value <= 0 for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Liber Liber local cache IDs are invalid")


def _positive_header(response: Any, key: str) -> int:
    try:
        value = int(response.headers[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"WordPress response is missing valid {key}") from error
    if value <= 0:
        raise ValueError(f"WordPress response has non-positive {key}")
    return value


def _rendered(page: dict[str, Any], key: str) -> str:
    value = page.get(key, {})
    return str(value.get("rendered", "")) if isinstance(value, dict) else ""


def _plain_text(value: str) -> str:
    value = _BREAK.sub("\n", value)
    value = _TAG.sub(" ", value)
    value = html.unescape(value).replace("\xa0", " ")
    return _SPACE.sub(" ", value).strip()


def _clean_credit(value: str) -> str:
    value = _EMAIL.sub("", value)
    value = re.sub(r"\s*,\s*(?=;|$)", "", value)
    return _SPACE.sub(" ", value).strip(" ,;")


def _normalize_url(value: str) -> str:
    return value.strip().rstrip("/").casefold()


def _portable(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary, payload)
    temporary.replace(path)


def _report(progress: Progress | None, message: str) -> None:
    if progress:
        progress(message)
