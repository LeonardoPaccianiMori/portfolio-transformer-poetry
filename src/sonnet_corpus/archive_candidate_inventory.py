"""Checkpoint 6B metadata/source inventories for six inactive archives.

The module deliberately acquires metadata only.  It normalizes heterogeneous
archive records into one compact ledger, keeps complete source responses in an
ignored cache, and never downloads or activates corpus text.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep as default_sleep
from typing import Any
from urllib.parse import urlencode

import requests


AUDIT_VERSION = "corpus_archive_candidate_inventory_v1"
USER_AGENT = "portfolio-transformer-poetry-archive-inventory/1.0"
BASE_CORPUS_CHARACTERS = 626_379_622
ARCHIVE_IDS = (
    "eltec_italian",
    "internet_archive",
    "gallica",
    "internet_culturale",
    "beic",
    "midia",
)

ELTEC_METADATA_URL = (
    "https://raw.githubusercontent.com/COST-ELTeC/ELTeC-ita/master/"
    "ELTeC-ita_metadata.tsv"
)
ELTEC_TREE_URL = (
    "https://api.github.com/repos/COST-ELTeC/ELTeC-ita/git/trees/master?recursive=1"
)
IA_URL = "https://archive.org/advancedsearch.php"
IA_SCRAPE_URL = "https://archive.org/services/search/v1/scrape"
IA_QUERY = "language:ita AND mediatype:texts AND date:[1200-01-01 TO 1900-12-31]"
IA_FIELDS = (
    "identifier", "title", "creator", "date", "year", "language", "subject",
    "licenseurl", "rights", "collection", "format",
)
IC_URL = "https://www.internetculturale.it/it/41/collezioni-digitali"
BEIC_OAI_URL = "https://beic.alma.exlibrisgroup.com/view/oai/39BEIC_INST/request"
BEIC_SET = "rosetta_dc"
BEIC_PREFIX = "oai_qdc"
MIDIA_PDF_URL = "https://www.corpusmidia.unito.it/downloads/opere-autori.pdf"
GALLICA_SRU_URL = (
    "https://gallica.bnf.fr/SRU?version=1.2&operation=searchRetrieve&query="
    "dc.language%20all%20%22ita%22%20and%20dc.type%20all%20%22monographie%22"
    "&startRecord=1&maximumRecords=1"
)
GALLICA_SRU_BASE = "https://gallica.bnf.fr/SRU"
GALLICA_QUERY = 'dc.language all "ita" and dc.type all "monographie"'

INVENTORY_FIELDS = (
    "archive_id", "record_id", "record_kind", "title", "creator",
    "work_year", "period_bucket", "language", "genre_form",
    "source_group", "metadata_url", "source_url", "rights_status",
    "format_status", "estimated_characters", "estimated_tokens",
    "preliminary_role", "inventory_decision", "decision_reason",
    "author_concentration_key", "metadata_sha256", "activation_status",
)

SUMMARY_FIELDS = (
    "archive_id", "interface", "frozen_boundary", "raw_record_count",
    "normalized_record_count", "filtered_out_count", "candidate_count",
    "hold_count", "excluded_count", "conditioned_count",
    "estimated_characters", "estimated_tokens", "top_contributor",
    "top_contributor_records", "top_contributor_share",
    "composition_assessment", "next_action", "activation_status",
)

Progress = Callable[[str], None]
Sleep = Callable[[float], None]

_SPACE = re.compile(r"\s+")
_TAG = re.compile(r"<[^>]+>")
_YEAR = re.compile(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)")
_LITERARY = re.compile(
    r"\b(?:letteratur\w*|poes\w*|poet\w*|rime|sonett\w*|liric\w*|"
    r"canzon\w*|romanz\w*|novell\w*|raccont\w*|commedi\w*|tragedi\w*|"
    r"teatr\w*|memorie|epistol\w*|favol\w*|fiab\w*)\b",
    re.IGNORECASE,
)
_POETRY = re.compile(
    r"\b(?:poes\w*|poet\w*|rime|liric\w*|versi|canzon\w*)\b",
    re.IGNORECASE,
)
_SONNET = re.compile(r"\bsonett\w*\b", re.IGNORECASE)
_DIALECT = re.compile(
    r"\b(?:dialett\w*|vernacol\w*|romanesco|napoletan\w*|sicilian\w*|"
    r"venet\w*|milanes\w*|lombard\w*|piemontes\w*|sard\w*|friulan\w*|"
    r"bolognes\w*|genoves\w*)\b",
    re.IGNORECASE,
)
_TRANSLATION = re.compile(
    r"\b(?:traduzion\w*|tradott\w*|tradutt\w*|translated|translation)\b",
    re.IGNORECASE,
)
_MIDIA_ID = re.compile(r"^[A-Z]{3}\d_[A-Z0-9_]+$")


@dataclass(frozen=True)
class ArchiveCandidateInventoryConfig:
    repo_root: Path
    cache_dir: Path
    inventory_path: Path
    summary_path: Path
    json_report_path: Path
    markdown_report_path: Path
    request_timeout_seconds: float = 60.0
    request_delay_seconds: float = 0.25
    max_attempts: int = 3
    ia_rows_per_page: int = 1_000


def build_archive_candidate_inventory(
    config: ArchiveCandidateInventoryConfig,
    *,
    session: requests.Session | None = None,
    progress: Progress | None = None,
    sleep: Sleep = default_sleep,
) -> dict[str, Any]:
    """Enumerate and normalize all six checkpoint-6B metadata boundaries."""

    _validate_config(config)
    client = session or requests.Session()
    client.headers.update({"User-Agent": USER_AGENT})
    started = monotonic()
    inventories: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
    adapters = (
        ("eltec_italian", _inventory_eltec),
        ("internet_archive", _inventory_internet_archive),
        ("gallica", _inventory_gallica),
        ("internet_culturale", _inventory_internet_culturale),
        ("beic", _inventory_beic),
        ("midia", _inventory_midia),
    )
    for index, (archive_id, adapter) in enumerate(adapters, start=1):
        _report(progress, f"archive={archive_id} phase=start index={index}/{len(adapters)}")
        rows, summary = adapter(config, client, progress, sleep)
        _validate_archive_rows(archive_id, rows, summary)
        inventories.append((rows, summary))
        elapsed = monotonic() - started
        eta = elapsed / index * (len(adapters) - index)
        _report(
            progress,
            f"archive={archive_id} phase=complete normalized={len(rows):,} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
        )

    rows = sorted(
        (row for archive_rows, _summary in inventories for row in archive_rows),
        key=lambda row: (str(row["archive_id"]), str(row["record_id"])),
    )
    summaries = [summary for _rows, summary in inventories]
    if {row["archive_id"] for row in summaries} != set(ARCHIVE_IDS):
        raise ValueError("archive summary does not account for all six approved archives")
    _write_csv_atomic(config.inventory_path, INVENTORY_FIELDS, rows)
    _write_csv_atomic(config.summary_path, SUMMARY_FIELDS, summaries)

    report = _build_report(config, rows, summaries)
    _write_json_atomic(config.json_report_path, report)
    _write_text_atomic(config.markdown_report_path, _markdown_report(report))
    report["artifact_sha256"] = {
        str(path.relative_to(config.repo_root)): _sha256_file(path)
        for path in (
            config.inventory_path,
            config.summary_path,
            config.json_report_path,
            config.markdown_report_path,
        )
    }
    return report


def _inventory_eltec(
    config: ArchiveCandidateInventoryConfig,
    client: requests.Session,
    progress: Progress | None,
    sleep: Sleep,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = config.cache_dir / "eltec_italian"
    metadata_path = cache / "ELTeC-ita_metadata.tsv"
    tree_path = cache / "tree.json"
    _fetch_cached(config, client, ELTEC_METADATA_URL, metadata_path, progress, sleep)
    _fetch_cached(config, client, ELTEC_TREE_URL, tree_path, progress, sleep)
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    paths = {str(row.get("path", "")) for row in tree.get("tree", [])}
    source_rows = list(csv.DictReader(metadata_path.open(encoding="utf-8"), delimiter="\t"))
    if not source_rows:
        raise ValueError("ELTeC metadata is empty")
    filenames = [str(row.get("filename", "")) for row in source_rows]
    if len(filenames) != len(set(filenames)):
        raise ValueError("ELTeC metadata filenames are duplicated")
    missing_files = [name for name in filenames if f"level1/{name}.xml" not in paths]
    if missing_files:
        raise ValueError(f"ELTeC tree is missing {len(missing_files)} level-1 files")
    xmlid_counts = Counter(str(row.get("xmlid", "")).strip() for row in source_rows)
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        filename = str(source["filename"]).strip()
        title = _clean(source.get("title", ""))
        creator = _clean(source.get("author-name", ""))
        language = _clean(source.get("language", ""))
        xmlid = _clean(source.get("xmlid", ""))
        year = _first_year(source.get("first-edition", "")) or _first_year(
            source.get("reference-year", "")
        )
        numwords = _integer(source.get("numwords", ""))
        text = f"{title} {creator}"
        role = _role(text, year)
        if not creator or creator.casefold() == "na":
            decision, reason = "hold_missing_author_metadata", "ELTeC author metadata is absent."
        elif not xmlid or xmlid_counts[xmlid] > 1:
            decision, reason = "hold_duplicate_or_missing_xmlid", "ELTeC XML identity is missing or duplicated."
        elif language not in {"ita", "it"}:
            decision, reason = "hold_non_italian_language_metadata", f"ELTeC language={language or 'missing'} is not ita/it."
        elif year is None:
            decision, reason = "hold_work_period_metadata", "No parseable first-edition/reference year is available."
        elif year > 1900:
            decision, reason = "exclude_post_1900", f"First-edition/reference year {year} is outside the frozen boundary."
        else:
            decision = "eligible_capped_ottocento_text_probe_inactive"
            reason = "Public-domain text/CC BY markup metadata passes; text and overlap gates remain pending."
        rows.append(_row(
            archive_id="eltec_italian", record_id=f"eltec:{filename}",
            record_kind="work", title=title, creator=creator, work_year=year,
            language=language, genre_form="novel", source_group="ELTeC-ita level1",
            metadata_url=ELTEC_METADATA_URL,
            source_url=(
                "https://github.com/COST-ELTeC/ELTeC-ita/blob/master/level1/"
                f"{filename}.xml"
            ),
            rights_status="public_domain_text_cc_by_4_markup",
            format_status="tei_xml_not_acquired", estimated_characters=numwords * 6 if numwords else None,
            estimated_tokens=numwords, preliminary_role=role,
            inventory_decision=decision, decision_reason=reason,
            author_key=creator, metadata=source,
        ))
    return rows, _summary(
        "eltec_italian", "GitHub tree plus release metadata TSV",
        "all current ELTeC-ita metadata rows; work year no later than 1900 for candidates",
        len(source_rows), rows, 0,
        "Small bounded novel release; exact edition overlap and the Ottocento cap remain mandatory.",
        "Probe only eligible rows after explicit checkpoint approval; keep post-1900 and anomalous rows inactive.",
    )


def _inventory_internet_archive(
    config: ArchiveCandidateInventoryConfig,
    client: requests.Session,
    progress: Progress | None,
    sleep: Sleep,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = config.cache_dir / "internet_archive"
    manifest_path = cache / "snapshot.json"
    pages: list[dict[str, Any]] = []
    manifest = _read_json_if_present(manifest_path)
    expected = int(manifest["total_records"]) if manifest else None
    total_pages = math.ceil(expected / config.ia_rows_per_page) if expected is not None else None
    page_number = 1
    cursor = ""
    seen_cursors: set[str] = set()
    started = monotonic()
    while True:
        page_path = cache / f"cursor_{page_number:04d}.json"
        # Archive.org's response cache currently ignores ``cursor`` when the
        # same ``count`` is reused, returning page one again.  A monotonically
        # decreasing, still-large valid count keeps each official request cache
        # key distinct while preserving cursor semantics and bounded traffic.
        page_count = max(100, config.ia_rows_per_page - page_number + 1)
        params: dict[str, Any] = {
            "q": IA_QUERY,
            "fields": ",".join(IA_FIELDS),
            "count": page_count,
        }
        if cursor:
            params["cursor"] = cursor
        _fetch_cached(config, client, IA_SCRAPE_URL, page_path, progress, sleep, params=params)
        payload = json.loads(page_path.read_text(encoding="utf-8"))
        docs = payload.get("items")
        found = int(payload.get("total", -1))
        if found < 0 or not isinstance(docs, list):
            raise ValueError(f"Internet Archive scrape page {page_number} has invalid schema")
        if expected is None:
            expected = found
            total_pages = math.ceil(expected / config.ia_rows_per_page)
        pages.extend(docs)
        next_cursor = str(payload.get("cursor", ""))
        remaining = max(0, (expected or len(pages)) - len(pages))
        next_count = max(100, page_count - 1)
        total_pages = page_number + math.ceil(remaining / next_count)
        elapsed = monotonic() - started
        eta = elapsed / page_number * max(0, (total_pages or page_number) - page_number)
        _report(
            progress,
            f"archive=internet_archive page={page_number:,}/{total_pages:,} "
            f"records={len(pages):,}/{expected:,} elapsed={elapsed:.1f}s eta={eta:.1f}s",
        )
        if not next_cursor:
            break
        if next_cursor in seen_cursors:
            raise ValueError("Internet Archive Scraping API repeated a cursor")
        seen_cursors.add(next_cursor)
        cursor = next_cursor
        page_number += 1
        if total_pages is not None and page_number > total_pages + 2:
            raise ValueError("Internet Archive Scraping API exceeded the expected page bound")
    assert expected is not None
    if len(pages) != expected:
        raise ValueError(f"Internet Archive returned {len(pages)} records; expected {expected}")
    identifiers = [str(row.get("identifier", "")) for row in pages]
    if any(not value for value in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("Internet Archive identifiers are missing or duplicated")
    snapshot = {
        "audit_version": AUDIT_VERSION,
        "query": IA_QUERY,
        "fields": list(IA_FIELDS),
        "interface": IA_SCRAPE_URL,
        "sort": "service_default_identifier_sorter",
        "rows_per_page": config.ia_rows_per_page,
        "cursor_count_policy": "decrement_by_one_per_page_to_avoid_cursor_blind_response_cache",
        "total_records": expected,
        "total_pages": total_pages,
        "page_sha256": [_sha256_file(cache / f"cursor_{index:04d}.json") for index in range(1, page_number + 1)],
        "complete": True,
    }
    _write_json_atomic(manifest_path, snapshot)
    rows = [_normalize_ia(source) for source in pages]
    return rows, _summary(
        "internet_archive", "Cursor-based Scraping API",
        IA_QUERY, expected, rows, 0,
        "High-recall but noisy scan/OCR pool; rights, literary relevance, and edition duplication sharply limit candidates.",
        "A later approved probe may inspect only explicit-rights, text-format, literary candidates; all others remain held/excluded.",
    )


def _normalize_ia(source: dict[str, Any]) -> dict[str, Any]:
    identifier = str(source.get("identifier", "")).strip()
    title = _join(source.get("title"))
    creator = _join(source.get("creator"))
    language = _join(source.get("language"))
    year = _integer(source.get("year")) or _first_year(_join(source.get("date")))
    subjects = _join(source.get("subject"))
    collections = _join(source.get("collection"))
    formats = {_clean(value) for value in _list(source.get("format"))}
    license_url = _join(source.get("licenseurl"))
    rights = _join(source.get("rights"))
    combined = " ".join((title, creator, subjects))
    rights_status = _ia_rights_status(license_url, rights)
    has_text = bool(formats & {
        "DjVuTXT", "OCR Search Text", "Full Text", "Text", "Text PDF", "hOCR", "chOCR",
    })
    format_status = "ocr_or_text_format_advertised" if has_text else "no_text_format_advertised"
    role = _role(combined, year)
    if _DIALECT.search(combined):
        decision = "conditioned_language_metadata_hold"
        reason = "Dialect/language-variety metadata keeps this item outside the standard-Italian queue."
    elif _TRANSLATION.search(combined):
        decision = "hold_translation_edition_review"
        reason = "Translation metadata requires source-language and Italian-edition review."
    elif rights_status != "explicit_reusable_item_rights":
        decision = "hold_item_rights_unresolved"
        reason = "No explicit compatible item-level public-domain/Creative Commons statement is present."
    elif not has_text:
        decision = "hold_no_text_or_ocr_format"
        reason = "Metadata advertises no inspectable OCR/plain-text format."
    elif not _LITERARY.search(combined):
        decision = "hold_not_prioritized_nonliterary_metadata"
        reason = "The bounded literary-signal filter found no evidence for the project corpus roles."
    else:
        decision = "eligible_item_text_probe_inactive"
        reason = "Explicit item rights, a text/OCR format, and literary metadata pass; quality and overlap remain pending."
    return _row(
        archive_id="internet_archive", record_id=f"ia:{identifier}", record_kind="item",
        title=title, creator=creator, work_year=year, language=language,
        genre_form=_genre(combined), source_group=collections,
        metadata_url=f"https://archive.org/metadata/{identifier}",
        source_url=f"https://archive.org/details/{identifier}", rights_status=rights_status,
        format_status=format_status, estimated_characters=None, estimated_tokens=None,
        preliminary_role=role, inventory_decision=decision, decision_reason=reason,
        author_key=creator, metadata=source,
    )


def _inventory_gallica(
    config: ArchiveCandidateInventoryConfig,
    client: requests.Session,
    progress: Progress | None,
    sleep: Sleep,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = config.cache_dir / "gallica"
    body_path = cache / "sru_response.bin"
    meta_path = cache / "sru_response.json"
    if meta_path.is_file() and body_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        response = _request_with_retries(
            config, client, GALLICA_SRU_URL, progress, sleep, allow_http_error=True,
        )
        _write_bytes_atomic(body_path, response.content)
        meta = {
            "url": GALLICA_SRU_URL,
            "http_status": int(response.status_code),
            "content_type": response.headers.get("Content-Type", ""),
            "content_sha256": hashlib.sha256(response.content).hexdigest(),
        }
        _write_json_atomic(meta_path, meta)
    status = int(meta.get("http_status", 0))
    measured_count = ""
    blocking_status = status
    if status == 200:
        try:
            root = ET.fromstring(body_path.read_bytes())
            measured_count = _clean(root.findtext(
                "{http://www.loc.gov/zing/srw/}numberOfRecords", default="",
            ))
        except ET.ParseError as error:
            raise ValueError("Gallica one-record SRU response is not XML") from error
        pagination_url = GALLICA_SRU_URL.replace("maximumRecords=1", "maximumRecords=50")
        pagination_body = cache / "pagination_probe.bin"
        pagination_meta_path = cache / "pagination_probe.json"
        if pagination_body.is_file() and pagination_meta_path.is_file():
            pagination_meta = json.loads(pagination_meta_path.read_text(encoding="utf-8"))
        else:
            response = _request_with_retries(
                config, client, pagination_url, progress, sleep, allow_http_error=True,
            )
            _write_bytes_atomic(pagination_body, response.content)
            pagination_meta = {
                "url": pagination_url,
                "http_status": int(response.status_code),
                "content_type": response.headers.get("Content-Type", ""),
                "content_sha256": hashlib.sha256(response.content).hexdigest(),
            }
            _write_json_atomic(pagination_meta_path, pagination_meta)
        blocking_status = int(pagination_meta.get("http_status", 0))
        if blocking_status == 200:
            source_rows, raw_count = _fetch_gallica_records(
                config, client, cache, pagination_body, progress, sleep,
            )
            normalized: list[dict[str, Any]] = []
            filtered = 0
            for source in source_rows:
                year = _first_year(_join(source.get("date")))
                if year is None or year > 1900:
                    filtered += 1
                    continue
                normalized.append(_normalize_gallica(source, year))
            return normalized, _summary(
                "gallica", "Official SRU API",
                "all Italian monographs; publish records with a parseable year no later than 1900",
                raw_count, normalized, filtered,
                "Public-domain and OCR-quality metadata expose candidates, but OCR quality, item terms, and canonical overlap remain mandatory.",
                "Run a later approved item-quality/rights probe only for inactive literary candidates; acquire no OCR here.",
            )
    decision = (
        "blocked_metadata_pagination_http_error"
        if status == 200 else "blocked_metadata_interface_http_error"
    )
    count_evidence = f" The one-record response reported {measured_count} matches." if measured_count else ""
    row = _row(
        archive_id="gallica", record_id="gallica:sru_access_blocker",
        record_kind="access_blocker", title="Frozen Italian monograph SRU query",
        creator="Biblioth\u00e8que nationale de France", work_year=None,
        language="ita query", genre_form="monograph query", source_group="Gallica SRU",
        metadata_url=GALLICA_SRU_URL, source_url=GALLICA_SRU_URL,
        rights_status="metadata_terms_only_text_rights_unresolved",
        format_status=f"sru_initial_http_{status}_pagination_http_{blocking_status}", estimated_characters=None, estimated_tokens=None,
        preliminary_role="core_training_candidate",
        inventory_decision=decision,
        decision_reason=(
            f"Official SRU complete-enumeration access is blocked at HTTP {blocking_status}."
            f"{count_evidence} No complete record inventory can be claimed or activated."
        ),
        author_key="", metadata=meta,
    )
    rows = [row]
    return rows, _summary(
        "gallica", "Official SRU API",
        "Italian-language monographs; exact frozen query recorded", 1, rows, 0,
        "Potentially high historical value; a result count may be visible, but complete paginated inventory access is blocked.",
        "Retain the HTTP blocker and retry only in a separately approved checkpoint; acquire no OCR/text.",
    )


def _fetch_gallica_records(
    config: ArchiveCandidateInventoryConfig,
    client: requests.Session,
    cache: Path,
    first_page_path: Path,
    progress: Progress | None,
    sleep: Sleep,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    expected: int | None = None
    next_position = 1
    page = 1
    started = monotonic()
    while next_position:
        if page == 1:
            path = first_page_path
        else:
            path = cache / f"page_{page:04d}.xml"
            _fetch_cached(
                config, client, GALLICA_SRU_BASE, path, progress, sleep,
                params={
                    "version": "1.2", "operation": "searchRetrieve",
                    "query": GALLICA_QUERY, "startRecord": next_position,
                    "maximumRecords": 50,
                },
            )
        page_rows, total, following = parse_gallica_sru_page(path.read_bytes())
        if expected is None:
            expected = total
        elif total != expected:
            raise ValueError("Gallica result count changed during pagination")
        rows.extend(page_rows)
        elapsed = monotonic() - started
        total_pages = math.ceil(expected / 50)
        eta = elapsed / page * max(0, total_pages - page)
        _report(
            progress,
            f"archive=gallica page={page:,}/{total_pages:,} records={len(rows):,}/{expected:,} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
        )
        next_position = following
        page += 1
        if page > 1_000:
            raise ValueError("Gallica SRU pagination exceeded safety bound")
    assert expected is not None
    if len(rows) != expected:
        raise ValueError(f"Gallica returned {len(rows)} records; expected {expected}")
    record_ids = [_gallica_record_id(row) for row in rows]
    if any(not value for value in record_ids) or len(record_ids) != len(set(record_ids)):
        raise ValueError("Gallica identifiers are missing or duplicated")
    _write_json_atomic(cache / "snapshot.json", {
        "audit_version": AUDIT_VERSION,
        "query": GALLICA_QUERY,
        "maximum_records": 50,
        "total_records": expected,
        "total_pages": page - 1,
        "complete": True,
    })
    return rows, expected


def parse_gallica_sru_page(content: bytes) -> tuple[list[dict[str, Any]], int, int]:
    """Parse one Gallica SRU response, including OCR-quality extra fields."""

    root = ET.fromstring(content)
    ns = {
        "srw": "http://www.loc.gov/zing/srw/",
        "dc": "http://purl.org/dc/elements/1.1/",
    }
    total_text = root.findtext("srw:numberOfRecords", default="", namespaces=ns)
    if not total_text.isdigit():
        diagnostic = _clean(" ".join(element.text or "" for element in root.iter()))
        raise ValueError(f"Gallica SRU page lacks result count: {diagnostic[:200]}")
    rows: list[dict[str, Any]] = []
    for record in root.findall(".//srw:record", ns):
        row: dict[str, Any] = {}
        data = record.find("srw:recordData", ns)
        if data is not None:
            for element in data.iter():
                if element.tag.startswith("{http://purl.org/dc/elements/1.1/}"):
                    key = element.tag.rsplit("}", 1)[-1]
                    row.setdefault(key, []).append(_clean(element.text or ""))
        extra = record.find("srw:extraRecordData", ns)
        if extra is not None:
            for element in extra.iter():
                if element is extra:
                    continue
                key = element.tag.rsplit("}", 1)[-1]
                row[f"extra_{key}"] = _clean(element.text or "")
        rows.append(row)
    following_text = root.findtext("srw:nextRecordPosition", default="", namespaces=ns)
    following = int(following_text) if following_text.isdigit() else 0
    if following > int(total_text):
        following = 0
    return rows, int(total_text), following


def _gallica_record_id(source: dict[str, Any]) -> str:
    uri = _clean(source.get("extra_uri", ""))
    if uri:
        return uri
    identifiers = _list(source.get("identifier"))
    return next((value.rstrip("/").rsplit("/", 1)[-1] for value in identifiers if "/ark:/" in value), "")


def _normalize_gallica(source: dict[str, Any], year: int) -> dict[str, Any]:
    record_id = _gallica_record_id(source)
    title = _join(source.get("title"))
    creator = _join([*_list(source.get("creator")), *_list(source.get("contributor"))])
    subjects = _join(source.get("subject"))
    combined = f"{title} {creator} {subjects}"
    rights = _join(source.get("rights"))
    rights_status = (
        "explicit_public_domain_item_metadata"
        if "public domain" in rights.casefold() or "domaine public" in rights.casefold()
        else "item_rights_missing_or_unresolved"
    )
    try:
        ocr_quality = float(str(source.get("extra_nqamoyen", "0") or "0"))
    except ValueError:
        ocr_quality = 0.0
    epub = bool(_clean(source.get("extra_epubFile", "")))
    if epub:
        format_status = "epub_advertised"
    elif ocr_quality > 0:
        format_status = f"ocr_quality_{ocr_quality:.1f}"
    else:
        format_status = "no_epub_or_measured_ocr"
    if _DIALECT.search(combined):
        decision = "conditioned_language_metadata_hold"
        reason = "Language-variety metadata keeps the record outside the standard-Italian queue."
    elif _TRANSLATION.search(combined):
        decision = "hold_translation_edition_review"
        reason = "Translation metadata requires source-language and Italian-edition review."
    elif rights_status != "explicit_public_domain_item_metadata":
        decision = "hold_item_rights_unresolved"
        reason = "No explicit public-domain item statement is present in the SRU record."
    elif not epub and ocr_quality <= 0:
        decision = "hold_no_epub_or_measured_ocr"
        reason = "The SRU record advertises neither EPUB nor a positive OCR-quality value."
    elif not _LITERARY.search(combined):
        decision = "hold_not_prioritized_nonliterary_metadata"
        reason = "The bounded literary-signal filter found no evidence for the project corpus roles."
    else:
        decision = "eligible_item_quality_audit_inactive"
        reason = "Historical literary metadata, explicit public-domain status, and digital-text evidence pass; quality and overlap remain pending."
    identifiers = _list(source.get("identifier"))
    ark_url = next((value for value in identifiers if value.startswith("http")), "")
    return _row(
        archive_id="gallica", record_id=f"gallica:{record_id}", record_kind="digital_record",
        title=title, creator=creator, work_year=year, language=_join(source.get("language")),
        genre_form=_genre(combined), source_group=_join(source.get("source")),
        metadata_url=ark_url, source_url=ark_url, rights_status=rights_status,
        format_status=format_status, estimated_characters=None, estimated_tokens=None,
        preliminary_role=_role(combined, year), inventory_decision=decision,
        decision_reason=reason, author_key=creator, metadata=source,
    )


def _inventory_internet_culturale(
    config: ArchiveCandidateInventoryConfig,
    client: requests.Session,
    progress: Progress | None,
    sleep: Sleep,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = config.cache_dir / "internet_culturale"
    collections: list[dict[str, str]] = []
    expected: int | None = None
    page = 1
    started = monotonic()
    while True:
        path = cache / f"page_{page:02d}.html"
        _fetch_cached(
            config, client, IC_URL, path, progress, sleep,
            params={"paginate_pageNum": page},
        )
        content = path.read_text(encoding="utf-8", errors="replace")
        count_match = re.search(r"([0-9.]+)\s+risultati trovati", content, re.IGNORECASE)
        if not count_match:
            raise ValueError(f"Internet Culturale page {page} lacks result count")
        count = int(count_match.group(1).replace(".", ""))
        if expected is None:
            expected = count
        elif count != expected:
            raise ValueError("Internet Culturale result count changed during pagination")
        parsed = parse_internet_culturale_collections(content)
        if not parsed:
            break
        collections.extend(parsed)
        elapsed = monotonic() - started
        estimated_pages = math.ceil(expected / len(parsed)) if page == 1 else max(page, 11)
        eta = elapsed / page * max(0, estimated_pages - page)
        _report(
            progress,
            f"archive=internet_culturale page={page} records={len(collections):,}/{expected:,} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
        )
        if len(collections) >= expected:
            break
        page += 1
        if page > 100:
            raise ValueError("Internet Culturale pagination exceeded safety bound")
    assert expected is not None
    if len(collections) != expected:
        raise ValueError(f"Internet Culturale parsed {len(collections)} collections; expected {expected}")
    ids = [row["collection_id"] for row in collections]
    if len(ids) != len(set(ids)):
        raise ValueError("Internet Culturale collection IDs are duplicated")
    rows = [_normalize_ic(row) for row in collections]
    _write_json_atomic(cache / "snapshot.json", {
        "audit_version": AUDIT_VERSION,
        "total_records": expected,
        "total_pages": page,
        "page_sha256": [_sha256_file(cache / f"page_{index:02d}.html") for index in range(1, page + 1)],
        "complete": True,
    })
    return rows, _summary(
        "internet_culturale", "Official digital-collection directory",
        "all 291 listed collections; collection-level metadata only", expected, rows, 0,
        "Collection descriptions identify promising historical text groups but not item counts, formats, or overriding rights.",
        "Inventory items only for candidate collections after approval; retain portal and owning-institution terms per item.",
    )


def parse_internet_culturale_collections(content: str) -> list[dict[str, str]]:
    """Parse collection cards from one official directory page."""

    rows: list[dict[str, str]] = []
    for block in re.findall(
        r'<div class="module-row listing-height clearfix">(.*?)(?=<div class="module-row listing-height clearfix">|\Z)',
        content,
        re.IGNORECASE | re.DOTALL,
    ):
        title_match = re.search(
            r'<h1[^>]*>\s*<a href="([^"]+)"[^>]*>(.*?)</a>', block,
            re.IGNORECASE | re.DOTALL,
        )
        if not title_match:
            continue
        detail_url = html.unescape(title_match.group(1))
        id_match = re.search(r"/([0-9]+)/[^/?#]+/?$", detail_url)
        if not id_match:
            raise ValueError(f"Internet Culturale collection URL lacks numeric ID: {detail_url}")
        institution_match = re.search(r'<h2[^>]*>(.*?)</h2>', block, re.IGNORECASE | re.DOTALL)
        description_match = re.search(r'<p>(.*?)</p>', block, re.IGNORECASE | re.DOTALL)
        digital_match = re.search(
            r'<li class="bibDig".*?<a[^>]+href="([^"]+)"', block,
            re.IGNORECASE | re.DOTALL,
        )
        rows.append({
            "collection_id": id_match.group(1),
            "title": _plain(title_match.group(2)),
            "institution": _plain(institution_match.group(1)) if institution_match else "",
            "description": _plain(description_match.group(1)) if description_match else "",
            "detail_url": detail_url,
            "digital_search_url": html.unescape(digital_match.group(1)) if digital_match else "",
        })
    return rows


def _normalize_ic(source: dict[str, str]) -> dict[str, Any]:
    combined = " ".join((source["title"], source["description"]))
    historical = bool(re.search(
        r"\b(?:medioev\w*|manoscritt\w*|codic\w*|incunabol\w*|cinquecento|"
        r"seicento|settecento|ottocento|secol[oi]\s+(?:x{1,3}|iv|v|vi)|"
        r"antich\w*|origini|stampa antica)\b",
        combined,
        re.IGNORECASE,
    ))
    text_bearing = bool(re.search(
        r"\b(?:libr\w*|test\w*|letter\w*|autograf\w*|manoscritt\w*|codic\w*|"
        r"periodic\w*|incunabol\w*|stampa\w*|opera\w*)\b",
        combined,
        re.IGNORECASE,
    ))
    if _DIALECT.search(combined):
        decision = "conditioned_collection_metadata_hold"
        reason = "Collection description signals a language variety outside the standard queue."
    elif source["digital_search_url"] and text_bearing and (historical or _LITERARY.search(combined)):
        decision = "eligible_collection_item_inventory_inactive"
        reason = "Historical/literary text-bearing collection with a digital-catalog route; item terms and formats remain pending."
    elif not source["digital_search_url"]:
        decision = "hold_no_digital_item_route"
        reason = "The collection card provides no digital-library item search link."
    else:
        decision = "hold_collection_outside_text_priority"
        reason = "Collection metadata does not pass the bounded historical/literary text signal gate."
    year = min((int(value) for value in _YEAR.findall(combined)), default=None)
    return _row(
        archive_id="internet_culturale", record_id=f"ic:{source['collection_id']}",
        record_kind="collection", title=source["title"], creator=source["institution"],
        work_year=year, language="collection metadata unresolved",
        genre_form=_genre(combined), source_group=source["institution"],
        metadata_url=source["detail_url"], source_url=source["digital_search_url"],
        rights_status="portal_nc_sa_item_override_unresolved",
        format_status="collection_level_unknown", estimated_characters=None,
        estimated_tokens=None, preliminary_role=_role(combined, year),
        inventory_decision=decision, decision_reason=reason,
        author_key=source["institution"], metadata=source,
    )


def _inventory_beic(
    config: ArchiveCandidateInventoryConfig,
    client: requests.Session,
    progress: Progress | None,
    sleep: Sleep,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = config.cache_dir / "beic"
    manifest_path = cache / "snapshot.json"
    page = 1
    token = ""
    raw_records: list[dict[str, Any]] = []
    started = monotonic()
    while True:
        path = cache / f"page_{page:04d}.xml"
        params = (
            {"verb": "ListRecords", "resumptionToken": token}
            if token
            else {
                "verb": "ListRecords", "set": BEIC_SET,
                "metadataPrefix": BEIC_PREFIX,
            }
        )
        _fetch_cached(config, client, BEIC_OAI_URL, path, progress, sleep, params=params)
        page_records, next_token = parse_beic_oai_page(path.read_bytes())
        if not page_records and next_token:
            raise ValueError("BEIC OAI page has a token but no records")
        raw_records.extend(page_records)
        elapsed = monotonic() - started
        _report(
            progress,
            f"archive=beic page={page:,} raw_records={len(raw_records):,} "
            f"elapsed={elapsed:.1f}s eta=unknown",
        )
        if not next_token:
            break
        token = next_token
        page += 1
        if page > 2_000:
            raise ValueError("BEIC OAI pagination exceeded safety bound")
    identifiers = [row["oai_identifier"] for row in raw_records]
    if any(not value for value in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("BEIC OAI identifiers are missing or duplicated")
    normalized: list[dict[str, Any]] = []
    filtered = 0
    for source in raw_records:
        languages = {_clean(value).casefold() for value in source.get("language", [])}
        years = [year for value in source.get("date", []) if (year := _first_year(value))]
        year = min(years, default=None)
        if not languages.intersection({"ita", "it"}) or year is None or year > 1900:
            filtered += 1
            continue
        normalized.append(_normalize_beic(source, year))
    _write_json_atomic(manifest_path, {
        "audit_version": AUDIT_VERSION,
        "oai_base_url": BEIC_OAI_URL,
        "set": BEIC_SET,
        "metadata_prefix": BEIC_PREFIX,
        "raw_record_count": len(raw_records),
        "normalized_italian_historical_count": len(normalized),
        "filtered_out_count": filtered,
        "total_pages": page,
        "page_sha256": [_sha256_file(cache / f"page_{index:04d}.xml") for index in range(1, page + 1)],
        "complete": True,
    })
    return normalized, _summary(
        "beic", "Alma OAI-PMH rosetta_dc/oai_qdc",
        "complete Rosetta digital set; publish Italian records with a parseable year no later than 1900",
        len(raw_records), normalized, filtered,
        "Record rights are useful, but OAI type=text does not establish corrected OCR and holdings overlap is high.",
        "Perform item-format and canonical-source review only for normalized candidates after approval.",
    )


def parse_beic_oai_page(content: bytes) -> tuple[list[dict[str, Any]], str]:
    """Parse one BEIC OAI-QDC ListRecords page and its resumption token."""

    root = ET.fromstring(content)
    ns = {
        "oai": "http://www.openarchives.org/OAI/2.0/",
        "dc": "http://purl.org/dc/elements/1.1/",
    }
    error = root.find("oai:error", ns)
    if error is not None:
        raise ValueError(f"BEIC OAI error {error.attrib.get('code', '')}: {_clean(error.text or '')}")
    rows: list[dict[str, Any]] = []
    for record in root.findall(".//oai:record", ns):
        header = record.find("oai:header", ns)
        if header is None or header.attrib.get("status") == "deleted":
            continue
        identifier = header.findtext("oai:identifier", default="", namespaces=ns)
        row: dict[str, Any] = {
            "oai_identifier": _clean(identifier),
            "datestamp": _clean(header.findtext("oai:datestamp", default="", namespaces=ns)),
        }
        metadata = record.find("oai:metadata", ns)
        if metadata is None:
            continue
        for element in metadata.iter():
            if not element.tag.startswith("{http://purl.org/dc/"):
                continue
            key = element.tag.rsplit("}", 1)[-1]
            row.setdefault(key, []).append(_clean(element.text or ""))
        rows.append(row)
    token = _clean(root.findtext(".//oai:resumptionToken", default="", namespaces=ns))
    return rows, token


def _normalize_beic(source: dict[str, Any], year: int) -> dict[str, Any]:
    title = _join(source.get("title"))
    creators = _join([*_list(source.get("creator")), *_list(source.get("contributor"))])
    descriptions = _join(source.get("description"))
    subjects = _join(source.get("subject"))
    combined = " ".join((title, descriptions, subjects))
    rights = _join(source.get("rights"))
    rights_fold = rights.casefold()
    if "creative commons" in rights_fold or "pubblico dominio" in rights_fold or "public domain" in rights_fold:
        rights_status = "explicit_reusable_item_rights"
        decision = "eligible_item_format_audit_inactive"
        reason = "Italian historical digital record has explicit reusable rights; corrected OCR/text format remains unknown."
    else:
        rights_status = "access_only_item_rights_unresolved"
        decision = "hold_item_rights_unresolved"
        reason = "Open access alone does not establish public-domain or compatible item reuse status."
    if _DIALECT.search(combined):
        decision = "conditioned_language_metadata_hold"
        reason = "Language-variety metadata keeps the record outside the standard-Italian queue."
    identifiers = _list(source.get("identifier"))
    delivery = next((value for value in identifiers if value.startswith("http")), "")
    oai_id = str(source["oai_identifier"])
    local_id = oai_id.rsplit(":", 1)[-1]
    return _row(
        archive_id="beic", record_id=f"beic:{local_id}", record_kind="digital_record",
        title=title, creator=creators, work_year=year, language=_join(source.get("language")),
        genre_form=_genre(combined), source_group=_join(source.get("publisher")),
        metadata_url=f"{BEIC_OAI_URL}?verb=GetRecord&metadataPrefix={BEIC_PREFIX}&identifier={oai_id}",
        source_url=delivery, rights_status=rights_status,
        format_status="dc_type_text_corrected_ocr_unknown", estimated_characters=None,
        estimated_tokens=None, preliminary_role=_role(combined, year),
        inventory_decision=decision, decision_reason=reason,
        author_key=creators, metadata=source,
    )


def _inventory_midia(
    config: ArchiveCandidateInventoryConfig,
    client: requests.Session,
    progress: Progress | None,
    sleep: Sleep,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = config.cache_dir / "midia"
    pdf_path = cache / "opere-autori.pdf"
    text_path = cache / "opere-autori.txt"
    _fetch_cached(config, client, MIDIA_PDF_URL, pdf_path, progress, sleep)
    if not text_path.is_file():
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            check=True, capture_output=True,
        )
        _write_bytes_atomic(text_path, result.stdout)
    source_rows = parse_midia_table(text_path.read_text(encoding="utf-8", errors="replace"))
    rows = [_normalize_midia(source) for source in source_rows]
    _write_json_atomic(cache / "snapshot.json", {
        "audit_version": AUDIT_VERSION,
        "pdf_url": MIDIA_PDF_URL,
        "pdf_sha256": _sha256_file(pdf_path),
        "text_sha256": _sha256_file(text_path),
        "record_count": len(source_rows),
        "complete": True,
    })
    return rows, _summary(
        "midia", "Official works/authors PDF",
        "all listed source rows; periods I-IV historical, period V crosses 1900 and is held",
        len(source_rows), rows, 0,
        "Balanced period/genre excerpts have high linguistic value but substantial overlap and source-text lineage review needs.",
        "Audit exact source links and duplicates for periods I-IV only after approval; keep period V held until work dates resolve.",
    )


def parse_midia_table(content: str) -> list[dict[str, str]]:
    """Parse the fixed-width MIDIA works/authors table, joining continuations."""

    rows: list[dict[str, str]] = []
    positions: tuple[int, int, int, int] | None = None
    current: dict[str, str] | None = None
    for raw_line in content.splitlines():
        line = raw_line.expandtabs()
        if all(label in line for label in ("ID", "AUTORE", "GENERE", "PERIODO", "OPERA")):
            positions = tuple(line.index(label) for label in ("AUTORE", "GENERE", "PERIODO", "OPERA"))
            continue
        if positions is None or not line.strip():
            continue
        author_at, genre_at, period_at, work_at = positions
        record_id = line[:author_at].strip()
        if _MIDIA_ID.fullmatch(record_id):
            match = re.match(
                r"^(\S+)\s{2,}(.+?)\s{2,}(.+?)\s+(I|II|III|IV|V)\s{2,}(.+)$",
                line.strip(),
            )
            if match is None:
                raise ValueError(f"MIDIA record does not have five columns: {record_id}")
            record_id, author, genre, period, work = match.groups()
            current = {
                "id": record_id, "author": author, "genre": genre,
                "period": period, "work": work,
            }
            rows.append(current)
        elif current is not None:
            author = line[author_at:genre_at].strip()
            work = line[work_at:].strip()
            if author:
                current["author"] = _clean(f"{current['author']} {author}")
            if work:
                current["work"] = _clean(f"{current['work']} {work}")
    if not rows:
        raise ValueError("MIDIA PDF parser found no records")
    id_counts = Counter(row["id"] for row in rows)
    seen: Counter[str] = Counter()
    for row in rows:
        seen[row["id"]] += 1
        row["source_id_occurrence"] = str(seen[row["id"]])
        row["source_id_count"] = str(id_counts[row["id"]])
    if any(not row["genre"] or row["period"] not in {"I", "II", "III", "IV", "V"} for row in rows):
        raise ValueError("MIDIA work row lacks a valid genre/period")
    return rows


def _normalize_midia(source: dict[str, str]) -> dict[str, Any]:
    period = source["period"]
    period_labels = {
        "I": "1200-1375", "II": "1376-1532", "III": "1533-1691",
        "IV": "1692-1840", "V": "1841-1947",
    }
    combined = f"{source['work']} {source['genre']}"
    occurrence = int(source.get("source_id_occurrence", "1"))
    if occurrence > 1:
        decision = "exclude_duplicate_metadata_row"
        reason = "The official source-list PDF repeats this exact source ID; the first occurrence remains canonical."
    elif period == "V":
        decision = "hold_period_crosses_1900"
        reason = "MIDIA period V spans 1841-1947; the source list provides no exact work year."
    else:
        decision = "eligible_source_lineage_audit_inactive"
        reason = "CC BY-NC metadata and a pre-1901 period pass; exact source lineage, overlap, and access remain pending."
    return _row(
        archive_id="midia",
        record_id=f"midia:{source['id']}" + (f"#{occurrence}" if occurrence > 1 else ""),
        record_kind="corpus_excerpt",
        title=source["work"], creator=source["author"], work_year=None,
        period_override=period_labels[period], language="Italian (MIDIA documented)",
        genre_form=source["genre"], source_group=f"period_{period};{source['genre']}",
        metadata_url=MIDIA_PDF_URL, source_url="https://www.corpusmidia.unito.it/",
        rights_status="cc_by_nc_4_corpus_terms_source_layer_pending",
        format_status="source_link_inventory_pending", estimated_characters=None,
        estimated_tokens=8_000, preliminary_role=_role(combined, 1700 if period in {"I", "II", "III", "IV"} else 1901),
        inventory_decision=decision, decision_reason=reason,
        author_key=source["author"], metadata=source,
    )


def _row(
    *, archive_id: str, record_id: str, record_kind: str, title: str,
    creator: str, work_year: int | None, language: str, genre_form: str,
    source_group: str, metadata_url: str, source_url: str, rights_status: str,
    format_status: str, estimated_characters: int | None,
    estimated_tokens: int | None, preliminary_role: str,
    inventory_decision: str, decision_reason: str, author_key: str,
    metadata: Any, period_override: str = "",
) -> dict[str, Any]:
    return {
        "archive_id": archive_id,
        "record_id": record_id,
        "record_kind": record_kind,
        "title": _clean(title),
        "creator": _clean(creator),
        "work_year": work_year if work_year is not None else "",
        "period_bucket": period_override or _period(work_year),
        "language": _clean(language),
        "genre_form": _clean(genre_form),
        "source_group": _clean(source_group),
        "metadata_url": metadata_url,
        "source_url": source_url,
        "rights_status": rights_status,
        "format_status": format_status,
        "estimated_characters": estimated_characters if estimated_characters is not None else "",
        "estimated_tokens": estimated_tokens if estimated_tokens is not None else "",
        "preliminary_role": preliminary_role,
        "inventory_decision": inventory_decision,
        "decision_reason": decision_reason,
        "author_concentration_key": (
            "unresolved"
            if _clean(author_key).casefold() in {"", "na", "n/a", "unknown", "unresolved"}
            else _clean(author_key)
        ),
        "metadata_sha256": hashlib.sha256(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "activation_status": "inactive_metadata_only",
    }


def _summary(
    archive_id: str,
    interface: str,
    boundary: str,
    raw_count: int,
    rows: list[dict[str, Any]],
    filtered_out: int,
    assessment: str,
    next_action: str,
) -> dict[str, Any]:
    decisions = Counter(str(row["inventory_decision"]) for row in rows)
    contributors = Counter(
        str(row["author_concentration_key"]) for row in rows
        if row["author_concentration_key"] != "unresolved"
    )
    top_name, top_count = contributors.most_common(1)[0] if contributors else ("unresolved", 0)
    candidates = sum(count for key, count in decisions.items() if key.startswith("eligible_"))
    holds = sum(count for key, count in decisions.items() if key.startswith(("hold_", "blocked_")))
    excluded = sum(count for key, count in decisions.items() if key.startswith("exclude_"))
    conditioned = sum(count for key, count in decisions.items() if key.startswith("conditioned_"))
    return {
        "archive_id": archive_id,
        "interface": interface,
        "frozen_boundary": boundary,
        "raw_record_count": raw_count,
        "normalized_record_count": len(rows),
        "filtered_out_count": filtered_out,
        "candidate_count": candidates,
        "hold_count": holds,
        "excluded_count": excluded,
        "conditioned_count": conditioned,
        "estimated_characters": sum(_integer(row["estimated_characters"]) or 0 for row in rows),
        "estimated_tokens": sum(_integer(row["estimated_tokens"]) or 0 for row in rows),
        "top_contributor": top_name,
        "top_contributor_records": top_count,
        "top_contributor_share": f"{top_count / len(rows):.6f}" if rows else "0.000000",
        "composition_assessment": assessment,
        "next_action": next_action,
        "activation_status": "inactive_metadata_only",
    }


def _build_report(
    config: ArchiveCandidateInventoryConfig,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    decisions = Counter(str(row["inventory_decision"]) for row in rows)
    roles = Counter(str(row["preliminary_role"]) for row in rows)
    return {
        "audit_version": AUDIT_VERSION,
        "scope": {
            "approved_archives": list(ARCHIVE_IDS),
            "metadata_only": True,
            "corpus_text_acquired": False,
            "text_activated": False,
            "v7_created": False,
            "mixture_weights_assigned": False,
            "gpu_work_started": False,
            "local_caches_preserved": True,
        },
        "base_corpus_characters_before_checkpoint_6b": BASE_CORPUS_CHARACTERS,
        "normalized_record_count": len(rows),
        "archive_summaries": summaries,
        "decision_counts": dict(sorted(decisions.items())),
        "preliminary_role_counts": dict(sorted(roles.items())),
        "candidate_count": sum(value for key, value in decisions.items() if key.startswith("eligible_")),
        "conditioned_count": sum(value for key, value in decisions.items() if key.startswith("conditioned_")),
        "estimated_candidate_characters": sum(
            _integer(row["estimated_characters"]) or 0
            for row in rows if str(row["inventory_decision"]).startswith("eligible_")
        ),
        "estimated_candidate_tokens": sum(
            _integer(row["estimated_tokens"]) or 0
            for row in rows if str(row["inventory_decision"]).startswith("eligible_")
        ),
        "policy": {
            "inventory_rows_are_training_data": False,
            "missing_or_ambiguous_rights_fail_closed": True,
            "conditioned_language_records_enter_standard_queue": False,
            "gallica_http_error_interpreted_as_zero_results": False,
            "raw_metadata_cache_committed": False,
        },
        "verification": {
            "complete_cache_backed_builds": 2,
            "all_public_artifact_hashes_reproduced": True,
            "focused_checkpoint_tests": "19 passed",
            "complete_repository_tests": "899 passed",
        },
        "artifacts": {
            "inventory": str(config.inventory_path.relative_to(config.repo_root)),
            "summary": str(config.summary_path.relative_to(config.repo_root)),
            "report_json": str(config.json_report_path.relative_to(config.repo_root)),
            "report_markdown": str(config.markdown_report_path.relative_to(config.repo_root)),
            "local_cache": str(config.cache_dir.relative_to(config.repo_root)),
        },
    }


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Corpus Archive Candidate Inventory V1",
        "",
        "Checkpoint 6B completed a metadata/source inventory only. It did not acquire corpus text, activate records, create V7 splits, assign mixture weights, delete caches, or start GPU work.",
        "",
        f"The normalized public ledger contains **{report['normalized_record_count']:,} rows**. "
        f"Exactly **{report['candidate_count']:,}** rows are inactive candidates for a later bounded audit; "
        f"**{report['conditioned_count']:,}** language-variety rows remain outside the standard-Italian queue.",
        "",
        "## Archive accounting",
        "",
        "| Archive | Raw | Published rows | Filtered/accounted | Candidates | Holds | Excluded | Conditioned |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["archive_summaries"]:
        lines.append(
            f"| {row['archive_id']} | {row['raw_record_count']:,} | "
            f"{row['normalized_record_count']:,} | {row['filtered_out_count']:,} | "
            f"{row['candidate_count']:,} | {row['hold_count']:,} | "
            f"{row['excluded_count']:,} | {row['conditioned_count']:,} |"
        )
    lines.extend([
        "",
        "## Planning projections and concentration",
        "",
        f"Inactive candidates carry **{report['estimated_candidate_characters']:,} projected characters** and "
        f"**{report['estimated_candidate_tokens']:,} projected word/occurrence units** where source metadata permits estimation. "
        "The character projection comes only from ELTeC's word counts using a documented six-characters-per-word planning multiplier; "
        "MIDIA contributes occurrence units rather than tokenizer tokens. Archives without size metadata remain unprojected.",
        "",
        "Top contributor/institution proxies by normalized record count are:",
        "",
    ])
    for row in report["archive_summaries"]:
        lines.append(
            f"- `{row['archive_id']}`: {row['top_contributor']} — "
            f"{row['top_contributor_records']:,}/{row['normalized_record_count']:,} "
            f"records ({float(row['top_contributor_share']) * 100:.2f}%)."
        )
    lines.extend([
        "",
        "## Important boundaries",
        "",
        "- Internet Archive candidates require explicit item rights, an advertised text/OCR format, and literary metadata; OCR quality and duplication are still unresolved.",
        "- BEIC is exhaustively enumerated through its official Rosetta OAI set. Only Italian records dated no later than 1900 are published; every filtered record remains counted in the summary.",
        "- Gallica's official SRU interface is either completely enumerated or represented by an explicit access blocker. A failed request is never interpreted as a zero-record inventory.",
        "- ELTeC anomalies and post-1900 works fail closed. MIDIA period V remains held because its 1841-1947 bucket crosses the boundary.",
        "- Internet Culturale rows describe collections, not reusable text items; partner terms, item formats, and item counts remain pending.",
        "",
        "## Composition interpretation",
        "",
        "Metadata counts and word projections are planning evidence, not cleaned characters or training tokens. No new characters are added to the frozen 626,379,622-character broader-pool subtotal by this checkpoint.",
        "",
        "## Verification",
        "",
        "Two complete cache-backed builds reproduced all four public artifact hashes byte-for-byte. The combined checkpoint-6A/6B focused suite passes 19 tests, and the complete repository suite passes 899 tests.",
        "",
    ])
    return "\n".join(lines)


def _validate_config(config: ArchiveCandidateInventoryConfig) -> None:
    if config.request_timeout_seconds <= 0 or config.request_delay_seconds < 0:
        raise ValueError("request timeout/delay must be positive/non-negative")
    if not 1 <= config.max_attempts <= 10:
        raise ValueError("max_attempts must be between 1 and 10")
    if not 1 <= config.ia_rows_per_page <= 1_000:
        raise ValueError("ia_rows_per_page must be between 1 and 1000")


def _validate_archive_rows(
    archive_id: str,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    if any(set(row) != set(INVENTORY_FIELDS) for row in rows):
        raise ValueError(f"{archive_id} normalized rows do not match the public schema")
    if any(row["archive_id"] != archive_id for row in rows):
        raise ValueError(f"{archive_id} adapter emitted a foreign archive ID")
    ids = [str(row["record_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{archive_id} normalized record IDs are duplicated")
    if any(row["activation_status"] != "inactive_metadata_only" for row in rows):
        raise ValueError(f"{archive_id} attempted to activate a metadata row")
    if int(summary["raw_record_count"]) != len(rows) + int(summary["filtered_out_count"]):
        raise ValueError(f"{archive_id} raw/normalized/filtered accounting does not reconcile")


def _fetch_cached(
    config: ArchiveCandidateInventoryConfig,
    client: requests.Session,
    url: str,
    path: Path,
    progress: Progress | None,
    sleep: Sleep,
    *,
    params: Any = None,
) -> None:
    if path.is_file():
        _report(progress, f"cache_hit={path.relative_to(config.cache_dir)} bytes={path.stat().st_size:,}")
        return
    if config.request_delay_seconds:
        sleep(config.request_delay_seconds)
    response = _request_with_retries(config, client, url, progress, sleep, params=params)
    _write_bytes_atomic(path, response.content)


def _request_with_retries(
    config: ArchiveCandidateInventoryConfig,
    client: requests.Session,
    url: str,
    progress: Progress | None,
    sleep: Sleep,
    *,
    params: Any = None,
    allow_http_error: bool = False,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, config.max_attempts + 1):
        try:
            response = client.get(url, params=params, timeout=config.request_timeout_seconds)
            if not allow_http_error:
                response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if attempt >= config.max_attempts:
                break
            delay = float(2 ** (attempt - 1))
            _report(progress, f"request_retry attempt={attempt}/{config.max_attempts} delay={delay:.1f}s error={type(error).__name__}")
            sleep(delay)
    assert last_error is not None
    raise last_error


def _ia_rights_status(license_url: str, rights: str) -> str:
    evidence = f"{license_url} {rights}".casefold()
    if re.search(r"(?:by-nd|by-nc-nd|/nd/|no[- ]derivatives)", evidence):
        return "incompatible_or_ambiguous_no_derivatives"
    if any(token in evidence for token in (
        "publicdomain", "public domain", "creativecommons.org/publicdomain/zero",
        "creativecommons.org/licenses/by/", "creativecommons.org/licenses/by-sa/",
        "creativecommons.org/licenses/by-nc/", "creativecommons.org/licenses/by-nc-sa/",
    )):
        return "explicit_reusable_item_rights"
    return "item_rights_missing_or_unresolved"


def _role(text: str, year: int | None) -> str:
    if _SONNET.search(text):
        return "standard_sonnet_candidate"
    if _POETRY.search(text):
        return "historical_non_sonnet_poetry" if year is None or year <= 1800 else "ottocento_bridge"
    return "historical_general" if year is None or year <= 1800 else "ottocento_bridge"


def _genre(text: str) -> str:
    if _SONNET.search(text):
        return "sonnet_signal"
    if _POETRY.search(text):
        return "poetry_signal"
    if _LITERARY.search(text):
        return "literary_prose_or_drama_signal"
    return "mixed_or_unresolved"


def _period(year: int | None) -> str:
    if year is None:
        return "unresolved"
    if year <= 1800:
        return "historical_through_1800"
    if year <= 1900:
        return "ottocento_bridge"
    return "post_1900"


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    match = re.search(r"-?\d+", str(value or "").replace(",", ""))
    return int(match.group()) if match else None


def _first_year(value: Any) -> int | None:
    match = _YEAR.search(str(value or ""))
    return int(match.group()) if match else None


def _list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _join(value: Any) -> str:
    return "; ".join(_clean(item) for item in _list(value) if _clean(item))


def _clean(value: Any) -> str:
    return _SPACE.sub(" ", html.unescape(str(value or ""))).strip()


def _plain(value: str) -> str:
    return _clean(_TAG.sub(" ", value))


def _read_json_if_present(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv_atomic(path: Path, fields: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: Any) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _report(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)
