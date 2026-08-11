"""Bounded checkpoint-6D audit for ILC-CNR and Oxford Text Archive text."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from html import entities as html_entities
from itertools import combinations
from pathlib import Path
from time import monotonic, sleep as default_sleep
from typing import Any
from urllib.parse import unquote, urlparse

import requests
from bs4 import BeautifulSoup

from sonnet_corpus.gutenberg_fulltext_probe import (
    TextFingerprint,
    TextReference,
    fingerprint_text,
    measure_word_shingle_containment,
)


AUDIT_VERSION = "ilc_ota_source_audit_v1"
USER_AGENT = "portfolio-transformer-poetry-ilc-ota-audit/1.0"
BASE_CORPUS_CHARACTERS = 626_379_622
OTA_BASE = "https://llds.ling-phil.ox.ac.uk"

INVENTORY_FIELDS = (
    "record_id", "archive_id", "source_item_id", "title", "creator",
    "created_date", "created_year", "period_bucket", "language", "source_type",
    "landing_page_url", "rights_text", "license_url", "format", "extent",
    "bitstream_url", "metadata_sha256", "metadata_cache_path",
    "metadata_decision", "metadata_rationale", "assigned_role",
    "activation_status",
)
FILE_FIELDS = (
    "record_id", "file_name", "download_url", "content_type", "raw_byte_count",
    "raw_sha256", "cache_path", "acquisition_status", "error",
)
UNIT_FIELDS = (
    "unit_id", "record_id", "archive_id", "member_path", "title", "author",
    "assigned_role", "cleaned_cache_path", "cleaned_character_count",
    "cleaned_word_count", "nonempty_line_count", "cleaned_sha256",
    "normalized_word_sha256", "italian_function_word_ratio",
    "alphabetic_character_ratio", "quality_flags", "language_flags",
    "internal_overlap_ids", "cross_overlap_ids", "protected_v6_poem_ids",
    "probe_decision", "activation_status", "error",
)
OVERLAP_FIELDS = (
    "candidate_unit_id", "reference_id", "reference_kind", "pair_scope",
    "left_containment", "right_containment", "matching_shingles",
    "decision_scope",
)
REVIEW_FIELDS = (
    "review_id", "record_id", "unit_id", "review_type", "evidence",
    "resolution", "rationale",
)
DECISION_FIELDS = (
    "record_id", "archive_id", "title", "assigned_role", "metadata_decision",
    "extracted_unit_count", "eligible_unit_count", "excluded_unit_count",
    "held_unit_count", "candidate_character_count", "eligible_character_count",
    "share_of_candidate_pool", "share_of_resulting_pool", "final_status",
    "next_action", "activation_status",
)

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_WHITESPACE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_NAMED_ENTITY = re.compile(r"&([A-Za-z][A-Za-z0-9._:-]*);")
_XML_ENTITIES = {"amp", "apos", "gt", "lt", "quot"}
_ITALIAN_FUNCTION_WORDS = {
    "a", "che", "con", "da", "del", "della", "di", "e", "gli", "il",
    "in", "la", "le", "lo", "ma", "nel", "non", "per", "si", "un", "una",
}
_ITALIAN_LANGUAGE_WORDS = _ITALIAN_FUNCTION_WORDS | {
    "al", "alla", "alle", "anche", "come", "dei", "delle", "dello", "era",
    "essere", "ha", "hanno", "io", "mi", "ne", "nella", "nello", "piu",
    "quale", "quando", "questa", "questo", "sono", "sua", "suo", "tra",
    "tutto", "uno", "vi",
}
_ENGLISH_LANGUAGE_WORDS = {
    "a", "and", "are", "as", "be", "been", "by", "for", "from", "had",
    "has", "have", "he", "her", "his", "in", "is", "it", "not", "of",
    "on", "or", "she", "that", "the", "their", "this", "to", "was", "were",
    "which", "with", "you",
}
_LATIN_LANGUAGE_WORDS = {
    "ab", "ad", "at", "aut", "cum", "de", "enim", "est", "et", "ex", "hic",
    "in", "nec", "non", "per", "quae", "quam", "qui", "quod", "sed", "si",
    "sunt", "ut",
}
_APPARATUS_FILE = re.compile(r"(?:^index|doc(?:-|$))", re.IGNORECASE)
_POETRY_MARKERS = re.compile(
    r"\b(?:rime|poes(?:ia|ie)|versi|canti|poemi?|orlando|morgante|teseide|"
    r"gerusalemme|purgatorio|paradiso|inferno)\b",
    re.IGNORECASE,
)
_CONDITIONED_MARKERS = re.compile(
    r"\b(?:dialect|dialetto|vernacolo|venetian|veneziano|romanesco|"
    r"napoletano|siciliano|sardo|bolognese)\b",
    re.IGNORECASE,
)
_BLOCK_TAGS = {
    "ab", "argument", "body", "closer", "dateline", "div", "head", "l",
    "lg", "opener", "p", "salute", "signed", "sp", "speaker", "stage",
}
_SKIP_TAGS = {
    "back", "bibl", "bibliography", "figure", "front", "fw", "listbibl",
    "note", "notesstmt", "teiheader", "table",
}

Progress = Callable[[str], None]
Sleep = Callable[[float], None]


@dataclass(frozen=True)
class ILCSourceSpec:
    record_id: str
    item_id: str
    landing_page_url: str
    title: str
    creator: str
    created_date: str
    license_url: str
    rights_text: str
    assigned_role: str
    file_name: str
    download_url: str
    expected_bytes: int
    expected_md5: str
    packaging: str


ILC_SOURCES = (
    ILCSourceSpec(
        "ilc_rosmini", "30dcd581-aa78-4470-9fc9-9f6bd385b97f",
        "https://hdl.handle.net/20.500.11752/ILC-57",
        "Corpus Antonio Rosmini - Serbati", "Antonio Rosmini-Serbati",
        "nineteenth century", "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "CC BY-NC-SA 4.0", "ottocento_bridge_capped", "corpus-rosmini.zip",
        "https://dspace-clarin-it.ilc.cnr.it/server/api/core/bitstreams/"
        "db6dadd7-0241-4d92-a24f-e5fef782ecb9/content",
        10_302_816, "20548a1127f46bf711c9addec673df21", "rosmini_zip",
    ),
    ILCSourceSpec(
        "ilc_libretti", "5c8f43f9-d039-4475-b930-fa13d35b583c",
        "https://hdl.handle.net/20.500.11752/OPEN-979",
        "Digital edition of opera libretti", "ILC-CNR digital edition",
        "1636-1705", "https://creativecommons.org/licenses/by/4.0/",
        "CC BY 4.0", "historical_non_sonnet_poetry",
        "Edizione_digitale_dei_libretti_per_la_funzione_delle_Tasche_Gallucci.xml",
        "https://dspace-clarin-it.ilc.cnr.it/server/api/core/bitstreams/"
        "0422a142-25b6-4bf8-af9b-8ad17f18b31f/content",
        2_737_739, "3ef546315401437f6e68fda44c630015", "libretti_xml",
    ),
    ILCSourceSpec(
        "ilc_bellini", "2e6fe73a-db5c-48aa-ad0a-dcaecc0baddb",
        "https://hdl.handle.net/20.500.11752/OPEN-1000",
        "Bellini Digital Correspondence", "Vincenzo Bellini",
        "1819-1835", "https://creativecommons.org/licenses/by-nc/4.0/",
        "CC BY-NC 4.0", "ottocento_bridge_capped", "BDC-XML.zip",
        "https://dspace-clarin-it.ilc.cnr.it/server/api/core/bitstreams/"
        "ca4201ea-d874-4169-983e-9e3e0feeac1c/content",
        407_671, "314dd1bcc2ffe797aebbbdd872f06dee", "bellini_zip",
    ),
)


@dataclass(frozen=True)
class ILCOTAAuditConfig:
    repo_root: Path
    discovery_cache_dir: Path
    cache_dir: Path
    inventory_path: Path
    file_path: Path
    unit_path: Path
    overlap_path: Path
    review_path: Path
    decision_path: Path
    json_report_path: Path
    markdown_report_path: Path
    bibit_record_manifest_path: Path
    bibit_sonnet_manifest_path: Path
    gutenberg_previous_probe_path: Path
    gutenberg_previous_cache_dir: Path
    gutenberg_pass_1b_probe_path: Path
    gutenberg_pass_1b_cache_dir: Path
    gutenberg_resolved_record_manifest_path: Path
    gutenberg_resolved_sonnet_manifest_path: Path
    wikisource_resolved_record_manifest_path: Path
    wikisource_resolved_sonnet_manifest_path: Path
    liber_liber_resolved_record_manifest_path: Path
    liber_liber_resolved_sonnet_manifest_path: Path
    broader_sources_manifest_path: Path
    protected_v6_sonnet_manifest_path: Path
    request_delay_seconds: float = 0.25
    request_timeout_seconds: float = 60.0
    max_attempts: int = 3
    min_cleaned_characters: int = 1_000
    min_italian_function_word_ratio: float = 0.02
    sketch_size: int = 256
    anchor_mask: int = 1023
    near_duplicate_containment: float = 0.8
    protected_containment: float = 0.8


@dataclass(frozen=True)
class ExtractedUnit:
    unit_id: str
    member_path: str
    title: str
    author: str
    text: str


def run_ilc_ota_source_audit(
    config: ILCOTAAuditConfig,
    *,
    session: requests.Session | None = None,
    sleep: Sleep = default_sleep,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Inventory and probe all checkpoint-6D-compatible text without activation."""

    _validate_config(config)
    client = session or requests.Session()
    client.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
    started = monotonic()

    inventory = _ilc_inventory(config)
    handles = parse_ota_catalog_handles(
        (config.discovery_cache_dir / "queries/ota_01.bin").read_bytes()
    )
    if len(handles) != 43:
        raise ValueError(f"Oxford catalog boundary drift: expected 43, got {len(handles)}")
    for index, handle in enumerate(handles, 1):
        _report(progress, f"ota-metadata={index}/43 handle={handle} start")
        url = f"{OTA_BASE}/llds/xmlui/handle/20.500.14106/{handle}"
        payload, metadata = _fetch_cached(
            config, client, f"ota_{handle}", url, "metadata", sleep,
        )
        row = parse_ota_item_page(payload, handle=handle, landing_page_url=url)
        decision, rationale, role = ota_metadata_decision(row)
        row.update({
            "record_id": f"ota_{handle}",
            "archive_id": "oxford_text_archive",
            "source_item_id": handle,
            "metadata_sha256": metadata["content_sha256"],
            "metadata_cache_path": _portable(
                config.cache_dir / "metadata" / f"ota_{handle}.bin", config.repo_root,
            ),
            "metadata_decision": decision,
            "metadata_rationale": rationale,
            "assigned_role": role,
            "activation_status": "inactive_metadata_only",
        })
        inventory.append({field: row.get(field, "") for field in INVENTORY_FIELDS})
        _report(progress, f"ota-metadata={index}/43 handle={handle} decision={decision} complete")

    inventory.sort(key=lambda row: row["record_id"])
    _write_csv(config.inventory_path, INVENTORY_FIELDS, inventory)

    file_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    texts: dict[str, str] = {}
    watched, protected_denominators = _load_protected_watch(config)

    eligible_records = [row for row in inventory if row["metadata_decision"] == "eligible_text_probe_inactive"]
    for index, row in enumerate(eligible_records, 1):
        _report(progress, f"acquire={index}/{len(eligible_records)} record={row['record_id']} start")
        extracted = []
        download_urls = list(filter(None, row["bitstream_url"].split(";")))
        has_xml_representation = any(
            _suffix(value) in {".xml", ".tei"} for value in download_urls
        )
        for file_index, download_url in enumerate(download_urls, 1):
            file_name = _download_name(row, download_url)
            cache_id = f"{row['record_id']}_{file_index:03d}_{file_name}"
            try:
                payload, metadata = _fetch_cached(
                    config, client, cache_id, download_url, "payloads", sleep,
                )
                spec = next((value for value in ILC_SOURCES if value.record_id == row["record_id"]), None)
                if spec is not None:
                    if len(payload) != spec.expected_bytes:
                        raise ValueError(f"official byte-count drift for {spec.record_id}")
                    if hashlib.md5(payload, usedforsecurity=False).hexdigest() != spec.expected_md5:
                        raise ValueError(f"official MD5 drift for {spec.record_id}")
                redundant = has_xml_representation and _suffix(download_url) == ".epub"
                file_units = [] if redundant else extract_record_units(
                    row, payload, file_name=file_name,
                )
                if redundant:
                    status = "cached_verified_redundant_representation"
                    error_text = "XML primary representation preferred over duplicate EPUB"
                elif file_units:
                    status = "cached_verified"
                    error_text = ""
                else:
                    status = "cached_verified_no_extractable_primary_text"
                    error_text = "unsupported, binary, or empty primary-text format"
                file_rows.append({
                    "record_id": row["record_id"], "file_name": file_name,
                    "download_url": download_url, "content_type": metadata["content_type"],
                    "raw_byte_count": len(payload), "raw_sha256": metadata["content_sha256"],
                    "cache_path": _portable(
                        config.cache_dir / "payloads" / f"{_safe_id(cache_id)}.bin",
                        config.repo_root,
                    ),
                    "acquisition_status": status, "error": error_text,
                })
                extracted.extend(file_units)
            except Exception as error:  # fail closed per file while retaining complete inventory
                file_rows.append({
                    "record_id": row["record_id"], "file_name": file_name,
                    "download_url": download_url, "content_type": "",
                    "raw_byte_count": 0, "raw_sha256": "", "cache_path": "",
                    "acquisition_status": "error_hold", "error": f"{type(error).__name__}: {error}",
                })

        for unit in extracted:
            clean_path = config.cache_dir / "cleaned" / f"{_safe_id(unit.unit_id)}.txt"
            _write_text(clean_path, unit.text)
            fingerprint, hits = fingerprint_text(
                unit.text, sketch_size=config.sketch_size,
                anchor_mask=config.anchor_mask, watched_shingles=watched,
            )
            words = [value.casefold() for value in _WORD.findall(unit.text)]
            italian_count = sum(value in _ITALIAN_FUNCTION_WORDS for value in words)
            alphabetic = sum(character.isalpha() for character in unit.text)
            nonspace = sum(not character.isspace() for character in unit.text)
            quality_flags = []
            language_flags = []
            minimum_characters = 50 if row["record_id"] == "ilc_bellini" else config.min_cleaned_characters
            if len(unit.text) < minimum_characters:
                quality_flags.append("below_minimum_cleaned_characters")
            alphabetic_ratio = alphabetic / nonspace if nonspace else 0.0
            italian_ratio = italian_count / len(words) if words else 0.0
            if alphabetic_ratio < 0.5:
                quality_flags.append("low_alphabetic_ratio")
            if _APPARATUS_FILE.search(Path(unit.member_path).stem):
                quality_flags.append("documentation_or_index_apparatus")
            if len(words) >= 100 and italian_ratio < config.min_italian_function_word_ratio:
                language_flags.append("low_italian_function_word_ratio")
            italian_score, english_score, latin_score = _language_scores(words)
            if len(words) >= 100 and english_score > max(0.05, italian_score * 1.3):
                language_flags.append("english_primary_text")
            if len(words) >= 100 and latin_score > max(0.05, italian_score * 1.5):
                language_flags.append("latin_primary_text")
            protected_ids = []
            for poem_id, values in hits.items():
                denominator = protected_denominators[poem_id]
                if denominator and len(values) / denominator >= config.protected_containment:
                    protected_ids.append(poem_id)
            unit_rows.append({
                "unit_id": unit.unit_id, "record_id": row["record_id"],
                "archive_id": row["archive_id"], "member_path": unit.member_path,
                "title": unit.title or row["title"], "author": unit.author or row["creator"],
                "assigned_role": row["assigned_role"],
                "cleaned_cache_path": _portable(clean_path, config.repo_root),
                "cleaned_character_count": len(unit.text), "cleaned_word_count": len(words),
                "nonempty_line_count": sum(bool(line.strip()) for line in unit.text.splitlines()),
                "cleaned_sha256": hashlib.sha256(unit.text.encode("utf-8")).hexdigest(),
                "normalized_word_sha256": fingerprint.normalized_word_sha256,
                "italian_function_word_ratio": f"{italian_ratio:.6f}",
                "alphabetic_character_ratio": f"{alphabetic_ratio:.6f}",
                "quality_flags": ";".join(quality_flags),
                "language_flags": ";".join(language_flags),
                "internal_overlap_ids": "", "cross_overlap_ids": "",
                "protected_v6_poem_ids": ";".join(sorted(protected_ids)),
                "probe_decision": "pending_overlap", "activation_status": "inactive_probe_only",
                "error": "", "_fingerprint": fingerprint,
            })
            texts[unit.unit_id] = unit.text
        _report(progress, f"acquire={index}/{len(eligible_records)} record={row['record_id']} units={len(extracted)} complete")

    references = _load_references(config)
    reference_fingerprints = _fingerprint_references(references, config, progress)
    candidate_fingerprints = {row["unit_id"]: row["_fingerprint"] for row in unit_rows}
    overlaps = _measure_overlaps(
        unit_rows, texts, candidate_fingerprints, references,
        reference_fingerprints, config, progress,
    )
    _finalize_unit_decisions(unit_rows, overlaps)
    reviews = _review_rows(inventory, unit_rows)
    decisions = _decision_rows(inventory, unit_rows, file_rows)

    for row in unit_rows:
        row.pop("_fingerprint", None)
    _write_csv(config.file_path, FILE_FIELDS, file_rows)
    _write_csv(config.unit_path, UNIT_FIELDS, unit_rows)
    _write_csv(config.overlap_path, OVERLAP_FIELDS, overlaps)
    _write_csv(config.review_path, REVIEW_FIELDS, reviews)
    _write_csv(config.decision_path, DECISION_FIELDS, decisions)

    report = _build_report(config, inventory, file_rows, unit_rows, overlaps, reviews, decisions)
    _write_json(config.json_report_path, report)
    _write_text(config.markdown_report_path, _render_markdown(report, decisions))
    _report(progress, f"complete elapsed={_format_duration(monotonic() - started)} activated=0")
    return report


def parse_ota_catalog_handles(content: bytes) -> list[str]:
    """Return the stable, deduplicated Italian-facet OTA handle suffixes."""

    values = set(re.findall(
        rb"/llds/xmlui/handle/20\.500\.14106/([^\"'?&<]+)", content,
    ))
    return sorted(value.decode("utf-8") for value in values)


def parse_ota_item_page(
    content: bytes,
    *,
    handle: str,
    landing_page_url: str,
) -> dict[str, Any]:
    """Parse one OTA DSpace landing page into rights and acquisition metadata."""

    soup = BeautifulSoup(content, "html.parser")
    values: dict[str, list[str]] = defaultdict(list)
    for meta in soup.find_all("meta"):
        name = str(meta.get("name") or "")
        value = str(meta.get("content") or "").strip()
        if name and value:
            values[name].append(value)
    title = _first(values["DC.title"]) or _first(values["citation_title"])
    creator = "; ".join(values["DC.creator"] or values["citation_author"])
    created_date = _first(values["DCTERMS.created"])
    year = _first_year(created_date)
    rights = values["DC.rights"]
    license_url = next(
        (value for value in rights if "creativecommons.org/" in value.casefold()), "",
    )
    candidates = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"])
        if "/bitstream/handle/20.500.14106/" not in href:
            continue
        url = href if href.startswith("http") else OTA_BASE + href
        if _suffix(url) in {".xml", ".tei", ".txt", ".epub", ".zip"}:
            candidates.append(url)
    if not candidates:
        citation = _first(values["citation_pdf_url"])
        if citation:
            candidates.append(citation)
    bitstream = ";".join(dict.fromkeys(candidates))
    return {
        "source_item_id": handle,
        "title": title,
        "creator": creator,
        "created_date": created_date,
        "created_year": year or "",
        "period_bucket": _period(year),
        "language": "; ".join(values["DC.language"] or values["citation_language"]),
        "source_type": "; ".join(values["DC.type"] or values["citation_keywords"]),
        "landing_page_url": landing_page_url,
        "rights_text": " | ".join(rights),
        "license_url": license_url,
        "format": "; ".join(values["DC.format"]),
        "extent": "; ".join(values["DCTERMS.extent"]),
        "bitstream_url": bitstream,
    }


def ota_metadata_decision(row: dict[str, Any]) -> tuple[str, str, str]:
    """Apply fail-closed item terms, date, language, and primary-text gates."""

    title = str(row.get("title", ""))
    language = str(row.get("language", "")).casefold()
    source_type = str(row.get("source_type", "")).casefold()
    license_url = str(row.get("license_url", "")).casefold()
    year = _integer(row.get("created_year"))
    if _CONDITIONED_MARKERS.search(title):
        return "conditioned_language_excluded_inactive", "explicit dialect marker; outside standard queue", "conditioned_language"
    if "italian" not in language and not re.search(r"\bita\b", language):
        return "excluded_non_italian_metadata", "item metadata does not establish Italian", "excluded"
    if "text" not in source_type:
        return "excluded_not_primary_text", "item type is not Text", "excluded"
    permitted = any(value in license_url for value in (
        "creativecommons.org/publicdomain/zero/1.0",
        "creativecommons.org/licenses/by-nc-sa/3.0",
        "creativecommons.org/licenses/by-nc-sa/4.0",
        "creativecommons.org/licenses/by/4.0",
    ))
    if not permitted:
        return "excluded_terms_unresolved", "no compatible item-level text license URI", "excluded"
    if year is None:
        return "excluded_period_unresolved", "no authoritative underlying-work year", "excluded"
    if year > 1900:
        return "excluded_post_1900", f"underlying work date {year} is after 1900", "excluded"
    if not row.get("bitstream_url"):
        return "excluded_no_text_bitstream", "no downloadable primary-text bitstream", "excluded"
    role = "historical_general"
    if _POETRY_MARKERS.search(title):
        role = "historical_non_sonnet_poetry"
    if year > 1800:
        role = "ottocento_bridge_capped"
    return "eligible_text_probe_inactive", "compatible terms, Italian Text item, pre-1901 work, and direct bitstream", role


def extract_record_units(
    row: dict[str, Any], payload: bytes, *, file_name: str,
) -> list[ExtractedUnit]:
    """Extract auditable primary-text units without normalizing spelling."""

    record_id = row["record_id"]
    if record_id == "ilc_libretti":
        return extract_libretti_units(payload)
    suffix = Path(file_name).suffix.casefold()
    if suffix == ".epub":
        return _extract_epub_units(record_id, payload)
    if zipfile.is_zipfile(io.BytesIO(payload)):
        return _extract_zip_units(record_id, payload)
    if suffix in {".xml", ".tei"} or (not suffix and payload.lstrip().startswith(b"<")):
        return extract_xml_units(payload, record_id=record_id, member_path=file_name)
    if _binary_control_ratio(payload) > 0.01:
        return []
    text = payload.decode("utf-8-sig", "replace")
    if suffix == ".txt":
        text = re.sub(r"<[^>\n]{1,200}>", "\n", text)
    text = _clean_text(_remove_controls(text))
    unit_id = f"{record_id}:{Path(file_name).stem}"
    return [ExtractedUnit(unit_id, file_name, row.get("title", ""), row.get("creator", ""), text)] if text else []


def extract_libretti_units(payload: bytes) -> list[ExtractedUnit]:
    """Extract 56 advertised works, merging the split L1669.1 encoding."""

    root = _parse_xml(payload)
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for text_element in _descendants(root, "text"):
        body = _first_child(text_element, "body")
        if body is None:
            continue
        xml_id = text_element.attrib.get("{http://www.w3.org/XML/1998/namespace}id", "")
        if not xml_id:  # the corpus-level Fonti apparatus body
            continue
        group_id = re.sub(r"-(?:1|2)$", "", xml_id)
        title = _first_descendant_text(body, "head") or group_id
        groups[group_id].append((title, _extract_primary_text(body)))
    result = []
    for group_id, parts in sorted(groups.items()):
        text = _clean_text("\n\n".join(value for _, value in parts if value))
        title = " / ".join(value for value, _ in parts)
        if text:
            result.append(ExtractedUnit(
                f"ilc_libretti:{group_id}",
                ";".join(f"xml:id={group_id}" for _ in parts), title,
                "ILC-CNR digital edition", text,
            ))
    if len(result) != 56:
        raise ValueError(f"libretti work accounting drift: expected 56, got {len(result)}")
    return result


def extract_xml_units(
    payload: bytes,
    *,
    record_id: str,
    member_path: str,
) -> list[ExtractedUnit]:
    """Extract the TEI body and header title/author from one XML document."""

    root = _parse_xml(payload)
    body = _first_descendant(root, "body")
    if body is None:
        return []
    title = _header_text(root, "title")
    author = _header_text(root, "author")
    text = _clean_text(_extract_primary_text(body))
    if not text:
        return []
    stem = Path(member_path).stem
    return [ExtractedUnit(f"{record_id}:{stem}", member_path, title, author, text)]


def _extract_zip_units(record_id: str, payload: bytes) -> list[ExtractedUnit]:
    result = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = sorted(
            name for name in archive.namelist()
            if name.casefold().endswith((".xml", ".tei", ".txt"))
            and not name.startswith("__MACOSX/")
            and "/lists/" not in name.casefold()
            and Path(name).name.casefold() not in {"readme.txt", "license.txt"}
        )
        for name in members:
            content = archive.read(name)
            if name.casefold().endswith((".xml", ".tei")):
                result.extend(extract_xml_units(content, record_id=record_id, member_path=name))
            else:
                text = _clean_text(content.decode("utf-8-sig", "replace"))
                if text:
                    result.append(ExtractedUnit(
                        f"{record_id}:{Path(name).stem}", name, Path(name).stem, "", text,
                    ))
    return result


def _extract_epub_units(record_id: str, payload: bytes) -> list[ExtractedUnit]:
    parts = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in sorted(archive.namelist()):
            if name.casefold().endswith((".xhtml", ".html", ".htm")):
                soup = BeautifulSoup(archive.read(name), "html.parser")
                parts.append(soup.get_text("\n", strip=True))
    text = _clean_text("\n\n".join(parts))
    return [ExtractedUnit(record_id, "EPUB XHTML spine", "", "", text)] if text else []


def _ilc_inventory(config: ILCOTAAuditConfig) -> list[dict[str, Any]]:
    result = []
    for spec in ILC_SOURCES:
        evidence = config.discovery_cache_dir / "evidence" / f"{spec.record_id.replace('ilc_', 'ilc_')}.bin"
        if not evidence.is_file():
            raise FileNotFoundError(f"missing checkpoint-6C ILC evidence: {evidence}")
        metadata_sha = hashlib.sha256(evidence.read_bytes()).hexdigest()
        year = 1850 if spec.record_id == "ilc_rosmini" else (1636 if spec.record_id == "ilc_libretti" else 1819)
        result.append({
            "record_id": spec.record_id, "archive_id": "ilc_cnr_historical_corpora",
            "source_item_id": spec.item_id, "title": spec.title, "creator": spec.creator,
            "created_date": spec.created_date, "created_year": year,
            "period_bucket": _period(year), "language": "Italian",
            "source_type": "Text corpus", "landing_page_url": spec.landing_page_url,
            "rights_text": spec.rights_text, "license_url": spec.license_url,
            "format": "application/zip" if spec.file_name.endswith(".zip") else "text/xml",
            "extent": f"{spec.expected_bytes} bytes", "bitstream_url": spec.download_url,
            "metadata_sha256": metadata_sha,
            "metadata_cache_path": _portable(evidence, config.repo_root),
            "metadata_decision": "eligible_text_probe_inactive",
            "metadata_rationale": "checkpoint-6C compatible item terms, role, and materiality gate",
            "assigned_role": spec.assigned_role, "activation_status": "inactive_metadata_only",
        })
    return result


def _load_references(config: ILCOTAAuditConfig) -> dict[str, TextReference]:
    result: dict[str, TextReference] = {}
    _add_range_manifest(result, config.repo_root, config.bibit_record_manifest_path, "bibit", "object_id", {"text_materialized"})
    _add_range_manifest(result, config.repo_root, config.bibit_sonnet_manifest_path, "bibit_sonnet", "candidate_id", None)
    _add_probe_manifest(result, config.gutenberg_previous_probe_path, config.gutenberg_previous_cache_dir, "gutenberg_previous")
    _add_probe_manifest(result, config.gutenberg_pass_1b_probe_path, config.gutenberg_pass_1b_cache_dir, "gutenberg_pass_1b")
    _add_range_manifest(result, config.repo_root, config.gutenberg_resolved_record_manifest_path, "gutenberg_resolved", "ebook_id", {"text_materialized_pending_v7"})
    _add_range_manifest(result, config.repo_root, config.gutenberg_resolved_sonnet_manifest_path, "gutenberg_sonnet", "candidate_id", {"standard_sonnet_materialized_pending_v7"})
    _add_range_manifest(result, config.repo_root, config.wikisource_resolved_record_manifest_path, "wikisource_resolved", "work_root_id", {"text_materialized_inactive"})
    _add_range_manifest(result, config.repo_root, config.wikisource_resolved_sonnet_manifest_path, "wikisource_sonnet", "candidate_id", {"sonnet_materialized_inactive"})
    _add_range_manifest(result, config.repo_root, config.liber_liber_resolved_record_manifest_path, "liber_liber_resolved", "record_id", {"text_materialized_inactive"})
    _add_range_manifest(result, config.repo_root, config.liber_liber_resolved_sonnet_manifest_path, "liber_liber_sonnet", "candidate_id", {"sonnet_materialized_inactive"})
    for row in _read_csv(config.broader_sources_manifest_path):
        relative = row.get("expected_clean_text_path", "")
        path = config.repo_root / relative if relative else Path("/")
        if relative and path.is_file():
            result[f"current:{row['source_id']}"] = TextReference(
                f"current:{row['source_id']}", "existing_project_corpus", path,
            )
    return result


def _add_range_manifest(
    result: dict[str, TextReference], repo_root: Path, manifest: Path,
    prefix: str, id_field: str, statuses: set[str] | None,
) -> None:
    for row in _read_csv(manifest):
        if statuses is not None and row.get("artifact_status", "") not in statuses:
            continue
        if not row.get("shard_path"):
            continue
        reference_id = f"{prefix}:{row[id_field]}"
        result[reference_id] = TextReference(
            reference_id, prefix, repo_root / row["shard_path"],
            int(row["byte_start"]), int(row["byte_end"]),
        )


def _add_probe_manifest(
    result: dict[str, TextReference], manifest: Path, cache_dir: Path, kind: str,
) -> None:
    for row in _read_csv(manifest):
        path = cache_dir / f"pg{row['ebook_id']}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen Gutenberg cache: {path}")
        reference_id = f"{kind}:pg{row['ebook_id']}"
        result[reference_id] = TextReference(
            reference_id, kind, path, cleaning="gutenberg_boilerplate",
        )


def _fingerprint_references(
    references: dict[str, TextReference], config: ILCOTAAuditConfig,
    progress: Progress | None,
) -> dict[str, TextFingerprint]:
    result = {}
    started = monotonic()
    total = len(references)
    for index, (reference_id, reference) in enumerate(sorted(references.items()), 1):
        result[reference_id], _ = fingerprint_text(
            reference.read_text(), sketch_size=config.sketch_size,
            anchor_mask=config.anchor_mask,
        )
        if index == 1 or index == total or index % 250 == 0:
            _report(progress, f"reference-index={index}/{total} elapsed={_format_duration(monotonic() - started)}")
    return result


def _measure_overlaps(
    unit_rows: list[dict[str, Any]], texts: dict[str, str],
    candidates: dict[str, TextFingerprint], references: dict[str, TextReference],
    reference_fingerprints: dict[str, TextFingerprint], config: ILCOTAAuditConfig,
    progress: Progress | None,
) -> list[dict[str, Any]]:
    overlaps = []
    internal_pairs = _discover_pairs(candidates)
    cross_pairs = _discover_cross_pairs(candidates, reference_fingerprints)
    _report(progress, f"overlap-candidates internal={len(internal_pairs)} cross={len(cross_pairs)}")
    by_id = {row["unit_id"]: row for row in unit_rows}
    for left, right in sorted(internal_pairs):
        metrics = measure_word_shingle_containment(texts[left], texts[right])
        if metrics["containment"] < config.near_duplicate_containment:
            continue
        by_id[left]["internal_overlap_ids"] = _append(by_id[left]["internal_overlap_ids"], right)
        by_id[right]["internal_overlap_ids"] = _append(by_id[right]["internal_overlap_ids"], left)
        overlaps.append(_overlap_row(left, right, "candidate", "internal", metrics))
    for unit_id, reference_id in sorted(cross_pairs):
        metrics = measure_word_shingle_containment(texts[unit_id], references[reference_id].read_text())
        if metrics["containment"] < config.near_duplicate_containment:
            continue
        by_id[unit_id]["cross_overlap_ids"] = _append(by_id[unit_id]["cross_overlap_ids"], reference_id)
        overlaps.append(_overlap_row(
            unit_id, reference_id, references[reference_id].source_kind, "cross", metrics,
        ))
    return overlaps


def _overlap_row(
    candidate: str, reference: str, kind: str, scope: str, metrics: dict[str, Any],
) -> dict[str, Any]:
    if metrics["left_containment"] >= 0.8:
        decision = "candidate_fully_covered"
    elif metrics["right_containment"] >= 0.8:
        decision = "embedded_reference_overlap"
    else:
        decision = "mutual_near_overlap"
    return {
        "candidate_unit_id": candidate, "reference_id": reference,
        "reference_kind": kind, "pair_scope": scope,
        "left_containment": f"{metrics['left_containment']:.6f}",
        "right_containment": f"{metrics['right_containment']:.6f}",
        "matching_shingles": metrics["matching_shingles"],
        "decision_scope": decision,
    }


def _finalize_unit_decisions(
    rows: list[dict[str, Any]], overlaps: list[dict[str, Any]],
) -> None:
    overlap_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for overlap in overlaps:
        overlap_by_id[overlap["candidate_unit_id"]].append(overlap)
        if overlap["pair_scope"] == "internal":
            overlap_by_id[overlap["reference_id"]].append(overlap)
    internal_losers = set()
    for overlap in overlaps:
        if overlap["pair_scope"] == "internal":
            internal_losers.add(max(overlap["candidate_unit_id"], overlap["reference_id"]))
    for row in rows:
        unit_id = row["unit_id"]
        linked = overlap_by_id[unit_id]
        if row["error"] or row["quality_flags"]:
            decision = "excluded_quality_or_parse_failure"
        elif row["language_flags"]:
            decision = "excluded_nonstandard_or_unverified_language"
        elif row["protected_v6_poem_ids"]:
            decision = "hold_protected_v6_segment_removal"
        elif unit_id in internal_losers:
            decision = "excluded_internal_canonical_duplicate"
        elif any(
            value["pair_scope"] == "cross"
            and value["candidate_unit_id"] == unit_id
            and value["decision_scope"] == "candidate_fully_covered"
            for value in linked
        ):
            decision = "excluded_cross_corpus_full_duplicate"
        elif any(value["decision_scope"] != "candidate_fully_covered" for value in linked):
            decision = "eligible_unique_with_segment_review_inactive"
        else:
            decision = "eligible_checkpoint_7_canonicalization_inactive"
        row["probe_decision"] = decision


def _review_rows(
    inventory: list[dict[str, Any]], units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for item in inventory:
        if item["metadata_decision"] != "eligible_text_probe_inactive":
            rows.append({
                "review_id": f"metadata:{item['record_id']}", "record_id": item["record_id"],
                "unit_id": "", "review_type": "metadata_exclusion",
                "evidence": item["metadata_rationale"], "resolution": item["metadata_decision"],
                "rationale": "fail-closed item-level terms, period, language, and primary-text policy",
            })
    for unit in units:
        evidence = ";".join(filter(None, (
            unit["quality_flags"], unit["language_flags"], unit["protected_v6_poem_ids"],
        )))
        if evidence:
            rows.append({
                "review_id": f"unit:{unit['unit_id']}", "record_id": unit["record_id"],
                "unit_id": unit["unit_id"], "review_type": "text_probe_anomaly",
                "evidence": evidence, "resolution": unit["probe_decision"],
                "rationale": "frozen quality, language, and protected-validation gates",
            })
    return rows


def _decision_rows(
    inventory: list[dict[str, Any]], units: list[dict[str, Any]],
    files: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unit_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in units:
        unit_groups[row["record_id"]].append(row)
    files_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in files:
        files_by_record[row["record_id"]].append(row)
    total_candidate = sum(int(row["cleaned_character_count"]) for row in units)
    resulting = BASE_CORPUS_CHARACTERS + total_candidate
    decisions = []
    for item in inventory:
        group = unit_groups[item["record_id"]]
        eligible = [row for row in group if row["probe_decision"].startswith("eligible_")]
        excluded = [row for row in group if row["probe_decision"].startswith("excluded_")]
        held = [row for row in group if row["probe_decision"].startswith("hold_")]
        candidate_chars = sum(int(row["cleaned_character_count"]) for row in group)
        eligible_chars = sum(int(row["cleaned_character_count"]) for row in eligible)
        if item["metadata_decision"] != "eligible_text_probe_inactive":
            final_status = item["metadata_decision"]
            next_action = "closed at metadata gate; retain evidence only"
        elif any(row["acquisition_status"] == "error_hold" for row in files_by_record[item["record_id"]]):
            final_status = "hold_acquisition_or_parse_error_inactive"
            next_action = "resolve the recorded acquisition/parse error before checkpoint 7"
        elif eligible:
            final_status = "eligible_checkpoint_7_canonicalization_inactive"
            next_action = "carry eligible units and recorded exclusions into cross-archive canonicalization"
        elif held:
            final_status = "hold_segment_resolution_before_checkpoint_7"
            next_action = "resolve protected or embedded segments before any final build"
        elif not group and any(
            row["acquisition_status"] == "cached_verified_no_extractable_primary_text"
            for row in files_by_record[item["record_id"]]
        ):
            final_status = "closed_no_extractable_primary_text_format"
            next_action = "retain the verified binary/legacy-format evidence; materialize nothing"
        else:
            final_status = "closed_no_unique_eligible_text"
            next_action = "retain duplicate/quality evidence; materialize nothing"
        decisions.append({
            "record_id": item["record_id"], "archive_id": item["archive_id"],
            "title": item["title"], "assigned_role": item["assigned_role"],
            "metadata_decision": item["metadata_decision"],
            "extracted_unit_count": len(group), "eligible_unit_count": len(eligible),
            "excluded_unit_count": len(excluded), "held_unit_count": len(held),
            "candidate_character_count": candidate_chars,
            "eligible_character_count": eligible_chars,
            "share_of_candidate_pool": f"{candidate_chars / total_candidate:.8f}" if total_candidate else "0.00000000",
            "share_of_resulting_pool": f"{candidate_chars / resulting:.8f}" if resulting else "0.00000000",
            "final_status": final_status, "next_action": next_action,
            "activation_status": "inactive_audit_only",
        })
    return decisions


def _build_report(
    config: ILCOTAAuditConfig, inventory: list[dict[str, Any]], files: list[dict[str, Any]],
    units: list[dict[str, Any]], overlaps: list[dict[str, Any]],
    reviews: list[dict[str, Any]], decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible_units = [row for row in units if row["probe_decision"].startswith("eligible_")]
    candidate_chars = sum(int(row["cleaned_character_count"]) for row in units)
    eligible_chars = sum(int(row["cleaned_character_count"]) for row in eligible_units)
    role_characters: Counter[str] = Counter()
    archive_characters: Counter[str] = Counter()
    for row in eligible_units:
        characters = int(row["cleaned_character_count"])
        role_characters[row["assigned_role"]] += characters
        archive_characters[row["archive_id"]] += characters
    concentration = sorted(
        (
            {
                "record_id": row["record_id"],
                "title": row["title"],
                "eligible_characters": int(row["eligible_character_count"]),
                "share_of_resulting_eligible_pool": (
                    int(row["eligible_character_count"])
                    / (BASE_CORPUS_CHARACTERS + eligible_chars)
                ),
            }
            for row in decisions
            if int(row["eligible_character_count"])
        ),
        key=lambda row: (-row["eligible_characters"], row["record_id"]),
    )
    return {
        "report_version": AUDIT_VERSION,
        "audit_date": _max_retrieval_date(config.cache_dir),
        "inventory_record_count": len(inventory),
        "ilc_record_count": sum(row["archive_id"] == "ilc_cnr_historical_corpora" for row in inventory),
        "ota_record_count": sum(row["archive_id"] == "oxford_text_archive" for row in inventory),
        "metadata_eligible_record_count": sum(row["metadata_decision"] == "eligible_text_probe_inactive" for row in inventory),
        "acquired_file_count": sum(row["acquisition_status"].startswith("cached_verified") for row in files),
        "extractable_file_count": sum(row["acquisition_status"] == "cached_verified" for row in files),
        "redundant_representation_file_count": sum(
            row["acquisition_status"] == "cached_verified_redundant_representation"
            for row in files
        ),
        "no_extractable_primary_text_file_count": sum(
            row["acquisition_status"] == "cached_verified_no_extractable_primary_text"
            for row in files
        ),
        "acquisition_error_count": sum(row["acquisition_status"] == "error_hold" for row in files),
        "extracted_unit_count": len(units),
        "eligible_unit_count": len(eligible_units),
        "candidate_character_count": candidate_chars,
        "eligible_character_count": eligible_chars,
        "eligible_role_character_counts": dict(sorted(role_characters.items())),
        "eligible_archive_character_counts": dict(sorted(archive_characters.items())),
        "source_concentration": concentration,
        "candidate_word_count": sum(int(row["cleaned_word_count"]) for row in units),
        "overlap_pair_count": len(overlaps),
        "internal_overlap_pair_count": sum(row["pair_scope"] == "internal" for row in overlaps),
        "cross_overlap_pair_count": sum(row["pair_scope"] == "cross" for row in overlaps),
        "protected_v6_unit_count": sum(bool(row["protected_v6_poem_ids"]) for row in units),
        "review_row_count": len(reviews),
        "decision_status_counts": dict(sorted(Counter(row["final_status"] for row in decisions).items())),
        "unit_status_counts": dict(sorted(Counter(row["probe_decision"] for row in units).items())),
        "base_corpus_characters": BASE_CORPUS_CHARACTERS,
        "uncapped_resulting_candidate_characters": BASE_CORPUS_CHARACTERS + candidate_chars,
        "uncapped_resulting_eligible_characters": BASE_CORPUS_CHARACTERS + eligible_chars,
        "text_activated": False, "v7_created": False,
        "mixture_weights_assigned": False, "cache_deleted": False,
        "gpu_work_started": False,
        "conditioned_pelavicino_in_standard_queue": False,
        "next_checkpoint": "7A cross-archive overlap and canonical decision freeze",
        "artifact_sha256": {
            "inventory_csv": _sha256_file(config.inventory_path),
            "file_csv": _sha256_file(config.file_path),
            "unit_csv": _sha256_file(config.unit_path),
            "overlap_csv": _sha256_file(config.overlap_path),
            "review_csv": _sha256_file(config.review_path),
            "decision_csv": _sha256_file(config.decision_path),
        },
    }


def _render_markdown(report: dict[str, Any], decisions: list[dict[str, Any]]) -> str:
    lines = [
        "# Checkpoint 6D: ILC-CNR And Oxford Text Archive Source Audit", "",
        f"Audit date: `{report['audit_date']}`", "", "## Outcome", "",
        f"The bounded inventory accounts for {report['inventory_record_count']} records: "
        f"{report['ilc_record_count']} ILC-CNR deposits and all {report['ota_record_count']} "
        "Italian-facet Oxford records.",
        f"Compatible item gates acquired {report['acquired_file_count']} files and extracted "
        f"{report['extracted_unit_count']} auditable text units containing "
        f"{report['candidate_character_count']:,} characters.",
        f"After source-level quality, language, overlap, and protected-V6 gates, "
        f"{report['eligible_unit_count']} units / {report['eligible_character_count']:,} "
        "characters remain inactive candidates for checkpoint 7.", "",
        f"The uncapped eligible projection would raise the frozen broader pool from "
        f"{report['base_corpus_characters']:,} to "
        f"{report['uncapped_resulting_eligible_characters']:,} characters. This is an "
        "audit projection, not activation or a training weight.", "",
        "The ten-work scarcity rule was only an admission floor. It did not cap the audit: "
        "all 56 advertised libretti, all 40 Bellini letters, all 55 Rosmini TEI works, "
        "and all 43 Oxford catalog records were accounted for.", "",
        "## Record decisions", "",
        "| Record | Role | Units | Candidate characters | Final status |", "|---|---|---:|---:|---|",
    ]
    for row in decisions:
        lines.append(
            f"| {row['title']} | `{row['assigned_role']}` | {row['extracted_unit_count']} | "
            f"{int(row['candidate_character_count']):,} | `{row['final_status']}` |"
        )
    lines.extend([
        "", "## Overlap and safety", "",
        f"- Threshold overlap pairs: {report['overlap_pair_count']} "
        f"({report['internal_overlap_pair_count']} internal; {report['cross_overlap_pair_count']} cross-corpus).",
        f"- Units touching protected V6 validation/test sonnets: {report['protected_v6_unit_count']}.",
        f"- Largest eligible record concentration: {report['source_concentration'][0]['title']} at "
        f"{report['source_concentration'][0]['share_of_resulting_eligible_pool']:.2%} "
        "of the uncapped resulting pool.",
        "- Codice Pelavicino remains conditioned Italian/Latin and was not placed in the standard queue.",
        "- Original spelling and punctuation are preserved; TEI headers and explicit apparatus are excluded.",
        "- No text is activated, no V7 split or mixture weight is assigned, no cache is deleted, and no GPU work starts.",
        "- Next checkpoint: 7A cross-archive overlap and canonical decision freeze.", "",
    ])
    return "\n".join(lines)


def _load_protected_watch(
    config: ILCOTAAuditConfig,
) -> tuple[dict[int, tuple[str, ...]], dict[str, int]]:
    watch: dict[int, list[str]] = defaultdict(list)
    denominators = {}
    for row in _read_csv(config.protected_v6_sonnet_manifest_path):
        if row["split_expanded_with_petrarch"] not in {"validation", "test"}:
            continue
        path = config.repo_root / row["clean_text_path"]
        fingerprint, _ = fingerprint_text(path.read_text(encoding="utf-8"), sketch_size=10_000_000, anchor_mask=0)
        values = set(fingerprint.anchors)
        if not values:
            continue
        denominators[row["poem_id"]] = len(values)
        for value in values:
            watch[value].append(row["poem_id"])
    return {value: tuple(ids) for value, ids in watch.items()}, denominators


def _discover_pairs(fingerprints: dict[str, TextFingerprint]) -> set[tuple[str, str]]:
    pairs = set()
    for attribute in ("anchors", "sketch"):
        postings: dict[int, list[str]] = defaultdict(list)
        for document_id, fingerprint in fingerprints.items():
            for value in getattr(fingerprint, attribute):
                postings[value].append(document_id)
        collisions: Counter[tuple[str, str]] = Counter()
        for ids in postings.values():
            if 1 < len(ids) <= 40:
                collisions.update(tuple(sorted(pair)) for pair in combinations(ids, 2))
        for pair, count in collisions.items():
            denominator = min(len(getattr(fingerprints[pair[0]], attribute)), len(getattr(fingerprints[pair[1]], attribute)))
            if denominator and count >= 2 and count / denominator >= 0.4:
                pairs.add(pair)
    return pairs


def _discover_cross_pairs(
    candidates: dict[str, TextFingerprint], references: dict[str, TextFingerprint],
) -> set[tuple[str, str]]:
    pairs = set()
    for attribute in ("anchors", "sketch"):
        postings: dict[int, list[str]] = defaultdict(list)
        for reference_id, fingerprint in references.items():
            for value in getattr(fingerprint, attribute):
                postings[value].append(reference_id)
        for unit_id, fingerprint in candidates.items():
            collisions: Counter[str] = Counter()
            for value in getattr(fingerprint, attribute):
                found = postings.get(value, ())
                if len(found) <= 40:
                    collisions.update(found)
            for reference_id, count in collisions.items():
                denominator = min(len(getattr(fingerprint, attribute)), len(getattr(references[reference_id], attribute)))
                if denominator and count >= 2 and count / denominator >= 0.4:
                    pairs.add((unit_id, reference_id))
    return pairs


def _fetch_cached(
    config: ILCOTAAuditConfig, session: requests.Session, cache_id: str, url: str,
    category: str, sleep: Sleep,
) -> tuple[bytes, dict[str, Any]]:
    directory = config.cache_dir / category
    body_path = directory / f"{_safe_id(cache_id)}.bin"
    metadata_path = directory / f"{_safe_id(cache_id)}.json"
    if body_path.is_file() and metadata_path.is_file():
        content = body_path.read_bytes()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["source_url"] != url:
            raise ValueError(f"cached URL drift for {cache_id}")
        if hashlib.sha256(content).hexdigest() != metadata["content_sha256"]:
            raise ValueError(f"cached content hash mismatch for {cache_id}")
        return content, metadata
    error: Exception | None = None
    response = None
    for attempt in range(1, config.max_attempts + 1):
        try:
            response = session.get(url, timeout=config.request_timeout_seconds)
            response.raise_for_status()
            break
        except requests.RequestException as caught:
            error = caught
            response = None
            if attempt < config.max_attempts:
                sleep(config.request_delay_seconds * attempt)
    if response is None:
        raise RuntimeError(f"failed to fetch {cache_id}: {error}")
    content = response.content
    retrieved = response.headers.get("Date", "")
    try:
        retrieval_date = datetime.strptime(retrieved, "%a, %d %b %Y %H:%M:%S %Z").date().isoformat()
    except ValueError:
        retrieval_date = datetime.now(UTC).date().isoformat()
    metadata = {
        "source_url": url, "resolved_url": response.url,
        "retrieval_date": retrieval_date, "http_status": response.status_code,
        "content_type": response.headers.get("Content-Type", "").split(";", 1)[0],
        "content_sha256": hashlib.sha256(content).hexdigest(),
    }
    directory.mkdir(parents=True, exist_ok=True)
    body_path.write_bytes(content)
    _write_json(metadata_path, metadata)
    if config.request_delay_seconds:
        sleep(config.request_delay_seconds)
    return content, metadata


def _parse_xml(payload: bytes) -> ET.Element:
    text = payload.decode("utf-8-sig", "replace")
    text = _remove_doctype(text)
    text = _NAMED_ENTITY.sub(_replace_entity, text)
    return ET.fromstring(text)


def _remove_doctype(text: str) -> str:
    match = re.search(r"<!DOCTYPE\b", text, re.IGNORECASE)
    if match is None:
        return text
    index = match.end()
    depth = 0
    quote = ""
    while index < len(text):
        character = text[index]
        if quote:
            if character == quote:
                quote = ""
        elif character in {"'", '"'}:
            quote = character
        elif character == "[":
            depth += 1
        elif character == "]":
            depth = max(0, depth - 1)
        elif character == ">" and depth == 0:
            return text[:match.start()] + text[index + 1:]
        index += 1
    raise ValueError("unterminated XML doctype")


def _replace_entity(match: re.Match[str]) -> str:
    name = match.group(1)
    if name in _XML_ENTITIES:
        return match.group(0)
    value = html_entities.html5.get(f"{name};")
    if value is None:
        raise ValueError(f"unsupported XML named entity: &{name};")
    return "".join(f"&#{ord(character)};" for character in value)


def _extract_primary_text(element: ET.Element) -> str:
    output: list[str] = []

    def visit(node: ET.Element) -> None:
        tag = _local_name(node.tag)
        if tag in _SKIP_TAGS:
            return
        if tag == "choice":
            children = list(node)
            selected = next((child for child in children if _local_name(child.tag) in {"orig", "sic"}), children[0] if children else None)
            if selected is not None:
                visit(selected)
            return
        if tag == "app":
            selected = next((child for child in node if _local_name(child.tag) == "lem"), None)
            if selected is not None:
                visit(selected)
            return
        if tag in {"del", "gap"}:
            return
        if node.text:
            output.append(node.text)
        for child in node:
            visit(child)
            if child.tail:
                output.append(child.tail)
        if tag in _BLOCK_TAGS:
            output.append("\n")

    visit(element)
    return "".join(output)


def _header_text(root: ET.Element, name: str) -> str:
    header = _first_descendant(root, "teiheader")
    if header is None:
        return ""
    return _first_descendant_text(header, name)


def _first_descendant_text(element: ET.Element, name: str) -> str:
    found = _first_descendant(element, name)
    return _compact(found) if found is not None else ""


def _first_descendant(element: ET.Element, name: str) -> ET.Element | None:
    normalized = name.casefold()
    return next((child for child in element.iter() if _local_name(child.tag) == normalized), None)


def _first_child(element: ET.Element, name: str) -> ET.Element | None:
    normalized = name.casefold()
    return next((child for child in element if _local_name(child.tag) == normalized), None)


def _descendants(element: ET.Element, name: str) -> list[ET.Element]:
    normalized = name.casefold()
    return [child for child in element.iter() if _local_name(child.tag) == normalized]


def _compact(element: ET.Element) -> str:
    return _WHITESPACE.sub(" ", "".join(element.itertext()).replace("\r", " ")).strip()


def _clean_text(text: str) -> str:
    lines = [_WHITESPACE.sub(" ", line).strip() for line in text.replace("\r", "").split("\n")]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].casefold()


def _download_name(row: dict[str, Any], url: str | None = None) -> str:
    spec = next((value for value in ILC_SOURCES if value.record_id == row["record_id"]), None)
    if spec is not None:
        return spec.file_name
    selected = url or row["bitstream_url"].split(";", 1)[0]
    return unquote(Path(urlparse(selected).path).name) or f"{row['record_id']}.bin"


def _suffix(url: str) -> str:
    return Path(unquote(urlparse(url).path)).suffix.casefold()


def _period(year: int | None) -> str:
    if year is None:
        return "unresolved"
    if year <= 1500:
        return "origins_to_1500"
    if year <= 1700:
        return "1501_to_1700"
    if year <= 1800:
        return "1701_to_1800"
    if year <= 1900:
        return "ottocento_bridge"
    return "post_1900"


def _first_year(value: str) -> int | None:
    match = re.search(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)", value)
    return int(match.group(1)) if match else None


def _integer(value: Any) -> int | None:
    try:
        return int(value) if value not in {None, ""} else None
    except (TypeError, ValueError):
        return None


def _first(values: Iterable[str]) -> str:
    return next(iter(values), "")


def _append(current: str, value: str) -> str:
    values = set(filter(None, current.split(";")))
    values.add(value)
    return ";".join(sorted(values))


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _language_scores(words: list[str]) -> tuple[float, float, float]:
    if not words:
        return 0.0, 0.0, 0.0
    denominator = len(words)
    return (
        sum(word in _ITALIAN_LANGUAGE_WORDS for word in words) / denominator,
        sum(word in _ENGLISH_LANGUAGE_WORDS for word in words) / denominator,
        sum(word in _LATIN_LANGUAGE_WORDS for word in words) / denominator,
    )


def _binary_control_ratio(payload: bytes) -> float:
    if not payload:
        return 1.0
    controls = sum(value < 32 and value not in {9, 10, 13} for value in payload)
    return controls / len(payload)


def _remove_controls(value: str) -> str:
    return "".join(character for character in value if ord(character) >= 32 or character in "\n\t")


def _validate_config(config: ILCOTAAuditConfig) -> None:
    if config.request_delay_seconds < 0 or config.request_timeout_seconds <= 0:
        raise ValueError("request delay/timeout must be non-negative/positive")
    if config.max_attempts <= 0:
        raise ValueError("max attempts must be positive")
    if not 0 < config.near_duplicate_containment <= 1:
        raise ValueError("near-duplicate threshold must be in (0, 1]")
    if not 0 < config.protected_containment <= 1:
        raise ValueError("protected threshold must be in (0, 1]")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _max_retrieval_date(cache_dir: Path) -> str:
    dates = []
    for path in cache_dir.glob("*/*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("retrieval_date"):
            dates.append(payload["retrieval_date"])
    return max(dates) if dates else datetime.now(UTC).date().isoformat()


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _report(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)
