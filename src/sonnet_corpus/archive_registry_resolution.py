"""Resolve the remaining corpus archive registry from pinned official evidence."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from time import monotonic, sleep as default_sleep
from typing import Any
from urllib.parse import quote_plus

import requests


AUDIT_VERSION = "corpus_archive_registry_resolution_v1"
USER_AGENT = "portfolio-transformer-poetry-archive-resolution/1.0"
BASE_CORPUS_CHARACTERS = 626_379_622

ARCHIVE_IDS = (
    "bibit_scrittori_italia",
    "bibit_incunaboli",
    "eltec_italian",
    "internet_archive",
    "gallica",
    "internet_culturale",
    "beic",
    "hathitrust",
    "google_books",
    "ovi_tlio",
    "midia",
    "diacoris",
)

EVIDENCE_FIELDS = (
    "evidence_id", "archive_id", "evidence_type", "authority",
    "source_url", "resolved_url", "retrieval_date", "http_status",
    "content_type", "content_sha256", "evidence_quote",
    "supports_decision", "limitation", "verification_status",
)

RESOLUTION_FIELDS = (
    "archive_id", "archive_name", "frozen_input_status",
    "official_evidence_ids", "training_permission", "reuse_obligations",
    "bulk_access_route", "assigned_corpus_role", "final_status",
    "measured_scope", "value_assessment", "blocking_condition",
    "next_action", "activation_status",
)

GATE_FIELDS = (
    "archive_id", "assigned_corpus_role", "language_variety",
    "historical_period", "register_genre_form", "scope_unit",
    "scope_count", "projected_characters", "projected_tokens",
    "projected_share_of_resulting_corpus", "concentration_risk",
    "expected_unique_value", "full_audit_runtime_lower_hours",
    "full_audit_runtime_upper_hours", "full_audit_value",
    "composition_gate_decision", "gate_reason", "activation_status",
)

Progress = Callable[[str], None]
Sleep = Callable[[float], None]


@dataclass(frozen=True)
class EvidenceSpec:
    evidence_id: str
    archive_id: str
    evidence_type: str
    url: str
    quote: str
    needles: tuple[str, ...]
    supports: str
    limitation: str
    allow_inaccessible: bool = False
    json_count_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArchiveRegistryResolutionConfig:
    repo_root: Path
    registry_path: Path
    cache_dir: Path
    evidence_path: Path
    resolution_path: Path
    composition_gate_path: Path
    json_report_path: Path
    markdown_report_path: Path
    request_timeout_seconds: float = 45.0
    request_delay_seconds: float = 0.25
    max_attempts: int = 3


def _ia_query_url() -> str:
    query = "language:ita AND mediatype:texts AND date:[1200-01-01 TO 1900-12-31]"
    return (
        "https://archive.org/advancedsearch.php?q=" + quote_plus(query)
        + "&fl%5B%5D=identifier&rows=0&page=1&output=json"
    )


EVIDENCE_SPECS = (
    EvidenceSpec(
        "bibit_scrittori_scope", "bibit_scrittori_italia", "scope_and_access",
        "http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/pages",
        "Scrittori d’Italia comprises 287 volumes, 179 works, and 125,171 image-text pages; the images are freely available under Laterza's permission.",
        ("287 volumi", "179 opere", "125.171 immagini-testo", "Tutte le immagini sono liberamente disponibili"),
        "Pins the complete advertised scope and confirms an image rather than transcription layer.",
        "Image availability does not establish an OCR/full-text layer or avoid overlap with BibIt.",
    ),
    EvidenceSpec(
        "bibit_incunaboli_scope", "bibit_incunaboli", "scope_and_access",
        "http://backend.bibliotecaitaliana.it/wp-json/muruca-core/v1/pages",
        "Incunaboli in volgare contains more than 1,600 incunabula from about 70 libraries and more than 200,000 freely consultable images with technical and management metadata.",
        ("più di 1600", "circa 70 biblioteche", "più di 200.000 immagini", "liberamente consultabili"),
        "Pins the archive-scale image inventory and confirms metadata availability.",
        "No corrected transcription, OCR download, or corpus-text reuse grant is advertised.",
    ),
    EvidenceSpec(
        "eltec_release_terms", "eltec_italian", "release_terms_and_scope",
        "https://raw.githubusercontent.com/COST-ELTeC/ELTeC-ita/master/README.md",
        "The ELTeC Italian release contains 36 level-1 novels; all included texts are public domain and the textual markup is CC BY 4.0.",
        ("contains 36 novels encoded at level 1", "All texts included in this collection are in the public domain", "Creative Commons Attribution International 4.0"),
        "Establishes release size, text status, markup license, contributors, and source families.",
        "The release overlaps BibIt/CLiGS sources and includes early-twentieth-century material, so it requires capped, record-level inventory.",
    ),
    EvidenceSpec(
        "internet_archive_automation", "internet_archive", "automated_access",
        "https://archive.org/developers/bots.html",
        "Internet Archive documents automated access for bots, AI agents, and LLMs and requires descriptive user agents, delays, and rate-limit handling.",
        ("Bots, LLMs, and Automated Access", "All automated requests", "Add delays between requests for bulk operations", "Honor 429"),
        "Authorizes a rate-limited metadata inventory through documented services.",
        "Operational access guidance is not a blanket license for item text.",
    ),
    EvidenceSpec(
        "internet_archive_historical_italian_count", "internet_archive", "metadata_count",
        _ia_query_url(), "",
        ("language:ita", "mediatype:texts"),
        "Measures the broad metadata candidate ceiling for Italian texts dated no later than 1900.",
        "Language/date metadata is noisy; the count is not a rights-cleared or OCR-quality-passing corpus.",
        json_count_path=("response", "numFound"),
    ),
    EvidenceSpec(
        "gallica_api_access", "gallica", "metadata_and_ocr_access",
        "https://api.bnf.fr/fr/api-gallica-de-recherche",
        "Gallica's SRU API searches the digital collection and exposes OCR-quality values from 0 to 100.",
        ("API de recherche de Gallica", "ocrquality", "100.00 étant un OCR sans faute"),
        "Supports a metadata-only inventory and a later measured OCR-quality gate.",
        "API access does not itself permit reuse of digitized documents.",
    ),
    EvidenceSpec(
        "gallica_api_terms", "gallica", "api_terms",
        "https://api.bnf.fr/fr/conditions-generales-dutilisation-du-site-bnf-api-et-jeux-de-donnees",
        "BnF API terms permit dataset downloads but require separate Gallica content terms or prior authorization for digitized-document reuse; BnF metadata is Etalab Open Licence 2.0 with source and update-date credit.",
        ("Télécharger les jeux de données", "obtenu les autorisations", "licence ouverte", "mentionner la source"),
        "Permits bounded metadata discovery with attribution.",
        "Corpus-text training remains blocked until the exact Gallica content and item rights pass.",
    ),
    EvidenceSpec(
        "internet_culturale_terms", "internet_culturale", "portal_terms",
        "https://www.internetculturale.it/it/15/termini-d-uso",
        "Internet Culturale permits non-commercial sharing and transformation of hosted digital content with attribution and ShareAlike, subject to different item-specific indications; metadata is CC0 1.0.",
        ("Autorizzazione al riuso", "Non-commerciale", "Condividi allo stesso modo", "CC0 1.0"),
        "Establishes a compatible non-commercial reuse route and its attribution/share-alike duties.",
        "Partner sites and item-specific notices can override portal defaults; web-resolution files are not automatically primary text.",
    ),
    EvidenceSpec(
        "internet_culturale_scope", "internet_culturale", "collection_scope",
        "https://www.internetculturale.it/it/41/collezioni-digitali",
        "The official collection directory reports 291 digital collections and distinguishes full-text and digital-text discovery routes.",
        ("291 risultati trovati", "Testi digitali", "Full-text"),
        "Measures the collection-level discovery universe before item inventory.",
        "Collection count is not a count of reusable Italian primary-text documents.",
    ),
    EvidenceSpec(
        "beic_terms", "beic", "digital_library_terms",
        "https://www.beic.it/wp-json/wp/v2/pages/2886",
        "BEIC states that its digital-library images are freely accessible, downloadable, and reusable; public-domain reproductions remain public domain, copyrighted BEIC-held works are CC BY-SA, and catalog/OAI-PMH metadata is CC0.",
        ("liberamente accessibili", "scaricabili", "restano nel pubblico dominio", "CC BY-SA", "CC-0"),
        "Establishes reusable digital objects and open metadata subject to exact work status.",
        "The image license does not establish corrected OCR or identify Italian literary text automatically.",
    ),
    EvidenceSpec(
        "beic_scope", "beic", "collection_scope",
        "https://www.beic.it/wp-json/wp/v2/pages/2881",
        "BEIC reports 39,821 digital resources, 98,327 bibliographic records, and 5,617 authors.",
        ("39.821 risorse digitali", "98.327 registrazioni bibliografiche", "5.617 autori"),
        "Measures the broad digital-library ceiling.",
        "The total is multidisciplinary and includes non-text, non-Italian, and in-copyright material.",
    ),
    EvidenceSpec(
        "hathitrust_terms_access", "hathitrust", "terms_access_blocker",
        "https://www.hathitrust.org/terms-of-use/",
        "The official HathiTrust terms page returned an automated-access challenge to this audit client.",
        (), "Records that first-party permission evidence could not be pinned.",
        "No training, bulk-download, or redistribution permission may be inferred from full-view status.",
        allow_inaccessible=True,
    ),
    EvidenceSpec(
        "google_books_api_terms", "google_books", "api_terms",
        "https://developers.google.com/books/terms",
        "Google Books API use is governed by the Books API and Google APIs terms, with removal duties for supplied content.",
        ("agree to these terms", "Google APIs Terms of Service", "remove infringing content"),
        "Supports metadata discovery under the API terms.",
        "The terms do not grant a blanket corpus-text training or bulk-download right.",
    ),
    EvidenceSpec(
        "google_books_api_access", "google_books", "metadata_access",
        "https://developers.google.com/books/docs/v1/using",
        "The Books API is a search and access interface; availability varies with copyright, contract, legal restrictions, and user location.",
        ("search and access book content", "copyright, contract, and other legal restrictions", "user's location"),
        "Supports discovery of editions that can be located in reusable archives.",
        "API discovery and preview availability are not reusable corpus-text provenance.",
    ),
    EvidenceSpec(
        "ovi_tlio_access", "ovi_tlio", "query_access",
        "http://tlio.ovi.cnr.it/TLIO/",
        "The official TLIO landing page exposes an interrogation menu.",
        ("sistema di interrogazione del TLIO", "Menù di interrogazione"),
        "Confirms query-interface access.",
        "No bulk text, license, redistribution, or model-training permission is stated.",
    ),
    EvidenceSpec(
        "midia_terms_and_scope", "midia", "terms_scope_and_access",
        "https://www.corpusmidia.unito.it/",
        "MIDIA reports about 7.8 million occurrences from about 800 texts spanning the thirteenth to mid-twentieth century and distributes the corpus under CC BY-NC 4.0.",
        ("7,8 milioni di occorrenze", "circa 800 testi", "licenza Creative Commons BY-NC 4.0"),
        "Establishes non-commercial reuse terms, composition scale, and attribution duty.",
        "The corpus crosses 1900 and still requires period filtering, source-level deduplication, and an access-method audit.",
    ),
    EvidenceSpec(
        "midia_result_access", "midia", "result_access",
        "https://www.corpusmidia.unito.it/documentation.php",
        "MIDIA documents downloadable CSV query results and links from concordance rows to the source text and work metadata.",
        ("scaricarli in formato CSV", "scaricare il testo del corpus", "metadati dell'opera"),
        "Supports a bounded source-list and access inventory.",
        "A query-result interface is not yet a verified bulk release endpoint.",
    ),
    EvidenceSpec(
        "diacoris_access_terms", "diacoris", "query_terms_and_scope",
        "https://corpora.ficlit.unibo.it/DiaCORIS/",
        "DiaCORIS exposes 1861-2001 sections and grants query access exclusively for scientific research with no economic benefit.",
        ("Section 1861-1900", "scopi di ricerca scientifica", "alcun beneficio economico", "L'accesso al corpus"),
        "Confirms the historical slice and non-commercial research query route.",
        "It does not grant bulk text, redistribution, derivative-dataset, or model-training permission.",
    ),
)


def _resolution_rows() -> list[dict[str, str]]:
    rows = [
        ("bibit_scrittori_italia", "Biblioteca Italiana - Scrittori d'Italia", "inventory_pending", "bibit_scrittori_scope", "not_established_for_corpus_text", "Retain Biblioteca Italiana/Laterza provenance; image availability only.", "image_only_no_ocr_or_transcription", "excluded", "closed_excluded_scan_only_overlap_risk", "179 works; 287 volumes; 125,171 image-text pages", "Low expected unique text after overlap with the completed BibIt and Wikisource/Laterza holdings.", "No reusable transcription layer; OCR would duplicate higher-quality canonical text.", "No further extraction; retain as edition/source-scan reference."),
        ("bibit_incunaboli", "Biblioteca Italiana - Incunaboli in volgare", "inventory_pending", "bibit_incunaboli_scope", "not_established_for_corpus_text", "Record library and item provenance for any future scan use.", "image_only_no_ocr_or_transcription", "auxiliary_ocr_candidate", "blocked_ocr_and_item_rights_gate", "More than 1,600 incunabula; about 70 libraries; more than 200,000 images", "Potentially high unique medieval value, but only after a separately approved OCR sample.", "No corrected text layer, corpus reuse grant, or measured OCR quality.", "Do not audit pages now; require a later approved OCR-quality and item-rights experiment."),
        ("eltec_italian", "ELTeC Italian novel collection", "terms_audit_pending", "eltec_release_terms", "permitted_with_conditions", "Texts are public domain; attribute CC BY 4.0 markup and named contributors.", "git_tei_release_bounded_inventory", "auxiliary_capped_ottocento_bridge", "eligible_bounded_inventory_inactive", "36 level-1 novels", "Moderate bridge value; likely substantial overlap because declared sources include BibIt and CLiGS Textbox.", "No blocker to metadata/source inventory; record-level dates, sources, and duplicate status remain unresolved.", "Checkpoint 6B may inventory the 36 records without activating or downloading corpus text through this decision alone."),
        ("internet_archive", "Internet Archive", "rights_and_quality_gate_required", "internet_archive_automation;internet_archive_historical_italian_count", "item_level_only_not_established", "Honor each item license/rights field, source credit, rate limits, and removal duties.", "advanced_search_metadata_only", "core_training_candidate", "eligible_bounded_inventory_inactive", "99,424 broad language/date metadata hits in the pinned 2026-08-11 query", "Potentially high recall but low precision and high duplicate/OCR risk.", "No blanket corpus license; item rights, corrected-text availability, and OCR quality must pass separately.", "Checkpoint 6B may run a bounded metadata inventory only, prioritizing explicit reusable rights and corrected OCR."),
        ("gallica", "Gallica / Bibliothèque nationale de France", "terms_audit_pending", "gallica_api_access;gallica_api_terms", "metadata_only_text_not_established", "Credit BnF and metadata update date under Etalab 2.0; obey exact Gallica/item content terms.", "sru_metadata_only", "core_training_candidate", "eligible_bounded_inventory_inactive", "SRU inventory supports language/type queries and 0-100 OCR-quality metadata; candidate count not retrievable during this audit", "Potentially high historical value, with OCR quality measurable before text acquisition.", "Digitized-document reuse requires exact Gallica content terms or rights-holder authorization.", "Checkpoint 6B may inventory metadata only; no OCR/full text until content and item rights pass."),
        ("internet_culturale", "Internet Culturale", "terms_and_access_audit_pending", "internet_culturale_terms;internet_culturale_scope", "permitted_noncommercial_with_item_override", "Attribute Internet Culturale and owning institutions; non-commercial use; ShareAlike; retain item-specific terms.", "collection_then_item_metadata_inventory", "core_training_candidate", "eligible_bounded_inventory_inactive", "291 digital collections in the official directory", "Moderate-to-high discovery value; unique corrected text availability is unknown.", "Partner/item terms and actual primary-text download formats remain unresolved.", "Checkpoint 6B may inventory text-bearing collections and item terms without acquiring corpus text."),
        ("beic", "BEIC Digital Library", "terms_and_access_audit_pending", "beic_terms;beic_scope", "permitted_subject_to_work_status", "Preserve BEIC/source credit and exact public-domain, CC BY-SA, or item-specific status; metadata is CC0.", "oai_pmh_and_catalog_metadata_inventory", "core_training_candidate", "eligible_bounded_inventory_inactive", "39,821 digital resources; 98,327 bibliographic records; 5,617 authors", "Moderate value after language/literature filtering; high scan overlap with Wikisource and other archives.", "Corrected OCR/full-text availability and work-level eligibility are not measured.", "Checkpoint 6B may inventory Italian literary records and text formats without acquiring corpus text."),
        ("hathitrust", "HathiTrust Digital Library", "blocked_pending_permission", "hathitrust_terms_access", "not_established", "None may be inferred while official terms evidence is inaccessible.", "access_and_terms_blocked", "core_training_candidate", "blocked_official_terms_and_bulk_access", "Large Italian holdings stated by registry; no auditable count pinned", "Potentially high, but lower immediate value than open archives because access and redistribution are unresolved.", "Official terms could not be pinned; full-view is not bulk reuse permission.", "Keep blocked; obtain accessible official terms or direct research-corpus permission before any inventory/acquisition."),
        ("google_books", "Google Books", "discovery_only", "google_books_api_terms;google_books_api_access", "discovery_only_no_corpus_grant", "Comply with Books/API terms and content-removal duties; do not redistribute previews.", "api_metadata_discovery_only", "excluded", "discovery_only_closed", "Large catalog; no corpus-eligible record count claimed", "Useful for edition identification, not as a corpus source.", "No blanket bulk-download or model-training permission for book content.", "Use only to locate the same edition in a reusable archive."),
        ("ovi_tlio", "OVI / TLIO", "terms_and_access_audit_pending", "ovi_tlio_access", "not_established", "No reuse obligations can be finalized without official permission terms.", "query_only", "core_training_candidate", "blocked_bulk_and_training_permission", "Official query interface confirmed; bulk scope not published", "Very high potential medieval-language value if a reusable release is authorized.", "Public query access does not establish bulk text, training, or redistribution permission.", "Request or locate explicit official bulk/research-training permission and a source list."),
        ("midia", "MIDIA historical Italian corpus", "terms_and_access_audit_pending", "midia_terms_and_scope;midia_result_access", "permitted_noncommercial_with_attribution", "CC BY-NC 4.0; cite MIDIA and preserve source/work metadata.", "bounded_source_and_access_inventory", "core_training_candidate", "eligible_bounded_inventory_inactive", "About 800 texts and 7.8 million occurrences from the 13th to mid-20th century", "High period/genre balancing value, but exact overlap and pre-1901 share are unknown.", "Bulk release/access method and record-level period/source lineage require verification.", "Checkpoint 6B may inventory the source list, periods, and access routes; this status alone does not authorize corpus-text download."),
        ("diacoris", "DiaCORIS historical corpus", "terms_and_access_audit_pending", "diacoris_access_terms", "research_query_only_not_training", "Scientific-research query access only; no economic benefit.", "query_only", "auxiliary_capped_ottocento_bridge", "blocked_bulk_and_training_permission", "Five time slices from 1861-2001; six subcorpus genres; no record/token count pinned", "Potentially useful 1861-1900 bridge slice, but later material is out of scope or preservation-only.", "No bulk text, derivative-dataset, redistribution, or training permission.", "Seek explicit permission and isolate the 1861-1900 source list before any acquisition."),
    ]
    keys = ("archive_id", "archive_name", "frozen_input_status", "official_evidence_ids", "training_permission", "reuse_obligations", "bulk_access_route", "assigned_corpus_role", "final_status", "measured_scope", "value_assessment", "blocking_condition", "next_action")
    return [dict(zip(keys, row, strict=True)) | {"activation_status": "inactive_metadata_only"} for row in rows]


def _gate_rows() -> list[dict[str, str]]:
    data = {
        "bibit_scrittori_italia": ("standard Italian", "historical through Ottocento", "canonical literary editions; page images", "works", "179", "high BibIt/Laterza overlap", "low", "0", "0", "low", "excluded", "Image OCR would duplicate higher-quality existing text."),
        "bibit_incunaboli": ("volgare; record review required", "fifteenth century", "incunabula; page images", "incunabula", ">1600", "unknown library/edition concentration", "high if unique", "40", "120", "conditional", "blocked", "A measured OCR and item-rights experiment is required first."),
        "eltec_italian": ("Italian", "nineteenth and early twentieth century", "novels", "novels", "36", "source overlap and author concentration unresolved", "moderate", "2", "5", "high", "eligible_bounded_inventory", "Small, bounded release suited to a capped bridge audit."),
        "internet_archive": ("metadata-tagged Italian", "through 1900 metadata filter", "mixed books; scan OCR", "metadata hits", "99424", "very high source/author/edition duplication", "potentially high", "20", "60", "conditional", "eligible_bounded_inventory", "Inventory must aggressively filter rights, duplicates, and OCR quality."),
        "gallica": ("Italian metadata query", "historical through Ottocento", "books/manuscripts; OCR", "not measured", "", "high edition and institution concentration possible", "potentially high", "12", "36", "conditional", "eligible_bounded_inventory", "Metadata and OCR quality are inspectable; text rights remain separate."),
        "internet_culturale": ("Italian collections", "medieval through contemporary", "mixed collections; some digital text", "collections", "291", "institution/item concentration unresolved", "moderate-high", "12", "30", "high", "eligible_bounded_inventory", "Terms support non-commercial reuse but partner/item access must be measured."),
        "beic": ("multilingual; Italian filter required", "ancient through contemporary", "multidisciplinary digital library", "digital resources", "39821", "high overlap with scan partners", "moderate", "12", "30", "high", "eligible_bounded_inventory", "Open rights metadata supports a bounded text-format inventory."),
        "hathitrust": ("Italian holdings", "historical through contemporary", "books", "not measured", "", "unknown", "potentially high", "0", "0", "low while blocked", "blocked", "Terms and research-corpus access are not pinned."),
        "google_books": ("Italian discovery", "historical through contemporary", "books/previews", "not measured", "", "very high duplicate inventory", "discovery only", "0", "0", "low", "excluded", "No corpus-text grant; use only as an edition locator."),
        "ovi_tlio": ("Old Italian", "medieval", "lexicographic/text corpus query", "not measured", "", "source-list concentration unknown", "very high if permitted", "0", "0", "low while blocked", "blocked", "Permission and bulk access are absent."),
        "midia": ("Italian", "thirteenth to mid-twentieth century", "seven text types; diachronic corpus", "texts", "~800", "period/genre/source concentration unresolved", "high", "4", "10", "high", "eligible_bounded_inventory", "CC BY-NC 4.0 supports a source/access inventory before text acquisition."),
        "diacoris": ("Italian", "1861-2001; only 1861-1900 relevant", "six prose/news/literary genres", "time slices", "5", "modern and news concentration high", "moderate bridge value", "0", "0", "low while blocked", "blocked", "Query-only research terms do not establish training permission."),
    }
    roles = {row["archive_id"]: row["assigned_corpus_role"] for row in _resolution_rows()}
    rows: list[dict[str, str]] = []
    for archive_id in ARCHIVE_IDS:
        language, period, genre, unit, count, concentration, value, low, high, audit_value, decision, reason = data[archive_id]
        projected_tokens = "7800000" if archive_id == "midia" else ""
        rows.append({
            "archive_id": archive_id,
            "assigned_corpus_role": roles[archive_id],
            "language_variety": language,
            "historical_period": period,
            "register_genre_form": genre,
            "scope_unit": unit,
            "scope_count": count,
            "projected_characters": "",
            "projected_tokens": projected_tokens,
            "projected_share_of_resulting_corpus": "not_computable_without_record_inventory",
            "concentration_risk": concentration,
            "expected_unique_value": value,
            "full_audit_runtime_lower_hours": low,
            "full_audit_runtime_upper_hours": high,
            "full_audit_value": audit_value,
            "composition_gate_decision": decision,
            "gate_reason": reason,
            "activation_status": "inactive_metadata_only",
        })
    return rows


class _PlainTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain_text(value: str) -> str:
    parser = _PlainTextParser()
    parser.feed(html.unescape(value))
    return re.sub(r"\s+", " ", html.unescape(" ".join(parser.parts))).strip()


def _all_json_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _all_json_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_json_strings(item)
    elif value is not None:
        yield str(value)


def _normalized_body(body: bytes, content_type: str) -> tuple[str, Any | None]:
    decoded = body.decode("utf-8", errors="replace")
    parsed: Any | None = None
    if "json" in content_type.lower() or decoded.lstrip().startswith(("{", "[")):
        try:
            parsed = json.loads(decoded)
            decoded = " ".join(_all_json_strings(parsed))
        except json.JSONDecodeError:
            parsed = None
    return _plain_text(decoded), parsed


def _cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _load_cached_response(cache_dir: Path, url: str) -> dict[str, Any] | None:
    stem = _cache_key(url)
    metadata_path = cache_dir / f"{stem}.json"
    body_path = cache_dir / f"{stem}.body"
    if not metadata_path.exists() and not body_path.exists():
        return None
    if not metadata_path.exists() or not body_path.exists():
        raise ValueError(f"incomplete archive evidence cache for {url}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    body = body_path.read_bytes()
    if metadata.get("source_url") != url:
        raise ValueError(f"archive evidence cache URL mismatch for {url}")
    digest = hashlib.sha256(body).hexdigest()
    if metadata.get("content_sha256") != digest:
        raise ValueError(f"archive evidence cache hash mismatch for {url}")
    return metadata | {"body": body}


def _fetch_one(
    spec: EvidenceSpec,
    config: ArchiveRegistryResolutionConfig,
    session: requests.Session,
    sleep: Sleep,
) -> dict[str, Any]:
    cached = _load_cached_response(config.cache_dir, spec.url)
    if cached is not None:
        return cached
    last_error: Exception | None = None
    for attempt in range(1, config.max_attempts + 1):
        try:
            response = session.get(spec.url, timeout=config.request_timeout_seconds, allow_redirects=True)
            status = int(response.status_code)
            if status >= 500:
                response.raise_for_status()
            body = bytes(response.content)
            metadata = {
                "schema_version": 1,
                "source_url": spec.url,
                "resolved_url": str(response.url),
                "retrieved_at_utc": datetime.now(UTC).isoformat(),
                "http_status": status,
                "content_type": str(response.headers.get("Content-Type", "")),
                "content_sha256": hashlib.sha256(body).hexdigest(),
            }
            stem = _cache_key(spec.url)
            _atomic_write(config.cache_dir / f"{stem}.body", body)
            _atomic_write(
                config.cache_dir / f"{stem}.json",
                (json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
            )
            return metadata | {"body": body}
        except requests.RequestException as error:
            last_error = error
            if attempt < config.max_attempts:
                sleep(float(2 ** (attempt - 1)))
    assert last_error is not None
    raise last_error


def fetch_official_evidence(
    config: ArchiveRegistryResolutionConfig,
    *,
    session: requests.Session | None = None,
    sleep: Sleep = default_sleep,
    progress: Progress | None = None,
    specs: tuple[EvidenceSpec, ...] = EVIDENCE_SPECS,
) -> list[dict[str, str]]:
    """Fetch or reuse official pages, verify quotations, and return public rows."""

    client = session or requests.Session()
    client.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
    fetched_by_url: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, str]] = []
    started = monotonic()
    for index, spec in enumerate(specs, 1):
        if spec.url not in fetched_by_url:
            fetched_by_url[spec.url] = _fetch_one(spec, config, client, sleep)
            if config.request_delay_seconds and index < len(specs):
                sleep(config.request_delay_seconds)
        fetched = fetched_by_url[spec.url]
        status = int(fetched["http_status"])
        normalized, parsed = _normalized_body(fetched["body"], str(fetched["content_type"]))
        if status != 200 and not spec.allow_inaccessible:
            raise ValueError(f"official evidence {spec.evidence_id} returned HTTP {status}")
        for needle in spec.needles:
            if needle not in normalized:
                raise ValueError(f"official evidence {spec.evidence_id} is missing expected text: {needle}")
        quote = spec.quote
        if spec.json_count_path:
            value = parsed
            for key in spec.json_count_path:
                if not isinstance(value, dict) or key not in value:
                    raise ValueError(f"official evidence {spec.evidence_id} lacks JSON path {spec.json_count_path}")
                value = value[key]
            quote = f"Pinned metadata query returned {int(value):,} records."
        verification = "verified_official_evidence" if status == 200 else f"official_page_inaccessible_http_{status}"
        rows.append({
            "evidence_id": spec.evidence_id,
            "archive_id": spec.archive_id,
            "evidence_type": spec.evidence_type,
            "authority": "official_first_party",
            "source_url": spec.url,
            "resolved_url": str(fetched["resolved_url"]),
            "retrieval_date": str(fetched["retrieved_at_utc"])[:10],
            "http_status": str(status),
            "content_type": str(fetched["content_type"]),
            "content_sha256": str(fetched["content_sha256"]),
            "evidence_quote": quote,
            "supports_decision": spec.supports,
            "limitation": spec.limitation,
            "verification_status": verification,
        })
        if progress is not None:
            elapsed = monotonic() - started
            rate = elapsed / index
            eta = max(0.0, rate * (len(specs) - index))
            progress(
                f"completed={index}/{len(specs)} percentage={100 * index / len(specs):.1f}% "
                f"evidence={spec.evidence_id} http={status} elapsed={elapsed:.1f}s eta={eta:.1f}s"
            )
    return rows


def _read_registry(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("archive registry has no header")
        return tuple(reader.fieldnames), list(reader)


def _validate_scope(registry_rows: list[dict[str, str]], resolutions: list[dict[str, str]]) -> None:
    ids = [row["archive_id"] for row in resolutions]
    if tuple(ids) != ARCHIVE_IDS or len(set(ids)) != len(ids):
        raise ValueError("archive resolution scope is not the frozen 12-row queue")
    registry = {row["archive_id"]: row for row in registry_rows}
    if any(archive_id not in registry for archive_id in ARCHIVE_IDS):
        raise ValueError("archive registry is missing a frozen checkpoint-6A row")
    final_by_id = {row["archive_id"]: row["final_status"] for row in resolutions}
    for row in resolutions:
        current = registry[row["archive_id"]]["status"]
        if current not in {row["frozen_input_status"], final_by_id[row["archive_id"]]}:
            raise ValueError(f"archive registry status drift for {row['archive_id']}: {current}")
    if registry.get("bibit_texts", {}).get("status") != "processed_build_complete":
        raise ValueError("completed BibIt text row must remain outside checkpoint 6A")


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _update_registry(
    path: Path,
    fields: tuple[str, ...],
    registry_rows: list[dict[str, str]],
    resolutions: list[dict[str, str]],
) -> None:
    decisions = {row["archive_id"]: row for row in resolutions}
    updated: list[dict[str, str]] = []
    for row in registry_rows:
        result = decisions.get(row["archive_id"])
        if result is not None:
            row = dict(row)
            row["license_or_reuse_status"] = result["training_permission"] + "; " + result["reuse_obligations"]
            row["bulk_access"] = result["bulk_access_route"]
            row["status"] = result["final_status"]
            row["next_action"] = result["next_action"]
            row["notes"] = result["measured_scope"] + "; " + result["blocking_condition"]
        updated.append(row)
    _write_csv(path, fields, updated)


def _artifact_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_markdown(report: dict[str, Any], resolutions: list[dict[str, str]]) -> str:
    lines = [
        "# Remaining Archive Registry Resolution, v1", "",
        f"Audit date: `{report['audit_date']}`", "",
        "This metadata-only checkpoint closes the twelve unresolved registry rows. "
        "`eligible_bounded_inventory_inactive` permits only the stated metadata/source inventory; "
        "it does not authorize corpus-text acquisition, activation, V7 splitting, mixture weighting, cache deletion, or GPU work.", "",
        "## Decisions", "",
        "| Archive | Role | Final status | Measured scope | Closure |", "| --- | --- | --- | --- | --- |",
    ]
    for row in resolutions:
        lines.append(
            f"| {row['archive_name']} | `{row['assigned_corpus_role']}` | `{row['final_status']}` | "
            f"{row['measured_scope']} | {row['next_action']} |"
        )
    lines.extend([
        "", "## Accounting", "",
        f"- Frozen registry rows: {report['archive_count']}",
        f"- Official evidence rows: {report['evidence_count']}",
        f"- Eligible bounded inventories: {report['eligible_bounded_inventory_count']}",
        f"- Concretely blocked rows: {report['blocked_count']}",
        f"- Closed exclusions/discovery-only rows: {report['closed_excluded_count']}",
        f"- Existing inactive broader-pool subtotal: {report['base_corpus_characters']:,} characters",
        "- New corpus characters activated: 0",
        "", "## Fail-closed constraints", "",
        "- First-party evidence is required for permission decisions.",
        "- Item/content terms override portal-level defaults where stated.",
        "- OCR-only archives need a separately approved measured quality gate.",
        "- HathiTrust, TLIO, and DiaCORIS remain blocked; Google Books remains discovery-only.",
        "- The open-ended final discovery pass remains checkpoint 6C.", "",
    ])
    return "\n".join(lines)


def build_archive_registry_resolution(
    config: ArchiveRegistryResolutionConfig,
    *,
    session: requests.Session | None = None,
    sleep: Sleep = default_sleep,
    progress: Progress | None = None,
    evidence_rows: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build the deterministic checkpoint-6A ledgers, reports, and registry update."""

    registry_fields, registry_rows = _read_registry(config.registry_path)
    resolutions = _resolution_rows()
    gates = _gate_rows()
    _validate_scope(registry_rows, resolutions)
    evidence = evidence_rows or fetch_official_evidence(
        config, session=session, sleep=sleep, progress=progress,
    )
    expected_evidence = {spec.evidence_id for spec in EVIDENCE_SPECS}
    actual_evidence = {row["evidence_id"] for row in evidence}
    if expected_evidence != actual_evidence or len(actual_evidence) != len(evidence):
        raise ValueError("official evidence accounting does not match the frozen specification")
    for row in resolutions:
        refs = set(row["official_evidence_ids"].split(";"))
        if not refs <= actual_evidence:
            raise ValueError(f"missing official evidence for {row['archive_id']}")
        if row["activation_status"] != "inactive_metadata_only":
            raise ValueError("checkpoint 6A cannot activate corpus text")
    if any(row["assigned_corpus_role"] not in {"core_training_candidate", "auxiliary_capped_ottocento_bridge", "auxiliary_ocr_candidate", "excluded"} for row in resolutions):
        raise ValueError("every archive must have exactly one approved composition role")

    _write_csv(config.evidence_path, EVIDENCE_FIELDS, sorted(evidence, key=lambda row: row["evidence_id"]))
    _write_csv(config.resolution_path, RESOLUTION_FIELDS, resolutions)
    _write_csv(config.composition_gate_path, GATE_FIELDS, gates)
    _update_registry(config.registry_path, registry_fields, registry_rows, resolutions)

    status_counts = Counter(row["final_status"] for row in resolutions)
    audit_date = max(row["retrieval_date"] for row in evidence)
    report: dict[str, Any] = {
        "report_version": AUDIT_VERSION,
        "audit_date": audit_date,
        "archive_count": len(resolutions),
        "archive_ids": list(ARCHIVE_IDS),
        "evidence_count": len(evidence),
        "official_first_party_evidence_count": sum(row["authority"] == "official_first_party" for row in evidence),
        "inaccessible_official_evidence_count": sum(row["verification_status"].startswith("official_page_inaccessible") for row in evidence),
        "eligible_bounded_inventory_count": sum(row["final_status"] == "eligible_bounded_inventory_inactive" for row in resolutions),
        "eligible_bounded_inventory_ids": [row["archive_id"] for row in resolutions if row["final_status"] == "eligible_bounded_inventory_inactive"],
        "blocked_count": sum(row["final_status"].startswith("blocked_") for row in resolutions),
        "blocked_ids": [row["archive_id"] for row in resolutions if row["final_status"].startswith("blocked_")],
        "closed_excluded_count": sum(row["final_status"] in {"closed_excluded_scan_only_overlap_risk", "discovery_only_closed"} for row in resolutions),
        "status_counts": dict(sorted(status_counts.items())),
        "assigned_role_counts": dict(sorted(Counter(row["assigned_corpus_role"] for row in resolutions).items())),
        "base_corpus_characters": BASE_CORPUS_CHARACTERS,
        "activated_corpus_characters": 0,
        "text_downloads": 0,
        "v7_split_created": False,
        "mixture_weights_assigned": False,
        "gpu_work_started": False,
        "cache_deleted": False,
        "next_checkpoint": "6B bounded inventories for the six eligible rows; 6C final archive discovery remains deferred",
        "artifact_sha256": {
            "evidence_csv": _artifact_hash(config.evidence_path),
            "resolution_csv": _artifact_hash(config.resolution_path),
            "composition_gate_csv": _artifact_hash(config.composition_gate_path),
            "registry_csv": _artifact_hash(config.registry_path),
        },
    }
    config.json_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.json_report_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    config.markdown_report_path.write_text(_render_markdown(report, resolutions), encoding="utf-8")
    return report
