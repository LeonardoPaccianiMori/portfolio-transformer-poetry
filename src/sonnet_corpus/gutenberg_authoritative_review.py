"""Authoritative bibliography pass for unresolved Italian Gutenberg records."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from time import monotonic, sleep as default_sleep
from typing import Any

import requests

from .gutenberg_metadata_review import RESOLUTION_FIELDS


FROZEN_PASS_1A_SHA256 = (
    "1093b7b55050d5ad1d6b8bfce4c505becad56c8f91f4ec8dba15046fb183bb2d"
)
FROZEN_MANUAL_SHA256 = (
    "995500c9de99e3de094b3969520dc5b9880757dfa5725d969b4490879908c0d4"
)
FROZEN_PASS_1A_COUNT = 673
FROZEN_MANUAL_COUNT = 364

SBN_SEARCH_URL = "https://opac.sbn.it/o/opac-api/titles-search-full-post"
SBN_DETAIL_URL = "https://opac.sbn.it/o/opac-api/title"
WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"

AUTHORITATIVE_FIELDS = (
    "resolution_pass",
    "authoritative_evidence_json",
    "authoritative_selected_source",
    "authoritative_record_id",
    "authoritative_evidence_url",
    "authoritative_evidence_year_start",
    "authoritative_evidence_year_end",
    "authoritative_evidence_text",
    "authoritative_method",
    "authoritative_confidence",
    "authoritative_retrieved_at",
    "authoritative_cache_sha256s",
    "final_period_bucket",
    "final_role",
    "final_decision",
    "final_resolution_status",
    "final_activation_class",
    "final_exclusion_reason",
)
FINAL_FIELDS = RESOLUTION_FIELDS + AUTHORITATIVE_FIELDS

_YEAR = re.compile(r"(?<!\d)(1[2-9]\d{2}|20\d{2})(?!\d)")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_VOLUME_PREFIX = re.compile(r"^\s*(?:vol(?:ume)?\.?\s*)?\d+\s*[:.-]\s*", re.I)
_VOLUME_SUFFIX = re.compile(
    r"\s*[,.-]?\s*vol(?:ume)?\.?\s*[ivxlcdm\d]+(?:\s*\([^)]*\))?\s*$", re.I
)
_WIKIDATA_EDITION_MARKERS = re.compile(
    r"\b(?:edizione|edition|film|produzione|production|episodio|episode|"
    r"adattamento|adaptation|collana|televisiv\w*)\b",
    re.I,
)
_WIKIDATA_WORK_MARKERS = re.compile(
    r"\b(?:opera|work|romanzo|novel|novella|racconto|story|poema|poem|"
    r"commedia|comedy|tragedia|play|saggio|essay|libro|book|testo|text)\b",
    re.I,
)

_LANGUAGE_POLICY = {
    "29873": {
        "route": "conditioned_mixed_language_prose_candidate",
        "decision": "route_mixed_language_segment_review",
        "activation_class": "conditioned_probe",
        "period": "nineteenth_century_bridge",
        "year_start": 1887,
        "year_end": 1887,
        "method": "primary_text_title_page_and_segment_policy",
        "text": (
            "MATTINATE NAPOLETANE | Secondo Migliaio | NAPOLI | "
            "LUIGI PIERRO EDITORE | 1887"
        ),
        "reason": "mixed_standard_italian_and_neapolitan_segments_require_extraction",
    },
    "34734": {
        "route": "conditioned_romanesco_sonnet_candidate",
        "decision": "route_conditioned_romanesco_sonnets",
        "activation_class": "conditioned_probe",
        "period": "language_conditioned_period_unresolved",
        "year_start": None,
        "year_end": None,
        "method": "primary_text_language_variety_policy",
        "text": (
            "Quanno quello piu basso e traccagnotto, / Facenno er mulinello, "
            "piano piano"
        ),
        "reason": "romanesco_sonnets_are_outside_the_standard_italian_core",
    },
    "49523": {
        "route": "nineteenth_century_bridge_candidate",
        "decision": "eligible_nineteenth_century_candidate",
        "activation_class": "eligible_probe",
        "period": "nineteenth_century_bridge",
        "year_start": 1880,
        "year_end": 1890,
        "method": "primary_text_collection_period_and_language_review",
        "text": (
            "Prose (1880-1890): furono pubblicati fra il 1880 e il 1890 "
            "nel Capitan Fracassa"
        ),
        "reason": "",
    },
    "57040": {
        "route": "conditioned_mixed_language_prose_candidate",
        "decision": "route_mixed_language_segment_review",
        "activation_class": "conditioned_probe",
        "period": "nineteenth_century_bridge",
        "year_start": 1883,
        "year_end": 1888,
        "method": "primary_text_preface_and_segment_policy",
        "text": (
            "Minuetto settecento (1883), Nennella (1884), Mattinate "
            "napoletane (1886) e Rosa Bellavita (1888)"
        ),
        "reason": "mixed_standard_italian_and_neapolitan_segments_require_extraction",
    },
    "48542": {
        "route": "conditioned_bolognese_prose_and_drama_candidate",
        "decision": "route_conditioned_bolognese_prose_and_drama",
        "activation_class": "conditioned_probe",
        "period": "language_conditioned_period_unresolved",
        "year_start": None,
        "year_end": None,
        "method": "primary_text_language_variety_policy",
        "text": (
            "ÈL SGNER PIREIN | Ai tempi dei tempi, viveva in Bologna un "
            "giornaletto ebdomadario nel quale era in grande onore il dialetto"
        ),
        "reason": "bolognese_dialect_collection_is_outside_standard_italian_core",
    },
}


_PRIMARY_TEXT_WORK_POLICY = {
    "22025": {
        "year_start": 1805,
        "year_end": 1810,
        "method": "project_gutenberg_dated_primary_documents",
        "text": (
            "Costituzione della Repubblica Italiana e Statuti Costituzionali "
            "del Regno d'Italia | dated constitutional documents from 1805 "
            "through 1810"
        ),
    },
    "22502": {
        "year_start": 1579,
        "year_end": 1613,
        "method": "project_gutenberg_dated_anthology_contents",
        "text": (
            "L'Alitinonfo: Di Reggio, il 16 maggio MDLXXIX | Breve "
            "instruzione: 1582 | Breve trattato ... Antonio Serra: 1613"
        ),
    },
    "28869": {
        "year_start": 1866,
        "year_end": 1874,
        "method": "project_gutenberg_dated_anthology_contents",
        "text": (
            "The three dramatic reviews were published in Il Politecnico in "
            "1866-67; the collected novellas were published from 1868 through "
            "1874"
        ),
    },
    "30738": {
        "year_start": 1518,
        "year_end": 1519,
        "method": "project_gutenberg_primary_text_edition_note",
        "text": "Edizione del 1519 | dell'edizione pubblicata nel 1518",
    },
    "31285": {
        "year_start": 1518,
        "year_end": 1519,
        "method": "project_gutenberg_primary_text_edition_note",
        "text": "Edizione del 1519 | dell'edizione pubblicata nel 1518",
    },
    "31818": {
        "year_start": 1518,
        "year_end": 1519,
        "method": "project_gutenberg_primary_text_edition_note",
        "text": "Edizione del 1519 | dell'edizione pubblicata nel 1518",
    },
    "32599": {
        "year_start": 1900,
        "year_end": 1900,
        "method": "project_gutenberg_primary_text_delivery_date",
        "text": (
            "Discorso pronunziato dall'autore a Torino l'11 aprile 1900 per "
            "invito di quella Società di Cultura"
        ),
    },
    "37776": {
        "year_start": 1900,
        "year_end": 1900,
        "method": "project_gutenberg_primary_text_title_page",
        "text": "NAPOLI--Tipografia Moderna 1900",
    },
    "37936": {
        "year_start": 1893,
        "year_end": 1893,
        "method": "project_gutenberg_primary_text_first_performance",
        "text": (
            "Dopo il veglione o viceversa | eseguito per la prima volta al "
            "Salone Margherita di Napoli, nel 1893"
        ),
    },
    "38216": {
        "year_start": 1898,
        "year_end": 1898,
        "method": "project_gutenberg_primary_text_first_performance",
        "text": (
            "Fiori d'arancio | Rappresentato per la prima volta nell'aprile "
            "del 1898 al teatro Sannazzaro di Napoli"
        ),
    },
    "38218": {
        "year_start": 1896,
        "year_end": 1896,
        "method": "project_gutenberg_primary_text_first_performance",
        "text": (
            "La fine dell'amore | Rappresentata per la prima volta al "
            "Sannazzaro di Napoli nel maggio del 1896"
        ),
    },
    "39239": {
        "year_start": 1200,
        "year_end": 1299,
        "method": "project_gutenberg_primary_text_collection_period",
        "text": (
            "Rimatori siculo-toscani del dugento. Serie prima - "
            "Pistoiesi-Lucchesi-Pisani"
        ),
    },
    "54070": {
        "year_start": 1682,
        "year_end": 1682,
        "method": "project_gutenberg_primary_text_title_page",
        "text": "IN VENETIA, M.DC.LXXXII. | Di Mantova li 23 Gennaro 1682",
    },
    "54167": {
        "year_start": 1550,
        "year_end": 1550,
        "method": "project_gutenberg_italian_edition_title_page",
        "text": (
            "Commentario de le piu notabili, & mostruose cose d'Italia | IN "
            "VINETIA AL SEGNO DEL POZZO | M D L"
        ),
    },
    "60249": {
        "year_start": 1895,
        "year_end": 1895,
        "method": "project_gutenberg_primary_text_delivery_date",
        "text": "La Vita Italiana nel Settecento | Conferenze tenute a Firenze nel 1895",
    },
    "63106": {
        "year_start": 1871,
        "year_end": 1877,
        "method": "project_gutenberg_dated_anthology_contents",
        "text": (
            "Una partita a scacchi; Il Trionfo d'amore; Intermezzi e scene | "
            "primary-text composition and first-performance dates span "
            "1871-1877"
        ),
    },
    "64393": {
        "year_start": 1549,
        "year_end": 1549,
        "method": "project_gutenberg_primary_text_title_page",
        "text": (
            "Lettere di molte valorose donne | IN VINEGIA APPRESSO GABRIEL "
            "GIOLITO DE FERRARI | MDXLIX"
        ),
    },
}


@dataclass(frozen=True)
class GutenbergAuthoritativeReviewConfig:
    repo_root: Path
    pass_1a_csv_path: Path
    manual_csv_path: Path
    cache_dir: Path
    final_csv_path: Path
    exclusion_csv_path: Path
    json_report_path: Path
    markdown_report_path: Path
    expected_pass_1a_sha256: str = FROZEN_PASS_1A_SHA256
    expected_manual_sha256: str = FROZEN_MANUAL_SHA256
    expected_pass_1a_count: int = FROZEN_PASS_1A_COUNT
    expected_manual_count: int = FROZEN_MANUAL_COUNT
    request_delay_seconds: float = 0.25
    request_timeout_seconds: float = 60.0
    fetch_attempts: int = 3


FetchJson = Callable[[str, str, dict[str, str], float], dict[str, Any]]
Progress = Callable[[str], None]
Sleep = Callable[[float], None]


def load_or_fetch_authority_json(
    cache_path: Path,
    *,
    source: str,
    method: str,
    url: str,
    params: dict[str, str],
    timeout_seconds: float,
    fetch_json: FetchJson,
    retrieved_at: str,
) -> tuple[dict[str, Any], str, str]:
    """Load a request-pinned cache envelope or fetch and atomically create it."""

    request = {
        "method": method.upper(),
        "url": url,
        "params": dict(sorted(params.items())),
    }
    if cache_path.is_file():
        envelope = json.loads(cache_path.read_text(encoding="utf-8"))
        _validate_cache_envelope(envelope, source=source, request=request)
        return (
            envelope["payload"],
            "hit",
            hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        )

    payload = fetch_json(method.upper(), url, params, timeout_seconds)
    if not isinstance(payload, dict):
        raise ValueError(f"{source} returned a non-object JSON payload")
    payload_sha256 = _sha256_json(payload)
    envelope = {
        "schema_version": 1,
        "source": source,
        "request": request,
        "retrieved_at": retrieved_at,
        "payload_sha256": payload_sha256,
        "payload": payload,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(cache_path)
    return payload, "downloaded", hashlib.sha256(cache_path.read_bytes()).hexdigest()


def build_sbn_search_params(
    row: dict[str, str],
    *,
    fallback: bool = False,
    title_override: str | None = None,
    include_author: bool = True,
) -> dict[str, str]:
    """Build the validated SBN advanced-search title/author request."""

    title = title_override or (
        row["title"].strip() if fallback else _core_title(row["title"])
    )
    params = {
        "core": "sbn",
        "fieldaccess[0]": "Titolo:4",
        "fieldvalue[0]": title,
        "fieldstruct[0]": "ricerca.parole_tutte:4=6",
        "sort_access": "Data_ascendente: min 31, min 3086, min 5003",
        "page-size": "50",
    }
    author = _primary_author(row["authors"])
    if author and include_author:
        params.update(
            {
                "fieldop[0]": "AND:@and@",
                "fieldaccess[1]": "Autore:1003:nocheck",
                "fieldvalue[1]": author,
                "fieldstruct[1]": "ricerca.parole_tutte:4=6",
            }
        )
    return params


def select_sbn_candidate(
    row: dict[str, str], payload: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Select the earliest directly title/author-matched dated SBN record."""

    data = payload.get("data", {})
    results = data.get("results", []) if isinstance(data, dict) else []
    matches: list[dict[str, Any]] = []
    for result in results if isinstance(results, list) else []:
        if not isinstance(result, dict) or not _sbn_result_matches(row, result):
            continue
        title_block = result.get("title", {})
        infos = [str(value) for value in result.get("infos", [])]
        publication = infos[0] if infos else ""
        years = _publication_years(publication)
        identifier = str(result.get("id", ""))
        bid = identifier.removeprefix("ITICCU")
        matches.append(
            {
                "id": identifier,
                "bid": bid,
                "title": str(title_block.get("info", "")),
                "author": str(title_block.get("text", "")),
                "publication": publication,
                "year_start": min(years) if years else None,
                "year_end": max(years) if years else None,
                "index": int(result.get("index", 0) or 0),
            }
        )
    dated = [item for item in matches if item["year_start"] is not None]
    selected = min(
        dated,
        key=lambda item: (item["year_start"], item["year_end"], item["index"]),
        default=None,
    )
    if selected is None and matches:
        selected = min(matches, key=lambda item: item["index"])
    audit = {
        "sbn_total_results": int(data.get("total", 0) or 0)
        if isinstance(data, dict)
        else 0,
        "sbn_inspected_results": len(results) if isinstance(results, list) else 0,
        "sbn_direct_match_count": len(matches),
        "sbn_dated_direct_match_count": len(dated),
        "sbn_direct_matches": matches[:10],
    }
    return selected, audit


def extract_sbn_detail(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract stable record fields from an SBN title-detail response."""

    data = payload.get("data", {})
    results = data.get("results", []) if isinstance(data, dict) else []
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("SBN detail response must contain exactly one record")
    result = results[0]
    fields = _sbn_table_fields(result)
    publication = fields.get("Pubblicazione", "")
    years = _publication_years(publication)
    identifier = str(result.get("id", ""))
    bid = identifier.removeprefix("ITICCU")
    return {
        "id": identifier,
        "bid": bid,
        "title": fields.get("Titolo", str(result.get("title", ""))).strip(),
        "author": fields.get("Autore principale", str(result.get("pretitle", ""))).strip(),
        "publication": publication.strip(),
        "year_start": min(years) if years else None,
        "year_end": max(years) if years else None,
        "permalink": f"https://opac.sbn.it/bid/{bid}",
    }


def select_wikidata_candidate(
    row: dict[str, str], payload: dict[str, Any]
) -> dict[str, Any] | None:
    """Choose a direct Wikidata work match; editions and adaptations rank last."""

    expected_title = _normalize(_core_title(row["title"]))
    surname = _primary_author_surname(row["authors"])
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    search = payload.get("search", [])
    for index, candidate in enumerate(search if isinstance(search, list) else []):
        if not isinstance(candidate, dict):
            continue
        label = str(candidate.get("label", ""))
        match_text = str(candidate.get("match", {}).get("text", ""))
        if expected_title not in {_normalize(label), _normalize(match_text)}:
            continue
        description = str(candidate.get("description", ""))
        if surname and surname not in _normalize(description):
            continue
        score = 2
        if _WIKIDATA_WORK_MARKERS.search(description):
            score += 2
        if _WIKIDATA_EDITION_MARKERS.search(description):
            score -= 4
        ranked.append(
            (
                -score,
                index,
                {
                    "id": str(candidate.get("id", "")),
                    "label": label,
                    "description": description,
                    "url": f"https://www.wikidata.org/wiki/{candidate.get('id', '')}",
                },
            )
        )
    return min(ranked, default=(0, 0, None))[2]


def extract_wikidata_p577(payload: dict[str, Any], qid: str) -> dict[str, Any] | None:
    """Extract publication-date claims and expose cross-period claim conflicts."""

    entity = payload.get("entities", {}).get(qid, {})
    claims = entity.get("claims", {}).get("P577", []) if isinstance(entity, dict) else []
    years: list[int] = []
    for claim in claims if isinstance(claims, list) else []:
        value = (
            claim.get("mainsnak", {})
            .get("datavalue", {})
            .get("value", {})
        )
        time_value = str(value.get("time", "")) if isinstance(value, dict) else ""
        match = re.match(r"^[+-](\d{4,})-", time_value)
        if match is not None:
            year = int(match.group(1))
            if 1200 <= year <= 2099:
                years.append(year)
    if not years:
        return None
    buckets = {_period_from_year(year) for year in years}
    return {
        "year_start": min(years),
        "year_end": max(years),
        "claim_years": sorted(set(years)),
        "claim_period_conflict": len(buckets) > 1,
    }


def resolve_authoritative_row(
    row: dict[str, str],
    *,
    evidence: list[dict[str, Any]],
    sbn_detail: dict[str, Any] | None,
    wikidata_detail: dict[str, Any] | None,
    cache_sha256s: list[str],
) -> dict[str, Any]:
    """Resolve one pass-1B row under conservative source and conflict rules."""

    result = dict(row)
    result.update({field: "" for field in AUTHORITATIVE_FIELDS})
    result["resolution_pass"] = "pass_1b"
    result["authoritative_evidence_json"] = json.dumps(
        evidence, ensure_ascii=False, separators=(",", ":")
    )
    result["authoritative_cache_sha256s"] = ";".join(sorted(set(cache_sha256s)))

    if row["ebook_id"] in _LANGUAGE_POLICY:
        policy = _LANGUAGE_POLICY[row["ebook_id"]]
        policy_evidence = _evidence(
            source="project_gutenberg_primary_text",
            record_id=f"pg{row['ebook_id']}",
            url=row["landing_page_url"],
            retrieved_at="2026-08-09",
            payload_sha256="",
            method=policy["method"],
            text=policy["text"],
            confidence="high",
            decisive=True,
            year_start=policy["year_start"],
            year_end=policy["year_end"],
        )
        evidence.append(policy_evidence)
        result["authoritative_evidence_json"] = json.dumps(
            evidence, ensure_ascii=False, separators=(",", ":")
        )
        return _finalize(
            result,
            selected=policy_evidence,
            period=policy["period"],
            role=policy["route"],
            decision=policy["decision"],
            status="pass_1b_policy_routed",
            activation_class=policy["activation_class"],
            exclusion_reason=policy["reason"],
        )

    if row["ebook_id"] in _PRIMARY_TEXT_WORK_POLICY:
        policy = _PRIMARY_TEXT_WORK_POLICY[row["ebook_id"]]
        selected = _evidence(
            source="project_gutenberg_primary_text",
            record_id=f"pg{row['ebook_id']}",
            url=row["landing_page_url"],
            retrieved_at="2026-08-10",
            payload_sha256="",
            method=policy["method"],
            text=policy["text"],
            confidence="high",
            decisive=True,
            year_start=policy["year_start"],
            year_end=policy["year_end"],
        )
        evidence.append(selected)
        result["authoritative_evidence_json"] = json.dumps(
            evidence, ensure_ascii=False, separators=(",", ":")
        )
        return _resolve_year_evidence(result, row, selected)

    primary_text_evidence = _pre_1901_primary_text_evidence(row)
    if primary_text_evidence is not None:
        evidence.append(primary_text_evidence)
        result["authoritative_evidence_json"] = json.dumps(
            evidence, ensure_ascii=False, separators=(",", ":")
        )
        return _resolve_year_evidence(result, row, primary_text_evidence)

    if sbn_detail is None:
        return _documented_exclusion(result, "no_direct_sbn_title_author_match")

    if row["inventory_status"] == "review_translation_edition_date":
        local_year = _translation_edition_year(row)
        if local_year is not None:
            selected = _evidence(
                source="project_gutenberg_primary_text",
                record_id=f"pg{row['ebook_id']}",
                url=row["landing_page_url"],
                retrieved_at="2026-08-09",
                payload_sha256="",
                method="italian_translation_title_page_confirmed_by_sbn",
                text=f"Italian translation edition title-page year: {local_year}",
                confidence="high",
                decisive=True,
                year_start=local_year,
                year_end=local_year,
            )
            evidence.append(selected)
            result["authoritative_evidence_json"] = json.dumps(
                evidence, ensure_ascii=False, separators=(",", ":")
            )
            return _resolve_year_evidence(result, row, selected)
        if sbn_detail.get("year_end") is not None and int(sbn_detail["year_end"]) <= 1900:
            selected = _find_evidence(evidence, "sbn_italian_edition_publication")
            return _resolve_year_evidence(result, row, selected)
        return _documented_exclusion(
            result, "italian_translation_edition_date_not_authoritatively_resolved"
        )

    if wikidata_detail is not None:
        if wikidata_detail.get("claim_period_conflict"):
            return _documented_exclusion(result, "conflicting_wikidata_p577_periods")
        wd_year_start = int(wikidata_detail["year_start"])
        if (
            sbn_detail.get("year_start") is not None
            and int(sbn_detail["year_start"]) <= 1900 < wd_year_start
        ):
            return _documented_exclusion(
                result, "sbn_edition_predates_wikidata_work_date_conflict"
            )
        selected = _find_evidence(evidence, "wikidata_p577_title_author_match")
        return _resolve_year_evidence(result, row, selected)

    if sbn_detail.get("year_start") is not None and int(sbn_detail["year_start"]) <= 1900:
        selected = _find_evidence(evidence, "sbn_edition_publication_upper_bound")
        return _resolve_year_evidence(result, row, selected)

    return _documented_exclusion(
        result, "post_1900_or_undated_sbn_edition_does_not_prove_work_date"
    )


def run_gutenberg_authoritative_review(
    config: GutenbergAuthoritativeReviewConfig,
    *,
    session: requests.Session | None = None,
    fetch_json: FetchJson | None = None,
    progress: Progress | None = None,
    sleep: Sleep = default_sleep,
) -> dict[str, Any]:
    """Resolve all pass-1B rows, merge them with pass 1A, and publish reports."""

    _validate_config(config)
    pass_1a_sha = _validate_frozen_file(
        config.pass_1a_csv_path, config.expected_pass_1a_sha256, "pass 1A"
    )
    manual_sha = _validate_frozen_file(
        config.manual_csv_path, config.expected_manual_sha256, "manual pass-1B input"
    )
    pass_1a = _read_csv(config.pass_1a_csv_path, RESOLUTION_FIELDS)
    manual = _read_csv(config.manual_csv_path, RESOLUTION_FIELDS)
    if len(pass_1a) != config.expected_pass_1a_count:
        raise ValueError("pass-1A record count does not match the frozen contract")
    if len(manual) != config.expected_manual_count:
        raise ValueError("pass-1B record count does not match the frozen contract")
    manual_ids = {row["ebook_id"] for row in manual}
    pass_1a_manual_ids = {
        row["ebook_id"] for row in pass_1a if row["resolution_status"] != "automatic_resolved"
    }
    if manual_ids != pass_1a_manual_ids:
        raise ValueError("pass-1B IDs do not exactly match unresolved pass-1A IDs")

    http = session or requests.Session()
    http.headers["User-Agent"] = (
        "portfolio-transformer-poetry/1.0 "
        "(https://github.com/LeonardoPaccianiMori/portfolio-transformer-poetry; "
        "bibliographic metadata audit)"
    )
    transport = fetch_json or _requests_fetch_json(http)
    retrieved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    last_network_at: list[float | None] = [None]

    def cached_request(
        cache_path: Path,
        *,
        source: str,
        method: str,
        url: str,
        params: dict[str, str],
    ) -> tuple[dict[str, Any], str, str]:
        if not cache_path.is_file() and last_network_at[0] is not None:
            remaining = config.request_delay_seconds - (monotonic() - last_network_at[0])
            if remaining > 0:
                sleep(remaining)
        errors: list[str] = []
        for attempt in range(1, config.fetch_attempts + 1):
            try:
                value = load_or_fetch_authority_json(
                    cache_path,
                    source=source,
                    method=method,
                    url=url,
                    params=params,
                    timeout_seconds=config.request_timeout_seconds,
                    fetch_json=transport,
                    retrieved_at=retrieved_at,
                )
                if value[1] == "downloaded":
                    last_network_at[0] = monotonic()
                return value
            except (requests.RequestException, RuntimeError) as error:
                last_network_at[0] = monotonic()
                errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
                if attempt < config.fetch_attempts:
                    sleep(max(1.0, config.request_delay_seconds) * attempt)
        raise RuntimeError("; ".join(errors))

    started = monotonic()
    pass_1b: dict[str, dict[str, Any]] = {}
    cache_counts: Counter[str] = Counter()
    for index, row in enumerate(sorted(manual, key=lambda item: int(item["ebook_id"])), 1):
        result, row_cache_counts = _resolve_row_from_network(
            row,
            config=config,
            cached_request=cached_request,
            retrieved_at=retrieved_at,
        )
        pass_1b[row["ebook_id"]] = result
        cache_counts.update(row_cache_counts)
        elapsed = monotonic() - started
        eta = elapsed / index * (len(manual) - index)
        _report(
            progress,
            f"record {index:,}/{len(manual):,} ({index / len(manual):.1%}) "
            f"id={row['ebook_id']} decision={result['final_decision']} "
            f"elapsed={_duration(elapsed)} eta={_duration(eta)}",
        )

    final_rows = [
        pass_1b[row["ebook_id"]]
        if row["ebook_id"] in pass_1b
        else _augment_pass_1a_row(row)
        for row in pass_1a
    ]
    if len(final_rows) != config.expected_pass_1a_count:
        raise AssertionError("final Gutenberg accounting lost records")
    exclusions = [
        row for row in final_rows if row["final_activation_class"] != "eligible_probe"
    ]
    _write_csv(config.final_csv_path, final_rows, FINAL_FIELDS)
    _write_csv(config.exclusion_csv_path, exclusions, FINAL_FIELDS)
    report = _build_report(
        config,
        final_rows=final_rows,
        pass_1b_rows=list(pass_1b.values()),
        exclusions=exclusions,
        pass_1a_sha=pass_1a_sha,
        manual_sha=manual_sha,
        cache_counts=cache_counts,
        elapsed=monotonic() - started,
    )
    _write_json(config.json_report_path, report)
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(
        render_gutenberg_authoritative_review_markdown(report), encoding="utf-8"
    )
    return report


def render_gutenberg_authoritative_review_markdown(report: dict[str, Any]) -> str:
    """Render the public pass-1B and final-accounting report."""

    lines = [
        "# Project Gutenberg Authoritative Metadata Resolution Pass 1B",
        "",
        "## Result",
        "",
        (
            f"Audited all {report['pass_1b_record_count']:,} pass-1B holds and "
            f"reconciled {report['final_record_count']:,} frozen Gutenberg records."
        ),
        "",
        "## Final Decisions",
        "",
        "| Decision | Records |",
        "| --- | ---: |",
    ]
    for decision, count in report["final_decision_counts"].items():
        lines.append(f"| `{decision}` | {count:,} |")
    lines.extend(["", "## Activation Classes", "", "| Class | Records |", "| --- | ---: |"])
    for status, count in report["final_activation_class_counts"].items():
        lines.append(f"| `{status}` | {count:,} |")
    lines.extend(
        ["", "## Pass 1B Exclusion Reasons", "", "| Reason | Records |", "| --- | ---: |"]
    )
    for reason, count in report["pass_1b_exclusion_reason_counts"].items():
        lines.append(f"| `{reason}` | {count:,} |")
    lines.extend(
        [
            "",
            "## Evidence Policy",
            "",
            "- SBN/ICCU title-and-author matches are the primary bibliographic identity evidence.",
            "- A matched SBN edition dated by 1900 proves only that the work existed by that date; a later edition does not prove a later first publication.",
            "- Wikidata P577 is accepted only for an exact title match whose description also identifies the expected author, and only after SBN has anchored the work identity.",
            "- Conflicting dates and unresolved identity/date evidence are documented exclusions, never guessed inclusions.",
            "- Record-specific primary-text resolutions are limited to explicit title-page, edition, composition, delivery, performance, or anthology-content dates.",
            "- Translation decisions use the Italian edition date, not the source work's original date.",
            "- Romanesco, Bolognese, and mixed-language records remain outside the standard-Italian core and are routed separately.",
            "- Open Library is not used as deciding evidence.",
            "- This checkpoint authorizes no download, corpus activation, V7 split, or training weight.",
            "",
            "## Artifacts",
            "",
            f"- Final 673-row resolution: `{report['outputs']['final_csv_path']}`",
            f"- Explicit unresolved/exclusion rows: `{report['outputs']['exclusion_csv_path']}`",
            f"- Machine report: `{report['outputs']['json_report_path']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve_row_from_network(
    row: dict[str, str],
    *,
    config: GutenbergAuthoritativeReviewConfig,
    cached_request: Callable[..., tuple[dict[str, Any], str, str]],
    retrieved_at: str,
) -> tuple[dict[str, Any], Counter[str]]:
    evidence: list[dict[str, Any]] = []
    cache_hashes: list[str] = []
    cache_counts: Counter[str] = Counter()
    selected_sbn: dict[str, Any] | None = None
    audit: dict[str, Any] = {}

    core_title = _core_title(row["title"])
    core_suffix = "_normalized" if core_title != _legacy_core_title(row["title"]) else ""
    variants: list[tuple[str, str, bool]] = [(core_suffix, core_title, True)]
    if _core_title(row["title"]) != row["title"].strip():
        variants.append(("_full", row["title"].strip(), True))
    short_title = _short_title(row["title"])
    if short_title not in {title for _, title, _ in variants}:
        variants.append(("_short", short_title, True))
    if _primary_author(row["authors"]):
        variants.append(("_title_only", short_title, False))

    for suffix, query_title, include_author in variants:
        params = build_sbn_search_params(
            row,
            title_override=query_title,
            include_author=include_author,
        )
        payload, cache_status, cache_sha = cached_request(
            config.cache_dir / "sbn/search" / f"pg{row['ebook_id']}{suffix}.json",
            source="sbn_iccu_search",
            method="POST",
            url=SBN_SEARCH_URL,
            params=params,
        )
        cache_counts[f"sbn_search_{cache_status}"] += 1
        cache_hashes.append(cache_sha)
        selected_sbn, audit = select_sbn_candidate(row, payload)
        evidence.append(
            _evidence(
                source="sbn_iccu",
                record_id="",
                url=SBN_SEARCH_URL,
                retrieved_at=_cache_retrieved_at(
                    config.cache_dir / "sbn/search" / f"pg{row['ebook_id']}{suffix}.json"
                ),
                payload_sha256=_sha256_json(payload),
                method=(
                    "sbn_title_author_search_direct_match"
                    if audit["sbn_direct_match_count"]
                    else "sbn_title_author_search_no_direct_match"
                ),
                text=(
                    f"query_title={params['fieldvalue[0]']!r}; "
                    f"total={audit['sbn_total_results']}; "
                    f"inspected={audit['sbn_inspected_results']}; "
                    f"direct_matches={audit['sbn_direct_match_count']}"
                ),
                confidence="high",
                decisive=False,
            )
        )
        if selected_sbn is not None:
            break

    sbn_detail: dict[str, Any] | None = None
    if selected_sbn is not None:
        bid = selected_sbn["bid"]
        payload, cache_status, cache_sha = cached_request(
            config.cache_dir / "sbn/detail" / f"{bid}.json",
            source="sbn_iccu_detail",
            method="GET",
            url=SBN_DETAIL_URL,
            params={"core": "sbn", "id": bid},
        )
        cache_counts[f"sbn_detail_{cache_status}"] += 1
        cache_hashes.append(cache_sha)
        sbn_detail = extract_sbn_detail(payload)
        if not _sbn_detail_matches(row, sbn_detail):
            sbn_detail = None
            evidence.append(
                _evidence(
                    source="sbn_iccu",
                    record_id=bid,
                    url=f"https://opac.sbn.it/bid/{bid}",
                    retrieved_at=_cache_retrieved_at(
                        config.cache_dir / "sbn/detail" / f"{bid}.json"
                    ),
                    payload_sha256=_sha256_json(payload),
                    method="sbn_detail_failed_direct_match",
                    text="The selected search result failed record-level title/author matching.",
                    confidence="high",
                    decisive=False,
                )
            )
        else:
            method = (
                "sbn_italian_edition_publication"
                if row["inventory_status"] == "review_translation_edition_date"
                else "sbn_edition_publication_upper_bound"
            )
            evidence.append(
                _evidence(
                    source="sbn_iccu",
                    record_id=bid,
                    url=sbn_detail["permalink"],
                    retrieved_at=_cache_retrieved_at(
                        config.cache_dir / "sbn/detail" / f"{bid}.json"
                    ),
                    payload_sha256=_sha256_json(payload),
                    method=method,
                    text=(
                        f"{sbn_detail['author']} | {sbn_detail['title']} | "
                        f"{sbn_detail['publication']}"
                    ),
                    confidence="high",
                    decisive=(sbn_detail.get("year_end") is not None),
                    year_start=sbn_detail.get("year_start"),
                    year_end=sbn_detail.get("year_end"),
                )
            )

    wikidata_detail: dict[str, Any] | None = None
    if sbn_detail is not None and row["inventory_status"] != "review_translation_edition_date":
        wikidata_suffix = (
            "_normalized"
            if _core_title(row["title"]) != _legacy_core_title(row["title"])
            else ""
        )
        wikidata_search_cache = (
            config.cache_dir
            / "wikidata/search"
            / f"pg{row['ebook_id']}{wikidata_suffix}.json"
        )
        params = {
            "action": "wbsearchentities",
            "search": _core_title(row["title"]),
            "language": "it",
            "uselang": "it",
            "type": "item",
            "limit": "10",
            "format": "json",
        }
        payload, cache_status, cache_sha = cached_request(
            wikidata_search_cache,
            source="wikidata_search",
            method="GET",
            url=WIKIDATA_SEARCH_URL,
            params=params,
        )
        cache_counts[f"wikidata_search_{cache_status}"] += 1
        cache_hashes.append(cache_sha)
        candidate = select_wikidata_candidate(row, payload)
        if candidate is None:
            evidence.append(
                _evidence(
                    source="wikidata",
                    record_id="",
                    url=WIKIDATA_SEARCH_URL,
                    retrieved_at=_cache_retrieved_at(
                        wikidata_search_cache
                    ),
                    payload_sha256=_sha256_json(payload),
                    method="wikidata_no_direct_title_author_match",
                    text=f"No direct work match for {_core_title(row['title'])!r}.",
                    confidence="high",
                    decisive=False,
                )
            )
        else:
            qid = candidate["id"]
            entity_url = WIKIDATA_ENTITY_URL.format(qid=qid)
            entity_payload, entity_status, entity_sha = cached_request(
                config.cache_dir / "wikidata/entity" / f"{qid}.json",
                source="wikidata_entity",
                method="GET",
                url=entity_url,
                params={},
            )
            cache_counts[f"wikidata_entity_{entity_status}"] += 1
            cache_hashes.append(entity_sha)
            wikidata_detail = extract_wikidata_p577(entity_payload, qid)
            if wikidata_detail is not None:
                evidence.append(
                    _evidence(
                        source="wikidata",
                        record_id=qid,
                        url=candidate["url"],
                        retrieved_at=_cache_retrieved_at(
                            config.cache_dir / "wikidata/entity" / f"{qid}.json"
                        ),
                        payload_sha256=_sha256_json(entity_payload),
                        method="wikidata_p577_title_author_match",
                        text=(
                            f"{candidate['label']} | {candidate['description']} | "
                            f"P577={wikidata_detail['claim_years']}"
                        ),
                        confidence="medium",
                        decisive=not wikidata_detail["claim_period_conflict"],
                        year_start=wikidata_detail["year_start"],
                        year_end=wikidata_detail["year_end"],
                    )
                )
            else:
                evidence.append(
                    _evidence(
                        source="wikidata",
                        record_id=qid,
                        url=candidate["url"],
                        retrieved_at=_cache_retrieved_at(
                            config.cache_dir / "wikidata/entity" / f"{qid}.json"
                        ),
                        payload_sha256=_sha256_json(entity_payload),
                        method="wikidata_direct_match_without_p577",
                        text=f"{candidate['label']} | {candidate['description']}",
                        confidence="medium",
                        decisive=False,
                    )
                )

    return (
        resolve_authoritative_row(
            row,
            evidence=evidence,
            sbn_detail=sbn_detail,
            wikidata_detail=wikidata_detail,
            cache_sha256s=cache_hashes,
        ),
        cache_counts,
    )


def _resolve_year_evidence(
    result: dict[str, Any], row: dict[str, str], selected: dict[str, Any] | None
) -> dict[str, Any]:
    if selected is None or selected.get("year_end") is None:
        return _documented_exclusion(result, "decisive_year_evidence_missing")
    decision_year = int(selected["year_end"])
    if selected["method"] in {
        "sbn_edition_publication_upper_bound",
        "project_gutenberg_title_page_edition_upper_bound",
    }:
        decision_year = int(selected["year_start"])
    period = _period_from_year(decision_year)
    role = _role_for_period(row["preliminary_role"], period)
    if period == "post_1900_excluded":
        return _finalize(
            result,
            selected=selected,
            period=period,
            role=role,
            decision="exclude_post_1900_original_text",
            status="pass_1b_authoritative_resolved",
            activation_class="excluded",
            exclusion_reason="authoritative_date_after_1900",
        )
    decision = (
        "eligible_historical_core_candidate"
        if period == "origins_through_1800"
        else "eligible_nineteenth_century_candidate"
    )
    return _finalize(
        result,
        selected=selected,
        period=period,
        role=role,
        decision=decision,
        status="pass_1b_authoritative_resolved",
        activation_class="eligible_probe",
        exclusion_reason="",
    )


def _documented_exclusion(result: dict[str, Any], reason: str) -> dict[str, Any]:
    return _finalize(
        result,
        selected=None,
        period="unresolved_excluded",
        role="excluded_unresolved_authoritative_metadata",
        decision="exclude_unresolved_authoritative_metadata",
        status="pass_1b_documented_exclusion",
        activation_class="excluded",
        exclusion_reason=reason,
    )


def _finalize(
    result: dict[str, Any],
    *,
    selected: dict[str, Any] | None,
    period: str,
    role: str,
    decision: str,
    status: str,
    activation_class: str,
    exclusion_reason: str,
) -> dict[str, Any]:
    if selected is not None:
        result.update(
            {
                "authoritative_selected_source": selected["source"],
                "authoritative_record_id": selected["record_id"],
                "authoritative_evidence_url": selected["url"],
                "authoritative_evidence_year_start": selected.get("year_start", "") or "",
                "authoritative_evidence_year_end": selected.get("year_end", "") or "",
                "authoritative_evidence_text": selected["text"],
                "authoritative_method": selected["method"],
                "authoritative_confidence": selected["confidence"],
                "authoritative_retrieved_at": selected["retrieved_at"],
            }
        )
    result.update(
        {
            "final_period_bucket": period,
            "final_role": role,
            "final_decision": decision,
            "final_resolution_status": status,
            "final_activation_class": activation_class,
            "final_exclusion_reason": exclusion_reason,
        }
    )
    return result


def _augment_pass_1a_row(row: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = dict(row)
    result.update({field: "" for field in AUTHORITATIVE_FIELDS})
    excluded = row["automatic_decision"] == "exclude_post_1900_original_text"
    result.update(
        {
            "resolution_pass": "pass_1a",
            "final_period_bucket": row["resolved_period_bucket"],
            "final_role": row["resolved_role"],
            "final_decision": row["automatic_decision"],
            "final_resolution_status": "pass_1a_resolved",
            "final_activation_class": "excluded" if excluded else "eligible_probe",
            "final_exclusion_reason": "pass_1a_post_1900_evidence" if excluded else "",
        }
    )
    return result


def _evidence(
    *,
    source: str,
    record_id: str,
    url: str,
    retrieved_at: str,
    payload_sha256: str,
    method: str,
    text: str,
    confidence: str,
    decisive: bool,
    year_start: int | None = None,
    year_end: int | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "record_id": record_id,
        "url": url,
        "retrieved_at": retrieved_at,
        "payload_sha256": payload_sha256,
        "method": method,
        "year_start": year_start,
        "year_end": year_end,
        "text": text[:1000],
        "confidence": confidence,
        "decisive": decisive,
    }


def _find_evidence(
    evidence: list[dict[str, Any]], method: str
) -> dict[str, Any] | None:
    return next((item for item in evidence if item["method"] == method), None)


def _translation_edition_year(row: dict[str, str]) -> int | None:
    try:
        evidence = json.loads(row["date_evidence_json"] or "[]")
    except json.JSONDecodeError:
        return None
    candidates = [
        int(item["year_start"])
        for item in evidence
        if item.get("kind") in {
            "explicit_first_italian_version",
            "title_page_edition_year",
            "copyright_year",
        }
        and item.get("year_start")
    ]
    return min(candidates) if candidates else None


def _pre_1901_primary_text_evidence(
    row: dict[str, str]
) -> dict[str, Any] | None:
    if row["inventory_status"] == "review_translation_edition_date":
        return None
    try:
        items = json.loads(row["date_evidence_json"] or "[]")
    except json.JSONDecodeError:
        return None
    candidates = [
        item
        for item in items
        if item.get("kind") in {"title_page_edition_year", "copyright_year"}
        and item.get("year_start")
        and int(item["year_start"]) <= 1900
    ]
    if not candidates:
        return None
    selected = min(candidates, key=lambda item: int(item["year_start"]))
    year = int(selected["year_start"])
    return _evidence(
        source="project_gutenberg_primary_text",
        record_id=f"pg{row['ebook_id']}",
        url=row["landing_page_url"],
        retrieved_at="2026-08-09",
        payload_sha256="",
        method="project_gutenberg_title_page_edition_upper_bound",
        text=str(selected.get("text", "")),
        confidence="medium",
        decisive=True,
        year_start=year,
        year_end=year,
    )


def _sbn_result_matches(row: dict[str, str], result: dict[str, Any]) -> bool:
    title = str(result.get("title", {}).get("info", ""))
    author = " ".join(
        (
            str(result.get("title", {}).get("text", "")),
            title,
        )
    )
    return _title_matches(row["title"], title) and _author_matches(row["authors"], author)


def _sbn_detail_matches(row: dict[str, str], detail: dict[str, Any]) -> bool:
    return _title_matches(row["title"], detail["title"]) and _author_matches(
        row["authors"], " ".join((detail["author"], detail["title"]))
    )


def _title_matches(expected: str, observed: str) -> bool:
    expected_normalized = _normalize(_core_title(expected))
    observed_normalized = _normalize(_VOLUME_PREFIX.sub("", observed))
    if not expected_normalized or not observed_normalized:
        return False
    if expected_normalized in observed_normalized:
        return True
    return SequenceMatcher(None, expected_normalized, observed_normalized).ratio() >= 0.84


def _author_matches(expected: str, observed: str) -> bool:
    surname = _primary_author_surname(expected)
    return not surname or surname in _normalize(observed)


def _primary_author(value: str) -> str:
    return value.split(";", 1)[0].strip()


def _primary_author_surname(value: str) -> str:
    author = _primary_author(value)
    if not author:
        return ""
    surname = author.split(",", 1)[0]
    return _normalize(surname)


def _core_title(value: str) -> str:
    title = _VOLUME_PREFIX.sub("", value.strip())
    first = title.split(":", 1)[0].strip()
    return _VOLUME_SUFFIX.sub("", first if first else title).strip(" ,.-")


def _legacy_core_title(value: str) -> str:
    title = _VOLUME_PREFIX.sub("", value.strip())
    first = title.split(":", 1)[0].strip()
    return first if first else title


def _short_title(value: str) -> str:
    title = _core_title(value).split(";", 1)[0].strip()
    words = title.split()
    return " ".join(words[:10])


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    ascii_value = "".join(character for character in decomposed if not unicodedata.combining(character))
    return " ".join(_NON_ALNUM.sub(" ", ascii_value).split())


def _publication_years(value: str) -> list[int]:
    return sorted({int(match.group(1)) for match in _YEAR.finditer(value)})


def _period_from_year(year: int) -> str:
    if year <= 1800:
        return "origins_through_1800"
    if year <= 1900:
        return "nineteenth_century_bridge"
    return "post_1900_excluded"


def _role_for_period(preliminary_role: str, period: str) -> str:
    if period == "post_1900_excluded":
        return "excluded_post_1900_original_text"
    if preliminary_role == "sonnet_specialization_candidate":
        return "sonnet_specialization_candidate"
    if preliminary_role == "historical_non_sonnet_poetry_candidate":
        return "historical_non_sonnet_poetry_candidate"
    if period == "nineteenth_century_bridge":
        return "nineteenth_century_bridge_candidate"
    return "historical_general_candidate"


def _sbn_table_fields(result: dict[str, Any]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for content in result.get("contents", []):
        if content.get("type") != "table":
            continue
        for row in content.get("body", []):
            if not isinstance(row, list) or len(row) < 2:
                continue
            label = str(row[0].get("value", "")).strip()
            value = _flatten_sbn_content(row[1]).strip()
            if label and value:
                fields[label] = value
    return fields


def _flatten_sbn_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_flatten_sbn_content(item) for item in value)
    if not isinstance(value, dict):
        return ""
    parts = []
    for key in ("value", "label", "text"):
        if key in value:
            parts.append(_flatten_sbn_content(value[key]))
    for key in ("contents", "values"):
        if key in value:
            parts.append(_flatten_sbn_content(value[key]))
    return " ".join(part for part in parts if part)


def _validate_cache_envelope(
    envelope: dict[str, Any], *, source: str, request: dict[str, Any]
) -> None:
    if envelope.get("schema_version") != 1:
        raise ValueError("authority cache schema version mismatch")
    if envelope.get("source") != source:
        raise ValueError("authority cache source mismatch")
    if envelope.get("request") != request:
        raise ValueError("authority cache request pin mismatch")
    if not isinstance(envelope.get("payload"), dict):
        raise ValueError("authority cache payload must be an object")
    actual = _sha256_json(envelope["payload"])
    if envelope.get("payload_sha256") != actual:
        raise ValueError("authority cache payload SHA-256 mismatch")
    if not envelope.get("retrieved_at"):
        raise ValueError("authority cache retrieval timestamp is missing")


def _cache_retrieved_at(path: Path) -> str:
    return str(json.loads(path.read_text(encoding="utf-8"))["retrieved_at"])


def _requests_fetch_json(session: requests.Session) -> FetchJson:
    def fetch(method: str, url: str, params: dict[str, str], timeout: float) -> dict[str, Any]:
        if method == "POST":
            response = session.post(url, data=params, timeout=timeout)
        else:
            response = session.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("status") == "error":
            raise RuntimeError(f"authority API error: {payload}")
        return payload

    return fetch


def _validate_config(config: GutenbergAuthoritativeReviewConfig) -> None:
    if config.request_delay_seconds < 0:
        raise ValueError("request_delay_seconds cannot be negative")
    if config.request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be positive")
    if config.fetch_attempts <= 0:
        raise ValueError("fetch_attempts must be positive")


def _validate_frozen_file(path: Path, expected_sha: str, label: str) -> str:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha:
        raise ValueError(f"{label} SHA-256 mismatch: expected={expected_sha} actual={actual}")
    return actual


def _read_csv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f"CSV schema mismatch: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    if len({row["ebook_id"] for row in rows}) != len(rows):
        raise ValueError(f"CSV contains duplicate ebook IDs: {path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _build_report(
    config: GutenbergAuthoritativeReviewConfig,
    *,
    final_rows: list[dict[str, Any]],
    pass_1b_rows: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    pass_1a_sha: str,
    manual_sha: str,
    cache_counts: Counter[str],
    elapsed: float,
) -> dict[str, Any]:
    return {
        "resolution_version": "project_gutenberg_authoritative_resolution_v1",
        "pass_1b_record_count": len(pass_1b_rows),
        "final_record_count": len(final_rows),
        "explicit_unresolved_or_exclusion_count": len(exclusions),
        "final_decision_counts": dict(
            sorted(Counter(row["final_decision"] for row in final_rows).items())
        ),
        "final_resolution_status_counts": dict(
            sorted(Counter(row["final_resolution_status"] for row in final_rows).items())
        ),
        "final_activation_class_counts": dict(
            sorted(Counter(row["final_activation_class"] for row in final_rows).items())
        ),
        "final_role_counts": dict(
            sorted(Counter(row["final_role"] for row in final_rows).items())
        ),
        "pass_1b_exclusion_reason_counts": dict(
            sorted(
                Counter(
                    row["final_exclusion_reason"]
                    for row in pass_1b_rows
                    if row["final_exclusion_reason"]
                ).items()
            )
        ),
        "cache_status_counts": dict(sorted(cache_counts.items())),
        "elapsed_seconds": round(elapsed, 3),
        "outputs": {
            "pass_1a_csv_path": _portable(config.pass_1a_csv_path, config.repo_root),
            "pass_1a_csv_sha256": pass_1a_sha,
            "manual_csv_path": _portable(config.manual_csv_path, config.repo_root),
            "manual_csv_sha256": manual_sha,
            "final_csv_path": _portable(config.final_csv_path, config.repo_root),
            "final_csv_sha256": _sha256_file(config.final_csv_path),
            "exclusion_csv_path": _portable(config.exclusion_csv_path, config.repo_root),
            "exclusion_csv_sha256": _sha256_file(config.exclusion_csv_path),
            "json_report_path": _portable(config.json_report_path, config.repo_root),
            "markdown_report_path": _portable(config.markdown_report_path, config.repo_root),
        },
        "policy": {
            "activation_authorized": False,
            "sbn_iccu_is_primary_authority": True,
            "wikidata_is_corroborating_only": True,
            "open_library_is_deciding_evidence": False,
            "post_1900_sbn_reprint_proves_work_date": False,
            "record_specific_primary_text_resolution_count": len(
                _PRIMARY_TEXT_WORK_POLICY
            ),
            "conflicts_and_unresolved_rows_are_excluded": True,
            "translations_use_italian_edition_period": True,
            "language_varieties_are_separate_from_standard_core": True,
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _report(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)
