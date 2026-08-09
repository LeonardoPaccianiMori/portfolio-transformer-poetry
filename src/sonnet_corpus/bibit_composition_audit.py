"""Metadata-first composition gate for the Biblioteca Italiana collection."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from time import monotonic
from typing import Any

from .biblioteca_italiana import (
    BIBIT_API_URL,
    BIBIT_FAQ_URL,
    BIBIT_LANDING_URL,
    BIBIT_PROJECT_URL,
    BibItCatalogRecord,
    fetch_bibit_catalog,
    fetch_bibit_rendered_texts,
)


HISTORICAL_PERIODS = ("Origini", "200", "300", "400", "500", "600", "700")
BRIDGE_PERIOD = "800"
PERIOD_LABELS = {
    "Origini": "Origini",
    "200": "Duecento",
    "300": "Trecento",
    "400": "Quattrocento",
    "500": "Cinquecento",
    "600": "Seicento",
    "700": "Settecento",
    "800": "Ottocento",
}
ROLE_HISTORICAL_GENERAL = "historical_general"
ROLE_HISTORICAL_POETRY = "historical_non_sonnet_poetry"
ROLE_SONNET_ONLY = "sonnet_only"
ROLE_BRIDGE = "nineteenth_century_bridge"
ROLE_EXCLUDED = "excluded"
ALLOWED_ROLES = {
    ROLE_HISTORICAL_GENERAL,
    ROLE_HISTORICAL_POETRY,
    ROLE_SONNET_ONLY,
    ROLE_BRIDGE,
    ROLE_EXCLUDED,
}

CANONICAL_OVERRIDES = {
    ("ariosto ludovico", "orlando furioso"): "bibit001135",
    ("manzoni alessandro", "promessi sposi"): "bibit000666",
}
CANONICAL_OVERRIDE_REASONS = {
    "bibit001135": "project override selects Ariosto's final 1532 authorial edition",
    "bibit000666": "project override selects Manzoni's later standard text over the 1827 redaction",
}
REPRESENTATIVE_INSPECTION_IDS = (
    "bibit000019",  # Dante, Commedia
    "bibit000049",  # Boiardo, Orlando innamorato
    "bibit001135",  # Ariosto, Orlando Furioso 1532
    "bibit001501",  # Tasso, Gerusalemme liberata
    "bibit000099",  # Tasso, Rime
    "bibit000760",  # Petrarch, Canzoniere
    "bibit000666",  # Manzoni, Promessi Sposi (later standard text)
    "bibit001238",  # Nievo, Confessioni di un Italiano
    "bibit001705",  # Leopardi, Zibaldone di pensieri
)

_POTENTIAL_SONNET_COLLECTION = re.compile(
    r"\b(?:canzoniere|liriche|poesie|rime|sonetti?)\b",
    re.IGNORECASE,
)
_DIALECT_TITLE = re.compile(
    r"(?:\bdialett\w*\b|\bsonetti romaneschi\b|"
    r"\bantichi testi siciliani in volgare\b|"
    r"\b(?:esopo|tristano|milione) veneto\b)",
    re.IGNORECASE,
)
_DIALECT_AUTHORS = {
    "belli giuseppe gioachino",
    "porta carlo",
}
_YEAR_SUFFIX = re.compile(r"\b(?:1[2-9][0-9]{2}|20[0-9]{2})\b")
_NON_WORD = re.compile(r"[^a-z0-9]+")

DECISION_FIELDS = (
    "object_id",
    "wordpress_id",
    "title",
    "authors",
    "periods",
    "genres",
    "languages",
    "source_publisher",
    "source_publication_place",
    "source_publication_date",
    "source_identifier",
    "family_key",
    "canonical_status",
    "canonical_object_id",
    "role",
    "role_reason",
    "estimated_characters",
    "estimated_min_tokens",
    "estimated_max_tokens",
    "landing_page_url",
    "xml_url",
)


@dataclass(frozen=True)
class BibItCompositionAuditConfig:
    """Artifact locations and bounded sampling settings for the audit."""

    repo_root: Path
    catalog_snapshot_path: Path
    decision_csv_path: Path
    json_report_path: Path
    markdown_report_path: Path
    sample_per_stratum: int = 3
    request_timeout_seconds: float = 300.0
    rendered_batch_size: int = 100
    characters_per_token_min: float = 3.0
    characters_per_token_max: float = 4.0
    bridge_share_recommendation: float = 0.10


CatalogFetcher = Callable[..., list[BibItCatalogRecord]]
RenderedTextFetcher = Callable[..., dict[str, str]]
Progress = Callable[[str], None]


def audit_bibit_composition(
    config: BibItCompositionAuditConfig,
    *,
    fetch_catalog: CatalogFetcher = fetch_bibit_catalog,
    fetch_rendered_texts: RenderedTextFetcher = fetch_bibit_rendered_texts,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Audit catalog composition and write public metadata-only decisions."""

    _validate_config(config)
    started = monotonic()
    started_at = _utc_now()
    _report(progress, "stage 1/5: fetching origins-through-Ottocento catalog")
    records = fetch_catalog(
        periods=(*HISTORICAL_PERIODS, BRIDGE_PERIOD),
        timeout=config.request_timeout_seconds,
        progress=lambda message: _report(progress, f"catalog | {message}"),
    )
    _validate_catalog(records)
    historical_count = sum(_primary_period(record) in HISTORICAL_PERIODS for record in records)
    bridge_count = sum(_primary_period(record) == BRIDGE_PERIOD for record in records)
    _report(
        progress,
        f"catalog complete records={len(records):,} historical={historical_count:,} "
        f"bridge_candidates={bridge_count:,} elapsed={_elapsed(started)}",
    )

    _report(progress, "stage 2/5: selecting deterministic composition sample")
    estimation_ids, inspection_ids = select_composition_sample(
        records,
        sample_per_stratum=config.sample_per_stratum,
    )
    sample_ids = tuple(dict.fromkeys((*estimation_ids, *inspection_ids)))
    _report(
        progress,
        f"sample selected estimation={len(estimation_ids):,} "
        f"inspection_union={len(sample_ids):,}",
    )

    _report(progress, "stage 3/5: fetching rendered text for size estimation")
    rendered_html = fetch_rendered_texts(
        sample_ids,
        timeout=config.request_timeout_seconds,
        batch_size=config.rendered_batch_size,
        progress=lambda message: _report(progress, f"sample | {message}"),
    )
    rendered_characters = {
        object_id: rendered_primary_text_character_count(html)
        for object_id, html in rendered_html.items()
    }
    empty_rendered_ids = {
        object_id for object_id, count in rendered_characters.items() if count <= 0
    }
    if len(empty_rendered_ids) == len(rendered_characters):
        raise ValueError("every BibIt rendered sample is empty after cleaning")
    if empty_rendered_ids:
        _report(
            progress,
            "sample records with no rendered primary text will be excluded: "
            + ", ".join(sorted(empty_rendered_ids)),
        )
    _report(
        progress,
        f"rendered sample complete characters={sum(rendered_characters.values()):,} "
        f"elapsed={_elapsed(started)}",
    )

    _report(progress, "stage 4/5: resolving editions, roles, and projected shares")
    canonical_by_family, duplicate_families = resolve_canonical_editions(records)
    estimates = estimate_record_characters(
        records,
        estimation_ids=estimation_ids,
        rendered_characters=rendered_characters,
    )
    decisions = build_composition_decisions(
        records,
        canonical_by_family=canonical_by_family,
        estimated_characters=estimates,
        forced_exclusions=empty_rendered_ids,
        characters_per_token_min=config.characters_per_token_min,
        characters_per_token_max=config.characters_per_token_max,
    )
    report = build_composition_report(
        config=config,
        records=records,
        decisions=decisions,
        duplicate_families=duplicate_families,
        estimation_ids=estimation_ids,
        inspection_ids=inspection_ids,
        rendered_characters=rendered_characters,
        empty_rendered_ids=empty_rendered_ids,
        started_at=started_at,
        finished_at=_utc_now(),
    )

    _report(progress, "stage 5/5: writing public audit artifacts")
    write_catalog_snapshot(records, config.catalog_snapshot_path, config.repo_root)
    write_decision_csv(decisions, config.decision_csv_path)
    _write_json(config.json_report_path, report)
    _write_text(config.markdown_report_path, render_composition_markdown(report))
    _report(
        progress,
        "audit artifacts complete "
        f"records={len(records):,} estimated_characters={report['estimated_total_characters']:,} "
        f"elapsed={_elapsed(started)}",
    )
    return report


def select_composition_sample(
    records: Iterable[BibItCatalogRecord],
    *,
    sample_per_stratum: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select a deterministic author/title-balanced sample per period and genre."""

    if sample_per_stratum <= 0:
        raise ValueError("sample_per_stratum must be greater than zero")
    records_list = list(records)
    by_stratum: dict[tuple[str, str], list[BibItCatalogRecord]] = defaultdict(list)
    for record in records_list:
        by_stratum[_stratum(record)].append(record)

    estimation_ids: list[str] = []
    for stratum in sorted(by_stratum):
        candidates = sorted(
            by_stratum[stratum],
            key=lambda record: _sample_key(record.object_id),
        )
        estimation_ids.extend(
            record.object_id for record in candidates[:sample_per_stratum]
        )
    estimation_ids = list(dict.fromkeys(estimation_ids))
    available_ids = {record.object_id for record in records_list}
    inspection_ids = [
        object_id for object_id in REPRESENTATIVE_INSPECTION_IDS if object_id in available_ids
    ]
    return tuple(estimation_ids), tuple(inspection_ids)


def rendered_primary_text_character_count(html: str) -> int:
    """Count visible literary characters after generated HTML wrappers are removed."""

    parser = _BibItRenderedTextParser()
    parser.feed(html)
    parser.close()
    text = " ".join("".join(parser.text_parts).split())
    return len(text)


def resolve_canonical_editions(
    records: Iterable[BibItCatalogRecord],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Select one deterministic candidate per normalized author/work family."""

    by_family: dict[str, list[BibItCatalogRecord]] = defaultdict(list)
    for record in records:
        by_family[work_family_key(record)].append(record)

    canonical_by_family: dict[str, str] = {}
    duplicate_families: list[dict[str, Any]] = []
    for family_key, family_records in sorted(by_family.items()):
        override_key = (_primary_author_key(family_records[0]), normalized_work_title(family_records[0].title))
        override = CANONICAL_OVERRIDES.get(override_key)
        if override and any(record.object_id == override for record in family_records):
            selected = next(record for record in family_records if record.object_id == override)
            reason = CANONICAL_OVERRIDE_REASONS[override]
        else:
            selected = max(family_records, key=_canonical_record_score)
            reason = "metadata-completeness tie-breaker; verify before activation"
        canonical_by_family[family_key] = selected.object_id
        if len(family_records) > 1:
            duplicate_families.append({
                "family_key": family_key,
                "author": " | ".join(selected.authors),
                "normalized_title": normalized_work_title(selected.title),
                "record_count": len(family_records),
                "canonical_object_id": selected.object_id,
                "canonical_title": selected.title,
                "selection_reason": reason,
                "alternates": [
                    {
                        "object_id": record.object_id,
                        "title": record.title,
                        "source_publication_date": record.source_publication_date,
                        "source_identifier": record.source_identifier,
                    }
                    for record in sorted(family_records, key=lambda item: item.object_id)
                    if record.object_id != selected.object_id
                ],
            })
    duplicate_families.sort(key=lambda row: (-row["record_count"], row["family_key"]))
    return canonical_by_family, duplicate_families


def estimate_record_characters(
    records: Iterable[BibItCatalogRecord],
    *,
    estimation_ids: Iterable[str],
    rendered_characters: dict[str, int],
) -> dict[str, int]:
    """Project per-record sizes from deterministic period/genre strata."""

    records_list = list(records)
    records_by_id = {record.object_id: record for record in records_list}
    sample_by_stratum: dict[tuple[str, str], list[int]] = defaultdict(list)
    for object_id in estimation_ids:
        record = records_by_id[object_id]
        character_count = rendered_characters[object_id]
        if character_count > 0:
            sample_by_stratum[_stratum(record)].append(character_count)
    if not sample_by_stratum:
        raise ValueError("BibIt size estimation sample is empty")

    global_median = int(statistics.median(
        count for count in rendered_characters.values() if count > 0
    ))
    estimates: dict[str, int] = {}
    for record in records_list:
        measured = rendered_characters.get(record.object_id, 0)
        if measured > 0:
            estimates[record.object_id] = measured
            continue
        values = sample_by_stratum.get(_stratum(record))
        estimate = statistics.mean(values) if values else global_median
        estimates[record.object_id] = max(1, round(estimate))
    return estimates


def build_composition_decisions(
    records: Iterable[BibItCatalogRecord],
    *,
    canonical_by_family: dict[str, str],
    estimated_characters: dict[str, int],
    forced_exclusions: set[str] | None = None,
    characters_per_token_min: float,
    characters_per_token_max: float,
) -> list[dict[str, Any]]:
    """Assign one explicit training role to every catalog record."""

    decisions: list[dict[str, Any]] = []
    excluded_ids = forced_exclusions or set()
    for record in records:
        family_key = work_family_key(record)
        canonical_id = canonical_by_family[family_key]
        if record.object_id in excluded_ids:
            role, reason = ROLE_EXCLUDED, "rendered catalog sample contains no primary text"
        else:
            role, reason = classify_record_role(
                record,
                is_canonical=record.object_id == canonical_id,
            )
        if role not in ALLOWED_ROLES:
            raise AssertionError(f"invalid BibIt role: {role}")
        character_count = estimated_characters[record.object_id]
        decisions.append({
            "object_id": record.object_id,
            "wordpress_id": record.wordpress_id,
            "title": record.title,
            "authors": " | ".join(record.authors),
            "periods": " | ".join(record.periods),
            "genres": " | ".join(record.genres),
            "languages": " | ".join(record.languages),
            "source_publisher": record.source_publisher,
            "source_publication_place": record.source_publication_place,
            "source_publication_date": record.source_publication_date,
            "source_identifier": record.source_identifier,
            "family_key": family_key,
            "canonical_status": "selected" if record.object_id == canonical_id else "alternate",
            "canonical_object_id": canonical_id,
            "role": role,
            "role_reason": reason,
            "estimated_characters": character_count,
            "estimated_min_tokens": math.floor(character_count / characters_per_token_max),
            "estimated_max_tokens": math.ceil(character_count / characters_per_token_min),
            "landing_page_url": record.landing_page_url,
            "xml_url": record.xml_url,
        })
    return sorted(decisions, key=lambda row: (row["role"], row["authors"], row["title"], row["object_id"]))


def classify_record_role(
    record: BibItCatalogRecord,
    *,
    is_canonical: bool,
) -> tuple[str, str]:
    """Classify one record under the approved staged-curriculum policy."""

    if not is_canonical:
        return ROLE_EXCLUDED, "alternate edition; retain exactly one canonical record per work"
    normalized_authors = {_normalize_key(author) for author in record.authors}
    if normalized_authors & _DIALECT_AUTHORS or _DIALECT_TITLE.search(record.title):
        return ROLE_EXCLUDED, "dialect-heavy source; keep outside the standard-Italian core"
    if _POTENTIAL_SONNET_COLLECTION.search(record.title):
        return (
            ROLE_SONNET_ONLY,
            "potential sonnet or mixed-lyric collection; route only explicit verified sonnet units to SFT",
        )
    period = _primary_period(record)
    if period == BRIDGE_PERIOD:
        return (
            ROLE_BRIDGE,
            "selected standard-literary Ottocento bridge candidate; final mixture share must be capped",
        )
    if _primary_genre(record).casefold() == "poesia":
        return (
            ROLE_HISTORICAL_POETRY,
            "historical poetry candidate; TEI audit must confirm that explicit sonnets are segmented out",
        )
    return (
        ROLE_HISTORICAL_GENERAL,
        "historical prose, theatre, letters, treatise, document, or related general-domain text",
    )


def build_composition_report(
    *,
    config: BibItCompositionAuditConfig,
    records: list[BibItCatalogRecord],
    decisions: list[dict[str, Any]],
    duplicate_families: list[dict[str, Any]],
    estimation_ids: tuple[str, ...],
    inspection_ids: tuple[str, ...],
    rendered_characters: dict[str, int],
    empty_rendered_ids: set[str],
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    """Build the compact machine-readable composition report."""

    role_counts = Counter(row["role"] for row in decisions)
    role_characters = Counter()
    for row in decisions:
        role_characters[row["role"]] += row["estimated_characters"]
    active_characters = sum(
        count for role, count in role_characters.items() if role != ROLE_EXCLUDED
    )
    role_projection = {
        role: {
            "record_count": role_counts.get(role, 0),
            "estimated_characters": role_characters.get(role, 0),
            "estimated_active_share": (
                role_characters.get(role, 0) / active_characters
                if active_characters and role != ROLE_EXCLUDED
                else 0.0
            ),
        }
        for role in sorted(ALLOWED_ROLES)
    }
    historical_decisions = [
        row for row in decisions if row["role"] not in {ROLE_BRIDGE, ROLE_EXCLUDED}
    ]
    historical_characters = sum(row["estimated_characters"] for row in historical_decisions)
    estimated_min_tokens = math.floor(
        historical_characters / config.characters_per_token_max
    )
    estimated_max_tokens = math.ceil(
        historical_characters / config.characters_per_token_min
    )
    period_counts = Counter(_primary_period(record) for record in records)
    genre_counts = Counter(_primary_genre(record) for record in records)
    author_counts = Counter(
        author for record in records for author in (record.authors or ("(missing)",))
    )
    records_by_id = {record.object_id: record for record in records}
    report = {
        "audit_version": "bibit_historical_composition_audit_v1",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "source_archive": "Biblioteca Italiana / BibIt",
        "source_landing_page": BIBIT_LANDING_URL,
        "source_project_page": BIBIT_PROJECT_URL,
        "source_faq": BIBIT_FAQ_URL,
        "source_api": BIBIT_API_URL,
        "reuse_terms": {
            "status": "copyrighted digital editions permitted for personal/scientific non-commercial use",
            "public_reuse_requirement": "cite Biblioteca Italiana as the source",
            "commercial_reuse": "prohibited",
            "downstream_decision": (
                "non-commercial model/data lineage; retain work-level edition and contributor attribution"
            ),
        },
        "catalog_record_count": len(records),
        "historical_catalog_record_count": sum(
            _primary_period(record) in HISTORICAL_PERIODS for record in records
        ),
        "nineteenth_century_candidate_count": sum(
            _primary_period(record) == BRIDGE_PERIOD for record in records
        ),
        "period_counts": _ordered_counts(period_counts, PERIOD_LABELS),
        "genre_counts": dict(sorted(genre_counts.items(), key=lambda item: (-item[1], item[0]))),
        "largest_authors": [
            {"author": author, "record_count": count, "catalog_share": count / len(records)}
            for author, count in author_counts.most_common(20)
        ],
        "stratum_count": len({_stratum(record) for record in records}),
        "estimation_sample_count": len(estimation_ids),
        "inspection_sample_count": len(set((*estimation_ids, *inspection_ids))),
        "estimation_sample_rendered_characters": sum(
            rendered_characters[object_id] for object_id in estimation_ids
        ),
        "empty_rendered_sample_ids": sorted(empty_rendered_ids),
        "representative_inspections": [
            {
                "object_id": object_id,
                "title": records_by_id[object_id].title,
                "authors": list(records_by_id[object_id].authors),
                "rendered_characters": rendered_characters[object_id],
            }
            for object_id in inspection_ids
        ],
        "estimated_total_characters": sum(
            row["estimated_characters"] for row in decisions
        ),
        "estimated_historical_active_characters": historical_characters,
        "estimated_historical_active_min_tokens": estimated_min_tokens,
        "estimated_historical_active_max_tokens": estimated_max_tokens,
        "token_estimation_assumption": {
            "minimum_characters_per_token": config.characters_per_token_min,
            "maximum_characters_per_token": config.characters_per_token_max,
        },
        "role_projection": role_projection,
        "duplicate_edition_family_count": len(duplicate_families),
        "duplicate_edition_families": duplicate_families,
        "bridge_share_recommendation": config.bridge_share_recommendation,
        "bridge_policy_status": "recommendation_pending_final_training-mixture freeze",
        "corpus_activation_status": "composition_gate_passed_audit_required_before_activation",
        "curriculum_decision": [
            "Stage 1: historical prose/general adaptation with limited preservation replay.",
            "Stage 2: historical non-sonnet poetry adaptation with prose and preservation replay.",
            "Stage 3: low-learning-rate sonnet specialization with author/epoch-balanced sampling.",
        ],
        "leakage_rule": (
            "Explicit sonnets and mixed collections that may contain sonnets are excluded from "
            "historical pretraining; held-out validation/test sonnets must be absent from every earlier stage."
        ),
        "artifacts": {
            "catalog_snapshot": _portable_path(config.catalog_snapshot_path, config.repo_root),
            "decision_csv": _portable_path(config.decision_csv_path, config.repo_root),
            "json_report": _portable_path(config.json_report_path, config.repo_root),
            "markdown_report": _portable_path(config.markdown_report_path, config.repo_root),
        },
    }
    return report


def write_catalog_snapshot(
    records: Iterable[BibItCatalogRecord],
    path: Path,
    repo_root: Path,
) -> None:
    payload = {
        "snapshot_version": "bibit_catalog_origins_through_ottocento_v1",
        "downloaded_at_utc": _utc_now(),
        "source_archive": "Biblioteca Italiana / BibIt",
        "source_landing_page": BIBIT_LANDING_URL,
        "source_api": BIBIT_API_URL,
        "scope": {
            "language": "ita",
            "periods": [*HISTORICAL_PERIODS, BRIDGE_PERIOD],
            "collection": "bibit",
        },
        "records": [record.to_dict() for record in records],
    }
    _write_json(path, payload)


def write_decision_csv(decisions: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=DECISION_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(decisions)


def render_composition_markdown(report: dict[str, Any]) -> str:
    """Render the public audit rationale and staged-curriculum decision."""

    roles = report["role_projection"]
    lines = [
        "# Biblioteca Italiana Historical Corpus Composition Audit",
        "",
        "## Decision",
        "",
        "The metadata composition gate passes. BibIt is a high-value expansion source,",
        "but this report does **not** activate all records as training data. Every work",
        "must still pass its role-specific TEI, edition, language-variety, and leakage audit.",
        "",
        f"- Catalog records reviewed: {report['catalog_record_count']:,}.",
        f"- Origins-through-Settecento records: {report['historical_catalog_record_count']:,}.",
        f"- Ottocento bridge candidates: {report['nineteenth_century_candidate_count']:,}.",
        f"- Duplicate/edition families requiring one canonical selection: "
        f"{report['duplicate_edition_family_count']:,}.",
        f"- Status: `{report['corpus_activation_status']}`.",
        "",
        "## Reuse Terms",
        "",
        "BibIt states that its digital resources are freely accessible for personal or",
        "scientific use, that public reuse must cite Biblioteca Italiana as the source,",
        "and that commercial reuse is prohibited. This source therefore creates a",
        "non-commercial data/model lineage. Work-level TEI headers must remain the",
        "canonical edition, editor, publisher, revision, and digitization record.",
        "",
        f"- [Biblioteca Italiana project page]({report['source_project_page']})",
        f"- [Biblioteca Italiana FAQ]({report['source_faq']})",
        f"- [BibIt catalog]({report['source_landing_page']})",
        "",
        "## Catalog Composition",
        "",
        "| Period | Records |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {label} | {count:,} |" for label, count in report["period_counts"].items()
    )
    lines.extend([
        "",
        "| Genre | Records |",
        "| --- | ---: |",
    ])
    lines.extend(
        f"| {genre} | {count:,} |" for genre, count in report["genre_counts"].items()
    )
    lines.extend([
        "",
        "## Projected Roles",
        "",
        "The projected sizes come from a deterministic period/genre sample and are",
        "planning estimates, not final post-cleaning token counts.",
        "",
        "| Role | Records | Estimated characters | Share of non-excluded estimate |",
        "| --- | ---: | ---: | ---: |",
    ])
    for role in sorted(ALLOWED_ROLES):
        row = roles[role]
        lines.append(
            f"| `{role}` | {row['record_count']:,} | "
            f"{row['estimated_characters']:,} | {row['estimated_active_share']:.1%} |"
        )
    lines.extend([
        "",
        "## Scale Estimate",
        "",
        f"- Deterministic estimation sample: {report['estimation_sample_count']:,} records "
        f"across {report['stratum_count']:,} period/genre strata.",
        "- Sample rendered literary characters: "
        f"{report['estimation_sample_rendered_characters']:,}.",
        "- Sample records with no rendered primary text: "
        f"{len(report['empty_rendered_sample_ids']):,}; these records are excluded.",
        "- Projected historical non-excluded characters before TEI cleaning and "
        f"deduplication: {report['estimated_historical_active_characters']:,}.",
        "- Approximate historical token range at 3-4 characters/token: "
        f"{report['estimated_historical_active_min_tokens']:,}-"
        f"{report['estimated_historical_active_max_tokens']:,}.",
        "",
        "The Ottocento bridge is not part of that historical token estimate. The current",
        f"recommendation is to cap it at no more than {report['bridge_share_recommendation']:.0%} "
        "of a future adaptation mixture, with the exact cap frozen only after full",
        "cleaning, deduplication, and language-variety review.",
        "",
        "### Measured Long-Work Anchors",
        "",
        "| Record | Work | Rendered characters |",
        "| --- | --- | ---: |",
    ])
    lines.extend(
        f"| `{row['object_id']}` | {row['authors'][0]} / {row['title']} | "
        f"{row['rendered_characters']:,} |"
        for row in report["representative_inspections"]
    )
    lines.extend([
        "",
        "These measured anchors prevent unusually long canonical works from being",
        "replaced by the median size of their period/genre stratum.",
        "",
        "## Canonical Editions",
        "",
        "Near-duplicate editions do not enter together. The decision CSV marks one",
        "candidate per normalized author/title family and excludes alternates. The",
        "Ariosto family has an explicit editorial override: use *Orlando Furioso 1532*,",
        "the final authorial edition, rather than combining the 1516, 1521, 1532, and",
        "modern-edition records. The Manzoni family similarly selects the later standard",
        "*Promessi Sposi* text and excludes the separate 1827 redaction from this mixture.",
        "",
        "| Family | Selected record | Alternates | Reason |",
        "| --- | --- | ---: | --- |",
    ])
    for family in report["duplicate_edition_families"][:20]:
        lines.append(
            f"| {family['author']} / {family['normalized_title']} | "
            f"`{family['canonical_object_id']}` | {len(family['alternates'])} | "
            f"{family['selection_reason']} |"
        )
    lines.extend([
        "",
        "## Leakage And Curriculum",
        "",
        f"{report['leakage_rule']}",
        "",
    ])
    lines.extend(
        f"{index}. {stage.split(': ', 1)[1]}"
        for index, stage in enumerate(report["curriculum_decision"], start=1)
    )
    lines.extend([
        "",
        "Long poems such as the *Commedia*, *Orlando innamorato*, *Orlando Furioso*,",
        "and *Gerusalemme liberata* belong in the non-sonnet poetry stage. Collections",
        "such as *Rime* and *Canzoniere* remain sonnet-only candidates until TEI-form",
        "segmentation proves which individual units are eligible. Sonnets retained for",
        "specialization will use author/epoch-balanced sampling, and the existing held-out",
        "validation/test assignments remain fixed.",
        "",
        "## Next Activation Gate",
        "",
        "1. Download only canonical TEI records selected for a named stage.",
        "2. Parse with external DTD/network resolution disabled and retain TEI-header provenance.",
        "3. Route every explicit `lg type=\"sonetto...\"` unit away from historical pretraining.",
        "4. Audit dialect, editorial apparatus, language, empty text, and exact/near duplicates.",
        "5. Freeze final post-cleaning token shares, the Ottocento cap, preservation replay,",
        "   and validation splits before restarting any GPU training.",
        "",
    ])
    return "\n".join(lines)


def work_family_key(record: BibItCatalogRecord) -> str:
    return f"{_primary_author_key(record)}::{normalized_work_title(record.title)}"


def normalized_work_title(title: str) -> str:
    normalized = _normalize_key(title)
    if normalized.startswith("orlando furioso"):
        normalized = _YEAR_SUFFIX.sub(" ", normalized)
    if "promessi sposi" in normalized:
        normalized = "promessi sposi"
    return " ".join(normalized.split())


def _canonical_record_score(record: BibItCatalogRecord) -> tuple[int, int, str]:
    metadata_score = sum(bool(value) for value in (
        record.source_publisher,
        record.source_publication_place,
        record.source_publication_date,
        record.source_identifier,
        record.source_authors,
    ))
    explicit_year = bool(_YEAR_SUFFIX.search(_normalize_key(record.title)))
    return metadata_score, int(explicit_year), record.object_id


def _primary_author_key(record: BibItCatalogRecord) -> str:
    author = record.authors[0] if record.authors else "non definito"
    return _normalize_key(author)


def _primary_period(record: BibItCatalogRecord) -> str:
    return record.periods[0] if record.periods else "(missing)"


def _primary_genre(record: BibItCatalogRecord) -> str:
    return record.genres[0] if record.genres else "(missing)"


def _stratum(record: BibItCatalogRecord) -> tuple[str, str]:
    return _primary_period(record), _primary_genre(record)


def _sample_key(object_id: str) -> str:
    return hashlib.sha256(f"bibit-composition-v1:{object_id}".encode("utf-8")).hexdigest()


def _normalize_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    ascii_value = "".join(character for character in normalized if not unicodedata.combining(character))
    return " ".join(_NON_WORD.sub(" ", ascii_value).split())


def _ordered_counts(counts: Counter[str], labels: dict[str, str]) -> dict[str, int]:
    ordered: dict[str, int] = {}
    for period in (*HISTORICAL_PERIODS, BRIDGE_PERIOD):
        ordered[labels[period]] = counts.get(period, 0)
    for period in sorted(set(counts) - set(labels)):
        ordered[period] = counts[period]
    return ordered


def _validate_config(config: BibItCompositionAuditConfig) -> None:
    if config.sample_per_stratum <= 0:
        raise ValueError("sample_per_stratum must be greater than zero")
    if config.request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be greater than zero")
    if config.rendered_batch_size <= 0:
        raise ValueError("rendered_batch_size must be greater than zero")
    if config.characters_per_token_min <= 0 or config.characters_per_token_max <= 0:
        raise ValueError("characters-per-token assumptions must be positive")
    if config.characters_per_token_min > config.characters_per_token_max:
        raise ValueError("minimum characters per token cannot exceed maximum")
    if not 0 < config.bridge_share_recommendation < 1:
        raise ValueError("bridge_share_recommendation must be between zero and one")


def _validate_catalog(records: list[BibItCatalogRecord]) -> None:
    if not records:
        raise ValueError("BibIt catalog is empty")
    object_ids = [record.object_id for record in records]
    duplicate_ids = [object_id for object_id, count in Counter(object_ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError("BibIt catalog has duplicate object IDs: " + ", ".join(duplicate_ids))
    if any(not record.title or not record.authors or not record.periods or not record.genres for record in records):
        raise ValueError("BibIt catalog contains incomplete composition metadata")


def _portable_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _elapsed(started: float) -> str:
    seconds = max(0, round(monotonic() - started))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _report(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)


class _BibItRenderedTextParser(HTMLParser):
    """Extract primary text while skipping generated headers, TOCs, and page labels."""

    SKIPPED_CLASSES = {"stdheader", "stdfooter", "toc", "pagebreak"}
    SKIPPED_TAGS = {"head", "script", "style"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = set(dict(attrs).get("class", "").split())
        should_skip = tag.casefold() in self.SKIPPED_TAGS or bool(classes & self.SKIPPED_CLASSES)
        if self._skip_depth or should_skip:
            self._skip_depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        return

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.text_parts.append(data)
