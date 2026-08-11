"""Resolve Italian Wikisource candidates into a metadata-only audit queue."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from sonnet_corpus.wikisource_archive_inventory import (
    DUMP_BASE_URL,
    DUMP_DATE,
    USER_AGENT,
    iter_sql_insert_rows,
)


PAGELINKS_FILE = "itwikisource-20260801-pagelinks.sql.gz"
PAGELINKS_SHA1 = "9bd9baad4e69c014c9d390f6679a34c2b33e0792"
NAMESPACES_FILE = "itwikisource-20260801-siteinfo-namespaces.json.gz"
NAMESPACES_SHA1 = "92c44b14e7bf9ef629d7ca421a3fe9b68d54d6c6"
PAGE_FILE = "itwikisource-20260801-page.sql.gz"
PAGE_SHA1 = "d346505404381269c65d933c2c1d65031693c615"
LINKTARGET_FILE = "itwikisource-20260801-linktarget.sql.gz"
LINKTARGET_SHA1 = "ba38f03d8c0127667d614e36a11f24ee40bb56f4"
INDEX_NAMESPACE_ID = 110

EXPECTED_INVENTORY_ROWS = 22_165
EXPECTED_HIERARCHY_ROWS = 117_297
EXPECTED_CANDIDATE_ROWS = 6_863

RESOLUTION_FIELDS = (
    "work_root_id",
    "root_title",
    "landing_page_url",
    "metadata_decision",
    "proposed_role",
    "author_evidence",
    "period_bucket",
    "language_route",
    "genre_route",
    "form_route",
    "projected_wikitext_bytes",
    "hierarchy_page_count",
    "direct_scan_link_count",
    "direct_scan_titles",
    "direct_scan_page_ids",
    "direct_scan_revision_ids",
    "scan_group_root_count",
    "scan_group_candidate_count",
    "scan_group_conditioned_count",
    "scan_group_nonitalian_count",
    "scan_group_nonstandard_hold_count",
    "scan_group_existing_reference_count",
    "scan_title_language_signals",
    "source_scan_resolution",
    "identity_resolution",
    "checkpoint_4b_decision",
    "review_reason",
    "next_action",
    "activation_status",
)

SCAN_LINK_FIELDS = (
    "work_root_id",
    "root_title",
    "metadata_decision",
    "proposed_role",
    "linking_page_count",
    "linking_page_ids",
    "scan_title",
    "scan_url",
    "scan_page_id",
    "scan_revision_id",
    "scan_touched_utc",
    "scan_wikitext_bytes",
    "scan_is_redirect",
    "scan_exists_in_dump",
    "scan_shared_root_count",
    "scan_shared_candidate_count",
    "scan_shared_conditioned_count",
    "scan_shared_nonitalian_count",
    "scan_shared_nonstandard_hold_count",
    "scan_shared_existing_reference_count",
    "scan_title_language_signals",
    "activation_status",
)

REVIEW_FIELDS = (
    "review_id",
    "checkpoint_4b_decision",
    "review_unit_type",
    "work_root_count",
    "projected_wikitext_bytes",
    "work_root_ids",
    "representative_root_titles",
    "scan_titles",
    "scan_title_language_signals",
    "scan_group_conditioned_count",
    "scan_group_nonitalian_count",
    "scan_group_nonstandard_hold_count",
    "review_status",
    "required_action",
    "activation_status",
)

_STANDARD_LANGUAGE_EVIDENCE = {
    "it",
    "italiano",
    "lingua italiana",
    "volgare italiano",
}

# These are deliberately specific title signals. Broad fragments such as
# ``venet`` or ``frances`` would incorrectly match places and personal names.
_SCAN_LANGUAGE_SIGNALS = {
    "abruzzese": "abruzzese",
    "bergamasco": "bergamasco",
    "bolognese": "bolognese",
    "bresciano": "bresciano",
    "calabrese": "calabrese",
    "dialect": "dialect",
    "dialetto": "dialetto",
    "friulano": "friulano",
    "gallurese": "gallurese",
    "genovese": "genovese",
    "istrioto": "istrioto",
    "ladino": "ladino",
    "milanese": "milanese",
    "muglisano": "muglisano",
    "napoletano": "napoletano",
    "piemontese": "piemontese",
    "potentino": "potentino",
    "provenzale": "provenzale",
    "romancio": "romancio",
    "romanesco": "romanesco",
    "romaneschi": "romanesco",
    "sardo": "sardo",
    "siciliano": "siciliano",
    "tergestino": "tergestino",
    "ticinese": "ticinese",
    "triestino": "triestino",
    "veneziano": "veneziano",
    "vernacolo": "vernacolo",
    "volgare lombardo": "volgare_lombardo",
}


@dataclass(frozen=True)
class WikisourceCandidateResolutionConfig:
    """Pinned inputs and public outputs for checkpoint 4B."""

    repo_root: Path
    cache_dir: Path
    inventory_path: Path
    hierarchy_path: Path
    resolution_path: Path
    scan_links_path: Path
    review_path: Path
    json_report_path: Path
    markdown_report_path: Path
    dump_date: str = DUMP_DATE
    dump_base_url: str = DUMP_BASE_URL
    page_filename: str = PAGE_FILE
    page_sha1: str = PAGE_SHA1
    linktarget_filename: str = LINKTARGET_FILE
    linktarget_sha1: str = LINKTARGET_SHA1
    pagelinks_filename: str = PAGELINKS_FILE
    pagelinks_sha1: str = PAGELINKS_SHA1
    namespaces_filename: str = NAMESPACES_FILE
    namespaces_sha1: str = NAMESPACES_SHA1
    expected_inventory_rows: int = EXPECTED_INVENTORY_ROWS
    expected_hierarchy_rows: int = EXPECTED_HIERARCHY_ROWS
    expected_candidate_rows: int = EXPECTED_CANDIDATE_ROWS
    progress_interval: int = 500_000


@dataclass(frozen=True)
class IndexPage:
    title: str
    page_id: int
    revision_id: int
    touched_utc: str
    wikitext_bytes: int
    is_redirect: bool


def build_wikisource_candidate_resolution(
    config: WikisourceCandidateResolutionConfig,
    *,
    session: requests.Session | None = None,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Build the complete 4B candidate queue without acquiring page text."""

    _validate_config(config)
    http = session or requests.Session()
    if session is None:
        http.headers.update({"User-Agent": USER_AGENT})

    dump_specs = {
        "page": (config.page_filename, config.page_sha1),
        "linktarget": (config.linktarget_filename, config.linktarget_sha1),
        "pagelinks": (config.pagelinks_filename, config.pagelinks_sha1),
        "namespaces": (config.namespaces_filename, config.namespaces_sha1),
    }
    dump_paths = {
        label: _acquire_metadata_file(
            config,
            filename=filename,
            expected_sha1=expected_sha1,
            session=http,
            progress=progress,
        )
        for label, (filename, expected_sha1) in dump_specs.items()
    }

    namespace = _load_index_namespace(dump_paths["namespaces"])
    inventory_rows = _read_csv(config.inventory_path)
    hierarchy_rows = _read_csv(config.hierarchy_path)
    if len(inventory_rows) != config.expected_inventory_rows:
        raise ValueError(
            "unexpected 4A inventory row count: "
            f"{len(inventory_rows)} != {config.expected_inventory_rows}"
        )
    if len(hierarchy_rows) != config.expected_hierarchy_rows:
        raise ValueError(
            "unexpected 4A hierarchy row count: "
            f"{len(hierarchy_rows)} != {config.expected_hierarchy_rows}"
        )

    inventory = {row["work_root_id"]: row for row in inventory_rows}
    candidates = [
        row
        for row in inventory_rows
        if row["metadata_decision"].endswith("metadata_candidate")
    ]
    if len(candidates) != config.expected_candidate_rows:
        raise ValueError(
            "unexpected 4A candidate row count: "
            f"{len(candidates)} != {config.expected_candidate_rows}"
        )

    page_to_root: dict[int, str] = {}
    for row in hierarchy_rows:
        root_id = row["work_root_id"]
        if root_id not in inventory:
            raise ValueError(f"hierarchy references an unknown work root: {root_id}")
        page_to_root[int(row["page_id"])] = root_id

    target_titles = _load_index_targets(dump_paths["linktarget"])
    index_pages = _load_index_pages(dump_paths["page"])
    root_scan_link_pages = _load_root_scan_links(
        dump_paths["pagelinks"],
        page_to_root=page_to_root,
        target_titles=target_titles,
        progress_interval=config.progress_interval,
        progress=progress,
    )
    scan_roots: dict[str, set[str]] = defaultdict(set)
    for root_id, scans in root_scan_link_pages.items():
        for scan_title in scans:
            scan_roots[scan_title].add(root_id)

    scan_link_rows = _build_scan_link_rows(
        candidates,
        inventory=inventory,
        root_scan_link_pages=root_scan_link_pages,
        scan_roots=scan_roots,
        index_pages=index_pages,
    )
    resolution_rows = _build_resolution_rows(
        candidates,
        inventory=inventory,
        root_scan_link_pages=root_scan_link_pages,
        scan_roots=scan_roots,
        index_pages=index_pages,
    )
    review_rows = _build_review_rows(resolution_rows)

    _write_csv(config.scan_links_path, SCAN_LINK_FIELDS, scan_link_rows)
    _write_csv(config.resolution_path, RESOLUTION_FIELDS, resolution_rows)
    _write_csv(config.review_path, REVIEW_FIELDS, review_rows)
    report = _build_report(
        config,
        dump_paths=dump_paths,
        namespace=namespace,
        inventory_rows=inventory_rows,
        hierarchy_rows=hierarchy_rows,
        resolution_rows=resolution_rows,
        scan_link_rows=scan_link_rows,
        review_rows=review_rows,
        scan_roots=scan_roots,
    )
    _write_json(config.json_report_path, report)
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(
        render_wikisource_candidate_resolution_markdown(report),
        encoding="utf-8",
    )
    return report


def classify_language_evidence(value: str) -> str:
    """Separate genuine language evidence from citation-index categories."""

    values = [item.strip() for item in value.split(" | ") if item.strip()]
    substantive = [
        item
        for item in values
        if not item.casefold().startswith("cui è citato")
    ]
    if not substantive:
        return "citation_only_or_unmarked"
    if all(item.casefold() in _STANDARD_LANGUAGE_EVIDENCE for item in substantive):
        return "standard_italian_explicit"
    return "nonstandard_or_unknown_language_evidence"


def scan_title_language_signals(title: str) -> list[str]:
    """Return conservative dialect/non-Italian signals present in a scan title."""

    folded = title.casefold()
    return sorted(
        {
            label
            for fragment, label in _SCAN_LANGUAGE_SIGNALS.items()
            if fragment in folded
        }
    )


def render_wikisource_candidate_resolution_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Italian Wikisource Candidate And Source-Scan Resolution",
        "",
        "## Result",
        "",
        (
            f"Checkpoint 4B resolves all {report['candidate_count']:,} provisional "
            f"checkpoint-4A candidates against the pinned `{report['dump']['date']}` "
            "metadata link graph."
        ),
        "",
        (
            f"Direct `Indice:` links exist for {report['direct_scan_linked_candidate_count']:,} "
            f"candidates across {report['distinct_candidate_scan_count']:,} source scans. "
            f"The metadata-only page-level audit queue contains "
            f"{report['eligible_page_level_audit_count']:,} candidates projecting "
            f"{report['eligible_projected_wikitext_bytes']:,} wikitext bytes."
        ),
        "",
        "## Decisions",
        "",
        "| Decision | Work roots | Projected wikitext bytes |",
        "| --- | ---: | ---: |",
    ]
    for decision, count in report["decision_counts"].items():
        lines.append(
            f"| `{decision}` | {count:,} | "
            f"{report['projected_wikitext_bytes_by_decision'][decision]:,} |"
        )
    lines.extend(
        [
            "",
            "## Eligible Role Projection",
            "",
            "| Role | Work roots | Projected wikitext bytes |",
            "| --- | ---: | ---: |",
        ]
    )
    for role, count in report["eligible_role_counts"].items():
        lines.append(
            f"| `{role}` | {count:,} | "
            f"{report['eligible_projected_wikitext_bytes_by_role'][role]:,} |"
        )
    language = report["language_evidence_audit"]
    lines.extend(
        [
            "",
            "## Language-Evidence Correction",
            "",
            (
                f"Checkpoint 4A contains {language['hold_language_variety_row_count']:,} "
                "language-review rows. Of these, "
                f"{language['citation_only_or_standard_row_count']:,} contain only "
                "citation-index labels and/or explicit standard Italian; they are not "
                "treated as dialect evidence in scan-group propagation."
            ),
            (
                f"The remaining {language['nonstandard_or_unknown_row_count']:,} rows "
                "retain genuine nonstandard or unresolved language evidence. No held "
                "row is promoted into the candidate queue by this correction."
            ),
            "",
            "## Source-Scan Boundaries",
            "",
            (
                f"The {report['direct_scan_linked_candidate_count']:,} linked candidates "
                f"map to {report['distinct_candidate_scan_count']:,} scans; "
                f"{report['shared_candidate_scan_count']:,} scans support more than one "
                "candidate root. Scan grouping is retained for later extraction and "
                "deduplication rather than flattening anthology contents."
            ),
            f"- Bounded review units: {report['review_unit_count']:,}.",
            "- Wikitext bytes remain projections, not cleaned characters or tokens.",
            (
                "- Current site terms remain CC BY-SA 4.0; scan/work rights "
                "still require final verification."
            ),
            "",
            "## Boundaries",
            "",
            "- The full page-text dump was not downloaded.",
            "- No primary text was extracted or activated.",
            "- Conditioned or unresolved language material was not admitted to standard core.",
            "- No V7 split, mixture weight, cache deletion, or GPU work occurred.",
            "",
            "## Artifacts",
            "",
            f"- Candidate resolution: `{report['outputs']['resolution_path']}`",
            f"- Source-scan links: `{report['outputs']['scan_links_path']}`",
            f"- Bounded review ledger: `{report['outputs']['review_path']}`",
            f"- Machine-readable report: `{report['outputs']['json_report_path']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _build_scan_link_rows(
    candidates: list[dict[str, str]],
    *,
    inventory: dict[str, dict[str, str]],
    root_scan_link_pages: dict[str, dict[str, set[int]]],
    scan_roots: dict[str, set[str]],
    index_pages: dict[str, IndexPage],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_ids = {row["work_root_id"] for row in candidates}
    for candidate in sorted(candidates, key=_row_sort_key):
        root_id = candidate["work_root_id"]
        for scan_title in sorted(root_scan_link_pages.get(root_id, {}), key=str.casefold):
            linked_roots = scan_roots[scan_title]
            counts = _shared_decision_counts(linked_roots, inventory, candidate_ids)
            page = index_pages.get(scan_title)
            rows.append(
                {
                    "work_root_id": root_id,
                    "root_title": candidate["root_title"],
                    "metadata_decision": candidate["metadata_decision"],
                    "proposed_role": candidate["proposed_role"],
                    "linking_page_count": len(root_scan_link_pages[root_id][scan_title]),
                    "linking_page_ids": " | ".join(
                        str(value)
                        for value in sorted(root_scan_link_pages[root_id][scan_title])
                    ),
                    "scan_title": scan_title,
                    "scan_url": _index_url(scan_title),
                    "scan_page_id": page.page_id if page else "",
                    "scan_revision_id": page.revision_id if page else "",
                    "scan_touched_utc": page.touched_utc if page else "",
                    "scan_wikitext_bytes": page.wikitext_bytes if page else 0,
                    "scan_is_redirect": page.is_redirect if page else "",
                    "scan_exists_in_dump": bool(page),
                    "scan_shared_root_count": len(linked_roots),
                    "scan_shared_candidate_count": counts["candidate"],
                    "scan_shared_conditioned_count": counts["conditioned"],
                    "scan_shared_nonitalian_count": counts["nonitalian"],
                    "scan_shared_nonstandard_hold_count": counts["nonstandard_hold"],
                    "scan_shared_existing_reference_count": counts["existing_reference"],
                    "scan_title_language_signals": " | ".join(
                        scan_title_language_signals(scan_title)
                    ),
                    "activation_status": "metadata_only_not_activated",
                }
            )
    return rows


def _build_resolution_rows(
    candidates: list[dict[str, str]],
    *,
    inventory: dict[str, dict[str, str]],
    root_scan_link_pages: dict[str, dict[str, set[int]]],
    scan_roots: dict[str, set[str]],
    index_pages: dict[str, IndexPage],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_ids = {row["work_root_id"] for row in candidates}
    for candidate in sorted(candidates, key=_row_sort_key):
        root_id = candidate["work_root_id"]
        scan_titles = sorted(root_scan_link_pages.get(root_id, {}), key=str.casefold)
        linked_roots: set[str] = set()
        for title in scan_titles:
            linked_roots.update(scan_roots[title])
        counts = _shared_decision_counts(linked_roots, inventory, candidate_ids)
        title_signals = sorted(
            {
                signal
                for title in scan_titles
                for signal in scan_title_language_signals(title)
            }
        )
        scan_pages = [index_pages.get(title) for title in scan_titles]
        decision, source_scan_resolution, identity_resolution, reason, action = (
            _candidate_decision(
                scan_titles=scan_titles,
                scan_pages=scan_pages,
                title_signals=title_signals,
                shared_counts=counts,
            )
        )
        rows.append(
            {
                **{field: candidate[field] for field in RESOLUTION_FIELDS if field in candidate},
                "direct_scan_link_count": len(scan_titles),
                "direct_scan_titles": " | ".join(scan_titles),
                "direct_scan_page_ids": " | ".join(
                    str(page.page_id) for page in scan_pages if page
                ),
                "direct_scan_revision_ids": " | ".join(
                    str(page.revision_id) for page in scan_pages if page
                ),
                "scan_group_root_count": len(linked_roots),
                "scan_group_candidate_count": counts["candidate"],
                "scan_group_conditioned_count": counts["conditioned"],
                "scan_group_nonitalian_count": counts["nonitalian"],
                "scan_group_nonstandard_hold_count": counts["nonstandard_hold"],
                "scan_group_existing_reference_count": counts["existing_reference"],
                "scan_title_language_signals": " | ".join(title_signals),
                "source_scan_resolution": source_scan_resolution,
                "identity_resolution": identity_resolution,
                "checkpoint_4b_decision": decision,
                "review_reason": reason,
                "next_action": action,
                "activation_status": "metadata_only_not_activated",
            }
        )
    return rows


def _candidate_decision(
    *,
    scan_titles: list[str],
    scan_pages: list[IndexPage | None],
    title_signals: list[str],
    shared_counts: Counter[str],
) -> tuple[str, str, str, str, str]:
    if not scan_titles:
        return (
            "hold_no_direct_scan_link",
            "no_direct_index_link_in_pinned_graph",
            "work_identity_not_scan_anchored",
            "No hierarchy page links directly to an Index-namespace source scan.",
            "Resolve the exact source scan or retain the work outside page-level extraction.",
        )
    if len(scan_titles) > 1:
        return (
            "hold_multiple_source_scans",
            "multiple_direct_index_links",
            "edition_boundary_ambiguous",
            "The work hierarchy links to multiple source scans.",
            "Select and justify one edition or define non-overlapping edition segments.",
        )
    if any(page is None for page in scan_pages):
        return (
            "hold_missing_index_page",
            "linked_index_target_missing_from_page_dump",
            "source_scan_target_unresolved",
            "The link target is absent from the pinned Index-namespace page table.",
            "Resolve the missing or renamed Index page before page-level extraction.",
        )
    if any(page and page.is_redirect for page in scan_pages):
        return (
            "hold_redirected_index_page",
            "linked_index_page_is_redirect",
            "source_scan_redirect_unresolved",
            "The pinned Index page is a redirect and its destination is not resolved here.",
            "Resolve and pin the canonical Index page before page-level extraction.",
        )
    if (
        title_signals
        or shared_counts["conditioned"]
        or shared_counts["nonitalian"]
        or shared_counts["nonstandard_hold"]
    ):
        return (
            "hold_scan_language_conflict",
            "direct_index_page_verified_language_conflict",
            "work_scan_identity_linked_but_language_boundary_unresolved",
            (
                "The scan title or a root sharing the scan has genuine "
                "nonstandard/unknown language evidence."
            ),
            (
                "Resolve language boundaries at scan and root level; never "
                "default the material to standard core."
            ),
        )
    return (
        "eligible_page_level_audit_queue",
        "single_direct_index_page_verified",
        "work_root_and_scan_identity_metadata_linked",
        (
            "One existing non-redirect Index page anchors the candidate without "
            "a propagated language conflict."
        ),
        (
            "Audit revision-pinned page boundaries, primary text, rights, quality, "
            "and cross-corpus overlap."
        ),
    )


def _build_review_rows(resolution_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in resolution_rows:
        decision = str(row["checkpoint_4b_decision"])
        if decision == "eligible_page_level_audit_queue":
            continue
        scan_key = str(row["direct_scan_titles"])
        key = (decision, scan_key if scan_key else str(row["work_root_id"]))
        grouped[key].append(row)

    rows: list[dict[str, Any]] = []
    ordered = sorted(
        grouped.values(),
        key=lambda values: (
            str(values[0]["checkpoint_4b_decision"]),
            str(values[0]["direct_scan_titles"]).casefold(),
            str(values[0]["root_title"]).casefold(),
        ),
    )
    for index, values in enumerate(ordered, start=1):
        decision = str(values[0]["checkpoint_4b_decision"])
        scan_titles = sorted(
            {
                title
                for row in values
                for title in str(row["direct_scan_titles"]).split(" | ")
                if title
            },
            key=str.casefold,
        )
        rows.append(
            {
                "review_id": f"itws-4b-review-{index:04d}",
                "checkpoint_4b_decision": decision,
                "review_unit_type": (
                    "shared_scan_cluster" if scan_titles and len(values) > 1 else "work_root"
                ),
                "work_root_count": len(values),
                "projected_wikitext_bytes": sum(
                    int(row["projected_wikitext_bytes"]) for row in values
                ),
                "work_root_ids": " | ".join(
                    sorted(str(row["work_root_id"]) for row in values)
                ),
                "representative_root_titles": " | ".join(
                    sorted(
                        {str(row["root_title"]) for row in values},
                        key=str.casefold,
                    )[:10]
                ),
                "scan_titles": " | ".join(scan_titles),
                "scan_title_language_signals": " | ".join(
                    sorted(
                        {
                            signal
                            for row in values
                            for signal in str(row["scan_title_language_signals"]).split(" | ")
                            if signal
                        }
                    )
                ),
                "scan_group_conditioned_count": max(
                    int(row["scan_group_conditioned_count"]) for row in values
                ),
                "scan_group_nonitalian_count": max(
                    int(row["scan_group_nonitalian_count"]) for row in values
                ),
                "scan_group_nonstandard_hold_count": max(
                    int(row["scan_group_nonstandard_hold_count"]) for row in values
                ),
                "review_status": "unresolved_hold",
                "required_action": values[0]["next_action"],
                "activation_status": "metadata_only_not_activated",
            }
        )
    return rows


def _shared_decision_counts(
    root_ids: set[str],
    inventory: dict[str, dict[str, str]],
    candidate_ids: set[str],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for root_id in root_ids:
        row = inventory[root_id]
        decision = row["metadata_decision"]
        if root_id in candidate_ids:
            counts["candidate"] += 1
        if decision == "conditioned_language_candidate":
            counts["conditioned"] += 1
        if decision == "exclude_explicit_non_italian":
            counts["nonitalian"] += 1
        if (
            decision == "hold_language_variety_review"
            and classify_language_evidence(row["language_evidence"])
            == "nonstandard_or_unknown_language_evidence"
        ):
            counts["nonstandard_hold"] += 1
        if decision == "existing_project_reference":
            counts["existing_reference"] += 1
    return counts


def _build_report(
    config: WikisourceCandidateResolutionConfig,
    *,
    dump_paths: dict[str, Path],
    namespace: dict[str, Any],
    inventory_rows: list[dict[str, str]],
    hierarchy_rows: list[dict[str, str]],
    resolution_rows: list[dict[str, Any]],
    scan_link_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    scan_roots: dict[str, set[str]],
) -> dict[str, Any]:
    decision_counts = Counter(row["checkpoint_4b_decision"] for row in resolution_rows)
    decision_bytes: Counter[str] = Counter()
    for row in resolution_rows:
        decision_bytes[row["checkpoint_4b_decision"]] += int(
            row["projected_wikitext_bytes"]
        )
    eligible = [
        row
        for row in resolution_rows
        if row["checkpoint_4b_decision"] == "eligible_page_level_audit_queue"
    ]
    role_counts = Counter(row["proposed_role"] for row in eligible)
    role_bytes: Counter[str] = Counter()
    for row in eligible:
        role_bytes[row["proposed_role"]] += int(row["projected_wikitext_bytes"])
    candidate_scans = {row["scan_title"] for row in scan_link_rows}
    language_holds = [
        row
        for row in inventory_rows
        if row["metadata_decision"] == "hold_language_variety_review"
    ]
    language_classes = Counter(
        classify_language_evidence(row["language_evidence"]) for row in language_holds
    )
    top_authors = _top_projection(eligible, "author_evidence", limit=15)
    scan_bytes: Counter[str] = Counter()
    scan_candidate_counts: Counter[str] = Counter()
    for row in eligible:
        title = str(row["direct_scan_titles"])
        scan_bytes[title] += int(row["projected_wikitext_bytes"])
        scan_candidate_counts[title] += 1
    largest_scans = [
        {
            "scan_title": title,
            "eligible_work_root_count": scan_candidate_counts[title],
            "projected_wikitext_bytes": size,
            "share": _ratio(size, sum(scan_bytes.values())),
        }
        for title, size in scan_bytes.most_common(20)
    ]
    return {
        "resolution_version": "italian_wikisource_candidate_resolution_v1",
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
        "index_namespace": namespace,
        "inputs": {
            "inventory_path": _portable(config.inventory_path, config.repo_root),
            "inventory_sha256": _sha256_file(config.inventory_path),
            "inventory_row_count": len(inventory_rows),
            "hierarchy_path": _portable(config.hierarchy_path, config.repo_root),
            "hierarchy_sha256": _sha256_file(config.hierarchy_path),
            "hierarchy_row_count": len(hierarchy_rows),
        },
        "candidate_count": len(resolution_rows),
        "candidate_projected_wikitext_bytes": sum(
            int(row["projected_wikitext_bytes"]) for row in resolution_rows
        ),
        "direct_scan_linked_candidate_count": sum(
            int(row["direct_scan_link_count"]) > 0 for row in resolution_rows
        ),
        "distinct_candidate_scan_count": len(candidate_scans),
        "shared_candidate_scan_count": sum(
            sum(root in {row["work_root_id"] for row in resolution_rows} for root in roots)
            > 1
            for title, roots in scan_roots.items()
            if title in candidate_scans
        ),
        "decision_counts": dict(sorted(decision_counts.items())),
        "projected_wikitext_bytes_by_decision": dict(sorted(decision_bytes.items())),
        "eligible_page_level_audit_count": len(eligible),
        "eligible_projected_wikitext_bytes": sum(
            int(row["projected_wikitext_bytes"]) for row in eligible
        ),
        "eligible_role_counts": dict(sorted(role_counts.items())),
        "eligible_projected_wikitext_bytes_by_role": dict(sorted(role_bytes.items())),
        "review_unit_count": len(review_rows),
        "language_evidence_audit": {
            "hold_language_variety_row_count": len(language_holds),
            "citation_only_or_standard_row_count": (
                language_classes["citation_only_or_unmarked"]
                + language_classes["standard_italian_explicit"]
            ),
            "citation_only_or_unmarked_row_count": language_classes[
                "citation_only_or_unmarked"
            ],
            "standard_italian_explicit_row_count": language_classes[
                "standard_italian_explicit"
            ],
            "nonstandard_or_unknown_row_count": language_classes[
                "nonstandard_or_unknown_language_evidence"
            ],
            "policy": (
                "citation-index categories do not propagate language hazards; held rows "
                "are not promoted into the 4B candidate queue"
            ),
        },
        "concentration": {
            "top_author_proxies": top_authors,
            "largest_eligible_scan_groups": largest_scans,
            "warning": (
                "wikitext bytes are projections; cleaned text, tokens, duplicate removal, "
                "and unknown/multi-author identities can change final concentration"
            ),
        },
        "outputs": {
            "resolution_path": _portable(config.resolution_path, config.repo_root),
            "resolution_sha256": _sha256_file(config.resolution_path),
            "scan_links_path": _portable(config.scan_links_path, config.repo_root),
            "scan_links_sha256": _sha256_file(config.scan_links_path),
            "review_path": _portable(config.review_path, config.repo_root),
            "review_sha256": _sha256_file(config.review_path),
            "json_report_path": _portable(config.json_report_path, config.repo_root),
            "markdown_report_path": _portable(config.markdown_report_path, config.repo_root),
        },
        "policy": {
            "metadata_only": True,
            "source_text_extracted": False,
            "corpus_text_activated": False,
            "conditioned_material_standard_core_eligible": False,
            "v7_split_assigned": False,
            "training_mixture_weight_assigned": False,
            "cache_deleted": False,
            "gpu_work_started": False,
            "eligible_queue_authorizes_extraction": False,
        },
        "next_checkpoint": (
            "Propose revision-pinned page-boundary extraction and cross-corpus probe for "
            "the eligible 4B queue; keep every 4B hold inactive."
        ),
    }


def _top_projection(
    rows: list[dict[str, Any]], field: str, *, limit: int
) -> list[dict[str, Any]]:
    total = sum(int(row["projected_wikitext_bytes"]) for row in rows)
    values: Counter[str] = Counter()
    for row in rows:
        labels = [item for item in str(row[field]).split(" | ") if item]
        for label in labels:
            values[label] += int(row["projected_wikitext_bytes"])
    return [
        {
            field: label,
            "projected_wikitext_bytes": size,
            "share": _ratio(size, total),
        }
        for label, size in values.most_common(limit)
    ]


def _load_index_namespace(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    namespaces = payload.get("query", {}).get("namespaces", {})
    namespace = namespaces.get(str(INDEX_NAMESPACE_ID))
    if not isinstance(namespace, dict):
        raise ValueError("pinned namespace metadata omitted Index namespace 110")
    if namespace.get("canonical") != "Index" or namespace.get("*") != "Indice":
        raise ValueError(f"unexpected Index namespace metadata: {namespace}")
    return {
        "id": INDEX_NAMESPACE_ID,
        "canonical_name": namespace["canonical"],
        "localized_name": namespace["*"],
    }


def _load_index_targets(path: Path) -> dict[int, str]:
    targets: dict[int, str] = {}
    for row in iter_sql_insert_rows(path, "linktarget"):
        if int(row[1]) == INDEX_NAMESPACE_ID:
            targets[int(row[0])] = _decode_title(str(row[2]))
    if not targets:
        raise ValueError("linktarget dump contains no Index-namespace targets")
    return targets


def _load_index_pages(path: Path) -> dict[str, IndexPage]:
    pages: dict[str, IndexPage] = {}
    for row in iter_sql_insert_rows(path, "page"):
        if int(row[1]) != INDEX_NAMESPACE_ID:
            continue
        title = _decode_title(str(row[2]))
        pages[title] = IndexPage(
            title=title,
            page_id=int(row[0]),
            revision_id=int(row[8]),
            touched_utc=_timestamp_utc(str(row[6])),
            wikitext_bytes=int(row[9]),
            is_redirect=row[3] == "1",
        )
    if not pages:
        raise ValueError("page dump contains no Index-namespace pages")
    return pages


def _load_root_scan_links(
    path: Path,
    *,
    page_to_root: dict[int, str],
    target_titles: dict[int, str],
    progress_interval: int,
    progress: Any | None,
) -> dict[str, dict[str, set[int]]]:
    links: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    parsed = 0
    for row in iter_sql_insert_rows(path, "pagelinks"):
        parsed += 1
        page_id = int(row[0])
        root_id = page_to_root.get(page_id)
        target = target_titles.get(int(row[2]))
        if root_id and target:
            links[root_id][target].add(page_id)
        if parsed % progress_interval == 0:
            _emit(
                progress,
                f"pagelinks parsed={parsed:,} linked_roots={len(links):,}",
            )
    if not links:
        raise ValueError("pagelinks dump contains no main-to-Index links")
    return {root: dict(scans) for root, scans in links.items()}


def _acquire_metadata_file(
    config: WikisourceCandidateResolutionConfig,
    *,
    filename: str,
    expected_sha1: str,
    session: requests.Session,
    progress: Any | None,
) -> Path:
    path = config.cache_dir / filename
    if path.is_file():
        actual = _sha1_file(path)
        if actual != expected_sha1:
            raise ValueError(f"cached dump hash mismatch: {filename}")
        _emit(progress, f"dump-cache-hit {filename} bytes={path.stat().st_size:,}")
        return path

    config.cache_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{filename}.", dir=config.cache_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    url = f"{config.dump_base_url}/{filename}"
    try:
        response = session.get(url, stream=True, timeout=60)
        response.raise_for_status()
        hasher = hashlib.sha1()
        byte_count = 0
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                hasher.update(chunk)
                byte_count += len(chunk)
                if byte_count % (8 * 1024 * 1024) < len(chunk):
                    _emit(progress, f"dump-download-progress {filename} bytes={byte_count:,}")
        if hasher.hexdigest() != expected_sha1:
            raise ValueError(f"downloaded dump hash mismatch: {filename}")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    _emit(progress, f"dump-download-complete {filename} bytes={path.stat().st_size:,}")
    return path


def _validate_config(config: WikisourceCandidateResolutionConfig) -> None:
    if config.dump_date != DUMP_DATE or config.dump_base_url != DUMP_BASE_URL:
        raise ValueError("checkpoint 4B requires the pinned 20260801 dump")
    if config.progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    for value in (
        config.expected_inventory_rows,
        config.expected_hierarchy_rows,
        config.expected_candidate_rows,
    ):
        if value <= 0:
            raise ValueError("expected row counts must be positive")
    for path in (config.inventory_path, config.hierarchy_path):
        if not path.is_file():
            raise FileNotFoundError(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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


def _row_sort_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["root_title"]).casefold(), str(row["work_root_id"])


def _decode_title(value: str) -> str:
    return value.replace("_", " ")


def _index_url(title: str) -> str:
    encoded = quote(f"Indice:{title}".replace(" ", "_"), safe="()_',:–—")
    return f"https://it.wikisource.org/wiki/{encoded}"


def _timestamp_utc(value: str) -> str:
    return (
        f"{value[0:4]}-{value[4:6]}-{value[6:8]}T"
        f"{value[8:10]}:{value[10:12]}:{value[12:14]}Z"
    )


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


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


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


def _emit(progress: Any | None, message: str) -> None:
    if progress:
        progress(message)
