"""Resolve checkpoint-4C Wikisource holds without activating corpus text.

The resolver is deliberately conservative.  It freezes canonical precedence,
removes only located duplicate/protected spans, rechecks sonnet form, and fails
closed when the source scan or a modern edition contributor is not covered by
usable rights evidence.
"""

from __future__ import annotations

import bz2
import csv
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

import requests

from .gutenberg_extraction_audit import locate_reference_segments
from .gutenberg_fulltext_probe import (
    TextReference,
    _normalized_words,
    _rolling_shingle_hashes,
    fingerprint_text,
    measure_word_shingle_containment,
)
from .wikisource_page_extraction import (
    DUMP_SHA1,
    _discover_pairs,
    iter_mediawiki_dump,
    normalize_title,
    sha1_file,
)


SITE_LICENSE = "Creative Commons Attribution-Share Alike 4.0"
SITE_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/deed.it"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
NEAR_DUPLICATE_THRESHOLD = 0.8

ROOT_FIELDS = (
    "work_root_id", "root_title", "landing_page_url", "author_evidence",
    "period_bucket", "input_role", "final_broader_role", "direct_scan_title",
    "scan_rights_id", "checkpoint_4c_decision", "quality_flags",
    "source_cache_path", "source_sha256", "source_character_count",
    "canonical_reference_ids", "removed_reference_ids", "rights_decision",
    "final_decision", "resolution_reason", "retained_broader_character_count",
    "excluded_character_count", "sonnet_candidate_count", "activation_status",
)

SEGMENT_FIELDS = (
    "segment_id", "work_root_id", "source_sha256", "character_start",
    "character_end", "character_count", "segment_sha256", "segment_decision",
    "final_role", "reason", "reference_ids", "activation_status",
)

SONNET_FIELDS = (
    "candidate_id", "work_root_id", "root_title", "source_record_author",
    "poem_author", "poem_author_resolution", "period_bucket", "source_url",
    "source_scan_title", "source_kind", "stanza_pattern", "line_count",
    "first_line", "last_line", "character_start", "character_end",
    "source_text_sha256", "cleaned_text_sha256", "exact_reference_ids",
    "near_reference_ids", "protected_v6_reference_ids", "candidate_decision",
    "final_role", "activation_status",
)

RIGHTS_FIELDS = (
    "scan_rights_id", "scan_title", "index_url", "index_revision_id",
    "commons_file_title", "commons_description_url", "commons_page_id",
    "index_author", "index_editor", "index_translator", "index_illustrator",
    "index_publisher", "index_city", "index_year", "index_source",
    "commons_artist", "commons_credit", "commons_license_short_name",
    "commons_license_code", "commons_usage_terms", "commons_copyrighted",
    "commons_attribution_required", "commons_categories", "site_license",
    "site_license_url", "underlying_work_evidence", "edition_rights_evidence",
    "scan_file_rights_evidence", "required_notice", "modification_notice",
    "downstream_note", "evidence_retrieved_utc", "rights_decision", "rights_reason",
)

REVIEW_FIELDS = (
    "review_id", "work_root_id", "checkpoint_4c_decision", "resolution",
    "canonical_reference_ids", "removed_reference_ids", "rights_decision",
    "rationale", "review_status", "activation_status",
)

# A 4C flag means content was already removed by the conservative cleaner.
# Without inspecting the exact source boundary we cannot prove that a table,
# score, or formula was apparatus rather than primary content.  Therefore no
# quality flag is promoted automatically in 4D.
_SAFE_REMOVAL_FLAGS: set[str] = set()
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_YEAR = re.compile(r"(?<!\d)(1\d{3}|20\d{2})(?!\d)")
_INDEX_FIELD = re.compile(r"^\|([^=]+)=(.*)$")
_HTML_TAG = re.compile(r"<[^>]+>")
_SONNET_PATTERNS = ((4, 4, 3, 3), (8, 6), (4, 4, 6), (14,))

Progress = Callable[[str], None]


@dataclass(frozen=True)
class WikisourceReviewResolutionConfig:
    repo_root: Path
    dump_path: Path
    extraction_path: Path
    boundaries_path: Path
    review_path: Path
    inventory_path: Path
    candidate_resolution_path: Path
    scan_links_path: Path
    siteinfo_rights_path: Path
    local_cache_dir: Path
    bibit_record_manifest_path: Path
    broader_sources_manifest_path: Path
    gutenberg_previous_probe_path: Path
    gutenberg_previous_cache_dir: Path
    gutenberg_pass_1b_probe_path: Path
    gutenberg_pass_1b_cache_dir: Path
    gutenberg_resolved_record_manifest_path: Path
    protected_sonnet_manifest_path: Path
    bibit_sonnet_manifest_path: Path
    gutenberg_resolved_sonnet_manifest_path: Path
    root_decisions_path: Path
    segment_decisions_path: Path
    sonnet_decisions_path: Path
    scan_rights_path: Path
    review_resolution_path: Path
    json_report_path: Path
    markdown_report_path: Path
    expected_dump_sha1: str = DUMP_SHA1
    expected_root_count: int = 4_641
    expected_review_count: int = 2_095
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD
    progress_interval: int = 100
    request_timeout: float = 60.0


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    decision: str
    role: str
    reason: str
    reference_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Poem:
    start: int
    end: int
    stanza_pattern: str
    source_kind: str
    cleaned_text: str


def run_wikisource_review_resolution(
    config: WikisourceReviewResolutionConfig,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Freeze all root, segment, sonnet, review, and source-rights decisions."""

    _validate_config(config)
    started = monotonic()
    extraction = _read_csv(config.extraction_path)
    reviews = _read_csv(config.review_path)
    inventory = _unique(_read_csv(config.inventory_path), "work_root_id")
    resolution = _unique(_read_csv(config.candidate_resolution_path), "work_root_id")
    scans = [
        row for row in _read_csv(config.scan_links_path)
        if row["work_root_id"] in {item["work_root_id"] for item in extraction}
    ]
    if len(extraction) != config.expected_root_count:
        raise ValueError(f"expected {config.expected_root_count} roots, found {len(extraction)}")
    if len(reviews) != config.expected_review_count:
        raise ValueError(f"expected {config.expected_review_count} reviews, found {len(reviews)}")
    if {row["work_root_id"] for row in reviews} != {
        row["work_root_id"] for row in extraction
        if row["checkpoint_4c_decision"] != "eligible_inactive_pending_processed_build"
    }:
        raise ValueError("4C review ledger does not match held extraction roots")

    scan_by_root = _scan_rows_by_root(scans)
    index_pages = _load_index_pages(config, scans, progress=progress)
    commons = _load_commons_metadata(config, index_pages, progress=progress)
    rights_rows, rights_by_title = _resolve_scan_rights(
        config, index_pages=index_pages, commons=commons, scans=scans
    )
    references = _load_text_references(config)
    protected = _load_protected_sonnets(config)
    sonnet_references = _load_sonnet_references(config)

    rows_by_id = {row["work_root_id"]: row for row in extraction}
    texts = {
        root_id: _read_4c_root_cache(config.repo_root / row["local_text_cache_path"])
        for root_id, row in rows_by_id.items()
    }
    for root_id, text in texts.items():
        row = rows_by_id[root_id]
        if len(text) != int(row["extracted_character_count"]):
            raise ValueError(f"4C root cache length mismatch: {root_id}")
        if _normalized_sha(text) != row["normalized_word_sha256"]:
            raise ValueError(f"4C root normalized hash mismatch: {root_id}")

    state: dict[str, dict[str, Any]] = {}
    for row in extraction:
        root_id = row["work_root_id"]
        scan_rows = scan_by_root.get(root_id, [])
        scan_title = scan_rows[0]["scan_title"] if len(scan_rows) == 1 else ""
        rights = rights_by_title.get(scan_title)
        flags = {value for value in row["quality_flags"].split(";") if value}
        decision = "eligible_for_canonical_review"
        reason = "4C extraction passed or contains only explicitly removable apparatus."
        if len(scan_rows) != 1:
            decision, reason = "exclude_scan_identity_unresolved", "Root does not resolve to exactly one source scan."
        elif rights is None or rights["rights_decision"] != "rights_pass":
            decision, reason = "exclude_source_rights_unresolved", (
                rights["rights_reason"] if rights else "No source-scan rights row exists."
            )
        elif row["revision_mismatches"]:
            decision, reason = "exclude_revision_mismatch", "Pinned current revision did not match extraction input."
        elif row["missing_proofread_pages"]:
            decision, reason = "exclude_incomplete_transcription", "Missing requested transcription pages make the work incomplete."
        elif not texts[root_id].strip():
            decision, reason = "exclude_empty_extraction", "No primary text remains."
        elif row["unresolved_markup"]:
            decision, reason = "exclude_unresolved_markup", "Unknown markup may contain or corrupt primary text."
        elif row["rendered_validation_status"] == "hold_rendered_validation":
            decision, reason = "exclude_rendered_boundary_failure", "Local extraction failed rendered-text containment."
        elif flags - _SAFE_REMOVAL_FLAGS:
            decision, reason = "exclude_quality_language_or_editorial_hold", (
                "Non-removable quality, language-variety, transcription-gap, or editorial evidence remains."
            )
        state[root_id] = {
            "decision": decision, "reason": reason, "scan_title": scan_title,
            "rights": rights, "canonical": set(), "removed": set(), "spans": [],
        }

    # Existing resolved corpora have precedence.  A reference-only containment
    # is surgically removed; candidate containment excludes the Wikisource root.
    for index, row in enumerate(extraction, start=1):
        root_id = row["work_root_id"]
        current = state[root_id]
        if current["decision"] != "eligible_for_canonical_review":
            continue
        for field in _cross_overlap_fields():
            for metric in _parse_cross_metrics(row[field]):
                reference = references.get(metric["reference_id"])
                if reference is None:
                    current["decision"] = "exclude_unresolved_canonical_reference"
                    current["reason"] = f"Missing frozen comparison reference {metric['reference_id']}."
                    break
                if metric["candidate"] >= config.near_duplicate_threshold:
                    current["decision"] = "exclude_canonical_cross_corpus_duplicate"
                    current["canonical"].add(metric["reference_id"])
                    current["reason"] = "An already resolved project source contains this Wikisource root."
                    break
                if metric["reference"] >= config.near_duplicate_threshold:
                    located = locate_reference_segments(texts[root_id], reference.read_text())
                    if not located:
                        current["decision"] = "exclude_unlocatable_embedded_duplicate"
                        current["reason"] = f"Embedded canonical reference could not be bounded: {metric['reference_id']}."
                        break
                    for start, end, _anchors in located:
                        current["spans"].append(_Span(
                            start, end, "exclude_cross_corpus_duplicate_segment", "excluded",
                            "Located text already exists in a higher-precedence corpus.",
                            (metric["reference_id"],),
                        ))
                        current["removed"].add(metric["reference_id"])
            if current["decision"] != "eligible_for_canonical_review":
                break
        if progress and (index == 1 or index % config.progress_interval == 0):
            _emit_progress(progress, "cross-canonical", index, len(extraction), started)

    _resolve_internal_exact(extraction, state, texts)
    _resolve_internal_near(extraction, state, texts, config.near_duplicate_threshold)

    root_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    sonnet_rows: list[dict[str, Any]] = []
    candidate_payloads: list[tuple[dict[str, Any], str]] = []
    for index, row in enumerate(sorted(extraction, key=lambda item: _root_number(item["work_root_id"])), start=1):
        root_id = row["work_root_id"]
        text = texts[root_id]
        current = state[root_id]
        poems: list[_Poem] = []
        if current["decision"] == "eligible_for_canonical_review" and row["proposed_role"] in {
            "historical_non_sonnet_poetry", "standard_sonnets"
        }:
            poems = discover_wikisource_sonnet_candidates(
                text, metadata_sonnet=row["proposed_role"] == "standard_sonnets"
            )

        # Protected V6 material is always removed.  A large otherwise-usable
        # root is retained only when every protected reference is bounded.
        for reference_id in _metric_ids(row["protected_v6_overlap_metrics"]):
            reference_text = protected.get(reference_id)
            located = locate_reference_segments(text, reference_text) if reference_text else []
            if not located:
                current["decision"] = "exclude_protected_v6_overlap_unbounded"
                current["reason"] = "Protected V6 overlap could not be isolated safely."
                current["canonical"].add(f"protected_v6:{reference_id}")
                break
            for start, end, _anchors in located:
                current["spans"].append(_Span(
                    start, end, "exclude_protected_v6_segment", "excluded",
                    "Protected V6 validation/test sonnet quarantine.",
                    (f"protected_v6:{reference_id}",),
                ))
                current["removed"].add(f"protected_v6:{reference_id}")

        if current["decision"] == "eligible_for_canonical_review":
            for poem_index, poem in enumerate(poems, start=1):
                candidate_id = f"{root_id}:sonnet{poem_index:04d}"
                candidate = _sonnet_row(candidate_id, row, current["scan_title"], poem)
                candidate["source_text_sha256"] = _sha(text[poem.start:poem.end])
                candidate_payloads.append((candidate, poem.cleaned_text))
                current["spans"].append(_Span(
                    poem.start, poem.end, "quarantine_sonnet_candidate", "standard_sonnets",
                    "Verified fourteen-line poem is isolated from broader text.", (candidate_id,),
                ))
            current["decision"] = "eligible_inactive_processed_build"
            current["reason"] = "Rights, quality, canonicalization, and protected-material gates passed."

        spans = _partition(
            len(text), current["spans"],
            default_role=(
                "historical_non_sonnet_poetry"
                if row["proposed_role"] == "standard_sonnets"
                else row["proposed_role"]
            ),
            include=current["decision"] == "eligible_inactive_processed_build",
        )
        if row["proposed_role"] == "standard_sonnets" and poems:
            broader_words = sum(
                len(_WORD.findall(text[span.start:span.end]))
                for span in spans if span.decision == "include_broader_text"
            )
            if broader_words < 100:
                spans = [
                    _Span(
                        span.start, span.end, "exclude_sonnet_wrapper_or_heading", "excluded",
                        "Short title, numbering, or wrapper text surrounding isolated sonnets.",
                    ) if span.decision == "include_broader_text" else span
                    for span in spans
                ]
        if current["decision"] != "eligible_inactive_processed_build":
            spans = [_Span(0, len(text), "exclude_source", "excluded", current["reason"], tuple(sorted(current["canonical"])))] if text else []
        retained = sum(span.end - span.start for span in spans if span.decision == "include_broader_text")
        for segment_index, span in enumerate(spans, start=1):
            payload = text[span.start:span.end]
            segment_rows.append({
                "segment_id": f"{root_id}:seg{segment_index:04d}",
                "work_root_id": root_id,
                "source_sha256": _sha(text),
                "character_start": span.start,
                "character_end": span.end,
                "character_count": len(payload),
                "segment_sha256": _sha(payload),
                "segment_decision": span.decision,
                "final_role": span.role,
                "reason": span.reason,
                "reference_ids": ";".join(span.reference_ids),
                "activation_status": "inactive_pending_cross_archive_freeze",
            })
        rights = current["rights"] or {}
        root_rows.append({
            "work_root_id": root_id, "root_title": row["root_title"],
            "landing_page_url": row["landing_page_url"], "author_evidence": row["author_evidence"],
            "period_bucket": row["period_bucket"], "input_role": row["proposed_role"],
            "final_broader_role": (
                "historical_non_sonnet_poetry" if row["proposed_role"] == "standard_sonnets" else row["proposed_role"]
            ),
            "direct_scan_title": current["scan_title"], "scan_rights_id": rights.get("scan_rights_id", ""),
            "checkpoint_4c_decision": row["checkpoint_4c_decision"], "quality_flags": row["quality_flags"],
            "source_cache_path": row["local_text_cache_path"], "source_sha256": _sha(text),
            "source_character_count": len(text), "canonical_reference_ids": ";".join(sorted(current["canonical"])),
            "removed_reference_ids": ";".join(sorted(current["removed"])),
            "rights_decision": rights.get("rights_decision", "rights_missing"),
            "final_decision": current["decision"], "resolution_reason": current["reason"],
            "retained_broader_character_count": retained,
            "excluded_character_count": len(text) - retained,
            "sonnet_candidate_count": len(poems) if current["decision"] == "eligible_inactive_processed_build" else 0,
            "activation_status": "inactive_pending_cross_archive_freeze",
        })
        if progress and (index == 1 or index % config.progress_interval == 0 or index == len(extraction)):
            _emit_progress(progress, "root-resolution", index, len(extraction), started)

    sonnet_rows = _resolve_sonnet_duplicates(
        candidate_payloads, sonnet_references, threshold=config.near_duplicate_threshold
    )
    _apply_sonnet_segment_decisions(segment_rows, sonnet_rows)
    _reconcile_root_sonnet_counts(root_rows, sonnet_rows)
    _resolve_post_segmentation_duplicates(
        root_rows, segment_rows, sonnet_rows, texts,
        threshold=config.near_duplicate_threshold,
    )
    resolved_reviews = _build_review_resolutions(reviews, {row["work_root_id"]: row for row in root_rows})
    _validate_outputs(config, root_rows, segment_rows, sonnet_rows, rights_rows, resolved_reviews)

    _write_csv(config.root_decisions_path, ROOT_FIELDS, root_rows)
    _write_csv(config.segment_decisions_path, SEGMENT_FIELDS, segment_rows)
    _write_csv(config.sonnet_decisions_path, SONNET_FIELDS, sonnet_rows)
    _write_csv(config.scan_rights_path, RIGHTS_FIELDS, rights_rows)
    _write_csv(config.review_resolution_path, REVIEW_FIELDS, resolved_reviews)
    report = _build_report(config, root_rows, segment_rows, sonnet_rows, rights_rows, resolved_reviews)
    _write_json(config.json_report_path, report)
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(render_markdown(report), encoding="utf-8")
    return report


def discover_wikisource_sonnet_candidates(text: str, *, metadata_sonnet: bool) -> list[_Poem]:
    """Return conservative fourteen-line blocks while preserving source offsets."""

    blocks = _line_blocks(text)
    result: list[_Poem] = []
    used: set[tuple[int, int]] = set()
    for index in range(len(blocks)):
        for pattern in _SONNET_PATTERNS:
            selected = blocks[index:index + len(pattern)]
            if len(selected) != len(pattern) or tuple(len(item[2]) for item in selected) != pattern:
                continue
            lines = [line for _start, _end, values in selected for line in values]
            if not _verse_like(lines):
                continue
            start, end = lines[0][0], lines[-1][1]
            if (start, end) in used:
                continue
            previous = ""
            if index:
                previous = " ".join(value[2] for value in blocks[index - 1][2]).casefold()
            explicit = "sonett" in previous
            if not metadata_sonnet and not explicit and pattern != (14,):
                continue
            cleaned = "\n".join(_clean_verse_line(value[2]) for value in lines).strip() + "\n"
            if not metadata_sonnet and not explicit and not _plausible_sonnet_rhyme(cleaned):
                continue
            result.append(_Poem(start, end, "-".join(map(str, pattern)), (
                "explicit_sonnet_heading" if explicit else
                "source_metadata_sonnet" if metadata_sonnet else
                "structural_14_line_in_poetry_root"
            ), cleaned))
            used.add((start, end))
            break
    return sorted(result, key=lambda item: (item.start, item.end))


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Italian Wikisource Review Resolution",
        "",
        "## Result",
        "",
        f"Checkpoint 4D resolves all {report['review_count']:,} checkpoint-4C review rows and accounts for all {report['root_count']:,} extracted roots.",
        "",
        f"- Source scans with rights evidence: {report['scan_rights_count']:,}.",
        f"- Rights-passing scans: {report['rights_pass_scan_count']:,}.",
        f"- Roots eligible for an inactive processed build: {report['eligible_root_count']:,}.",
        f"- Retained broader-text characters: {report['retained_broader_character_count']:,}.",
        f"- Verified sonnet candidates: {report['eligible_sonnet_count']:,}.",
        "",
        "## Final Root Decisions",
        "",
        "| Decision | Roots |",
        "| --- | ---: |",
    ]
    lines.extend(f"| `{key}` | {value:,} |" for key, value in report["root_decision_counts"].items())
    lines.extend([
        "", "## Boundary", "",
        "These decisions remain inactive. They create no V7 split, training mixture, GPU job, or cache deletion. "
        "Conditioned and checkpoint-4B-held material never enters the standard-Italian queue.",
    ])
    return "\n".join(lines).rstrip() + "\n"


def _load_index_pages(
    config: WikisourceReviewResolutionConfig,
    scans: list[dict[str, str]],
    *, progress: Progress | None,
) -> dict[str, dict[str, Any]]:
    cache_path = config.local_cache_dir / "index_pages.json"
    wanted = {normalize_title(f"Indice:{row['scan_title']}") for row in scans}
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("dump_sha1") == config.expected_dump_sha1 and set(payload.get("pages", {})) == wanted:
            return payload["pages"]
    pages: dict[str, dict[str, Any]] = {}
    for page in iter_mediawiki_dump(config.dump_path, selected_titles=wanted, progress=progress):
        pages[normalize_title(page.title)] = {
            "page_id": page.page_id, "revision_id": page.revision_id,
            "timestamp": page.timestamp, "wikitext": page.text,
        }
    if set(pages) != wanted:
        raise ValueError(f"missing {len(wanted - set(pages))} pinned Index pages")
    config.local_cache_dir.mkdir(parents=True, exist_ok=True)
    _write_json(cache_path, {"dump_sha1": config.expected_dump_sha1, "pages": pages})
    return pages


def _load_commons_metadata(
    config: WikisourceReviewResolutionConfig,
    index_pages: dict[str, dict[str, Any]],
    *, progress: Progress | None,
) -> dict[str, dict[str, Any]]:
    cache_path = config.local_cache_dir / "commons_file_metadata.json"
    file_titles = sorted(f"File:{title.split(':', 1)[1]}" for title in index_pages)
    cached: dict[str, dict[str, Any]] = {}
    retrieved = ""
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        cached = payload.get("files", {})
        retrieved = payload.get("retrieved_utc", "")
    missing = [title for title in file_titles if title not in cached]
    if missing:
        session = requests.Session()
        session.headers.update({"User-Agent": "portfolio-transformer-poetry/4D (corpus rights audit)"})
        for offset in range(0, len(missing), 50):
            batch = missing[offset:offset + 50]
            response = session.get(COMMONS_API, params={
                "action": "query", "format": "json", "formatversion": "2",
                "prop": "imageinfo", "iiprop": "url|extmetadata",
                "titles": "|".join(batch),
            }, timeout=config.request_timeout)
            response.raise_for_status()
            body = response.json()
            if "error" in body:
                raise RuntimeError(f"Commons API error: {body['error']}")
            for page in body.get("query", {}).get("pages", []):
                cached[normalize_title(page["title"])] = page
            if progress:
                progress(f"commons-rights completed={min(offset + 50, len(missing)):,}/{len(missing):,}")
        retrieved = _utc_now()
        _write_json(cache_path, {"retrieved_utc": retrieved, "files": cached})
    if not retrieved:
        retrieved = _utc_now()
    cached["__retrieved_utc__"] = {"value": retrieved}
    return cached


def _resolve_scan_rights(
    config: WikisourceReviewResolutionConfig,
    *, index_pages: dict[str, dict[str, Any]],
    commons: dict[str, dict[str, Any]],
    scans: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    scan_revision = {}
    for row in scans:
        scan_revision.setdefault(row["scan_title"], row)
    result = []
    by_title = {}
    retrieved = commons.get("__retrieved_utc__", {}).get("value", "")
    for scan_title, link in sorted(scan_revision.items(), key=lambda item: item[0].casefold()):
        index_title = normalize_title(f"Indice:{scan_title}")
        page = index_pages.get(index_title, {})
        fields = _parse_index_fields(page.get("wikitext", ""))
        file_title = normalize_title(f"File:{scan_title}")
        common = commons.get(file_title, {})
        imageinfo = (common.get("imageinfo") or [{}])[0]
        metadata = imageinfo.get("extmetadata", {})
        value = lambda key: _clean_metadata(metadata.get(key, {}).get("value", ""))
        license_name = value("LicenseShortName")
        license_code = value("License")
        copyrighted = value("Copyrighted")
        year_match = _YEAR.search(fields.get("anno", ""))
        modern_contributors = ";".join(filter(None, (
            fields.get("curatore", ""), fields.get("traduttore", ""), fields.get("illustratore", "")
        )))
        decision = "rights_pass"
        reason = "Commons marks the exact scan file non-copyrighted/public domain and Wikisource pins CC BY-SA 4.0 transcription terms."
        if common.get("missing") or not imageinfo:
            decision, reason = "rights_hold_missing_commons_file", "Exact Commons source file or image metadata is missing."
        elif copyrighted.casefold() not in {"false", "no", "0"} or "public domain" not in license_name.casefold():
            decision, reason = "rights_hold_incompatible_or_unclear_scan_license", "Commons does not explicitly mark the exact scan file public domain and non-copyrighted."
        elif not fields.get("autore") or not year_match or not fields.get("fonte"):
            decision, reason = "rights_hold_incomplete_index_provenance", "Index metadata lacks author, year, or source evidence."
        elif int(year_match.group(1)) > 1930 and modern_contributors:
            decision, reason = "rights_hold_modern_edition_contributor", "Post-1930 edition names a curator, translator, or illustrator whose contribution needs separate rights evidence."
        row = {
            "scan_rights_id": f"itws-scan:{link['scan_page_id']}", "scan_title": scan_title,
            "index_url": link["scan_url"], "index_revision_id": page.get("revision_id", ""),
            "commons_file_title": file_title, "commons_description_url": imageinfo.get("descriptionurl", ""),
            "commons_page_id": common.get("pageid", ""), "index_author": fields.get("autore", ""),
            "index_editor": fields.get("curatore", ""), "index_translator": fields.get("traduttore", ""),
            "index_illustrator": fields.get("illustratore", ""), "index_publisher": fields.get("editore", ""),
            "index_city": fields.get("città", ""), "index_year": fields.get("anno", ""),
            "index_source": fields.get("fonte", ""), "commons_artist": value("Artist"),
            "commons_credit": value("Credit"), "commons_license_short_name": license_name,
            "commons_license_code": license_code, "commons_usage_terms": value("UsageTerms"),
            "commons_copyrighted": copyrighted, "commons_attribution_required": value("AttributionRequired"),
            "commons_categories": value("Categories"), "site_license": SITE_LICENSE,
            "site_license_url": SITE_LICENSE_URL,
            "underlying_work_evidence": f"Wikisource work period={fields.get('anno', '')}; author={fields.get('autore', '')}; Commons license={license_name}",
            "edition_rights_evidence": f"editor={fields.get('curatore', '')}; translator={fields.get('traduttore', '')}; illustrator={fields.get('illustratore', '')}; publisher={fields.get('editore', '')}",
            "scan_file_rights_evidence": imageinfo.get("descriptionurl", ""),
            "required_notice": "Attribute Italian Wikisource contributors through the stable work page/history; retain the CC BY-SA 4.0 license link and exact source-scan credit.",
            "modification_notice": "MediaWiki/ProofreadPage markup and identified apparatus were removed; spelling and punctuation of retained primary text were preserved.",
            "downstream_note": "Processed text remains inactive pending cross-archive and training-lineage freeze; redistributed adaptations must preserve applicable ShareAlike obligations.",
            "evidence_retrieved_utc": retrieved, "rights_decision": decision, "rights_reason": reason,
        }
        result.append(row)
        by_title[scan_title] = row
    return result, by_title


def _resolve_internal_exact(rows: list[dict[str, str]], state: dict[str, dict[str, Any]], texts: dict[str, str]) -> None:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["normalized_word_sha256"] and int(row["extracted_word_count"]):
            groups[row["normalized_word_sha256"]].append(row["work_root_id"])
    for ids in groups.values():
        active = [root_id for root_id in ids if state[root_id]["decision"] == "eligible_for_canonical_review"]
        if len(active) < 2:
            continue
        winner = min(active, key=lambda root_id: (-len(texts[root_id]), _root_number(root_id)))
        for root_id in active:
            if root_id == winner:
                continue
            state[root_id]["decision"] = "exclude_internal_exact_duplicate"
            state[root_id]["canonical"].add(winner)
            state[root_id]["reason"] = "Deterministic complete/long/stable-ID precedence selected another exact Wikisource root."


def _resolve_internal_near(
    rows: list[dict[str, str]], state: dict[str, dict[str, Any]], texts: dict[str, str], threshold: float
) -> None:
    pairs = set()
    for row in rows:
        left = row["work_root_id"]
        for value in row["internal_near_duplicate_metrics"].split(";"):
            right = value.split("|", 1)[0]
            if right:
                pairs.add(tuple(sorted((left, right))))
    for left, right in sorted(pairs):
        if state[left]["decision"] != "eligible_for_canonical_review" or state[right]["decision"] != "eligible_for_canonical_review":
            continue
        metric = measure_word_shingle_containment(texts[left], texts[right])
        if metric["left_containment"] < threshold and metric["right_containment"] < threshold:
            continue
        if metric["left_containment"] >= threshold and metric["right_containment"] < threshold:
            loser, winner = left, right
        elif metric["right_containment"] >= threshold and metric["left_containment"] < threshold:
            loser, winner = right, left
        else:
            winner = min((left, right), key=lambda root_id: (-len(texts[root_id]), _root_number(root_id)))
            loser = right if winner == left else left
        state[loser]["decision"] = "exclude_internal_near_duplicate"
        state[loser]["canonical"].add(winner)
        state[loser]["reason"] = "Directional containment and deterministic completeness precedence selected another Wikisource root."


def _resolve_sonnet_duplicates(
    candidates: list[tuple[dict[str, Any], str]],
    references: dict[str, str],
    *, threshold: float,
) -> list[dict[str, Any]]:
    postings: dict[int, set[str]] = defaultdict(set)
    reference_hashes = {}
    for reference_id, text in references.items():
        hashes = set(_rolling_shingle_hashes(_normalized_words(text)))
        reference_hashes[reference_id] = hashes
        for value in hashes:
            postings[value].add(reference_id)
    seen_exact: dict[str, str] = {}
    seen_text: dict[str, str] = {}
    result = []
    for row, text in sorted(candidates, key=lambda item: item[0]["candidate_id"]):
        exact_ids = []
        near_ids = []
        protected_ids = []
        digest = _normalized_sha(text)
        if digest in seen_exact:
            exact_ids.append(seen_exact[digest])
        possible = Counter(reference_id for value in set(_rolling_shingle_hashes(_normalized_words(text))) for reference_id in postings.get(value, ()))
        for reference_id, _count in possible.most_common(30):
            metric = measure_word_shingle_containment(text, references[reference_id])
            if metric["left_containment"] >= threshold:
                (exact_ids if _normalized_sha(references[reference_id]) == digest else near_ids).append(reference_id)
                if reference_id.startswith("v6_protected:"):
                    protected_ids.append(reference_id)
        for earlier_id, earlier_text in seen_text.items():
            metric = measure_word_shingle_containment(text, earlier_text)
            if metric["left_containment"] >= threshold and metric["right_containment"] >= threshold:
                near_ids.append(earlier_id)
        if protected_ids:
            decision = "exclude_protected_v6_sonnet"
        elif exact_ids or near_ids:
            decision = "exclude_canonical_sonnet_duplicate"
        else:
            decision = "eligible_standard_sonnet_inactive_pending_v7"
            seen_exact[digest] = row["candidate_id"]
            seen_text[row["candidate_id"]] = text
        row.update({
            "exact_reference_ids": ";".join(sorted(set(exact_ids))),
            "near_reference_ids": ";".join(sorted(set(near_ids))),
            "protected_v6_reference_ids": ";".join(sorted(set(protected_ids))),
            "candidate_decision": decision, "final_role": "standard_sonnets",
            "activation_status": "inactive_pending_v7_attribution_and_split_freeze",
        })
        result.append(row)
    return result


def _sonnet_row(candidate_id: str, root: dict[str, str], scan_title: str, poem: _Poem) -> dict[str, Any]:
    lines = poem.cleaned_text.strip().splitlines()
    author = root["author_evidence"] if root["author_evidence"].casefold() not in {"", "autori vari"} else ""
    return {
        "candidate_id": candidate_id, "work_root_id": root["work_root_id"],
        "root_title": root["root_title"], "source_record_author": root["author_evidence"],
        "poem_author": author,
        "poem_author_resolution": "root_metadata_proxy_pending_v7_review" if author else "unresolved",
        "period_bucket": root["period_bucket"], "source_url": root["landing_page_url"],
        "source_scan_title": scan_title, "source_kind": poem.source_kind,
        "stanza_pattern": poem.stanza_pattern, "line_count": len(lines),
        "first_line": lines[0], "last_line": lines[-1], "character_start": poem.start,
        "character_end": poem.end, "source_text_sha256": "", "cleaned_text_sha256": _sha(poem.cleaned_text),
        "exact_reference_ids": "", "near_reference_ids": "", "protected_v6_reference_ids": "",
        "candidate_decision": "", "final_role": "standard_sonnets", "activation_status": "inactive",
    }


def _apply_sonnet_segment_decisions(segments: list[dict[str, Any]], sonnets: list[dict[str, Any]]) -> None:
    by_id = {row["candidate_id"]: row for row in sonnets}
    for segment in segments:
        candidate_ids = [value for value in segment["reference_ids"].split(";") if value in by_id]
        if not candidate_ids:
            continue
        decisions = {by_id[candidate_id]["candidate_decision"] for candidate_id in candidate_ids}
        if decisions == {"eligible_standard_sonnet_inactive_pending_v7"} and segment["segment_decision"] == "quarantine_sonnet_candidate":
            segment["segment_decision"] = "materialize_standard_sonnet_inactive"
        else:
            segment["segment_decision"] = "exclude_sonnet_duplicate_or_protected"
            segment["final_role"] = "excluded"
        segment["reason"] = ";".join(sorted(decisions))


def _reconcile_root_sonnet_counts(roots: list[dict[str, Any]], sonnets: list[dict[str, Any]]) -> None:
    counts = Counter(row["work_root_id"] for row in sonnets)
    for root in roots:
        root["sonnet_candidate_count"] = counts[root["work_root_id"]]


def _resolve_post_segmentation_duplicates(
    roots: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    sonnets: list[dict[str, Any]],
    source_texts: dict[str, str],
    *,
    threshold: float,
) -> None:
    """Deduplicate broader remainders after sonnet/apparatus removal."""

    roots_by_id = {row["work_root_id"]: row for row in roots}
    segments_by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in segments:
        segments_by_root[row["work_root_id"]].append(row)

    def materialized_text(root_id: str) -> str:
        parts = [
            source_texts[root_id][int(row["character_start"]):int(row["character_end"])]
            for row in sorted(segments_by_root[root_id], key=lambda item: int(item["character_start"]))
            if row["segment_decision"] == "include_broader_text"
        ]
        return "\n".join(parts).rstrip() + "\n" if any(part.strip() for part in parts) else ""

    texts = {
        row["work_root_id"]: materialized_text(row["work_root_id"])
        for row in roots
        if row["final_decision"] == "eligible_inactive_processed_build"
        and int(row["retained_broader_character_count"])
    }
    fingerprints = {root_id: fingerprint_text(text)[0] for root_id, text in texts.items()}
    exact: dict[str, list[str]] = defaultdict(list)
    for root_id, value in fingerprints.items():
        if value.word_count:
            exact[value.normalized_word_sha256].append(root_id)
    losers: dict[str, str] = {}
    for ids in exact.values():
        if len(ids) < 2:
            continue
        winner = min(ids, key=_root_number)
        for root_id in ids:
            if root_id != winner:
                losers[root_id] = winner
    active_fingerprints = {
        root_id: value for root_id, value in fingerprints.items() if root_id not in losers
    }
    for left, right in sorted(_discover_pairs(active_fingerprints)):
        metric = measure_word_shingle_containment(texts[left], texts[right])
        if metric["containment"] < threshold:
            continue
        if metric["left_containment"] >= threshold and metric["right_containment"] < threshold:
            loser, winner = left, right
        elif metric["right_containment"] >= threshold and metric["left_containment"] < threshold:
            loser, winner = right, left
        else:
            winner = min((left, right), key=lambda root_id: (-len(texts[root_id]), _root_number(root_id)))
            loser = right if winner == left else left
        losers.setdefault(loser, winner)

    eligible_sonnet_roots = {
        row["work_root_id"] for row in sonnets
        if row["candidate_decision"] == "eligible_standard_sonnet_inactive_pending_v7"
    }
    for loser, winner in sorted(losers.items()):
        for segment in segments_by_root[loser]:
            if segment["segment_decision"] == "include_broader_text":
                segment["segment_decision"] = "exclude_post_segmentation_duplicate"
                segment["final_role"] = "excluded"
                segment["reason"] = "Broader remainder duplicates a canonical post-segmentation Wikisource record."
                segment["reference_ids"] = winner
        root = roots_by_id[loser]
        root["canonical_reference_ids"] = ";".join(sorted(set(filter(None, root["canonical_reference_ids"].split(";"))) | {winner}))
        root["retained_broader_character_count"] = 0
        root["excluded_character_count"] = root["source_character_count"]
        if loser in eligible_sonnet_roots:
            root["final_decision"] = "eligible_sonnets_only_inactive"
            root["resolution_reason"] = "Duplicate broader remainder excluded; unique verified sonnet artifacts remain eligible and inactive."
        else:
            root["final_decision"] = "exclude_post_segmentation_duplicate"
            root["resolution_reason"] = "Broader remainder duplicates a canonical post-segmentation Wikisource record."
    for root in roots:
        if (
            root["final_decision"] == "eligible_inactive_processed_build"
            and not materialized_text(root["work_root_id"]).strip()
        ):
            for segment in segments_by_root[root["work_root_id"]]:
                if segment["segment_decision"] == "include_broader_text":
                    segment["segment_decision"] = "exclude_no_unique_material_after_segmentation"
                    segment["final_role"] = "excluded"
                    segment["reason"] = "Retained span contains no non-whitespace material after segmentation."
            root["retained_broader_character_count"] = 0
            root["excluded_character_count"] = root["source_character_count"]
            if root["work_root_id"] in eligible_sonnet_roots:
                root["final_decision"] = "eligible_sonnets_only_inactive"
                root["resolution_reason"] = "Only verified unique sonnet artifacts remain after wrapper and duplicate removal."
            else:
                root["final_decision"] = "exclude_no_unique_material_after_segmentation"
                root["resolution_reason"] = "No unique broader text or eligible sonnet remains after segmentation."


def _build_review_resolutions(reviews: list[dict[str, str]], roots: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for review in reviews:
        root = roots[review["work_root_id"]]
        result.append({
            "review_id": review["review_id"], "work_root_id": review["work_root_id"],
            "checkpoint_4c_decision": review["checkpoint_4c_decision"], "resolution": root["final_decision"],
            "canonical_reference_ids": root["canonical_reference_ids"],
            "removed_reference_ids": root["removed_reference_ids"], "rights_decision": root["rights_decision"],
            "rationale": root["resolution_reason"], "review_status": "resolved",
            "activation_status": "inactive_pending_cross_archive_freeze",
        })
    return result


def _partition(length: int, removals: list[_Span], *, default_role: str, include: bool) -> list[_Span]:
    if not length:
        return []
    if not include:
        return [_Span(0, length, "exclude_source", "excluded", "Source-level exclusion.")]
    merged = _merge_spans(removals)
    result = []
    cursor = 0
    for span in merged:
        if span.start > cursor:
            result.append(_Span(cursor, span.start, "include_broader_text", default_role, "Retained primary text."))
        result.append(span)
        cursor = max(cursor, span.end)
    if cursor < length:
        result.append(_Span(cursor, length, "include_broader_text", default_role, "Retained primary text."))
    return [span for span in result if span.end > span.start]


def _merge_spans(spans: Iterable[_Span]) -> list[_Span]:
    result: list[_Span] = []
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if not result or span.start >= result[-1].end:
            result.append(span)
            continue
        prior = result[-1]
        decisions = {prior.decision, span.decision}
        decision = "exclude_overlapping_quarantine_segments" if len(decisions) > 1 else prior.decision
        result[-1] = _Span(
            prior.start, max(prior.end, span.end), decision,
            prior.role if prior.role == span.role else "excluded",
            "; ".join(sorted({prior.reason, span.reason})),
            tuple(sorted(set(prior.reference_ids) | set(span.reference_ids))),
        )
    return result


def _line_blocks(text: str) -> list[tuple[int, int, list[tuple[int, int, str]]]]:
    blocks = []
    lines = []
    position = 0
    for raw in text.splitlines(keepends=True):
        end = position + len(raw)
        value = raw.rstrip("\r\n")
        if value.strip():
            lines.append((position, end, value))
        elif lines:
            blocks.append((lines[0][0], lines[-1][1], lines))
            lines = []
        position = end
    if lines:
        blocks.append((lines[0][0], lines[-1][1], lines))
    return blocks


def _verse_like(lines: list[tuple[int, int, str]]) -> bool:
    if len(lines) != 14:
        return False
    values = [_clean_verse_line(line[2]) for line in lines]
    counts = [len(_WORD.findall(value)) for value in values]
    return all(values) and sum(2 <= count <= 20 for count in counts) >= 12 and sum(len(value) <= 110 for value in values) >= 12


def _clean_verse_line(value: str) -> str:
    return re.sub(r"^\s*:+\s*", "", value).strip()


def _plausible_sonnet_rhyme(text: str) -> bool:
    """Apply a documented loose Italian end-rhyme gate to inferred blocks."""

    import unicodedata

    keys = []
    for line in text.splitlines():
        normalized = "".join(
            character for character in unicodedata.normalize("NFKD", line.casefold())
            if not unicodedata.combining(character)
        )
        words = _WORD.findall(normalized)
        keys.append(words[-1][-2:] if words else "")
    octave = Counter(keys[:8])
    sestet = Counter(keys[8:])
    return sum(count >= 2 for count in octave.values()) >= 2 and any(
        count >= 2 for count in sestet.values()
    )


def _load_text_references(config: WikisourceReviewResolutionConfig) -> dict[str, TextReference]:
    result: dict[str, TextReference] = {}
    for row in _read_csv(config.bibit_record_manifest_path):
        if row["artifact_status"] == "text_materialized" and row["shard_path"]:
            result[f"bibit:{row['object_id']}"] = TextReference(
                f"bibit:{row['object_id']}", "bibit", config.repo_root / row["shard_path"],
                int(row["byte_start"]), int(row["byte_end"]),
            )
    for row in _read_csv(config.broader_sources_manifest_path):
        relative = row.get("expected_clean_text_path", "")
        if relative and (config.repo_root / relative).is_file():
            result[f"current:{row['source_id']}"] = TextReference(
                f"current:{row['source_id']}", "current", config.repo_root / relative
            )
    _add_probe_refs(result, config.gutenberg_previous_probe_path, config.gutenberg_previous_cache_dir, "gutenberg_previous")
    _add_probe_refs(result, config.gutenberg_pass_1b_probe_path, config.gutenberg_pass_1b_cache_dir, "gutenberg_pass1b")
    for row in _read_csv(config.gutenberg_resolved_record_manifest_path):
        if row["artifact_status"] == "text_materialized_pending_v7" and row["shard_path"]:
            key = f"gutenberg_resolved:pg{row['ebook_id']}"
            result[key] = TextReference(key, "gutenberg_resolved", config.repo_root / row["shard_path"], int(row["byte_start"]), int(row["byte_end"]))
    return result


def _add_probe_refs(result: dict[str, TextReference], path: Path, cache: Path, prefix: str) -> None:
    for row in _read_csv(path):
        source = cache / f"pg{row['ebook_id']}.txt"
        if source.is_file():
            key = f"{prefix}:pg{row['ebook_id']}"
            result[key] = TextReference(key, prefix, source, cleaning="gutenberg_boilerplate")


def _load_protected_sonnets(config: WikisourceReviewResolutionConfig) -> dict[str, str]:
    result = {}
    for row in _read_csv(config.protected_sonnet_manifest_path):
        if row["split_expanded_with_petrarch"] in {"validation", "test"}:
            path = config.repo_root / row["clean_text_path"]
            if path.is_file():
                result[row["poem_id"]] = path.read_text(encoding="utf-8")
    return result


def _load_sonnet_references(config: WikisourceReviewResolutionConfig) -> dict[str, str]:
    result = {f"v6_protected:{key}": value for key, value in _load_protected_sonnets(config).items()}
    for row in _read_csv(config.protected_sonnet_manifest_path):
        path = config.repo_root / row["clean_text_path"]
        if path.is_file() and f"v6_protected:{row['poem_id']}" not in result:
            result[f"v6:{row['poem_id']}"] = path.read_text(encoding="utf-8")
    for prefix, manifest in (("bibit_sonnet", config.bibit_sonnet_manifest_path), ("gutenberg_sonnet", config.gutenberg_resolved_sonnet_manifest_path)):
        cache: dict[Path, bytes] = {}
        for row in _read_csv(manifest):
            if not row.get("shard_path"):
                continue
            path = config.repo_root / row["shard_path"]
            payload = cache.setdefault(path, path.read_bytes())
            key = row.get("candidate_id") or row.get("poem_id")
            result[f"{prefix}:{key}"] = payload[int(row["byte_start"]):int(row["byte_end"])].decode("utf-8")
    return result


def _scan_rows_by_root(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        result[row["work_root_id"]].append(row)
    return result


def _parse_index_fields(wikitext: str) -> dict[str, str]:
    result = {}
    for line in wikitext.splitlines():
        match = _INDEX_FIELD.match(line)
        if match:
            result[match.group(1).strip().casefold()] = match.group(2).strip()
    return result


def _clean_metadata(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_HTML_TAG.sub("", value))).strip()


def _cross_overlap_fields() -> tuple[str, ...]:
    return (
        "bibit_overlap_metrics", "gutenberg_previous_pool_overlap_metrics",
        "gutenberg_pass_1b_overlap_metrics", "gutenberg_resolved_overlap_metrics",
        "existing_project_corpus_overlap_metrics",
    )


def _parse_cross_metrics(value: str) -> list[dict[str, Any]]:
    result = []
    for item in value.split(";"):
        if not item:
            continue
        parts = item.split("|")
        metrics = {key: float(number) for key, number in (part.split("=", 1) for part in parts[1:] if "=" in part)}
        result.append({"reference_id": parts[0], "candidate": metrics.get("candidate", 0.0), "reference": metrics.get("reference", 0.0)})
    return result


def _metric_ids(value: str) -> list[str]:
    return [item.split("|", 1)[0] for item in value.split(";") if item]


def _normalized_sha(text: str) -> str:
    digest = hashlib.sha256()
    for word in _normalized_words(text):
        digest.update(word.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_4c_root_cache(path: Path) -> str:
    value = path.read_text(encoding="utf-8")
    return value[:-1] if value.endswith("\n") else value


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _root_number(root_id: str) -> int:
    return int(root_id.split(":")[-1])


def _unique(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        if row[key] in result:
            raise ValueError(f"duplicate {key}: {row[key]}")
        result[row[key]] = row
    return result


def _validate_config(config: WikisourceReviewResolutionConfig) -> None:
    required = (
        config.dump_path, config.extraction_path, config.boundaries_path, config.review_path,
        config.inventory_path, config.candidate_resolution_path, config.scan_links_path,
        config.siteinfo_rights_path, config.bibit_record_manifest_path,
        config.broader_sources_manifest_path, config.gutenberg_previous_probe_path,
        config.gutenberg_pass_1b_probe_path, config.gutenberg_resolved_record_manifest_path,
        config.protected_sonnet_manifest_path, config.bibit_sonnet_manifest_path,
        config.gutenberg_resolved_sonnet_manifest_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing 4D inputs: {missing}")
    if sha1_file(config.dump_path) != config.expected_dump_sha1:
        raise ValueError("Wikisource dump SHA-1 mismatch")
    siteinfo = json.loads(config.siteinfo_rights_path.read_text(encoding="utf-8"))
    rights = siteinfo["query"]["rightsinfo"]
    if rights["text"] != SITE_LICENSE or rights["url"] != SITE_LICENSE_URL:
        raise ValueError("pinned Wikisource site license changed")


def _validate_outputs(
    config: WikisourceReviewResolutionConfig,
    roots: list[dict[str, Any]], segments: list[dict[str, Any]], sonnets: list[dict[str, Any]],
    rights: list[dict[str, Any]], reviews: list[dict[str, Any]],
) -> None:
    if len(roots) != config.expected_root_count or len(reviews) != config.expected_review_count:
        raise ValueError("4D output accounting mismatch")
    if any(row["review_status"] != "resolved" for row in reviews):
        raise ValueError("unresolved 4D review row")
    if any(row["rights_decision"] == "rights_missing" and row["final_decision"].startswith("eligible_") for row in roots):
        raise ValueError("eligible root lacks rights evidence")
    if any(int(row["line_count"]) != 14 for row in sonnets):
        raise ValueError("non-fourteen-line sonnet candidate")
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in segments:
        by_root[row["work_root_id"]].append(row)
    for root in roots:
        values = sorted(by_root[root["work_root_id"]], key=lambda row: int(row["character_start"]))
        if values and (int(values[0]["character_start"]) != 0 or int(values[-1]["character_end"]) != int(root["source_character_count"])):
            raise ValueError(f"segment partition does not cover {root['work_root_id']}")
        for left, right in zip(values, values[1:]):
            if int(left["character_end"]) != int(right["character_start"]):
                raise ValueError(f"segment gap/overlap: {root['work_root_id']}")


def _build_report(
    config: WikisourceReviewResolutionConfig, roots: list[dict[str, Any]],
    segments: list[dict[str, Any]], sonnets: list[dict[str, Any]],
    rights: list[dict[str, Any]], reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible = [row for row in roots if row["final_decision"].startswith("eligible_")]
    eligible_broader = [row for row in eligible if int(row["retained_broader_character_count"])]
    eligible_sonnets = [row for row in sonnets if row["candidate_decision"] == "eligible_standard_sonnet_inactive_pending_v7"]
    return {
        "checkpoint": "4D-review-resolution", "root_count": len(roots), "review_count": len(reviews),
        "segment_count": len(segments), "sonnet_candidate_count": len(sonnets),
        "eligible_sonnet_count": len(eligible_sonnets), "scan_rights_count": len(rights),
        "rights_pass_scan_count": sum(row["rights_decision"] == "rights_pass" for row in rights),
        "eligible_root_count": len(eligible),
        "eligible_broader_root_count": len(eligible_broader),
        "eligible_sonnet_only_root_count": sum(row["final_decision"] == "eligible_sonnets_only_inactive" for row in eligible),
        "retained_broader_character_count": sum(int(row["retained_broader_character_count"]) for row in eligible_broader),
        "root_decision_counts": dict(sorted(Counter(row["final_decision"] for row in roots).items())),
        "rights_decision_counts": dict(sorted(Counter(row["rights_decision"] for row in rights).items())),
        "eligible_role_counts": dict(sorted(Counter(row["final_broader_role"] for row in eligible_broader).items())),
        "sonnet_decision_counts": dict(sorted(Counter(row["candidate_decision"] for row in sonnets).items())),
        "input_sha256": {
            "extraction": _sha_file(config.extraction_path), "boundaries": _sha_file(config.boundaries_path),
            "review": _sha_file(config.review_path), "inventory": _sha_file(config.inventory_path),
            "candidate_resolution": _sha_file(config.candidate_resolution_path), "scan_links": _sha_file(config.scan_links_path),
        },
        "outputs": {
            "roots": config.root_decisions_path.relative_to(config.repo_root).as_posix(),
            "segments": config.segment_decisions_path.relative_to(config.repo_root).as_posix(),
            "sonnets": config.sonnet_decisions_path.relative_to(config.repo_root).as_posix(),
            "scan_rights": config.scan_rights_path.relative_to(config.repo_root).as_posix(),
            "reviews": config.review_resolution_path.relative_to(config.repo_root).as_posix(),
        },
        "output_sha256": {
            "roots": _sha_file(config.root_decisions_path),
            "segments": _sha_file(config.segment_decisions_path),
            "sonnets": _sha_file(config.sonnet_decisions_path),
            "scan_rights": _sha_file(config.scan_rights_path),
            "reviews": _sha_file(config.review_resolution_path),
        },
        "policy": {
            "canonical_precedence": "protected V6 > current corpora/BibIt/resolved Gutenberg > complete clean rights-cleared Wikisource > stable root ID",
            "rights_fail_closed": True, "conditioned_material_excluded": True,
            "text_activated": False, "v7_created": False, "mixture_assigned": False,
            "cache_deleted": False, "gpu_work_started": False,
        },
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _emit_progress(progress: Progress, phase: str, completed: int, total: int, started: float) -> None:
    elapsed = monotonic() - started
    rate = completed / elapsed if elapsed else 0.0
    eta = (total - completed) / rate if rate else 0.0
    progress(f"{phase} completed={completed:,}/{total:,} percent={completed/max(1,total):.1%} elapsed={elapsed:.1f}s eta={eta:.1f}s")
