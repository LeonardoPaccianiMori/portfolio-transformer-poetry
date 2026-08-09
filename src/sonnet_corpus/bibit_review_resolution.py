"""Deterministic resolution of Biblioteca Italiana record and sonnet queues."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .biblioteca_italiana import ParsedBibItTEI, parse_bibit_tei
from .bibit_composition_audit import ROLE_HISTORICAL_POETRY
from .bibit_role_audit import normalize_loose_text, write_csv


RECORD_DECISION_FIELDS = (
    "object_id",
    "title",
    "authors",
    "assigned_role",
    "audited_route",
    "source_audit_status",
    "source_audit_flags",
    "decision",
    "final_role",
    "cleaning_policy",
    "included_characters",
    "duplicate_of_object_id",
    "reason",
    "landing_page_url",
    "tei_sha256",
)

SONNET_DECISION_FIELDS = (
    "candidate_id",
    "object_id",
    "title",
    "candidate_author",
    "author_resolution",
    "periods",
    "source_kind",
    "tei_type",
    "heading_path",
    "line_count",
    "stanza_pattern",
    "rhyme_signature",
    "rhyme_evidence",
    "source_audit_status",
    "decision",
    "final_role",
    "cleaning_policy",
    "reason",
    "character_count",
    "text_sha256",
    "normalized_sha256",
    "first_line",
    "last_line",
    "landing_page_url",
)

_BRACKETED_TEXT = re.compile(r"\[([^\]\n]{1,160})\]")
_EDITORIAL_LABEL = re.compile(
    r"(?:\d+(?:\s*[-–—,;]\s*\d+)*|[A-Z]{1,3}|[IVXLCDM]{1,6})"
)
_ELLIPSIS = re.compile(r"[.·…\s–—-]+")
_SONNET_CUE = re.compile(r"\bsonett\w*\b", re.IGNORECASE)
_COLLECTION_CUE = re.compile(
    r"\b(?:rime|canzonier\w*|amoros\w*|amori|versi d['’]amore|"
    r"petrarca spirituale|vita nuova|eroici furori)\b",
    re.IGNORECASE,
)
_STANDARD_STANZA_PATTERNS = {"4+4+3+3", "4+4+6", "8+6", "8+3+3"}
_DIRECT_EXCLUSION_STATUSES = {
    "excluded_empty",
    "excluded_exact_active_duplicate",
    "excluded_exact_bibit_duplicate",
    "excluded_held_out_identity_conflict",
    "excluded_implausible_sonnet_length",
}
_ARCADIA_OBJECT_IDS = {
    "bibit000807",
    "bibit001063",
    "bibit000532",
    "bibit000524",
    "bibit000384",
    "bibit001414",
}


@dataclass(frozen=True)
class BibItReviewResolutionConfig:
    repo_root: Path
    record_audit_csv_path: Path
    sonnet_audit_csv_path: Path
    tei_cache_dir: Path
    record_decision_csv_path: Path
    sonnet_decision_csv_path: Path
    json_report_path: Path
    markdown_report_path: Path
    progress_interval: int = 25


Progress = Callable[[str], None]


def resolve_bibit_review_queues(
    config: BibItReviewResolutionConfig,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Resolve every BibIt audit row into an activation or exclusion decision."""

    if config.progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    record_rows = _read_csv(config.record_audit_csv_path)
    sonnet_rows = _read_csv(config.sonnet_audit_csv_path)
    _report(
        progress,
        f"loaded records={len(record_rows):,} sonnet_candidates={len(sonnet_rows):,}",
    )

    record_decisions = [_resolve_record(row) for row in record_rows]
    record_by_id = {row["object_id"]: row for row in record_decisions}
    if len(record_by_id) != len(record_decisions):
        raise ValueError("record audit contains duplicate object IDs")

    text_loader = _CandidateTextLoader(config.tei_cache_dir)
    sonnet_decisions: list[dict[str, Any]] = []
    for index, row in enumerate(sonnet_rows, start=1):
        sonnet_decisions.append(
            _resolve_sonnet_candidate(
                row,
                record_decision=record_by_id[row["object_id"]],
                text_loader=text_loader,
            )
        )
        if (
            index == 1
            or index % config.progress_interval == 0
            or index == len(sonnet_rows)
        ):
            _report(
                progress,
                f"resolved candidate {index:,}/{len(sonnet_rows):,} "
                f"({index / len(sonnet_rows):.1%}) parsed_tei={text_loader.parse_count:,}",
            )

    unresolved_records = [
        row for row in record_decisions if row["decision"].startswith("review_")
    ]
    unresolved_sonnets = [
        row for row in sonnet_decisions if row["decision"].startswith("review_")
    ]
    if unresolved_records or unresolved_sonnets:
        raise ValueError(
            "BibIt resolution left unresolved rows: "
            f"records={len(unresolved_records)} sonnets={len(unresolved_sonnets)}"
        )

    report = _build_report(config, record_decisions, sonnet_decisions)
    write_csv(config.record_decision_csv_path, RECORD_DECISION_FIELDS, record_decisions)
    write_csv(config.sonnet_decision_csv_path, SONNET_DECISION_FIELDS, sonnet_decisions)
    _write_json(config.json_report_path, report)
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(
        render_bibit_resolution_markdown(report),
        encoding="utf-8",
    )
    return report


def clean_bibit_editorial_brackets(text: str) -> str:
    """Remove editorial square delimiters while retaining supplied source text."""

    def replace(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        if _EDITORIAL_LABEL.fullmatch(content):
            return ""
        if _ELLIPSIS.fullmatch(content):
            return "..."
        return content

    return _BRACKETED_TEXT.sub(replace, text)


def coarse_sonnet_rhyme_evidence(text: str) -> tuple[str, bool]:
    """Return a visual three-letter rhyme signature and conservative evidence."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 14:
        return "", False
    keys: list[str] = []
    for line in lines:
        words = normalize_loose_text(line).split()
        if not words:
            return "", False
        word = words[-1]
        keys.append(word[-3:] if len(word) >= 3 else word)

    labels: dict[str, str] = {}
    signature: list[str] = []
    for key in keys:
        if key not in labels:
            labels[key] = chr(ord("A") + len(labels))
        signature.append(labels[key])

    octave = Counter(keys[:8])
    sestet = Counter(keys[8:])
    octave_coverage = sum(count for count in octave.values() if count >= 2)
    sestet_coverage = sum(count for count in sestet.values() if count >= 2)
    evidence = (
        octave_coverage >= 6
        and sum(count >= 2 for count in octave.values()) >= 2
        and sestet_coverage >= 4
        and sum(count >= 2 for count in sestet.values()) >= 1
    )
    return "".join(signature), evidence


def _resolve_record(row: dict[str, str]) -> dict[str, Any]:
    flags = {value for value in row["audit_flags"].split(";") if value}
    characters = int(row["routed_training_characters"] or 0)
    decision = "activate_core"
    final_role = row["route"]
    cleaning_policy = "preserve_rendered_tei_text"
    reason = "passed canonical TEI role audit"

    if row["error"] or "fetch_or_parse_error" in flags:
        decision = "exclude_empty_or_unavailable"
        final_role = "excluded"
        reason = row["error"] or "TEI unavailable"
    elif "duplicate_exact_bibit_document" in flags:
        decision = "exclude_duplicate_document"
        final_role = "excluded"
        reason = f"exact canonical-text duplicate of {row['duplicate_of_object_id']}"
    elif "review_non_italian_language" in flags:
        decision = "exclude_non_italian"
        final_role = "excluded"
        reason = "TEI language metadata identifies a non-Italian work"
    elif "review_language_variety" in flags:
        decision = "exclude_core_language_variety"
        final_role = "excluded_language_variety_evidence"
        reason = "dialect, vernacular, or mixed-language variety is not core standard Italian"
    elif "review_missing_source_edition" in flags:
        decision = "exclude_missing_source_provenance"
        final_role = "excluded"
        reason = "TEI lacks both a printed source title and source identifier"
    elif "empty_or_too_short_after_sonnet_quarantine" in flags:
        decision = "exclude_empty_or_too_short"
        final_role = "excluded"
        reason = "fewer than 200 routed characters remain after poem quarantine"
    elif "review_no_sonnet_candidates" in flags:
        if characters >= 200:
            decision = "activate_as_non_sonnet_poetry"
            final_role = ROLE_HISTORICAL_POETRY
            reason = "sonnet composition guess was false; TEI contains reusable non-sonnet verse"
        else:
            decision = "exclude_empty_or_too_short"
            final_role = "excluded"
            reason = "no sonnet candidate and too little residual verse"
    elif "review_editorial_brackets" in flags:
        decision = "activate_core_with_bracket_cleanup"
        cleaning_policy = "strip_editorial_square_delimiters_and_labels"
        reason = "retain supplied historical text while removing editorial bracket notation"
    elif "review_editorial_references" in flags:
        decision = "activate_core"
        reason = "cross-reference phrases are authorial text and remain unchanged"
    elif row["audit_status"] != "activation_candidate":
        raise ValueError(
            f"unhandled record review flags for {row['object_id']}: {row['audit_flags']}"
        )

    if (
        decision == "activate_as_non_sonnet_poetry"
        and "review_editorial_brackets" in flags
    ):
        decision = "activate_as_non_sonnet_poetry_with_bracket_cleanup"
        cleaning_policy = "strip_editorial_square_delimiters_and_labels"

    included_characters = characters if decision.startswith("activate_") else 0
    return {
        "object_id": row["object_id"],
        "title": row["title"],
        "authors": row["authors"],
        "assigned_role": row["assigned_role"],
        "audited_route": row["route"],
        "source_audit_status": row["audit_status"],
        "source_audit_flags": row["audit_flags"],
        "decision": decision,
        "final_role": final_role,
        "cleaning_policy": cleaning_policy,
        "included_characters": included_characters,
        "duplicate_of_object_id": row["duplicate_of_object_id"],
        "reason": reason,
        "landing_page_url": row["landing_page_url"],
        "tei_sha256": row["tei_sha256"],
    }


def _resolve_sonnet_candidate(
    row: dict[str, str],
    *,
    record_decision: dict[str, Any],
    text_loader: "_CandidateTextLoader",
) -> dict[str, Any]:
    source_status = row["status"]
    source_kind = row["source_kind"]
    line_count = int(row["line_count"] or 0)
    candidate_author, author_resolution = _resolve_candidate_author(row)
    rhyme_signature = ""
    rhyme_evidence = False
    cleaning_policy = record_decision["cleaning_policy"]

    if row["held_out_duplicate_poem_ids"]:
        decision = "exclude_held_out_identity"
        final_role = "excluded"
        reason = "matches a fixed V6 validation/test identity"
    elif source_status in {
        "excluded_exact_active_duplicate",
        "excluded_exact_bibit_duplicate",
    }:
        decision = "exclude_exact_duplicate"
        final_role = "excluded"
        reason = "exact duplicate of active V6 or earlier canonical BibIt candidate"
    elif source_status == "review_near_active_duplicate":
        decision = "exclude_near_active_duplicate"
        final_role = "excluded"
        reason = "near-duplicate of an active V6 poem; preserve only the canonical edition"
    elif source_status in _DIRECT_EXCLUSION_STATUSES:
        decision = source_status
        final_role = "excluded"
        reason = "failed the source audit's form or identity gate"
    elif not str(record_decision["decision"]).startswith("activate_"):
        decision = "exclude_with_source_record"
        final_role = "excluded"
        reason = f"source record decision is {record_decision['decision']}"
    elif source_kind == "explicit_tei_sonnet" and line_count == 14:
        decision = "activate_standard_explicit_sonnet"
        final_role = "sonnet_core_standard_14_line"
        reason = "explicit TEI sonnet label and exact 14-line form"
    elif source_kind == "explicit_tei_sonnet" and 15 <= line_count <= 32:
        decision = "activate_explicit_sonnet_variant"
        final_role = "sonnet_variant_conditioned_auxiliary"
        reason = "explicit TEI sonnet label with a non-standard or tailed line count"
    elif source_kind == "structural_sonnet_variant":
        decision = "activate_heading_backed_sonnet_variant"
        final_role = "sonnet_variant_conditioned_auxiliary"
        reason = "15-32-line unit is nested under an explicit sonnet section heading"
    elif source_kind == "structural_14_line":
        text = text_loader.text_for(row)
        rhyme_signature, rhyme_evidence = coarse_sonnet_rhyme_evidence(text)
        evidence_text = " ".join(
            (row["title"], row["heading_path"], row["tei_type"])
        )
        if _SONNET_CUE.search(evidence_text):
            evidence_reason = "source title or heading explicitly identifies a sonnet"
        elif row["stanza_pattern"] in _STANDARD_STANZA_PATTERNS:
            evidence_reason = f"TEI stanza pattern {row['stanza_pattern']}"
        elif _COLLECTION_CUE.search(evidence_text) and rhyme_evidence:
            evidence_reason = "rime/canzoniere context plus repeated octave/sestet rhyme evidence"
        else:
            evidence_reason = ""
        if evidence_reason:
            decision = "activate_inferred_standard_sonnet"
            final_role = "sonnet_core_inferred_14_line"
            reason = f"exact 14-line unit; {evidence_reason}"
        else:
            decision = "exclude_unverified_structural_14_line_unit"
            final_role = "excluded"
            reason = "14 lines alone are insufficient evidence of sonnet form"
    else:
        decision = "exclude_unhandled_form"
        final_role = "excluded"
        reason = f"unsupported source kind or line count: {source_kind}/{line_count}"

    if "editorial" in source_status and final_role != "excluded":
        cleaning_policy = "strip_editorial_square_delimiters_and_labels"

    return {
        "candidate_id": row["candidate_id"],
        "object_id": row["object_id"],
        "title": row["title"],
        "candidate_author": candidate_author,
        "author_resolution": author_resolution,
        "periods": row["periods"],
        "source_kind": source_kind,
        "tei_type": row["tei_type"],
        "heading_path": row["heading_path"],
        "line_count": line_count,
        "stanza_pattern": row["stanza_pattern"],
        "rhyme_signature": rhyme_signature,
        "rhyme_evidence": rhyme_evidence,
        "source_audit_status": source_status,
        "decision": decision,
        "final_role": final_role,
        "cleaning_policy": cleaning_policy,
        "reason": reason,
        "character_count": int(row["character_count"] or 0),
        "text_sha256": row["text_sha256"],
        "normalized_sha256": row["normalized_sha256"],
        "first_line": row["first_line"],
        "last_line": row["last_line"],
        "landing_page_url": row["landing_page_url"],
    }


def _resolve_candidate_author(row: dict[str, str]) -> tuple[str, str]:
    author = row["authors"].strip()
    if normalize_loose_text(author) not in {"", "non definito"}:
        return author, "catalog_author"

    parts = [part.strip() for part in row["heading_path"].split(" > ") if part.strip()]
    object_id = row["object_id"]
    if object_id in _ARCADIA_OBJECT_IDS and parts:
        return parts[0], "source_heading_arcadian_name"
    if object_id == "bibit000709" and len(parts) >= 2:
        return parts[1].title(), "source_heading_named_author"
    if object_id == "bibit001319" and parts:
        return parts[0], "source_heading_named_author"
    if object_id == "bibit000818" and parts:
        return "Anonimo", "source_heading_attribution_retained_separately"
    if object_id == "bibit000909" and parts:
        return parts[0], "source_heading_named_author"
    if object_id == "bibit001194":
        return "Anonimo", "anonymous_source_collection"

    return "Anonimo", "anonymous_or_unresolved_in_source_edition"


class _CandidateTextLoader:
    def __init__(self, tei_cache_dir: Path):
        self.tei_cache_dir = tei_cache_dir
        self._units_by_object_id: dict[str, dict[str, str]] = {}
        self.parse_count = 0

    def text_for(self, row: dict[str, str]) -> str:
        object_id = row["object_id"]
        if object_id not in self._units_by_object_id:
            path = self.tei_cache_dir / f"{object_id}.xml"
            if not path.is_file():
                raise FileNotFoundError(f"missing cached BibIt TEI: {path}")
            parsed = parse_bibit_tei(path.read_bytes(), object_id=object_id)
            self._units_by_object_id[object_id] = _candidate_text_map(parsed)
            self.parse_count += 1
        unit_id = row["candidate_id"].split(":", maxsplit=1)[1]
        try:
            return self._units_by_object_id[object_id][unit_id]
        except KeyError as error:
            raise ValueError(f"candidate unit not found in pinned TEI: {row['candidate_id']}") from error


def _candidate_text_map(parsed: ParsedBibItTEI) -> dict[str, str]:
    units = (
        *parsed.sonnets,
        *parsed.structural_sonnet_candidates,
        *parsed.structural_sonnet_variants,
    )
    return {unit.unit_id: unit.text for unit in units}


def _build_report(
    config: BibItReviewResolutionConfig,
    records: list[dict[str, Any]],
    sonnets: list[dict[str, Any]],
) -> dict[str, Any]:
    record_counts = Counter(row["decision"] for row in records)
    sonnet_counts = Counter(row["decision"] for row in sonnets)
    role_characters = Counter()
    for row in records:
        if row["decision"].startswith("activate_"):
            role_characters[row["final_role"]] += int(row["included_characters"])
    role_poems = Counter()
    role_poem_characters = Counter()
    for row in sonnets:
        if str(row["decision"]).startswith("activate_"):
            role_poems[row["final_role"]] += 1
            role_poem_characters[row["final_role"]] += int(row["character_count"])
    return {
        "resolution_version": "bibit_review_resolution_v1",
        "inputs": {
            "record_audit_csv_path": _portable(config.record_audit_csv_path, config.repo_root),
            "sonnet_audit_csv_path": _portable(config.sonnet_audit_csv_path, config.repo_root),
            "local_tei_cache_path": _portable(config.tei_cache_dir, config.repo_root),
        },
        "outputs": {
            "record_decision_csv_path": _portable(config.record_decision_csv_path, config.repo_root),
            "sonnet_decision_csv_path": _portable(config.sonnet_decision_csv_path, config.repo_root),
            "json_report_path": _portable(config.json_report_path, config.repo_root),
            "markdown_report_path": _portable(config.markdown_report_path, config.repo_root),
        },
        "record_count": len(records),
        "record_decision_counts": dict(sorted(record_counts.items())),
        "activated_record_characters_by_role": dict(sorted(role_characters.items())),
        "sonnet_candidate_count": len(sonnets),
        "sonnet_decision_counts": dict(sorted(sonnet_counts.items())),
        "activated_sonnet_counts_by_role": dict(sorted(role_poems.items())),
        "activated_sonnet_characters_by_role": dict(
            sorted(role_poem_characters.items())
        ),
        "unresolved_record_count": 0,
        "unresolved_sonnet_count": 0,
        "policy": {
            "near_active_duplicates_excluded": True,
            "held_out_identities_excluded": True,
            "language_variety_excluded_from_core": True,
            "explicit_nonstandard_sonnets_conditioned_separately": True,
            "structural_14_line_units_require_additional_form_evidence": True,
            "editorial_square_brackets_use_documented_cleanup": True,
        },
    }


def render_bibit_resolution_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Biblioteca Italiana Review Resolution",
        "",
        "## Decision",
        "",
        (
            f"Resolved {report['record_count']:,} canonical records and "
            f"{report['sonnet_candidate_count']:,} poem candidates with no open rows."
        ),
        "",
        "Dialect, vernacular, and Franco-Italian records remain documented evidence",
        "but are excluded from the unconditioned standard-Italian core. Explicit",
        "non-14-line sonnets are retained only in a conditioned variant role.",
        "",
        "## Record Decisions",
        "",
        "| Decision | Records |",
        "| --- | ---: |",
    ]
    for decision, count in report["record_decision_counts"].items():
        lines.append(f"| `{decision}` | {count:,} |")
    lines.extend(
        [
            "",
            "## Activated Text",
            "",
            "| Role | Characters |",
            "| --- | ---: |",
        ]
    )
    for role, characters in report["activated_record_characters_by_role"].items():
        lines.append(f"| `{role}` | {characters:,} |")
    lines.extend(
        [
            "",
            "## Poem Decisions",
            "",
            "| Decision | Candidates |",
            "| --- | ---: |",
        ]
    )
    for decision, count in report["sonnet_decision_counts"].items():
        lines.append(f"| `{decision}` | {count:,} |")
    lines.extend(
        [
            "",
            "## Activated Poem Roles",
            "",
            "| Role | Poems | Characters |",
            "| --- | ---: | ---: |",
        ]
    )
    for role, count in report["activated_sonnet_counts_by_role"].items():
        characters = report["activated_sonnet_characters_by_role"][role]
        lines.append(f"| `{role}` | {count:,} | {characters:,} |")
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- Record decisions: `{report['outputs']['record_decision_csv_path']}`",
            f"- Poem decisions: `{report['outputs']['sonnet_decision_csv_path']}`",
            f"- Machine-readable report: `{report['outputs']['json_report_path']}`",
            "- Raw TEI remains machine-local; decisions retain source hashes and URLs.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _report(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)
