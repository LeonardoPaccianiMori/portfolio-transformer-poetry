"""Freeze the unresolved Project Gutenberg metadata-review queue."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gutenberg_catalog_inventory import INVENTORY_FIELDS


REVIEW_STATUS_EVIDENCE = {
    "review_work_publication_date": (
        "work first-publication year from a title page, catalog record, or "
        "authoritative bibliography"
    ),
    "review_missing_period_evidence": (
        "author/work dates from a title page, catalog record, or authoritative "
        "bibliography"
    ),
    "review_translation_edition_date": (
        "Italian translation edition date and source-language/translator evidence"
    ),
    "review_language_variety_before_download": (
        "primary-text evidence for standard Italian, dialect, or mixed-language routing"
    ),
    "review_rights": "record-level public-domain or compatible reuse evidence",
}

QUEUE_FIELDS = (*INVENTORY_FIELDS, "resolution_evidence_required")


@dataclass(frozen=True)
class GutenbergMetadataReviewQueueConfig:
    repo_root: Path
    inventory_csv_path: Path
    queue_csv_path: Path
    json_report_path: Path
    markdown_report_path: Path


def freeze_gutenberg_metadata_review_queue(
    config: GutenbergMetadataReviewQueueConfig,
) -> dict[str, Any]:
    """Write the exact unresolved queue from a frozen Gutenberg inventory."""

    rows = _read_inventory(config.inventory_csv_path)
    ebook_ids = [row["ebook_id"] for row in rows]
    if len(ebook_ids) != len(set(ebook_ids)):
        raise ValueError("Gutenberg inventory contains duplicate eBook IDs")

    queue_rows: list[dict[str, str]] = []
    for row in rows:
        status = row["inventory_status"]
        if status not in REVIEW_STATUS_EVIDENCE:
            continue
        queued = dict(row)
        queued["resolution_evidence_required"] = REVIEW_STATUS_EVIDENCE[status]
        queue_rows.append(queued)
    queue_rows.sort(key=lambda row: int(row["ebook_id"]))

    status_counts = Counter(row["inventory_status"] for row in rows)
    review_counts = Counter(row["inventory_status"] for row in queue_rows)
    accounting = {
        "eligible_fulltext_probe": (
            status_counts["audit_then_deduplicate"]
            + status_counts["deduplicate_before_full_text_audit"]
        ),
        "metadata_review_queue": len(queue_rows),
        "already_registered": status_counts["already_registered_project_gutenberg_source"],
        "metadata_exclusions": (
            status_counts["exclude_core_language_variety_metadata"]
            + status_counts["exclude_metadata_scope"]
        ),
    }
    if sum(accounting.values()) != len(rows):
        raise ValueError(
            "Gutenberg inventory accounting is incomplete: "
            f"accounted={sum(accounting.values())} inventory={len(rows)}"
        )

    _write_csv(config.queue_csv_path, queue_rows)
    report = {
        "queue_version": "project_gutenberg_metadata_review_queue_v1",
        "inventory_record_count": len(rows),
        "review_record_count": len(queue_rows),
        "review_status_counts": dict(sorted(review_counts.items())),
        "inventory_status_counts": dict(sorted(status_counts.items())),
        "inventory_accounting": accounting,
        "preliminary_role_count_is_not_queue_count": True,
        "preliminary_date_and_role_review_count": sum(
            row["preliminary_role"] == "date_and_role_review" for row in rows
        ),
        "required_evidence_by_status": REVIEW_STATUS_EVIDENCE,
        "outputs": {
            "inventory_csv_path": _portable(config.inventory_csv_path, config.repo_root),
            "inventory_csv_sha256": _sha256_file(config.inventory_csv_path),
            "queue_csv_path": _portable(config.queue_csv_path, config.repo_root),
            "queue_csv_sha256": _sha256_file(config.queue_csv_path),
            "json_report_path": _portable(config.json_report_path, config.repo_root),
            "markdown_report_path": _portable(
                config.markdown_report_path, config.repo_root
            ),
        },
        "policy": {
            "metadata_only": True,
            "full_text_downloaded": False,
            "activation_authorized": False,
            "queue_resolution_required_before_new_fulltext_probe": True,
        },
    }
    _write_json(config.json_report_path, report)
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(
        render_gutenberg_metadata_review_queue_markdown(report),
        encoding="utf-8",
    )
    return report


def render_gutenberg_metadata_review_queue_markdown(report: dict[str, Any]) -> str:
    """Render the public queue-accounting report."""

    lines = [
        "# Project Gutenberg Metadata Review Queue",
        "",
        "## Result",
        "",
        (
            f"Frozen {report['review_record_count']:,} unresolved metadata-review "
            f"records from the {report['inventory_record_count']:,}-record Italian "
            "catalog inventory. This artifact activates no text."
        ),
        "",
        "## Review Statuses",
        "",
        "| Status | Records | Evidence required |",
        "| --- | ---: | --- |",
    ]
    for status, count in report["review_status_counts"].items():
        evidence = report["required_evidence_by_status"][status]
        lines.append(f"| `{status}` | {count:,} | {evidence} |")
    lines.extend(
        [
            "",
            "## Complete Inventory Accounting",
            "",
            "| Route | Records |",
            "| --- | ---: |",
        ]
    )
    for route, count in report["inventory_accounting"].items():
        lines.append(f"| `{route}` | {count:,} |")
    lines.extend(
        [
            f"| **Total** | **{report['inventory_record_count']:,}** |",
            "",
            "## Count Clarification",
            "",
            (
                f"The earlier {report['preliminary_date_and_role_review_count']:,} "
                "figure counts records whose preliminary *role* is "
                "`date_and_role_review`. It is not the unresolved queue size. The "
                f"{report['review_record_count']:,}-record queue also includes poetry, "
                "translation, sonnet, and language-variety candidates whose status "
                "still requires evidence."
            ),
            "",
            "## Boundaries",
            "",
            "- Resolve each queued record before deciding whether to download it.",
            "- A resolved record still requires full-text quality and deduplication gates.",
            "- Dialect and mixed-language records remain outside the unconditioned core.",
            "- No V7 split or training-mixture weight is assigned here.",
            "",
            "## Artifacts",
            "",
            f"- Frozen input inventory: `{report['outputs']['inventory_csv_path']}`",
            f"- Review queue: `{report['outputs']['queue_csv_path']}`",
            f"- Machine-readable report: `{report['outputs']['json_report_path']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _read_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != INVENTORY_FIELDS:
            raise ValueError("Gutenberg inventory schema does not match INVENTORY_FIELDS")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Gutenberg inventory is empty: {path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)
