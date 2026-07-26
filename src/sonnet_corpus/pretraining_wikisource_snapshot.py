"""Create committed Wikisource source snapshots from reviewed local audits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .italian_wikisource_probe import WORK_BOUNDARIES
from .pretraining_manifest import read_pretraining_manifest


@dataclass(frozen=True)
class PretrainingWikisourceSnapshotSelection:
    """One reviewed source and its approved immutable page scope."""

    source_id: str
    scope: str
    excluded_page_titles: tuple[str, ...] = ()


def create_pretraining_wikisource_snapshots(
    *,
    manifest_path: Path,
    audit_report_path: Path,
    snapshot_dir: Path,
    selections: tuple[PretrainingWikisourceSnapshotSelection, ...],
) -> list[dict[str, object]]:
    """Write immutable source snapshots from successful audit-report results."""

    if not selections:
        raise ValueError("at least one Wikisource snapshot selection is required")
    if len({selection.source_id for selection in selections}) != len(selections):
        raise ValueError("Wikisource snapshot selections contain duplicate source IDs")

    rows_by_id = {row.source_id: row for row in read_pretraining_manifest(manifest_path)}
    audit_report = json.loads(audit_report_path.read_text(encoding="utf-8"))
    results_by_id = {
        str(result["source_id"]): result for result in audit_report.get("results", [])
    }
    snapshots: list[dict[str, object]] = []
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for selection in selections:
        if selection.scope not in {"explicit_subpages", "recursive_leaf_pages"}:
            raise ValueError(f"unsupported pretraining snapshot scope: {selection.scope}")
        row = rows_by_id.get(selection.source_id)
        if row is None:
            raise ValueError(f"Wikisource snapshot source is absent from manifest: {selection.source_id}")
        if row.source_archive != "Italian Wikisource":
            raise ValueError(f"snapshot source is not Italian Wikisource: {selection.source_id}")
        result = results_by_id.get(selection.source_id)
        if result is None or result.get("status") != "ok":
            raise ValueError(
                "Wikisource snapshot requires a successful reviewed audit: "
                f"{selection.source_id}"
            )
        boundaries = WORK_BOUNDARIES.get(selection.source_id)
        if boundaries is None:
            raise ValueError(f"Wikisource snapshot source has no recorded boundaries: {selection.source_id}")

        excluded_titles = set(selection.excluded_page_titles)
        page_revisions = [
            page
            for page in result["page_revisions"]
            if page["title"] not in excluded_titles
        ]
        if not page_revisions:
            raise ValueError(f"Wikisource snapshot selected no primary pages: {selection.source_id}")
        unexpected_exclusions = excluded_titles - {
            page["title"] for page in result["page_revisions"]
        }
        if unexpected_exclusions:
            raise ValueError(
                "Wikisource snapshot exclusions were absent from the audit: "
                + ", ".join(sorted(unexpected_exclusions))
            )

        root_title = boundaries.root_page_title or row.title
        payload: dict[str, object] = {
            "source_id": row.source_id,
            "landing_page_url": row.landing_page_url,
            "title": row.title,
            "scope": selection.scope,
            "root_revision": {
                "title": root_title,
                "revision_id": result["root_revision_id"],
                "revision_timestamp": result["root_revision_timestamp"],
            },
            "page_revisions": page_revisions,
        }
        snapshot_path = snapshot_dir / f"{row.source_id}.json"
        snapshot_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        snapshots.append(payload)

    return snapshots
