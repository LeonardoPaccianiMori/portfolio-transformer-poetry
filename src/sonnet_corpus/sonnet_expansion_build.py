"""Build a versioned sonnet corpus from approved revision-pinned sources."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from .cleaning import clean_poem_text, count_poem_lines
from .italian_wikisource import (
    FetchedItalianWikisourcePageCollection,
    WikisourcePageRevision,
    WikisourceWorkSnapshot,
    fetch_pinned_italian_wikisource_page_collection,
)
from .manifest import ManifestRow, write_manifest
from .splits import assign_split
from .wikisource import extract_poem_text, url_from_title


FetchPinnedCollection = Callable[..., FetchedItalianWikisourcePageCollection]


@dataclass(frozen=True)
class ActivatedSourceMetadata:
    source_id: str
    author: str
    source_archive: str
    source_collection: str
    source_edition: str
    license_notes: str
    period: str
    poem_id_prefix: str = ""


ACTIVATED_SOURCE_METADATA = {
    "ws_alfieri_rime_1912": ActivatedSourceMetadata(
        source_id="ws_alfieri_rime_1912",
        author="Vittorio Alfieri",
        source_archive="Italian Wikisource",
        source_collection="Rime varie (Alfieri, 1912)",
        source_edition="Rime scelte, Sansoni, 1912",
        license_notes="CC BY-SA / GFDL metadata on Italian Wikisource; retain page URL, author, source edition, and license metadata.",
        period="XVIII secolo",
        poem_id_prefix="alfieri",
    ),
    "ws_foscolo_sonetti": ActivatedSourceMetadata(
        source_id="ws_foscolo_sonetti",
        author="Ugo Foscolo",
        source_archive="Italian Wikisource",
        source_collection="Sonetti (Foscolo)",
        source_edition="Opere scelte di Ugo Foscolo II, Poligrafica Fiesolana, 1835; curated by Giuseppe Caleffi.",
        license_notes="CC BY-SA 3.0 / GFDL metadata on Italian Wikisource; retain page URL, author Ugo Foscolo, editor Giuseppe Caleffi, source scan, license links, and share-alike notice.",
        period="XIX secolo",
        poem_id_prefix="foscolo",
    ),
}


def create_sonnet_source_snapshot(
    *,
    audit_report_path: Path,
    snapshot_path: Path,
    source_id: str,
) -> dict[str, object]:
    """Convert reviewed eligible audit records into a committed page snapshot."""

    report = json.loads(audit_report_path.read_text(encoding="utf-8"))
    if report.get("source", {}).get("source_id") != source_id:
        raise ValueError("audit report source ID does not match requested snapshot")
    if report.get("activation_status") != "audit_then_include":
        raise ValueError("audit report is not an activation candidate")

    eligible = [
        candidate
        for candidate in report.get("candidates", [])
        if candidate.get("status") == "eligible_14_lines"
        and candidate.get("line_count_clean") == 14
        and not candidate.get("exact_active_duplicate_poem_ids")
    ]
    if not eligible:
        raise ValueError("audit report has no eligible non-duplicate sonnets")

    root = report["root_revision"]
    metadata = ACTIVATED_SOURCE_METADATA.get(source_id)
    if metadata is None:
        raise ValueError(f"no activation metadata configured for source: {source_id}")
    source_records = [candidate.get("source_record") for candidate in eligible]
    uses_edition_records = any(source_records)
    if uses_edition_records and not all(isinstance(record, dict) for record in source_records):
        raise ValueError("edition-page audit has incomplete bibliographic-record provenance")
    if uses_edition_records and not report.get("edition_page_title_suffix"):
        raise ValueError("edition-page audit has no approved edition title suffix")

    payload = {
        "source_id": source_id,
        "landing_page_url": report["source"]["landing_page_url"],
        "title": root["title"],
        "scope": "explicit_edition_pages" if uses_edition_records else "explicit_subpages",
        "root_revision": root,
        "page_revisions": [
            {
                "title": candidate["page_title"],
                "revision_id": candidate["revision_id"],
                "revision_timestamp": candidate["revision_timestamp"],
            }
            for candidate in eligible
        ],
        "source_record_revisions": (
            [
                {
                    "title": record["page_title"],
                    "revision_id": record["revision_id"],
                    "revision_timestamp": record["revision_timestamp"],
                }
                for record in source_records
                if isinstance(record, dict)
            ]
            if uses_edition_records
            else []
        ),
        "edition_page_title_suffix": (
            report["edition_page_title_suffix"] if uses_edition_records else ""
        ),
        "source_metadata": asdict(metadata),
        "audit_report": {
            "page_count": report["page_count"],
            "eligible_sonnet_count": len(eligible),
            "started_at_utc": report["started_at_utc"],
            "finished_at_utc": report["finished_at_utc"],
        },
    }
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def build_sonnets_expanded(
    *,
    repo_root: Path,
    base_manifest_path: Path,
    snapshot_path: Path,
    output_dataset_id: str = "sonnets_expanded_v2",
    request_delay: float = 6.0,
    fetch_collection: FetchPinnedCollection = fetch_pinned_italian_wikisource_page_collection,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Create a new versioned corpus from an active base plus one pinned source."""

    snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot = read_snapshot(snapshot_payload)
    metadata = ActivatedSourceMetadata(**snapshot_payload["source_metadata"])
    if metadata.source_id != snapshot.source_id:
        raise ValueError("snapshot metadata source ID does not match page snapshot")

    output_root = repo_root / "data" / "processed" / output_dataset_id
    output_poems_dir = output_root / "poems"
    output_manifest_path = repo_root / "data" / "metadata" / f"{output_dataset_id}_manifest.csv"
    output_report_path = repo_root / "data" / "metadata" / f"{output_dataset_id}_build_report.json"
    output_attribution_path = repo_root / "data" / "metadata" / f"{output_dataset_id}_attribution.md"
    if output_root.exists() or output_manifest_path.exists():
        raise FileExistsError(f"versioned corpus already exists: {output_dataset_id}")

    base_rows = [row for row in read_manifest_rows(base_manifest_path) if row.include_in_training]
    if not base_rows:
        raise ValueError("base manifest has no active processed poems")
    duplicate_index = index_texts(base_rows, repo_root)
    started_at = utc_now()

    try:
        _write_progress(progress, f"copying {len(base_rows)} base processed poems")
        output_poems_dir.mkdir(parents=True)
        copied_rows = copy_base_rows(base_rows, repo_root, output_poems_dir)

        _write_progress(progress, f"rendering {len(snapshot.page_revisions)} approved pinned source pages")
        collection = fetch_collection(
            snapshot,
            request_delay=request_delay,
            progress=progress,
        )
        if [page.revision for page in collection.pages] != snapshot.page_revisions:
            raise ValueError("pinned fetch returned a page sequence different from the snapshot")

        source_rows = build_source_rows(
            collection=collection,
            metadata=metadata,
            output_poems_dir=output_poems_dir,
            repo_root=repo_root,
            duplicate_index=duplicate_index,
        )
        combined_rows = copied_rows + source_rows
        write_manifest(combined_rows, output_manifest_path)
        write_attribution(output_attribution_path, output_dataset_id, metadata, snapshot, source_rows)
        report = {
            "dataset_id": output_dataset_id,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "base_manifest_path": portable_path(base_manifest_path, repo_root),
            "snapshot_path": portable_path(snapshot_path, repo_root),
            "base_poem_count": len(copied_rows),
            "added_poem_count": len(source_rows),
            "total_poem_count": len(combined_rows),
            "split_counts": split_counts(combined_rows),
        }
        output_report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        delete_temporary_workspaces(repo_root)
        _write_progress(progress, f"wrote versioned corpus: {output_dataset_id}")
        return report
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        output_manifest_path.unlink(missing_ok=True)
        output_report_path.unlink(missing_ok=True)
        output_attribution_path.unlink(missing_ok=True)
        raise


def build_sonnets_expanded_v2(
    *,
    repo_root: Path,
    base_manifest_path: Path,
    snapshot_path: Path,
    output_dataset_id: str = "sonnets_expanded_v2",
    request_delay: float = 6.0,
    fetch_collection: FetchPinnedCollection = fetch_pinned_italian_wikisource_page_collection,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Backward-compatible v2 entry point for the original Alfieri build."""

    return build_sonnets_expanded(
        repo_root=repo_root,
        base_manifest_path=base_manifest_path,
        snapshot_path=snapshot_path,
        output_dataset_id=output_dataset_id,
        request_delay=request_delay,
        fetch_collection=fetch_collection,
        progress=progress,
    )


def read_snapshot(payload: dict[str, object]) -> WikisourceWorkSnapshot:
    """Validate the pinned source fields needed for an immutable rebuild."""

    root = payload["root_revision"]
    pages = payload["page_revisions"]
    if not isinstance(root, dict) or not isinstance(pages, list):
        raise ValueError("invalid sonnet source snapshot structure")
    snapshot = WikisourceWorkSnapshot(
        source_id=str(payload["source_id"]),
        landing_page_url=str(payload["landing_page_url"]),
        title=str(payload["title"]),
        scope=str(payload["scope"]),
        root_revision=WikisourcePageRevision(**root),
        page_revisions=[WikisourcePageRevision(**page) for page in pages],
        source_record_revisions=[
            WikisourcePageRevision(**record)
            for record in payload.get("source_record_revisions", [])
        ],
        edition_page_title_suffix=str(payload.get("edition_page_title_suffix", "")),
    )
    if snapshot.scope not in {"explicit_subpages", "explicit_edition_pages"}:
        raise ValueError("sonnet source snapshot must declare explicit eligible pages")
    if not snapshot.page_revisions:
        raise ValueError("sonnet source snapshot has no eligible pages")
    if snapshot.scope == "explicit_edition_pages":
        if len(snapshot.source_record_revisions) != len(snapshot.page_revisions):
            raise ValueError("edition-page snapshot has mismatched record and page counts")
        if not snapshot.edition_page_title_suffix:
            raise ValueError("edition-page snapshot has no edition title suffix")
    return snapshot


def read_manifest_rows(path: Path) -> list[ManifestRow]:
    """Read an existing processed corpus manifest as typed rows."""

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    boolean_fields = {
        "include_in_core_pre_petrarch",
        "include_in_expanded_with_petrarch",
        "include_in_training",
        "editorial_brackets_removed",
        "line_markers_removed",
    }
    typed_rows = []
    for row in rows:
        typed = {
            field: value == "True" if field in boolean_fields else value
            for field, value in row.items()
        }
        typed_rows.append(ManifestRow(**typed))
    return typed_rows


def copy_base_rows(
    rows: list[ManifestRow], repo_root: Path, output_poems_dir: Path
) -> list[ManifestRow]:
    """Copy v1 processed files and retarget their manifest paths to the new version."""

    copied: list[ManifestRow] = []
    for row in rows:
        source_path = repo_root / row.clean_text_path
        destination = output_poems_dir / source_path.name
        shutil.copyfile(source_path, destination)
        copied.append(
            replace(
                row,
                clean_text_path=str(destination.relative_to(repo_root)),
            )
        )
    return copied


def build_source_rows(
    *,
    collection: FetchedItalianWikisourcePageCollection,
    metadata: ActivatedSourceMetadata,
    output_poems_dir: Path,
    repo_root: Path,
    duplicate_index: dict[str, list[str]],
) -> list[ManifestRow]:
    """Revalidate and publish only the pre-approved source sonnet revisions."""

    rows: list[ManifestRow] = []
    for page in collection.pages:
        extracted = extract_poem_text(page.html)
        cleaned = clean_poem_text(extracted)
        if count_poem_lines(cleaned) != 14:
            raise ValueError(f"pinned source page is no longer a 14-line sonnet: {page.revision.title}")
        normalized = normalize_for_duplicate_check(cleaned)
        if duplicate_index.get(normalized):
            raise ValueError(
                "pinned source page duplicates an existing poem: "
                f"{page.revision.title} -> {duplicate_index[normalized]}"
            )
        poem_id = source_poem_id(metadata, page.revision.title)
        if any(row.poem_id == poem_id for row in rows):
            raise ValueError(f"duplicate source poem ID: {poem_id}")
        destination = output_poems_dir / f"{poem_id}.txt"
        destination.write_text(cleaned, encoding="utf-8")
        duplicate_index[normalized] = [poem_id]
        rows.append(
            ManifestRow(
                poem_id=poem_id,
                title_or_first_line=source_title(metadata, page.revision.title),
                author=metadata.author,
                displayed_author=metadata.author,
                source_archive=metadata.source_archive,
                source_collection=metadata.source_collection,
                source_subcollection="approved 14-line sonnets",
                source_url=url_from_title(page.revision.title),
                source_revision_id=str(page.revision.revision_id),
                source_revision_timestamp=page.revision.revision_timestamp,
                downloaded_at_utc=utc_now(),
                source_edition=metadata.source_edition,
                license_notes=metadata.license_notes,
                period=metadata.period,
                form="sonnet",
                form_evidence="revision-pinned source audit; exactly 14 cleaned poetic lines",
                count_method="line_count_14",
                attribution_status="secure",
                line_count_raw=count_poem_lines(extracted),
                line_count_clean=14,
                raw_text_path="",
                clean_text_path=str(destination.relative_to(repo_root)),
                include_in_core_pre_petrarch=False,
                include_in_expanded_with_petrarch=True,
                include_in_training=True,
                split_core_pre_petrarch="excluded",
                split_expanded_with_petrarch=assign_split(poem_id),
                editorial_brackets_removed=True,
                line_markers_removed=True,
                cleaning_notes="Removed isolated rendered bracket lines and displayed line markers; joined inline Wikisource markup; preserved spelling, punctuation, and verse line breaks.",
                audit_notes=f"Activated from revision-pinned source snapshot: {metadata.source_id}",
            )
        )
    return rows


def index_texts(rows: list[ManifestRow], repo_root: Path) -> dict[str, list[str]]:
    """Index source text before writing the expanded corpus to prevent leakage."""

    index: dict[str, list[str]] = {}
    for row in rows:
        text = (repo_root / row.clean_text_path).read_text(encoding="utf-8")
        index.setdefault(normalize_for_duplicate_check(text), []).append(row.poem_id)
    return index


def normalize_for_duplicate_check(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def source_poem_id(metadata: ActivatedSourceMetadata, page_title: str) -> str:
    """Create a stable filesystem-safe ID from an activated source page title."""

    suffix = page_title.rsplit("/", maxsplit=1)[-1]
    ascii_suffix = unicodedata.normalize("NFKD", suffix).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_suffix.casefold()).strip("_")
    if not slug:
        raise ValueError(f"could not derive source poem ID: {page_title}")
    prefix = metadata.poem_id_prefix or metadata.source_id.removeprefix("ws_").split("_", maxsplit=1)[0]
    return f"{prefix}_{slug}"


def source_title(metadata: ActivatedSourceMetadata, page_title: str) -> str:
    """Present a source page title without a collection-subpage prefix when present."""

    return page_title.removeprefix(f"{metadata.source_collection}/")


def split_counts(rows: list[ManifestRow]) -> dict[str, int]:
    counts = {"train": 0, "validation": 0, "test": 0}
    for row in rows:
        if row.include_in_expanded_with_petrarch:
            counts[row.split_expanded_with_petrarch] += 1
    return counts


def write_attribution(
    path: Path,
    output_dataset_id: str,
    metadata: ActivatedSourceMetadata,
    snapshot: WikisourceWorkSnapshot,
    rows: list[ManifestRow],
) -> None:
    """Write source-specific public attribution for the activated addition."""

    lines = [
        f"# {output_dataset_id.replace('_', ' ').title()} Attribution",
        "",
        f"This dataset version contains its declared base corpus plus {len(rows)} activated {metadata.author} sonnets.",
        "",
        "## Added Source",
        "",
        f"- Author: {metadata.author}",
        f"- Collection: {metadata.source_collection}",
        f"- Edition: {metadata.source_edition}",
        f"- Archive: {metadata.source_archive}",
        f"- Landing page: {snapshot.landing_page_url}",
        f"- Reuse metadata: {metadata.license_notes}",
        f"- Root revision: {snapshot.root_revision.revision_id} ({snapshot.root_revision.revision_timestamp})",
        f"- Activated revisions: {len(snapshot.page_revisions)}",
        f"- Snapshot scope: {snapshot.scope}",
        "",
        "Each poem's exact page URL and revision are recorded in the versioned manifest.",
    ]
    if snapshot.scope == "explicit_edition_pages":
        lines.extend(
            [
                "",
                "## Edition Resolution",
                "",
                "Each poem also records its bibliographic `Opera:` record in the committed snapshot.",
                f"- Approved edition selector: page title ending in `{snapshot.edition_page_title_suffix}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def delete_temporary_workspaces(repo_root: Path) -> None:
    """Remove temporary corpus workspaces after a successful versioned build."""

    for path in (repo_root / "data" / "raw", repo_root / "data" / "interim"):
        shutil.rmtree(path, ignore_errors=True)


def portable_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
