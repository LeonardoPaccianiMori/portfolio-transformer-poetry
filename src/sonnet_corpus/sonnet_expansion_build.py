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
from .sonnet_wikisource_probe import extract_sonnet_candidate_segments
from .splits import assign_split
from .wikisource import url_from_title


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
    snapshot_scope: str = "explicit_subpages"


@dataclass(frozen=True)
class ApprovedSonnetCandidate:
    """One audited poem segment expected within a pinned source page."""

    page_title: str
    revision_id: int
    revision_timestamp: str
    segment_index: int = 1
    segment_label: str = ""
    cleaned_text_sha256: str = ""


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
    "ws_varchi_infermita": ActivatedSourceMetadata(
        source_id="ws_varchi_infermita",
        author="Benedetto Varchi",
        source_archive="Italian Wikisource",
        source_collection="Sonetti per la infermità, e guarigione di Cosimo I dei Medici",
        source_edition=(
            "Sonetti di mess. Benedetto Varchi per la infermità, e guarigione "
            "di Cosimo I dei Medici, Firenze, per il Magheri, 1821; original "
            "sonnets dated 1563."
        ),
        license_notes=(
            "CC BY-SA 3.0 / GFDL metadata on Italian Wikisource; retain page "
            "URL, author Benedetto Varchi, source scan Indice:Varchi - "
            "Sonetti.pdf, license links, and share-alike notice."
        ),
        period="XVI secolo",
        poem_id_prefix="varchi",
    ),
    "ws_andreini_rime_1601": ActivatedSourceMetadata(
        source_id="ws_andreini_rime_1601",
        author="Isabella Andreini",
        source_archive="Italian Wikisource",
        source_collection="Rime (Andreini)",
        source_edition=(
            "Rime d'Isabella Andreini Padovana Comica Gelosa, Milano, "
            "Girolamo Bordone e Pietromartire Locarni, 1601."
        ),
        license_notes=(
            "Italian Wikisource CC BY-SA 3.0 / GFDL page metadata; retain "
            "page URL, author, linked public-domain 1601 scan, license links, "
            "and share-alike notice."
        ),
        period="XVII secolo",
        poem_id_prefix="andreini",
    ),
    "ws_colonna_rime_1760": ActivatedSourceMetadata(
        source_id="ws_colonna_rime_1760",
        author="Vittoria Colonna",
        source_archive="Italian Wikisource",
        source_collection="Rime (Vittoria Colonna)",
        source_edition=(
            "Rime di Vittoria Colonna, Bergamo, Pietro Lancellotti, 1760."
        ),
        license_notes=(
            "Italian Wikisource CC BY-SA 3.0 / GFDL page metadata; retain "
            "page URL, author, source scan, license links, and share-alike notice."
        ),
        period="XVI secolo",
        poem_id_prefix="colonna",
    ),
    "ws_stampa_rime_1913": ActivatedSourceMetadata(
        source_id="ws_stampa_rime_1913",
        author="Gaspara Stampa",
        source_archive="Italian Wikisource",
        source_collection="Rime (Stampa)",
        source_edition=(
            "Gaspara Stampa - Veronica Franco, Rime, edited by Abdelkader "
            "Salza, Bari, Laterza, 1913; Gaspara Stampa section only."
        ),
        license_notes=(
            "Italian Wikisource CC BY-SA 3.0 / GFDL page metadata; retain "
            "page URL, author, source scan, editor, license links, and "
            "share-alike notice."
        ),
        period="XVI secolo",
        poem_id_prefix="stampa",
    ),
    "ws_ariosto_rime_varie_1857": ActivatedSourceMetadata(
        source_id="ws_ariosto_rime_varie_1857",
        author="Ludovico Ariosto",
        source_archive="Italian Wikisource",
        source_collection="Opere minori (Ariosto)/Rime varie",
        source_edition="Opere minori di Ludovico Ariosto, Firenze, 1857.",
        license_notes=(
            "Italian Wikisource CC BY-SA 3.0 / GFDL page metadata; retain "
            "page URL, author, source scan, license links, and share-alike notice."
        ),
        period="XVI secolo",
        poem_id_prefix="ariosto",
    ),
    "ws_sannazaro_rime_disperse": ActivatedSourceMetadata(
        source_id="ws_sannazaro_rime_disperse",
        author="Jacopo Sannazaro",
        source_archive="Italian Wikisource",
        source_collection="Rime disperse",
        source_edition=(
            "Opere volgari di Jacopo Sannazaro, edited by Alfredo Mauro, "
            "Bari, Laterza, 1961."
        ),
        license_notes=(
            "Italian Wikisource CC BY-SA 3.0 / GFDL page metadata; retain "
            "page URL, author, source edition, editor, license links, and "
            "share-alike notice."
        ),
        period="XVI secolo",
        poem_id_prefix="sannazaro",
        snapshot_scope="explicit_linked_pages",
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

    unique_page_candidates: list[dict[str, object]] = []
    unique_source_records: list[dict[str, object]] = []
    seen_pages: dict[str, tuple[int, str]] = {}
    for candidate in eligible:
        page_identity = (
            int(candidate["revision_id"]),
            str(candidate["revision_timestamp"]),
        )
        prior_identity = seen_pages.get(candidate["page_title"])
        if prior_identity is not None and prior_identity != page_identity:
            raise ValueError("one audited page has conflicting revision identities")
        if prior_identity is None:
            seen_pages[candidate["page_title"]] = page_identity
            unique_page_candidates.append(candidate)
            if uses_edition_records:
                unique_source_records.append(candidate["source_record"])

    payload = {
        "source_id": source_id,
        "landing_page_url": report["source"]["landing_page_url"],
        "title": root["title"],
        "scope": (
            "explicit_edition_pages"
            if uses_edition_records
            else metadata.snapshot_scope
        ),
        "root_revision": root,
        "page_revisions": [
            {
                "title": candidate["page_title"],
                "revision_id": candidate["revision_id"],
                "revision_timestamp": candidate["revision_timestamp"],
            }
            for candidate in unique_page_candidates
        ],
        "eligible_candidates": [
            {
                "page_title": candidate["page_title"],
                "revision_id": candidate["revision_id"],
                "revision_timestamp": candidate["revision_timestamp"],
                "segment_index": candidate.get("segment_index", 1),
                "segment_label": candidate.get("segment_label", ""),
                "cleaned_text_sha256": candidate.get("cleaned_text_sha256", ""),
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
                for record in unique_source_records
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
            "pinned_page_count": len(unique_page_candidates),
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

    return build_sonnets_expanded_from_snapshots(
        repo_root=repo_root,
        base_manifest_path=base_manifest_path,
        snapshot_paths=[snapshot_path],
        output_dataset_id=output_dataset_id,
        request_delay=request_delay,
        fetch_collection=fetch_collection,
        progress=progress,
    )


def build_sonnets_expanded_from_snapshots(
    *,
    repo_root: Path,
    base_manifest_path: Path,
    snapshot_paths: list[Path],
    output_dataset_id: str,
    request_delay: float = 6.0,
    fetch_collection: FetchPinnedCollection = fetch_pinned_italian_wikisource_page_collection,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Create one versioned corpus from a base plus multiple pinned sources."""

    if not snapshot_paths:
        raise ValueError("at least one source snapshot is required")
    sources = [read_activated_source(path) for path in snapshot_paths]
    source_ids = [snapshot.source_id for _, snapshot, _, _ in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source snapshots contain duplicate source IDs")

    output_root = repo_root / "data" / "processed" / output_dataset_id
    output_manifest_path = repo_root / "data" / "metadata" / f"{output_dataset_id}_manifest.csv"
    output_report_path = repo_root / "data" / "metadata" / f"{output_dataset_id}_build_report.json"
    output_attribution_path = repo_root / "data" / "metadata" / f"{output_dataset_id}_attribution.md"
    output_paths = (
        output_root,
        output_manifest_path,
        output_report_path,
        output_attribution_path,
    )
    prepare_output_destination(output_dataset_id, output_paths)
    staging_root = repo_root / "data" / "interim" / f"{output_dataset_id}_build"
    staging_poems_dir = staging_root / "poems"
    staging_manifest_path = repo_root / "data" / "interim" / f"{output_dataset_id}_manifest.csv"
    staging_report_path = repo_root / "data" / "interim" / f"{output_dataset_id}_build_report.json"
    staging_attribution_path = repo_root / "data" / "interim" / f"{output_dataset_id}_attribution.md"
    staging_paths = (
        staging_root,
        staging_manifest_path,
        staging_report_path,
        staging_attribution_path,
    )
    cleanup_paths(staging_paths)

    base_rows = [row for row in read_manifest_rows(base_manifest_path) if row.include_in_training]
    if not base_rows:
        raise ValueError("base manifest has no active processed poems")
    validate_candidate_poem_ids(sources, {row.poem_id for row in base_rows})
    duplicate_index = index_texts(base_rows, repo_root)
    started_at = utc_now()

    try:
        _write_progress(progress, f"copying {len(base_rows)} base processed poems")
        staging_poems_dir.mkdir(parents=True)
        copied_rows = copy_base_rows(base_rows, repo_root, staging_poems_dir)

        source_rows: list[ManifestRow] = []
        source_reports: list[dict[str, object]] = []
        occupied_poem_ids = {row.poem_id for row in copied_rows}
        total_pages = sum(len(snapshot.page_revisions) for _, snapshot, _, _ in sources)
        _write_progress(
            progress,
            f"rendering {total_pages} unique pinned pages from {len(sources)} sources",
        )
        for source_number, (
            snapshot_path,
            snapshot,
            metadata,
            candidates,
        ) in enumerate(sources, start=1):
            _write_progress(
                progress,
                f"source {source_number}/{len(sources)}: {snapshot.source_id} "
                f"({len(snapshot.page_revisions)} pages, {len(candidates)} sonnets)",
            )
            collection = fetch_collection(
                snapshot,
                request_delay=request_delay,
                progress=progress,
            )
            if [page.revision for page in collection.pages] != snapshot.page_revisions:
                raise ValueError(
                    "pinned fetch returned a page sequence different from the snapshot"
                )
            added_rows = build_source_rows(
                collection=collection,
                metadata=metadata,
                candidates=candidates,
                output_poems_dir=staging_poems_dir,
                repo_root=repo_root,
                duplicate_index=duplicate_index,
                occupied_poem_ids=occupied_poem_ids,
            )
            source_rows.extend(added_rows)
            source_reports.append(
                {
                    "source_id": snapshot.source_id,
                    "snapshot_path": portable_path(snapshot_path, repo_root),
                    "pinned_page_count": len(snapshot.page_revisions),
                    "added_poem_count": len(added_rows),
                    "root_revision_id": snapshot.root_revision.revision_id,
                    "root_revision_timestamp": (
                        snapshot.root_revision.revision_timestamp
                    ),
                }
            )
            _write_progress(
                progress,
                f"validated source {snapshot.source_id}: {len(added_rows)} sonnets",
            )
        combined_rows = retarget_clean_text_paths(
            copied_rows + source_rows,
            old_root=staging_root,
            new_root=output_root,
            repo_root=repo_root,
        )
        write_manifest(combined_rows, staging_manifest_path)
        write_attribution_for_sources(
            staging_attribution_path,
            output_dataset_id,
            [(snapshot, metadata) for _, snapshot, metadata, _ in sources],
            source_reports,
        )
        report = {
            "dataset_id": output_dataset_id,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "base_manifest_path": portable_path(base_manifest_path, repo_root),
            "snapshot_paths": [
                portable_path(snapshot_path, repo_root)
                for snapshot_path in snapshot_paths
            ],
            "base_poem_count": len(copied_rows),
            "added_poem_count": len(source_rows),
            "total_poem_count": len(combined_rows),
            "split_counts": split_counts(combined_rows),
            "sources": source_reports,
        }
        if len(snapshot_paths) == 1:
            report["snapshot_path"] = portable_path(snapshot_paths[0], repo_root)
        staging_report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_root.parent.mkdir(parents=True, exist_ok=True)
        staging_root.replace(output_root)
        staging_manifest_path.replace(output_manifest_path)
        staging_report_path.replace(output_report_path)
        staging_attribution_path.replace(output_attribution_path)
        delete_temporary_workspaces(repo_root)
        _write_progress(progress, f"wrote versioned corpus: {output_dataset_id}")
        return report
    except Exception:
        cleanup_paths(output_paths)
        cleanup_paths(staging_paths)
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
    if snapshot.scope not in {
        "explicit_subpages",
        "explicit_linked_pages",
        "explicit_edition_pages",
    }:
        raise ValueError("sonnet source snapshot must declare explicit eligible pages")
    if not snapshot.page_revisions:
        raise ValueError("sonnet source snapshot has no eligible pages")
    if snapshot.scope == "explicit_edition_pages":
        if len(snapshot.source_record_revisions) != len(snapshot.page_revisions):
            raise ValueError("edition-page snapshot has mismatched record and page counts")
        if not snapshot.edition_page_title_suffix:
            raise ValueError("edition-page snapshot has no edition title suffix")
    return snapshot


def read_activated_source(
    snapshot_path: Path,
) -> tuple[
    Path,
    WikisourceWorkSnapshot,
    ActivatedSourceMetadata,
    list[ApprovedSonnetCandidate],
]:
    """Read and cross-check one committed source snapshot and its poem segments."""

    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot = read_snapshot(payload)
    metadata = ActivatedSourceMetadata(**payload["source_metadata"])
    if metadata.source_id != snapshot.source_id:
        raise ValueError("snapshot metadata source ID does not match page snapshot")

    raw_candidates = payload.get("eligible_candidates")
    if raw_candidates is None:
        candidates = [
            ApprovedSonnetCandidate(
                page_title=revision.title,
                revision_id=revision.revision_id,
                revision_timestamp=revision.revision_timestamp,
            )
            for revision in snapshot.page_revisions
        ]
    elif isinstance(raw_candidates, list):
        candidates = [
            ApprovedSonnetCandidate(
                page_title=str(candidate["page_title"]),
                revision_id=int(candidate["revision_id"]),
                revision_timestamp=str(candidate["revision_timestamp"]),
                segment_index=int(candidate.get("segment_index", 1)),
                segment_label=str(candidate.get("segment_label", "")),
                cleaned_text_sha256=str(candidate.get("cleaned_text_sha256", "")),
            )
            for candidate in raw_candidates
        ]
    else:
        raise ValueError("snapshot eligible_candidates must be a list")
    if not candidates:
        raise ValueError("source snapshot has no eligible sonnet candidates")

    revision_by_title = {
        revision.title: revision for revision in snapshot.page_revisions
    }
    seen_segments: set[tuple[str, int]] = set()
    for candidate in candidates:
        revision = revision_by_title.get(candidate.page_title)
        if revision is None:
            raise ValueError("eligible candidate references an unpinned source page")
        if (
            candidate.revision_id != revision.revision_id
            or candidate.revision_timestamp != revision.revision_timestamp
        ):
            raise ValueError("eligible candidate revision differs from pinned page")
        segment_key = (candidate.page_title, candidate.segment_index)
        if segment_key in seen_segments:
            raise ValueError("source snapshot repeats an eligible poem segment")
        if candidate.segment_index < 1:
            raise ValueError("eligible candidate segment index must be positive")
        seen_segments.add(segment_key)
    return snapshot_path, snapshot, metadata, candidates


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


def retarget_clean_text_paths(
    rows: list[ManifestRow],
    *,
    old_root: Path,
    new_root: Path,
    repo_root: Path,
) -> list[ManifestRow]:
    """Retarget staged text paths to their final published corpus directory."""

    old_relative_root = old_root.relative_to(repo_root)
    new_relative_root = new_root.relative_to(repo_root)
    retargeted: list[ManifestRow] = []
    for row in rows:
        old_path = Path(row.clean_text_path)
        try:
            suffix = old_path.relative_to(old_relative_root)
        except ValueError as error:
            raise ValueError(
                "staged manifest row does not point inside the staged corpus: "
                f"{row.clean_text_path}"
            ) from error
        retargeted.append(
            replace(row, clean_text_path=str(new_relative_root / suffix))
        )
    return retargeted


def prepare_output_destination(output_dataset_id: str, paths: tuple[Path, ...]) -> None:
    """Reject a completed dataset and remove only an incomplete prior attempt."""

    existing_paths = [path for path in paths if path.exists()]
    if not existing_paths:
        return
    if len(existing_paths) == len(paths):
        raise FileExistsError(f"versioned corpus already exists: {output_dataset_id}")
    cleanup_paths(paths)


def cleanup_paths(paths: tuple[Path, ...]) -> None:
    """Remove explicit failed-build paths without touching other project data."""

    for path in paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def build_source_rows(
    *,
    collection: FetchedItalianWikisourcePageCollection,
    metadata: ActivatedSourceMetadata,
    candidates: list[ApprovedSonnetCandidate],
    output_poems_dir: Path,
    repo_root: Path,
    duplicate_index: dict[str, list[str]],
    occupied_poem_ids: set[str] | None = None,
) -> list[ManifestRow]:
    """Revalidate and publish only the pre-approved source sonnet revisions."""

    candidates_by_page: dict[str, list[ApprovedSonnetCandidate]] = {}
    for candidate in candidates:
        candidates_by_page.setdefault(candidate.page_title, []).append(candidate)
    occupied_poem_ids = occupied_poem_ids if occupied_poem_ids is not None else set()

    rows: list[ManifestRow] = []
    for page in collection.pages:
        page_candidates = candidates_by_page.pop(page.revision.title, [])
        if not page_candidates:
            raise ValueError(
                f"pinned source page has no approved candidate: {page.revision.title}"
            )
        extracted_segments = {
            segment.index: segment
            for segment in extract_sonnet_candidate_segments(
                source_id=metadata.source_id,
                page_title=page.revision.title,
                html=page.html,
            )
        }
        for candidate in sorted(page_candidates, key=lambda item: item.segment_index):
            segment = extracted_segments.get(candidate.segment_index)
            if segment is None:
                raise ValueError(
                    "approved poem segment is absent from pinned source page: "
                    f"{candidate.page_title} segment {candidate.segment_index}"
                )
            if candidate.segment_label and segment.label != candidate.segment_label:
                raise ValueError(
                    "approved poem segment label changed: "
                    f"{candidate.page_title} segment {candidate.segment_index}"
                )
            cleaned = clean_poem_text(segment.raw_text)
            if count_poem_lines(cleaned) != 14:
                raise ValueError(
                    "pinned source segment is no longer a 14-line sonnet: "
                    f"{candidate.page_title} segment {candidate.segment_index}"
                )
            cleaned_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
            if (
                candidate.cleaned_text_sha256
                and cleaned_hash != candidate.cleaned_text_sha256
            ):
                raise ValueError(
                    "pinned source text differs from its audited fingerprint: "
                    f"{candidate.page_title} segment {candidate.segment_index}"
                )
            normalized = normalize_for_duplicate_check(cleaned)
            if duplicate_index.get(normalized):
                raise ValueError(
                    "pinned source segment duplicates an existing poem: "
                    f"{candidate.page_title} -> {duplicate_index[normalized]}"
                )
            poem_id = source_poem_id(
                metadata,
                candidate.page_title,
                segment_label=candidate.segment_label,
            )
            if poem_id in occupied_poem_ids:
                raise ValueError(f"duplicate source poem ID: {poem_id}")
            occupied_poem_ids.add(poem_id)
            destination = output_poems_dir / f"{poem_id}.txt"
            destination.write_text(cleaned, encoding="utf-8")
            duplicate_index[normalized] = [poem_id]
            rows.append(
                ManifestRow(
                    poem_id=poem_id,
                    title_or_first_line=source_title(
                        metadata,
                        candidate.page_title,
                        segment_label=candidate.segment_label,
                    ),
                    author=metadata.author,
                    displayed_author=metadata.author,
                    source_archive=metadata.source_archive,
                    source_collection=metadata.source_collection,
                    source_subcollection="approved 14-line sonnets",
                    source_url=url_from_title(candidate.page_title),
                    source_revision_id=str(candidate.revision_id),
                    source_revision_timestamp=candidate.revision_timestamp,
                    downloaded_at_utc=utc_now(),
                    source_edition=metadata.source_edition,
                    license_notes=metadata.license_notes,
                    period=metadata.period,
                    form="sonnet",
                    form_evidence=(
                        "revision-pinned source audit; exactly 14 cleaned poetic lines"
                    ),
                    count_method="line_count_14",
                    attribution_status="secure",
                    line_count_raw=count_poem_lines(segment.raw_text),
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
                    cleaning_notes=(
                        "Removed isolated rendered bracket lines and displayed "
                        "line markers; joined inline Wikisource markup; preserved "
                        "spelling, punctuation, and verse line breaks."
                    ),
                    audit_notes=(
                        "Activated from revision-pinned source snapshot: "
                        f"{metadata.source_id}; segment {candidate.segment_index}"
                    ),
                )
            )
    if candidates_by_page:
        raise ValueError("eligible candidates were not present in the pinned fetch")
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


def source_poem_id(
    metadata: ActivatedSourceMetadata,
    page_title: str,
    *,
    segment_label: str = "",
) -> str:
    """Create a stable filesystem-safe ID from an activated source page title."""

    relative_page_title = page_title.removeprefix(
        f"{metadata.source_collection}/"
    )
    suffix = (
        f"{relative_page_title}/{segment_label}"
        if segment_label
        else relative_page_title
    )
    ascii_suffix = unicodedata.normalize("NFKD", suffix).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_suffix.casefold()).strip("_")
    if not slug:
        raise ValueError(f"could not derive source poem ID: {page_title}")
    prefix = metadata.poem_id_prefix or metadata.source_id.removeprefix("ws_").split("_", maxsplit=1)[0]
    return f"{prefix}_{slug}"


def validate_candidate_poem_ids(
    sources: list[
        tuple[
            Path,
            WikisourceWorkSnapshot,
            ActivatedSourceMetadata,
            list[ApprovedSonnetCandidate],
        ]
    ],
    occupied_poem_ids: set[str],
) -> None:
    """Reject all candidate ID collisions before making network requests."""

    seen = set(occupied_poem_ids)
    for _, _, metadata, candidates in sources:
        for candidate in candidates:
            poem_id = source_poem_id(
                metadata,
                candidate.page_title,
                segment_label=candidate.segment_label,
            )
            if poem_id in seen:
                raise ValueError(f"duplicate source poem ID: {poem_id}")
            seen.add(poem_id)


def source_title(
    metadata: ActivatedSourceMetadata,
    page_title: str,
    *,
    segment_label: str = "",
) -> str:
    """Present a source page title without a collection-subpage prefix when present."""

    if segment_label:
        return segment_label
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

    write_attribution_for_sources(
        path,
        output_dataset_id,
        [(snapshot, metadata)],
        [
            {
                "source_id": snapshot.source_id,
                "pinned_page_count": len(snapshot.page_revisions),
                "added_poem_count": len(rows),
            }
        ],
    )


def write_attribution_for_sources(
    path: Path,
    output_dataset_id: str,
    sources: list[tuple[WikisourceWorkSnapshot, ActivatedSourceMetadata]],
    source_reports: list[dict[str, object]],
) -> None:
    """Write public attribution for every source added in a dataset version."""

    report_by_source = {
        str(report["source_id"]): report for report in source_reports
    }
    total_added = sum(int(report["added_poem_count"]) for report in source_reports)
    lines = [
        f"# {output_dataset_id.replace('_', ' ').title()} Attribution",
        "",
        (
            "This dataset version contains its declared base corpus plus "
            f"{total_added} activated sonnets from {len(sources)} source collections."
        ),
        "",
        "## Added Sources",
        "",
    ]
    for snapshot, metadata in sources:
        report = report_by_source[snapshot.source_id]
        lines.extend(
            [
                f"### {metadata.author}",
                "",
                f"- Collection: {metadata.source_collection}",
                f"- Edition: {metadata.source_edition}",
                f"- Archive: {metadata.source_archive}",
                f"- Landing page: {snapshot.landing_page_url}",
                f"- Reuse metadata: {metadata.license_notes}",
                (
                    f"- Root revision: {snapshot.root_revision.revision_id} "
                    f"({snapshot.root_revision.revision_timestamp})"
                ),
                f"- Activated sonnets: {report['added_poem_count']}",
                f"- Pinned pages: {report['pinned_page_count']}",
                f"- Snapshot scope: {snapshot.scope}",
                "",
            ]
        )
        if snapshot.scope == "explicit_edition_pages":
            lines.extend(
                [
                    (
                        "Each poem also records its bibliographic `Opera:` "
                        "record in the committed snapshot."
                    ),
                    (
                        "- Approved edition selector: page title ending in "
                        f"`{snapshot.edition_page_title_suffix}`"
                    ),
                    "",
                ]
            )
    lines.extend(
        [
            "## Record-Level Provenance",
            "",
            (
                "Each poem's exact page URL and revision are recorded in the "
                "versioned manifest. The committed snapshots also record the "
                "audited poem segment and its cleaned-text fingerprint."
            ),
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
