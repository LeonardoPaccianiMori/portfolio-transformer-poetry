"""Manifest rows and CSV writing."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


MANIFEST_FIELDS = [
    "poem_id",
    "title_or_first_line",
    "author",
    "displayed_author",
    "source_archive",
    "source_collection",
    "source_subcollection",
    "source_url",
    "source_revision_id",
    "source_revision_timestamp",
    "downloaded_at_utc",
    "source_edition",
    "license_notes",
    "period",
    "form",
    "form_evidence",
    "count_method",
    "attribution_status",
    "line_count_raw",
    "line_count_clean",
    "raw_text_path",
    "clean_text_path",
    "include_in_core_pre_petrarch",
    "include_in_expanded_with_petrarch",
    "include_in_training",
    "split_core_pre_petrarch",
    "split_expanded_with_petrarch",
    "editorial_brackets_removed",
    "line_markers_removed",
    "cleaning_notes",
    "audit_notes",
]


@dataclass
class ManifestRow:
    poem_id: str
    title_or_first_line: str
    author: str
    displayed_author: str
    source_archive: str
    source_collection: str
    source_subcollection: str
    source_url: str
    source_revision_id: str
    source_revision_timestamp: str
    downloaded_at_utc: str
    source_edition: str
    license_notes: str
    period: str
    form: str
    form_evidence: str
    count_method: str
    attribution_status: str
    line_count_raw: int | str
    line_count_clean: int | str
    raw_text_path: str
    clean_text_path: str
    include_in_core_pre_petrarch: bool
    include_in_expanded_with_petrarch: bool
    include_in_training: bool
    split_core_pre_petrarch: str
    split_expanded_with_petrarch: str
    editorial_brackets_removed: bool
    line_markers_removed: bool
    cleaning_notes: str
    audit_notes: str

    def validate(self) -> None:
        required = [
            self.poem_id,
            self.title_or_first_line,
            self.author,
            self.displayed_author,
            self.source_archive,
            self.source_collection,
            self.source_url,
            self.downloaded_at_utc,
            self.license_notes,
            self.period,
            self.form,
            self.form_evidence,
            self.count_method,
            self.attribution_status,
        ]
        if any(value == "" for value in required):
            raise ValueError(f"manifest row has empty required field: {self.poem_id}")

        if self.form != "sonnet":
            raise ValueError(f"only sonnet rows are supported: {self.poem_id}")

        allowed_methods = {
            "explicit_index_section",
            "wikisource_category",
            "line_count_14",
            "canonical_external_count",
            "manual_exclusion",
        }
        if self.count_method not in allowed_methods:
            raise ValueError(f"invalid count_method for {self.poem_id}: {self.count_method}")

        allowed_attribution = {"secure", "doubtful", "correspondence", "attributed", "unknown"}
        if self.attribution_status not in allowed_attribution:
            raise ValueError(
                f"invalid attribution_status for {self.poem_id}: {self.attribution_status}"
            )

        allowed_splits = {"", "train", "validation", "test", "excluded"}
        if self.split_core_pre_petrarch not in allowed_splits:
            raise ValueError(f"invalid core split for {self.poem_id}")
        if self.split_expanded_with_petrarch not in allowed_splits:
            raise ValueError(f"invalid expanded split for {self.poem_id}")


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    """Write manifest rows as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row.validate()

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def validate_processed_files(rows: list[ManifestRow], repo_root: Path) -> None:
    """Ensure every included row points to an existing processed poem file."""

    for row in rows:
        if not row.include_in_training:
            continue
        clean_path = repo_root / row.clean_text_path
        if not clean_path.is_file():
            raise FileNotFoundError(f"missing processed poem for {row.poem_id}: {clean_path}")
