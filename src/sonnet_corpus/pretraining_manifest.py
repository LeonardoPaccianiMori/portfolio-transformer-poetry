"""Manifest rows and CSV writing for broader pretraining sources."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


PRETRAINING_MANIFEST_FIELDS = [
    "source_id",
    "title",
    "author",
    "source_archive",
    "source_collection",
    "landing_page_url",
    "download_url",
    "ebook_id",
    "language",
    "period_bucket",
    "approx_date",
    "genre",
    "text_kind",
    "inclusion_status",
    "public_domain_status",
    "license_notes",
    "edition_notes",
    "source_release_date",
    "source_last_updated",
    "expected_clean_text_path",
    "token_count_report_path",
    "split",
    "boilerplate_strategy",
    "mixed_text_strategy",
    "cleaning_notes",
    "audit_notes",
]


ALLOWED_SOURCE_ARCHIVES = {
    "Project Gutenberg",
    "Liber Liber",
    "Italian Wikisource",
    "Biblioteca Italiana",
}

ALLOWED_TEXT_KINDS = {
    "prose",
    "mixed",
    "poetry",
    "drama",
}

ALLOWED_INCLUSION_STATUSES = {
    "include_probe",
    "conditional_extract_prose",
    "audit_then_include",
    "tier_c_scale",
    "defer",
    "exclude",
}

ALLOWED_PERIOD_BUCKETS = {
    "tier_a_pre_1375",
    "tier_a_b_borderline",
    "tier_b_1376_1400",
    "tier_c_1401_1600",
    "tier_d_post_1600",
    "unknown",
}

ALLOWED_SPLITS = {
    "",
    "train",
    "validation",
    "test",
    "excluded",
}

@dataclass
class PretrainingSourceRow:
    source_id: str
    title: str
    author: str
    source_archive: str
    source_collection: str
    landing_page_url: str
    download_url: str
    ebook_id: str
    language: str
    period_bucket: str
    approx_date: str
    genre: str
    text_kind: str
    inclusion_status: str
    public_domain_status: str
    license_notes: str
    edition_notes: str
    source_release_date: str
    source_last_updated: str
    expected_clean_text_path: str
    token_count_report_path: str
    split: str
    boilerplate_strategy: str
    mixed_text_strategy: str
    cleaning_notes: str
    audit_notes: str

    def validate(self) -> None:
        required = [
            self.source_id,
            self.title,
            self.author,
            self.source_archive,
            self.landing_page_url,
            self.language,
            self.period_bucket,
            self.genre,
            self.text_kind,
            self.inclusion_status,
            self.public_domain_status,
            self.license_notes,
        ]
        if any(value == "" for value in required):
            raise ValueError(f"pretraining manifest row has empty required field: {self.source_id}")

        if self.source_archive not in ALLOWED_SOURCE_ARCHIVES:
            raise ValueError(f"invalid source_archive for {self.source_id}: {self.source_archive}")

        if self.text_kind not in ALLOWED_TEXT_KINDS:
            raise ValueError(f"invalid text_kind for {self.source_id}: {self.text_kind}")

        if self.inclusion_status not in ALLOWED_INCLUSION_STATUSES:
            raise ValueError(
                f"invalid inclusion_status for {self.source_id}: {self.inclusion_status}"
            )

        if self.period_bucket not in ALLOWED_PERIOD_BUCKETS:
            raise ValueError(f"invalid period_bucket for {self.source_id}: {self.period_bucket}")

        if self.split not in ALLOWED_SPLITS:
            raise ValueError(f"invalid split for {self.source_id}: {self.split}")

        self._validate_genre_policy()
        self._validate_source_policy()
        self._validate_split_policy()

    def _validate_genre_policy(self) -> None:
        if self.text_kind == "poetry" and self.inclusion_status not in {"exclude", "defer"}:
            raise ValueError(f"poetry source must be excluded or deferred: {self.source_id}")

        if self.text_kind == "drama" and self.inclusion_status not in {"exclude", "defer"}:
            raise ValueError(f"drama source must be excluded or deferred: {self.source_id}")

        if self.text_kind == "mixed" and self.inclusion_status != "conditional_extract_prose":
            raise ValueError(
                f"mixed source must use conditional_extract_prose: {self.source_id}"
            )

        if self.inclusion_status == "conditional_extract_prose" and self.mixed_text_strategy == "":
            raise ValueError(f"mixed/prose extraction strategy is required: {self.source_id}")

    def _validate_source_policy(self) -> None:
        if self.source_archive == "Project Gutenberg":
            if self.ebook_id == "":
                raise ValueError(f"Project Gutenberg row requires ebook_id: {self.source_id}")
            if "public domain" not in self.public_domain_status.lower():
                raise ValueError(
                    f"Project Gutenberg row requires public-domain status: {self.source_id}"
                )
            if self.boilerplate_strategy == "":
                raise ValueError(
                    f"Project Gutenberg row requires boilerplate_strategy: {self.source_id}"
                )

        if self.source_archive == "Liber Liber":
            license_text = f"{self.license_notes} {self.edition_notes}".lower()
            if "liber liber" not in license_text or "license" not in license_text:
                raise ValueError(
                    "Liber Liber row requires a separate license/edition-layer note: "
                    f"{self.source_id}"
                )

    def _validate_split_policy(self) -> None:
        if self.inclusion_status in {"exclude", "defer"} and self.split not in {"", "excluded"}:
            raise ValueError(f"excluded/deferred row cannot use active split: {self.source_id}")

        if self.split == "excluded" and self.inclusion_status not in {"exclude", "defer"}:
            raise ValueError(f"only excluded/deferred rows should use excluded split: {self.source_id}")


def write_pretraining_manifest(rows: list[PretrainingSourceRow], path: Path) -> None:
    """Write broader pretraining source rows as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row.validate()

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRETRAINING_MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_pretraining_manifest(path: Path) -> list[PretrainingSourceRow]:
    """Read broader pretraining source rows from CSV."""

    with path.open(encoding="utf-8", newline="") as handle:
        rows = [PretrainingSourceRow(**row) for row in csv.DictReader(handle)]

    for row in rows:
        row.validate()

    return rows
