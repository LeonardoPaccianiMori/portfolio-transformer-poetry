import csv
import json
from pathlib import Path

import pytest

from sonnet_corpus.gutenberg_catalog_inventory import INVENTORY_FIELDS
from sonnet_corpus.gutenberg_metadata_review_queue import (
    GutenbergMetadataReviewQueueConfig,
    freeze_gutenberg_metadata_review_queue,
)


def _row(ebook_id: str, status: str, role: str = "date_and_role_review"):
    row = {field: "" for field in INVENTORY_FIELDS}
    row.update(
        {
            "ebook_id": ebook_id,
            "title": f"Title {ebook_id}",
            "inventory_status": status,
            "preliminary_role": role,
        }
    )
    return row


def _write_inventory(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _config(tmp_path: Path) -> GutenbergMetadataReviewQueueConfig:
    return GutenbergMetadataReviewQueueConfig(
        repo_root=tmp_path,
        inventory_csv_path=tmp_path / "inventory.csv",
        queue_csv_path=tmp_path / "queue.csv",
        json_report_path=tmp_path / "report.json",
        markdown_report_path=tmp_path / "report.md",
    )


def test_freeze_gutenberg_metadata_review_queue_accounts_for_every_route(tmp_path):
    config = _config(tmp_path)
    _write_inventory(
        config.inventory_csv_path,
        [
            _row("5", "review_work_publication_date"),
            _row("4", "review_missing_period_evidence"),
            _row("3", "review_translation_edition_date", "historical_general_candidate"),
            _row("2", "review_language_variety_before_download", "language_variety_review_required"),
            _row("1", "review_rights"),
            _row("6", "audit_then_deduplicate", "historical_general_candidate"),
            _row("7", "deduplicate_before_full_text_audit", "historical_general_candidate"),
            _row("8", "already_registered_project_gutenberg_source"),
            _row("9", "exclude_core_language_variety_metadata"),
            _row("10", "exclude_metadata_scope"),
        ],
    )

    report = freeze_gutenberg_metadata_review_queue(config)

    assert report["review_record_count"] == 5
    assert report["inventory_accounting"] == {
        "eligible_fulltext_probe": 2,
        "metadata_review_queue": 5,
        "already_registered": 1,
        "metadata_exclusions": 2,
    }
    with config.queue_csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["ebook_id"] for row in rows] == ["1", "2", "3", "4", "5"]
    assert all(row["resolution_evidence_required"] for row in rows)
    assert json.loads(config.json_report_path.read_text())["policy"][
        "activation_authorized"
    ] is False
    assert "It is not the unresolved queue size" in config.markdown_report_path.read_text()


def test_freeze_gutenberg_metadata_review_queue_rejects_incomplete_accounting(tmp_path):
    config = _config(tmp_path)
    _write_inventory(config.inventory_csv_path, [_row("1", "unexpected_status")])

    with pytest.raises(ValueError, match="accounting is incomplete"):
        freeze_gutenberg_metadata_review_queue(config)
