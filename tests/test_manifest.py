from pathlib import Path

import pytest

from sonnet_corpus.manifest import ManifestRow, validate_processed_files, write_manifest


def make_row(**overrides):
    values = {
        "poem_id": "test_poem",
        "title_or_first_line": "Test poem",
        "author": "Author",
        "displayed_author": "Author",
        "source_archive": "Italian Wikisource",
        "source_collection": "Collection",
        "source_subcollection": "Sonetti",
        "source_url": "https://it.wikisource.org/wiki/Test",
        "source_revision_id": "",
        "source_revision_timestamp": "",
        "downloaded_at_utc": "2026-06-02T00:00:00+00:00",
        "source_edition": "",
        "license_notes": "CC BY-SA",
        "period": "XIII secolo",
        "form": "sonnet",
        "form_evidence": "line_count_14",
        "count_method": "line_count_14",
        "attribution_status": "secure",
        "line_count_raw": 14,
        "line_count_clean": 14,
        "raw_text_path": "",
        "clean_text_path": "data/processed/poems/test_poem.txt",
        "include_in_core_pre_petrarch": True,
        "include_in_expanded_with_petrarch": True,
        "include_in_training": True,
        "split_core_pre_petrarch": "train",
        "split_expanded_with_petrarch": "train",
        "editorial_brackets_removed": True,
        "line_markers_removed": True,
        "cleaning_notes": "",
        "audit_notes": "",
    }
    values.update(overrides)
    return ManifestRow(**values)


def test_manifest_validation_rejects_bad_method():
    row = make_row(count_method="bad")

    with pytest.raises(ValueError, match="invalid count_method"):
        row.validate()


def test_write_manifest_outputs_required_header(tmp_path: Path):
    path = tmp_path / "manifest.csv"

    write_manifest([make_row()], path)

    text = path.read_text(encoding="utf-8")
    assert "poem_id,title_or_first_line,author" in text
    assert "split_core_pre_petrarch" in text
    assert "split_expanded_with_petrarch" in text
    assert "\r" not in text


def test_validate_processed_files_checks_included_rows(tmp_path: Path):
    row = make_row()

    with pytest.raises(FileNotFoundError):
        validate_processed_files([row], tmp_path)

    poem_path = tmp_path / row.clean_text_path
    poem_path.parent.mkdir(parents=True)
    poem_path.write_text("line\n", encoding="utf-8")

    validate_processed_files([row], tmp_path)
