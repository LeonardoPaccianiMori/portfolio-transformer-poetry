import csv
from pathlib import Path

from sonnet_corpus.manifest import MANIFEST_FIELDS
from sonnet_corpus.minerva_sft_audit import audit_minerva_sft_corpus


def _row(poem_id: str, split: str, path: str, period: str) -> dict[str, str]:
    row = {field: "" for field in MANIFEST_FIELDS}
    row.update({
        "poem_id": poem_id,
        "title_or_first_line": poem_id,
        "author": f"Author {period}",
        "source_archive": "Archive",
        "source_collection": f"Collection {period}",
        "period": period,
        "clean_text_path": path,
        "include_in_expanded_with_petrarch": "True",
        "split_expanded_with_petrarch": split,
        "editorial_brackets_removed": "True",
        "line_markers_removed": "False",
        "cleaning_notes": "Preserved text.",
    })
    return row


def _write_fixture(tmp_path: Path, *, duplicate: bool = False) -> Path:
    rows = []
    base_lines = [f"Verso numero {index}" for index in range(1, 15)]
    for index, split in enumerate(("train", "validation", "test")):
        poem_id = f"poem_{index}"
        relative_path = f"poems/{poem_id}.txt"
        poem_path = tmp_path / relative_path
        poem_path.parent.mkdir(parents=True, exist_ok=True)
        lines = base_lines if duplicate else [f"{line} variante {index}" for line in base_lines]
        poem_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        rows.append(_row(poem_id, split, relative_path, f"Period {index}"))

    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def test_minerva_sft_audit_writes_composition_and_review_artifacts(tmp_path):
    manifest_path = _write_fixture(tmp_path)
    report = audit_minerva_sft_corpus(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        dataset="expanded_with_petrarch",
        json_report_path=Path("report.json"),
        markdown_report_path=Path("report.md"),
        review_sample_path=Path("review.md"),
        review_sample_size=1,
    )

    assert report["selected_poem_count"] == 3
    assert report["split_poem_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert report["automated_structural_gate"] == "pass"
    assert "Minerva V5 SFT Corpus Audit" in (tmp_path / "report.md").read_text()
    assert "Syntax/edition review: TODO" in (tmp_path / "review.md").read_text()


def test_minerva_sft_audit_flags_duplicate_and_markup(tmp_path):
    manifest_path = _write_fixture(tmp_path, duplicate=True)
    train_path = tmp_path / "poems/poem_0.txt"
    train_path.write_text(
        train_path.read_text().replace("Verso numero 1", "{{Verso}}", 1),
        encoding="utf-8",
    )

    report = audit_minerva_sft_corpus(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        dataset="expanded_with_petrarch",
        json_report_path=Path("report.json"),
        markdown_report_path=Path("report.md"),
        review_sample_path=Path("review.md"),
    )

    assert report["automated_structural_gate"] == "review_required"
    assert report["suspicious_marker_counts"]["wiki_template"] == 2
    assert report["cross_split_duplicate_group_count"] == 1
    assert any(
        issue["issue_type"] == "marker:wiki_template"
        for issue in report["structural_issues"]
    )
