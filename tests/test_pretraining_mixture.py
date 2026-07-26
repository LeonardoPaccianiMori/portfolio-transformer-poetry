import json
from pathlib import Path

import pytest

from sonnet_corpus.pretraining_mixture import (
    PretrainingComponent,
    PretrainingMixtureConfig,
    build_pretraining_mixture,
    publish_pretraining_component,
)


def write_component(
    root: Path,
    *,
    source_id: str,
    text: str,
    author: str = "Test Author",
) -> tuple[Path, Path]:
    processed_dir = root / "processed"
    source_dir = processed_dir / "sources"
    source_dir.mkdir(parents=True)
    source_path = source_dir / f"{source_id}.txt"
    source_path.write_text(text, encoding="utf-8")
    (processed_dir / "corpus.txt").write_text(text + "\n", encoding="utf-8")
    report = {
        "corpus_version": "test_component",
        "processed_dir": str(processed_dir),
        "combined_corpus_path": str(processed_dir / "corpus.txt"),
        "sources": [
            {
                "source_id": source_id,
                "title": source_id,
                "author": author,
                "source_archive": "Test archive",
                "processed_path": str(source_path),
                "cleaned_character_count": len(text),
                "cleaned_word_count": len(text.split()),
            }
        ],
    }
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return processed_dir, report_path


def test_publish_pretraining_component_copies_text_and_rewrites_paths(tmp_path: Path):
    source_dir, source_report = write_component(
        tmp_path / "local", source_id="one", text="Uno due"
    )
    target_dir = tmp_path / "public" / "component"
    target_report = tmp_path / "public" / "report.json"

    report = publish_pretraining_component(
        source_processed_dir=source_dir,
        source_report_path=source_report,
        target_processed_dir=target_dir,
        target_report_path=target_report,
    )

    assert (target_dir / "sources/one.txt").read_text(encoding="utf-8") == "Uno due"
    assert report["processed_dir"] == str(target_dir)
    assert report["sources"][0]["processed_path"] == str(target_dir / "sources/one.txt")
    assert json.loads(target_report.read_text(encoding="utf-8")) == report


def test_build_pretraining_mixture_joins_unique_sources_and_reports_exception(tmp_path: Path):
    first_dir, first_report = write_component(
        tmp_path / "first",
        source_id="ramusio",
        text="Primo testo molto piu lungo",
        author="Ramusio",
    )
    second_dir, second_report = write_component(
        tmp_path / "second", source_id="verri", text="Secondo testo lungo", author="Verri"
    )
    config = PretrainingMixtureConfig(
        corpus_version="mixture",
        components=(
            PretrainingComponent("first", first_dir, first_report),
            PretrainingComponent("second", second_dir, second_report),
        ),
        processed_dir=tmp_path / "output" / "processed",
        report_path=tmp_path / "output" / "report.json",
        markdown_report_path=tmp_path / "output" / "report.md",
        work_cap_warning=0.5,
        author_cap_warning=0.5,
        approved_work_cap_exceptions=("ramusio",),
        approved_author_cap_exceptions=("Ramusio",),
    )

    report = build_pretraining_mixture(config)

    assert (config.processed_dir / "corpus.txt").read_text(encoding="utf-8") == (
        "Primo testo molto piu lungo\n\nSecondo testo lungo\n"
    )
    assert report["source_count"] == 2
    assert {warning["name"] for warning in report["concentration_warnings"]} == {
        "ramusio",
        "Ramusio",
    }
    assert all(warning["approved_exception"] for warning in report["concentration_warnings"])


def test_build_pretraining_mixture_rejects_duplicate_source_ids(tmp_path: Path):
    first_dir, first_report = write_component(tmp_path / "first", source_id="same", text="Uno")
    second_dir, second_report = write_component(tmp_path / "second", source_id="same", text="Due")
    config = PretrainingMixtureConfig(
        corpus_version="mixture",
        components=(
            PretrainingComponent("first", first_dir, first_report),
            PretrainingComponent("second", second_dir, second_report),
        ),
        processed_dir=tmp_path / "output" / "processed",
        report_path=tmp_path / "output" / "report.json",
        markdown_report_path=tmp_path / "output" / "report.md",
        work_cap_warning=0.5,
        author_cap_warning=0.5,
    )

    with pytest.raises(ValueError, match="duplicate source ID: same"):
        build_pretraining_mixture(config)
