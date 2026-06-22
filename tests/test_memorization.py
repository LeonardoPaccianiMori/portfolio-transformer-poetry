import csv
import json
from pathlib import Path

import pytest

from sonnet_evaluation.memorization import (
    build_memorization_report,
    character_ngram_set,
    find_nearest_training_record,
    load_training_records,
    longest_common_substring_length,
    memorization_risk_level,
    ngram_containment,
    normalize_for_memorization,
    score_generation_memorization,
    write_memorization_report,
)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def write_manifest(path: Path) -> None:
    rows = [
        {
            "poem_id": "train_poem",
            "title_or_first_line": "Amor che move",
            "author": "Author One",
            "clean_text_path": "data/processed/poems/train_poem.txt",
            "include_in_expanded_with_petrarch": "True",
            "split_expanded_with_petrarch": "train",
        },
        {
            "poem_id": "validation_poem",
            "title_or_first_line": "Donna gentile",
            "author": "Author Two",
            "clean_text_path": "data/processed/poems/validation_poem.txt",
            "include_in_expanded_with_petrarch": "True",
            "split_expanded_with_petrarch": "validation",
        },
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_processed_poems(repo_root: Path) -> None:
    poem_dir = repo_root / "data" / "processed" / "poems"
    poem_dir.mkdir(parents=True)
    (poem_dir / "train_poem.txt").write_text(
        "Amor che move il sole e l'altre stelle\n",
        encoding="utf-8",
    )
    (poem_dir / "validation_poem.txt").write_text(
        "Donna gentile porta luce nova\n",
        encoding="utf-8",
    )


def write_generation_directory(generation_dir: Path) -> None:
    generation_dir.mkdir(parents=True)
    output_path = generation_dir / "amor.txt"
    output_path.write_text(
        "Amor che move il sole e poi ritorna\n",
        encoding="utf-8",
    )
    write_json(
        generation_dir / "metadata.json",
        {
            "generated_files": [
                {
                    "prompt_id": "amor",
                    "prompt_text": "Amor",
                    "path": str(output_path),
                    "seed": 1337,
                },
            ],
        },
    )


def test_normalize_for_memorization_lowercases_and_collapses_whitespace():
    assert normalize_for_memorization(" Amor\n  CHE\tmove ") == "amor che move"


def test_character_ngram_set_returns_unique_ngrams():
    assert character_ngram_set("ababa", ngram_size=3) == {"aba", "bab"}


def test_character_ngram_set_rejects_invalid_ngram_size():
    with pytest.raises(ValueError, match="ngram_size"):
        character_ngram_set("abc", ngram_size=0)


def test_ngram_containment_measures_generated_overlap():
    containment = ngram_containment(
        generated_text="abcdef",
        reference_text="xxabcxy",
        ngram_size=3,
    )

    assert containment == 1 / 4


def test_longest_common_substring_length_uses_normalized_text():
    length = longest_common_substring_length(
        "Prima\nSeconda",
        "xx prima seconda yy",
    )

    assert length == len("prima seconda")


def test_memorization_risk_level_uses_thresholds():
    assert memorization_risk_level(0.01, 10) == "low"
    assert memorization_risk_level(0.15, 10) == "medium"
    assert memorization_risk_level(0.30, 10) == "high"
    assert memorization_risk_level(0.01, 160) == "high"


def test_load_training_records_selects_only_requested_split(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path)
    write_processed_poems(tmp_path)

    records = load_training_records(
        manifest_path=manifest_path,
        repo_root=tmp_path,
        dataset="expanded_with_petrarch",
        split="train",
    )

    assert len(records) == 1
    assert records[0]["poem_id"] == "train_poem"
    assert records[0]["text"].startswith("Amor")


def test_find_nearest_training_record_selects_best_overlap():
    records = [
        {
            "poem_id": "weak",
            "title_or_first_line": "Weak",
            "author": "A",
            "clean_text_path": "weak.txt",
            "text": "completely different text",
        },
        {
            "poem_id": "strong",
            "title_or_first_line": "Strong",
            "author": "B",
            "clean_text_path": "strong.txt",
            "text": "amor che move il sole e l'altre stelle",
        },
    ]

    nearest = find_nearest_training_record(
        generated_text="amor che move il sole",
        training_records=records,
        ngram_size=5,
    )

    assert nearest["nearest_poem_id"] == "strong"
    assert nearest["ngram_containment"] > 0.0


def test_find_nearest_training_record_rejects_empty_records():
    with pytest.raises(ValueError, match="training_records"):
        find_nearest_training_record(
            generated_text="Amor",
            training_records=[],
        )


def test_score_generation_memorization_scores_each_output(tmp_path):
    generation_dir = tmp_path / "generation"
    write_generation_directory(generation_dir)
    training_records = [
        {
            "poem_id": "train_poem",
            "title_or_first_line": "Amor che move",
            "author": "Author One",
            "clean_text_path": "train.txt",
            "text": "Amor che move il sole e l'altre stelle",
        },
    ]

    rows = score_generation_memorization(
        generation_dir=generation_dir,
        training_records=training_records,
        ngram_size=5,
    )

    assert len(rows) == 1
    assert rows[0]["prompt_id"] == "amor"
    assert rows[0]["nearest_poem_id"] == "train_poem"
    assert rows[0]["longest_common_substring_chars"] > 0


def test_build_memorization_report_contains_table_and_notes():
    rows = [
        {
            "prompt_id": "amor",
            "generated_character_count": 100,
            "nearest_title_or_first_line": "Amor | che move",
            "nearest_author": "Author One",
            "ngram_containment": 0.25,
            "longest_common_substring_chars": 90,
            "risk_level": "medium",
            "seed": 1337,
        },
    ]

    report = build_memorization_report(
        generation_dir=Path("outputs/generations/run"),
        dataset="expanded_with_petrarch",
        split="train",
        ngram_size=40,
        rows=rows,
    )

    assert "# Memorization Checks" in report
    assert "| Prompt | Chars |" in report
    assert r"Amor \| che move" in report
    assert "## Notes" in report


def test_write_memorization_report_writes_markdown(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    generation_dir = tmp_path / "generation"
    output_path = tmp_path / "reports" / "memorization.md"
    write_manifest(manifest_path)
    write_processed_poems(tmp_path)
    write_generation_directory(generation_dir)

    rows = write_memorization_report(
        generation_dir=generation_dir,
        manifest_path=manifest_path,
        repo_root=tmp_path,
        dataset="expanded_with_petrarch",
        split="train",
        output_path=output_path,
        ngram_size=5,
    )

    report = output_path.read_text(encoding="utf-8")

    assert len(rows) == 1
    assert output_path.is_file()
    assert "Amor che move" in report
