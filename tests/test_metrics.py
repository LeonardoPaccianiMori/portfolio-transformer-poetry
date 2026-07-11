import json
from pathlib import Path

import pytest

from sonnet_evaluation.metrics import (
    build_generation_metrics_report,
    non_empty_line_count,
    repeated_ngram_ratio,
    resolve_generated_path,
    score_generated_text,
    score_generation_directory,
    unique_character_ratio,
    write_generation_metrics_report,
)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def write_generation_directory(generation_dir: Path) -> None:
    generation_dir.mkdir(parents=True)
    first_path = generation_dir / "amor.txt"
    second_path = generation_dir / "donna.txt"
    first_path.write_text(
        "Amor\nche move\n<|poem_end|>\n",
        encoding="utf-8",
    )
    second_path.write_text(
        "Donnaaaaa",
        encoding="utf-8",
    )
    write_json(
        generation_dir / "metadata.json",
        {
            "generated_files": [
                {
                    "prompt_id": "amor",
                    "prompt_text": "Amor",
                    "path": str(first_path),
                    "seed": 1337,
                },
                {
                    "prompt_id": "donna",
                    "prompt_text": "Donna",
                    "path": str(second_path),
                    "seed": 1338,
                },
            ],
            "stop_text": "<|poem_end|>",
        },
    )


def test_non_empty_line_count_ignores_blank_lines():
    assert non_empty_line_count("Amor\n\nche move\n") == 2


def test_unique_character_ratio_returns_fraction_of_unique_characters():
    assert unique_character_ratio("aabb") == 0.5


def test_unique_character_ratio_returns_zero_for_empty_text():
    assert unique_character_ratio("") == 0.0


def test_repeated_ngram_ratio_detects_repeated_ngrams():
    assert repeated_ngram_ratio("aaaaaa", ngram_size=2) > 0.0


def test_repeated_ngram_ratio_returns_zero_for_short_text():
    assert repeated_ngram_ratio("abc", ngram_size=4) == 0.0


def test_repeated_ngram_ratio_rejects_invalid_ngram_size():
    with pytest.raises(ValueError, match="ngram_size"):
        repeated_ngram_ratio("abc", ngram_size=0)


def test_resolve_generated_path_falls_back_to_metadata_directory(tmp_path):
    generation_dir = tmp_path / "generation"
    generation_dir.mkdir()
    output_path = generation_dir / "amor.txt"
    output_path.write_text("Amor", encoding="utf-8")
    metadata_path = generation_dir / "metadata.json"

    resolved_path = resolve_generated_path(
        path_text="missing/prefix/amor.txt",
        metadata_path=metadata_path,
    )

    assert resolved_path == output_path


def test_score_generated_text_returns_basic_metrics():
    metrics = score_generated_text(
        text="Amor\nche move\n<|poem_end|>\n",
        prompt_text="Amor",
        ngram_size=4,
    )

    assert metrics["character_count"] == 27
    assert metrics["non_empty_line_count"] == 3
    assert metrics["boundary_marker_count"] == 1
    assert metrics["unique_character_ratio"] > 0.0
    assert metrics["repetition_ratio"] >= 0.0
    assert metrics["prompt_preserved"]


def test_score_generation_directory_scores_each_generated_file(tmp_path):
    generation_dir = tmp_path / "generation"
    write_generation_directory(generation_dir)

    rows = score_generation_directory(generation_dir)

    assert len(rows) == 2
    assert rows[0]["prompt_id"] == "amor"
    assert rows[0]["prompt_preserved"]
    assert rows[0]["boundary_marker_count"] == 1
    assert rows[1]["prompt_id"] == "donna"


def test_build_generation_metrics_report_contains_table_and_notes(tmp_path):
    generation_dir = tmp_path / "generation"
    rows = [
        {
            "prompt_id": "amor",
            "character_count": 10,
            "non_empty_line_count": 2,
            "boundary_marker_count": 1,
            "boundary_marker": "<|poem_end|>",
            "unique_character_ratio": 0.5,
            "repetition_ratio": 0.25,
            "prompt_preserved": True,
            "seed": 1337,
        },
    ]

    report = build_generation_metrics_report(
        generation_dir=generation_dir,
        rows=rows,
    )

    assert "# Generation Metrics" in report
    assert "| Prompt | Chars | Lines |" in report
    assert "amor" in report
    assert "## Notes" in report


def test_score_generation_directory_uses_stop_text_as_boundary_marker(tmp_path):
    generation_dir = tmp_path / "generation"
    write_generation_directory(generation_dir)
    metadata_path = generation_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["stop_text"] = "<|endoftext|>"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    rows = score_generation_directory(generation_dir)

    assert rows[0]["boundary_marker"] == "<|endoftext|>"
    assert rows[0]["boundary_marker_count"] == 0


def test_write_generation_metrics_report_writes_markdown(tmp_path):
    generation_dir = tmp_path / "generation"
    output_path = tmp_path / "reports" / "generation_metrics.md"
    write_generation_directory(generation_dir)

    rows = write_generation_metrics_report(
        generation_dir=generation_dir,
        output_path=output_path,
    )

    report = output_path.read_text(encoding="utf-8")

    assert len(rows) == 2
    assert output_path.is_file()
    assert "amor" in report
    assert "donna" in report
