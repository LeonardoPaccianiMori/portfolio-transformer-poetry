import json
from pathlib import Path

import pytest

from sonnet_evaluation.qualitative import (
    build_qualitative_review_report,
    fenced_text_block,
    load_generated_reviews,
    markdown_heading_text,
    markdown_review_section,
    read_generation_metadata,
    write_qualitative_review_report,
)


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def write_generation_directory(generation_dir: Path) -> None:
    generation_dir.mkdir(parents=True)
    output_path = generation_dir / "amor.txt"
    output_path.write_text(
        "Amor\nche move\n",
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


def test_read_generation_metadata_rejects_non_object(tmp_path):
    generation_dir = tmp_path / "generation"
    generation_dir.mkdir()
    write_json(generation_dir / "metadata.json", [])

    with pytest.raises(ValueError, match="JSON object"):
        read_generation_metadata(generation_dir)


def test_read_generation_metadata_requires_generated_files(tmp_path):
    generation_dir = tmp_path / "generation"
    generation_dir.mkdir()
    write_json(generation_dir / "metadata.json", {})

    with pytest.raises(ValueError, match="generated_files"):
        read_generation_metadata(generation_dir)


def test_load_generated_reviews_reads_text_and_metadata(tmp_path):
    generation_dir = tmp_path / "generation"
    write_generation_directory(generation_dir)

    reviews = load_generated_reviews(generation_dir)

    assert len(reviews) == 1
    assert reviews[0]["prompt_id"] == "amor"
    assert reviews[0]["prompt_text"] == "Amor"
    assert reviews[0]["seed"] == 1337
    assert reviews[0]["generated_text"] == "Amor\nche move\n"


def test_markdown_heading_text_removes_newlines():
    assert markdown_heading_text("amor\nprompt") == "amor prompt"


def test_fenced_text_block_wraps_generated_text():
    block = fenced_text_block("Amor\nche move\n")

    assert block.startswith("```text\n")
    assert "Amor\nche move" in block
    assert block.endswith("\n```")


def test_fenced_text_block_removes_nested_backtick_fences():
    block = fenced_text_block("before ``` after")

    assert "``` after" not in block
    assert "''' after" in block


def test_markdown_review_section_contains_review_fields():
    section = markdown_review_section({
        "prompt_id": "amor",
        "prompt_text": "Amor",
        "path": "outputs/amor.txt",
        "seed": 1337,
        "generated_text": "Amor\n",
    })

    assert "## Prompt: amor" in section
    assert "- Sonnet-like structure: TODO" in section
    assert "- Strongest failure mode: TODO" in section
    assert "```text\nAmor\n```" in section


def test_build_qualitative_review_report_contains_instructions_and_sections():
    report = build_qualitative_review_report(
        generation_dir=Path("outputs/generations/run"),
        reviews=[
            {
                "prompt_id": "amor",
                "prompt_text": "Amor",
                "path": "outputs/amor.txt",
                "seed": 1337,
                "generated_text": "Amor\n",
            },
        ],
    )

    assert "# Qualitative Generation Review" in report
    assert "## Review Instructions" in report
    assert "## Prompt: amor" in report
    assert "Keep weak and failed samples" in report


def test_write_qualitative_review_report_writes_markdown(tmp_path):
    generation_dir = tmp_path / "generation"
    output_path = tmp_path / "reports" / "qualitative_review.md"
    write_generation_directory(generation_dir)

    reviews = write_qualitative_review_report(
        generation_dir=generation_dir,
        output_path=output_path,
    )

    report = output_path.read_text(encoding="utf-8")

    assert len(reviews) == 1
    assert output_path.is_file()
    assert "## Prompt: amor" in report
    assert "Amor\nche move" in report
