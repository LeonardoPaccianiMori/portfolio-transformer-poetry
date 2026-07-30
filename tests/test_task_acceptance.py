import json
from pathlib import Path

import pytest

from sonnet_evaluation.task_acceptance import (
    build_task_format_acceptance_report,
    score_task_format_acceptance_directory,
    write_task_format_acceptance_report,
)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_task_generation_directory(generation_dir: Path) -> None:
    generation_dir.mkdir()
    output_path = generation_dir / "first__seed_1337.txt"
    output_path.write_text("Prima linea\nseconda linea\n", encoding="utf-8")
    write_json(
        generation_dir / "metadata.json",
        {
            "generation_format": "task_format_opening_line_continuation",
            "total_line_target": 2,
            "generated_files": [
                {
                    "prompt_id": "first__seed_1337",
                    "source_prompt_id": "first",
                    "poem_id": "test_poem",
                    "author": "Author",
                    "opening_line": "Prima linea",
                    "prompt_text": "Prima linea",
                    "path": str(output_path),
                    "seed": 1337,
                    "stop_reason": "target_lines",
                    "completed_continuation_lines": 1,
                },
            ],
            "stop_text": "<|endoftext|>",
        },
    )


def test_score_task_format_acceptance_checks_exact_opening_and_line_target(tmp_path):
    generation_dir = tmp_path / "generation"
    write_task_generation_directory(generation_dir)

    rows = score_task_format_acceptance_directory(generation_dir)

    assert len(rows) == 1
    assert rows[0]["opening_line_preserved"]
    assert rows[0]["control_tokens_hidden"]
    assert rows[0]["controlled_sonnet_form"]
    assert rows[0]["automatic_control_pass"]


def test_score_task_format_acceptance_rejects_non_task_metadata(tmp_path):
    generation_dir = tmp_path / "generation"
    write_task_generation_directory(generation_dir)
    metadata_path = generation_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["generation_format"] = "generic"
    write_json(metadata_path, metadata)

    with pytest.raises(ValueError, match="not task-format"):
        score_task_format_acceptance_directory(generation_dir)


def test_task_format_acceptance_report_states_automatic_and_remaining_gates(tmp_path):
    generation_dir = tmp_path / "generation"
    write_task_generation_directory(generation_dir)
    rows = score_task_format_acceptance_directory(generation_dir)

    report = build_task_format_acceptance_report(generation_dir, rows)

    assert "# Task-Format Acceptance Controls" in report
    assert "Controlled prompt/form outputs: **1/1**" in report
    assert "Remaining Acceptance Evidence" in report
    assert "decoder-enforced" in report


def test_write_task_format_acceptance_report_writes_markdown(tmp_path):
    generation_dir = tmp_path / "generation"
    output_path = tmp_path / "reports" / "acceptance.md"
    write_task_generation_directory(generation_dir)

    rows = write_task_format_acceptance_report(generation_dir, output_path)

    assert len(rows) == 1
    assert output_path.is_file()
    assert "first__seed_1337" in output_path.read_text(encoding="utf-8")
