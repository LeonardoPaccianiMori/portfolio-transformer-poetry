"""Automatic controls for the fixed task-format sonnet acceptance set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sonnet_corpus.task_format import (
    SONNET_CONTINUATION_TOKEN,
    SONNET_OPENING_TOKEN,
)
from sonnet_evaluation.metrics import score_generation_directory


TASK_CONTROL_TOKENS = (
    "<|endoftext|>",
    SONNET_OPENING_TOKEN,
    SONNET_CONTINUATION_TOKEN,
)
REQUIRED_CONTROLLED_OUTPUTS = 18


def score_task_format_acceptance_directory(
    generation_dir: Path,
) -> list[dict[str, Any]]:
    """Score prompt preservation and the controlled 14-line form per output."""
    metadata = _read_metadata(generation_dir)
    if metadata.get("generation_format") != "task_format_opening_line_continuation":
        raise ValueError("generation metadata is not task-format generation")

    total_line_target = metadata.get("total_line_target")
    if not isinstance(total_line_target, int) or total_line_target <= 1:
        raise ValueError("task-format generation metadata has an invalid line target")

    metric_rows = score_generation_directory(generation_dir)
    generated_files = metadata["generated_files"]
    if len(metric_rows) != len(generated_files):
        raise AssertionError("generation metrics and metadata rows must align")

    rows = []
    for metric_row, generated_file in zip(metric_rows, generated_files, strict=True):
        text = Path(metric_row["path"]).read_text(encoding="utf-8")
        first_line = text.splitlines()[0] if text.splitlines() else ""
        opening_line_preserved = first_line == generated_file["opening_line"]
        control_tokens_hidden = not any(
            token in text
            for token in TASK_CONTROL_TOKENS
        )
        controlled_sonnet_form = (
            metric_row["non_empty_line_count"] == total_line_target
        )
        rows.append({
            **metric_row,
            "source_prompt_id": generated_file["source_prompt_id"],
            "poem_id": generated_file["poem_id"],
            "author": generated_file.get("author", ""),
            "opening_line_preserved": opening_line_preserved,
            "control_tokens_hidden": control_tokens_hidden,
            "controlled_sonnet_form": controlled_sonnet_form,
            "automatic_control_pass": (
                opening_line_preserved
                and control_tokens_hidden
                and controlled_sonnet_form
            ),
            "stop_reason": generated_file["stop_reason"],
            "completed_continuation_lines": generated_file[
                "completed_continuation_lines"
            ],
        })

    return rows


def build_task_format_acceptance_report(
    generation_dir: Path,
    rows: list[dict[str, Any]],
) -> str:
    """Render the automatic portion of the predeclared acceptance protocol."""
    total_outputs = len(rows)
    automatic_passes = sum(row["automatic_control_pass"] for row in rows)
    headers = [
        "Output",
        "Author",
        "Lines",
        "Prompt Exact",
        "Markers Hidden",
        "14-Line Form",
        "Automatic Pass",
        "Stop",
    ]
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row["prompt_id"],
            row["author"],
            str(row["non_empty_line_count"]),
            _yes_no(row["opening_line_preserved"]),
            _yes_no(row["control_tokens_hidden"]),
            _yes_no(row["controlled_sonnet_form"]),
            _yes_no(row["automatic_control_pass"]),
            row["stop_reason"],
        ]
        table_lines.append("| " + " | ".join(_markdown_cell(value) for value in values) + " |")

    return "\n\n".join([
        "# Task-Format Acceptance Controls",
        f"Generation directory: `{generation_dir}`",
        "## Automatic Result",
        (
            f"- Controlled prompt/form outputs: **{automatic_passes}/{total_outputs}** "
            f"(required: at least {REQUIRED_CONTROLLED_OUTPUTS}/20)."
        ),
        (
            "- Automatic control gate: **pass**."
            if automatic_passes >= REQUIRED_CONTROLLED_OUTPUTS
            else "- Automatic control gate: **fail**."
        ),
        "## Per-Output Controls",
        "\n".join(table_lines),
        "## Remaining Acceptance Evidence",
        "- The qualitative review must assess grammatical Italian, seven-line topic/argument continuity, and severe repetition/collapse.",
        "- The memorization report must show zero high-risk outputs.",
        "- Automatic form control is decoder-enforced and must not be claimed as learned metre or rhyme.",
    ]) + "\n"


def write_task_format_acceptance_report(
    generation_dir: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    """Score one task-format generation directory and write its control report."""
    rows = score_task_format_acceptance_directory(generation_dir)
    report = build_task_format_acceptance_report(
        generation_dir=generation_dir,
        rows=rows,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    return rows


def _read_metadata(generation_dir: Path) -> dict[str, Any]:
    payload = json.loads((generation_dir / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("generation metadata must contain a JSON object")
    if not isinstance(payload.get("generated_files"), list):
        raise ValueError("generation metadata must contain generated_files")
    return payload


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _markdown_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", r"\|")
