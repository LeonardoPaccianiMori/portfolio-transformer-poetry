"""Inspect editorial material in the Project Gutenberg Sidrac source."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


PRIMARY_TEXT_START = "Questo è lo libro lo quale si chiama Sidracco"
PRIMARY_TEXT_END = "_Conpiuto di scrivere a' dì XIIII di febraio, 1382"
INLINE_NOTE_PATTERN = re.compile(r"\(\d{1,4}\)")
NOTE_LINE_PATTERN = re.compile(r"^\(\d{1,4}\)")


@dataclass(frozen=True)
class SidracEditorialAuditReport:
    """Read-only measurements of candidate Sidrac primary-text boundaries."""

    source_path: str
    report_path: str
    source_character_count: int
    source_line_count: int
    primary_text_start_line: int
    primary_text_end_line: int
    front_matter_character_count: int
    candidate_primary_character_count: int
    end_matter_character_count: int
    candidate_inline_note_marker_count: int
    candidate_note_line_count: int
    candidate_note_line_examples: list[dict[str, str | int]]
    primary_text_start_context: str
    primary_text_end_context: str


def audit_sidrac_editorial_content(
    *,
    source_path: Path,
    report_path: Path,
) -> SidracEditorialAuditReport:
    """Write a bounded report that supports a later source-policy decision."""

    text = source_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    start_index = _find_line_index(lines, PRIMARY_TEXT_START)
    end_index = _find_line_index(lines, PRIMARY_TEXT_END)
    if end_index < start_index:
        raise ValueError("Sidrac primary-text end marker occurs before its start marker")

    front_matter = "".join(lines[:start_index])
    candidate_primary_text = "".join(lines[start_index : end_index + 1])
    end_matter = "".join(lines[end_index + 1 :])
    candidate_lines = lines[start_index : end_index + 1]
    note_examples = _note_line_examples(candidate_lines, start_index=start_index)
    report = SidracEditorialAuditReport(
        source_path=_portable_path(source_path),
        report_path=_portable_path(report_path),
        source_character_count=len(text),
        source_line_count=len(lines),
        primary_text_start_line=start_index + 1,
        primary_text_end_line=end_index + 1,
        front_matter_character_count=len(front_matter),
        candidate_primary_character_count=len(candidate_primary_text),
        end_matter_character_count=len(end_matter),
        candidate_inline_note_marker_count=len(INLINE_NOTE_PATTERN.findall(candidate_primary_text)),
        candidate_note_line_count=sum(
            NOTE_LINE_PATTERN.match(line) is not None for line in candidate_lines
        ),
        candidate_note_line_examples=note_examples,
        primary_text_start_context=_context(lines, start_index),
        primary_text_end_context=_context(lines, end_index),
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _find_line_index(lines: list[str], marker: str) -> int:
    for index, line in enumerate(lines):
        if marker in line:
            return index
    raise ValueError(f"Sidrac marker was not found: {marker}")


def _note_line_examples(
    lines: list[str],
    *,
    start_index: int,
    limit: int = 10,
) -> list[dict[str, str | int]]:
    examples = []
    for offset, line in enumerate(lines):
        if NOTE_LINE_PATTERN.match(line) is not None:
            examples.append(
                {
                    "line_number": start_index + offset + 1,
                    "text": line.strip(),
                }
            )
        if len(examples) == limit:
            break
    return examples


def _context(lines: list[str], index: int, radius: int = 2) -> str:
    start = max(index - radius, 0)
    end = min(index + radius + 1, len(lines))
    return " ".join(line.strip() for line in lines[start:end] if line.strip())


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)
