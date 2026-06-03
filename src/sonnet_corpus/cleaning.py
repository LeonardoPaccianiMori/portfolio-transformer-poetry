"""Text cleanup helpers for Wikisource poem pages."""

from __future__ import annotations

import re


_EDITORIAL_BRACKET_RE = re.compile(r"\[([A-Za-zÀ-ÖØ-öø-ÿ])\]")
_LINE_MARKER_RE = re.compile(r"^\s*\d{1,3}\s*$")


def remove_editorial_brackets(text: str) -> str:
    """Remove brackets around editorial single-letter expansions.

    Example:
        ``ben[e]`` becomes ``bene``.
    """

    return _EDITORIAL_BRACKET_RE.sub(r"\1", text)


def normalize_apostrophes(text: str) -> str:
    """Normalize apostrophes used by Wikisource titles/content."""

    return text.replace("’", "'")


def clean_poem_text(text: str) -> str:
    """Clean extracted poem text while preserving poetic line breaks."""

    text = normalize_apostrophes(text)
    text = remove_editorial_brackets(text)

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            continue
        if _LINE_MARKER_RE.match(line):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip() + "\n"


def count_poem_lines(text: str) -> int:
    """Count non-empty poem lines."""

    return sum(1 for line in text.splitlines() if line.strip())
