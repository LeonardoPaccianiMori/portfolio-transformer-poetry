"""Conservative text cleanup for broader Italian pretraining sources."""

from __future__ import annotations

import re


def normalize_text_boundaries(text: str) -> str:
    """Normalize line endings and blank lines without changing source spelling."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip() + "\n"


def clean_pretraining_text(text: str, *, source_id: str) -> str:
    """Apply source-level cleanup before writing broader pretraining text."""

    if not text.strip():
        raise ValueError(f"cleaned text is empty for {source_id}")

    return normalize_text_boundaries(text)


def validate_cleaned_text(
    text: str,
    *,
    source_id: str,
    min_character_count: int = 200,
) -> None:
    """Reject obviously broken text extraction results."""

    if len(text) < min_character_count:
        raise ValueError(
            f"cleaned text for {source_id} is too short: "
            f"{len(text)} characters; expected at least {min_character_count}"
        )

    lower_text = text.casefold()
    forbidden_markers = [
        "*** start of the project gutenberg ebook",
        "*** end of the project gutenberg ebook",
    ]
    for marker in forbidden_markers:
        if marker in lower_text:
            raise ValueError(
                f"cleaned text for {source_id} still contains wrapper marker: {marker}"
            )
