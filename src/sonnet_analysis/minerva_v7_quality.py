"""Conservative surface diagnostics for generated Italian sonnets.

These checks identify observable decoder and response-format failures.  They do
not attempt to score meter, rhyme, grammar, argument, or poetic quality.
"""

from __future__ import annotations

import re
from typing import Any


META_TEXT_PATTERNS = {
    "numbered_verse_label": re.compile(
        r"(?im)^\s*(?:primo|secondo|terzo|quarto|quinto|sesto|settimo|ottavo|"
        r"nono|decimo|undicesimo|dodicesimo|tredicesimo|quattordicesimo)\s+"
        r"verso\s*:"
    ),
    "generic_sonnet_label": re.compile(
        r"(?im)^\s*sonetto(?:\s+classico)?(?:\s+in\s+italiano)?\s*[.:]?\s*$"
    ),
    "sonnet_explanation": re.compile(
        r"(?im)^\s*(?:in\s+questo\s+sonetto|il\s+sonetto\s+(?:proposto|"
        r"presentato|è|si\s+compone|si\s+descrive)|questo\s+sonetto)\b"
    ),
    "commentary_label": re.compile(
        r"(?im)^\s*(?:spiegazione|commento|analisi|nota)\s*:"
    ),
}

TERMINAL_PUNCTUATION = frozenset(".!?…")
TRAILING_CLOSERS = frozenset("'\"’”»)]}")
DEFAULT_LONG_LINE_CHARACTERS = 120
DEFAULT_HIGH_REPETITION_RATIO = 0.35


def generated_sonnet_surface_diagnostics(
    text: str,
    *,
    non_empty_line_count: int,
    repetition_ratio: float,
    long_line_characters: int = DEFAULT_LONG_LINE_CHARACTERS,
    high_repetition_ratio: float = DEFAULT_HIGH_REPETITION_RATIO,
) -> dict[str, Any]:
    """Return literal response-format diagnostics without judging poetry."""

    if long_line_characters <= 0:
        raise ValueError("long_line_characters must be positive")
    if not 0 <= high_repetition_ratio <= 1:
        raise ValueError("high_repetition_ratio must be between zero and one")

    markers = [
        marker
        for marker, pattern in META_TEXT_PATTERNS.items()
        if pattern.search(text)
    ]
    line_lengths = [len(line) for line in text.splitlines() if line.strip()]
    maximum_line_characters = max(line_lengths, default=0)
    terminal_punctuation = ends_with_terminal_punctuation(text)
    meta_text_free = not markers
    no_very_long_line = maximum_line_characters < long_line_characters
    below_high_repetition_threshold = repetition_ratio < high_repetition_ratio
    exact_fourteen_lines = non_empty_line_count == 14
    stanza_pattern = non_empty_stanza_line_pattern(text)
    return {
        "meta_text_detected": not meta_text_free,
        "meta_text_markers": markers,
        "meta_text_free": meta_text_free,
        "ends_with_terminal_punctuation": terminal_punctuation,
        "maximum_line_characters": maximum_line_characters,
        "no_line_at_or_above_120_characters": no_very_long_line,
        "below_035_repetition_ratio": below_high_repetition_threshold,
        "non_empty_stanza_line_pattern": list(stanza_pattern),
        "explicit_4433_stanza_pattern": stanza_pattern == (4, 4, 3, 3),
        "surface_screen_pass": (
            exact_fourteen_lines
            and meta_text_free
            and terminal_punctuation
            and no_very_long_line
            and below_high_repetition_threshold
        ),
        "surface_screen_is_not_poetic_quality_judgment": True,
    }


def ends_with_terminal_punctuation(text: str) -> bool:
    """Test literal final punctuation after removing quotes/brackets only."""

    stripped = text.rstrip()
    while stripped and stripped[-1] in TRAILING_CLOSERS:
        stripped = stripped[:-1].rstrip()
    return bool(stripped) and stripped[-1] in TERMINAL_PUNCTUATION


def non_empty_stanza_line_pattern(text: str) -> tuple[int, ...]:
    """Count non-empty lines in blank-line-delimited stanza groups."""

    groups: list[int] = []
    current = 0
    for line in text.splitlines():
        if line.strip():
            current += 1
        elif current:
            groups.append(current)
            current = 0
    if current:
        groups.append(current)
    return tuple(groups)
