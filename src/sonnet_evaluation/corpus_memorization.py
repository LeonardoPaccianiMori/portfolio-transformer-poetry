"""Corpus memorization screening for generated sonnet text.

The screen measures verbatim overlap between generated continuations and a
training corpus. The opening line is excluded because generation prompts
provide it. Exact corpus lines and word n-gram shingles are separate signals.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

MIN_LINE_WORDS = 4
DEFAULT_SHINGLE_SIZE = 5


def screen_normalize(text: str) -> str:
    text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    text = unicodedata.normalize("NFC", text).casefold()
    text = re.sub(r"[^a-zàèéìòùç' ]", " ", text)
    return " ".join(text.split())


def build_corpus_line_index(corpus_texts: Iterable[str]) -> set[str]:
    line_index: set[str] = set()
    for text in corpus_texts:
        for line in str(text).splitlines():
            normalized = screen_normalize(line)
            if len(normalized.split()) >= MIN_LINE_WORDS:
                line_index.add(normalized)
    return line_index


def continuation_lines(text: str) -> list[str]:
    raw_lines = [line for line in str(text).splitlines() if line.strip()]
    return [screen_normalize(line) for line in raw_lines[1:]]


def copied_continuation_lines(text: str, line_index: set[str]) -> int:
    """Count continuation lines that appear verbatim in the corpus index."""

    return sum(1 for line in continuation_lines(text) if line in line_index)


def word_shingles(
    tokens: Sequence[str], size: int = DEFAULT_SHINGLE_SIZE
) -> set[tuple[str, ...]]:
    if size <= 0:
        raise ValueError("shingle size must be positive")
    return {
        tuple(tokens[index : index + size])
        for index in range(max(0, len(tokens) - size + 1))
    }


def memorization_screen(
    records: Sequence[Mapping[str, Any]],
    corpus_texts: Iterable[str],
    *,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
) -> dict[str, Any]:
    """Measure verbatim corpus overlap in outputs, excluding each opening."""

    line_index: set[str] = set()
    shingle_index: set[tuple[str, ...]] = set()
    corpus_documents = 0
    for text in corpus_texts:
        corpus_documents += 1
        for line in str(text).splitlines():
            normalized = screen_normalize(line)
            if len(normalized.split()) >= MIN_LINE_WORDS:
                line_index.add(normalized)
        shingle_index.update(
            word_shingles(screen_normalize(str(text)).split(), shingle_size)
        )
    per_condition: dict[str, dict[str, Any]] = {}
    for record in records:
        condition = str(record["system_id"])
        lines = continuation_lines(record["text"])
        exact_lines = sum(1 for line in lines if line in line_index)
        tokens = [token for line in lines for token in line.split()]
        shingles = word_shingles(tokens, shingle_size)
        hit_rate = len(shingles & shingle_index) / len(shingles) if shingles else 0.0
        stats = per_condition.setdefault(
            condition,
            {
                "outputs": 0,
                "outputs_with_exact_line": 0,
                "outputs_with_four_exact_lines": 0,
                "max_exact_lines": 0,
                "outputs_shingle_hit_at_least_half": 0,
                "shingle_hit_sum": 0.0,
            },
        )
        stats["outputs"] += 1
        stats["outputs_with_exact_line"] += int(exact_lines > 0)
        stats["outputs_with_four_exact_lines"] += int(exact_lines >= 4)
        stats["max_exact_lines"] = max(stats["max_exact_lines"], exact_lines)
        stats["outputs_shingle_hit_at_least_half"] += int(hit_rate >= 0.5)
        stats["shingle_hit_sum"] += hit_rate
    for stats in per_condition.values():
        stats["mean_shingle_hit"] = stats["shingle_hit_sum"] / max(
            1, stats["outputs"]
        )
        del stats["shingle_hit_sum"]
    return {
        "shingle_size": shingle_size,
        "corpus_documents": corpus_documents,
        "corpus_lines": len(line_index),
        "corpus_shingles": len(shingle_index),
        "conditions": per_condition,
    }
