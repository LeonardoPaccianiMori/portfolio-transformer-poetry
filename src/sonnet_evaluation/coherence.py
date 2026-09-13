"""Coherence proxies and a calibrated model-judge panel.

The proxies are structural and cheap. The judge panel asks cheap capable
models to rate grammar, continuity, imagery, and meta-text absence on a fixed
rubric, and is calibrated on known-good sonnets against deterministic
corruptions before any produced output is scored.
"""

from __future__ import annotations

import json
import random
import re
import statistics
import subprocess
from collections import Counter
from typing import Any, Callable, Mapping, Sequence

WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÿ']+")
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
JUDGE_KEYS = ("grammar", "continuity", "imagery", "meta_text_absence")
JUDGE_INSTRUCTION = (
    "You are a strict evaluator of Italian sonnets. Rate the text on four "
    "dimensions from 1 (bad) to 5 (excellent): grammar (grammatical Italian), "
    "continuity (semantic flow across lines), imagery (coherent images, not "
    "random word salad), and meta_text_absence (no explanations, labels, "
    "numbered lists, or prose; 5 means none). Reply with ONLY a JSON object "
    "with those four integer keys and a short note string. Text:\n\n"
)


def coherence_proxies(text: str) -> dict[str, float]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    words = WORD_PATTERN.findall(text.lower())
    counts = Counter(words)
    bigrams = [tuple(words[index : index + 2]) for index in range(len(words) - 1)]
    final_words = [
        matches[-1].lower()
        for line in lines
        if (matches := WORD_PATTERN.findall(line))
    ]
    return {
        "line_count": float(len(lines)),
        "type_token_ratio": (len(counts) / len(words)) if words else 0.0,
        "repeated_line_ratio": (
            1 - len(set(lines)) / len(lines) if lines else 0.0
        ),
        "repeated_bigram_ratio": (
            1 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0
        ),
        "mean_line_length": (
            statistics.fmean(len(line) for line in lines) if lines else 0.0
        ),
        "line_length_std": (
            statistics.pstdev(len(line) for line in lines)
            if len(lines) > 1
            else 0.0
        ),
        "final_word_repetition": (
            1 - len(set(final_words)) / len(final_words) if final_words else 0.0
        ),
        "punctuation_ratio": (
            sum(1 for character in text if character in ".,;:!?")
            / max(1, len(text))
        ),
    }


def corrupt_shuffle_lines(text: str, *, seed: int = 7) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    random.Random(seed).shuffle(lines)
    return "\n".join(lines)


def corrupt_swap_words(text: str, *, seed: int = 7) -> str:
    rng = random.Random(seed)
    result = []
    for line in text.splitlines():
        matches = list(WORD_PATTERN.finditer(line))
        if len(matches) >= 2:
            first, last = matches[0], matches[-1]
            line = (
                line[: first.start()]
                + last.group(0)
                + line[first.end() : last.start()]
                + first.group(0)
                + line[last.end() :]
            )
            rng.random()
        result.append(line)
    return "\n".join(result)


def corrupt_truncate(text: str, *, keep: int = 7) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[:keep])


def judge_prompt(text: str) -> str:
    return f"{JUDGE_INSTRUCTION}{text}"


def strip_ansi(text: str) -> str:
    return ANSI_PATTERN.sub("", text)


def parse_judge_output(raw: str) -> dict[str, Any] | None:
    cleaned = strip_ansi(raw)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not all(key in payload for key in JUDGE_KEYS):
        return None
    scores = {}
    for key in JUDGE_KEYS:
        value = payload[key]
        if not isinstance(value, int) or not 1 <= value <= 5:
            return None
        scores[key] = value
    scores["note"] = str(payload.get("note", ""))[:200]
    scores["mean"] = sum(scores[key] for key in JUDGE_KEYS) / len(JUDGE_KEYS)
    return scores


def run_judge(
    model: str, prompt: str, *, timeout_seconds: int = 240
) -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            ["opencode", "run", "-m", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    if completed.returncode != 0:
        return None
    return parse_judge_output(completed.stdout)


def calibrate_judge(
    model: str,
    originals: Sequence[str],
    *,
    timeout_seconds: int = 240,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    original_scores = []
    corruption_scores = []
    rows = []
    for index, text in enumerate(originals):
        original = run_judge(model, judge_prompt(text), timeout_seconds=timeout_seconds)
        corruptions = {
            "shuffle": corrupt_shuffle_lines(text),
            "swap": corrupt_swap_words(text),
            "truncate": corrupt_truncate(text),
        }
        corruption_results = {
            name: run_judge(model, judge_prompt(value), timeout_seconds=timeout_seconds)
            for name, value in corruptions.items()
        }
        if original is not None:
            original_scores.append(original["mean"])
        valid = [value["mean"] for value in corruption_results.values() if value]
        corruption_scores.extend(valid)
        rows.append(
            {
                "index": index,
                "original": original,
                "corruptions": corruption_results,
            }
        )
        if progress:
            progress(f"calibration sample {index + 1}/{len(originals)}")
    separation = (
        statistics.fmean(original_scores) - statistics.fmean(corruption_scores)
        if original_scores and corruption_scores
        else None
    )
    return {
        "judge": model,
        "original_count": len(original_scores),
        "corruption_count": len(corruption_scores),
        "mean_original": (
            statistics.fmean(original_scores) if original_scores else None
        ),
        "mean_corruption": (
            statistics.fmean(corruption_scores) if corruption_scores else None
        ),
        "separation": separation,
        "rows": rows,
    }


def judge_texts(
    model: str,
    texts: Sequence[str],
    *,
    timeout_seconds: int = 240,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any] | None]:
    results = []
    for index, text in enumerate(texts):
        results.append(
            run_judge(model, judge_prompt(text), timeout_seconds=timeout_seconds)
        )
        if progress and (index + 1) % 10 == 0:
            progress(f"{index + 1}/{len(texts)}")
    return results
