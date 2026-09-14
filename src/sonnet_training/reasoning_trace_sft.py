"""Reasoning-trace SFT for autonomous rhyme planning.

The model first writes a rhyme plan (scheme plus one ending word for each of
lines 2-14) and then the sonnet. Training targets pair corpus sonnets with a
trace derived from their own rhyme structure, so the plan is always valid and
comes from real poetry.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from sonnet_analysis.minerva_v7_prompt_intervention import build_intervention_prompt
from sonnet_analysis.plan_then_poem_validation import PROMPT_ARM
from sonnet_evaluation.rhyme_lexicon import line_final_word
from sonnet_evaluation.sonnet_prosody import rhyme_key, soft_rhyme_key
from sonnet_evaluation.sonnet_prosody_sealed import (
    QUATRAIN_PATTERNS,
    TERCET_PATTERNS,
)
from sonnet_training.self_play_rft import generate_autonomous_validation

TRACE_VERSION = "reasoning_trace_sft_v1"
TRACE_ARMS = ("trace_baseline", "trace_sft")

TRACE_INSTRUCTION = (
    "Prima scrivi il piano di rime in questo formato esatto:\n"
    "Schema: <quattordici lettere maiuscole, per esempio ABBAABBACDECDE>\n"
    "Rime:\n"
    "2. <parola>\n"
    "3. <parola>\n"
    "... e così fino a 14\n"
    "Poi scrivi esclusivamente i quattordici versi poetici, senza titolo, "
    "introduzione, spiegazione, commento, numeri o prosa."
)
SCHEMA_PATTERN = re.compile(r"^\s*Schema:\s*([A-Za-z]{14})\s*$")
RIME_PATTERN = re.compile(r"^\s*Rime:\s*$", re.IGNORECASE)
ENTRY_PATTERN = re.compile(r"^\s*(\d{1,2})[.)]\s*(\S.*?)\s*$")


def trace_prompt(tokenizer: Any, opening_line: str) -> str:
    """Render the plan-then-poem instruction with the exact opening prefill."""

    return build_intervention_prompt(
        tokenizer, opening_line, PROMPT_ARM, extra_instruction=TRACE_INSTRUCTION
    )


def _relabel(scheme: str) -> str | None:
    cleaned = "".join(str(scheme).split()).upper()
    if not cleaned or not cleaned.isalpha():
        return None
    mapping: dict[str, str] = {}
    output = []
    for char in cleaned:
        if char not in mapping:
            if len(mapping) >= 26:
                return None
            mapping[char] = chr(ord("A") + len(mapping))
        output.append(mapping[char])
    return "".join(output)


def canonical_scheme(scheme: str) -> str | None:
    """Relabel a scheme so keys follow first-occurrence order, or None."""

    cleaned = "".join(str(scheme).split()).upper()
    if len(cleaned) != 14:
        return None
    return _relabel(cleaned)


def derive_trace(scheme: str, words: Sequence[str]) -> str:
    canonical = canonical_scheme(scheme)
    if canonical is None:
        raise ValueError("invalid rhyme scheme")
    if canonical != "".join(str(scheme).split()).upper():
        raise ValueError("rhyme scheme must already be canonical")
    if len(words) != 14:
        raise ValueError("expected 14 ending words")
    lines = [f"Schema: {canonical}", "Rime:"]
    lines.extend(f"{index}. {str(words[index - 1])}" for index in range(2, 15))
    return "\n".join(lines)


def parse_trace(text: str) -> dict[str, Any] | None:
    """Split a generation into a rhyme plan and the 14 poem lines."""

    raw_lines = str(text).splitlines()
    schema = None
    rime_index = None
    for index, line in enumerate(raw_lines):
        match = SCHEMA_PATTERN.match(line)
        if match and schema is None:
            schema = match.group(1)
        if RIME_PATTERN.match(line) and rime_index is None:
            rime_index = index
    if schema is None or rime_index is None:
        return None
    entries: dict[int, str] = {}
    last_trace_index = rime_index
    for index in range(rime_index + 1, len(raw_lines)):
        line = raw_lines[index]
        if not line.strip():
            continue
        match = ENTRY_PATTERN.match(line)
        if not match:
            break
        number = int(match.group(1))
        if number < 2 or number > 14 or number in entries:
            break
        entries[number] = match.group(2).strip()
        last_trace_index = index
    if set(entries) != set(range(2, 15)):
        return None
    poem_lines = []
    for line in raw_lines[last_trace_index + 1 :]:
        stripped = line.strip()
        if stripped:
            poem_lines.append(stripped)
        if len(poem_lines) == 14:
            break
    if len(poem_lines) < 14:
        return None
    return {
        "raw_schema": schema.upper(),
        "schema": canonical_scheme(schema) or schema.upper(),
        "entries": entries,
        "poem_lines": poem_lines,
        "poem_text": "\n".join(poem_lines),
    }


def validate_trace(
    parsed: Mapping[str, Any],
    opening_line: str,
    *,
    allow_soft: bool = True,
) -> dict[str, Any]:
    """Check a parsed trace for canonical form, valid scheme, and true rhymes."""

    raw_schema = str(parsed.get("raw_schema", ""))
    canonical = canonical_scheme(raw_schema)
    opening_word = line_final_word(opening_line)
    schema_valid = False
    rhyme_consistent = False
    if canonical is not None:
        quatrain = _relabel(canonical[:8]) or ""
        tercet = _relabel(canonical[8:]) or ""
        schema_valid = quatrain in QUATRAIN_PATTERNS and tercet in TERCET_PATTERNS
        groups: dict[str, list[str]] = {}
        for number, entry in dict(parsed.get("entries", {})).items():
            word = line_final_word(str(entry))
            if word:
                groups.setdefault(canonical[int(number) - 1], []).append(word)
        if opening_word:
            groups.setdefault(canonical[0], []).append(opening_word)
        hard = all(
            len(words) >= 2
            and all(rhyme_key(word) is not None for word in words)
            and len({rhyme_key(word) for word in words}) == 1
            for words in groups.values()
        )
        soft = False
        if allow_soft and not hard:
            soft = all(
                len(words) >= 2
                and all(soft_rhyme_key(rhyme_key(word)) is not None for word in words)
                and len({soft_rhyme_key(rhyme_key(word)) for word in words}) == 1
                for words in groups.values()
            )
        rhyme_consistent = bool(hard or soft)
    return {
        "well_formed": canonical is not None,
        "canonical_scheme": canonical,
        "schema_valid": schema_valid,
        "rhyme_consistent": rhyme_consistent,
        "valid": bool(canonical is not None and schema_valid and rhyme_consistent),
        "opening_word": opening_word,
    }


def build_trace_examples(
    cards: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_sequence_tokens: int = 1536,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    examples: list[dict[str, Any]] = []
    skipped = {"malformed": 0, "too_long": 0}
    for card in cards:
        lines = [
            line.strip() for line in str(card["poem_text"]).splitlines() if line.strip()
        ]
        if len(lines) != 14:
            skipped["malformed"] += 1
            continue
        target = (
            str(card["trace_text"]).strip("\n")
            + "\n\n"
            + str(card["poem_text"]).strip("\n")
            + "\n"
        )
        prompt = trace_prompt(tokenizer, str(card["opening_line"]))
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        target_ids = list(target_ids) + [int(tokenizer.eos_token_id)]
        if len(prompt_ids) + len(target_ids) > max_sequence_tokens:
            skipped["too_long"] += 1
            continue
        examples.append(
            {
                "unit_id": str(card["unit_id"]),
                "prompt_ids": list(prompt_ids),
                "target_ids": target_ids,
                "opening_line": str(card["opening_line"]),
                "scheme": str(card["scheme"]),
            }
        )
    if not examples:
        raise ValueError("no reasoning-trace examples could be built")
    return examples, skipped


def generate_trace_validation(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Mapping[str, Any]],
    arm: str,
    seeds: Sequence[int],
    recipe: Mapping[str, Any],
    output_dir: Any,
    device: Any,
    batch_size: int,
    model_identity: str | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    return generate_autonomous_validation(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        arm=arm,
        seeds=seeds,
        recipe=recipe,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        model_identity=model_identity,
        progress=progress,
        prompt_builder=trace_prompt,
        version=TRACE_VERSION,
        analysis_role="reasoning_trace_sft_validation",
        arms=TRACE_ARMS,
    )
