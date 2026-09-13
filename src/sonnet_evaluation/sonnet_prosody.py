"""Rule-based Italian prosody checker for sonnet evaluation.

The checker measures sonnet structure, hendecasyllable metre, and rhyme.  It
follows one documented set of scansion conventions and marks ambiguous cases as
``uncertain`` instead of forcing a pass or fail.  Accuracy and uncertainty
rates must be measured against reviewed ground truth before the checker is used
as an evaluation metric, selector, label source, or reward.

Scansion conventions:

- An unaccented ``i`` or ``u`` next to another vowel is a glide and joins the
  same syllable ("piano", "causa", "miei").
- A repeated high vowel stays two nuclei ("zii").
- An accented vowel is a nucleus ("più", "città").
- Apostrophes join the elided article or preposition to the next word
  ("l'amor" is "lamor").
- A vowel-final word merges with a vowel-initial next word (sinalefe).  A
  stressed monosyllable or an accented final vowel can take a dialefe instead;
  those boundaries are reported as candidates and can make a line uncertain.
- The final stress comes from a written accent when present, otherwise from the
  default paroxytone convention.  A stress lexicon can set proparoxytone
  (sdrucciolo) or other forms.  A 12-syllable line with an assumed paroxytone
  stress is reported as possibly sdrucciolo, not as a failure.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

DIACRITIC_VOWELS = "àáâäèéêëìíîïòóôöùúûü"
STRESS_ACCENTS = "àáâèéêìíîòóôùúû"
VOWELS = "aeiou" + DIACRITIC_VOWELS
HIGH_VOWELS = "iu"
WORD_NORMALIZATION = str.maketrans({"ſ": "s"})
ACCENT_TRANSLATION = str.maketrans(
    {
        "à": "a", "á": "a", "â": "a", "ä": "a",
        "è": "e", "é": "e", "ê": "e", "ë": "e",
        "ì": "i", "í": "i", "î": "i", "ï": "i",
        "ò": "o", "ó": "o", "ô": "o", "ö": "o",
        "ù": "u", "ú": "u", "û": "u", "ü": "u",
    }
)
SOFT_RHYME_TRANSLATION = str.maketrans({"i": "e", "u": "o"})

SONNET_LINE_COUNT = 14
SONNET_STANZA_PATTERN = (4, 4, 3, 3)

STRESSED_MONOSYLLABLES = frozenset(
    {
        "è", "é", "ho", "hai", "ha", "dà", "fa", "sta", "va", "so", "sto",
        "do", "dò", "può", "più", "già", "giù", "ciò", "sé", "né", "me", "te",
        "tu", "no", "sì", "là", "lì", "qua", "qui", "tre", "re", "fé", "chi",
        "cui", "lui", "lei", "noi", "voi", "sù", "sú", "dì", "dí", "piè",
    }
)

WORD_PATTERN = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿſ]+(?:['’ʼ][A-Za-zÀ-ÖØ-öø-ÿſ]+)*['’ʼ]?",
    re.UNICODE,
)
VOWEL_RUN_PATTERN = re.compile(f"[{VOWELS}]+")


def count_word_syllables(word: str) -> int:
    """Count poetic syllables in one word with the documented conventions."""

    if not word:
        return 0
    normalized = _normalize_word(word)
    return sum(
        _vowel_run_nuclei(run)
        for run in VOWEL_RUN_PATTERN.findall(normalized)
    )


def rhyme_key(word: str) -> str | None:
    """Return the phonetic rhyme key from the stressed vowel to the word end."""

    normalized = _normalize_word(word)
    if not normalized:
        return None
    start = _stressed_vowel_index(normalized)
    if start is None:
        return None
    return normalized[start:].translate(ACCENT_TRANSLATION)


def soft_rhyme_key(key: str | None) -> str | None:
    """Return a key with the Sicilian vowel equivalences i/e and u/o."""

    if not key:
        return None
    return key.translate(SOFT_RHYME_TRANSLATION)


def analyse_line(
    line: str, *, stress_overrides: Mapping[str, int] | None = None
) -> dict[str, Any]:
    """Analyse one verse line for syllable count, stress, and rhyme."""

    overrides = dict(stress_overrides or {})
    for word, value in overrides.items():
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"invalid stress override for {word!r}: {value!r}")

    tokens = WORD_PATTERN.findall(line)
    normalized = [_normalize_word(token) for token in tokens]
    syllable_counts = [count_word_syllables(word) for word in normalized]
    raw_count = sum(syllable_counts)

    boundary_merges = 0
    vowel_boundaries = 0
    dialefe_candidates: list[int] = []
    for index in range(len(normalized) - 1):
        left = normalized[index]
        right = normalized[index + 1]
        if left and left[-1] in VOWELS and _initial_vowel(right) is not None:
            boundary_merges += 1
            vowel_boundaries += 1
            if _dialefe_candidate(left):
                dialefe_candidates.append(index)
    syllable_count = raw_count - boundary_merges

    hiatus_sites = 0
    synaeresis_sites = 0
    for word, word_count in zip(normalized, syllable_counts):
        if word_count:
            merged, splittable = _within_word_ambiguity(word)
            hiatus_sites += merged
            synaeresis_sites += splittable

    final_index = next(
        (index for index in range(len(normalized) - 1, -1, -1)
         if syllable_counts[index] > 0),
        None,
    )
    if final_index is None:
        return {
            "text": line,
            "syllable_count": syllable_count,
            "stress_from_end": None,
            "stress_source": "missing_word",
            "stress_position": None,
            "verse_type": None,
            "is_hendecasyllable": None,
            "uncertain": True,
            "uncertainty_reasons": ["no_vowel_final_word"],
            "dialefe_candidate_boundaries": tuple(dialefe_candidates),
            "final_word": None,
            "rhyme_key": None,
            "soft_rhyme_key": None,
        }

    final_word = normalized[final_index]
    final_word_syllables = syllable_counts[final_index]
    stress_from_end, stress_source = _final_stress(
        final_word, final_word_syllables, overrides
    )
    stress_position = syllable_count - stress_from_end + 1
    verse_type = _verse_type(syllable_count, stress_position)

    uncertain = False
    reasons: list[str] = []
    if verse_type is None:
        reachable_min = syllable_count - synaeresis_sites
        reachable_max = syllable_count + vowel_boundaries + hiatus_sites
        target_counts = {9 + stress_from_end}
        sdrucciolo_possible = stress_from_end == 2 and final_word_syllables >= 3
        if sdrucciolo_possible:
            target_counts.add(12)
        reachable_targets = [
            target
            for target in sorted(target_counts)
            if reachable_min <= target <= reachable_max
        ]
        if reachable_targets:
            uncertain = True
            if any(target > syllable_count for target in reachable_targets):
                if dialefe_candidates:
                    reasons.append("dialefe_possible")
                if hiatus_sites:
                    reasons.append("hiatus_possible")
            if any(target < syllable_count for target in reachable_targets):
                reasons.append("synaeresis_possible")
            if sdrucciolo_possible and 12 in reachable_targets:
                reasons.append("sdrucciolo_possible")

    key = rhyme_key(final_word)
    return {
        "text": line,
        "syllable_count": syllable_count,
        "stress_from_end": stress_from_end,
        "stress_source": stress_source,
        "stress_position": stress_position,
        "verse_type": verse_type,
        "is_hendecasyllable": True if verse_type else (None if uncertain else False),
        "uncertain": uncertain,
        "uncertainty_reasons": reasons,
        "dialefe_candidate_boundaries": tuple(dialefe_candidates),
        "final_word": final_word,
        "rhyme_key": key,
        "soft_rhyme_key": soft_rhyme_key(key),
    }


def analyse_sonnet(
    text: str,
    *,
    expected_scheme: str | None = None,
    stress_overrides: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Analyse a sonnet for structure, metre, and rhyme."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line_results = [
        analyse_line(line, stress_overrides=stress_overrides) for line in lines
    ]
    stanza_pattern = non_empty_stanza_pattern(text)
    structure_ok = len(lines) == SONNET_LINE_COUNT
    stanza_pattern_ok: bool | None = (
        stanza_pattern == SONNET_STANZA_PATTERN
        if len(stanza_pattern) > 1
        else None
    )
    valid_metre = sum(
        1 for result in line_results
        if result["is_hendecasyllable"] is True
    )
    uncertain_metre = sum(
        1 for result in line_results
        if result["is_hendecasyllable"] is None
    )
    metre_ok = len(lines) == SONNET_LINE_COUNT and valid_metre == SONNET_LINE_COUNT

    keys = [result["rhyme_key"] for result in line_results]
    scheme = _rhyme_scheme(keys)
    scheme_ok: bool | None = None
    normalized_expected: str | None = None
    if expected_scheme:
        normalized_expected = "".join(expected_scheme.split()).upper()
        scheme_ok = scheme == normalized_expected

    perfect_pairs, soft_pairs = _rhyme_pair_counts(line_results)
    repeated_pairs = perfect_pairs + soft_pairs
    rhyme_score = perfect_pairs / repeated_pairs if repeated_pairs else 0.0
    hard_gate_pass = (
        structure_ok
        and stanza_pattern_ok is not False
        and metre_ok
        and scheme_ok is not False
    )
    return {
        "line_count": len(lines),
        "structure_ok": structure_ok,
        "stanza_pattern": list(stanza_pattern),
        "stanza_pattern_ok": stanza_pattern_ok,
        "metre_valid_lines": valid_metre,
        "metre_uncertain_lines": uncertain_metre,
        "metre_failed_lines": len(lines) - valid_metre - uncertain_metre,
        "metre_ok": metre_ok,
        "metre_uncertain": uncertain_metre > 0,
        "rhyme_scheme": scheme,
        "expected_scheme": normalized_expected,
        "scheme_ok": scheme_ok,
        "perfect_rhyme_pairs": perfect_pairs,
        "soft_rhyme_pairs": soft_pairs,
        "rhyme_score": rhyme_score,
        "hard_gate_pass": hard_gate_pass,
        "lines": line_results,
        "checker_is_not_a_quality_judgment": True,
    }


def non_empty_stanza_pattern(text: str) -> tuple[int, ...]:
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


def _normalize_word(word: str) -> str:
    normalized = word.translate(WORD_NORMALIZATION).lower()
    return normalized.translate({ord("'"): None, ord("’"): None, ord("ʼ"): None})


def _initial_vowel(word: str) -> str | None:
    """Return the first vowel, treating a silent leading h as transparent."""

    trimmed = word[1:] if word.startswith("h") else word
    if trimmed and trimmed[0] in VOWELS:
        return trimmed[0]
    return None


def _vowel_class(char: str) -> str:
    if char in "aeo" or char in "àáâäèéêëòóôö":
        return "A"
    if char in "iu":
        return "H"
    if char in "ìíîïùúûü":
        return "X"
    return "?"


def _vowel_run_nuclei(run: str) -> int:
    classes = [_vowel_class(char) for char in run]
    count = 0
    index = 0
    while index < len(run):
        char_class = classes[index]
        if char_class in "AX":
            count += 1
            index += 1
            if index < len(run) and classes[index] == "H":
                index += 1
        elif char_class == "H":
            count += 1
            index += 1
            if index < len(run):
                if classes[index] in "AX":
                    index += 1
                    if index < len(run) and classes[index] == "H":
                        index += 1
                elif classes[index] == "H" and run[index] != run[index - 1]:
                    index += 1
        else:
            index += 1
    return count


def _stressed_vowel_index(word: str) -> int | None:
    for index, char in enumerate(word):
        if char in STRESS_ACCENTS:
            return index
    vowel_positions = [
        index for index, char in enumerate(word) if char in VOWELS
    ]
    if not vowel_positions:
        return None
    if len(vowel_positions) == 1:
        return vowel_positions[0]
    return vowel_positions[-2]


def _final_stress(
    word: str, syllable_count: int, overrides: Mapping[str, int]
) -> tuple[int, str]:
    if word in overrides:
        return int(overrides[word]), "stress_lexicon"
    for index, char in enumerate(word):
        if char in STRESS_ACCENTS:
            tail = word[index + 1:]
            after = sum(
                _vowel_run_nuclei(run)
                for run in VOWEL_RUN_PATTERN.findall(tail)
            )
            return after + 1, "written_accent"
    if word and word[-1] not in VOWELS:
        return 1, "final_consonant"
    if syllable_count <= 1:
        return 1, "monosyllable"
    return 2, "default_paroxytone"


def _dialefe_candidate(left_word: str) -> bool:
    if left_word in STRESSED_MONOSYLLABLES:
        return True
    return bool(left_word) and left_word[-1] in STRESS_ACCENTS


def _within_word_ambiguity(word: str) -> tuple[int, int]:
    """Return (merged_pairs, splittable_pairs) for one normalized word."""

    merged = 0
    splittable = 0
    for run in VOWEL_RUN_PATTERN.findall(word):
        classes = [_vowel_class(char) for char in run]
        merged += len(run) - _vowel_run_nuclei(run)
        for left, right in zip(classes, classes[1:]):
            if left == "A" and right == "A":
                splittable += 1
    return merged, splittable


def _verse_type(syllable_count: int, stress_position: int | None) -> str | None:
    if stress_position != 10:
        return None
    if syllable_count == 11:
        return "piano"
    if syllable_count == 10:
        return "tronco"
    if syllable_count == 12:
        return "sdrucciolo"
    return None


def _rhyme_scheme(keys: list[str | None]) -> str:
    mapping: dict[str, str] = {}
    letters: list[str] = []
    for key in keys:
        if key is None:
            letters.append("?")
            continue
        if key not in mapping:
            mapping[key] = chr(ord("A") + len(mapping))
        letters.append(mapping[key])
    return "".join(letters)


def _rhyme_pair_counts(
    line_results: list[dict[str, Any]],
) -> tuple[int, int]:
    exact_groups: dict[str, int] = {}
    soft_groups: dict[str, int] = {}
    for result in line_results:
        exact = result["rhyme_key"]
        soft = result["soft_rhyme_key"]
        if exact is not None:
            exact_groups[exact] = exact_groups.get(exact, 0) + 1
        if soft is not None:
            soft_groups[soft] = soft_groups.get(soft, 0) + 1

    def pairs(groups: dict[str, int]) -> int:
        return sum(size * (size - 1) // 2 for size in groups.values())

    perfect = pairs(exact_groups)
    soft = pairs(soft_groups) - perfect
    return perfect, soft
