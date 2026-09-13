import subprocess
import sys
from pathlib import Path

import pytest

from sonnet_evaluation.sonnet_prosody import (
    SONNET_STANZA_PATTERN,
    analyse_line,
    analyse_sonnet,
    count_word_syllables,
    rhyme_key,
    soft_rhyme_key,
)

ROOT = Path(__file__).resolve().parents[1]

PIANO_LINES = [
    "Nel mezzo del cammin di nostra vita",
    "Tanto gentile e tanto onesta pare",
    "Amor, ch'a nullo amato amar perdona",
    "Mille fïate, o dolce mia guerrera",
    "Dolce mio caro et precïoso pegno",
]

TRONCO_LINES = [
    "Nel mezzo del cammin di nostra età",
    "e il naufragar m'è dolce in questo mar",
]


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("piano", 2),
        ("poeta", 3),
        ("causa", 2),
        ("fiume", 2),
        ("miei", 1),
        ("più", 1),
        ("zii", 2),
        ("core", 2),
        ("vita", 2),
        ("musica", 3),
        ("città", 2),
        ("fïate", 3),
        ("precïoso", 4),
        ("ſtato", 2),
        ("sí", 1),
        ("piú", 1),
        ("cosí", 2),
        ("m'â", 1),
    ],
)
def test_count_word_syllables_uses_documented_conventions(word, expected):
    assert count_word_syllables(word) == expected


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("vita", "ita"),
        ("salita", "ita"),
        ("fiore", "ore"),
        ("core", "ore"),
        ("cori", "ori"),
        ("città", "a"),
        ("precïoso", "oso"),
        ("piú", "u"),
        ("ſtato", "ato"),
    ],
)
def test_rhyme_key_starts_at_the_stressed_vowel(word, expected):
    assert rhyme_key(word) == expected


def test_soft_rhyme_key_equates_i_and_e():
    assert soft_rhyme_key(rhyme_key("core")) == soft_rhyme_key(rhyme_key("cori"))


def test_soft_rhyme_key_keeps_open_vowels_distinct():
    assert soft_rhyme_key(rhyme_key("core")) != soft_rhyme_key(rhyme_key("cara"))
    assert soft_rhyme_key(rhyme_key("vita")) == soft_rhyme_key(rhyme_key("salita"))


@pytest.mark.parametrize("line", PIANO_LINES)
def test_hendecasyllables_are_piano(line):
    result = analyse_line(line)
    assert result["syllable_count"] == 11
    assert result["verse_type"] == "piano"
    assert result["is_hendecasyllable"] is True
    assert result["uncertain"] is False


@pytest.mark.parametrize("line", TRONCO_LINES)
def test_hendecasyllables_are_tronco(line):
    result = analyse_line(line)
    assert result["syllable_count"] == 10
    assert result["verse_type"] == "tronco"
    assert result["is_hendecasyllable"] is True
    assert result["uncertain"] is False


def test_sdrucciolo_needs_a_stress_lexicon_and_is_uncertain_without_one():
    line = "Nel mezzo del cammin di nostra musica"
    default = analyse_line(line)
    assert default["syllable_count"] == 12
    assert default["verse_type"] is None
    assert default["is_hendecasyllable"] is None
    assert default["uncertain"] is True
    assert "sdrucciolo_possible" in default["uncertainty_reasons"]

    resolved = analyse_line(line, stress_overrides={"musica": 3})
    assert resolved["syllable_count"] == 12
    assert resolved["verse_type"] == "sdrucciolo"
    assert resolved["is_hendecasyllable"] is True
    assert resolved["uncertain"] is False


def test_possible_hiatus_word_marks_line_uncertain():
    result = analyse_line("che la diritta via era smarrita")
    assert result["syllable_count"] == 10
    assert result["is_hendecasyllable"] is None
    assert result["uncertain"] is True
    assert "hiatus_possible" in result["uncertainty_reasons"]


def test_dialefe_candidate_boundary_is_reported():
    result = analyse_line("e già il naufragar m'è dolce in questo mar")
    assert result["dialefe_candidate_boundaries"] == (1,)


def test_invalid_stress_override_is_rejected():
    with pytest.raises(ValueError, match="stress override"):
        analyse_line(
            "Nel mezzo del cammin di nostra vita",
            stress_overrides={"vita": 0},
        )


def sonnet_text() -> str:
    line = "Nel mezzo del cammin di nostra vita"
    return "\n".join(
        [line] * 4 + [""] + [line] * 4 + [""] + [line] * 3 + [""] + [line] * 3
    )


def test_sonnet_structure_and_stanza_pattern():
    result = analyse_sonnet(sonnet_text())
    assert result["line_count"] == 14
    assert result["structure_ok"] is True
    assert result["stanza_pattern"] == list(SONNET_STANZA_PATTERN)
    assert result["stanza_pattern_ok"] is True
    assert result["metre_ok"] is True
    assert result["hard_gate_pass"] is True


def test_single_group_poem_has_unknown_stanza_pattern():
    poem = "\n".join(["Nel mezzo del cammin di nostra vita"] * 14)
    result = analyse_sonnet(poem)
    assert result["structure_ok"] is True
    assert result["stanza_pattern_ok"] is None


def test_scheme_comparison_uses_exact_rhyme_keys():
    text = "dolce vita\ngentil fiore\ndolce core\nalta salita"
    result = analyse_sonnet(text, expected_scheme="ABBA")
    assert result["rhyme_scheme"] == "ABBA"
    assert result["scheme_ok"] is True
    assert result["perfect_rhyme_pairs"] == 2
    assert result["soft_rhyme_pairs"] == 0
    assert result["rhyme_score"] == 1.0


def test_near_rhyme_lowers_the_soft_score():
    text = "dolce vita\nalta salita\ngentil fiore\nsospiri e cori"
    result = analyse_sonnet(text, expected_scheme="AABC")
    assert result["scheme_ok"] is True
    assert result["perfect_rhyme_pairs"] == 1
    assert result["soft_rhyme_pairs"] == 1
    assert result["rhyme_score"] == 0.5


def test_wrong_expected_scheme_fails_the_gate():
    text = "dolce vita\ngentil fiore\ndolce core\nalta salita"
    result = analyse_sonnet(text, expected_scheme="ABAB")
    assert result["scheme_ok"] is False
    assert result["hard_gate_pass"] is False


def test_cli_prints_a_summary_and_writes_json(tmp_path):
    output = tmp_path / "poem.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_sonnet_prosody.py"),
            "--text",
            "dolce vita\ngentil fiore\ndolce core\nalta salita",
            "--scheme",
            "ABBA",
            "--json",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "scheme_ok: True" in completed.stdout
    assert output.is_file()
