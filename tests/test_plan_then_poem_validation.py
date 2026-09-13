from pathlib import Path

from sonnet_analysis.minerva_v7_prompt_intervention import build_intervention_prompt
from sonnet_analysis.plan_then_poem_validation import (
    adherence,
    build_control_words,
    build_plan_words,
    echo_detected,
    mismatched_prompt_id,
    output_name,
    plan_instruction,
)


class FakeTokenizer:
    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True
    ) -> str:
        return f"U:{messages[0]['content']}|A:"


def make_lexicon() -> dict:
    keys = [
        "ita",
        "ore",
        "are",
        "ente",
        "ando",
        "ura",
        "ezza",
        "aggio",
        "ore2",
        "are2",
        "ente2",
        "ando2",
        "ura2",
        "ezza2",
        "aggio2",
        "ita2",
    ]
    entries = []
    for index, key in enumerate(keys):
        entries.append(
            {
                "key": key,
                "soft_key": key,
                "count": 100 - index,
                "words": [f"w{key}{n}" for n in range(6)],
            }
        )
    return {"lexicon_version": "sonnet_rhyme_lexicon_v1", "entries": entries}


def test_plan_instruction_numbers_words():
    text = plan_instruction(["vita", "core"])
    assert text.startswith("Termina ogni verso")
    assert "1. vita" in text and "2. core" in text


def test_adherence_counts_word_and_key_matches():
    planned = ["vita", "core"] * 7
    text = "\n".join(f"verso {index} {word}" for index, word in enumerate(planned))
    result = adherence(text, planned)
    assert result["word_matches"] == 14
    assert result["key_matches"] == 14
    assert result["missing"] == 0


def test_adherence_separates_key_matches_from_word_matches():
    planned = ["vita", "core"] * 7
    text = "\n".join(f"verso {index} salita" for index in range(14))
    result = adherence(text, planned)
    assert result["word_matches"] == 0
    assert result["key_matches"] == 7


def test_build_plan_words_uses_distinct_keys():
    plan = build_plan_words(make_lexicon(), seed=11)
    assert len(plan["words"]) == 14
    assert len({row["key"] for row in plan["lines"]}) == len(set(plan["scheme"]))


def test_build_control_words_uses_distinct_keys():
    words = build_control_words(make_lexicon(), seed=11)
    assert len(words) == 14
    assert len(set(words)) == 14


def test_output_name_is_deterministic():
    assert output_name("planned", "p1", 5200) == output_name("planned", "p1", 5200)
    assert output_name("planned", "p1", 5200) != output_name("control", "p1", 5200)


def test_extra_instruction_is_appended_to_the_user_content():
    tokenizer = FakeTokenizer()
    prompt = build_intervention_prompt(
        tokenizer,
        "Nel mezzo del cammin",
        "explicit_no_labels_or_prose",
        extra_instruction="LISTA",
    )
    assert "LISTA" in prompt
    assert "Primo verso: Nel mezzo del cammin" in prompt
    plain = build_intervention_prompt(
        tokenizer, "Nel mezzo del cammin", "explicit_no_labels_or_prose"
    )
    assert "LISTA" not in plain


def test_adherence_reports_per_line_hits():
    planned = ["vita", "core"] * 7
    text = "\n".join(
        "verso numero vita" if index % 2 == 0 else "verso numero cosa"
        for index in range(14)
    )
    result = adherence(text, planned)
    assert result["key_by_line"][0] == 1
    assert result["key_by_line"][1] == 0
    assert result["word_by_line"][0] == 1


def test_echo_detected_flags_numbered_lists():
    assert echo_detected("1. vita\n2. core\n3. amore") is True
    assert echo_detected("Termina ogni verso con la parola indicata") is True
    assert echo_detected("Nel mezzo del cammin di nostra vita") is False


def test_mismatched_prompt_id_cycles():
    prompt_ids = ["a", "b", "c"]
    assert mismatched_prompt_id(prompt_ids, 0) == "b"
    assert mismatched_prompt_id(prompt_ids, 2) == "a"
