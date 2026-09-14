from sonnet_training.reasoning_trace_sft import (
    build_plan_examples,
    build_trace_examples,
    canonical_scheme,
    plan_prompt,
    planned_words_from_trace,
    derive_trace,
    parse_trace,
    validate_trace,
)


class FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True
    ) -> str:
        rendered = "|".join(message["content"] for message in messages)
        return f"{rendered}|assistant:"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1] * len(str(text).split())}


def build_trace(scheme: str, entries: dict[int, str]) -> str:
    lines = [f"Schema: {scheme}", "Rime:"]
    lines.extend(f"{index}. {entries[index]}" for index in range(2, 15))
    return "\n".join(lines)


def valid_entries() -> dict[int, str]:
    return {
        2: "salita",
        3: "ferita",
        4: "vita",
        5: "salita",
        6: "ferita",
        7: "salita",
        8: "vita",
        9: "sole",
        10: "fiore",
        11: "stella",
        12: "parole",
        13: "amore",
        14: "bella",
    }


def test_canonical_scheme_relabels_and_rejects_malformed():
    assert canonical_scheme("ABBAABBACDECDE") == "ABBAABBACDECDE"
    assert canonical_scheme("XYYXXYYXZWVWVW") == "ABBAABBACDEDED"
    assert canonical_scheme("ABC") is None
    assert canonical_scheme("ABC123") is None


def test_derive_trace_formats_thirteen_entries():
    words = ["vita"] * 14
    trace = derive_trace("ABBAABBACDECDE", words)
    lines = trace.splitlines()
    assert lines[0] == "Schema: ABBAABBACDECDE"
    assert lines[1] == "Rime:"
    assert len(lines) == 15
    assert lines[2] == "2. vita"
    assert lines[-1] == "14. vita"


def test_parse_trace_round_trip_and_missing_trace():
    trace = build_trace("ABBAABBACDECDE", valid_entries())
    poem = "\n".join(f"verso numero {index} qui" for index in range(14))
    parsed = parse_trace(trace + "\n\n" + poem)
    assert parsed is not None
    assert parsed["schema"] == "ABBAABBACDECDE"
    assert len(parsed["entries"]) == 13
    assert parsed["entries"][9] == "sole"
    assert len(parsed["poem_lines"]) == 14
    assert parse_trace("solo testo senza piano") is None
    assert parse_trace("Schema: ABBAABBACDECDE\nRime:\n2. sole\n") is None


def test_validate_trace_accepts_a_true_rhyme_plan():
    trace = build_trace("ABBAABBACDECDE", valid_entries())
    opening = "Nel mezzo del cammin di nostra vita"
    parsed = parse_trace(
        trace + "\n\n" + "\n".join("verso numero qui ora" for _ in range(14))
    )
    result = validate_trace(parsed, opening)
    assert result["well_formed"] is True
    assert result["schema_valid"] is True
    assert result["rhyme_consistent"] is True
    assert result["valid"] is True


def test_validate_trace_rejects_broken_rhyme_and_bad_scheme():
    entries = valid_entries()
    entries[9] = "cane"
    parsed = parse_trace(
        build_trace("ABBAABBACDECDE", entries)
        + "\n\n"
        + "\n".join("verso numero qui ora" for _ in range(14))
    )
    result = validate_trace(parsed, "Nel mezzo del cammin di nostra vita")
    assert result["schema_valid"] is True
    assert result["rhyme_consistent"] is False
    assert result["valid"] is False
    parsed_bad = parse_trace(
        build_trace("ABCDEFGHIJKLMN", valid_entries())
        + "\n\n"
        + "\n".join("verso numero qui ora" for _ in range(14))
    )
    bad = validate_trace(parsed_bad, "Nel mezzo del cammin di nostra vita")
    assert bad["schema_valid"] is False
    assert bad["valid"] is False


def test_build_trace_examples_counts_trace_and_poem_tokens():
    card = {
        "unit_id": "unit-1",
        "opening_line": "Nel mezzo del cammin di nostra vita",
        "scheme": "ABBAABBACDECDE",
        "trace_text": build_trace("ABBAABBACDECDE", valid_entries()),
        "poem_text": "\n".join(f"verso numero {index} qui" for index in range(14)),
    }
    examples, skipped = build_trace_examples([card], FakeTokenizer())
    assert skipped == {"malformed": 0, "too_long": 0}
    assert len(examples) == 1
    example = examples[0]
    expected = len((card["trace_text"] + "\n\n" + card["poem_text"]).split()) + 1
    assert len(example["target_ids"]) == expected
    assert example["opening_line"] == card["opening_line"]


def test_build_trace_examples_skips_malformed_cards():
    card = {
        "unit_id": "unit-1",
        "opening_line": "Nel mezzo del cammin di nostra vita",
        "scheme": "ABBAABBACDECDE",
        "trace_text": build_trace("ABBAABBACDECDE", valid_entries()),
        "poem_text": "solo tre versi qui",
    }
    try:
        build_trace_examples([card], FakeTokenizer())
    except ValueError:
        pass
    else:
        raise AssertionError("malformed cards must not produce examples")


def test_plan_prompt_asks_for_the_plan_only():
    prompt = plan_prompt(FakeTokenizer(), "Nel mezzo del cammin di nostra vita")
    assert "Nel mezzo del cammin di nostra vita" in prompt
    assert "Non scrivere i versi" in prompt


def test_build_plan_examples_target_is_the_trace_only():
    card = {
        "unit_id": "unit-1",
        "opening_line": "Nel mezzo del cammin di nostra vita",
        "scheme": "ABBAABBACDECDE",
        "trace_text": build_trace("ABBAABBACDECDE", valid_entries()),
    }
    examples, skipped = build_plan_examples([card], FakeTokenizer())
    assert skipped == {"malformed": 0, "too_long": 0}
    assert len(examples) == 1
    expected = len(card["trace_text"].split()) + 1
    assert len(examples[0]["target_ids"]) == expected


def test_planned_words_from_trace_includes_the_opening_ending():
    trace = build_trace("ABBAABBACDECDE", valid_entries())
    parsed = parse_trace(
        trace + "\n\n" + "\n".join("verso numero qui ora" for _ in range(14))
    )
    words = planned_words_from_trace(parsed, "Nel mezzo del cammin di nostra vita")
    assert words is not None
    assert len(words) == 14
    assert words[0] == "vita"
    assert words[1] == "salita"
    assert words[8] == "sole"
