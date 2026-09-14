from sonnet_training.self_play_rft import (
    ARMS,
    autonomous_prompt,
    build_no_plan_examples,
    candidate_keep,
    filter_cards,
    self_play_reading,
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


def good_card(**overrides):
    card = {
        "path": "candidate.json",
        "opening_line": "verso 0 della poesia vita",
        "text": "\n".join(
            [f"verso {index} della poesia vita" for index in range(14)]
        ),
        "plan_scheme_ok": True,
        "hendecasyllable_lines": 10,
        "failed_lines": 1,
        "type_token_ratio": 0.8,
        "repeated_line_ratio": 0.0,
        "repeated_bigram_ratio": 0.01,
        "final_word_repetition": 0.01,
        "copied_lines": 0,
    }
    card.update(overrides)
    return card


def test_candidate_keep_requires_scheme_and_quality():
    assert candidate_keep(good_card())
    assert not candidate_keep(good_card(plan_scheme_ok=False))
    assert not candidate_keep(good_card(copied_lines=1))
    assert not candidate_keep(good_card(hendecasyllable_lines=6))
    assert not candidate_keep(good_card(failed_lines=3))
    assert not candidate_keep(good_card(type_token_ratio=0.4))
    assert not candidate_keep(good_card(repeated_line_ratio=0.2))
    assert not candidate_keep(good_card(repeated_bigram_ratio=0.2))


def test_filter_cards_reports_rejection_reasons():
    cards = [
        good_card(),
        good_card(path="scheme.json", plan_scheme_ok=False),
        good_card(path="copied.json", copied_lines=2),
        good_card(path="accepted.json", hendecasyllable_lines=3),
        good_card(path="failed.json", failed_lines=5),
        good_card(path="degenerate.json", repeated_bigram_ratio=0.5),
    ]
    kept, rejected = filter_cards(cards)
    assert len(kept) == 1
    assert rejected == {
        "scheme": 1,
        "copied": 1,
        "accepted_lines": 1,
        "failed_lines": 1,
        "degenerate": 1,
    }


def test_autonomous_prompt_has_no_plan_instruction():
    prompt = autonomous_prompt(FakeTokenizer(), "Qualsivoglia scrittore asino o dotto")
    assert "Qualsivoglia scrittore asino o dotto" in prompt
    assert "Termina ogni verso con la parola indicata" not in prompt


def test_build_no_plan_examples_excludes_the_opening_from_the_target():
    card = good_card()
    examples, skipped = build_no_plan_examples([card], FakeTokenizer())
    assert skipped == {"malformed": 0, "too_long": 0}
    assert len(examples) == 1
    example = examples[0]
    assert len(example["target_ids"]) == 13 * 5 + 1
    assert example["opening_line"] == "verso 0 della poesia vita"


def test_build_no_plan_examples_skips_malformed_cards():
    card = good_card(text="\n".join(f"verso {index}" for index in range(13)))
    try:
        build_no_plan_examples([card], FakeTokenizer())
    except ValueError:
        pass
    else:
        raise AssertionError("malformed cards must not produce examples")


def test_self_play_reading_gate():
    signal = self_play_reading(
        baseline_valid=0.10,
        rft_valid=0.35,
        baseline_accepted=7.0,
        rft_accepted=7.1,
        judge_gain=0.0,
    )
    assert signal["result"] == "SELF_PLAY_SIGNAL"
    below = self_play_reading(
        baseline_valid=0.10,
        rft_valid=0.29,
        baseline_accepted=7.0,
        rft_accepted=7.1,
        judge_gain=0.0,
    )
    assert below["result"] == "SELF_PLAY_NULL"
    coherence_loss = self_play_reading(
        baseline_valid=0.10,
        rft_valid=0.35,
        baseline_accepted=7.0,
        rft_accepted=7.1,
        judge_gain=-0.5,
    )
    assert coherence_loss["result"] == "SELF_PLAY_NULL"
    assert ARMS[1].startswith("autonomous_")
