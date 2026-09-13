from sonnet_evaluation.coherence import (
    coherence_proxies,
    corrupt_shuffle_lines,
    corrupt_swap_words,
    corrupt_truncate,
    parse_judge_output,
)

GOOD = "\n".join(
    [
        "Nel mezzo del cammin di nostra vita",
        "mi ritrovai per una selva oscura",
        "ché la diritta via era smarrita",
        "e quanto a dir qual era è cosa dura",
    ]
)


def test_proxies_detect_repetition():
    repeated = "\n".join(["Nel mezzo del cammin di nostra vita"] * 4)
    good = coherence_proxies(GOOD)
    bad = coherence_proxies(repeated)
    assert good["repeated_line_ratio"] == 0.0
    assert bad["repeated_line_ratio"] > 0.5
    assert bad["final_word_repetition"] > 0.5


def test_corruptions_change_the_text():
    text = "\n".join([f"verso numero {index} vita" for index in range(8)])
    assert corrupt_shuffle_lines(text) != text
    assert corrupt_swap_words(text) != text
    assert corrupt_truncate(text) != text
    assert len(corrupt_truncate(text, keep=2).splitlines()) == 2


def test_parse_judge_output_accepts_banner_and_json():
    raw = (
        "\x1b[0m> build \xc2\xb7 model\n\n"
        '{"grammar": 4, "continuity": 3, "imagery": 2, '
        '"meta_text_absence": 5, "note": "ok"}\n'
    )
    parsed = parse_judge_output(raw)
    assert parsed is not None
    assert parsed["grammar"] == 4
    assert parsed["mean"] == 3.5


def test_parse_judge_output_rejects_invalid_payloads():
    assert parse_judge_output("no json here") is None
    assert parse_judge_output('{"grammar": 9, "continuity": 1, "imagery": 1, "meta_text_absence": 1}') is None
    assert parse_judge_output('{"grammar": 1}') is None
