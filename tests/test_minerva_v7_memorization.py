from sonnet_analysis.minerva_v7_memorization import score_texts_against_reference


def _record(record_id: str, text: str) -> dict[str, str]:
    return {"record_id": record_id, "source_id": record_id, "text": text}


def test_high_lcs_risk_cannot_be_hidden_by_higher_medium_containment():
    generated = "|".join(f"token{index:03d}" for index in range(120))
    high_lcs = _record("high-lcs", generated[100:260] + "\N{SECTION SIGN}high-tail")
    medium_containment = _record(
        "medium-containment",
        "~".join(
            generated[start : start + 60]
            for start in (300, 390, 480, 570, 660, 750, 840, 930)
        ),
    )

    forward = score_texts_against_reference(
        [generated], [high_lcs, medium_containment]
    )[0]
    reversed_order = score_texts_against_reference(
        [generated], [medium_containment, high_lcs]
    )[0]
    medium_only = score_texts_against_reference(
        [generated], [medium_containment]
    )[0]

    assert forward == reversed_order
    assert forward["nearest_poem_id"] == "high-lcs"
    assert forward["risk_level"] == "high"
    assert forward["longest_common_substring_chars"] == 160
    assert medium_only["risk_level"] == "medium"
    assert forward["ngram_containment"] < medium_only["ngram_containment"]


def test_equal_scores_use_a_stable_reference_identity_tie_breaker():
    generated = "a" * 200
    first = _record("a-reference", generated)
    second = _record("z-reference", generated)

    forward = score_texts_against_reference([generated], [first, second])[0]
    reversed_order = score_texts_against_reference([generated], [second, first])[0]

    assert forward == reversed_order
    assert forward["nearest_poem_id"] == "a-reference"
