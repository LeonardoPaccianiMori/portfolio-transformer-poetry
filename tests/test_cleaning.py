from sonnet_corpus.cleaning import clean_poem_text, count_poem_lines


def test_clean_poem_text_removes_editorial_brackets_and_line_markers():
    raw = """
Oi deo d'amore
son ben[e] nato a tua isperagione.
4
ché l'alma mia ti segue

8
"""

    cleaned = clean_poem_text(raw)

    assert "ben[e]" not in cleaned
    assert "bene" in cleaned
    assert "\n4\n" not in cleaned
    assert "\n8\n" not in cleaned
    assert count_poem_lines(cleaned) == 3


def test_clean_poem_text_preserves_line_boundaries():
    raw = "\n".join(f"line {idx}" for idx in range(1, 15))

    cleaned = clean_poem_text(raw)

    assert count_poem_lines(cleaned) == 14
    assert cleaned.splitlines()[0] == "line 1"
    assert cleaned.splitlines()[-1] == "line 14"
