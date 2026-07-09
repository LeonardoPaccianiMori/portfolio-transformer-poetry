import pytest

from sonnet_corpus.pretraining_cleaning import (
    clean_pretraining_text,
    normalize_text_boundaries,
    validate_cleaned_text,
)


def test_normalize_text_boundaries_preserves_text_but_normalizes_spacing():
    text = "Prima riga.  \r\n\r\n\r\nSeconda riga.\r\n"

    normalized = normalize_text_boundaries(text)

    assert normalized == "Prima riga.\n\nSeconda riga.\n"


def test_clean_pretraining_text_rejects_empty_text():
    with pytest.raises(ValueError, match="cleaned text is empty"):
        clean_pretraining_text(" \n\t ", source_id="empty")


def test_validate_cleaned_text_rejects_too_short_extractions():
    with pytest.raises(ValueError, match="too short"):
        validate_cleaned_text("Breve.\n", source_id="short", min_character_count=20)


def test_validate_cleaned_text_rejects_project_gutenberg_markers():
    text = (
        "Corpo del testo.\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        "Altro testo.\n"
    )

    with pytest.raises(ValueError, match="wrapper marker"):
        validate_cleaned_text(text, source_id="pg_bad", min_character_count=1)
