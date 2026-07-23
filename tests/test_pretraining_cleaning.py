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


def test_clean_pretraining_text_removes_only_the_audited_vico_figure_placeholder():
    cleaned = clean_pretraining_text(
        "Titolo\n[inserisci figura1]\n[Conclusione]\n[inserisci figura2]\n",
        source_id="ll_vico_principj_scienza_nuova",
    )

    assert cleaned == "Titolo\n[Conclusione]\n[inserisci figura2]\n"


def test_clean_pretraining_text_removes_audited_liber_liber_edition_credits():
    decameron = clean_pretraining_text(
        "DECAMERON\n\na cura di Vittore Branca\n\nArnoldo Mondadori Editore\n\nPROEMIO\n",
        source_id="ll_boccaccio_decameron_mondadori",
    )
    trecentonovelle = clean_pretraining_text(
        "Il Trecentonovelle\na cura di Emilio Faccioli\nProemio\n",
        source_id="ll_sacchetti_trecentonovelle",
    )

    assert decameron == "DECAMERON\n\nPROEMIO\n"
    assert trecentonovelle == "Il Trecentonovelle\nProemio\n"


def test_clean_pretraining_text_removes_audited_terminal_editorial_notes():
    villani = clean_pretraining_text(
        "Testo primario.\n\n ERRORI                    CORREZIONI\n"
        "Nota del Trascrittore\n",
        source_id="pg_villani_cronica_vol1_69898",
    )
    novellino = clean_pretraining_text(
        "Novella finale.\n\nLE CIENTO NOVELLE ANTIKE\nTavola dei contenuti\n"
        "1 Qui vengono riportate le diciotto novelle incluse "
        "dal Borghini (1572), ma escluse dal Gualteruzzi (1525). "
        "[Nota per l'edizione elettronica Manuzio]\n",
        source_id="ll_novellino",
    )

    assert villani == "Testo primario.\n"
    assert novellino == "Novella finale.\n"


def test_clean_pretraining_text_removes_only_the_terminal_novellino_summary():
    cleaned = clean_pretraining_text(
        "Il sommario del racconto e nel corpo del testo.\n\n"
        "Novella finale.\n\nSOMMARIO\nIndice terminale\n",
        source_id="ll_novellino",
    )

    assert cleaned == "Il sommario del racconto e nel corpo del testo.\n\nNovella finale.\n"


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


def test_validate_cleaned_text_rejects_audited_editorial_markers():
    with pytest.raises(ValueError, match="editorial marker"):
        validate_cleaned_text(
            "Testo.\nA cura di Vittore Branca\n",
            source_id="ll_boccaccio_decameron_mondadori",
            min_character_count=1,
        )
