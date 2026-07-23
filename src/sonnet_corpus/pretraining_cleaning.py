"""Conservative text cleanup for broader Italian pretraining sources."""

from __future__ import annotations

import re


VICO_FIGURE_PLACEHOLDER_SOURCE_ID = "ll_vico_principj_scienza_nuova"
DECAMERON_SOURCE_ID = "ll_boccaccio_decameron_mondadori"
TRECENTONOVELLE_SOURCE_ID = "ll_sacchetti_trecentonovelle"
NOVELLINO_SOURCE_ID = "ll_novellino"
MATTEO_VILLANI_SOURCE_IDS = {
    "pg_villani_cronica_vol1_69898",
    "pg_villani_cronica_vol2_69899",
    "pg_villani_cronica_vol3_69900",
    "pg_villani_cronica_vol4_69901",
    "pg_villani_cronica_vol5_69902",
}


def normalize_text_boundaries(text: str) -> str:
    """Normalize line endings and blank lines without changing source spelling."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip() + "\n"


def clean_pretraining_text(text: str, *, source_id: str) -> str:
    """Apply source-level cleanup before writing broader pretraining text."""

    if not text.strip():
        raise ValueError(f"cleaned text is empty for {source_id}")

    if source_id == VICO_FIGURE_PLACEHOLDER_SOURCE_ID:
        text = re.sub(
            r"(?m)^[ \t]*\[inserisci figura1\][ \t]*\n?",
            "",
            text,
        )

    if source_id == DECAMERON_SOURCE_ID:
        text = text.replace(
            "a cura di Vittore Branca\n\nArnoldo Mondadori Editore\n\n",
            "",
            1,
        )

    if source_id == TRECENTONOVELLE_SOURCE_ID:
        text = re.sub(r"(?mi)^a cura di Emilio Faccioli\n", "", text, count=1)

    if source_id == NOVELLINO_SOURCE_ID:
        text = re.sub(
            r"\n?1 Qui vengono riportate le diciotto novelle incluse dal Borghini "
            r"\(1572\), ma escluse dal Gualteruzzi \(1525\)\. "
            r"\[Nota per l'edizione elettronica Manuzio\]\s*$",
            "\n",
            text,
        )
        table_start = text.rfind("\nLE CIENTO NOVELLE ANTIKE\n")
        if table_start > 0:
            text = text[:table_start]
        summary_start = text.rfind("\nSOMMARIO\n")
        if summary_start > 0:
            text = text[:summary_start]

    if source_id in MATTEO_VILLANI_SOURCE_IDS:
        text = re.sub(
            r"\n\s*ERRORI\s+CORREZIONI\s*\n.*\Z",
            "\n",
            text,
            flags=re.DOTALL,
        )

    return normalize_text_boundaries(text)


def validate_cleaned_text(
    text: str,
    *,
    source_id: str,
    min_character_count: int = 200,
) -> None:
    """Reject obviously broken text extraction results."""

    if len(text) < min_character_count:
        raise ValueError(
            f"cleaned text for {source_id} is too short: "
            f"{len(text)} characters; expected at least {min_character_count}"
        )

    lower_text = text.casefold()
    forbidden_markers = [
        "*** start of the project gutenberg ebook",
        "*** end of the project gutenberg ebook",
    ]
    for marker in forbidden_markers:
        if marker in lower_text:
            raise ValueError(
                f"cleaned text for {source_id} still contains wrapper marker: {marker}"
            )

    source_specific_markers = {
        DECAMERON_SOURCE_ID: ["a cura di vittore branca", "arnoldo mondadori editore"],
        TRECENTONOVELLE_SOURCE_ID: ["a cura di emilio faccioli"],
        NOVELLINO_SOURCE_ID: [
            "nota per l'edizione elettronica manuzio",
            "\nsommario\n",
        ],
    }
    if source_id in MATTEO_VILLANI_SOURCE_IDS:
        source_specific_markers[source_id] = [
            "nota del trascrittore",
            "errori                    correzioni",
        ]

    for marker in source_specific_markers.get(source_id, []):
        if marker in lower_text:
            raise ValueError(
                f"cleaned text for {source_id} still contains editorial marker: {marker}"
            )
