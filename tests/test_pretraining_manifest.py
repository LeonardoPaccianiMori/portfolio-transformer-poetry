import csv
from pathlib import Path

import pytest

from sonnet_corpus.pretraining_manifest import (
    PretrainingSourceRow,
    write_pretraining_manifest,
)


def make_row(**overrides):
    values = {
        "source_id": "pg_sidrac_44549",
        "title": "Il libro di Sidrach: testo inedito del secolo XIV",
        "author": "Sidrac / anonymous tradition",
        "source_archive": "Project Gutenberg",
        "source_collection": "Project Gutenberg Italian",
        "landing_page_url": "https://www.gutenberg.org/ebooks/44549",
        "download_url": "",
        "ebook_id": "44549",
        "language": "Italian",
        "period_bucket": "tier_a_b_borderline",
        "approx_date": "XIV secolo",
        "genre": "encyclopedic / philosophical prose",
        "text_kind": "prose",
        "inclusion_status": "include_probe",
        "public_domain_status": "Public domain in the USA",
        "license_notes": "Project Gutenberg public-domain status on landing page.",
        "edition_notes": "",
        "source_release_date": "",
        "source_last_updated": "",
        "expected_clean_text_path": "data/processed/pretraining/pg_sidrac_44549.txt",
        "token_count_report_path": "",
        "split": "",
        "boilerplate_strategy": "strip Project Gutenberg header and footer",
        "mixed_text_strategy": "",
        "cleaning_notes": "",
        "audit_notes": "",
    }
    values.update(overrides)
    return PretrainingSourceRow(**values)


def test_pretraining_manifest_validation_rejects_empty_required_field():
    row = make_row(title="")

    with pytest.raises(ValueError, match="empty required field"):
        row.validate()


def test_pretraining_manifest_validation_rejects_active_poetry_rows():
    row = make_row(
        source_id="pg_orlando_furioso",
        title="Orlando Furioso",
        text_kind="poetry",
        inclusion_status="include_probe",
    )

    with pytest.raises(ValueError, match="poetry source must be excluded"):
        row.validate()


def test_pretraining_manifest_validation_allows_excluded_poetry_rows():
    row = make_row(
        source_id="pg_orlando_furioso",
        title="Orlando Furioso",
        text_kind="poetry",
        inclusion_status="exclude",
        split="excluded",
    )

    row.validate()


def test_pretraining_manifest_validation_requires_mixed_text_strategy():
    row = make_row(
        source_id="pg_vita_nuova_71218",
        title="La vita nuova",
        text_kind="mixed",
        inclusion_status="conditional_extract_prose",
        mixed_text_strategy="",
    )

    with pytest.raises(ValueError, match="strategy is required"):
        row.validate()


def test_pretraining_manifest_validation_requires_gutenberg_ebook_id():
    row = make_row(ebook_id="")

    with pytest.raises(ValueError, match="requires ebook_id"):
        row.validate()


def test_pretraining_manifest_validation_requires_gutenberg_public_domain_status():
    row = make_row(public_domain_status="unknown")

    with pytest.raises(ValueError, match="requires public-domain status"):
        row.validate()


def test_pretraining_manifest_validation_requires_liber_liber_license_layer_note():
    row = make_row(
        source_id="liberliber_decameron",
        source_archive="Liber Liber",
        source_collection="Liber Liber",
        landing_page_url="https://liberliber.it/autori/autori-b/giovanni-boccaccio/",
        ebook_id="",
        public_domain_status="underlying work out of copyright",
        license_notes="underlying work is out of copyright",
        edition_notes="",
        boilerplate_strategy="remove Liber Liber wrapper text",
    )

    with pytest.raises(ValueError, match="license/edition-layer note"):
        row.validate()


def test_active_liber_liber_row_requires_exact_creative_commons_terms():
    row = make_row(
        source_id="liberliber_decameron",
        source_archive="Liber Liber",
        source_collection="Liber Liber / Progetto Manuzio",
        landing_page_url="https://liberliber.it/example",
        ebook_id="",
        inclusion_status="include_probe",
        public_domain_status="underlying work out of copyright",
        license_notes="Liber Liber license applies to the digital edition.",
        edition_notes="Modern digital edition.",
        boilerplate_strategy="extract primary text from the ODT file",
    )

    with pytest.raises(ValueError, match="exact CC BY-NC-SA 4.0 terms"):
        row.validate()


def test_write_pretraining_manifest_outputs_expected_header(tmp_path: Path):
    path = tmp_path / "broader_prose_sources_manifest.csv"

    write_pretraining_manifest([make_row()], path)

    text = path.read_text(encoding="utf-8")
    assert "source_id,title,author,source_archive" in text
    assert "mixed_text_strategy" in text
    assert "pg_sidrac_44549" in text


def test_starter_broader_prose_manifest_rows_are_valid():
    path = Path("data/metadata/broader_prose_sources_manifest.csv")

    with path.open(encoding="utf-8", newline="") as handle:
        rows = [PretrainingSourceRow(**row) for row in csv.DictReader(handle)]

    assert {row.source_id for row in rows} >= {
        "pg_sidrac_44549",
        "pg_villani_cronica_vol1_69898",
        "pg_jacopo_alighieri_chiose_30766",
        "pg_caterina_dialogo_26961",
        "pg_dante_vita_nuova_71218",
        "pg_villani_cronica_vol2_69899",
        "pg_villani_cronica_vol3_69900",
        "pg_villani_cronica_vol4_69901",
        "pg_villani_cronica_vol5_69902",
        "ll_novellino",
        "ll_giovanni_villani_nuova_cronica",
        "ll_boccaccio_decameron_mondadori",
        "ll_sacchetti_trecentonovelle",
        "ll_bandello_novelle",
        "ll_bembo_asolani",
        "ll_bembo_prose_volgar_lingua",
        "ll_castiglione_cortegiano",
        "ll_cellini_vita",
        "ll_machiavelli_discorsi",
        "ll_machiavelli_istorie_fiorentine",
        "ll_machiavelli_arte_guerra",
        "ll_machiavelli_principe",
        "ll_guicciardini_storia_italia",
        "ll_guicciardini_storie_fiorentine",
        "ll_guicciardini_discorsi_politici",
        "ll_guicciardini_ricordi",
        "ll_vasari_vite_1568",
        "ll_masuccio_novellino",
        "ll_ramusio_navigazioni_viaggi",
    }
    for row in rows:
        row.validate()

    rows_by_id = {row.source_id: row for row in rows}
    assert rows_by_id["ll_cellini_vita"].inclusion_status == "defer"
    assert rows_by_id["ll_cellini_vita"].split == "excluded"
