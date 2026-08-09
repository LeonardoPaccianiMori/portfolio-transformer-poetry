import csv
import json
from pathlib import Path

from sonnet_corpus.biblioteca_italiana import BibItCatalogRecord
from sonnet_corpus.bibit_composition_audit import (
    BibItCompositionAuditConfig,
    ROLE_BRIDGE,
    ROLE_EXCLUDED,
    ROLE_HISTORICAL_GENERAL,
    ROLE_HISTORICAL_POETRY,
    ROLE_SONNET_ONLY,
    audit_bibit_composition,
    classify_record_role,
    rendered_primary_text_character_count,
    resolve_canonical_editions,
)


def _record(
    object_id: str,
    title: str,
    author: str,
    period: str,
    genre: str,
    *,
    date: str = "1990",
) -> BibItCatalogRecord:
    return BibItCatalogRecord(
        object_id=object_id,
        wordpress_id=str(int(object_id[-6:])),
        title=title,
        authors=(author,),
        genres=(genre,),
        periods=(period,),
        languages=("ita",),
        source_authors=(author,),
        source_publisher="Editore",
        source_publication_place="Roma",
        source_publication_date=date,
        source_identifier=f"SBN-{object_id}",
        source_modified_utc="2019-01-01T00:00:00Z",
    )


def _records() -> list[BibItCatalogRecord]:
    return [
        _record("bibit000001", "Trattato", "Autore, Primo", "500", "Trattati"),
        _record("bibit000002", "Poema eroico", "Autore, Secondo", "500", "Poesia"),
        _record("bibit000003", "Rime", "Tasso, Torquato", "500", "Poesia"),
        _record("bibit000004", "Romanzo", "Autrice, Terza", "800", "Narrativa"),
        _record("bibit000005", "Sonetti romaneschi", "Belli, Giuseppe Gioachino", "800", "Poesia"),
        _record("bibit000006", "Orlando Furioso 1516", "Ariosto, Ludovico", "500", "Poesia", date="1516"),
        _record("bibit001135", "Orlando Furioso 1532", "Ariosto, Ludovico", "500", "Poesia", date="1532"),
    ]


def test_rendered_character_count_excludes_generated_wrappers():
    html = """
    <html><head><title>Generated title</title></head><body>
    <div class="stdheader"><h1>Repeated title</h1></div>
    <ul class="toc"><li>Table of contents</li></ul>
    <div><p>Testo letterario principale.</p><span class="pagebreak">[Page 1]</span></div>
    <div class="stdfooter">Generated footer</div>
    </body></html>
    """

    assert rendered_primary_text_character_count(html) == len("Testo letterario principale.")


def test_resolve_canonical_editions_uses_ariosto_1532_override():
    canonical, families = resolve_canonical_editions(_records())
    ariosto_family = next(key for key in canonical if key.startswith("ariosto ludovico::"))

    assert canonical[ariosto_family] == "bibit001135"
    assert any(row["canonical_object_id"] == "bibit001135" for row in families)


def test_composition_rules_do_not_merge_dated_volumes_or_geographic_titles():
    records = [
        _record("bibit000010", "Epistolario: lettere dal 1801 al 1820", "Foscolo, Ugo", "800", "Lettere ed epistolari"),
        _record("bibit000011", "Epistolario: lettere dal 1821 al 1824", "Foscolo, Ugo", "800", "Lettere ed epistolari"),
    ]
    _, families = resolve_canonical_editions(records)
    role, _ = classify_record_role(
        _record(
            "bibit000012",
            "Saggio storico sulla rivoluzione napoletana del 1799",
            "Cuoco, Vincenzo",
            "800",
            "Trattati",
        ),
        is_canonical=True,
    )

    assert families == []
    assert role == ROLE_BRIDGE


def test_resolve_canonical_editions_selects_later_promessi_sposi_text():
    records = [
        _record(
            "bibit000484",
            "I promessi sposi [redazione 1827]",
            "Manzoni, Alessandro",
            "800",
            "Narrativa",
        ),
        _record(
            "bibit000666",
            "Promessi Sposi",
            "Manzoni, Alessandro",
            "800",
            "Narrativa",
        ),
    ]

    canonical, families = resolve_canonical_editions(records)

    assert set(canonical.values()) == {"bibit000666"}
    assert families[0]["canonical_object_id"] == "bibit000666"


def test_audit_bibit_composition_writes_roles_and_public_artifacts(tmp_path):
    records = _records()

    def fetch_catalog(**kwargs):
        kwargs["progress"]("fixture catalog")
        return records

    def fetch_rendered(object_ids, **kwargs):
        kwargs["progress"]("fixture sample")
        return {
            object_id: f"<html><body><p>{'testo ' * (20 + index)}</p></body></html>"
            for index, object_id in enumerate(object_ids)
        }

    config = BibItCompositionAuditConfig(
        repo_root=tmp_path,
        catalog_snapshot_path=tmp_path / "catalog.json",
        decision_csv_path=tmp_path / "decisions.csv",
        json_report_path=tmp_path / "report.json",
        markdown_report_path=tmp_path / "report.md",
        sample_per_stratum=1,
    )
    messages = []
    report = audit_bibit_composition(
        config,
        fetch_catalog=fetch_catalog,
        fetch_rendered_texts=fetch_rendered,
        progress=messages.append,
    )

    assert report["catalog_record_count"] == 7
    assert report["duplicate_edition_family_count"] == 1
    assert report["corpus_activation_status"].startswith("composition_gate_passed")
    assert config.catalog_snapshot_path.is_file()
    assert config.decision_csv_path.is_file()
    assert config.json_report_path.is_file()
    assert config.markdown_report_path.is_file()
    assert "stage 5/5" in " ".join(messages)
    assert json.loads(config.catalog_snapshot_path.read_text())["records"]

    with config.decision_csv_path.open(encoding="utf-8", newline="") as handle:
        rows = {row["object_id"]: row for row in csv.DictReader(handle)}
    assert rows["bibit000001"]["role"] == ROLE_HISTORICAL_GENERAL
    assert rows["bibit000002"]["role"] == ROLE_HISTORICAL_POETRY
    assert rows["bibit000003"]["role"] == ROLE_SONNET_ONLY
    assert rows["bibit000004"]["role"] == ROLE_BRIDGE
    assert rows["bibit000005"]["role"] == ROLE_EXCLUDED
    assert rows["bibit000006"]["role"] == ROLE_EXCLUDED
    assert rows["bibit001135"]["role"] == ROLE_HISTORICAL_POETRY
    assert "does **not** activate" in config.markdown_report_path.read_text()
