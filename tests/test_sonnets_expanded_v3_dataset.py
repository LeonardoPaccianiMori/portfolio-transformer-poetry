from pathlib import Path

from sonnet_corpus.sonnet_expansion_build import read_manifest_rows, split_counts


ROOT = Path(__file__).resolve().parents[1]


def test_expanded_v3_contains_the_pinned_foscolo_addition():
    manifest_path = ROOT / "data/metadata/sonnets_expanded_v3_manifest.csv"
    rows = read_manifest_rows(manifest_path)
    foscolo_rows = [row for row in rows if row.audit_notes.endswith("ws_foscolo_sonetti")]

    assert len(rows) == 978
    assert len(foscolo_rows) == 12
    assert {row.author for row in foscolo_rows} == {"Ugo Foscolo"}
    assert all(int(row.line_count_clean) == 14 for row in foscolo_rows)
    assert all(row.source_edition.startswith("Opere scelte di Ugo Foscolo II") for row in foscolo_rows)
    assert all((ROOT / row.clean_text_path).is_file() for row in rows)
    assert split_counts(rows) == {"train": 783, "validation": 98, "test": 97}
