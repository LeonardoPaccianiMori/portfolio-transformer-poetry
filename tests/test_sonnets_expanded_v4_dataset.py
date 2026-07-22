from pathlib import Path

from sonnet_corpus.sonnet_expansion_build import read_manifest_rows, split_counts


ROOT = Path(__file__).resolve().parents[1]


def test_expanded_v4_contains_the_pinned_varchi_addition():
    manifest_path = ROOT / "data/metadata/sonnets_expanded_v4_manifest.csv"
    rows = read_manifest_rows(manifest_path)
    varchi_rows = [row for row in rows if row.audit_notes.endswith("ws_varchi_infermita")]

    assert len(rows) == 1011
    assert len(varchi_rows) == 33
    assert {row.author for row in varchi_rows} == {"Benedetto Varchi"}
    assert all(int(row.line_count_clean) == 14 for row in varchi_rows)
    assert all("Firenze, per il Magheri, 1821" in row.source_edition for row in varchi_rows)
    assert all(row.license_notes.startswith("CC BY-SA 3.0 / GFDL") for row in varchi_rows)
    assert all((ROOT / row.clean_text_path).is_file() for row in rows)
    assert split_counts(rows) == {"train": 809, "validation": 103, "test": 99}
