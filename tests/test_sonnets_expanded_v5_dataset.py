from collections import Counter
from pathlib import Path

from sonnet_corpus.sonnet_expansion_build import (
    normalize_for_duplicate_check,
    read_manifest_rows,
    split_counts,
)


ROOT = Path(__file__).resolve().parents[1]
V5_MANIFEST = ROOT / "data/metadata/sonnets_expanded_v5_manifest.csv"


def test_expanded_v5_contains_every_approved_source_addition():
    rows = read_manifest_rows(V5_MANIFEST)
    author_counts = Counter(row.author for row in rows)

    assert len(rows) == 1875
    assert author_counts["Isabella Andreini"] == 196
    assert author_counts["Vittoria Colonna"] == 336
    assert author_counts["Gaspara Stampa"] == 282
    assert author_counts["Ludovico Ariosto"] == 29
    assert author_counts["Jacopo Sannazaro"] == 21
    assert split_counts(rows) == {
        "train": 1486,
        "validation": 191,
        "test": 198,
    }


def test_expanded_v5_processed_files_are_complete_and_exactly_deduplicated():
    rows = read_manifest_rows(V5_MANIFEST)
    v4_ids = {
        row.poem_id
        for row in read_manifest_rows(
            ROOT / "data/metadata/sonnets_expanded_v4_manifest.csv"
        )
    }
    base_texts = []
    added_texts = []
    for row in rows:
        text = (ROOT / row.clean_text_path).read_text(encoding="utf-8")
        if row.poem_id in v4_ids:
            base_texts.append(text)
        else:
            added_texts.append(text)

    assert len({row.poem_id for row in rows}) == len(rows)
    assert all(int(row.line_count_clean) == 14 for row in rows)
    assert all(text.count("\n") == 14 for text in base_texts + added_texts)
    normalized_base = {
        normalize_for_duplicate_check(text) for text in base_texts
    }
    normalized_added = [
        normalize_for_duplicate_check(text) for text in added_texts
    ]
    assert len(normalized_added) == 864
    assert len(set(normalized_added)) == len(normalized_added)
    assert normalized_base.isdisjoint(normalized_added)


def test_expanded_v5_preserves_every_v4_poem_verbatim():
    v4_rows = read_manifest_rows(
        ROOT / "data/metadata/sonnets_expanded_v4_manifest.csv"
    )
    v5_rows_by_id = {
        row.poem_id: row for row in read_manifest_rows(V5_MANIFEST)
    }

    assert len(v4_rows) == 1011
    for v4_row in v4_rows:
        v5_row = v5_rows_by_id[v4_row.poem_id]
        assert (ROOT / v5_row.clean_text_path).read_bytes() == (
            ROOT / v4_row.clean_text_path
        ).read_bytes()
