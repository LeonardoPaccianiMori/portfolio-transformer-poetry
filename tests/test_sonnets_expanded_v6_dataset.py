import hashlib
import json
from pathlib import Path

from sonnet_corpus.sonnet_expansion_build import (
    normalize_for_duplicate_check,
    read_manifest_rows,
    split_counts,
)
from sonnet_corpus.sonnet_v6_correction import (
    V6_CANONICAL_DUPLICATE_RECORDS,
    V6_EXPECTED_COUNTS,
    V6_REMOVALS,
)


ROOT = Path(__file__).resolve().parents[1]
V5_MANIFEST = ROOT / "data/metadata/sonnets_expanded_v5_manifest.csv"
V6_MANIFEST = ROOT / "data/metadata/sonnets_expanded_v6_manifest.csv"


def test_expanded_v6_removes_only_the_fixed_v5_defects():
    v5_rows = read_manifest_rows(V5_MANIFEST)
    v6_rows = read_manifest_rows(V6_MANIFEST)
    v5_by_id = {row.poem_id: row for row in v5_rows}
    v6_by_id = {row.poem_id: row for row in v6_rows}

    assert len(v5_rows) == 1875
    assert len(v6_rows) == 1868
    assert set(v5_by_id) - set(v6_by_id) == set(V6_REMOVALS)
    assert set(V6_CANONICAL_DUPLICATE_RECORDS) <= set(v6_by_id)
    assert split_counts(v6_rows) == V6_EXPECTED_COUNTS

    for poem_id, row in v6_by_id.items():
        v6_text = (ROOT / row.clean_text_path).read_bytes()
        v5_text = (ROOT / v5_by_id[poem_id].clean_text_path).read_bytes()
        assert v6_text == v5_text
        assert v6_text.count(b"\n") == 14


def test_expanded_v6_has_unique_texts_and_preserves_prompt_isolation():
    rows = read_manifest_rows(V6_MANIFEST)
    rows_by_id = {row.poem_id: row for row in rows}
    fingerprints = [
        normalize_for_duplicate_check(
            (ROOT / row.clean_text_path).read_text(encoding="utf-8")
        )
        for row in rows
    ]
    assert len(fingerprints) == len(set(fingerprints))

    for config_name, expected_split in (
        ("minerva_3b_validation_sanity_prompts.json", "validation"),
        ("task_format_acceptance_prompts.json", "test"),
    ):
        prompts = json.loads((ROOT / "configs" / config_name).read_text())
        for prompt in prompts:
            assert rows_by_id[prompt["poem_id"]].split_expanded_with_petrarch == (
                expected_split
            )


def test_expanded_v6_report_matches_committed_manifest():
    report = json.loads(
        (ROOT / "data/metadata/sonnets_expanded_v6_build_report.json").read_text()
    )
    manifest_bytes = V6_MANIFEST.read_bytes()

    assert report["output_poem_count"] == 1868
    assert report["removed_poem_count"] == 7
    assert report["split_counts"] == V6_EXPECTED_COUNTS
    assert report["exact_duplicate_group_count"] == 0
    assert report["output_manifest_sha256"] == hashlib.sha256(
        manifest_bytes
    ).hexdigest()
