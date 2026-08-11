import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from sonnet_corpus.sonnet_v7_split import (
    V7SplitConfig,
    author_group_id,
    build_v7_sonnet_split,
    canonicalize_author_label,
    derive_work_group_id,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "data/processed/canonical_italian_corpora_v1/sonnets_manifest.csv"
V6 = ROOT / "data/metadata/sonnets_expanded_v6_manifest.csv"
V7 = ROOT / "data/metadata/sonnets_expanded_v7_manifest.csv"
AUTHOR_GROUPS = ROOT / "data/metadata/sonnets_expanded_v7_author_groups_v1.csv"
REPORT = ROOT / "reports/sonnets_expanded_v7_split_v1.json"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_author_keys_merge_reviewed_order_particle_and_historical_aliases():
    assert canonicalize_author_label("Varchi, Benedetto") == canonicalize_author_label(
        "Benedetto Varchi"
    )
    assert author_group_id("Sannazaro, Iacopo") == author_group_id(
        "Jacopo Sannazaro"
    )
    assert author_group_id("Folgore di San Gimignano") == author_group_id(
        "Folgore da San Gimignano"
    )
    assert author_group_id("Gaspara Stampa") != author_group_id("Baldassarre Stampa")


def test_generic_author_labels_are_work_grouped_instead_of_conflated():
    assert author_group_id("") == ""
    assert author_group_id("Anonimo") == ""
    assert author_group_id("Poesie anonime") == ""
    assert author_group_id("Varie Rime degli Arcadi") == ""


def test_work_groups_collapse_only_candidate_suffixes_within_one_source():
    assert derive_work_group_id("gutenberg", "pg100:char10-20") == derive_work_group_id(
        "gutenberg", "pg100:char30-40"
    )
    assert derive_work_group_id("bibit", "bibit1:sonnet_0001") == derive_work_group_id(
        "bibit", "bibit1:sonnet_0002"
    )
    assert derive_work_group_id("gutenberg", "pg100:char10-20") != derive_work_group_id(
        "gutenberg", "pg101:char10-20"
    )


def test_v7_manifest_accounts_for_every_canonical_identity_and_exclusion():
    canonical = _rows(CANONICAL)
    v7 = _rows(V7)
    assert len(canonical) == len(v7) == 22_693
    assert [row["unit_id"] for row in canonical] == [row["unit_id"] for row in v7]
    assert Counter(row["v7_split"] for row in v7) == {
        "train": 19_899,
        "validation": 1_247,
        "test": 1_244,
        "excluded": 303,
    }
    assert sum(row["include_in_v7"] == "true" for row in v7) == 22_390
    assert sum(row["v7_training_eligible"] == "true" for row in v7) == 19_899
    assert all(not row["storage_path"].startswith("data/local/") for row in v7)


def test_every_v6_identity_and_split_is_preserved_exactly():
    v6 = {row["poem_id"]: row for row in _rows(V6)}
    v7 = {
        row["source_id"]: row
        for row in _rows(V7)
        if row["source_group"] == "v6_sonnets"
    }

    assert len(v6) == len(v7) == 1_868
    assert set(v6) == set(v7)
    assert Counter(row["v7_split"] for row in v7.values()) == {
        "train": 1_481,
        "validation": 190,
        "test": 197,
    }
    for poem_id, row in v6.items():
        assert v7[poem_id]["v7_split"] == row["split_expanded_with_petrarch"]
        assert v7[poem_id]["v7_split_tier"] == "legacy_v6_locked"


def test_approved_legacy_author_additions_remain_in_training_and_are_disclosed():
    rows = _rows(V7)
    approved = [
        row
        for row in rows
        if row["v7_split_decision"] == "approved_legacy_protected_author_overlap_train"
    ]

    assert len(approved) == 2_118
    assert all(row["v7_split"] == "train" for row in approved)
    assert {row["author"] for row in approved} >= {
        "Varchi, Benedetto",
        "Alighieri, Dante",
        "Petrarca, Francesco",
        "Sannazaro, Iacopo",
    }


def test_clean_v7_heldout_authors_and_works_are_absent_from_training_and_v6():
    rows = _rows(V7)
    train = [row for row in rows if row["v7_split"] == "train"]
    clean = [row for row in rows if row["v7_split_tier"] == "clean_v7_grouped"]
    v6 = [row for row in rows if row["source_group"] == "v6_sonnets"]

    train_authors = {row["author_group_id"] for row in train if row["author_group_id"]}
    v6_authors = {row["author_group_id"] for row in v6 if row["author_group_id"]}
    heldout_authors = {row["author_group_id"] for row in clean if row["author_group_id"]}
    train_works = {row["work_group_id"] for row in train}
    heldout_works = {row["work_group_id"] for row in clean}

    assert heldout_authors.isdisjoint(train_authors)
    assert heldout_authors.isdisjoint(v6_authors)
    assert heldout_works.isdisjoint(train_works)
    assert Counter(row["v7_split"] for row in clean) == {
        "validation": 1_057,
        "test": 1_047,
    }


def test_new_author_work_components_never_cross_splits():
    rows = [
        row
        for row in _rows(V7)
        if row["training_eligible"] == "true" and row["source_group"] != "v6_sonnets"
    ]
    splits_by_group = defaultdict(set)
    for row in rows:
        splits_by_group[row["split_group_id"]].add(row["v7_split"])

    assert len(splits_by_group) == 602
    assert all(len(splits) == 1 for splits in splits_by_group.values())


def test_author_group_ledger_exposes_alias_and_protected_group_membership():
    rows = _rows(AUTHOR_GROUPS)
    by_label = {row["raw_author_label"]: row for row in rows}

    assert len(rows) == 481
    assert by_label["Varchi, Benedetto"]["author_group_id"] == by_label[
        "Benedetto Varchi"
    ]["author_group_id"]
    assert by_label["Varchi, Benedetto"]["protected_v6_presence"] == "true"
    assert by_label["Sannazaro, Iacopo"]["protected_v6_presence"] == "true"
    assert by_label["Anonimo"]["resolution_status"] == "generic_author_work_grouped"


def test_v7_report_freezes_counts_policy_and_no_gpu_boundary():
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["v7_split_counts"] == {
        "test": 1_244,
        "train": 19_899,
        "validation": 1_247,
    }
    assert report["new_split_counts"] == {
        "test": 1_047,
        "train": 18_418,
        "validation": 1_057,
    }
    assert report["v7_identity_sha256"] == (
        "a636a1d62a624c300cc782bfc4fa1b54e77d5bea8b4e6783f77b8c618b845d1e"
    )
    assert report["verification"]["all_v6_splits_preserved"] is True
    assert report["verification"]["legacy_v6_author_overlap_disclosed"] is True
    assert report["verification"]["minerva_tokenization_performed"] is False
    assert report["verification"]["mixture_weights_assigned"] is False
    assert report["verification"]["gpu_work_started"] is False


def test_v7_builder_reproduces_every_committed_artifact(tmp_path):
    generated = tmp_path / "generated"
    report = build_v7_sonnet_split(
        V7SplitConfig(
            repo_root=ROOT,
            canonical_sonnet_manifest_path=CANONICAL,
            v6_manifest_path=V6,
            author_group_path=generated / AUTHOR_GROUPS.name,
            v7_manifest_path=generated / V7.name,
            json_report_path=generated / REPORT.name,
            markdown_report_path=generated / "sonnets_expanded_v7_split_v1.md",
        )
    )

    assert report["v7_identity_sha256"] == (
        "a636a1d62a624c300cc782bfc4fa1b54e77d5bea8b4e6783f77b8c618b845d1e"
    )
    for committed in (
        AUTHOR_GROUPS,
        V7,
        REPORT,
        ROOT / "reports/sonnets_expanded_v7_split_v1.md",
    ):
        assert (generated / committed.name).read_bytes() == committed.read_bytes()
