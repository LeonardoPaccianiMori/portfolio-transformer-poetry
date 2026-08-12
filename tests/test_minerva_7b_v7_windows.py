import hashlib
import json
from pathlib import Path

import pytest

from sonnet_training.minerva_7b_v7_windows import (
    DocumentSpan,
    PoolIndex,
    ShardSpan,
    compare_window_indexes,
    enumerate_pool_windows,
    interleave_stage_windows,
    load_sampling_policy,
    pool_index_identity,
    sample_component_windows,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "data/metadata/minerva_7b_v7_sampling_policy_v1.json"
ENCODED_REPORT_PATH = ROOT / "reports/minerva_7b_v7_encoded_data_v1.json"


def _pool(tmp_path: Path) -> PoolIndex:
    first = tmp_path / "pool-00000.int32.bin"
    second = tmp_path / "pool-00001.int32.bin"
    first.write_bytes(b"\0" * 20)
    second.write_bytes(b"\0" * 20)
    return PoolIndex(
        pool_id="synthetic_train",
        corpus_role="historical_general",
        split="train",
        tokens=10,
        documents=(
            DocumentSpan(0, "doc:a", 0, 3, "author:a", "work:a", "epoch:a"),
            DocumentSpan(1, "doc:b", 3, 7, "author:b", "work:b", "epoch:b"),
            DocumentSpan(2, "doc:c", 7, 10, "author:c", "work:c", "epoch:c"),
        ),
        shards=(
            ShardSpan(0, 0, 5, 5, hashlib.sha256(first.read_bytes()).hexdigest(), first),
            ShardSpan(1, 5, 10, 5, hashlib.sha256(second.read_bytes()).hexdigest(), second),
        ),
        content_identity_sha256="a" * 64,
    )


def test_sampling_policy_pins_encoded_lineage_and_non_gpu_boundaries():
    policy = load_sampling_policy(POLICY_PATH, ENCODED_REPORT_PATH)

    assert policy["expected"]["training_windows"] == 47_360
    assert policy["expected"]["training_target_tokens"] == 96_993_280
    assert policy["windowing"]["source_span_tokens"] == 2049
    assert policy["publication"]["individual_window_indexes_public"] is False
    assert policy["publication"]["gpu_work_authorized"] is False
    assert policy["publication"]["cache_deletion_authorized"] is False


def test_policy_rejects_changed_encoded_report(tmp_path):
    changed = tmp_path / "encoded.json"
    changed.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        load_sampling_policy(POLICY_PATH, changed)


def test_window_enumeration_records_cross_document_and_cross_shard_spans(tmp_path):
    windows = enumerate_pool_windows(
        _pool(tmp_path), source_span_tokens=5, target_stride_tokens=4
    )

    assert len(windows) == 2
    assert windows[0].source_start == 0
    assert windows[0].source_slices == ((0, 0, 5),)
    assert [(row.unit_id, row.tokens) for row in windows[0].contributions] == [
        ("doc:a", 2),
        ("doc:b", 2),
    ]
    assert windows[1].source_start == 4
    assert windows[1].source_slices == ((0, 4, 1), (1, 0, 4))
    assert [(row.unit_id, row.tokens) for row in windows[1].contributions] == [
        ("doc:b", 2),
        ("doc:c", 2),
    ]


def test_concentration_caps_use_exact_target_contributions_and_repeat_cycles(tmp_path):
    candidates = enumerate_pool_windows(
        _pool(tmp_path), source_span_tokens=5, target_stride_tokens=4
    )

    selected = sample_component_windows(
        candidates,
        count=4,
        target_tokens_per_window=4,
        seed=7,
        stage_id="stage",
        component="component",
        ceilings={"author_key": 0.5, "work_key": 0.5},
    )

    assert len(selected) == 4
    assert max(row.selection_cycle for row in selected) >= 1
    author_exposure = {}
    for sampled in selected:
        for contribution in sampled.candidate.contributions:
            author_exposure[contribution.author_key] = (
                author_exposure.get(contribution.author_key, 0)
                + contribution.tokens
            )
    assert max(author_exposure.values()) <= 8


def test_infeasible_concentration_caps_fail_closed(tmp_path):
    candidate = enumerate_pool_windows(
        _pool(tmp_path), source_span_tokens=5, target_stride_tokens=4
    )[0]

    with pytest.raises(ValueError, match="infeasible"):
        sample_component_windows(
            [candidate],
            count=1,
            target_tokens_per_window=4,
            seed=7,
            stage_id="stage",
            component="component",
            ceilings={"author_key": 0.25},
        )


def test_stage_interleaving_preserves_exact_component_counts(tmp_path):
    candidates = enumerate_pool_windows(
        _pool(tmp_path), source_span_tokens=5, target_stride_tokens=4
    )
    first = sample_component_windows(
        candidates,
        count=2,
        target_tokens_per_window=4,
        seed=1,
        stage_id="stage",
        component="first",
        ceilings={},
    )
    second = sample_component_windows(
        candidates,
        count=4,
        target_tokens_per_window=4,
        seed=1,
        stage_id="stage",
        component="second",
        ceilings={},
    )

    ordered = interleave_stage_windows({"first": first, "second": second})

    assert len(ordered) == 6
    assert [component for component, _ in ordered].count("first") == 2
    assert [component for component, _ in ordered].count("second") == 4
    assert [component for component, _ in ordered][:3] == [
        "second",
        "first",
        "second",
    ]


def test_path_independent_pool_and_window_identities(tmp_path):
    primary = _pool(tmp_path)
    reproduction_dir = tmp_path / "reproduction"
    reproduction_dir.mkdir()
    reproduction = _pool(reproduction_dir)

    assert pool_index_identity({primary.pool_id: primary}) == pool_index_identity(
        {reproduction.pool_id: reproduction}
    )
    comparison = compare_window_indexes(
        {
            "window_index_content_identity_sha256": "f" * 64,
            "training": {"windows": 1},
            "evaluation": {"validation": {"windows": 1}},
        },
        {
            "window_index_content_identity_sha256": "f" * 64,
            "training": {"windows": 1},
            "evaluation": {"validation": {"windows": 1}},
        },
    )
    assert comparison["match"] is True


def test_policy_expected_stage_counts_match_checkpoint_8c_report():
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    report = json.loads(ENCODED_REPORT_PATH.read_text(encoding="utf-8"))

    assert policy["expected"]["stage_windows"] == {
        row["stage_id"]: row["target_windows_2048"]
        for row in report["stage_plan"]["stages"]
    }
    assert policy["expected"]["training_windows"] == report["stage_plan"][
        "total_target_windows_2048"
    ]


def test_completed_checkpoint_artifacts_freeze_window_counts_caps_and_identity():
    report_path = ROOT / "reports/minerva_7b_v7_stage_windows_v1.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["status"] == "active_verified"
    assert report["training"]["windows"] == 47_360
    assert report["training"]["target_tokens"] == 96_993_280
    assert [row["windows"] for row in report["training"]["stages"]] == [
        33_040,
        12_160,
        2_160,
    ]
    assert report["evaluation"]["validation"]["windows"] == 959
    assert report["evaluation"]["test"]["windows"] == 106
    assert report["window_index_content_identity_sha256"] == (
        "e821e3afdc3bd7aa6874180509ba756f942e651980f6455469722c13f8f7424c"
    )
    sonnet = report["training"]["stages"][2]["components"][0]
    concentration = {row["field"]: row for row in sonnet["concentration"]}
    assert concentration["author_key"]["maximum_tokens"] == 163_415
    assert concentration["author_key"]["passes"] is True
    assert concentration["epoch_key"]["maximum_tokens"] == 1_061_670
    assert concentration["epoch_key"]["passes"] is True
    assert report["reproduction"]["match"] is True
    assert report["verification"]["conditioned_material_included"] is False
    assert report["verification"]["gpu_work_started"] is False
    assert report["verification"]["cache_deleted"] is False
