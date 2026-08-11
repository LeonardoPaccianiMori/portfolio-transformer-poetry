import hashlib
import json
from pathlib import Path

import pytest

from sonnet_corpus.minerva_v7_composition import (
    COMPOSITION_VERSION,
    _count_text_tokens,
    build_staged_composition_gate,
    load_composition_policy,
    render_composition_markdown,
    tokenizer_sha256,
    write_composition_reports,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "data/metadata/minerva_7b_v7_composition_policy_v1.json"
REPORT_PATH = ROOT / "reports/minerva_7b_v7_token_counts_v1.json"


class FakeTokenizer:
    eos_token_id = 2

    def __len__(self):
        return 100

    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        return_attention_mask,
        return_token_type_ids,
    ):
        assert add_special_tokens is False
        assert return_attention_mask is False
        assert return_token_type_ids is False
        return {"input_ids": list(range(len(text.split())))}


def _row(key: str, tokens: int) -> dict[str, object]:
    return {
        "key": key,
        "documents": 1,
        "characters": tokens * 4,
        "text_tokens": tokens - 1,
        "eos_tokens": 1,
        "training_tokens": tokens,
        "characters_per_text_token": 4.0,
    }


def _aggregates() -> dict[str, list[dict[str, object]]]:
    return {
        "roles": [
            _row("historical_general", 1_000),
            _row("historical_non_sonnet_poetry", 500),
            _row("nineteenth_century_bridge", 200),
            _row("standard_sonnets", 400),
        ],
        "v7_splits": [
            _row("train", 320),
            _row("validation", 40),
            _row("test", 40),
        ],
        "source_groups": [_row("fixture", 2_100)],
        "broader_works": [_row(f"work-{index}", 100) for index in range(17)],
        "broader_authors": [_row(f"author-{index}", 100) for index in range(8)],
        "sonnet_train_works": [_row(f"sonnet-work-{index}", 20) for index in range(20)],
        "sonnet_train_authors": [
            _row("dominant-author", 80),
            *[_row(f"sonnet-author-{index}", 20) for index in range(20)],
        ],
        "sonnet_train_epochs": [
            _row("16th_century", 180),
            *[_row(f"epoch-{index}", 60) for index in range(6)],
        ],
    }


def test_policy_freezes_approved_tokenizer_ratios_and_caps():
    policy = load_composition_policy(POLICY_PATH)

    assert policy["composition_version"] == COMPOSITION_VERSION
    assert policy["stages"]["stage_1_historical_general"]["components"] == {
        "historical_general": 0.85,
        "nineteenth_century_bridge": 0.1,
        "modern_preservation_replay": 0.05,
    }
    assert policy["concentration_ceilings"]["sonnet_author"] == 0.05
    assert policy["concentration_ceilings"]["sonnet_epoch"] == 0.3


def test_policy_rejects_stage_shares_that_do_not_sum_to_one(tmp_path):
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy["stages"]["stage_3_sonnets"]["components"][
        "standard_sonnets_v7_train"
    ] = 0.7
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ValueError, match="shares must sum to one"):
        load_composition_policy(path)


def test_policy_rejects_bridge_share_above_approved_ceiling(tmp_path):
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    stage = policy["stages"]["stage_1_historical_general"]["components"]
    stage["historical_general"] = 0.8
    stage["nineteenth_century_bridge"] = 0.15
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ValueError, match="bridge share exceeds"):
        load_composition_policy(path)


def test_token_count_disables_wrappers_and_requires_nonempty_result():
    tokenizer = FakeTokenizer()

    assert _count_text_tokens(tokenizer, "uno due tre") == 3
    with pytest.raises(ValueError, match="zero tokens"):
        _count_text_tokens(tokenizer, "")


def test_fallback_tokenizer_fingerprint_is_deterministic():
    tokenizer = FakeTokenizer()

    assert tokenizer_sha256(tokenizer) == tokenizer_sha256(tokenizer)
    assert len(tokenizer_sha256(tokenizer)) == 64


def test_gate_freezes_all_stage_components_and_detects_required_reweighting():
    policy = load_composition_policy(POLICY_PATH)

    gate = build_staged_composition_gate(
        aggregates=_aggregates(),
        replay=_row("replay", 100),
        policy=policy,
    )

    assert gate["pass"] is True
    assert gate["component_available_training_tokens"][
        "standard_sonnets_v7_train"
    ] == 320
    assert gate["component_available_training_tokens"][
        "stage_1_historical_replay"
    ] == 1_200
    assert gate["component_available_training_tokens"][
        "stage_2_historical_replay"
    ] == 1_700
    assert gate["concentration"]["sonnet_author"]["reweighting_required"] is True
    assert gate["concentration"]["sonnet_epoch"]["reweighting_required"] is True
    assert all(
        stage["shares_sum"] == pytest.approx(1.0)
        for stage in gate["stage_mixtures"]
    )


def test_gate_fails_when_a_required_role_or_v7_training_split_is_empty():
    policy = load_composition_policy(POLICY_PATH)
    aggregates = _aggregates()
    aggregates["roles"] = [
        row for row in aggregates["roles"] if row["key"] != "historical_general"
    ]
    aggregates["v7_splits"] = [_row("validation", 40), _row("test", 40)]

    gate = build_staged_composition_gate(
        aggregates=aggregates,
        replay=_row("replay", 100),
        policy=policy,
    )

    assert gate["pass"] is False
    assert gate["missing_roles"] == ["historical_general"]
    assert "standard_sonnets_v7_train" in gate["unavailable_components"]


def test_markdown_and_report_writers_are_deterministic(tmp_path):
    report = {
        "status": "pass",
        "totals": {
            "documents": 4,
            "characters": 2_000,
            "text_tokens": 496,
            "eos_tokens": 4,
            "training_tokens": 500,
        },
        "modern_preservation_replay": _row("replay", 100),
        "aggregates": _aggregates(),
        "composition_gate": build_staged_composition_gate(
            aggregates=_aggregates(),
            replay=_row("replay", 100),
            policy=load_composition_policy(POLICY_PATH),
        ),
    }
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    write_composition_reports(report, json_path, markdown_path)
    first = (json_path.read_bytes(), markdown_path.read_bytes())
    write_composition_reports(report, json_path, markdown_path)

    assert first == (json_path.read_bytes(), markdown_path.read_bytes())
    markdown = render_composition_markdown(report)
    assert "No token IDs or encoded training shards" in markdown
    assert "Instruction-following preservation remains" in markdown
    assert markdown.endswith("\n")


def test_real_report_freezes_exact_minerva_counts_and_gate_results():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    roles = {row["key"]: row for row in report["aggregates"]["roles"]}
    splits = {row["key"]: row for row in report["aggregates"]["v7_splits"]}

    assert report["status"] == "pass"
    assert report["totals"] == {
        "characters": 644_027_809,
        "characters_per_text_token": pytest.approx(3.6038097686256503),
        "documents": 26_934,
        "eos_tokens": 26_934,
        "text_tokens": 178_707_493,
        "training_tokens": 178_734_427,
    }
    assert {key: row["training_tokens"] for key, row in roles.items()} == {
        "historical_general": 58_105_538,
        "historical_non_sonnet_poetry": 18_877_139,
        "nineteenth_century_bridge": 97_763_895,
        "standard_sonnets": 3_987_855,
    }
    assert {key: row["training_tokens"] for key, row in splits.items()} == {
        "train": 3_551_021,
        "validation": 219_470,
        "test": 217_364,
    }
    assert report["modern_preservation_replay"]["training_tokens"] == 2_034_777
    assert report["tokenizer"]["serialized_sha256"] == (
        "11fbe803977e9d6dc1a50e6bb088be5b550f5e26da2a82fbfd7b41a045853a8c"
    )
    assert report["logical_unit_token_identity_sha256"] == (
        "8f33891e684dea673538c20822371da21c7de2207dff80e4a93850b2caea1772"
    )
    gate = report["composition_gate"]
    assert gate["pass"] is True
    assert gate["concentration"]["sonnet_author"]["reweighting_required"] is True
    assert gate["concentration"]["sonnet_epoch"]["reweighting_required"] is True
    assert gate["infeasible_caps"] == []


def test_real_reports_preserve_inactive_and_nonencoded_safety_boundary():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    serialized = REPORT_PATH.read_text(encoding="utf-8")

    assert report["activation_status"] == "inactive_pending_encoded_mixtures"
    assert report["verification"]["v7_validation_test_training_excluded"] is True
    assert report["verification"]["protected_v6_training_excluded"] is True
    assert report["verification"]["conditioned_material_included"] is False
    assert report["verification"]["token_ids_persisted"] is False
    assert report["verification"]["corpus_roles_activated"] is False
    assert report["verification"]["gpu_work_started"] is False
    assert "/home/" not in serialized
    assert "file://" not in serialized


def test_real_replay_count_is_bound_to_local_lineage_report():
    replay_path = ROOT / "data/local/minerva_7b_staged/replay_train.txt"
    lineage_path = ROOT / "data/local/minerva_7b_staged/replay_sample_report.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    public = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(replay_path.read_bytes()).hexdigest()

    assert lineage["sample_version"] == "paisa_even_byte_windows_v1"
    assert lineage["output_size_bytes"] == replay_path.stat().st_size
    assert lineage["output_sha256"] == digest
    assert public["provenance"]["replay_sample_sha256"] == digest
    assert public["provenance"]["replay_public"] is False
