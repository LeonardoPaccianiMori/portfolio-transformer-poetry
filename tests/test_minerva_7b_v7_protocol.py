import json
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from sonnet_training.minerva_7b_v7_protocol import (
    FullWeightProtocolConfig,
    abort_reasons,
    build_hardware_candidates,
    build_modern_preservation_index,
    candidate_passes_gates,
    learning_rate_at_update,
    load_full_weight_protocol,
    prepare_full_weight_protocol,
    project_all_in_cost,
    select_stage_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/minerva_7b_v7_full_weight_protocol.json"


def _protocol():
    return load_full_weight_protocol(PROTOCOL_PATH, ROOT)


def test_protocol_pins_exact_stage_updates_context_and_authorization_boundary():
    protocol = _protocol()

    assert protocol["data"]["context_length"] == 2048
    assert protocol["data"]["global_windows_per_update"] == 16
    assert protocol["data"]["global_target_tokens_per_update"] == 32_768
    assert [row["optimizer_updates"] for row in protocol["stages"]] == [
        2_065,
        760,
        135,
    ]
    assert sum(row["optimizer_updates"] for row in protocol["stages"]) == 2_960
    assert protocol["cost"]["all_in_spend_ceiling"] == 60.0
    assert protocol["cost"]["maximum_projected_all_in_cost_to_launch"] == 48.0
    assert protocol["authorization"] == {
        "cache_deletion_authorized": False,
        "gpu_benchmark_authorized": False,
        "gpu_rental_authorized": False,
        "long_training_authorized": False,
        "protocol_design_approved": True,
    }


def test_hardware_candidates_keep_exact_global_batch_and_context():
    candidates = build_hardware_candidates(_protocol())

    assert len(candidates) == 12
    assert {row.local_microbatch_size for row in candidates} == {1, 2, 4}
    assert {row.gradient_accumulation_steps for row in candidates} == {8, 4, 2}
    assert {row.global_windows_per_update for row in candidates} == {16}
    assert {row.global_target_tokens_per_update for row in candidates} == {32_768}
    assert {
        row.execution_mode for row in candidates
    } == {"torch_compile_default", "eager"}


def test_learning_rates_warm_up_and_decay_to_exact_stage_minimum():
    protocol = _protocol()
    for stage in protocol["stages"]:
        assert learning_rate_at_update(stage, 1) < stage["peak_learning_rate"]
        assert learning_rate_at_update(
            stage, stage["warmup_updates"]
        ) == pytest.approx(stage["peak_learning_rate"])
        assert learning_rate_at_update(
            stage, stage["optimizer_updates"]
        ) == pytest.approx(stage["minimum_learning_rate"])


def test_stage_checkpoint_requires_improvement_retention_and_preservation():
    protocol = _protocol()
    stage = protocol["stages"][1]
    baseline = {
        "historical_non_sonnet_poetry_loss": 3.0,
        "stage_1_historical_general_bridge_token_weighted_loss": 2.0,
        "modern_validation_loss": 2.0,
        "instruction_validation_loss": 3.0,
    }
    passing = {
        "update": 50,
        "historical_non_sonnet_poetry_loss": 2.98,
        "stage_1_historical_general_bridge_token_weighted_loss": 2.03,
        "modern_validation_loss": 2.09,
        "instruction_validation_loss": 3.29,
    }

    assert candidate_passes_gates(
        stage=stage,
        metrics=passing,
        baseline_metrics=baseline,
        preservation=protocol["preservation"],
    )
    better_primary_but_failed_retention = {
        **passing,
        "update": 100,
        "historical_non_sonnet_poetry_loss": 2.9,
        "stage_1_historical_general_bridge_token_weighted_loss": 2.05,
    }
    selected = select_stage_checkpoint(
        stage=stage,
        history=[passing, better_primary_but_failed_retention],
        baseline_metrics=baseline,
        preservation=protocol["preservation"],
    )
    assert selected is passing


def test_abort_rules_detect_sustained_loss_gradient_preservation_and_cost():
    protocol = _protocol()
    rows = [
        {"loss": 1.0, "gradient_norm": 1.0} for _ in range(20)
    ] + [
        {"loss": 2.1, "gradient_norm": 101.0} for _ in range(3)
    ]

    reasons = abort_reasons(
        protocol=protocol,
        recent_updates=rows,
        consecutive_preservation_failures=2,
        projected_or_spent_cost_usd=60.0,
    )

    assert set(reasons) == {
        "spend_ceiling_reached",
        "sustained_gradient_norm_limit",
        "sustained_training_loss_spike",
        "repeated_preservation_gate_failure",
    }


def test_cost_projection_requires_twenty_percent_launch_contingency():
    projection = project_all_in_cost(
        training_tokens=96_993_280,
        measured_tokens_per_second=10_000.0,
        hourly_rate_usd=3.495,
        protocol=_protocol(),
    )

    assert projection["projected_all_in_cost_usd"] == pytest.approx(
        11.7705386667
    )
    assert projection["launch_limit_usd"] == 48.0
    assert projection["passes_launch_gate"] is True


def test_analysis_snapshots_and_activation_probe_contract_are_frozen():
    protocol = _protocol()
    snapshots = protocol["analysis_snapshots"]
    activations = protocol["activation_analysis_preservation"]

    assert snapshots["midpoint_updates"] == {
        "stage_1_historical_general": 1_033,
        "stage_2_non_sonnet_poetry": 380,
        "stage_3_sonnets": 68,
    }
    assert snapshots["estimated_new_snapshot_count"] == 6
    assert activations["required_before_long_run"] is True
    assert len(activations["model_states"]) == 7
    assert set(activations["probe_domains"]) == {
        "modern_instruction",
        "historical_general",
        "historical_non_sonnet_poetry",
        "standard_sonnet",
    }
    assert activations["do_not_capture_every_training_batch"] is True


@pytest.mark.local_artifact
def test_modern_preservation_index_is_deterministic_and_spans_are_complete(tmp_path):
    protocol = _protocol()
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    first_report = build_modern_preservation_index(
        protocol=protocol,
        repo_root=ROOT,
        modern_encoded_dir=ROOT / "data/local/minerva_7b_full_weight/encoded",
        output_path=first,
    )
    second_report = build_modern_preservation_index(
        protocol=protocol,
        repo_root=ROOT,
        modern_encoded_dir=ROOT / "data/local/minerva_7b_full_weight/encoded",
        output_path=second,
    )

    rows = [json.loads(line) for line in first.read_text().splitlines()]
    assert first.read_bytes() == second.read_bytes()
    assert first_report == second_report
    assert first_report["selected_windows"] == 128
    assert first_report["target_tokens"] == 128 * 2048
    assert first_report["first_candidate_index"] == 0
    assert first_report["last_candidate_index"] == 1696
    assert all(
        sum(piece["token_count"] for piece in row["source_slices"]) == 2049
        for row in rows
    )


@pytest.mark.local_artifact
def test_prepare_protocol_writes_public_aggregate_report_and_local_index(tmp_path):
    config = FullWeightProtocolConfig(
        repo_root=ROOT,
        protocol_path=PROTOCOL_PATH,
        modern_encoded_dir=ROOT / "data/local/minerva_7b_full_weight/encoded",
        preservation_index_path=tmp_path / "local" / "preservation.jsonl",
        json_report_path=tmp_path / "report.json",
        markdown_report_path=tmp_path / "report.md",
    )

    report = prepare_full_weight_protocol(config)

    assert report["status"] == "frozen_verified_gpu_unauthorized"
    assert report["training"]["total_optimizer_updates"] == 2_960
    assert report["hardware_qualification"]["candidate_count"] == 12
    assert report["verification"]["gpu_work_started"] is False
    assert report["verification"]["cache_deleted"] is False
    assert "token IDs" not in config.json_report_path.read_text()
    assert config.preservation_index_path.exists()


def test_completed_checkpoint_artifacts_freeze_protocol_counts_and_hashes():
    report_path = ROOT / "reports/minerva_7b_v7_full_weight_protocol_v1.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["status"] == "frozen_verified_gpu_unauthorized"
    assert report["protocol_sha256"] == (
        "4c24868febe95ea064176fd7868a5515d8bf121f521401f33b3374e1c1736c8a"
    )
    assert report["training"]["total_windows"] == 47_360
    assert report["training"]["total_target_tokens"] == 96_993_280
    assert report["training"]["total_optimizer_updates"] == 2_960
    assert report["preservation"]["local_index"]["index_sha256"] == (
        "01299b685515fffdd5d3a1ec00204d2038d1639365efb8a2e0e2575e0e3fc582"
    )
    assert report["analysis_snapshots"]["estimated_new_snapshot_count"] == 6
    assert len(report["activation_analysis_preservation"]["model_states"]) == 7
    assert report["hardware_qualification"]["candidate_count"] == 12
    assert report["authorization"]["gpu_benchmark_authorized"] is False
    assert report["authorization"]["long_training_authorized"] is False
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == (
        "b4e50dcd0ca9584beb334a4b6593c39584253c03cddb47e55e3df20197f0bd70"
    )
