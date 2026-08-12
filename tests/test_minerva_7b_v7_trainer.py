import json
import random
from pathlib import Path

import pytest
import torch
from types import SimpleNamespace

from sonnet_training.minerva_7b_v7_trainer import (
    apply_stage_learning_rate,
    capture_local_rng_state,
    checkpoint_metadata,
    compose_validation_metrics,
    evaluate_all_gates,
    qualification_runtime_candidates,
    qualification_result_passes,
    record_stage_evaluation,
    restore_local_rng_state,
    runtime_candidate_from_environment,
    shifted_causal_loss,
    should_evaluate,
    should_save_analysis_snapshot,
    should_save_resume,
    stage_global_update,
    validate_long_run_authorization,
)


ROOT = Path(__file__).resolve().parents[1]


def _protocol():
    return json.loads(
        (ROOT / "configs/minerva_7b_v7_full_weight_protocol.json").read_text()
    )


def test_shifted_causal_loss_scores_exact_supplied_targets():
    logits = torch.tensor(
        [[[10.0, 0.0], [0.0, 10.0], [10.0, 0.0]]], dtype=torch.float32
    )
    targets = torch.tensor([[0, 1, 0]])

    assert shifted_causal_loss(logits, targets).item() < 0.001


def test_stage_global_updates_are_contiguous_across_resets():
    protocol = _protocol()
    assert stage_global_update(protocol, "stage_1_historical_general", 2065) == 2065
    assert stage_global_update(protocol, "stage_2_non_sonnet_poetry", 1) == 2066
    assert stage_global_update(protocol, "stage_3_sonnets", 135) == 2960


def test_evaluation_resume_and_analysis_boundaries_are_frozen():
    stage = _protocol()["stages"][2]

    assert should_evaluate(stage, 15)
    assert should_evaluate(stage, 135)
    assert not should_evaluate(stage, 14)
    assert should_save_resume(stage, 30)
    assert should_save_analysis_snapshot(stage=stage, update=68, midpoint_update=68)
    assert should_save_analysis_snapshot(stage=stage, update=135, midpoint_update=68)


def test_learning_rate_is_applied_to_every_optimizer_group():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD(
        [{"params": [parameter], "lr": 0.0}, {"params": [], "lr": 0.0}]
    )
    stage = _protocol()["stages"][0]

    value = apply_stage_learning_rate(optimizer, stage, 62)

    assert value == pytest.approx(1e-5)
    assert {group["lr"] for group in optimizer.param_groups} == {value}


def test_runtime_candidate_requires_exact_global_batch(monkeypatch):
    monkeypatch.setenv("V7_LOCAL_MICROBATCH_SIZE", "2")
    monkeypatch.setenv("V7_GRADIENT_ACCUMULATION_STEPS", "4")
    monkeypatch.setenv("V7_GRADIENT_CHECKPOINTING", "true")
    monkeypatch.setenv("V7_EXECUTION_MODE", "eager")

    candidate = runtime_candidate_from_environment()

    assert candidate.local_microbatch_size == 2
    assert candidate.gradient_accumulation_steps == 4
    assert candidate.gradient_checkpointing


def test_long_run_guard_rejects_checkpoint_8f_execution():
    execution = json.loads(
        (ROOT / "configs/minerva_7b_v7_execution.json").read_text()
    )
    with pytest.raises(PermissionError, match="unauthorized"):
        validate_long_run_authorization(execution)


def test_composite_validation_metrics_are_target_weighted():
    rows = {
        "validation_historical_general": {"target_tokens": 1, "loss": 1.0},
        "validation_historical_non_sonnet_poetry": {
            "target_tokens": 2,
            "loss": 2.0,
        },
        "validation_nineteenth_century_bridge": {
            "target_tokens": 3,
            "loss": 3.0,
        },
        "sonnets_validation": {"target_tokens": 4, "loss": 4.0},
    }

    metrics = compose_validation_metrics(rows)

    assert metrics["historical_general_bridge_token_weighted_loss"] == 2.5
    assert metrics["all_broader_validation_token_weighted_loss"] == pytest.approx(
        14 / 6
    )


def test_qualification_matrix_contains_exact_twelve_global_batch_candidates():
    candidates = qualification_runtime_candidates(_protocol())
    assert len(candidates) == 12
    assert {
        row.local_microbatch_size * row.gradient_accumulation_steps * 2
        for row in candidates
    } == {16}


def test_stage_evaluation_records_gate_and_current_selection():
    protocol = _protocol()
    stage = protocol["stages"][0]
    baseline = {
        "historical_general_bridge_token_weighted_loss": 3.0,
        "modern_validation_loss": 2.0,
        "instruction_validation_loss": 2.0,
    }
    history = []

    row = record_stage_evaluation(
        stage=stage,
        stage_update=100,
        evaluation={
            "metrics": {
                "historical_general_bridge_token_weighted_loss": 2.99,
                "modern_validation_loss": 2.01,
                "instruction_validation_loss": 2.01,
            }
        },
        baseline_metrics=baseline,
        preservation=protocol["preservation"],
        history=history,
    )

    assert row["passes_all_gates"]
    assert row["is_current_selected_candidate"]


def test_checkpoint_metadata_contains_exact_resume_and_analysis_state():
    protocol = _protocol()
    metadata = checkpoint_metadata(
        protocol=protocol,
        stage_id="stage_1_historical_general",
        stage_update=100,
        global_update=100,
        next_stage_window_index=1600,
        next_window_identity_sha256="a" * 64,
        next_learning_rate=1e-5,
        world_size=2,
        git_commit="deadbeef",
        package_versions={"torch": "x"},
        hardware_topology={"gpu": "h100"},
        validation_history=[{"update": 100}],
        protocol_sha256="e" * 64,
        parent_baseline_metrics={"modern_validation_loss": 2.0},
        stage_start_metrics={"historical_general_bridge_token_weighted_loss": 3.0},
        recent_updates=[{"loss": 1.0, "gradient_norm": 2.0}],
        preservation_failures=1,
    )

    assert metadata["next_stage_window_index"] == 1600
    assert metadata["protocol_sha256"] == "e" * 64
    assert metadata["parent_baseline_metrics"]["modern_validation_loss"] == 2.0
    assert metadata["recent_updates"][0]["gradient_norm"] == 2.0
    assert metadata["preservation_failures"] == 1


def test_qualification_result_requires_validation_and_resume_proofs():
    protocol = _protocol()
    rank_metrics = [
        {
            "mean_loss": 2.0,
            "mean_gradient_norm": 3.0,
            "reserved_headroom_mib": 9000.0,
        },
        {
            "mean_loss": 2.1,
            "mean_gradient_norm": 3.1,
            "reserved_headroom_mib": 9000.0,
        },
    ]
    assert qualification_result_passes(
        rank_metrics=rank_metrics,
        measured_nccl_gbps=101.0,
        projected_cost_usd=47.0,
        validation_transition_passes=True,
        resume_proof_passes=True,
        protocol=protocol,
    )
    assert not qualification_result_passes(
        rank_metrics=rank_metrics,
        measured_nccl_gbps=101.0,
        projected_cost_usd=47.0,
        validation_transition_passes=True,
        resume_proof_passes=False,
        protocol=protocol,
    )


def test_all_gate_evaluation_returns_every_metric_with_fake_cpu_model():
    class Reader:
        def __init__(self, pool_ids):
            self.pool_ids = pool_ids

        def rows(self, index_id):
            assert index_id in self.pool_ids
            return ({"target_tokens": 3, "source": (0, 1, 2, 3)},)

        def source_tokens(self, row):
            return row["source"]

    class Model(torch.nn.Module):
        def forward(self, input_ids, use_cache=False):
            vocabulary = 16
            logits = torch.zeros(*input_ids.shape, vocabulary)
            return SimpleNamespace(logits=logits)

    class Tokenizer:
        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt
        ):
            if add_generation_prompt:
                return [0, 1, 2]
            return [0, 1, 2, 3, 4]

    broader = Reader(
        {
            "validation_historical_general",
            "validation_historical_non_sonnet_poetry",
            "validation_nineteenth_century_bridge",
            "sonnets_validation",
        }
    )
    modern = Reader({"modern_preservation_validation_v1"})

    result = evaluate_all_gates(
        model=Model(),
        reader=broader,
        modern_reader=modern,
        tokenizer=Tokenizer(),
        prompts=[{"id": "x", "prompt": "p", "response": "r"}],
        device=torch.device("cpu"),
    )

    assert set(result["metrics"]) == {
        "historical_general_bridge_token_weighted_loss",
        "stage_1_historical_general_bridge_token_weighted_loss",
        "historical_non_sonnet_poetry_loss",
        "v7_sonnet_validation_loss",
        "all_broader_validation_token_weighted_loss",
        "modern_validation_loss",
        "instruction_validation_loss",
    }
    assert result["instruction"]["rows"][0]["target_tokens"] == 2


def test_cpu_rng_capture_and_restore_reproduces_python_and_torch_streams():
    random.seed(123)
    torch.manual_seed(123)
    state = capture_local_rng_state()
    expected = (random.random(), torch.rand(1).item())
    random.random()
    torch.rand(1)

    restore_local_rng_state(state)

    assert (random.random(), torch.rand(1).item()) == expected
