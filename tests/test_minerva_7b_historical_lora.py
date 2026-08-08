import json

import pytest
import torch

from sonnet_training.minerva_7b_historical_lora import (
    Minerva7BHistoricalLoRAConfig,
    _stream_window,
    build_instruction_preservation_batches,
    build_historical_training_plan,
    historical_learning_rate,
    preservation_gate,
    select_lowest_qualifying_historical_row,
)


def test_historical_plan_uses_seven_to_one_replay_updates():
    config = Minerva7BHistoricalLoRAConfig()
    plan = build_historical_training_plan(
        config=config,
        historical_token_count=512 * 15 + 1,
        replay_token_count=512 * 3 + 1,
    )

    assert plan.historical_window_count == 15
    assert plan.replay_window_count == 3
    assert plan.updates_per_epoch == 3
    assert plan.planned_updates == 6
    assert plan.nominal_tokens_per_update == 4096


def test_historical_learning_rate_warms_up_and_decays():
    config = Minerva7BHistoricalLoRAConfig()
    plan = build_historical_training_plan(
        config=config,
        historical_token_count=512 * 700 + 1,
        replay_token_count=512 * 10 + 1,
    )

    assert historical_learning_rate(config=config, plan=plan, step=1) < config.learning_rate
    assert historical_learning_rate(
        config=config, plan=plan, step=plan.warmup_steps
    ) == pytest.approx(config.learning_rate)
    assert historical_learning_rate(
        config=config, plan=plan, step=plan.planned_updates
    ) == pytest.approx(config.min_learning_rate)


def test_preservation_gate_enforces_both_relative_limits():
    config = Minerva7BHistoricalLoRAConfig()

    assert preservation_gate(
        modern_loss=2.05,
        instruction_loss=3.2,
        baseline_modern_loss=2.0,
        baseline_instruction_loss=3.0,
        config=config,
    )
    assert not preservation_gate(
        modern_loss=2.2,
        instruction_loss=3.2,
        baseline_modern_loss=2.0,
        baseline_instruction_loss=3.0,
        config=config,
    )
    assert not preservation_gate(
        modern_loss=2.05,
        instruction_loss=3.31,
        baseline_modern_loss=2.0,
        baseline_instruction_loss=3.0,
        config=config,
    )


def test_hugging_face_causal_labels_match_inputs_before_internal_shift():
    stream = torch.arange(10, dtype=torch.int32)

    input_ids, labels = _stream_window(
        stream,
        window_index=0,
        context_length=4,
        device=torch.device("cpu"),
    )

    assert input_ids.tolist() == [[0, 1, 2, 3]]
    assert labels.tolist() == input_ids.tolist()


def test_instruction_preservation_masks_chat_prompt(tmp_path):
    class FakeChatTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2] if len(messages) == 1 else [1, 2, 3, 4]

    path = tmp_path / "prompts.json"
    path.write_text(json.dumps([{"prompt": "Domanda", "response": "Risposta"}]))

    batches = build_instruction_preservation_batches(
        tokenizer=FakeChatTokenizer(),
        path=path,
        context_length=8,
    )

    assert batches[0][0].tolist() == [[1, 2, 3, 4]]
    assert batches[0][1].tolist() == [[-100, -100, 3, 4]]


def test_stage_a_final_selection_uses_lowest_qualifying_loss():
    config = Minerva7BHistoricalLoRAConfig()
    baseline = {"historical_validation_loss": 3.3}
    history = [
        {"step": 3000, "historical_validation_loss": 3.19, "preservation_gate_passed": True},
        {"step": 3381, "historical_validation_loss": 3.18, "preservation_gate_passed": True},
        {"step": 4000, "historical_validation_loss": 3.17, "preservation_gate_passed": False},
    ]

    selected = select_lowest_qualifying_historical_row(
        history=history,
        baseline_metrics=baseline,
        config=config,
    )

    assert selected is history[1]
