import torch
import pytest

from sonnet_training.minerva_v7_ai_dpo import (
    DPOExample,
    build_training_plan,
    dpo_loss,
    learning_rate_for_step,
    sequence_response_logps,
    split_examples_by_prompt,
)


def _example(index: int, prompt: str) -> DPOExample:
    return DPOExample(
        pair_id=f"pair_{index}", pair_type="literary", prompt_id=prompt,
        opening_line="Prima", chosen="Prima\nScelta.", rejected="Prima\nRespinta.",
        vote_counts={"A": 2, "B": 1, "tie": 0},
    )


def test_split_examples_keeps_prompt_groups_disjoint_and_reproducible():
    rows = [_example(index, f"p{index // 2}") for index in range(20)]
    first = split_examples_by_prompt(rows, validation_fraction=0.2, seed=7)
    second = split_examples_by_prompt(rows, validation_fraction=0.2, seed=7)
    assert first == second
    train, validation = first
    assert {row.prompt_id for row in train}.isdisjoint(
        {row.prompt_id for row in validation}
    )
    assert len(train) + len(validation) == 20


def test_sequence_response_logps_masks_prompt_and_padding():
    input_ids = torch.tensor([[0, 1, 2, 3], [0, 1, 3, 0]])
    mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 0]], dtype=torch.bool)
    logits = torch.zeros(2, 4, 4)
    result = sequence_response_logps(logits, input_ids, mask)
    log_uniform = -torch.log(torch.tensor(4.0))
    assert torch.allclose(result, torch.tensor([2 * log_uniform, log_uniform]))


def test_dpo_loss_rewards_policy_margin_beyond_reference():
    good_loss, good = dpo_loss(
        policy_logps=torch.tensor([-1.0, -3.0]),
        reference_logps=torch.tensor([-2.0, -3.0]), beta=1.0,
    )
    bad_loss, bad = dpo_loss(
        policy_logps=torch.tensor([-3.0, -1.0]),
        reference_logps=torch.tensor([-2.0, -3.0]), beta=1.0,
    )
    assert good_loss < bad_loss
    assert good["preference_accuracy"] == 1.0
    assert bad["preference_accuracy"] == 0.0


def test_learning_rate_warms_up_and_decays_to_floor():
    config = {
        "warmup_fraction": 0.1, "learning_rate": 1e-5,
        "minimum_learning_rate": 1e-6,
    }
    assert learning_rate_for_step(config, step=1, total_steps=100) == pytest.approx(1e-6)
    assert learning_rate_for_step(config, step=10, total_steps=100) == pytest.approx(1e-5)
    assert learning_rate_for_step(config, step=100, total_steps=100) == pytest.approx(1e-6)


def test_training_plan_has_deterministic_complete_order_and_prompt_disjoint_split():
    rows = [_example(index, f"p{index // 2}") for index in range(40)]
    config = {
        "validation_fraction": 0.1,
        "split_seed": 11,
        "training_seed": 13,
        "gradient_accumulation_steps": 8,
    }
    first = build_training_plan(rows, config=config)
    second = build_training_plan(rows, config=config)
    assert first == second
    assert sorted(first["training_order"]) == list(range(len(first["train"])))
    assert first["total_steps"] == 5
    split = first["split_manifest"]
    assert set(split["train_prompt_ids"]).isdisjoint(split["validation_prompt_ids"])
    assert split["v7_test_accessed"] is False
