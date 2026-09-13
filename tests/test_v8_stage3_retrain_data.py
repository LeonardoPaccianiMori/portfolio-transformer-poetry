import pytest

from sonnet_training.v8_stage3_retrain_data import (
    build_window_plan,
    plan_summary,
    replay_every,
    replay_token_target,
)


def test_replay_token_target():
    assert replay_token_target(95, 0.05) == 5
    with pytest.raises(ValueError):
        replay_token_target(0, 0.05)
    with pytest.raises(ValueError):
        replay_token_target(100, 1.0)


def test_replay_every_for_five_percent():
    assert replay_every(1416, 0.05) == 19


def test_build_window_plan_is_deterministic_and_uses_five_percent_replay():
    plan = build_window_plan(100 * 2048, 100 * 2048, context_length=2048, fraction=0.05)
    again = build_window_plan(100 * 2048, 100 * 2048, context_length=2048, fraction=0.05)
    assert plan == again
    kinds = [row["kind"] for row in plan]
    assert kinds.count("train") == 99
    assert kinds.count("replay") == 5
    summary = plan_summary(plan, windows_per_update=16)
    assert summary["total_windows"] == 104
    assert summary["train_windows"] == 99
    assert summary["replay_windows"] == 5
    assert summary["optimizer_updates"] == 6
    assert summary["trailing_windows_dropped"] == 8
    assert abs(summary["replay_share"] - 5 / 104) < 1e-12


def test_build_window_plan_drops_a_window_without_a_lookahead_token():
    plan = build_window_plan(2 * 2048, 10 * 2048, context_length=2048, fraction=0.5)
    train = [row for row in plan if row["kind"] == "train"]
    assert len(train) == 1
