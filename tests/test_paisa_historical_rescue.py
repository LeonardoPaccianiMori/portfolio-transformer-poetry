from pathlib import Path

import pytest

from sonnet_training.paisa_historical_rescue import (
    RESCUE_TRAINING_PLAN_MARKDOWN_PATH,
)
from sonnet_training.paisa_historical_rescue import build_rescue_stage_config
from sonnet_training.paisa_historical_rescue import build_rescue_training_plan
from sonnet_training.paisa_historical_rescue import build_rescue_training_plan_markdown


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_rescue_plan_locks_the_measured_capped_stage_budgets():
    plan = build_rescue_training_plan(REPO_ROOT)
    modern, historical = plan.stages

    assert plan.tokens_per_optimizer_update == 4_096
    assert plan.architecture["microbatch_size"] == 4
    assert plan.architecture["gradient_accumulation_steps"] == 2
    assert modern.train_steps == 412_998
    assert modern.stable_steps == 330_398
    assert modern.unused_target_token_budget == 1_527
    assert historical.train_steps == 56_213
    assert historical.stable_steps == 44_970
    assert historical.unused_target_token_budget == 3_316
    assert modern.train_steps * plan.tokens_per_optimizer_update <= (
        modern.train_tokens * modern.max_passes
    )
    assert historical.train_steps * plan.tokens_per_optimizer_update <= (
        historical.train_tokens * historical.max_passes
    )


def test_rescue_stage_configs_preserve_the_approved_stage_boundary():
    plan = build_rescue_training_plan(REPO_ROOT)
    modern = build_rescue_stage_config(
        plan=plan,
        stage_id="modern_italian_pretraining",
        device="cuda:0",
    )
    historical = build_rescue_stage_config(
        plan=plan,
        stage_id="historical_italian_annealing",
        device="cuda:0",
        historical_parent_checkpoint_path="runs/modern/best_validation.pt",
    )

    assert modern.train_split_id == "paisa_train"
    assert modern.learning_rate_schedule == "warmup_stable_cosine"
    assert modern.validation_mode == "random_batches"
    assert modern.eval_batches == 20
    assert modern.initialization_checkpoint_path == ""
    assert historical.train_split_id == "historical_train"
    assert historical.validation_mode == "sequential_windows"
    assert historical.initialization_checkpoint_path == "runs/modern/best_validation.pt"
    assert historical.checkpoint_retention == "latest_only"


def test_historical_rescue_stage_requires_the_modern_parent_checkpoint():
    plan = build_rescue_training_plan(REPO_ROOT)

    with pytest.raises(ValueError, match="requires a PAISA"):
        build_rescue_stage_config(
            plan=plan,
            stage_id="historical_italian_annealing",
            device="cuda:0",
        )


def test_rescue_training_plan_markdown_records_runtime_and_hardware_choice():
    plan = build_rescue_training_plan(REPO_ROOT)

    markdown = build_rescue_training_plan_markdown(plan)

    assert "rescue_upper_micro4" in markdown
    assert "412,998" in markdown
    assert "56,213" in markdown
    assert "67.4 hours" in markdown
    assert str(RESCUE_TRAINING_PLAN_MARKDOWN_PATH) not in markdown
