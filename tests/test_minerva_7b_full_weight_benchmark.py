from dataclasses import replace

import pytest

from sonnet_training.minerva_7b_full_weight_benchmark import (
    Minerva7BFullWeightBenchmarkConfig,
    build_throughput_candidates,
    project_full_run,
    select_fastest_fit_candidate,
    validate_full_weight_benchmark_config,
)


def test_full_weight_benchmark_builds_fixed_effective_batch_candidates():
    config = Minerva7BFullWeightBenchmarkConfig()

    candidates = build_throughput_candidates(config)

    assert len(candidates) == 8
    assert [row.microbatch_size for row in candidates[:4]] == [1, 2, 4, 8]
    assert [row.gradient_accumulation_steps for row in candidates[:4]] == [8, 4, 2, 1]
    assert all(row.tokens_per_update == 4096 for row in candidates)
    assert candidates[0].candidate_id == "gc_on_micro1"
    assert candidates[-1].candidate_id == "gc_off_micro8"


def test_full_weight_benchmark_recipe_is_locked():
    config = Minerva7BFullWeightBenchmarkConfig()

    validate_full_weight_benchmark_config(config)

    with pytest.raises(ValueError, match="locked"):
        validate_full_weight_benchmark_config(
            replace(config, timed_updates=2)
        )


def test_projection_and_selection_use_fastest_fitting_candidate():
    config = Minerva7BFullWeightBenchmarkConfig()
    projection = project_full_run(tokens_per_second=4000.0, config=config)
    rows = [
        {"candidate_id": "slow", "fit_decision": "pass", "tokens_per_second": 2000.0},
        {"candidate_id": "fast", "fit_decision": "pass", "tokens_per_second": 4000.0},
        {"candidate_id": "oom", "fit_decision": "reject", "tokens_per_second": None},
    ]

    selected = select_fastest_fit_candidate(rows)

    assert selected["candidate_id"] == "fast"
    assert projection["update_only_hours"] == pytest.approx(24.3938400694)
    assert projection["projected_hours_with_overhead"] == pytest.approx(28.0529160798)
    assert projection["projected_cost_usd"] == pytest.approx(59.4721820892)
