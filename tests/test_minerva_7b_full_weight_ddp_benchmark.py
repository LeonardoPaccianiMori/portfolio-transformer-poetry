from dataclasses import replace

import pytest

from sonnet_training.minerva_7b_full_weight_ddp_benchmark import (
    H100_BENCHMARK_VERSION,
    H100_INTERMEDIATE_BENCHMARK_VERSION,
    Minerva7BDualH100DdpBenchmarkConfig,
    Minerva7BDualH100IntermediateDdpBenchmarkConfig,
    Minerva7BFullWeightDdpBenchmarkConfig,
    build_ddp_throughput_candidates,
    project_distributed_full_run,
    select_fastest_ddp_candidate,
    validate_full_weight_ddp_benchmark_config,
)


def test_ddp_benchmark_builds_two_fixed_global_batch_groups():
    config = Minerva7BFullWeightDdpBenchmarkConfig()

    candidates = build_ddp_throughput_candidates(config)

    assert len(candidates) == 6
    assert [row.global_sequences_per_update for row in candidates] == [
        8, 8, 8, 16, 16, 16,
    ]
    assert [row.local_microbatch_size for row in candidates] == [
        4, 4, 4, 8, 8, 8,
    ]
    assert [row.bucket_cap_mib for row in candidates] == [25, 100, 250] * 2
    assert candidates[0].candidate_id == "global4096_micro4_bucket25"
    assert candidates[-1].candidate_id == "global8192_micro8_bucket250"
    assert {row.tokens_per_update for row in candidates} == {4096, 8192}


def test_ddp_benchmark_recipe_is_locked():
    config = Minerva7BFullWeightDdpBenchmarkConfig()

    validate_full_weight_ddp_benchmark_config(config)

    with pytest.raises(ValueError, match="locked"):
        validate_full_weight_ddp_benchmark_config(
            replace(config, timed_updates=2)
        )


def test_dual_h100_profile_is_separately_locked():
    config = Minerva7BDualH100DdpBenchmarkConfig()

    validate_full_weight_ddp_benchmark_config(config)

    assert config.benchmark_version == H100_BENCHMARK_VERSION
    assert config.expected_gpu_name_substring == "h100 80gb hbm3"
    assert config.minimum_total_memory_mib == 75 * 1024
    assert config.minimum_communication_gigabytes_per_second == 100.0
    assert config.hourly_rate_usd == 3.495
    with pytest.raises(ValueError, match="locked"):
        validate_full_weight_ddp_benchmark_config(
            replace(config, hourly_rate_usd=3.0)
        )


def test_dual_h100_intermediate_profile_tests_three_local_batches():
    config = Minerva7BDualH100IntermediateDdpBenchmarkConfig()

    validate_full_weight_ddp_benchmark_config(config)
    candidates = build_ddp_throughput_candidates(config)

    assert config.benchmark_version == H100_INTERMEDIATE_BENCHMARK_VERSION
    assert config.global_sequence_counts == (10, 12, 14)
    assert config.bucket_cap_mib == (25, 250)
    assert len(candidates) == 6
    assert [row.local_microbatch_size for row in candidates] == [5, 5, 6, 6, 7, 7]
    assert {row.tokens_per_update for row in candidates} == {5120, 6144, 7168}


def test_ddp_projection_and_selection_use_fastest_fitting_candidate():
    config = Minerva7BFullWeightDdpBenchmarkConfig()
    projection = project_distributed_full_run(
        tokens_per_second=8000.0,
        config=config,
    )
    rows = [
        {"candidate_id": "slow", "fit_decision": "pass", "tokens_per_second": 6000.0},
        {"candidate_id": "fast", "fit_decision": "pass", "tokens_per_second": 8000.0},
        {"candidate_id": "reject", "fit_decision": "reject", "tokens_per_second": 9000.0},
    ]

    selected = select_fastest_ddp_candidate(rows)

    assert selected["candidate_id"] == "fast"
    assert projection["update_only_hours"] == pytest.approx(12.1969200347)
    assert projection["projected_hours_with_overhead"] == pytest.approx(14.0264580399)
    assert projection["projected_cost_usd"] == pytest.approx(30.3252022823)
