import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import pytest

from sonnet_training import minerva_7b_v7_gpu_qualification as gpu_qualification
from sonnet_training.minerva_7b_v7_qualification import (
    build_qualification_candidates,
    candidate_by_id,
    load_hardware_qualification,
    preliminary_gate_reasons,
    project_candidate_cost,
    project_stage_costs,
    qualification_gate_reasons,
    select_preliminary_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/minerva_7b_v7_hardware_qualification.json"
REPORT_PATH = ROOT / "reports/minerva_7b_v7_dual_a6000_qualification_v1.json"


def _config():
    return load_hardware_qualification(CONFIG_PATH, ROOT)


def test_qualification_pins_a6000_primary_and_single_h100_fallback():
    config = _config()

    assert config["primary_profile"]["profile_id"] == "dual_rtx_a6000_ddp"
    assert config["primary_profile"]["world_size"] == 2
    assert config["primary_profile"]["minimum_memory_mib_per_gpu"] == 48_000
    assert config["primary_profile"]["minimum_peak_reserved_headroom_mib"] == 8_192
    assert config["fallback_profile"]["profile_id"] == "single_h100_sxm"
    assert config["cost"]["hourly_rate_usd"] == 1.008
    assert config["cost"]["maximum_projected_all_in_cost_to_launch_usd"] == 48.0
    assert config["cost"]["absolute_spend_ceiling_usd"] == 60.0
    assert config["authorization"]["gpu_qualification_authorized"] is True
    assert config["authorization"]["long_training_authorized"] is False


def test_a6000_matrix_has_exact_eight_candidates_and_preserves_batch():
    candidates = build_qualification_candidates(_config())

    assert len(candidates) == 8
    assert {row.local_microbatch_size for row in candidates} == {1, 2}
    assert {row.gradient_accumulation_steps for row in candidates} == {8, 4}
    assert {row.gradient_checkpointing for row in candidates} == {True, False}
    assert {row.execution_mode for row in candidates} == {
        "eager",
        "torch_compile_default",
    }
    assert {
        row.local_microbatch_size * row.gradient_accumulation_steps * 2
        for row in candidates
    } == {16}
    assert {row.global_target_tokens_per_update for row in candidates} == {32_768}


def test_candidate_lookup_rejects_unapproved_runtime():
    config = _config()
    selected = candidate_by_id(
        config, "a6000_context2048_micro1_accum8_gc_on_eager"
    )
    assert selected.local_microbatch_size == 1
    with pytest.raises(KeyError, match="unknown qualification candidate"):
        candidate_by_id(config, "micro8_unapproved")


def test_cost_projection_applies_exact_token_budget_and_contingency():
    projection = project_candidate_cost(
        config=_config(), measured_tokens_per_second=707.0
    )

    assert projection["projected_all_in_cost_usd"] == pytest.approx(
        48.0164752475
    )
    assert projection["passes_launch_gate"] is False
    passing = project_candidate_cost(
        config=_config(), measured_tokens_per_second=708.0
    )
    assert passing["passes_launch_gate"] is True


def test_stage_projection_sums_to_complete_run_projection():
    config = _config()
    protocol = json.loads(
        (ROOT / config["scientific_protocol"]["path"]).read_text(encoding="utf-8")
    )
    stages = project_stage_costs(
        config=config,
        stages=protocol["stages"],
        measured_tokens_per_second=1_000.0,
    )
    complete = project_candidate_cost(
        config=config, measured_tokens_per_second=1_000.0
    )

    assert [row["stage_id"] for row in stages] == [
        "stage_1_historical_general",
        "stage_2_non_sonnet_poetry",
        "stage_3_sonnets",
    ]
    assert sum(row["target_tokens"] for row in stages) == 96_993_280
    assert sum(row["projected_all_in_hours"] for row in stages) == pytest.approx(
        complete["projected_all_in_hours"]
    )
    assert sum(row["projected_all_in_cost_usd"] for row in stages) == pytest.approx(
        complete["projected_all_in_cost_usd"]
    )


def test_preliminary_gates_require_finite_memory_stability_hardware_and_cost():
    config = _config()
    passing_metrics = [
        {
            "mean_loss": 2.0,
            "mean_gradient_norm": 3.0,
            "tokens_per_second": 1_000.0,
            "reserved_headroom_mib": 9_000.0,
            "reserved_memory_growth_mib": 0.0,
        },
        {
            "mean_loss": 2.1,
            "mean_gradient_norm": 3.1,
            "tokens_per_second": 1_000.0,
            "reserved_headroom_mib": 9_100.0,
            "reserved_memory_growth_mib": 10.0,
        },
    ]
    assert preliminary_gate_reasons(
        config=config,
        rank_metrics=passing_metrics,
        hardware={"profile_passed": True},
        communication={"algorithmic_gigabytes_per_second": 3.61},
        projection={"passes_launch_gate": True},
    ) == ()

    failed = preliminary_gate_reasons(
        config=config,
        rank_metrics=[
            {**passing_metrics[0], "reserved_headroom_mib": 8_000.0},
            {**passing_metrics[1], "reserved_memory_growth_mib": 600.0},
        ],
        hardware={"profile_passed": False},
        communication={"algorithmic_gigabytes_per_second": 2.0},
        projection={"passes_launch_gate": False},
    )
    assert set(failed) == {
        "insufficient_peak_reserved_headroom",
        "progressive_reserved_memory_growth",
        "hardware_profile_failed",
        "nccl_bandwidth_below_profile_floor",
        "projected_cost_exceeds_launch_gate",
    }


def test_selection_is_fastest_passing_candidate_and_proofs_remain_mandatory():
    rows = [
        {
            "candidate": {"candidate_id": "slow"},
            "mean_tokens_per_second": 800.0,
            "projection": {"projected_all_in_cost_usd": 40.0},
            "preliminary_gate_reasons": [],
        },
        {
            "candidate": {"candidate_id": "fast"},
            "mean_tokens_per_second": 1_000.0,
            "projection": {"projected_all_in_cost_usd": 32.0},
            "preliminary_gate_reasons": [],
        },
        {
            "candidate": {"candidate_id": "failed"},
            "mean_tokens_per_second": 2_000.0,
            "projection": {"projected_all_in_cost_usd": 16.0},
            "preliminary_gate_reasons": ["insufficient_peak_reserved_headroom"],
        },
    ]

    assert select_preliminary_candidate(rows)["candidate"]["candidate_id"] == "fast"
    assert qualification_gate_reasons(
        preliminary_reasons=(),
        validation_transition_passed=True,
        atomic_checkpoint_passed=True,
        fresh_process_resume_passed=False,
    ) == ("fresh_process_resume_failed",)


def test_observed_preflight_records_pytorch_nccl_and_storage_without_claiming_pass():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    observed = config["observed_machine_preflight"]

    assert observed["pytorch"] == "2.12.0+cu130"
    assert observed["measured_nccl_algorithmic_gigabytes_per_second"] == 3.610
    assert observed["nvidia_smi_topology_label"] == "PIX"
    assert observed["nvidia_smi_nvlink_links_per_gpu"] == 4
    assert observed["visible_root_filesystem_gib"] == 350
    assert observed["workspace_is_persistent_volume"] is False


def test_public_a6000_report_records_fail_closed_result_without_test_access():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["result"] == "failed_use_single_h100_sxm_fallback"
    assert len(report["candidate_results"]) == 8
    assert sum(
        row["result"] == "failed_cuda_out_of_memory"
        for row in report["candidate_results"]
    ) == 6
    assert report["fastest_runnable_candidate"]["candidate_id"] == (
        "a6000_context2048_micro1_accum8_gc_on_torch_compile_default"
    )
    assert report["state_transition_proofs"]["fresh_process_resume_run"] is False
    assert report["v7_test_accessed"] is False
    assert report["training_started"] is False


def test_transfer_reader_ignores_absent_protected_test_pool(monkeypatch, tmp_path):
    encoded_dir = tmp_path / "encoded"
    index_dir = tmp_path / "indexes"
    encoded_dir.mkdir()
    (index_dir / "training").mkdir(parents=True)
    train_shard = encoded_dir / "train_pool-00000.int32.bin"
    train_shard.write_bytes(struct.pack("<3i", 11, 12, 13))
    train_index = index_dir / "training/stage.jsonl"
    train_row = {
        "pool_id": "train_pool",
        "source_slices": [
            {"shard_index": 0, "token_offset": 0, "token_count": 3}
        ],
        "source_span_tokens": 3,
        "target_tokens": 2,
    }
    train_index.write_text(json.dumps(train_row) + "\n", encoding="utf-8")

    def artifact(path):
        return {
            "path": path.relative_to(index_dir).as_posix(),
            "bytes": path.stat().st_size,
            "rows": 1,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    train_artifact = artifact(train_index)
    test_artifact = {
        "path": "test/sonnets_test.jsonl",
        "bytes": 123,
        "rows": 1,
        "sha256": "0" * 64,
    }
    encoded_report = {
        "pools": [
            {
                "pool_id": "train_pool",
                "shards": [
                    {
                        "path": str(train_shard),
                        "bytes": train_shard.stat().st_size,
                        "token_count": 3,
                        "shard_index": 0,
                        "sha256": hashlib.sha256(
                            train_shard.read_bytes()
                        ).hexdigest(),
                    }
                ],
            },
            {
                "pool_id": "sonnets_test",
                "shards": [
                    {
                        "path": "missing/sonnets_test-00000.int32.bin",
                        "bytes": 400,
                        "token_count": 100,
                        "shard_index": 0,
                        "sha256": "0" * 64,
                    }
                ],
            },
        ]
    }
    window_manifest = {
        "files": [test_artifact, train_artifact],
        "training": {
            "stages": [
                {
                    "index": train_artifact,
                    "components": [
                        {"pool_windows": {"train_pool": 1}}
                    ],
                }
            ]
        },
        "evaluation": {"validation": {"pools": []}},
    }
    execution_path = tmp_path / "execution.json"
    execution_path.write_text(
        json.dumps(
            {
                "local_paths": {
                    "encoded_dir": "encoded",
                    "window_index_dir": "indexes",
                    "modern_encoded_dir": "modern",
                    "modern_index_path": "modern.jsonl",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        gpu_qualification,
        "build_execution_context",
        lambda _config: {
            "encoded_report": encoded_report,
            "window_manifest": window_manifest,
        },
    )

    _context, reader, store = gpu_qualification._open_training_reader(
        tmp_path,
        {"scientific_protocol": {"execution_path": "execution.json"}},
    )
    try:
        assert reader.source_tokens(reader.rows("stage")[0]) == (11, 12, 13)
    finally:
        store.close()


def test_matrix_worker_uses_current_python_for_distributed_launch(monkeypatch):
    script_path = ROOT / "scripts/qualify_minerva_7b_v7_dual_a6000.py"
    spec = importlib.util.spec_from_file_location("v7_a6000_matrix", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._run_worker("candidate", "safe")

    assert captured["command"][:3] == [
        module.sys.executable,
        "-m",
        "torch.distributed.run",
    ]
    assert captured["command"][-3:] == [
        "candidate",
        "--candidate-id",
        "safe",
    ]
    assert captured["kwargs"]["cwd"] == module.ROOT
