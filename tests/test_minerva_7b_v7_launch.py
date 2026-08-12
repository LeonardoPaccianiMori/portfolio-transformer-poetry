import json
from pathlib import Path

import pytest

from sonnet_training.minerva_7b_v7_execution import atomic_install_checkpoint
from sonnet_training.minerva_7b_v7_launch import (
    load_single_h100_launch_config,
    stage_launch,
    validate_stage_boundary,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCH_PATH = ROOT / "configs/minerva_7b_v7_single_h100_launch.json"


def test_launch_contract_verifies_immutable_qualification_lineage():
    launch = load_single_h100_launch_config(LAUNCH_PATH, ROOT)

    assert launch["qualified_runtime"] == {
        "candidate_id": "h100_context2048_micro1_accum16_gc_off_torch_compile_default",
        "world_size": 1,
        "gpu_count": 1,
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "minimum_visible_memory_mib": 76800,
        "context_length": 2048,
        "global_windows_per_update": 16,
        "local_microbatch_size": 1,
        "gradient_accumulation_steps": 16,
        "gradient_checkpointing": False,
        "execution_mode": "torch_compile_default",
        "ddp_backend": "nccl",
        "ddp_bucket_cap_mib": 25,
    }
    assert launch["authorization"]["launch_owner"] == "user"
    assert launch["authorization"]["v7_test_access_authorized"] is False


def test_launch_contract_rejects_hash_changed_lineage(tmp_path):
    launch = json.loads(LAUNCH_PATH.read_text())
    launch["lineage"]["protocol_sha256"] = "0" * 64
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(launch))

    with pytest.raises(ValueError, match="protocol"):
        load_single_h100_launch_config(path, ROOT)


def test_stage_launch_is_explicit_and_ordered():
    launch = load_single_h100_launch_config(LAUNCH_PATH, ROOT)

    first = stage_launch(launch, "stage_1_historical_general")
    second = stage_launch(launch, "stage_2_non_sonnet_poetry")

    assert first["required_boundary"] is None
    assert second["required_boundary"].endswith("stage_1_historical_general_selected")
    with pytest.raises(ValueError, match="outside"):
        stage_launch(launch, "stage_4")


def test_stage_boundary_requires_complete_selected_endpoint_lineage(tmp_path):
    launch = load_single_h100_launch_config(LAUNCH_PATH, ROOT)
    boundary = tmp_path / "stage_1"
    metadata = {
        "artifact_type": "model_only_analysis_snapshot",
        "snapshot_role": "validation_selected_endpoint",
        "stage_id": "stage_1_historical_general",
        "protocol_sha256": launch["lineage"]["protocol_sha256"],
        "launch_config_sha256": launch["_launch_config_sha256"],
        "preceding_model_identity_sha256": "a" * 64,
        "source_candidate_manifest_sha256": "b" * 64,
        "selected_metrics": {"update": 100},
        "parent_baseline_metrics": {"modern_validation_loss": 2.0},
        "validation_history": [{"stage_id": "stage_1_historical_general"}],
    }
    atomic_install_checkpoint(
        destination=boundary,
        files={"model/model.safetensors": b"weights"},
        metadata=metadata,
    )

    manifest = validate_stage_boundary(
        path=boundary,
        expected_stage_id="stage_1_historical_general",
        launch=launch,
    )

    assert manifest["metadata"]["selected_metrics"]["update"] == 100


def test_stage_boundary_rejects_midpoint(tmp_path):
    launch = load_single_h100_launch_config(LAUNCH_PATH, ROOT)
    boundary = tmp_path / "midpoint"
    atomic_install_checkpoint(
        destination=boundary,
        files={"model/model.safetensors": b"weights"},
        metadata={
            "artifact_type": "model_only_analysis_snapshot",
            "snapshot_role": "midpoint",
            "stage_id": "stage_1_historical_general",
            "protocol_sha256": launch["lineage"]["protocol_sha256"],
        },
    )

    with pytest.raises(ValueError, match="selected endpoint"):
        validate_stage_boundary(
            path=boundary,
            expected_stage_id="stage_1_historical_general",
            launch=launch,
        )


def test_public_readiness_report_matches_launch_and_bundle():
    launch = load_single_h100_launch_config(LAUNCH_PATH, ROOT)
    report = json.loads(
        (ROOT / "reports/minerva_7b_v7_single_h100_launch_readiness_v1.json").read_text()
    )

    assert report["status"] == "ready_for_user_launched_stage_1"
    assert report["launch_config"]["sha256"] == launch["_launch_config_sha256"]
    assert report["runtime"]["candidate_id"] == launch["qualified_runtime"][
        "candidate_id"
    ]
    assert report["private_bundle"]["v7_test_material_included"] is False
    assert report["remote_preflight"]["stage_1_started"] is False
