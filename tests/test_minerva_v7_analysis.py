import hashlib
import json
import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from sonnet_analysis.minerva_v7_dynamics import (
    build_dynamics_report,
    read_jsonl_tolerant,
    write_dynamics_exports,
)
from sonnet_analysis.minerva_v7_gpu_plan import (
    build_gpu_extraction_plan,
    validate_causal_experiment_proposal,
    validate_probe_manifest,
)
from sonnet_analysis.minerva_v7_registry import MODEL_STATES, audit_research_states
from sonnet_analysis.minerva_v7_weights import (
    compare_model_weights,
    parameter_group,
    plan_weight_comparison,
)


def _protocol(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "protocol_sha256": "p" * 64,
                "model": {"model_id": "parent", "revision": "revision"},
                "training": {
                    "stages": [
                        {"stage_id": "stage_1_historical_general", "optimizer_updates": 2},
                        {"stage_id": "stage_2_non_sonnet_poetry", "optimizer_updates": 2},
                        {"stage_id": "stage_3_sonnets", "optimizer_updates": 1},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _snapshot(
    path: Path, *, stage_id: str, role: str, update: int,
    preceding: str | None = None,
) -> None:
    (path / "model").mkdir(parents=True)
    weight = path / "model/model.safetensors"
    save_file({"weight": torch.tensor([1.0, 2.0])}, weight)
    manifest = {
        "checkpoint_version": "minerva_7b_v7_atomic_checkpoint_v1",
        "files": [
            {
                "path": "model/model.safetensors",
                "bytes": weight.stat().st_size,
                "sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
            }
        ],
        "metadata": {
            "artifact_type": "model_only_analysis_snapshot",
            "stage_id": stage_id,
            "snapshot_role": role,
            "update": update,
            "protocol_sha256": "p" * 64,
            "preceding_model_identity_sha256": preceding or hashlib.sha256(
                b"parent@revision"
            ).hexdigest(),
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_registry_distinguishes_complete_missing_partial_and_invalid(tmp_path):
    protocol = _protocol(tmp_path / "protocol.json")
    run = tmp_path / "run"
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "config.json").write_text("{}", encoding="utf-8")
    save_file({"weight": torch.ones(1)}, parent / "model.safetensors")
    _snapshot(
        run / "analysis_snapshots/stage_1_historical_general_update_001033",
        stage_id="stage_1_historical_general", role="midpoint", update=1033,
    )
    partial = run / "stage_boundaries/stage_1_historical_general_selected"
    partial.mkdir(parents=True)
    _snapshot(
        run / "analysis_snapshots/stage_2_non_sonnet_poetry_update_000380",
        stage_id="wrong_stage", role="midpoint", update=380,
    )

    report = audit_research_states(
        run_dir=run, protocol_path=protocol, parent_model_dir=parent, verify_hashes=True
    )
    statuses = {row["state_id"]: row["status"] for row in report["states"]}

    assert statuses["untouched_parent"] == "complete"
    assert statuses["stage_1_midpoint"] == "complete"
    assert statuses["stage_1_selected"] == "partial"
    assert statuses["stage_2_midpoint"] == "invalid"
    assert statuses["stage_3_selected"] == "missing"
    assert not report["all_seven_states_complete"]
    assert not report["causal_experiments_authorized"]


def test_registry_rejects_cross_state_lineage_mismatch(tmp_path):
    protocol = _protocol(tmp_path / "protocol.json")
    run = tmp_path / "run"
    _snapshot(
        run / "analysis_snapshots/stage_1_historical_general_update_001033",
        stage_id="stage_1_historical_general", role="midpoint", update=1033,
    )
    selected = run / "stage_boundaries/stage_1_historical_general_selected"
    _snapshot(
        selected, stage_id="stage_1_historical_general",
        role="validation_selected_endpoint", update=2, preceding="x" * 64,
    )
    manifest_path = selected / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"].update(
        {
            "selected_metrics": {}, "parent_baseline_metrics": {},
            "validation_history": [], "source_candidate_manifest_sha256": "s" * 64,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = audit_research_states(run_dir=run, protocol_path=protocol)
    rows = {row["state_id"]: row for row in report["states"]}

    assert rows["stage_1_midpoint"]["status"] == "invalid"
    assert rows["stage_1_selected"]["status"] == "invalid"
    assert any("different preceding identities" in issue for issue in rows["stage_1_selected"]["issues"])


def _telemetry(stage: str, update: int, global_update: int) -> dict:
    return {
        "stage_id": stage,
        "stage_update": update,
        "global_update": global_update,
        "mean_training_loss": 3.0 - 0.1 * global_update,
        "preclip_global_gradient_norm": 1.0,
        "learning_rate": 1e-5,
        "tokens_per_second": 1000.0,
        "elapsed_seconds": float(global_update),
        "eta_seconds": 1.0,
        "cumulative_cost_usd": global_update / 10,
        "first_window_index": (update - 1) * 16,
        "next_window_index": update * 16,
        "window_identity_sha256": str(global_update) * 64,
    }


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_dynamics_accepts_completed_prefix_and_exports_tables(tmp_path):
    protocol = _protocol(tmp_path / "protocol.json")
    run = tmp_path / "run"
    _jsonl(
        run / "telemetry.jsonl",
        [
            _telemetry("stage_1_historical_general", 1, 1),
            _telemetry("stage_1_historical_general", 2, 2),
            _telemetry("stage_2_non_sonnet_poetry", 1, 3),
        ],
    )
    _jsonl(
        run / "evaluations.jsonl",
        [
            {
                "stage_id": "stage_1_historical_general", "update": 2,
                "passes_all_gates": True, "is_current_selected_candidate": True,
            }
        ],
    )
    report = build_dynamics_report(run_dir=run, protocol_path=protocol)
    paths = write_dynamics_exports(report, tmp_path / "exports")

    assert report["status"] == "valid_in_progress"
    assert report["stages"][0]["complete"]
    assert report["stages"][1]["latest_update"] == 1
    assert Path(paths["telemetry"]).read_text(encoding="utf-8").count("\n") == 4


def test_jsonl_tolerates_only_a_truncated_final_line(tmp_path):
    path = tmp_path / "active.jsonl"
    path.write_bytes(b'{"stage_id":"ok"}\n{"stage_id":')
    rows, ignored = read_jsonl_tolerant(path)
    assert rows == [{"stage_id": "ok"}]
    assert ignored
    path.write_bytes(b'{bad}\n{"stage_id":"ok"}\n')
    with pytest.raises(ValueError, match="row 1"):
        read_jsonl_tolerant(path)


def test_dynamics_rejects_conflicting_duplicate_updates(tmp_path):
    protocol = _protocol(tmp_path / "protocol.json")
    run = tmp_path / "run"
    first = _telemetry("stage_1_historical_general", 1, 1)
    second = {**first, "mean_training_loss": 99.0}
    _jsonl(run / "telemetry.jsonl", [first, second])
    with pytest.raises(ValueError, match="conflicting duplicate"):
        build_dynamics_report(run_dir=run, protocol_path=protocol)


def _weights(path: Path, *, offset: float = 0.0, shape=(2, 3)) -> Path:
    path.mkdir()
    save_file(
        {
            "model.embed_tokens.weight": torch.arange(6, dtype=torch.float32).reshape(shape) + offset,
            "model.layers.0.self_attn.q_proj.weight": torch.eye(3) + offset,
            "model.layers.8.mlp.down_proj.weight": torch.ones((2, 3)) + offset,
            "model.norm.weight": torch.ones(3) + offset,
            "lm_head.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3) + offset,
        },
        path / "model.safetensors",
    )
    return path


def test_weight_plan_and_comparison_are_chunked_and_exact(tmp_path):
    left = _weights(tmp_path / "left")
    right = _weights(tmp_path / "right", offset=1.0)
    plan = plan_weight_comparison(
        left_model_dir=left, right_model_dir=right, chunk_bytes=16
    )
    report = compare_model_weights(
        left_model_dir=left, right_model_dir=right, chunk_bytes=16
    )

    assert plan["maximum_simultaneous_input_chunks"] == 2
    assert not plan["full_delta_tensor_materialized"]
    assert plan["maximum_projected_working_bytes"] <= 512 * 1024**2
    embedding = next(row for row in report["tensors"] if row["name"] == "model.embed_tokens.weight")
    assert embedding["delta_l2"] == pytest.approx(6**0.5)
    assert sum(row["fraction_of_total_delta_energy"] for row in report["tensors"]) == pytest.approx(1.0)
    assert parameter_group("model.layers.23.mlp.gate_proj.weight") == "blocks_16_23/mlp"


def test_weight_plan_rejects_tensor_shape_mismatch(tmp_path):
    left = _weights(tmp_path / "left")
    right = _weights(tmp_path / "right", shape=(3, 2))
    with pytest.raises(ValueError, match="shape mismatch"):
        plan_weight_comparison(left_model_dir=left, right_model_dir=right)


def test_weight_plan_accounts_for_a_row_wider_than_requested_chunk(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    save_file({"wide": torch.ones((2, 100), dtype=torch.float32)}, left / "model.safetensors")
    save_file({"wide": torch.ones((2, 100), dtype=torch.float32)}, right / "model.safetensors")
    plan = plan_weight_comparison(
        left_model_dir=left, right_model_dir=right, chunk_bytes=16
    )
    assert plan["largest_chunk_elements"] == 100
    assert plan["maximum_projected_working_bytes"] == 100 * 4 * 6


def _probe_manifest(path: Path) -> Path:
    probes = []
    for domain in sorted(
        ("modern_instruction", "historical_general", "historical_non_sonnet_poetry", "standard_sonnet")
    ):
        for index in range(12):
            tokens = [index + 1, index + 2, index + 3]
            digest = hashlib.sha256(
                b"".join(struct.pack("<I", value) for value in tokens)
            ).hexdigest()
            probes.append(
                {
                    "probe_id": f"{domain}:{index}", "domain": domain,
                    "source_identity": f"{domain}-source-{index}",
                    "source_split": "instruction_preservation" if domain == "modern_instruction" else f"validation_{domain}",
                    "input_ids": tokens, "attention_mask": [1, 1, 1],
                    "selected_positions": [0, 2], "input_ids_sha256": digest,
                }
            )
    payload = {
        "probe_version": "minerva_7b_v7_activation_probes_v1",
        "probe_count": 48,
        "v7_test_accessed": False,
        "extraction": {
            "bounded_raw_attention": {
                "layer_indices": [0, 8, 16, 24, 31], "maximum_tokens": 256
            },
            "fixed_logit_summary": {"top_k": 20},
        },
        "probes": probes,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_gpu_plan_is_bounded_resumable_and_noncausal(tmp_path):
    probes = _probe_manifest(tmp_path / "probes.json")
    audit = {
        "states": [
            {"state_id": state.state_id, "status": "complete", "path": f"/states/{state.state_id}"}
            for state in MODEL_STATES
        ]
    }
    plan = build_gpu_extraction_plan(
        probe_manifest_path=probes,
        state_audit=audit,
        output_root=tmp_path / "outputs",
        model_config={"hidden_size": 16, "num_hidden_layers": 32, "num_attention_heads": 4},
    )

    assert plan["ready_state_count"] == 7
    assert plan["estimates"]["all_seven_states_bytes"] > 0
    assert all(job["resumable_unit"] == "one_complete_model_state" for job in plan["jobs"])
    assert not plan["causal_experiments_authorized"]
    assert not plan["execution"]["v7_test_accessed"]


def test_probe_manifest_rejects_test_material(tmp_path):
    path = _probe_manifest(tmp_path / "probes.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["probes"][0]["source_split"] = "sonnets_test"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="test split"):
        validate_probe_manifest(path)


def test_complete_causal_proposal_still_requires_separate_approval():
    proposal = {
        "descriptive_finding_id": "finding-001",
        "hypothesis": "blocks 16--23 mediate the poetry gain",
        "intervention": "restore blocks 16--23 from stage 1 into stage 2",
        "negative_control": "restore a parameter-matched early-block set",
        "predicted_adaptation_effect": "poetry loss worsens",
        "predicted_preservation_effect": "modern loss remains stable",
        "stopping_rule": "one intervention and one control only",
        "model_state_comparisons": ["stage_1_selected_to_stage_2_selected"],
        "evaluation_domains": [
            "modern_instruction", "historical_general",
            "historical_non_sonnet_poetry", "standard_sonnet",
        ],
        "primary_metrics": ["historical_non_sonnet_poetry_loss"],
        "v7_test_accessed": False,
    }
    result = validate_causal_experiment_proposal(proposal)
    assert result["proposal_complete"]
    assert not result["execution_authorized"]
    assert result["separate_user_approval_required"]


def test_incomplete_causal_proposal_is_rejected():
    with pytest.raises(ValueError, match="causal proposal is incomplete"):
        validate_causal_experiment_proposal({"v7_test_accessed": False})
