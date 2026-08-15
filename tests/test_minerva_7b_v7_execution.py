import hashlib
import json
import struct
from pathlib import Path

import pytest
import torch

from sonnet_training.minerva_7b_v7_execution import (
    CHECKPOINT_VERSION,
    FrozenWindowReader,
    Int32ShardStore,
    V7ExecutionConfig,
    atomic_install_checkpoint,
    atomic_install_checkpoint_writer,
    build_execution_context,
    fresh_process_resume_contract,
    load_execution_config,
    make_update_telemetry,
    optimizer_state_inventory,
    rotate_resume_checkpoints,
    select_document_probe_rows,
    selected_probe_positions,
    summarize_named_tensors,
    verify_checkpoint_directory,
)


ROOT = Path(__file__).resolve().parents[1]
EXECUTION_PATH = ROOT / "configs/minerva_7b_v7_execution.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_reader(tmp_path: Path):
    encoded = tmp_path / "encoded"
    encoded.mkdir()
    shard = encoded / "pool-00000.int32.bin"
    shard.write_bytes(struct.pack("<10i", *range(10)))
    report = {
        "pools": [
            {
                "pool_id": "pool",
                "shards": [
                    {
                        "shard_index": 0,
                        "path": "pool-00000.int32.bin",
                        "token_count": 10,
                        "bytes": 40,
                        "sha256": _sha(shard),
                    }
                ],
            }
        ]
    }
    index_root = tmp_path / "indexes"
    (index_root / "training").mkdir(parents=True)
    rows = []
    for index, offset in enumerate((0, 3)):
        rows.append(
            {
                "pool_id": "pool",
                "stage_window_index": index,
                "source_span_tokens": 4,
                "target_tokens": 3,
                "source_slices": [
                    {"shard_index": 0, "token_offset": offset, "token_count": 4}
                ],
            }
        )
    index_path = index_root / "training/stage.jsonl"
    index_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest = {
        "files": [
            {
                "path": "training/stage.jsonl",
                "rows": 2,
                "bytes": index_path.stat().st_size,
                "sha256": _sha(index_path),
            }
        ]
    }
    store = Int32ShardStore(
        encoded_dir=encoded, encoded_report=report, required_pools=["pool"]
    )
    reader = FrozenWindowReader(
        index_root=index_root,
        encoded_store=store,
        window_manifest=manifest,
    )
    return store, reader


def test_execution_config_pins_lineage_evidence_and_authorization():
    execution = load_execution_config(EXECUTION_PATH, ROOT)

    assert execution["activation_probes"]["probes_per_domain"] == 12
    assert execution["training_runtime"]["world_size"] == 2
    assert execution["evidence_retention"]["model_states"] == 7
    assert execution["authorization"]["cpu_implementation_and_bundle_build_approved"]
    assert not execution["authorization"]["gpu_qualification_authorized"]
    assert not execution["authorization"]["long_training_authorized"]


@pytest.mark.local_artifact
def test_execution_context_matches_local_frozen_window_manifest():
    config = V7ExecutionConfig(
        repo_root=ROOT,
        execution_path=EXECUTION_PATH,
        encoded_dir=ROOT / "data/local/minerva_7b_v7/encoded",
        window_index_dir=ROOT / "data/local/minerva_7b_v7/window_indexes",
        modern_encoded_dir=ROOT / "data/local/minerva_7b_full_weight/encoded",
        modern_index_path=ROOT
        / "data/local/minerva_7b_v7/modern_preservation_validation_v1.jsonl",
    )

    context = build_execution_context(config)

    assert context["window_manifest"]["window_index_content_identity_sha256"] == (
        "e821e3afdc3bd7aa6874180509ba756f942e651980f6455469722c13f8f7424c"
    )


def test_window_reader_reconstructs_shifted_inputs_and_targets(tmp_path):
    store, reader = _tiny_reader(tmp_path)
    try:
        batch = reader.optimizer_batch(
            stage_id="stage", update=1, global_windows_per_update=2
        )
    finally:
        store.close()

    assert batch.input_ids == ((0, 1, 2), (3, 4, 5))
    assert batch.target_ids == ((1, 2, 3), (4, 5, 6))
    assert batch.first_window_index == 0
    assert batch.next_window_index == 2
    assert len(batch.identity_sha256) == 64


def test_window_reader_rank_striding_preserves_each_global_window_once(tmp_path):
    store, reader = _tiny_reader(tmp_path)
    try:
        batch = reader.optimizer_batch(
            stage_id="stage", update=1, global_windows_per_update=2
        )
        rank_zero = reader.rank_microbatches(
            batch, rank=0, world_size=2, local_microbatch_size=1
        )
        rank_one = reader.rank_microbatches(
            batch, rank=1, world_size=2, local_microbatch_size=1
        )
    finally:
        store.close()

    assert rank_zero == (((0, 1, 2),),)
    assert rank_one == (((3, 4, 5),),)


def test_shard_reader_rejects_hash_mismatch(tmp_path):
    store, _ = _tiny_reader(tmp_path)
    store.close()
    shard = tmp_path / "encoded/pool-00000.int32.bin"
    shard.write_bytes(shard.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="hash mismatch"):
        Int32ShardStore(
            encoded_dir=tmp_path / "encoded",
            encoded_report={
                "pools": [
                    {
                        "pool_id": "pool",
                        "shards": [
                            {
                                "shard_index": 0,
                                "path": "pool-00000.int32.bin",
                                "token_count": 10,
                                "bytes": 40,
                                "sha256": "0" * 64,
                            }
                        ],
                    }
                ]
            },
        )


def test_atomic_checkpoint_verifies_and_rejects_tampering(tmp_path):
    destination = tmp_path / "resume_000001"
    manifest = atomic_install_checkpoint(
        destination=destination,
        files={"model/state.bin": b"weights", "rng.json": b"{}"},
        metadata={"stage_id": "stage_1"},
    )

    assert manifest["checkpoint_version"] == CHECKPOINT_VERSION
    assert verify_checkpoint_directory(destination) == manifest
    (destination / "model/state.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="mismatch"):
        verify_checkpoint_directory(destination)


def test_atomic_checkpoint_rejects_unsafe_paths(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        atomic_install_checkpoint(
            destination=tmp_path / "resume",
            files={"../escape": b"bad"},
            metadata={},
        )
    assert not (tmp_path / "resume").exists()
    assert not (tmp_path / "resume.tmp").exists()


def test_atomic_checkpoint_writer_hashes_nested_large_artifact_layout(tmp_path):
    destination = tmp_path / "snapshot"

    def populate(directory):
        (directory / "model").mkdir()
        (directory / "model/weights.safetensors").write_bytes(b"weights")
        (directory / "optimizer.pt").write_bytes(b"optimizer")

    manifest = atomic_install_checkpoint_writer(
        destination=destination,
        populate=populate,
        metadata={"stage_update": 100},
    )

    assert {row["path"] for row in manifest["files"]} == {
        "model/weights.safetensors",
        "optimizer.pt",
    }
    assert verify_checkpoint_directory(destination) == manifest


def test_resume_rotation_keeps_newest_two_verified_generations(tmp_path):
    for update in (1, 2, 3):
        atomic_install_checkpoint(
            destination=tmp_path / f"resume_{update:06d}",
            files={"state.json": b"{}"},
            metadata={"update": update},
        )

    rotate_resume_checkpoints(tmp_path, retain=2)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "resume_000002",
        "resume_000003",
    ]


def test_fresh_process_contract_detects_exact_state_mismatch():
    expected = {
        "stage_id": "stage_1",
        "stage_update": 100,
        "global_update": 100,
        "next_stage_window_index": 1600,
        "next_window_identity_sha256": "a" * 64,
        "next_learning_rate": 1e-5,
        "protocol_sha256": "b" * 64,
        "encoded_content_identity_sha256": "c" * 64,
        "window_content_identity_sha256": "d" * 64,
        "world_size": 2,
    }
    passing = fresh_process_resume_contract(
        manifest={"metadata": expected}, expected=expected
    )
    failing = fresh_process_resume_contract(
        manifest={"metadata": {**expected, "world_size": 1}}, expected=expected
    )

    assert passing["passes"]
    assert failing["mismatches"] == ["world_size"]


def test_compact_tensor_and_optimizer_summaries_keep_no_full_values():
    tensors = [
        ("model.layers.0.weight", torch.tensor([3.0, 4.0, 0.0])),
        ("model.layers.0.bias", torch.tensor([0.0, 2.0])),
    ]
    summary = summarize_named_tensors(tensors)
    inventory = optimizer_state_inventory(
        {
            "state": {0: {"step": 2, "moment": torch.tensor([3.0, 4.0])}},
            "param_groups": [{}],
        }
    )

    row = summary["model.layers.0"]
    assert row["element_count"] == 5
    assert row["l2_norm"] == pytest.approx((29.0) ** 0.5)
    assert row["zero_fraction"] == pytest.approx(0.4)
    assert inventory["rows"][0]["l2_norm"] == 5.0
    assert "tensor_values" not in json.dumps(inventory)


def test_probe_positions_skip_special_tokens_and_add_markers():
    positions = selected_probe_positions(
        [1, 10, 11, 12, 13, 2], special_token_ids=[1, 2], rare_token_positions=[2]
    )
    assert 0 not in positions
    assert 5 not in positions
    assert 2 in positions


def test_probe_selection_maximizes_documents_then_uses_more_excerpts():
    documents = [
        {"tokens": 2000, "unit_id": "a"},
        {"tokens": 2000, "unit_id": "b"},
    ]
    selected = select_document_probe_rows(
        documents, count=4, seed=7, minimum_tokens=64, maximum_tokens=512
    )

    assert {row["unit_id"] for row in selected[:2]} == {"a", "b"}
    assert len(selected) == 4
    assert all("probe_token_offset" in row for row in selected)


def test_update_telemetry_records_window_identity_and_rejects_nonfinite(tmp_path):
    store, reader = _tiny_reader(tmp_path)
    try:
        batch = reader.optimizer_batch(
            stage_id="stage", update=1, global_windows_per_update=2
        )
    finally:
        store.close()
    row = make_update_telemetry(
        stage_id="stage",
        stage_update=1,
        global_update=1,
        batch=batch,
        loss=2.0,
        gradient_norm=3.0,
        learning_rate=1e-5,
        tokens_per_second=1000.0,
        elapsed_seconds=1.0,
        eta_seconds=9.0,
        rank_memory=[{"rank": 0, "reserved_mib": 100.0}],
        cumulative_cost_usd=0.01,
    )
    assert row["window_identity_sha256"] == batch.identity_sha256
    with pytest.raises(ValueError, match="non-finite"):
        make_update_telemetry(
            stage_id="stage",
            stage_update=1,
            global_update=1,
            batch=batch,
            loss=float("nan"),
            gradient_norm=3.0,
            learning_rate=1e-5,
            tokens_per_second=1000.0,
            elapsed_seconds=1.0,
            eta_seconds=9.0,
            rank_memory=[],
            cumulative_cost_usd=None,
        )


@pytest.mark.local_artifact
def test_completed_public_report_and_local_probe_manifest_are_consistent():
    report = json.loads(
        (ROOT / "reports/minerva_7b_v7_execution_v1.json").read_text()
    )
    probe_path = ROOT / "data/local/minerva_7b_v7/activation_probes_v1.json"
    probe = json.loads(probe_path.read_text())

    assert report["status"] == "cpu_execution_artifacts_verified_gpu_unauthorized"
    assert report["reader"]["total_updates"] == 2960
    assert report["activation_probes"]["manifest_sha256"] == _sha(probe_path)
    assert probe["probe_count"] == 48
    assert set(probe["domains"].values()) == {12}
    assert probe["v7_test_accessed"] is False
    assert "sonnets_test" not in probe_path.read_text()
    assert report["transfer_bundle"]["local_build"] == {
        "bytes": 283_511_320,
        "files": 77,
        "public": False,
        "sha256": "034406e616987ab07d090d9c9928b1fa0368d6f334ffeb883d48cb2b40fd672f",
        "v7_test_material_included": False,
        "verified": True,
    }
