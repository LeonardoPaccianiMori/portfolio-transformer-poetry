from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import pytest
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
RECONCILE_SCRIPT = ROOT / "scripts/reconcile_huggingface_lineage.py"
EXPORT_SCRIPT = ROOT / "scripts/export_huggingface_artifacts.py"
VALIDATE_SCRIPT = ROOT / "scripts/validate_huggingface_artifacts.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tracked_huggingface_release_metadata_is_published_and_bounded():
    validator = load_module("hf_validator", VALIDATE_SCRIPT)
    validator.validate_tracked_release_metadata()


def test_release_plan_uses_one_repository_with_four_subfolders():
    plan = json.loads((ROOT / "release/huggingface/release_plan.yml").read_text(encoding="utf-8"))
    repository = "LPM93/teaching-transformers-classical-italian-sonnets"
    assert plan["repository"] == repository
    assert plan["status"] == "published_public_ungated"
    assert plan["repository_visibility"] == "public_ungated"
    assert plan["publication_url"] == f"https://huggingface.co/{repository}"
    assert plan["publication_commit"] == "3581abbb1023c77f784b37aa152cdb6c0447fa73"
    assert plan["publication_date"] == "2026-08-21"
    assert plan["package_manifest_sha256"] == "cedc265ece2b0faf81cf03861ac2f64faa8dea90856dd919a5990e238f2746e2"
    assert plan["package_bytes"] == 44_492_691_499
    assert plan["package_files"] == 69
    assert plan["decision_record_id"] == "decision-2026-08-21-huggingface-single-repository-release"
    assert plan["authorization_date"] == "2026-08-21"
    assert {artifact["repository"] for artifact in plan["artifacts"]} == {repository}
    assert {artifact["subfolder"] for artifact in plan["artifacts"]} == {
        "stage1", "stage2", "stage3", "dpo_adapter",
    }
    adapter = next(row for row in plan["artifacts"] if row["artifact_id"] == "dpo_adapter")
    assert adapter["parent_repository"] == repository
    assert adapter["parent_subfolder"] == "stage3"


def test_aggregate_stage_counts_exact_target_tokens_without_text(tmp_path):
    reconcile = load_module("hf_reconcile", RECONCILE_SCRIPT)
    index = tmp_path / "stage.jsonl"
    rows = [
        {
            "component": "historical_general",
            "target_tokens": 2048,
            "target_contributions": [
                {"unit_id": "bibit:record:one", "tokens": 1024},
                {"unit_id": "gutenberg:record:two", "tokens": 1024},
            ],
        },
        {
            "component": "modern_preservation_replay",
            "target_tokens": 2048,
            "target_contributions": [
                {"unit_id": "paisa_even_byte_windows_v1", "tokens": 2048},
            ],
        },
    ]
    index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    first = reconcile.aggregate_stage(
        index,
        selected_windows=2,
        existing_families={},
        v6_families={},
    )
    second = reconcile.aggregate_stage(
        index,
        selected_windows=2,
        existing_families={},
        v6_families={},
    )
    assert first == second
    assert first["target_tokens"] == 4096
    assert first["source_family_target_tokens"] == {
        "Biblioteca Italiana": 1024,
        "PAISA": 2048,
        "Project Gutenberg": 1024,
    }
    serialized = json.dumps(first).lower()
    assert all(key not in serialized for key in ("poem", "opening", "generation", "corpus_text"))


def test_adapter_export_strips_private_training_state(tmp_path):
    exporter = load_module("hf_exporter", EXPORT_SCRIPT)
    checkpoint_path = tmp_path / "selected.pt"
    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
    }
    checkpoint = {
        "parent_state_identity_sha256": "p" * 64,
        "adapter_state_dict": tensors,
        "optimizer_state_dict": {"private": True},
        "torch_rng_state": torch.arange(4),
        "split_manifest": {"train_pair_ids": ["private-pair"]},
        "history": [{"private": True}],
        "config": {
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_rank": 8,
            "target_modules": ["q_proj"],
        },
    }
    torch.save(checkpoint, checkpoint_path)
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    artifact = {
        "artifact_id": "dpo_adapter",
        "parent_repository": "LPM93/teaching-transformers-classical-italian-sonnets",
        "parent_subfolder": "stage3",
        "parent_state_identity_sha256": "p" * 64,
        "research_checkpoint_sha256": digest,
    }
    output = tmp_path / "output"
    exporter.export_adapter(
        artifact=artifact,
        checkpoint_path=checkpoint_path,
        output_root=output,
    )
    exported = load_file(output / "dpo_adapter/adapter_model.safetensors")
    assert set(exported) == set(tensors)
    assert all(torch.equal(exported[key], value) for key, value in tensors.items())
    serialized_config = (output / "dpo_adapter/adapter_config.json").read_text(encoding="utf-8")
    assert "private-pair" not in serialized_config
    assert "optimizer" not in serialized_config
    assert "history" not in serialized_config
    config = json.loads(serialized_config)
    assert config["base_model_name_or_path"] == artifact["parent_repository"]


def test_export_rejects_nonignored_repository_destination(tmp_path):
    exporter = load_module("hf_exporter_output_root", EXPORT_SCRIPT)
    exporter.validate_output_root(tmp_path / "outside-repository")
    with pytest.raises(ValueError, match="must be ignored"):
        exporter.validate_output_root(ROOT / "release" / "unignored-model-package")


def test_stage3_lineage_stops_at_selected_update_120():
    lineage = json.loads((ROOT / "release/huggingface/stage3/lineage.json").read_text(encoding="utf-8"))
    stage3 = lineage["completed_training_stages"][-1]
    assert lineage["artifact"]["selected_update"] == 120
    assert lineage["artifact"]["planned_updates"] == 135
    assert stage3["selected_windows"] == 1920
    assert stage3["target_tokens"] == 3_932_160
