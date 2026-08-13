"""Fail-closed real-model loading and runtime preflight for V7 research jobs."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import torch

from sonnet_analysis.minerva_v7_registry import COMPARISONS


RUNTIME_VERSION = "minerva_7b_v7_research_runtime_v1"
MINIMUM_FREE_DISK_GIB = 40
MINIMUM_GPU_MEMORY_MIB = 76800


def load_research_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "research_version": "minerva_7b_v7_post_training_research_v1",
        "model_id": "sapienzanlp/Minerva-7B-instruct-v1.0",
        "revision": "d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d",
        "state_count": 7,
        "probe_manifest_sha256": "3557b4e455357ca165b4689a3876de7965ad59677cba6f0c0b00d2fad956488b",
        "prompt_sha256": "98f429aeb04c4491517b3e1c218d21a98596476d163c87d62f3d09d535ea70e5",
        "encoded_data_report_sha256": "2acaf9c8a598e2543017b17b4b60f2d9d4a4b18520345ded1cd8712bc9304f3e",
        "embedding_registry_sha256": "439a90cd6721dc19121a180c2f46782588bdd880b7eaa5c7686ae1663ddbbf54",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"research configuration mismatch: {key}")
    authorization = config.get("authorization", {})
    if not authorization.get("descriptive_analysis_approved"):
        raise PermissionError("descriptive analysis is not approved")
    if authorization.get("causal_experiments_authorized") or authorization.get("v7_test_access_authorized"):
        raise PermissionError("research configuration crosses a prohibited boundary")
    if not authorization.get("gpu_execution_requires_user_manual_launch"):
        raise PermissionError("research configuration does not preserve manual launch ownership")
    return config


def load_verified_state(audit_path: Path, state_id: str) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("hash_verification_performed"):
        raise ValueError("GPU research requires a full state hash audit")
    rows = {row["state_id"]: row for row in audit.get("states", [])}
    if state_id not in rows or rows[state_id].get("status") != "complete":
        raise ValueError(f"research state is not complete and verified: {state_id}")
    row = rows[state_id]
    if not row.get("state_identity_sha256"):
        raise ValueError("verified state lacks a stable research identity")
    model_dir = Path(str(row.get("model_dir") or Path(str(row["path"])) / "model"))
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"verified state model directory is absent: {model_dir}")
    return {**row, "model_dir": str(model_dir)}


def load_verified_comparison(
    audit_path: Path, left_state_id: str, right_state_id: str
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Resolve one frozen comparison only through a full state-integrity audit."""

    matches = [
        row for row in COMPARISONS
        if row["left"] == left_state_id and row["right"] == right_state_id
    ]
    if len(matches) != 1:
        raise ValueError("state pair is not present in the frozen comparison registry")
    left = load_verified_state(audit_path, left_state_id)
    right = load_verified_state(audit_path, right_state_id)
    return left, right, str(matches[0]["comparison_id"])


def gpu_preflight(*, output_root: Path, required_output_bytes: int, hourly_rate: float) -> dict[str, Any]:
    if hourly_rate <= 0:
        raise ValueError("hourly_rate must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("V7 research extraction requires exactly one CUDA GPU")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    memory_mib = properties.total_memory / 1024**2
    if memory_mib < MINIMUM_GPU_MEMORY_MIB or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not meet the qualified H100-class BF16 contract")
    output_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(output_root)
    required_free = max(MINIMUM_FREE_DISK_GIB * 1024**3, required_output_bytes * 2)
    if disk.free < required_free:
        raise RuntimeError("insufficient disk for atomic research extraction")
    return {
        "runtime_version": RUNTIME_VERSION,
        "device": str(device),
        "gpu_name": properties.name,
        "gpu_memory_mib": memory_mib,
        "bfloat16_supported": True,
        "free_disk_bytes": disk.free,
        "required_free_disk_bytes": required_free,
        "hourly_rate_usd": hourly_rate,
        "training_authorized": False,
        "causal_experiments_authorized": False,
        "instance_lifecycle_action_authorized": False,
    }


def load_bf16_model_and_tokenizer(
    *, state: Mapping[str, Any], config: Mapping[str, Any], device: torch.device
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = str(state["model_dir"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    if any(parameter.dtype != torch.bfloat16 for parameter in model.parameters()):
        raise ValueError("research model is not uniformly BF16")
    architecture = model.config
    expected = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 51264,
    }
    for key, value in expected.items():
        if int(getattr(architecture, key)) != value:
            raise ValueError(f"research model architecture mismatch: {key}")
    return model, tokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
