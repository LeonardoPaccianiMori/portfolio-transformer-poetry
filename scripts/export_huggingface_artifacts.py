#!/usr/bin/env python3
"""Build fail-closed local packages for the planned Hugging Face release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = REPOSITORY_ROOT / "release/huggingface"
RELEASE_PLAN = RELEASE_ROOT / "release_plan.yml"
UPSTREAM_FILE_MANIFEST = RELEASE_ROOT / "upstream_file_manifest.json"
CC_BY_NC = REPOSITORY_ROOT / "LICENSES/CC-BY-NC-4.0.txt"
APACHE = REPOSITORY_ROOT / "LICENSE"
CC_BY = REPOSITORY_ROOT / "LICENSES/CC-BY-4.0.txt"

FULL_MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "model.safetensors.index.json",
)
UPSTREAM_METADATA_FILES = (
    "chat_template.jinja",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
DOCUMENT_FILES = ("README.md", "NOTICE.md", "RIGHTS_SCOPE.md", "TRAINING_CONTENT_SUMMARY.md", "lineage.json")
ROOT_README = RELEASE_ROOT / "README.md"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_verified(source: Path, destination: Path, expected_sha256: str | None = None) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    actual = sha256_path(source)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"Hash mismatch for {source}: {actual} != {expected_sha256}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256_path(destination) != actual:
        raise ValueError(f"Copy verification failed for {destination}")


def _copy_release_documents(artifact_id: str, destination: Path) -> None:
    source_root = RELEASE_ROOT / artifact_id
    for name in DOCUMENT_FILES:
        copy_verified(source_root / name, destination / name)
    copy_verified(CC_BY_NC, destination / "LICENSE")
    copy_verified(APACHE, destination / "LICENSES/APACHE-2.0.txt")
    copy_verified(CC_BY, destination / "LICENSES/CC-BY-4.0.txt")


def _verify_boundary_manifest(source_root: Path, expected_identity: str) -> dict[str, str]:
    manifest_path = source_root / "manifest.json"
    if sha256_path(manifest_path) != expected_identity:
        raise ValueError(f"Unexpected selected-state identity for {source_root}")
    manifest = load_json(manifest_path)
    return {str(row["path"]): str(row["sha256"]) for row in manifest["files"]}


def export_full_model(
    *, artifact: dict[str, Any], boundaries_root: Path, output_root: Path,
    include_verified_upstream_files: bool,
) -> None:
    artifact_id = str(artifact["artifact_id"])
    source_root = boundaries_root / str(artifact["source_directory"])
    source_model = source_root / "model"
    destination = output_root / artifact_id
    expected_files = _verify_boundary_manifest(source_root, str(artifact["state_identity_sha256"]))
    upstream_manifest = load_json(UPSTREAM_FILE_MANIFEST)
    _copy_release_documents(artifact_id, destination)
    for name in FULL_MODEL_FILES:
        if name in upstream_manifest["package_files"]:
            upstream_hash = str(upstream_manifest["package_files"][name]["sha256"])
            if expected_files[f"model/{name}"] != upstream_hash:
                raise ValueError(f"Selected {name} does not match reviewed upstream-derived metadata")
        copy_verified(source_model / name, destination / name, expected_files[f"model/{name}"])
    if include_verified_upstream_files:
        for name in UPSTREAM_METADATA_FILES:
            row = upstream_manifest["package_files"][name]
            copy_verified(source_model / name, destination / name, str(row["sha256"]))


def _adapter_config(parent_repository: str, checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint["config"]
    return {
        "base_model_name_or_path": parent_repository,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": int(config["lora_alpha"]),
        "lora_dropout": float(config["lora_dropout"]),
        "r": int(config["lora_rank"]),
        "target_modules": sorted(str(value) for value in config["target_modules"]),
        "task_type": "CAUSAL_LM",
        "peft_type": "LORA",
        "use_dora": False,
        "use_rslora": False,
    }


def export_adapter(
    *, artifact: dict[str, Any], checkpoint_path: Path, output_root: Path
) -> None:
    expected_hash = str(artifact["research_checkpoint_sha256"])
    if sha256_path(checkpoint_path) != expected_hash:
        raise ValueError("The DPO checkpoint is not the frozen selected adapter")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("parent_state_identity_sha256") != artifact["parent_state_identity_sha256"]:
        raise ValueError("DPO parent identity does not match selected Stage 3")
    state = checkpoint.get("adapter_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("DPO checkpoint has no adapter state dictionary")
    tensors: dict[str, torch.Tensor] = {}
    for key, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Non-tensor adapter value: {key}")
        if "lora_" not in key or not key.endswith(".weight"):
            raise ValueError(f"Unexpected adapter tensor: {key}")
        tensors[str(key)] = value.detach().cpu().contiguous()

    artifact_id = str(artifact["artifact_id"])
    destination = output_root / artifact_id
    _copy_release_documents(artifact_id, destination)
    destination.mkdir(parents=True, exist_ok=True)
    save_file(tensors, destination / "adapter_model.safetensors", metadata={"format": "pt"})
    (destination / "adapter_config.json").write_text(
        json.dumps(
            _adapter_config(str(artifact["parent_repository"]), checkpoint),
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def write_package_manifest(output_root: Path) -> None:
    rows = []
    for path in sorted(value for value in output_root.rglob("*") if value.is_file()):
        rows.append({
            "bytes": path.stat().st_size,
            "path": path.relative_to(output_root).as_posix(),
            "sha256": sha256_path(path),
        })
    (output_root / "package_manifest.json").write_text(
        json.dumps(
            {"schema_version": "transformer_poetry_hf_package_manifest_v1", "files": rows},
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def copy_repository_root_documents(output_root: Path) -> None:
    copy_verified(ROOT_README, output_root / "README.md")
    copy_verified(CC_BY_NC, output_root / "LICENSE")
    copy_verified(APACHE, output_root / "LICENSES/APACHE-2.0.txt")
    copy_verified(CC_BY, output_root / "LICENSES/CC-BY-4.0.txt")


def validate_output_root(output_root: Path) -> None:
    resolved_root = output_root.resolve()
    try:
        relative = resolved_root.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError:
        return
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("Output root inside the repository must be ignored by Git")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-boundaries", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--include-verified-upstream-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_output_root(args.output_root)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError("Output root must be absent or empty")
    plan = load_json(RELEASE_PLAN)
    copy_repository_root_documents(args.output_root)
    for artifact in plan["artifacts"]:
        if artifact["kind"] == "full_model":
            export_full_model(
                artifact=artifact,
                boundaries_root=args.stage_boundaries,
                output_root=args.output_root,
                include_verified_upstream_files=args.include_verified_upstream_files,
            )
        else:
            export_adapter(
                artifact=artifact,
                checkpoint_path=args.adapter_checkpoint,
                output_root=args.output_root,
            )
    write_package_manifest(args.output_root)


if __name__ == "__main__":
    main()
