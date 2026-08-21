#!/usr/bin/env python3
"""Validate tracked HF release metadata and optional local export packages."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = REPOSITORY_ROOT / "release/huggingface"
PLAN_PATH = RELEASE_ROOT / "release_plan.yml"
UPSTREAM_FILE_MANIFEST = RELEASE_ROOT / "upstream_file_manifest.json"
TEXT_NAMES = {
    "README.md", "NOTICE.md", "RIGHTS_SCOPE.md", "TRAINING_CONTENT_SUMMARY.md",
    "LICENSE", "APACHE-2.0.txt", "CC-BY-4.0.txt", "adapter_config.json",
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "lineage.json", "tokenizer_config.json", "special_tokens_map.json",
    "tokenizer.json", "chat_template.jinja",
}
PRIVATE_PATTERNS = (
    re.compile(r"/home/", re.IGNORECASE),
    re.compile(r"file:" r"//", re.IGNORECASE),
    re.compile(r"artifacts/local/", re.IGNORECASE),
    re.compile(r"runs/minerva", re.IGNORECASE),
    re.compile(r"(?:password|api[_-]?key|access[_-]?token)\s*[:=]", re.IGNORECASE),
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["artifact_id"]): row for row in plan["artifacts"]}


def validate_tracked_release_metadata() -> None:
    plan = load_json(PLAN_PATH)
    upstream_files = load_json(UPSTREAM_FILE_MANIFEST)
    if plan["status"] != "published_public_ungated":
        raise ValueError("Unexpected Hugging Face publication status")
    if plan["license_metadata"] != "cc-by-nc-4.0":
        raise ValueError("Unexpected weight-license metadata")
    artifacts = _artifact_map(plan)
    if set(artifacts) != {"stage1", "stage2", "stage3", "dpo_adapter"}:
        raise ValueError("The release plan must define exactly four artifacts")
    repository = "LPM93/teaching-transformers-classical-italian-sonnets"
    if plan["repository"] != repository:
        raise ValueError("Unexpected single Hugging Face repository")
    if plan["repository_visibility"] != "public_ungated":
        raise ValueError("Unexpected Hugging Face repository visibility")
    if plan["publication_url"] != f"https://huggingface.co/{repository}":
        raise ValueError("Unexpected Hugging Face publication URL")
    if plan["publication_commit"] != "3581abbb1023c77f784b37aa152cdb6c0447fa73":
        raise ValueError("Unexpected Hugging Face publication commit")
    if plan["publication_date"] != "2026-08-21":
        raise ValueError("Unexpected Hugging Face publication date")
    if plan["package_manifest_sha256"] != "cedc265ece2b0faf81cf03861ac2f64faa8dea90856dd919a5990e238f2746e2":
        raise ValueError("Unexpected Hugging Face package-manifest identity")
    if plan["package_bytes"] != 44_492_691_499:
        raise ValueError("Unexpected Hugging Face package size")
    if plan["package_files"] != 69:
        raise ValueError("Unexpected Hugging Face package file count")
    if plan["decision_record_id"] != "decision-2026-08-21-huggingface-single-repository-release":
        raise ValueError("Unexpected model-release decision record")
    if plan["authorization_date"] != "2026-08-21":
        raise ValueError("Unexpected model-release authorization date")
    if {artifact["repository"] for artifact in artifacts.values()} != {repository}:
        raise ValueError("Every artifact must use the single Hugging Face repository")
    if {artifact["subfolder"] for artifact in artifacts.values()} != {
        "stage1", "stage2", "stage3", "dpo_adapter",
    }:
        raise ValueError("The single-repository subfolder map is incomplete")
    if artifacts["dpo_adapter"]["parent_repository"] != repository:
        raise ValueError("The adapter must identify the single repository as its base location")
    if artifacts["dpo_adapter"]["parent_subfolder"] != "stage3":
        raise ValueError("The adapter must identify Stage 3 as its exact base subfolder")
    if artifacts["stage3"]["selected_update"] != 120:
        raise ValueError("Stage 3 must identify selected update 120")
    expected_upstream_files = {
        "config.json", "generation_config.json", "chat_template.jinja",
        "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    }
    if set(upstream_files["package_files"]) != expected_upstream_files:
        raise ValueError("Verified upstream-file manifest has unexpected coverage")
    for name, row in upstream_files["package_files"].items():
        for field in ("sha256", "parent_sha256"):
            if field in row and re.fullmatch(r"[0-9a-f]{64}", str(row[field])) is None:
                raise ValueError(f"Invalid {field} for verified upstream file {name}")
    reviewed_stage_records: dict[str, dict[str, Any]] = {}
    for artifact_id, artifact in artifacts.items():
        root = RELEASE_ROOT / artifact_id
        required = {
            "README.md", "NOTICE.md", "RIGHTS_SCOPE.md",
            "TRAINING_CONTENT_SUMMARY.md", "lineage.json",
        }
        missing = sorted(name for name in required if not (root / name).is_file())
        if missing:
            raise ValueError(f"Missing tracked {artifact_id} files: {missing}")
        lineage = load_json(root / "lineage.json")
        if lineage["artifact_id"] != artifact_id:
            raise ValueError(f"Lineage artifact mismatch for {artifact_id}")
        if lineage["artifact"] != artifact:
            raise ValueError(f"Lineage artifact identity mismatch for {artifact_id}")
        if lineage["parent"] != plan["parent"]:
            raise ValueError(f"Lineage parent mismatch for {artifact_id}")
        stage_sequence = {
            "stage1": (("stage_1_historical_general", artifacts["stage1"]["selected_windows"]),),
            "stage2": (
                ("stage_1_historical_general", artifacts["stage1"]["selected_windows"]),
                ("stage_2_non_sonnet_poetry", artifacts["stage2"]["selected_windows"]),
            ),
            "stage3": (
                ("stage_1_historical_general", artifacts["stage1"]["selected_windows"]),
                ("stage_2_non_sonnet_poetry", artifacts["stage2"]["selected_windows"]),
                ("stage_3_sonnets", artifacts["stage3"]["selected_windows"]),
            ),
            "dpo_adapter": (
                ("stage_1_historical_general", artifacts["stage1"]["selected_windows"]),
                ("stage_2_non_sonnet_poetry", artifacts["stage2"]["selected_windows"]),
                ("stage_3_sonnets", artifacts["stage3"]["selected_windows"]),
            ),
        }[artifact_id]
        completed_stages = lineage["completed_training_stages"]
        if len(completed_stages) != len(stage_sequence):
            raise ValueError(f"Lineage stage count mismatch for {artifact_id}")
        accumulated_families: dict[str, int] = {}
        for stage, (expected_stage_id, expected_windows) in zip(completed_stages, stage_sequence):
            if stage["stage_id"] != expected_stage_id or stage["selected_windows"] != expected_windows:
                raise ValueError(f"Lineage stage boundary mismatch for {artifact_id}")
            expected_tokens = int(expected_windows) * 2048
            if stage["target_tokens"] != expected_tokens:
                raise ValueError(f"Lineage stage token mismatch for {artifact_id}")
            if re.fullmatch(r"[0-9a-f]{64}", stage["selected_window_rows_sha256"]) is None:
                raise ValueError(f"Invalid selected-window hash for {artifact_id}")
            if sum(stage["component_target_tokens"].values()) != expected_tokens:
                raise ValueError(f"Lineage component total mismatch for {artifact_id}")
            if sum(stage["source_family_target_tokens"].values()) != expected_tokens:
                raise ValueError(f"Lineage source-family total mismatch for {artifact_id}")
            if expected_stage_id in reviewed_stage_records:
                if stage != reviewed_stage_records[expected_stage_id]:
                    raise ValueError(f"Shared lineage stage differs for {artifact_id}")
            else:
                reviewed_stage_records[expected_stage_id] = stage
            for family, tokens in stage["source_family_target_tokens"].items():
                accumulated_families[family] = accumulated_families.get(family, 0) + int(tokens)
        if accumulated_families != lineage["cumulative_source_family_target_tokens"]:
            raise ValueError(f"Lineage cumulative source-family mismatch for {artifact_id}")
        if sum(accumulated_families.values()) != lineage["cumulative_target_tokens"]:
            raise ValueError(f"Lineage token mismatch for {artifact_id}")
        card = (root / "README.md").read_text(encoding="utf-8")
        normalized_card = " ".join(card.split()).lower()
        required_card_phrases = (
            "license: cc-by-nc-4.0",
            "not trained from scratch",
            "0/100",
            "12/20",
            "ai-judged",
        )
        if any(phrase not in normalized_card for phrase in required_card_phrases):
            raise ValueError(f"Required claim boundary missing from {artifact_id} card")
        rights = (root / "RIGHTS_SCOPE.md").read_text(encoding="utf-8")
        for phrase in ("Apache-2.0", "CC BY-NC 4.0", "training-data licenses do not govern the weights"):
            if phrase not in rights:
                raise ValueError(f"Rights-scope boundary missing from {artifact_id}")
    _scan_text_files(RELEASE_ROOT)


def _scan_text_files(root: Path) -> None:
    for path in sorted(value for value in root.rglob("*") if value.is_file() and value.name in TEXT_NAMES):
        text = path.read_text(encoding="utf-8")
        for pattern in PRIVATE_PATTERNS:
            if pattern.search(text):
                raise ValueError(f"Private or machine-local pattern in {path}: {pattern.pattern}")


def _validate_model_safetensors(root: Path) -> None:
    index = load_json(root / "model.safetensors.index.json")
    referenced = set(index["weight_map"].values())
    actual = {path.name for path in root.glob("model-*.safetensors")}
    if referenced != actual:
        raise ValueError(f"Shard/index mismatch in {root}")
    indexed_keys = set(index["weight_map"])
    actual_keys: set[str] = set()
    for name in sorted(actual):
        with safe_open(root / name, framework="pt", device="cpu") as handle:
            if handle.metadata() != {"format": "pt"}:
                raise ValueError(f"Unexpected safetensors metadata in {root / name}")
            actual_keys.update(handle.keys())
    if indexed_keys != actual_keys:
        raise ValueError(f"Tensor-key mismatch in {root}")


def _validate_adapter(
    root: Path, checkpoint_path: Path | None, artifact: dict[str, Any]
) -> None:
    config = load_json(root / "adapter_config.json")
    if config["base_model_name_or_path"] != artifact["parent_repository"]:
        raise ValueError("Adapter does not identify the planned single repository")
    exported = load_file(root / "adapter_model.safetensors", device="cpu")
    if not exported or any("lora_" not in key for key in exported):
        raise ValueError("Unexpected PEFT adapter tensor set")
    with safe_open(root / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
        if handle.metadata() not in ({"format": "pt"}, None):
            raise ValueError("Unexpected adapter safetensors metadata")
    if checkpoint_path is not None:
        if sha256_path(checkpoint_path) != artifact["research_checkpoint_sha256"]:
            raise ValueError("Adapter checkpoint is not the selected research checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("parent_state_identity_sha256") != artifact["parent_state_identity_sha256"]:
            raise ValueError("Adapter checkpoint parent identity does not match selected Stage 3")
        source = checkpoint["adapter_state_dict"]
        if set(source) != set(exported):
            raise ValueError("Exported adapter keys differ from the selected checkpoint")
        for key in source:
            if not torch.equal(source[key].detach().cpu(), exported[key]):
                raise ValueError(f"Exported adapter tensor differs: {key}")


def _load_models(package_root: Path, checkpoint_path: Path) -> None:
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict

    for artifact_id in ("stage1", "stage2", "stage3"):
        model = AutoModelForCausalLM.from_pretrained(
            package_root,
            subfolder=artifact_id,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        if artifact_id == "stage3":
            attached = PeftModel.from_pretrained(
                model,
                package_root,
                subfolder="dpo_adapter",
                local_files_only=True,
            )
            attached.eval()
            fixed_ids = torch.tensor([[1, 42, 314, 2718]], dtype=torch.long)
            with torch.inference_mode():
                exported_logits = attached(input_ids=fixed_ids).logits.detach().cpu()
            del attached, model
            gc.collect()

            model = AutoModelForCausalLM.from_pretrained(
                package_root,
                subfolder=artifact_id,
                local_files_only=True,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            config = checkpoint["config"]
            source_model = get_peft_model(
                model,
                LoraConfig(
                    base_model_name_or_path=str(package_root / artifact_id),
                    bias="none",
                    inference_mode=True,
                    lora_alpha=int(config["lora_alpha"]),
                    lora_dropout=float(config["lora_dropout"]),
                    r=int(config["lora_rank"]),
                    target_modules=sorted(str(value) for value in config["target_modules"]),
                    task_type="CAUSAL_LM",
                ),
            )
            set_peft_model_state_dict(source_model, checkpoint["adapter_state_dict"])
            source_model.eval()
            with torch.inference_mode():
                source_logits = source_model(input_ids=fixed_ids).logits.detach().cpu()
            if not torch.equal(exported_logits, source_logits):
                raise ValueError("Exported PEFT reconstruction changes fixed-token logits")
            del source_model, model, checkpoint, exported_logits, source_logits
        else:
            del model
        gc.collect()


def validate_package(
    package_root: Path, *, checkpoint_path: Path | None = None, load_models: bool = False
) -> None:
    plan = load_json(PLAN_PATH)
    artifacts = _artifact_map(plan)
    artifact_folders = set(artifacts)
    root_files = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file()
        and path.relative_to(package_root).parts[0] not in artifact_folders
    }
    if root_files != set(plan["repository_root_allowlist"]):
        raise ValueError(f"Repository-root allowlist failure: {sorted(root_files)}")
    _scan_text_files(package_root)
    for artifact_id, artifact in artifacts.items():
        root = package_root / artifact_id
        if not root.is_dir():
            raise ValueError(f"Missing package directory: {root}")
        allowed = set(plan["adapter_allowlist"] if artifact["kind"] == "peft_adapter" else plan["full_model_allowlist"])
        if artifact["kind"] == "full_model":
            allowed.update(plan["optional_verified_upstream_files"])
        actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
        unexpected = sorted(actual - allowed)
        required = set(plan["adapter_allowlist"] if artifact["kind"] == "peft_adapter" else plan["full_model_allowlist"])
        missing = sorted(required - actual)
        if unexpected or missing:
            raise ValueError(f"Allowlist failure for {artifact_id}: missing={missing}, unexpected={unexpected}")
        if artifact["kind"] == "full_model":
            _validate_model_safetensors(root)
        else:
            _validate_adapter(root, checkpoint_path, artifact)
    manifest_path = package_root / "package_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != "transformer_poetry_hf_package_manifest_v1":
        raise ValueError("Unexpected package-manifest schema")
    manifest_rows = manifest.get("files")
    if not isinstance(manifest_rows, list):
        raise ValueError("Package manifest must contain a file list")
    listed_paths = [str(row["path"]) for row in manifest_rows]
    if len(listed_paths) != len(set(listed_paths)):
        raise ValueError("Duplicate package-manifest path")
    if any(Path(value).is_absolute() or ".." in Path(value).parts for value in listed_paths):
        raise ValueError("Unsafe package-manifest path")
    actual_manifested = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if set(listed_paths) != actual_manifested:
        raise ValueError("Package manifest does not cover the complete package")
    for row in manifest_rows:
        path = package_root / str(row["path"])
        if path.stat().st_size != row["bytes"] or sha256_path(path) != row["sha256"]:
            raise ValueError(f"Package-manifest mismatch: {path}")
    if load_models:
        if checkpoint_path is None:
            raise ValueError("Model-load certification requires the selected adapter checkpoint")
        _load_models(package_root, checkpoint_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--load-models", action="store_true")
    parser.add_argument("--certify-release-candidate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_tracked_release_metadata()
    if args.certify_release_candidate and (
        args.package_root is None or args.adapter_checkpoint is None
    ):
        raise ValueError(
            "Release-candidate certification requires --package-root and --adapter-checkpoint"
        )
    if args.package_root is not None:
        validate_package(
            args.package_root,
            checkpoint_path=args.adapter_checkpoint,
            load_models=args.load_models or args.certify_release_candidate,
        )


if __name__ == "__main__":
    main()
