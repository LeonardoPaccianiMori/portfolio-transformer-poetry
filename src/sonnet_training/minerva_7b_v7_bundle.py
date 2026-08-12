"""Hash-verified private transfer bundle for Minerva 7B V7 execution."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any


BUNDLE_VERSION = "minerva_7b_v7_execution_bundle_v2"
PUBLIC_PATHS = (
    "configs/minerva_7b_v7_full_weight_protocol.json",
    "configs/minerva_7b_v7_execution.json",
    "configs/minerva_7b_v7_hardware_qualification.json",
    "configs/minerva_7b_preservation_prompts.json",
    "configs/minerva_7b_parent_decoding_confirmation_prompts.json",
    "data/metadata/minerva_7b_v7_sampling_policy_v1.json",
    "reports/minerva_7b_v7_encoded_data_v1.json",
    "reports/minerva_7b_v7_stage_windows_v1.json",
    "requirements.txt",
    "requirements/minerva_qlora.txt",
    "scripts/qualify_minerva_7b_v7_full_weight.py",
    "scripts/qualify_minerva_7b_v7_dual_a6000.py",
    "scripts/run_minerva_7b_v7_qualification_worker.py",
    "scripts/train_minerva_7b_v7_full_weight.py",
    "src/sonnet_training/cuda_compat.py",
    "src/sonnet_training/minerva_7b_full_weight_calibration.py",
    "src/sonnet_training/minerva_7b_model_audit.py",
    "src/sonnet_training/minerva_7b_qlora.py",
    "src/sonnet_training/minerva_7b_v7_execution.py",
    "src/sonnet_training/minerva_7b_v7_gpu_qualification.py",
    "src/sonnet_training/minerva_7b_v7_protocol.py",
    "src/sonnet_training/minerva_7b_v7_qualification.py",
    "src/sonnet_training/minerva_7b_v7_trainer.py",
)
LOCAL_EXACT_PATHS = (
    "data/local/minerva_7b_v7/window_indexes/manifest.json",
    "data/local/minerva_7b_v7/modern_preservation_validation_v1.jsonl",
    "data/local/minerva_7b_v7/activation_probes_v1.json",
)
EXCLUDED_POOL_IDS = {"sonnets_test"}


def bundle_file_paths(repo_root: Path) -> tuple[Path, ...]:
    """Resolve the execution payload while excluding frozen V7 test artifacts."""

    paths = [repo_root / value for value in PUBLIC_PATHS + LOCAL_EXACT_PATHS]
    encoded_dir = repo_root / "data/local/minerva_7b_v7/encoded"
    for path in sorted(encoded_dir.iterdir()):
        if not path.is_file() or path.name.startswith("sonnets_test"):
            continue
        paths.append(path)
    index_root = repo_root / "data/local/minerva_7b_v7/window_indexes"
    for category in ("training", "validation"):
        paths.extend(sorted((index_root / category).glob("*.jsonl")))
    modern_dir = repo_root / "data/local/minerva_7b_full_weight/encoded"
    paths.extend(
        [
            modern_dir / "paisa_validation-00000.int32.bin",
            modern_dir / "paisa_validation.metadata.json",
            modern_dir / "paisa_validation.documents.jsonl",
        ]
    )
    unique = tuple(dict.fromkeys(paths))
    for path in unique:
        if not path.is_file():
            raise FileNotFoundError(f"required V7 bundle file is missing: {path}")
        if "sonnets_test" in path.name or "/test/" in path.as_posix():
            raise ValueError("V7 test material may not enter the pre-training bundle")
    return unique


def package_v7_execution_bundle(
    *, repo_root: Path, output_path: Path
) -> dict[str, Any]:
    """Write a deterministic gzip-compressed tar and embedded file manifest."""

    files = []
    for path in bundle_file_paths(repo_root):
        relative = path.relative_to(repo_root).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    files.sort(key=lambda row: row["path"])
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "files": files,
        "scope": (
            "V7 training and validation token shards, frozen window indexes, PAISA "
            "preservation validation, activation probes, and execution code"
        ),
        "v7_test_material_included": False,
        "model_weights_included": False,
        "raw_source_text_included": False,
        "public_distribution": False,
        "extraction_target": "root of the exact public checkpoint-8G repository clone",
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                filename="", fileobj=raw, mode="wb", mtime=0
            ) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as archive:
                    for row in files:
                        data = (repo_root / row["path"]).read_bytes()
                        _add_bytes(archive, row["path"], data)
                    _add_bytes(archive, "bundle_manifest.json", manifest_bytes)
            raw.flush()
            os.fsync(raw.fileno())
        verify_v7_execution_bundle(temporary)
        os.replace(temporary, output_path)
        _fsync_directory(output_path.parent)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        **manifest,
        "output_path": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": _sha256(output_path),
    }


def verify_v7_execution_bundle(path: Path) -> dict[str, Any]:
    """Stream-verify every archive member against the embedded manifest."""

    with tarfile.open(path, mode="r:gz") as archive:
        names = archive.getnames()
        if len(names) != len(set(names)) or "bundle_manifest.json" not in names:
            raise ValueError("V7 bundle has duplicate members or no manifest")
        manifest_file = archive.extractfile("bundle_manifest.json")
        if manifest_file is None:
            raise ValueError("V7 bundle manifest is unreadable")
        manifest = json.loads(manifest_file.read())
        if manifest.get("bundle_version") != BUNDLE_VERSION:
            raise ValueError("unexpected V7 bundle version")
        expected = {row["path"]: row for row in manifest["files"]}
        if set(names) != set(expected) | {"bundle_manifest.json"}:
            raise ValueError("V7 bundle membership differs from its manifest")
        for name, row in expected.items():
            if "sonnets_test" in name or "/test/" in name:
                raise ValueError("V7 test material is present in the bundle")
            member = archive.extractfile(name)
            if member is None:
                raise ValueError(f"V7 bundle member is unreadable: {name}")
            digest = hashlib.sha256()
            size = 0
            for block in iter(lambda: member.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
            if size != int(row["bytes"]) or digest.hexdigest() != row["sha256"]:
                raise ValueError(f"V7 bundle member failed verification: {name}")
    return manifest


def _add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
