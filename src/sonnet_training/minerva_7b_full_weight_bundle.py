"""Package the small remote Minerva full-weight calibration payload."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


BUNDLE_VERSION = "minerva_7b_full_weight_h100_calibration_bundle_v1"
BUNDLE_PATHS = (
    "configs/minerva_7b_full_weight_calibration.json",
    "docs/minerva_7b_full_weight_calibration_protocol.md",
    "requirements.txt",
    "requirements/minerva_qlora.txt",
    "scripts/calibrate_minerva_7b_full_weight.py",
    "src/sonnet_training/cuda_compat.py",
    "src/sonnet_training/minerva_7b_full_weight_calibration.py",
    "src/sonnet_training/minerva_7b_full_weight_data.py",
    "src/sonnet_training/minerva_7b_qlora.py",
    "src/sonnet_corpus/paisa_build.py",
    "data/local/minerva_7b_full_weight/encoded/report.json",
    "data/local/minerva_7b_full_weight/encoded/calibration_windows.pt",
)


def package_minerva_7b_full_weight_calibration(
    *, repo_root: Path, output_path: Path
) -> dict[str, Any]:
    """Create a verified archive without full corpus shards or model weights."""
    files = []
    for relative_path in BUNDLE_PATHS:
        path = repo_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"required calibration bundle file is missing: {path}")
        files.append({
            "path": relative_path,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    report_path = repo_root / "data/local/minerva_7b_full_weight/encoded/report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise ValueError("full-weight data report must be complete before packaging")
    calibration = report.get("calibration_windows")
    if not isinstance(calibration, dict):
        raise ValueError("full-weight data report is missing calibration windows")
    windows_path = repo_root / str(calibration["path"])
    if _sha256(windows_path) != calibration.get("sha256"):
        raise ValueError("calibration-window hash does not match the data report")

    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": files,
        "data_scope": (
            "seven deterministic 512-token calibration windows plus aggregate "
            "full-corpus metadata; no full token shards, text, model weights, or checkpoints"
        ),
        "license_lineage": (
            "PAISA-derived windows remain non-commercial CC BY-NC-SA project data; "
            "the remote VM is temporary compute, not a distribution endpoint"
        ),
        "extraction_target": "root of an up-to-date portfolio-transformer-poetry clone",
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, mode="w:gz") as archive:
        for row in files:
            archive.add(repo_root / row["path"], arcname=row["path"], recursive=False)
        info = tarfile.TarInfo("bundle_manifest.json")
        info.size = len(manifest_bytes)
        info.mtime = 0
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(manifest_bytes))
    return {
        **manifest,
        "output_path": str(output_path),
        "output_bytes": output_path.stat().st_size,
        "output_sha256": _sha256(output_path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
