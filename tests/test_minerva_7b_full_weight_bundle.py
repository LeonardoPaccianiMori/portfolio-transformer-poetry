import hashlib
import json
import tarfile

import torch

from sonnet_training.minerva_7b_full_weight_bundle import (
    BUNDLE_PATHS,
    package_minerva_7b_full_weight_calibration,
)


def test_full_weight_bundle_contains_windows_but_no_shards_or_weights(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    for relative_path in BUNDLE_PATHS:
        path = repo_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative_path.endswith("calibration_windows.pt"):
            torch.save({"test": torch.tensor([1], dtype=torch.int32)}, path)
        elif relative_path.endswith("encoded/report.json"):
            continue
        else:
            path.write_text(f"fixture: {relative_path}\n", encoding="utf-8")
    windows_path = repo_root / BUNDLE_PATHS[-1]
    windows_sha = hashlib.sha256(windows_path.read_bytes()).hexdigest()
    report_path = repo_root / "data/local/minerva_7b_full_weight/encoded/report.json"
    report_path.write_text(json.dumps({
        "status": "complete",
        "calibration_windows": {
            "path": "data/local/minerva_7b_full_weight/encoded/calibration_windows.pt",
            "sha256": windows_sha,
        },
    }))
    output_path = tmp_path / "bundle.tar.gz"

    report = package_minerva_7b_full_weight_calibration(
        repo_root=repo_root,
        output_path=output_path,
    )

    assert report["output_bytes"] > 0
    with tarfile.open(output_path, "r:gz") as archive:
        names = archive.getnames()
    assert "bundle_manifest.json" in names
    assert str(BUNDLE_PATHS[-1]) in names
    assert not any(name.endswith(".safetensors") for name in names)
    assert not any("int32.bin" in name for name in names)
