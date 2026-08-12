import json
import tarfile
from pathlib import Path

import pytest

from sonnet_training import minerva_7b_v7_bundle as bundle


def test_bundle_paths_exclude_v7_test_material(monkeypatch, tmp_path):
    monkeypatch.setattr(bundle, "PUBLIC_PATHS", ("public.txt",))
    monkeypatch.setattr(
        bundle,
        "LOCAL_EXACT_PATHS",
        (
            "data/local/minerva_7b_v7/window_indexes/manifest.json",
            "data/local/minerva_7b_v7/modern_preservation_validation_v1.jsonl",
            "data/local/minerva_7b_v7/activation_probes_v1.json",
        ),
    )
    paths = [
        "public.txt",
        "data/local/minerva_7b_v7/window_indexes/manifest.json",
        "data/local/minerva_7b_v7/modern_preservation_validation_v1.jsonl",
        "data/local/minerva_7b_v7/activation_probes_v1.json",
        "data/local/minerva_7b_v7/encoded/train-00000.int32.bin",
        "data/local/minerva_7b_v7/encoded/sonnets_test-00000.int32.bin",
        "data/local/minerva_7b_v7/window_indexes/training/stage.jsonl",
        "data/local/minerva_7b_v7/window_indexes/validation/heldout.jsonl",
        "data/local/minerva_7b_full_weight/encoded/paisa_validation-00000.int32.bin",
        "data/local/minerva_7b_full_weight/encoded/paisa_validation.metadata.json",
        "data/local/minerva_7b_full_weight/encoded/paisa_validation.documents.jsonl",
    ]
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("payload")

    selected = bundle.bundle_file_paths(tmp_path)

    assert all("sonnets_test" not in path.name for path in selected)
    assert any(path.name == "train-00000.int32.bin" for path in selected)


def test_small_bundle_is_deterministic_and_verifiable(monkeypatch, tmp_path):
    monkeypatch.setattr(bundle, "bundle_file_paths", lambda root: (root / "a.bin",))
    (tmp_path / "a.bin").write_bytes(b"abc")
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    first_report = bundle.package_v7_execution_bundle(
        repo_root=tmp_path, output_path=first
    )
    second_report = bundle.package_v7_execution_bundle(
        repo_root=tmp_path, output_path=second
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_report["output_sha256"] == second_report["output_sha256"]
    manifest = bundle.verify_v7_execution_bundle(first)
    assert manifest["v7_test_material_included"] is False


def test_bundle_install_is_atomic_when_verification_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(bundle, "bundle_file_paths", lambda root: (root / "a.bin",))
    (tmp_path / "a.bin").write_bytes(b"abc")
    destination = tmp_path / "bundle.tar.gz"
    destination.write_bytes(b"previous-valid-artifact")

    def reject(_path):
        raise ValueError("synthetic verification failure")

    monkeypatch.setattr(bundle, "verify_v7_execution_bundle", reject)
    with pytest.raises(ValueError, match="synthetic"):
        bundle.package_v7_execution_bundle(
            repo_root=tmp_path, output_path=destination
        )

    assert destination.read_bytes() == b"previous-valid-artifact"
    assert not (tmp_path / "bundle.tar.gz.tmp").exists()


def test_bundle_verifier_rejects_unmanifested_member(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        manifest = {
            "bundle_version": bundle.BUNDLE_VERSION,
            "files": [],
        }
        payload = json.dumps(manifest).encode()
        info = tarfile.TarInfo("bundle_manifest.json")
        info.size = len(payload)
        import io

        handle.addfile(info, io.BytesIO(payload))
        extra = tarfile.TarInfo("extra")
        extra.size = 1
        handle.addfile(extra, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="membership"):
        bundle.verify_v7_execution_bundle(archive)
