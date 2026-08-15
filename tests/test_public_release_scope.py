from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_public_release_inventory.py"


def load_inventory_module():
    spec = importlib.util.spec_from_file_location("release_inventory", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_current_and_published_history_manifests_are_complete():
    inventory = load_inventory_module()
    assert inventory.structural_errors(["origin/main"]) == []


def test_raw_history_parser_preserves_non_ascii_corpus_paths():
    inventory = load_inventory_module()
    oid = "1" * 40
    payload = (
        f":000000 100644 {'0' * 40} {oid} A".encode("ascii")
        + b"\0data/processed/example/citt\xc3\xa0.txt\0"
    )
    assert inventory.parse_raw_history_pairs(payload) == [
        ("data/processed/example/città.txt", oid)
    ]
    assert inventory.artifact_class("data/processed/example/città.txt") == "processed_corpus"


def test_license_scope_map_is_classified_as_license_or_notice():
    inventory = load_inventory_module()
    assert inventory.artifact_class("LICENSE.md") == "license_or_notice"


def test_clearance_validation_is_fail_closed():
    inventory = load_inventory_module()
    unresolved = {field: inventory.PENDING for field in inventory.FIELDS}
    unresolved["repository_relative_path"] = "example.txt"
    assert inventory.clearance_errors([unresolved])


def test_clearance_requires_complete_provenance_and_notice_fields():
    inventory = load_inventory_module()
    row = {field: "not_applicable" for field in inventory.FIELDS}
    row.update({
        "repository_relative_path": "example.txt",
        "scope": "current_tree",
        "current_tree_disposition": "approved_for_public_tree",
        "historical_retention_disposition": "approved_for_public_history",
        "privacy_security_disposition": "approved",
        "decision_authority_role": "reviewer",
        "decision_record_id": "memo-id",
        "review_date": "2026-08-15",
        "modifications_made_by_project": "needs_review",
    })
    assert any("unresolved modifications_made_by_project" in error for error in inventory.clearance_errors([row]))


def test_clearance_halts_for_required_current_or_history_removal():
    inventory = load_inventory_module()
    resolved = {field: "not_applicable" for field in inventory.FIELDS}
    resolved.update({
        "repository_relative_path": "example.txt",
        "scope": "current_tree",
        "git_blob_oid": "example",
        "sha256": "example",
        "current_tree_disposition": "remove_from_current_tree",
        "historical_retention_disposition": "approved_for_public_history",
        "privacy_security_disposition": "approved",
        "decision_authority_role": "reviewer",
        "decision_record_id": "memo-id",
        "review_date": "2026-08-15",
    })
    assert any("still present" in error for error in inventory.clearance_errors([resolved]))

    resolved["scope"] = "published_history"
    resolved["current_tree_disposition"] = "not_applicable"
    resolved["historical_retention_disposition"] = "remove_from_history"
    assert any("destructive-action plan" in error for error in inventory.clearance_errors([resolved]))


def test_release_allowlist_prohibits_manual_assets():
    policy = json.loads((ROOT / "release/github_release_allowlist.yml").read_text(encoding="utf-8"))
    assert policy["automatic_source_snapshots_only"] is True
    assert policy["manual_assets"] == []
    assert {"model_weights", "corpus_archives", "preferences", "generations", "checkpoints", "adapters"} <= set(policy["prohibited_manual_asset_classes"])
