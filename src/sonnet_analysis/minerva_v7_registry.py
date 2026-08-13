"""Fail-closed registry and integrity audit for the seven Minerva V7 states."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REGISTRY_VERSION = "minerva_7b_v7_research_registry_v1"


@dataclass(frozen=True)
class ModelState:
    state_id: str
    order: int
    stage_id: str | None
    snapshot_role: str
    expected_update: int | None


MODEL_STATES = (
    ModelState("untouched_parent", 0, None, "published_parent", None),
    ModelState("stage_1_midpoint", 1, "stage_1_historical_general", "midpoint", 1033),
    ModelState(
        "stage_1_selected", 2, "stage_1_historical_general",
        "validation_selected_endpoint", None,
    ),
    ModelState("stage_2_midpoint", 3, "stage_2_non_sonnet_poetry", "midpoint", 380),
    ModelState(
        "stage_2_selected", 4, "stage_2_non_sonnet_poetry",
        "validation_selected_endpoint", None,
    ),
    ModelState("stage_3_midpoint", 5, "stage_3_sonnets", "midpoint", 68),
    ModelState(
        "stage_3_selected", 6, "stage_3_sonnets",
        "validation_selected_endpoint", None,
    ),
)

COMPARISONS = tuple(
    {"comparison_id": f"{left.state_id}_to_{right.state_id}", "left": left.state_id, "right": right.state_id}
    for left, right in zip(MODEL_STATES, MODEL_STATES[1:])
) + (
    {
        "comparison_id": "untouched_parent_to_stage_3_selected",
        "left": "untouched_parent",
        "right": "stage_3_selected",
    },
)


def default_state_paths(run_dir: Path, parent_model_dir: Path | None = None) -> dict[str, Path | None]:
    """Return stable expected locations without requiring unfinished states."""

    paths: dict[str, Path | None] = {"untouched_parent": parent_model_dir}
    for state in MODEL_STATES[1:]:
        assert state.stage_id is not None
        if state.snapshot_role == "midpoint":
            paths[state.state_id] = (
                run_dir / "analysis_snapshots" /
                f"{state.stage_id}_update_{state.expected_update:06d}"
            )
        else:
            paths[state.state_id] = run_dir / "stage_boundaries" / f"{state.stage_id}_selected"
    return paths


def audit_research_states(
    *,
    run_dir: Path,
    protocol_path: Path,
    parent_model_dir: Path | None = None,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    """Audit available states while distinguishing expected absence from corruption."""

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_protocol_sha = str(protocol.get("protocol_sha256") or _sha256(protocol_path))
    paths = default_state_paths(run_dir, parent_model_dir)
    rows = []
    for state in MODEL_STATES:
        path = paths[state.state_id]
        if state.state_id == "untouched_parent":
            rows.append(_audit_parent(state, path))
        else:
            assert path is not None
            rows.append(
                _audit_snapshot(
                    state, path, expected_protocol_sha=expected_protocol_sha,
                    verify_hashes=verify_hashes,
                )
            )

    _validate_cross_state_lineage(rows, protocol)

    row_by_id = {row["state_id"]: row for row in rows}
    comparisons = []
    for comparison in COMPARISONS:
        ready = all(row_by_id[key]["status"] == "complete" for key in (comparison["left"], comparison["right"]))
        comparisons.append({**comparison, "ready": ready})
    statuses = {status: sum(row["status"] == status for row in rows) for status in ("complete", "missing", "partial", "invalid")}
    return {
        "registry_version": REGISTRY_VERSION,
        "run_dir": str(run_dir),
        "protocol_path": str(protocol_path),
        "protocol_sha256": expected_protocol_sha,
        "hash_verification_performed": verify_hashes,
        "states": rows,
        "comparisons": comparisons,
        "status_counts": statuses,
        "all_seven_states_complete": statuses["complete"] == 7,
        "causal_experiments_authorized": False,
    }


def _audit_parent(state: ModelState, path: Path | None) -> dict[str, Any]:
    row = {**asdict(state), "path": str(path) if path else None, "issues": []}
    if path is None or not path.exists():
        row["status"] = "missing"
        row["issues"].append("pinned parent cache path was not supplied or is absent")
        return row
    model_dir = path / "model" if (path / "model").is_dir() else path
    config = model_dir / "config.json"
    weights = sorted(model_dir.glob("*.safetensors"))
    if not config.is_file() or not weights:
        row["status"] = "partial"
        row["issues"].append("parent directory lacks config.json or SafeTensors weights")
    else:
        row["status"] = "complete"
        row["model_dir"] = str(model_dir)
        row["weight_bytes"] = sum(item.stat().st_size for item in weights)
    return row


def _audit_snapshot(
    state: ModelState,
    path: Path,
    *,
    expected_protocol_sha: str,
    verify_hashes: bool,
) -> dict[str, Any]:
    row = {**asdict(state), "path": str(path), "issues": []}
    manifest_path = path / "manifest.json"
    if not path.exists():
        row["status"] = "missing"
        row["issues"].append("state has not been produced or downloaded")
        return row
    if not manifest_path.is_file():
        row["status"] = "partial"
        row["issues"].append("snapshot directory has no manifest.json")
        return row
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        row["status"] = "invalid"
        row["issues"].append(f"manifest cannot be read: {error}")
        return row

    metadata = manifest.get("metadata", {})
    checks = {
        "artifact_type": metadata.get("artifact_type") == "model_only_analysis_snapshot",
        "stage_id": metadata.get("stage_id") == state.stage_id,
        "snapshot_role": metadata.get("snapshot_role") == state.snapshot_role,
        "protocol_sha256": metadata.get("protocol_sha256") == expected_protocol_sha,
        "expected_update": state.expected_update is None or int(metadata.get("update", -1)) == state.expected_update,
    }
    row["metadata"] = {
        key: metadata.get(key)
        for key in (
            "stage_id", "snapshot_role", "update", "protocol_sha256", "git_commit",
            "preceding_model_identity_sha256", "source_candidate_manifest_sha256",
        )
        if key in metadata
    }
    row["manifest_sha256"] = _sha256(manifest_path)
    if state.snapshot_role == "validation_selected_endpoint":
        endpoint_fields = (
            "selected_metrics", "parent_baseline_metrics", "validation_history",
            "source_candidate_manifest_sha256", "preceding_model_identity_sha256",
        )
        for field in endpoint_fields:
            if field not in metadata:
                row["issues"].append(f"selected endpoint lacks metadata: {field}")
    for name, passed in checks.items():
        if not passed:
            row["issues"].append(f"manifest metadata mismatch: {name}")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        row["issues"].append("manifest file inventory is absent")
        row["status"] = "invalid"
        return row
    missing = []
    wrong_size = []
    wrong_hash = []
    weight_bytes = 0
    for item in files:
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            row["issues"].append("manifest contains an unsafe path")
            continue
        file_path = path / relative
        if not file_path.is_file():
            missing.append(str(relative))
            continue
        if file_path.stat().st_size != int(item["bytes"]):
            wrong_size.append(str(relative))
            continue
        if str(relative).endswith(".safetensors"):
            weight_bytes += file_path.stat().st_size
        if verify_hashes and _sha256(file_path) != item["sha256"]:
            wrong_hash.append(str(relative))
    if missing:
        row["issues"].append(f"missing manifest files: {len(missing)}")
    if wrong_size:
        row["issues"].append(f"wrong-size manifest files: {len(wrong_size)}")
    if wrong_hash:
        row["issues"].append(f"hash-mismatched manifest files: {len(wrong_hash)}")
    row["weight_bytes"] = weight_bytes
    row["verified_file_count"] = len(files) - len(missing) - len(wrong_size) - len(wrong_hash)
    if missing or wrong_size:
        row["status"] = "partial"
    elif row["issues"]:
        row["status"] = "invalid"
    else:
        row["status"] = "complete"
    return row


def _validate_cross_state_lineage(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> None:
    """Reject internally valid states assembled from different training lineages."""

    row_by_id = {row["state_id"]: row for row in rows}
    expected_parent = hashlib.sha256(
        f"{protocol['model']['model_id']}@{protocol['model']['revision']}".encode("utf-8")
    ).hexdigest() if isinstance(protocol.get("model"), dict) else None
    stage_pairs = (
        ("stage_1_midpoint", "stage_1_selected", None),
        ("stage_2_midpoint", "stage_2_selected", "stage_1_selected"),
        ("stage_3_midpoint", "stage_3_selected", "stage_2_selected"),
    )
    for midpoint_id, selected_id, preceding_id in stage_pairs:
        midpoint = row_by_id[midpoint_id]
        selected = row_by_id[selected_id]
        complete = [row for row in (midpoint, selected) if row["status"] == "complete"]
        preceding_values = {
            row.get("metadata", {}).get("preceding_model_identity_sha256")
            for row in complete
        }
        preceding_values.discard(None)
        if len(preceding_values) > 1:
            for row in complete:
                _invalidate(row, "midpoint and selected endpoint have different preceding identities")
        expected_preceding = (
            expected_parent if preceding_id is None else row_by_id[preceding_id].get("manifest_sha256")
        )
        if expected_preceding is not None:
            for row in complete:
                actual = row.get("metadata", {}).get("preceding_model_identity_sha256")
                if actual != expected_preceding:
                    _invalidate(row, "preceding model identity breaks the seven-state lineage")


def _invalidate(row: dict[str, Any], issue: str) -> None:
    if issue not in row["issues"]:
        row["issues"].append(issue)
    row["status"] = "invalid"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
