"""Paired representation, attention, and logit analysis for extracted V7 probes."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPRESENTATION_VERSION = "minerva_7b_v7_representation_analysis_v1"


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float | None:
    """Compute linear CKA across paired samples, returning null for zero variance."""

    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("linear CKA requires paired 2D sample matrices")
    left = left.double() - left.double().mean(dim=0, keepdim=True)
    right = right.double() - right.double().mean(dim=0, keepdim=True)
    cross = torch.linalg.matrix_norm(left.T @ right) ** 2
    left_norm = torch.linalg.matrix_norm(left.T @ left)
    right_norm = torch.linalg.matrix_norm(right.T @ right)
    denominator = left_norm * right_norm
    return float(cross / denominator) if denominator > 0 else None


def analyze_state_pair(
    *, left_state_dir: Path, right_state_dir: Path, comparison_id: str,
    authoritative: bool = True,
    expected_state_identities: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compare matching pooled probes without loading raw sequence activations."""

    left = _probe_catalog(left_state_dir, authoritative=authoritative)
    right = _probe_catalog(right_state_dir, authoritative=authoritative)
    if expected_state_identities is not None:
        for label, completion, expected in (
            ("left", left["completion"], expected_state_identities["left"]),
            ("right", right["completion"], expected_state_identities["right"]),
        ):
            if completion.get("state", {}).get("state_identity_sha256") != expected:
                raise ValueError(f"{label} extraction state identity mismatch")
    left_probes = left["probes"]
    right_probes = right["probes"]
    if set(left_probes) != set(right_probes):
        raise ValueError("state extraction probe identities differ")
    identities = sorted(left_probes)
    domains: dict[str, list[int]] = defaultdict(list)
    pooled_left = []
    pooled_right = []
    selected_left = []
    selected_right = []
    attention_rows = []
    logit_rows = []
    for sample_index, identity in enumerate(identities):
        left_manifest, left_tensors = _load_aggregate(left_probes[identity])
        right_manifest, right_tensors = _load_aggregate(right_probes[identity])
        if left_manifest["probe_identity_sha256"] != right_manifest["probe_identity_sha256"]:
            raise ValueError("paired probe token identities differ")
        domains[str(left_manifest["domain"])].append(sample_index)
        pooled_left.append(left_tensors["pooled_hidden_states"])
        pooled_right.append(right_tensors["pooled_hidden_states"])
        selected_left.append(left_tensors["selected_hidden_states"].mean(dim=1))
        selected_right.append(right_tensors["selected_hidden_states"].mean(dim=1))
        attention_rows.append(
            {
                "probe_identity_sha256": identity,
                "domain": left_manifest["domain"],
                "entropy_mean_absolute_change": float(
                    torch.mean(torch.abs(right_tensors["attention_entropy"] - left_tensors["attention_entropy"]))
                ),
                "distance_mean_absolute_change": float(
                    torch.mean(torch.abs(right_tensors["attention_distance"] - left_tensors["attention_distance"]))
                ),
            }
        )
        logit_rows.append(
            {
                "probe_identity_sha256": identity,
                "domain": left_manifest["domain"],
                "mean_entropy_change": float(
                    torch.mean(right_tensors["logit_entropy"] - left_tensors["logit_entropy"])
                ),
                "top20_id_overlap": _top_id_overlap(
                    left_tensors["top_logit_ids"], right_tensors["top_logit_ids"]
                ),
            }
        )
    left_all = torch.stack(pooled_left)  # probes, streams, hidden
    right_all = torch.stack(pooled_right)
    selected_left_all = torch.stack(selected_left)
    selected_right_all = torch.stack(selected_right)
    if left_all.shape != right_all.shape:
        raise ValueError("paired pooled hidden-state shapes differ")
    layer_rows = []
    for layer in range(left_all.shape[1]):
        delta = right_all[:, layer] - left_all[:, layer]
        left_norm = torch.linalg.vector_norm(left_all[:, layer], dim=-1)
        relative = torch.linalg.vector_norm(delta, dim=-1) / left_norm.clamp_min(1e-30)
        cosine = torch.nn.functional.cosine_similarity(left_all[:, layer], right_all[:, layer], dim=-1)
        layer_rows.append(
            {
                "stream_index": layer,
                "all_probe_linear_cka": linear_cka(left_all[:, layer], right_all[:, layer]),
                "mean_relative_drift": float(relative.mean()),
                "mean_cosine_similarity": float(cosine.mean()),
                "selected_position_linear_cka": linear_cka(
                    selected_left_all[:, layer], selected_right_all[:, layer]
                ),
                "left_effective_rank": effective_rank(left_all[:, layer]),
                "right_effective_rank": effective_rank(right_all[:, layer]),
                "domains": {
                    domain: {
                        "linear_cka": linear_cka(left_all[indexes, layer], right_all[indexes, layer]),
                        "mean_relative_drift": float(relative[indexes].mean()),
                        "mean_cosine_similarity": float(cosine[indexes].mean()),
                    }
                    for domain, indexes in sorted(domains.items())
                },
            }
        )
    return {
        "representation_version": REPRESENTATION_VERSION,
        "comparison_id": comparison_id,
        "probe_count": len(identities),
        "stream_count": left_all.shape[1],
        "layer_rows": layer_rows,
        "attention_rows": attention_rows,
        "logit_rows": logit_rows,
        "v7_test_accessed": False,
        "causal_experiments_performed": False,
    }


def effective_rank(matrix: torch.Tensor) -> float:
    """Entropy effective rank of centered probe representations."""

    if matrix.ndim != 2:
        raise ValueError("effective rank requires a 2D matrix")
    centered = matrix.double() - matrix.double().mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    total = singular_values.sum()
    if total <= 0:
        return 0.0
    probabilities = singular_values / total
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return float(torch.exp(entropy))


def _probe_catalog(state_dir: Path, *, authoritative: bool) -> dict[str, Any]:
    completion_path = state_dir / "complete.json"
    if not completion_path.is_file():
        raise ValueError(f"state extraction is incomplete: {state_dir}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("v7_test_accessed") is not False:
        raise ValueError("state extraction does not prove test isolation")
    if authoritative and (
        completion.get("probe_count") != 48
        or completion.get("completion_scope") != "authoritative_48_probe_suite"
        or len(completion.get("probe_results", [])) != 48
    ):
        raise ValueError("authoritative representation analysis requires all 48 probes")
    result = {}
    for row in completion.get("probe_results", []):
        path = Path(str(row["path"]))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("state extraction completion contains an unsafe path")
        path = state_dir / path
        manifest_path = path / "manifest.json"
        if _sha256(manifest_path) != row.get("manifest_sha256"):
            raise ValueError("state extraction completion manifest hash mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = str(manifest["probe_identity_sha256"])
        if identity in result:
            raise ValueError("duplicate probe identity in state extraction")
        result[identity] = path
    return {"completion": completion, "probes": result}


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_aggregate(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    from safetensors.torch import load_file

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    for row in manifest.get("files", []):
        relative = Path(str(row.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("probe manifest contains an unsafe path")
        file_path = path / relative
        if (
            not file_path.is_file()
            or file_path.stat().st_size != int(row["bytes"])
            or _sha256(file_path) != row["sha256"]
        ):
            raise ValueError("probe aggregate file hash mismatch")
    tensors = load_file(path / "aggregates.safetensors", device="cpu")
    return manifest, tensors


def _top_id_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("top-logit ID arrays must have matching position x k shape")
    overlaps = []
    for left_row, right_row in zip(left, right, strict=True):
        left_ids = set(int(value) for value in left_row)
        right_ids = set(int(value) for value in right_row)
        overlaps.append(len(left_ids & right_ids) / len(left_ids | right_ids))
    return sum(overlaps) / len(overlaps)
