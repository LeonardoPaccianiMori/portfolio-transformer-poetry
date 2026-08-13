"""Frequency-controlled, streamed embedding and LM-head analysis for V7 states."""

from __future__ import annotations

import array
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from sonnet_analysis.minerva_v7_weights import TensorCatalog
from sonnet_training.minerva_7b_full_weight_data import tokenizer_sha256


EMBEDDING_VERSION = "minerva_7b_v7_embedding_analysis_v1"
EXPECTED_TOKENIZER_SHA256 = "11fbe803977e9d6dc1a50e6bb088be5b550f5e26da2a82fbfd7b41a045853a8c"


def verify_embedding_tokenizer(tokenizer: Any) -> None:
    if tokenizer_sha256(tokenizer) != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("embedding analysis tokenizer fingerprint mismatch")


def resolve_verified_training_shards(
    encoded_report_path: Path, *, repo_root: Path
) -> list[Path]:
    """Resolve and hash every frozen training shard without permitting test data."""

    report = json.loads(encoded_report_path.read_text(encoding="utf-8"))
    shards = []
    for pool in report.get("pools", []):
        if pool.get("split") != "train":
            continue
        if "test" in str(pool.get("pool_id", "")).lower():
            raise ValueError("embedding frequency source contains a test pool")
        for row in pool.get("shards", []):
            relative = Path(str(row["path"]))
            if relative.is_absolute() or ".." in relative.parts or "test" in str(relative).lower():
                raise ValueError("embedding frequency source contains an unsafe or test path")
            path = repo_root / relative
            if not path.is_file() or path.stat().st_size != int(row["bytes"]) or _sha256(path) != row["sha256"]:
                raise ValueError(f"embedding frequency shard integrity mismatch: {relative}")
            shards.append(path)
    if not shards:
        raise ValueError("encoded report contains no verified training shards")
    return shards


def resolve_token_registry(tokenizer: Any, registry: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze only terms represented by one non-special token, recording rejects."""

    accepted = []
    rejected = []
    seen = set()
    for group, terms in registry["groups"].items():
        for term in terms:
            variants = [str(term), f" {term}"] if registry["policy"]["leading_space_variant"] else [str(term)]
            for variant in variants:
                ids = tokenizer.encode(variant, add_special_tokens=False)
                row = {"group": group, "term": term, "variant": variant, "token_ids": [int(value) for value in ids]}
                if len(ids) == 1 and int(ids[0]) not in set(getattr(tokenizer, "all_special_ids", [])):
                    identity = (group, int(ids[0]))
                    if identity not in seen:
                        accepted.append({**row, "token_id": int(ids[0])})
                        seen.add(identity)
                else:
                    rejected.append(row)
    if not accepted:
        raise ValueError("embedding registry contains no single-token terms")
    return {
        "registry_version": registry["registry_version"],
        "accepted": accepted,
        "rejected_fragmented_terms": rejected,
        "v7_test_accessed": False,
    }


def count_selected_token_ids(
    shard_paths: Sequence[Path], token_ids: Sequence[int], *, chunk_bytes: int = 16 * 1024 * 1024
) -> dict[int, int]:
    """Count selected int32 IDs without retaining corpus tokens."""

    if chunk_bytes <= 0 or chunk_bytes % 4:
        raise ValueError("chunk_bytes must be a positive multiple of four")
    selected = set(int(value) for value in token_ids)
    counts = {value: 0 for value in selected}
    typecode = "i"
    if array.array(typecode).itemsize != 4:
        raise RuntimeError("native signed integer is not 32-bit")
    for path in shard_paths:
        with path.open("rb") as handle:
            while block := handle.read(chunk_bytes):
                values = array.array(typecode)
                values.frombytes(block)
                if sys.byteorder != "little":
                    values.byteswap()
                for value in values:
                    if value in selected:
                        counts[value] += 1
    return counts


def analyze_embedding_pair(
    *,
    left_model_dir: Path,
    right_model_dir: Path,
    resolved_registry: Mapping[str, Any],
    frequencies: Mapping[int, int],
    top_k: int = 20,
    vocabulary_chunk_rows: int = 4096,
) -> dict[str, Any]:
    """Compare selected rows and exact streamed nearest neighborhoods."""

    if top_k <= 0 or vocabulary_chunk_rows <= 0:
        raise ValueError("top_k and vocabulary_chunk_rows must be positive")
    token_ids = sorted({int(row["token_id"]) for row in resolved_registry["accepted"]})
    tensor_names = ("model.embed_tokens.weight", "lm_head.weight")
    tensor_reports = {}
    for tensor_name in tensor_names:
        left_rows, left_neighbors = _selected_rows_and_neighbors(
            left_model_dir, tensor_name, token_ids, top_k, vocabulary_chunk_rows
        )
        right_rows, right_neighbors = _selected_rows_and_neighbors(
            right_model_dir, tensor_name, token_ids, top_k, vocabulary_chunk_rows
        )
        rows = []
        for index, token_id in enumerate(token_ids):
            delta = right_rows[index] - left_rows[index]
            left_set = set(left_neighbors[token_id])
            right_set = set(right_neighbors[token_id])
            rows.append(
                {
                    "token_id": token_id,
                    "frequency": int(frequencies.get(token_id, 0)),
                    "delta_l2": float(torch.linalg.vector_norm(delta)),
                    "relative_delta_l2": _ratio(
                        float(torch.linalg.vector_norm(delta)),
                        float(torch.linalg.vector_norm(left_rows[index])),
                    ),
                    "cosine_similarity": float(
                        torch.nn.functional.cosine_similarity(
                            left_rows[index].unsqueeze(0), right_rows[index].unsqueeze(0)
                        )[0]
                    ),
                    "left_neighbors": left_neighbors[token_id],
                    "right_neighbors": right_neighbors[token_id],
                    "neighbor_jaccard": len(left_set & right_set) / len(left_set | right_set),
                }
            )
        tensor_reports[tensor_name] = rows
    return {
        "embedding_version": EMBEDDING_VERSION,
        "accepted_registry": list(resolved_registry["accepted"]),
        "rejected_fragmented_terms": list(resolved_registry["rejected_fragmented_terms"]),
        "tensors": tensor_reports,
        "frequency_control_recorded": True,
        "v7_test_accessed": False,
    }


def _selected_rows_and_neighbors(
    model_dir: Path, tensor_name: str, token_ids: list[int], top_k: int, chunk_rows: int
) -> tuple[torch.Tensor, dict[int, list[int]]]:
    from safetensors import safe_open

    catalog = TensorCatalog(model_dir)
    if tensor_name not in catalog.locations:
        raise ValueError(f"state lacks tensor: {tensor_name}")
    with safe_open(catalog.locations[tensor_name], framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(tensor_name)
        shape = tensor_slice.get_shape()
        if len(shape) != 2 or max(token_ids) >= shape[0]:
            raise ValueError(f"invalid vocabulary tensor: {tensor_name}")
        queries = torch.stack([tensor_slice[token_id : token_id + 1][0].float() for token_id in token_ids])
        normalized_queries = torch.nn.functional.normalize(queries, dim=-1)
        best_values = torch.full((len(token_ids), top_k + 1), -torch.inf)
        best_ids = torch.full((len(token_ids), top_k + 1), -1, dtype=torch.long)
        for start in range(0, shape[0], chunk_rows):
            end = min(shape[0], start + chunk_rows)
            chunk = tensor_slice[start:end].float()
            similarities = normalized_queries @ torch.nn.functional.normalize(chunk, dim=-1).T
            ids = torch.arange(start, end).expand(len(token_ids), -1)
            values = torch.cat([best_values, similarities], dim=1)
            identifiers = torch.cat([best_ids, ids], dim=1)
            best_values, indexes = torch.topk(values, k=top_k + 1, dim=1)
            best_ids = torch.gather(identifiers, 1, indexes)
    neighbors = {}
    for index, token_id in enumerate(token_ids):
        neighbors[token_id] = [int(value) for value in best_ids[index] if int(value) != token_id][:top_k]
    return queries, neighbors


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
