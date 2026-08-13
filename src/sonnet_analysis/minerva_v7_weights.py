"""Memory-bounded SafeTensors comparisons for Minerva V7 model states."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator


WEIGHT_ANALYSIS_VERSION = "minerva_7b_v7_weight_change_v1"
DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024
MAX_PROJECTED_WORKING_BYTES = 512 * 1024 * 1024


class TensorCatalog:
    """Map each tensor name to exactly one SafeTensors shard."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.locations: dict[str, Path] = {}
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            for name, relative in index.get("weight_map", {}).items():
                self.locations[str(name)] = model_dir / str(relative)
        else:
            from safetensors import safe_open

            for shard in sorted(model_dir.glob("*.safetensors")):
                with safe_open(shard, framework="pt", device="cpu") as handle:
                    for name in handle.keys():
                        if name in self.locations:
                            raise ValueError(f"tensor occurs in multiple shards: {name}")
                        self.locations[name] = shard
        if not self.locations:
            raise ValueError(f"no SafeTensors weights found: {model_dir}")
        missing = sorted({path for path in self.locations.values() if not path.is_file()})
        if missing:
            raise FileNotFoundError(f"weight shard is absent: {missing[0]}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.locations))


def plan_weight_comparison(
    *, left_model_dir: Path, right_model_dir: Path, chunk_bytes: int = DEFAULT_CHUNK_BYTES
) -> dict[str, Any]:
    """Validate tensor structure and estimate bounded resident memory and I/O."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    left = TensorCatalog(left_model_dir)
    right = TensorCatalog(right_model_dir)
    if left.names != right.names:
        missing_left = sorted(set(right.names) - set(left.names))
        missing_right = sorted(set(left.names) - set(right.names))
        raise ValueError(
            f"tensor-name mismatch: missing_left={missing_left[:3]} missing_right={missing_right[:3]}"
        )
    tensors = []
    total_elements = 0
    largest_tensor_elements = 0
    largest_chunk_elements = 0
    for name in left.names:
        left_meta = _tensor_metadata(left.locations[name], name)
        right_meta = _tensor_metadata(right.locations[name], name)
        if left_meta["shape"] != right_meta["shape"]:
            raise ValueError(f"tensor-shape mismatch: {name}")
        if left_meta["dtype"] != right_meta["dtype"]:
            raise ValueError(f"tensor-dtype mismatch: {name}")
        elements = math.prod(left_meta["shape"]) if left_meta["shape"] else 1
        total_elements += elements
        largest_tensor_elements = max(largest_tensor_elements, elements)
        trailing = math.prod(left_meta["shape"][1:]) if len(left_meta["shape"]) > 1 else 1
        requested_elements = max(1, chunk_bytes // 4)
        rows_per_chunk = max(1, requested_elements // trailing)
        largest_chunk_elements = max(
            largest_chunk_elements, min(elements, rows_per_chunk * trailing)
        )
        tensors.append({"name": name, **left_meta, "elements": elements})
    # Two input chunks plus FP32 left/right/delta/product workspaces and reduction slack.
    projected_working = largest_chunk_elements * 4 * 6
    if projected_working > MAX_PROJECTED_WORKING_BYTES:
        raise ValueError("projected tensor working set exceeds the frozen 512 MiB limit")
    unique_shards = set(left.locations.values()) | set(right.locations.values())
    return {
        "analysis_version": WEIGHT_ANALYSIS_VERSION,
        "left_model_dir": str(left_model_dir),
        "right_model_dir": str(right_model_dir),
        "tensor_count": len(tensors),
        "total_elements": total_elements,
        "largest_tensor_elements": largest_tensor_elements,
        "largest_chunk_elements": largest_chunk_elements,
        "chunk_bytes": chunk_bytes,
        "maximum_projected_working_bytes": projected_working,
        "input_bytes_to_scan": sum(path.stat().st_size for path in unique_shards),
        "full_delta_tensor_materialized": False,
        "maximum_simultaneous_input_chunks": 2,
        "tensors": tensors,
    }


def compare_model_weights(
    *,
    left_model_dir: Path,
    right_model_dir: Path,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    progress: callable | None = None,
) -> dict[str, Any]:
    """Compute exact norm/cosine summaries with chunked FP32 arithmetic."""

    import torch
    from safetensors import safe_open

    plan = plan_weight_comparison(
        left_model_dir=left_model_dir, right_model_dir=right_model_dir,
        chunk_bytes=chunk_bytes,
    )
    left_catalog = TensorCatalog(left_model_dir)
    right_catalog = TensorCatalog(right_model_dir)
    rows = []
    started = time.monotonic()
    for index, item in enumerate(plan["tensors"], start=1):
        name = item["name"]
        shape = tuple(item["shape"])
        elements_per_slice = max(1, chunk_bytes // 4)
        sums = {"left_sq": 0.0, "right_sq": 0.0, "delta_sq": 0.0, "dot": 0.0, "max_abs_delta": 0.0}
        with safe_open(left_catalog.locations[name], framework="pt", device="cpu") as left_handle, safe_open(
            right_catalog.locations[name], framework="pt", device="cpu"
        ) as right_handle:
            left_slice = left_handle.get_slice(name)
            right_slice = right_handle.get_slice(name)
            for selector in _selectors(shape, elements_per_slice):
                left_tensor = left_slice[selector].to(torch.float32)
                right_tensor = right_slice[selector].to(torch.float32)
                delta = right_tensor - left_tensor
                sums["left_sq"] += torch.sum(left_tensor * left_tensor, dtype=torch.float64).item()
                sums["right_sq"] += torch.sum(right_tensor * right_tensor, dtype=torch.float64).item()
                sums["delta_sq"] += torch.sum(delta * delta, dtype=torch.float64).item()
                sums["dot"] += torch.sum(left_tensor * right_tensor, dtype=torch.float64).item()
                sums["max_abs_delta"] = max(sums["max_abs_delta"], torch.max(torch.abs(delta)).item())
        left_norm = math.sqrt(sums["left_sq"])
        right_norm = math.sqrt(sums["right_sq"])
        delta_norm = math.sqrt(sums["delta_sq"])
        rows.append(
            {
                "name": name,
                "group": parameter_group(name),
                "elements": item["elements"],
                "left_l2": left_norm,
                "right_l2": right_norm,
                "delta_l2": delta_norm,
                "relative_delta_l2": _ratio(delta_norm, left_norm),
                "cosine_similarity": _ratio(sums["dot"], left_norm * right_norm),
                "max_abs_delta": sums["max_abs_delta"],
            }
        )
        if progress is not None:
            progress(index, len(plan["tensors"]), name, time.monotonic() - started)
    groups = _aggregate_groups(rows)
    total_delta_sq = sum(row["delta_l2"] ** 2 for row in rows)
    for row in rows:
        row["fraction_of_total_delta_energy"] = _ratio(row["delta_l2"] ** 2, total_delta_sq)
    return {
        **{key: value for key, value in plan.items() if key != "tensors"},
        "elapsed_seconds": time.monotonic() - started,
        "tensors": rows,
        "groups": groups,
    }


def parameter_group(name: str) -> str:
    if name == "model.embed_tokens.weight":
        return "embedding"
    if name.startswith("lm_head."):
        return "lm_head"
    if name.startswith("model.norm."):
        return "final_norm"
    if name.startswith("model.layers."):
        parts = name.split(".")
        layer = int(parts[2])
        band = f"blocks_{(layer // 8) * 8:02d}_{(layer // 8) * 8 + 7:02d}"
        if ".self_attn." in name:
            module = "attention"
        elif ".mlp." in name:
            module = "mlp"
        elif "layernorm" in name or "layer_norm" in name:
            module = "norm"
        else:
            module = "other"
        return f"{band}/{module}"
    return "other"


def _tensor_metadata(path: Path, name: str) -> dict[str, Any]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(name)
        return {"shape": list(tensor_slice.get_shape()), "dtype": str(tensor_slice.get_dtype())}


def _selectors(shape: tuple[int, ...], elements_per_slice: int) -> Iterator[Any]:
    if not shape:
        yield ()
        return
    trailing = math.prod(shape[1:]) if len(shape) > 1 else 1
    rows = max(1, elements_per_slice // trailing)
    for start in range(0, shape[0], rows):
        selection = [slice(start, min(shape[0], start + rows))]
        selection.extend(slice(None) for _ in shape[1:])
        yield tuple(selection)


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _aggregate_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accum: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        group = accum[row["group"]]
        group["elements"] += row["elements"]
        group["left_sq"] += row["left_l2"] ** 2
        group["right_sq"] += row["right_l2"] ** 2
        group["delta_sq"] += row["delta_l2"] ** 2
    result = []
    total_delta_sq = sum(value["delta_sq"] for value in accum.values())
    for name, value in sorted(accum.items()):
        left_norm = math.sqrt(value["left_sq"])
        delta_norm = math.sqrt(value["delta_sq"])
        result.append(
            {
                "group": name,
                "elements": int(value["elements"]),
                "left_l2": left_norm,
                "right_l2": math.sqrt(value["right_sq"]),
                "delta_l2": delta_norm,
                "relative_delta_l2": _ratio(delta_norm, left_norm),
                "fraction_of_total_delta_energy": _ratio(value["delta_sq"], total_delta_sq),
            }
        )
    return result
