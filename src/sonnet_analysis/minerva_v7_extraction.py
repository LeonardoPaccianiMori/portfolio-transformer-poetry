"""Resumable hidden-state, attention, and logit extraction for frozen V7 probes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


EXTRACTION_VERSION = "minerva_7b_v7_probe_extraction_v1"


def resolve_module(root: torch.nn.Module, dotted_name: str) -> torch.nn.Module:
    """Resolve a frozen dotted module name including numeric ModuleList indices."""

    current: Any = root
    for part in dotted_name.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    if not isinstance(current, torch.nn.Module):
        raise ValueError(f"resolved object is not a module: {dotted_name}")
    return current


def capture_probe(
    *,
    model: torch.nn.Module,
    probe: Mapping[str, Any],
    device: torch.device | str,
    block_count: int,
    raw_attention_layers: Sequence[int],
    raw_attention_maximum_tokens: int,
    retain_raw_attention: bool,
) -> dict[str, Any]:
    """Capture one exact probe in eval/inference mode using frozen module names."""

    resolved_device = torch.device(device)
    streams: dict[str, torch.Tensor] = {}
    handles = []

    def hook(name: str) -> Callable[..., None]:
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"module did not return a tensor: {name}")
            streams[name] = tensor.detach()

        return capture

    names = ["model.embed_tokens"] + [f"model.layers.{index}" for index in range(block_count)] + ["model.norm"]
    for name in names:
        handles.append(resolve_module(model, name).register_forward_hook(hook(name)))
    input_ids = torch.tensor([probe["input_ids"]], dtype=torch.long, device=resolved_device)
    attention_mask = torch.tensor([probe["attention_mask"]], dtype=torch.long, device=resolved_device)
    model.eval()
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    missing = [name for name in names if name not in streams]
    if missing:
        raise ValueError(f"hidden-state hooks did not fire: {missing[:3]}")
    ordered = [streams[name][0] for name in names]
    sequence_length = len(probe["input_ids"])
    if any(tuple(tensor.shape[:1]) != (sequence_length,) for tensor in ordered):
        raise ValueError("captured hidden-state sequence length changed")
    raw_hidden = torch.stack([tensor.to(device="cpu", dtype=torch.bfloat16) for tensor in ordered])
    mask = attention_mask[0].bool()
    selected = torch.tensor(probe["selected_positions"], dtype=torch.long, device=resolved_device)
    pooled = torch.stack(
        [tensor[mask].float().mean(dim=0).cpu() for tensor in ordered]
    )
    selected_hidden = torch.stack(
        [tensor.index_select(0, selected).float().cpu() for tensor in ordered]
    )
    logits = outputs.logits[0].index_select(0, selected).float()
    logsumexp = torch.logsumexp(logits, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)
    entropy = logsumexp - torch.sum(probabilities * logits, dim=-1)
    top_values, top_ids = torch.topk(logits, k=min(20, logits.shape[-1]), dim=-1)
    attentions = getattr(outputs, "attentions", None)
    if not isinstance(attentions, (tuple, list)) or len(attentions) != block_count:
        raise ValueError("model did not return attention for every transformer block")
    attention_entropy = []
    attention_distance = []
    raw_attention = []
    for layer_index, attention in enumerate(attentions):
        layer = attention[0].float()
        entropy_row, distance_row = summarize_attention(layer, mask)
        attention_entropy.append(entropy_row.cpu())
        attention_distance.append(distance_row.cpu())
        if retain_raw_attention and layer_index in raw_attention_layers:
            bounded = attention[
                0,
                :,
                :raw_attention_maximum_tokens,
                :raw_attention_maximum_tokens,
            ]
            raw_attention.append(bounded.to(device="cpu", dtype=torch.bfloat16))
    return {
        "raw_hidden_states": raw_hidden,
        "pooled_hidden_states": pooled,
        "selected_hidden_states": selected_hidden,
        "selected_positions": selected.cpu(),
        "top_logit_ids": top_ids.to(torch.int32).cpu(),
        "top_logit_values": top_values.cpu(),
        "logsumexp": logsumexp.cpu(),
        "logit_entropy": entropy.cpu(),
        "attention_entropy": torch.stack(attention_entropy),
        "attention_distance": torch.stack(attention_distance),
        "raw_attention": torch.stack(raw_attention) if raw_attention else None,
    }


def summarize_attention(attention: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-head entropy and mean absolute attended-token distance."""

    if attention.ndim != 3 or attention.shape[-1] != attention.shape[-2]:
        raise ValueError("attention must have shape (heads, query, key)")
    length = attention.shape[-1]
    mask = valid_mask[:length].bool()
    weights = attention[:, mask, :][:, :, mask]
    normalizer = weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    probabilities = weights / normalizer
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1).mean(dim=-1)
    positions = torch.arange(length, device=attention.device)[mask].float()
    distance = torch.abs(positions[:, None] - positions[None, :])
    mean_distance = (probabilities * distance).sum(dim=-1).mean(dim=-1)
    return entropy, mean_distance


def write_probe_result(
    *, destination: Path, probe: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    """Atomically save one resumable probe and its complete hash manifest."""

    from safetensors.torch import save_file

    temporary = destination.with_name(destination.name + ".tmp")
    if destination.exists():
        return verify_probe_result(destination, probe)
    if temporary.exists():
        import shutil

        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    raw_path = temporary / "raw_hidden_states.safetensors"
    aggregate_path = temporary / "aggregates.safetensors"
    save_file({"hidden_states": result["raw_hidden_states"].contiguous()}, raw_path)
    tensors = {
        key: value.contiguous()
        for key, value in result.items()
        if isinstance(value, torch.Tensor) and key not in {"raw_hidden_states", "raw_attention"}
    }
    save_file(tensors, aggregate_path)
    files = [_file_row(raw_path, temporary), _file_row(aggregate_path, temporary)]
    raw_attention = result.get("raw_attention")
    if isinstance(raw_attention, torch.Tensor):
        attention_path = temporary / "raw_attention.safetensors"
        save_file({"attention": raw_attention.contiguous()}, attention_path)
        files.append(_file_row(attention_path, temporary))
    probe_identity = _probe_identity_sha256(probe)
    manifest = {
        "extraction_version": EXTRACTION_VERSION,
        "probe_id": probe["probe_id"],
        "source_identity": probe["source_identity"],
        "probe_identity_sha256": probe_identity,
        "input_ids_sha256": probe["input_ids_sha256"],
        "domain": probe["domain"],
        "sequence_tokens": len(probe["input_ids"]),
        "selected_positions": list(probe["selected_positions"]),
        "files": files,
        "v7_test_accessed": False,
    }
    manifest_path = temporary / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _fsync_tree(temporary)
    temporary.rename(destination)
    return verify_probe_result(destination, probe)


def verify_probe_result(destination: Path, probe: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"probe result is incomplete: {destination}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("extraction_version") != EXTRACTION_VERSION:
        raise ValueError("probe extraction version mismatch")
    if manifest.get("probe_identity_sha256") != _probe_identity_sha256(probe):
        raise ValueError("probe result identity mismatch")
    if manifest.get("input_ids_sha256") != probe["input_ids_sha256"]:
        raise ValueError("probe result token identity mismatch")
    for row in manifest.get("files", []):
        path = destination / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
            raise ValueError(f"probe result file mismatch: {path}")
    return {**manifest, "manifest_sha256": _sha256(manifest_path)}


def extract_state(
    *,
    model: torch.nn.Module,
    probes: Sequence[Mapping[str, Any]],
    destination: Path,
    state_metadata: Mapping[str, Any],
    device: torch.device | str,
    block_count: int,
    raw_attention_layers: Sequence[int],
    raw_attention_maximum_tokens: int,
    raw_attention_probe_hashes: set[str],
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Extract or resume one state; install complete.json only after all probes verify."""

    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.monotonic()
    for index, probe in enumerate(probes, start=1):
        probe_key = hashlib.sha256(
            f"{probe['probe_id']}|{probe['input_ids_sha256']}".encode("utf-8")
        ).hexdigest()[:16]
        probe_dir = destination / "probes" / probe_key
        if probe_dir.exists():
            manifest = verify_probe_result(probe_dir, probe)
            status = "reused"
        else:
            result = capture_probe(
                model=model,
                probe=probe,
                device=device,
                block_count=block_count,
                raw_attention_layers=raw_attention_layers,
                raw_attention_maximum_tokens=raw_attention_maximum_tokens,
                retain_raw_attention=probe["input_ids_sha256"] in raw_attention_probe_hashes,
            )
            probe_dir.parent.mkdir(parents=True, exist_ok=True)
            manifest = write_probe_result(destination=probe_dir, probe=probe, result=result)
            status = "written"
            del result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append(
            {
                "probe_id": probe["probe_id"],
                "path": str(probe_dir.relative_to(destination)),
                "manifest_sha256": manifest["manifest_sha256"],
            }
        )
        if progress:
            elapsed = time.monotonic() - started
            eta = elapsed / index * (len(probes) - index)
            progress(
                f"probe={index}/{len(probes)} progress={100 * index / len(probes):.1f}% "
                f"status={status} elapsed={elapsed:.1f}s eta={eta:.1f}s"
            )
    completion = {
        "extraction_version": EXTRACTION_VERSION,
        "state": dict(state_metadata),
        "probe_count": len(rows),
        "probe_results": rows,
        "completion_scope": "authoritative_48_probe_suite" if len(rows) == 48 else "bounded_non_authoritative_run",
        "elapsed_seconds": time.monotonic() - started,
        "v7_test_accessed": False,
        "causal_experiments_performed": False,
    }
    completion_path = destination / "complete.json"
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return completion


def select_raw_attention_probe_hashes(probes: Sequence[Mapping[str, Any]]) -> set[str]:
    selected = set()
    for domain in sorted({str(row["domain"]) for row in probes}):
        row = next(item for item in probes if item["domain"] == domain)
        selected.add(str(row["input_ids_sha256"]))
    return selected


def _file_row(path: Path, root: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _probe_identity_sha256(probe: Mapping[str, Any]) -> str:
    payload = {
        "probe_id": str(probe["probe_id"]),
        "source_identity": str(probe["source_identity"]),
        "input_ids_sha256": str(probe["input_ids_sha256"]),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
