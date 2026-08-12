"""CPU-verifiable execution machinery for the Minerva 7B V7 curriculum."""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import random
import shutil
import struct
import tarfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXECUTION_VERSION = "minerva_7b_v7_execution_v1"
CHECKPOINT_VERSION = "minerva_7b_v7_atomic_checkpoint_v1"
PROBE_VERSION = "minerva_7b_v7_activation_probes_v1"
UINT32 = struct.Struct("<I")


@dataclass(frozen=True)
class V7ExecutionConfig:
    """Resolved committed policy and machine-local training artifacts."""

    repo_root: Path
    execution_path: Path
    encoded_dir: Path
    window_index_dir: Path
    modern_encoded_dir: Path
    modern_index_path: Path


@dataclass(frozen=True)
class WindowBatch:
    """One complete optimizer update in global frozen order."""

    stage_id: str
    update: int
    first_window_index: int
    next_window_index: int
    input_ids: tuple[tuple[int, ...], ...]
    target_ids: tuple[tuple[int, ...], ...]
    identity_sha256: str


class Int32ShardStore:
    """Memory-map verified signed-int32 token shards and read exact slices."""

    def __init__(
        self,
        *,
        encoded_dir: Path,
        encoded_report: Mapping[str, Any],
        required_pools: Iterable[str] | None = None,
        verify_hashes: bool = True,
        path_overrides: Mapping[str, Path] | None = None,
    ) -> None:
        required = set(required_pools or ())
        self._pools: dict[str, dict[int, tuple[Any, mmap.mmap, int]]] = {}
        self._files: list[Any] = []
        pool_rows = encoded_report.get("pools")
        if not isinstance(pool_rows, list):
            raise ValueError("encoded report is missing pools")
        for pool in pool_rows:
            pool_id = str(pool["pool_id"])
            if required and pool_id not in required:
                continue
            shards: dict[int, tuple[Any, mmap.mmap, int]] = {}
            for shard in pool["shards"]:
                override = (path_overrides or {}).get(pool_id)
                path = (
                    override
                    if override is not None
                    else encoded_dir / Path(str(shard["path"])).name
                )
                expected_bytes = int(shard["bytes"])
                if not path.is_file() or path.stat().st_size != expected_bytes:
                    raise ValueError(f"encoded shard size mismatch: {path}")
                if expected_bytes != int(shard["token_count"]) * 4:
                    raise ValueError("encoded shard is not packed signed int32")
                if verify_hashes and _sha256(path) != shard["sha256"]:
                    raise ValueError(f"encoded shard hash mismatch: {path}")
                handle = path.open("rb")
                mapping = mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ)
                self._files.append(handle)
                shards[int(shard["shard_index"])] = (
                    handle,
                    mapping,
                    int(shard["token_count"]),
                )
            self._pools[pool_id] = shards
        if required - self._pools.keys():
            raise ValueError(
                "encoded report lacks required pools: "
                + ", ".join(sorted(required - self._pools.keys()))
            )

    def read_slices(
        self, pool_id: str, slices: Sequence[Mapping[str, Any]]
    ) -> tuple[int, ...]:
        """Read and concatenate the exact physical slices in one window row."""

        if pool_id not in self._pools:
            raise KeyError(f"unknown encoded pool: {pool_id}")
        values: list[int] = []
        for piece in slices:
            shard_index = int(piece["shard_index"])
            offset = int(piece["token_offset"])
            count = int(piece["token_count"])
            try:
                _, mapping, token_count = self._pools[pool_id][shard_index]
            except KeyError as error:
                raise ValueError("window references an unknown encoded shard") from error
            if offset < 0 or count <= 0 or offset + count > token_count:
                raise ValueError("window slice is outside its encoded shard")
            raw = mapping[offset * 4 : (offset + count) * 4]
            values.extend(value[0] for value in struct.iter_unpack("<i", raw))
        return tuple(values)

    def close(self) -> None:
        for shards in self._pools.values():
            for _, mapping, _ in shards.values():
                mapping.close()
        for handle in self._files:
            handle.close()
        self._pools.clear()
        self._files.clear()

    def __enter__(self) -> Int32ShardStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class FrozenWindowReader:
    """Reconstruct the exact 2,049-token sources used by checkpoint 8D."""

    def __init__(
        self,
        *,
        index_root: Path,
        encoded_store: Int32ShardStore,
        window_manifest: Mapping[str, Any],
        verify_index_hashes: bool = True,
    ) -> None:
        self.index_root = index_root
        self.encoded_store = encoded_store
        self._entries: dict[str, Mapping[str, Any]] = {}
        files = window_manifest.get("files")
        if not isinstance(files, list):
            raise ValueError("window manifest is missing files")
        for row in files:
            path = index_root / str(row["path"])
            if not path.is_file() or path.stat().st_size != int(row["bytes"]):
                raise ValueError(f"window index size mismatch: {path}")
            if verify_index_hashes and _sha256(path) != row["sha256"]:
                raise ValueError(f"window index hash mismatch: {path}")
            key = path.stem
            self._entries[key] = row

    def rows(self, index_id: str) -> tuple[dict[str, Any], ...]:
        if index_id not in self._entries:
            raise KeyError(f"unknown window index: {index_id}")
        path = self.index_root / str(self._entries[index_id]["path"])
        default_pool_id = self._entries[index_id].get("pool_id")
        loaded = []
        for row in _read_jsonl(path):
            if "pool_id" not in row and default_pool_id is not None:
                row["pool_id"] = str(default_pool_id)
            loaded.append(row)
        rows = tuple(loaded)
        if len(rows) != int(self._entries[index_id]["rows"]):
            raise ValueError("window index row count changed")
        return rows

    def source_tokens(self, row: Mapping[str, Any]) -> tuple[int, ...]:
        tokens = self.encoded_store.read_slices(
            str(row["pool_id"]), _list(row, "source_slices")
        )
        if len(tokens) != int(row["source_span_tokens"]):
            raise ValueError("window slices do not reconstruct the source span")
        if len(tokens) != int(row["target_tokens"]) + 1:
            raise ValueError("causal source must contain one more token than targets")
        return tokens

    def optimizer_batch(
        self,
        *,
        stage_id: str,
        update: int,
        global_windows_per_update: int,
    ) -> WindowBatch:
        if update <= 0 or global_windows_per_update <= 0:
            raise ValueError("update and global window count must be positive")
        rows = self.rows(stage_id)
        start = (update - 1) * global_windows_per_update
        end = start + global_windows_per_update
        if end > len(rows):
            raise ValueError("optimizer update exceeds the stage window index")
        sources = tuple(self.source_tokens(row) for row in rows[start:end])
        identity = _canonical_sha256(
            [
                {
                    "stage_id": stage_id,
                    "stage_window_index": row.get("stage_window_index"),
                    "pool_id": row["pool_id"],
                    "source_slices": row["source_slices"],
                }
                for row in rows[start:end]
            ]
        )
        return WindowBatch(
            stage_id=stage_id,
            update=update,
            first_window_index=start,
            next_window_index=end,
            input_ids=tuple(tokens[:-1] for tokens in sources),
            target_ids=tuple(tokens[1:] for tokens in sources),
            identity_sha256=identity,
        )

    @staticmethod
    def rank_microbatches(
        batch: WindowBatch,
        *,
        rank: int,
        world_size: int,
        local_microbatch_size: int,
    ) -> tuple[tuple[tuple[int, ...], ...], ...]:
        """Assign global windows by rank striding, then form local microbatches."""

        if rank < 0 or rank >= world_size or local_microbatch_size <= 0:
            raise ValueError("invalid DDP rank or local microbatch")
        local = batch.input_ids[rank::world_size]
        if len(local) % local_microbatch_size:
            raise ValueError("local window count does not divide into microbatches")
        return tuple(
            tuple(local[start : start + local_microbatch_size])
            for start in range(0, len(local), local_microbatch_size)
        )


def load_execution_config(path: Path, repo_root: Path) -> dict[str, Any]:
    """Load 8F and verify all committed lineage before local work."""

    execution = _read_json(path)
    if execution.get("execution_version") != EXECUTION_VERSION:
        raise ValueError("unexpected V7 execution version")
    lineage = _mapping(execution, "lineage")
    for path_key, hash_key in (
        ("protocol_path", "protocol_sha256"),
        ("encoded_report_path", "encoded_report_sha256"),
        ("window_report_path", "window_report_sha256"),
    ):
        artifact = _resolve(repo_root, str(lineage[path_key]))
        if _sha256(artifact) != lineage[hash_key]:
            raise ValueError(f"V7 execution lineage mismatch: {path_key}")
    protocol = _read_json(_resolve(repo_root, str(lineage["protocol_path"])))
    encoded = _read_json(_resolve(repo_root, str(lineage["encoded_report_path"])))
    windows = _read_json(_resolve(repo_root, str(lineage["window_report_path"])))
    if protocol["lineage"]["tokenizer_sha256"] != lineage["tokenizer_sha256"]:
        raise ValueError("V7 execution tokenizer lineage mismatch")
    if encoded["encoded_content_identity_sha256"] != protocol["lineage"][
        "encoded_content_identity_sha256"
    ]:
        raise ValueError("V7 encoded identity changed")
    if windows["reproduction"]["primary_content_identity_sha256"] != lineage[
        "window_content_identity_sha256"
    ]:
        raise ValueError("V7 window identity changed")
    _validate_execution_contract(execution, protocol)
    return execution


def build_execution_context(config: V7ExecutionConfig) -> dict[str, Any]:
    """Load committed reports and the private manifest used by the reader."""

    execution = load_execution_config(config.execution_path, config.repo_root)
    lineage = execution["lineage"]
    encoded_report = _read_json(
        _resolve(config.repo_root, str(lineage["encoded_report_path"]))
    )
    window_report = _read_json(
        _resolve(config.repo_root, str(lineage["window_report_path"]))
    )
    window_manifest = _read_json(config.window_index_dir / "manifest.json")
    if window_manifest.get("window_index_content_identity_sha256") != str(
        window_report["window_index_content_identity_sha256"]
    ):
        raise ValueError("local window manifest differs from the frozen public report")
    return {
        "execution": execution,
        "protocol": _read_json(
            _resolve(config.repo_root, str(lineage["protocol_path"]))
        ),
        "encoded_report": encoded_report,
        "window_report": window_report,
        "window_manifest": window_manifest,
    }


def make_update_telemetry(
    *,
    stage_id: str,
    stage_update: int,
    global_update: int,
    batch: WindowBatch,
    loss: float,
    gradient_norm: float,
    learning_rate: float,
    tokens_per_second: float,
    elapsed_seconds: float,
    eta_seconds: float,
    rank_memory: Sequence[Mapping[str, float]],
    cumulative_cost_usd: float | None,
) -> dict[str, Any]:
    """Validate one compact permanent row written at every optimizer update."""

    finite_values = (
        loss,
        gradient_norm,
        learning_rate,
        tokens_per_second,
        elapsed_seconds,
        eta_seconds,
    )
    if any(not math.isfinite(float(value)) for value in finite_values):
        raise ValueError("update telemetry contains non-finite values")
    if cumulative_cost_usd is not None and (
        cumulative_cost_usd < 0 or not math.isfinite(cumulative_cost_usd)
    ):
        raise ValueError("cumulative cost must be finite and non-negative")
    return {
        "stage_id": stage_id,
        "stage_update": stage_update,
        "global_update": global_update,
        "first_window_index": batch.first_window_index,
        "next_window_index": batch.next_window_index,
        "window_identity_sha256": batch.identity_sha256,
        "mean_training_loss": float(loss),
        "preclip_global_gradient_norm": float(gradient_norm),
        "learning_rate": float(learning_rate),
        "tokens_per_second": float(tokens_per_second),
        "elapsed_seconds": float(elapsed_seconds),
        "eta_seconds": float(eta_seconds),
        "rank_memory": [dict(row) for row in rank_memory],
        "cumulative_cost_usd": cumulative_cost_usd,
    }


def summarize_named_tensors(
    named_tensors: Iterable[tuple[str, Any]], *, group_depth: int = 3
) -> dict[str, dict[str, float | int]]:
    """Summarize parameters or gradients without retaining full tensors."""

    if group_depth <= 0:
        raise ValueError("group depth must be positive")
    accumulators: dict[str, dict[str, float | int]] = {}
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        group = ".".join(name.split(".")[:group_depth])
        detached = tensor.detach().float()
        count = int(detached.numel())
        row = accumulators.setdefault(
            group,
            {
                "tensor_count": 0,
                "element_count": 0,
                "sum_squares": 0.0,
                "max_abs": 0.0,
                "zero_count": 0,
            },
        )
        row["tensor_count"] = int(row["tensor_count"]) + 1
        row["element_count"] = int(row["element_count"]) + count
        row["sum_squares"] = float(row["sum_squares"]) + float(
            detached.square().sum().item()
        )
        row["max_abs"] = max(
            float(row["max_abs"]), float(detached.abs().max().item())
        )
        row["zero_count"] = int(row["zero_count"]) + int(
            (detached == 0).sum().item()
        )
    summaries: dict[str, dict[str, float | int]] = {}
    for group, row in accumulators.items():
        count = int(row["element_count"])
        sum_squares = float(row["sum_squares"])
        summaries[group] = {
            "tensor_count": int(row["tensor_count"]),
            "element_count": count,
            "l2_norm": math.sqrt(sum_squares),
            "rms": math.sqrt(sum_squares / count),
            "max_abs": float(row["max_abs"]),
            "zero_fraction": int(row["zero_count"]) / count,
        }
    return summaries


def optimizer_state_inventory(optimizer_state: Mapping[str, Any]) -> dict[str, Any]:
    """Record optimizer-state structure and norms, not another full copy."""

    rows = []
    state = optimizer_state.get("state", {})
    for parameter_id, parameter_state in sorted(
        state.items(), key=lambda item: str(item[0])
    ):
        for key, value in sorted(parameter_state.items()):
            row: dict[str, Any] = {
                "parameter_id": str(parameter_id),
                "state_key": str(key),
                "python_type": type(value).__name__,
            }
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                row["shape"] = list(value.shape)
                row["dtype"] = str(value.dtype)
                if getattr(value.dtype, "is_floating_point", False):
                    floating = value.detach().float()
                    row["l2_norm"] = float(floating.norm().item())
                    row["max_abs"] = float(floating.abs().max().item())
            elif isinstance(value, (int, float, bool, str)):
                row["value"] = value
            rows.append(row)
    return {
        "state_entries": len(state),
        "parameter_groups": len(optimizer_state.get("param_groups", [])),
        "rows": rows,
    }


def atomic_install_checkpoint(
    *,
    destination: Path,
    files: Mapping[str, bytes | Path],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Install a hash-verified checkpoint directory through a sibling temp dir."""

    def populate(directory: Path) -> None:
        for relative_name, source in sorted(files.items()):
            _validate_relative_path(relative_name)
            output = directory / relative_name
            output.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(source, Path):
                shutil.copyfile(source, output)
            else:
                output.write_bytes(source)

    return atomic_install_checkpoint_writer(
        destination=destination,
        populate=populate,
        metadata=metadata,
    )


def atomic_install_checkpoint_writer(
    *,
    destination: Path,
    populate: Callable[[Path], None],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically install a large checkpoint populated directly in its temp dir."""

    if destination.exists():
        raise FileExistsError(f"checkpoint destination already exists: {destination}")
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"checkpoint temporary directory exists: {temporary}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        populate(temporary)
        manifest_files = []
        for output in sorted(temporary.rglob("*")):
            if not output.is_file() or output.name == "manifest.json":
                continue
            relative_name = output.relative_to(temporary).as_posix()
            _validate_relative_path(relative_name)
            _fsync_file(output)
            manifest_files.append(
                {
                    "path": relative_name,
                    "bytes": output.stat().st_size,
                    "sha256": _sha256(output),
                }
            )
        manifest = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "metadata": dict(metadata),
            "files": manifest_files,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _fsync_file(manifest_path)
        _fsync_directory(temporary)
        verify_checkpoint_directory(temporary)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def verify_checkpoint_directory(path: Path) -> dict[str, Any]:
    """Reject incomplete, altered, or path-escaping checkpoint manifests."""

    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("checkpoint manifest is missing")
    manifest = _read_json(manifest_path)
    if manifest.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unexpected checkpoint version")
    for row in _list(manifest, "files"):
        relative = str(row["path"])
        _validate_relative_path(relative)
        artifact = path / relative
        if not artifact.is_file() or artifact.stat().st_size != int(row["bytes"]):
            raise ValueError(f"checkpoint file size mismatch: {relative}")
        if _sha256(artifact) != row["sha256"]:
            raise ValueError(f"checkpoint file hash mismatch: {relative}")
    return manifest


def rotate_resume_checkpoints(resume_root: Path, *, retain: int = 2) -> None:
    """Keep the newest verified resume generations after a successful install."""

    if retain <= 0:
        raise ValueError("retain must be positive")
    checkpoints = sorted(
        (path for path in resume_root.glob("resume_*" ) if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for path in checkpoints:
        verify_checkpoint_directory(path)
    for path in checkpoints[retain:]:
        shutil.rmtree(path)


def fresh_process_resume_contract(
    *,
    manifest: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare exact counters, next-window identity, LR, hashes, and topology."""

    metadata = _mapping(manifest, "metadata")
    required = (
        "stage_id",
        "stage_update",
        "global_update",
        "next_stage_window_index",
        "next_window_identity_sha256",
        "next_learning_rate",
        "protocol_sha256",
        "encoded_content_identity_sha256",
        "window_content_identity_sha256",
        "world_size",
    )
    mismatches = [key for key in required if metadata.get(key) != expected.get(key)]
    return {
        "passes": not mismatches,
        "checked_fields": list(required),
        "mismatches": mismatches,
        "finite_next_update_required": True,
    }


def select_document_probe_rows(
    documents: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
    minimum_tokens: int,
    maximum_tokens: int = 512,
) -> tuple[Mapping[str, Any], ...]:
    """Choose held-out excerpts, maximizing document diversity before reuse."""

    eligible = [row for row in documents if int(row["tokens"]) >= minimum_tokens]
    if not eligible:
        raise ValueError("no eligible held-out documents for activation probes")
    rng = random.Random(seed)
    document_order = list(range(len(eligible)))
    rng.shuffle(document_order)
    offsets: dict[int, list[int]] = {}
    for index, row in enumerate(eligible):
        token_count = int(row["tokens"])
        candidates = list(range(0, token_count - minimum_tokens + 1, maximum_tokens))
        rng.shuffle(candidates)
        offsets[index] = candidates
    selected: list[Mapping[str, Any]] = []
    cycle = 0
    while len(selected) < count:
        added = False
        for index in document_order:
            if cycle >= len(offsets[index]):
                continue
            row = dict(eligible[index])
            row["probe_token_offset"] = offsets[index][cycle]
            selected.append(row)
            added = True
            if len(selected) == count:
                break
        if not added:
            raise ValueError("too few eligible held-out excerpts for activation probes")
        cycle += 1
    return tuple(selected)


def selected_probe_positions(
    token_ids: Sequence[int],
    *,
    special_token_ids: Sequence[int] = (),
    rare_token_positions: Sequence[int] = (),
) -> tuple[int, ...]:
    """Freeze stable sequence landmarks plus at most three domain marker positions."""

    if len(token_ids) < 2:
        raise ValueError("activation probe must contain at least two tokens")
    special = set(int(value) for value in special_token_ids)
    non_special = [
        index for index, token_id in enumerate(token_ids) if int(token_id) not in special
    ]
    if not non_special:
        raise ValueError("activation probe contains no non-special tokens")
    last_index = len(non_special) - 1
    positions = {
        non_special[0],
        non_special[last_index // 4],
        non_special[last_index // 2],
        non_special[(3 * last_index) // 4],
        non_special[-1],
    }
    for position in rare_token_positions[:3]:
        if 0 <= int(position) < len(token_ids):
            positions.add(int(position))
    return tuple(sorted(positions))


def build_activation_probe_manifest(
    *,
    execution: Mapping[str, Any],
    tokenizer: Any,
    encoded_report: Mapping[str, Any],
    encoded_dir: Path,
    preservation_prompts_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Materialize 48 held-out probes with exact token IDs and extraction settings."""

    probe_config = _mapping(execution, "activation_probes")
    count = int(probe_config["probes_per_domain"])
    maximum = int(probe_config["maximum_tokens_per_probe"])
    minimum = int(probe_config["minimum_tokens_per_corpus_probe"])
    seed = int(probe_config["seed"])
    pools = {str(row["pool_id"]): row for row in encoded_report["pools"]}
    probes: list[dict[str, Any]] = []

    prompts = _read_json(preservation_prompts_path)
    if len(prompts) != count:
        raise ValueError("modern instruction probes must use all twelve prompts")
    for row in prompts:
        messages = [
            {"role": "user", "content": str(row["prompt"])},
            {"role": "assistant", "content": str(row["response"])},
        ]
        token_ids = list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        )[:maximum]
        probes.append(
            _probe_row(
                probe_id=f"modern_instruction:{row['id']}",
                domain="modern_instruction",
                source_identity=str(row["id"]),
                source_split="instruction_preservation",
                token_ids=token_ids,
                special_token_ids=tokenizer.all_special_ids,
            )
        )

    domain_pools = (
        ("historical_general", "validation_historical_general"),
        (
            "historical_non_sonnet_poetry",
            "validation_historical_non_sonnet_poetry",
        ),
        ("standard_sonnet", "sonnets_validation"),
    )
    required_pools = [pool_id for _, pool_id in domain_pools]
    with Int32ShardStore(
        encoded_dir=encoded_dir,
        encoded_report=encoded_report,
        required_pools=required_pools,
    ) as store:
        for domain_index, (domain, pool_id) in enumerate(domain_pools):
            document_path = encoded_dir / f"{pool_id}.documents.jsonl"
            document_row = pools[pool_id]["document_index"]
            if _sha256(document_path) != document_row["sha256"]:
                raise ValueError("probe document index hash mismatch")
            documents = tuple(_read_jsonl(document_path))
            selected = select_document_probe_rows(
                documents,
                count=count,
                seed=seed + domain_index,
                minimum_tokens=minimum,
                maximum_tokens=maximum,
            )
            for document in selected:
                token_ids = _read_document_prefix(
                    store=store,
                    pool_id=pool_id,
                    document=document,
                    maximum_tokens=maximum,
                )
                probes.append(
                    _probe_row(
                        probe_id=f"{domain}:{document['document_index']:06d}",
                        domain=domain,
                        source_identity=(
                            f"{document['unit_id']}#token_offset="
                            f"{int(document.get('probe_token_offset', 0))}"
                        ),
                        source_split=pool_id,
                        token_ids=token_ids,
                        special_token_ids=tokenizer.all_special_ids,
                    )
                )
    _add_domain_marker_positions(probes, tokenizer.all_special_ids)
    manifest = {
        "probe_version": PROBE_VERSION,
        "status": "frozen_local_token_ids",
        "seed": seed,
        "tokenizer_sha256": execution["lineage"]["tokenizer_sha256"],
        "model_revision": "d1fc0f0e589ae879c5ac763e0e4206a4d14a3f6d",
        "probe_count": len(probes),
        "probes_per_domain": count,
        "domains": {
            domain: sum(row["domain"] == domain for row in probes)
            for domain in sorted({row["domain"] for row in probes})
        },
        "extraction": {
            "module_names": probe_config["module_names"],
            "hidden_state_capture": probe_config["hidden_state_capture"],
            "hidden_state_storage_dtype": probe_config[
                "hidden_state_storage_dtype"
            ],
            "aggregate_computation_dtype": probe_config[
                "aggregate_computation_dtype"
            ],
            "pooling": probe_config["pooling"],
            "attention_summary": probe_config["attention_summary"],
            "bounded_raw_attention": probe_config["bounded_raw_attention"],
            "fixed_logit_summary": probe_config["fixed_logit_summary"],
        },
        "probes": probes,
        "v7_test_accessed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest_sha256": _sha256(output_path)}


def build_public_execution_report(
    *,
    execution_path: Path,
    repo_root: Path,
    local_probe_manifest_path: Path,
) -> dict[str, Any]:
    """Publish aggregate 8F evidence without private token IDs or corpus shards."""

    execution = load_execution_config(execution_path, repo_root)
    probe = _read_json(local_probe_manifest_path)
    if probe.get("probe_version") != PROBE_VERSION or probe.get("probe_count") != 48:
        raise ValueError("activation-probe manifest is not complete")
    protocol = _read_json(_resolve(repo_root, execution["lineage"]["protocol_path"]))
    evidence = execution["evidence_retention"]
    estimated_update_rows = sum(
        int(stage["optimizer_updates"]) for stage in protocol["stages"]
    )
    estimated_evaluations = sum(
        math.ceil(int(stage["optimizer_updates"]) / int(stage["evaluation_interval_updates"]))
        + 1
        for stage in protocol["stages"]
    )
    bundle_path = _resolve(
        repo_root, str(execution["local_paths"]["transfer_bundle_path"])
    )
    bundle_build = None
    if bundle_path.is_file():
        from sonnet_training.minerva_7b_v7_bundle import BUNDLE_VERSION

        with tarfile.open(bundle_path, mode="r:gz") as archive:
            manifest_file = archive.extractfile("bundle_manifest.json")
            if manifest_file is None:
                raise ValueError("local V7 bundle manifest is unreadable")
            verified_bundle = json.loads(manifest_file.read())
        if verified_bundle.get("bundle_version") != BUNDLE_VERSION:
            raise ValueError("local V7 bundle version changed")
        bundle_build = {
            "bytes": bundle_path.stat().st_size,
            "sha256": _sha256(bundle_path),
            "files": len(verified_bundle["files"]),
            "verified": True,
            "public": False,
            "v7_test_material_included": verified_bundle[
                "v7_test_material_included"
            ],
        }
    return {
        "execution_version": EXECUTION_VERSION,
        "build_date": execution["build_date"],
        "status": "cpu_execution_artifacts_verified_gpu_unauthorized",
        "execution_sha256": _sha256(execution_path),
        "lineage": execution["lineage"],
        "reader": {
            "source_span_tokens": protocol["data"]["source_span_tokens"],
            "target_tokens": protocol["data"]["target_tokens_per_window"],
            "global_windows_per_update": protocol["data"][
                "global_windows_per_update"
            ],
            "total_updates": estimated_update_rows,
            "uses_local_memory_mapped_int32_shards": True,
            "repacked_corpus_created": False,
        },
        "checkpointing": {
            "atomic_sibling_install_and_hash_reload": True,
            "fresh_process_resume_contract": True,
            "resume_generations_retained": 2,
        },
        "activation_probes": {
            "manifest_sha256": _sha256(local_probe_manifest_path),
            "manifest_bytes": local_probe_manifest_path.stat().st_size,
            "manifest_public": False,
            "probe_count": probe["probe_count"],
            "domains": probe["domains"],
            "v7_test_accessed": probe["v7_test_accessed"],
            "module_names": probe["extraction"]["module_names"],
            "bounded_raw_attention": probe["extraction"][
                "bounded_raw_attention"
            ],
        },
        "evidence_retention": {
            **evidence,
            "estimated_update_telemetry_rows": estimated_update_rows,
            "estimated_evaluation_events": estimated_evaluations,
        },
        "transfer_bundle": {
            **execution["transfer_bundle"],
            "local_build": bundle_build,
        },
        "hardware_qualification": {
            "candidate_count": 12,
            "context_length": 2048,
            "warmup_updates": 3,
            "timed_updates": 20,
            "execution_command": (
                "torchrun --standalone --nproc_per_node=2 "
                "scripts/qualify_minerva_7b_v7_full_weight.py"
            ),
            "command_not_executed": True,
        },
        "authorization": execution["authorization"],
        "verification": {
            "all_committed_lineage_matches": True,
            "activation_probe_contract_frozen": True,
            "test_index_or_shard_bundled": False,
            "gpu_work_started": False,
            "gpu_rental_started": False,
            "cache_deleted": False,
        },
    }


def render_public_execution_markdown(report: Mapping[str, Any]) -> str:
    evidence = _mapping(report, "evidence_retention")
    probes = _mapping(report, "activation_probes")
    return "\n".join(
        [
            "# Minerva 7B V7 Execution and Evidence Contract",
            "",
            "Checkpoint 8F implements the CPU-verifiable reader, atomic checkpoint",
            "contract, frozen activation probes, evidence-retention policy, transfer",
            "bundle definition, and dual-H100 qualification entry point. GPU execution",
            "remains unauthorized.",
            "",
            "## Exact data execution",
            "",
            f"The reader reconstructs each 2,049-token source span directly from the",
            f"local signed-int32 shards, then forms 2,048 input and shifted-target",
            f"sequences. It consumes 16 frozen windows per optimizer update across",
            f"{int(report['reader']['total_updates']):,} updates. No repacked corpus is created.",
            "",
            "## Preserved evidence",
            "",
            f"Permanent compact telemetry contains one row for every optimizer update",
            f"({int(evidence['estimated_update_telemetry_rows']):,} rows), plus approximately",
            f"{int(evidence['estimated_evaluation_events'])} evaluation events. It records loss,",
            "pre-clipping gradient norm, learning rate, throughput, memory, exact window",
            "identity, cost when available, validation/preservation results, and promotion",
            "decisions. Stage midpoints and ends also retain compact per-module parameter,",
            "gradient, optimizer-state, and allocator summaries. Fixed-probe logit summaries",
            "and deterministic generations accompany the seven model states.",
            "",
            "Full per-update gradients, optimizer copies, ordinary batch activations, and",
            "unbounded attention tensors are deliberately not retained because their cost",
            "would be disproportionate and the saved model states can reproduce probes.",
            "",
            "## Activation probes",
            "",
            f"The ignored local manifest contains {int(probes['probe_count'])} exact probes:",
            "12 modern instructions, 12 historical-general excerpts, 12 historical",
            "non-sonnet-poetry excerpts, and 12 validation sonnets. It freezes token IDs,",
            "masks, positions, all 32 block names, final normalization, pooling, BF16 local",
            "capture, FP32 aggregation, top-20 logits, and a bounded raw-attention sample.",
            f"Its SHA-256 is `{probes['manifest_sha256']}`. V7 test data was not accessed.",
            "",
            "## Checkpoint, transfer, and GPU boundary",
            "",
            "Resume checkpoints install through a sibling temporary directory, fsync and",
            "hash-verification, then atomic rename. The fresh-process proof compares exact",
            "stage/update/window/LR/data/topology state and still requires a finite next",
            "update. The private transfer bundle includes the training/validation shards,",
            "indexes, preservation material, and probe manifest, but excludes V7 test data,",
            "raw corpus caches, model weights, and prior runs.",
            "",
            "The frozen qualification command is:",
            "",
            "```bash",
            str(report["hardware_qualification"]["execution_command"]),
            "```",
            "",
            "It has not been run. GPU rental, qualification, long training, instance",
            "lifecycle actions, test access, and cache deletion remain unauthorized.",
            "",
        ]
    )


def _probe_row(
    *,
    probe_id: str,
    domain: str,
    source_identity: str,
    source_split: str,
    token_ids: Sequence[int],
    special_token_ids: Sequence[int],
) -> dict[str, Any]:
    ids = [int(value) for value in token_ids]
    return {
        "probe_id": probe_id,
        "domain": domain,
        "source_identity": source_identity,
        "source_split": source_split,
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "selected_positions": list(
            selected_probe_positions(ids, special_token_ids=special_token_ids)
        ),
        "input_ids_sha256": hashlib.sha256(
            b"".join(UINT32.pack(value) for value in ids)
        ).hexdigest(),
    }


def _add_domain_marker_positions(
    probes: list[dict[str, Any]], special_token_ids: Sequence[int]
) -> None:
    """Add up to three positions whose tokens are unusually domain-specific."""

    special = set(int(value) for value in special_token_ids)
    domain_counts: dict[str, dict[int, int]] = {}
    global_counts: dict[int, int] = {}
    for probe in probes:
        counts = domain_counts.setdefault(str(probe["domain"]), {})
        for token_id in probe["input_ids"]:
            token_id = int(token_id)
            if token_id in special:
                continue
            counts[token_id] = counts.get(token_id, 0) + 1
            global_counts[token_id] = global_counts.get(token_id, 0) + 1
    for probe in probes:
        domain = str(probe["domain"])
        candidates = []
        for position, token_id in enumerate(probe["input_ids"]):
            token_id = int(token_id)
            if token_id in special:
                continue
            within = domain_counts[domain][token_id]
            outside = global_counts[token_id] - within
            score = (within + 1) / (outside + 1)
            candidates.append((-score, global_counts[token_id], position, token_id))
        selected = []
        seen_tokens = set()
        for _, _, position, token_id in sorted(candidates):
            if token_id not in seen_tokens:
                selected.append(position)
                seen_tokens.add(token_id)
            if len(selected) == 3:
                break
        probe["domain_marker_positions"] = sorted(selected)
        probe["selected_positions"] = list(
            selected_probe_positions(
                probe["input_ids"],
                special_token_ids=special_token_ids,
                rare_token_positions=selected,
            )
        )


def _read_document_prefix(
    *,
    store: Int32ShardStore,
    pool_id: str,
    document: Mapping[str, Any],
    maximum_tokens: int,
) -> tuple[int, ...]:
    start = _mapping(document, "token_start")
    end = _mapping(document, "token_end")
    if int(start["shard_index"]) != int(end["shard_index"]):
        raise ValueError("probe document unexpectedly crosses encoded shards")
    local_offset = int(document.get("probe_token_offset", 0))
    count = min(int(document["tokens"]) - local_offset, maximum_tokens)
    return store.read_slices(
        pool_id,
        [
            {
                "shard_index": int(start["shard_index"]),
                "token_offset": int(start["token_offset"]) + local_offset,
                "token_count": count,
            }
        ],
    )


def _validate_execution_contract(
    execution: Mapping[str, Any], protocol: Mapping[str, Any]
) -> None:
    probes = _mapping(execution, "activation_probes")
    if int(probes["probes_per_domain"]) != 12:
        raise ValueError("8F freezes exactly 12 activation probes per domain")
    if int(probes["maximum_tokens_per_probe"]) > 512:
        raise ValueError("activation probes exceed the approved bounded length")
    runtime = _mapping(execution, "training_runtime")
    if int(runtime["world_size"]) != int(
        protocol["hardware_qualification"]["world_size"]
    ):
        raise ValueError("8F world size differs from the approved protocol")
    authorization = _mapping(execution, "authorization")
    if any(
        authorization[key]
        for key in (
            "gpu_qualification_authorized",
            "gpu_rental_authorized",
            "long_training_authorized",
            "instance_lifecycle_action_authorized",
            "cache_deletion_authorized",
        )
    ):
        raise ValueError("8F may not authorize GPU, lifecycle, or cache actions")


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSONL in {path} at line {line_number}"
                    ) from error


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return result


def _list(value: Mapping[str, Any], key: str) -> list[Any]:
    result = value.get(key)
    if not isinstance(result, list):
        raise ValueError(f"missing list: {key}")
    return result


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_relative_path(value: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value in ("", "."):
        raise ValueError(f"unsafe checkpoint relative path: {value}")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
