"""Freeze validation, encode canonical V7 pools, and plan 2,048-token stages."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sonnet_corpus.canonical_corpus_reader import (
    CanonicalCorpusReader,
    CanonicalTextUnit,
)
from sonnet_corpus.sonnet_v7_split import canonicalize_author_label
from sonnet_training.minerva_7b_full_weight_data import (
    INT32_BYTES,
    ShardedEncodingState,
    _ShardedInt32Writer,
    tokenizer_sha256,
)
from sonnet_training.minerva_7b_qlora import (
    MINERVA_7B_INSTRUCT_MODEL_ID,
    MINERVA_7B_INSTRUCT_REVISION,
)


DATA_VERSION = "minerva_7b_v7_encoded_data_v1"
BROADER_ROLES = (
    "historical_general",
    "historical_non_sonnet_poetry",
    "nineteenth_century_bridge",
)
SPLIT_FIELDS = (
    "unit_id",
    "final_role",
    "source_group",
    "source_id",
    "title",
    "author",
    "component_id",
    "component_key",
    "token_count",
    "split",
    "component_decision",
    "oversize_component",
)
Progress = Callable[[str], None]


@dataclass(frozen=True)
class MinervaV7DataConfig:
    """Pin checkpoint inputs, local outputs, and progress/resume controls."""

    repo_root: Path
    policy_path: Path
    composition_policy_path: Path
    composition_report_path: Path
    canonical_corpus_dir: Path
    v7_manifest_path: Path
    replay_text_path: Path
    replay_report_path: Path
    tokenizer_cache_dir: Path
    output_dir: Path
    reproduction_output_dir: Path
    broader_split_manifest_path: Path
    json_report_path: Path
    markdown_report_path: Path
    max_documents_per_pool_run: int | None = None
    expected_protected_v6_count: int = 387


@dataclass(frozen=True)
class CountedUnit:
    """One verified logical unit plus its exact Minerva accounting."""

    unit: CanonicalTextUnit
    text_tokens: int
    training_tokens: int
    component_key: str
    component_id: str
    v7_split: str
    author_key: str = ""
    work_key: str = ""
    epoch_key: str = ""


@dataclass(frozen=True)
class EncodedDocument:
    """One document assigned to exactly one encoded pool."""

    unit_id: str
    logical_sha256: str
    characters: int
    expected_tokens: int
    source_group: str
    source_id: str
    author_key: str
    work_key: str
    epoch_key: str
    unit: CanonicalTextUnit | None = None
    text_path: Path | None = None


@dataclass(frozen=True)
class PoolSpec:
    """Ordered documents that may share encoded shards and packed windows."""

    pool_id: str
    corpus_role: str
    split: str
    documents: tuple[EncodedDocument, ...]


def load_training_data_policy(path: Path) -> dict[str, Any]:
    """Load and validate the approved checkpoint-8C policy."""

    payload = _read_json(path)
    if payload.get("data_version") != DATA_VERSION:
        raise ValueError("unexpected Minerva V7 encoded-data policy version")
    if payload.get("model_id") != MINERVA_7B_INSTRUCT_MODEL_ID:
        raise ValueError("encoded-data policy is not pinned to Minerva 7B Instruct")
    if payload.get("revision") != MINERVA_7B_INSTRUCT_REVISION:
        raise ValueError("encoded-data policy has the wrong Minerva revision")
    validation = _mapping(payload, "broader_validation")
    target = float(validation["target_fraction_per_role"])
    maximum = float(validation["maximum_component_fraction_per_role"])
    tolerance = float(validation["acceptance_tolerance_fraction"])
    if not 0.0 < target < maximum < 1.0:
        raise ValueError("broader validation fractions are inconsistent")
    if not 0.0 < tolerance < target:
        raise ValueError("broader validation tolerance is inconsistent")
    encoding = _mapping(payload, "encoding")
    if encoding.get("dtype") != "int32" or int(encoding["bytes_per_token"]) != 4:
        raise ValueError("encoded-data storage must remain signed int32")
    if int(encoding["shard_target_tokens"]) <= 2049:
        raise ValueError("encoded shard target is too small")
    windowing = _mapping(payload, "windowing")
    if int(windowing["context_length"]) != 2048:
        raise ValueError("Minerva V7 primary context must remain 2,048")
    if int(windowing["source_span_tokens"]) != 2049:
        raise ValueError("2,048 next-token targets require 2,049 source tokens")
    if int(windowing["budget_alignment_tokens"]) != 40_960:
        raise ValueError("stage budget alignment must preserve ratios and windows")
    return payload


def prepare_and_verify_minerva_v7_data(
    config: MinervaV7DataConfig,
    *,
    tokenizer: Any | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Build primary and reproduction encodings, then activate on exact identity."""

    _validate_config(config)
    policy = load_training_data_policy(config.policy_path)
    if tokenizer is None:
        from transformers import AutoTokenizer

        _report(progress, "loading pinned Minerva 7B tokenizer from local cache")
        tokenizer = AutoTokenizer.from_pretrained(
            MINERVA_7B_INSTRUCT_MODEL_ID,
            revision=MINERVA_7B_INSTRUCT_REVISION,
            cache_dir=config.tokenizer_cache_dir,
            local_files_only=True,
        )
    fingerprint = tokenizer_sha256(tokenizer)
    if fingerprint != policy["tokenizer_sha256"]:
        raise ValueError("loaded Minerva tokenizer fingerprint does not match policy")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token_id, int) or eos_token_id < 0:
        raise ValueError("Minerva tokenizer must define a non-negative EOS token ID")

    _report(progress, "counting verified logical units and freezing validation once")
    reader = CanonicalCorpusReader(
        config.repo_root,
        config.canonical_corpus_dir,
        expected_protected_v6_count=config.expected_protected_v6_count,
    )
    v7_rows = _load_v7_rows(config.v7_manifest_path)
    composition_policy = _read_json(config.composition_policy_path)
    counted = count_canonical_units(
        reader=reader,
        v7_rows=v7_rows,
        tokenizer=tokenizer,
        eos_token_id=eos_token_id,
        epoch_harmonization=_mapping(composition_policy, "epoch_harmonization"),
        progress=progress,
    )
    composition_report = _read_json(config.composition_report_path)
    _validate_count_reproduction(
        counted=counted,
        composition_report=composition_report,
        tokenizer_fingerprint=fingerprint,
    )
    validation = _mapping(policy, "broader_validation")
    split_result = select_broader_validation(
        [row for row in counted if row.unit.unit_kind == "broader"],
        target_fraction=float(validation["target_fraction_per_role"]),
        maximum_component_fraction=float(
            validation["maximum_component_fraction_per_role"]
        ),
        tolerance=float(validation["acceptance_tolerance_fraction"]),
        seed=int(validation["seed"]),
    )
    pools = build_pool_specs(
        counted=counted,
        broader_splits=split_result["unit_splits"],
        replay_text_path=config.replay_text_path,
        replay_report_path=config.replay_report_path,
        tokenizer=tokenizer,
        eos_token_id=eos_token_id,
    )

    _report(progress, "primary build: encoding frozen pools")
    primary = _prepare_one_build(
        config=config,
        policy=policy,
        tokenizer=tokenizer,
        tokenizer_fingerprint=fingerprint,
        eos_token_id=eos_token_id,
        reader=reader,
        counted=counted,
        split_result=split_result,
        pools=pools,
        output_dir=config.output_dir,
        progress=progress,
    )
    if primary["status"] != "complete":
        return primary

    _report(progress, "reproduction build: independently encoding identical pools")
    reproduction = _prepare_one_build(
        config=config,
        policy=policy,
        tokenizer=tokenizer,
        tokenizer_fingerprint=fingerprint,
        eos_token_id=eos_token_id,
        reader=reader,
        counted=counted,
        split_result=split_result,
        pools=pools,
        output_dir=config.reproduction_output_dir,
        progress=progress,
    )
    if reproduction["status"] != "complete":
        return {
            "data_version": DATA_VERSION,
            "status": "incomplete_reproduction",
            "primary": primary,
            "reproduction": reproduction,
        }

    comparison = compare_encoded_builds(primary, reproduction)
    if not comparison["match"]:
        raise ValueError("independent Minerva V7 encoded builds do not match")
    report = dict(primary)
    report.update(
        {
            "status": "active_verified",
            "activation_status": "active_for_local_training_data_use",
            "reproduction": comparison,
            "verification": {
                **primary["verification"],
                "independent_encoded_build_count": 2,
                "independent_content_identities_match": True,
                "broader_roles_activated": True,
                "v7_training_sonnets_activated": True,
                "gpu_work_started": False,
                "cache_deleted": False,
            },
        }
    )
    write_broader_split_manifest(primary["broader_split_rows"], config.broader_split_manifest_path)
    public_report = {key: value for key, value in report.items() if key != "broader_split_rows"}
    _write_json(config.json_report_path, public_report)
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(
        render_encoded_data_markdown(public_report), encoding="utf-8"
    )
    _report(progress, f"activated verified local data: {config.json_report_path}")
    return public_report


def _prepare_one_build(
    *,
    config: MinervaV7DataConfig,
    policy: Mapping[str, Any],
    tokenizer: Any,
    tokenizer_fingerprint: str,
    eos_token_id: int,
    reader: CanonicalCorpusReader,
    counted: Sequence[CountedUnit],
    split_result: Mapping[str, Any],
    pools: Sequence[PoolSpec],
    output_dir: Path,
    progress: Progress | None,
) -> dict[str, Any]:
    encoding = _mapping(policy, "encoding")
    pool_reports: list[dict[str, Any]] = []
    for index, pool in enumerate(pools, start=1):
        _report(progress, f"pool {index}/{len(pools)} start: {pool.pool_id}")
        pool_report = encode_pool(
            pool=pool,
            reader=reader,
            output_dir=output_dir,
            tokenizer=tokenizer,
            tokenizer_fingerprint=tokenizer_fingerprint,
            eos_token_id=eos_token_id,
            shard_target_tokens=int(encoding["shard_target_tokens"]),
            checkpoint_interval_documents=int(
                encoding["checkpoint_interval_documents"]
            ),
            progress_interval_documents=int(encoding["progress_interval_documents"]),
            max_documents=config.max_documents_per_pool_run,
            progress=progress,
        )
        pool_reports.append(pool_report)
        if pool_report["status"] != "complete":
            return {
                "data_version": DATA_VERSION,
                "status": "incomplete",
                "output_dir": _portable(output_dir, config.repo_root),
                "pools": pool_reports,
            }

    pool_by_id = {row["pool_id"]: row for row in pool_reports}
    training_role_tokens = {
        role: int(pool_by_id[f"train_{role}"]["tokens"])
        for role in BROADER_ROLES
    }
    stage_plan = build_exact_stage_plan(
        training_role_tokens=training_role_tokens,
        v7_train_tokens=int(pool_by_id["sonnets_train"]["tokens"]),
        replay_tokens=int(pool_by_id["modern_preservation_replay"]["tokens"]),
        composition_policy=_read_json(config.composition_policy_path),
        data_policy=policy,
    )
    split_rows = build_broader_split_rows(
        [row for row in counted if row.unit.unit_kind == "broader"],
        split_result,
    )
    content_identity = encoded_content_identity(pool_reports)
    return {
        "data_version": DATA_VERSION,
        "build_date": policy["build_date"],
        "status": "complete",
        "activation_status": "encoded_pending_independent_reproduction",
        "model_id": MINERVA_7B_INSTRUCT_MODEL_ID,
        "revision": MINERVA_7B_INSTRUCT_REVISION,
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer),
            "eos_token_id": eos_token_id,
            "serialized_sha256": tokenizer_fingerprint,
        },
        "provenance": {
            "training_data_policy_path": _portable(config.policy_path, config.repo_root),
            "training_data_policy_sha256": _sha256(config.policy_path),
            "composition_policy_path": _portable(
                config.composition_policy_path, config.repo_root
            ),
            "composition_policy_sha256": _sha256(config.composition_policy_path),
            "composition_report_path": _portable(
                config.composition_report_path, config.repo_root
            ),
            "composition_report_sha256": _sha256(config.composition_report_path),
            "v7_manifest_path": _portable(config.v7_manifest_path, config.repo_root),
            "v7_manifest_sha256": _sha256(config.v7_manifest_path),
        },
        "format": {
            "dtype": "int32",
            "bytes_per_token": INT32_BYTES,
            "shard_target_tokens": int(encoding["shard_target_tokens"]),
            "document_boundary": "one Minerva EOS token",
            "document_index_sampling_metadata": (
                "author_key, work_key, and harmonized epoch_key per logical unit"
            ),
            "materialization_level": "role_and_split_token_pools_not_sampled_windows",
            "token_ids_public": False,
        },
        "windowing": {
            **_mapping(policy, "windowing"),
            "sampling_assignment_status": "pending_deterministic_training_sampler",
        },
        "broader_validation": split_result["report"],
        "broader_split_rows": split_rows,
        "stage_plan": stage_plan,
        "totals": {
            "documents": sum(int(row["documents"]) for row in pool_reports),
            "characters": sum(int(row["characters"]) for row in pool_reports),
            "tokens": sum(int(row["tokens"]) for row in pool_reports),
            "eos_tokens": sum(int(row["eos_tokens"]) for row in pool_reports),
            "shards": sum(len(row["shards"]) for row in pool_reports),
            "encoded_bytes": sum(
                int(shard["bytes"])
                for row in pool_reports
                for shard in row["shards"]
            ),
        },
        "pools": pool_reports,
        "encoded_content_identity_sha256": content_identity,
        "verification": {
            "composition_counts_reproduced": True,
            "broader_validation_author_work_disjoint": True,
            "v7_validation_test_training_excluded": True,
            "protected_v6_training_excluded": True,
            "conditioned_material_included": False,
            "all_shards_int32_hash_verified": True,
            "context_length_frozen_2048": True,
            "gpu_work_started": False,
            "cache_deleted": False,
        },
    }


def count_canonical_units(
    *,
    reader: CanonicalCorpusReader,
    v7_rows: Mapping[str, Mapping[str, str]],
    tokenizer: Any,
    eos_token_id: int,
    epoch_harmonization: Mapping[str, Any],
    progress: Progress | None = None,
) -> list[CountedUnit]:
    """Reproduce checkpoint-8B counts while retaining per-unit identities."""

    del eos_token_id  # Its existence is validated; accounting appends exactly one.
    counted: list[CountedUnit] = []
    units = reader.units
    for index, unit in enumerate(units, start=1):
        text_tokens = len(_token_ids(tokenizer, reader.read_text(unit)))
        if text_tokens <= 0:
            raise ValueError(f"logical unit tokenized to zero tokens: {unit.unit_id}")
        component_key = ""
        component_id = ""
        v7_split = ""
        author_key = ""
        work_key = ""
        epoch_key = unit.epoch_bucket
        if unit.unit_kind == "broader":
            work_key = f"{unit.source_group}:{unit.source_id}"
            canonical_author = canonicalize_author_label(unit.author)
            component_key = (
                f"author:{canonical_author}" if canonical_author else f"work:{work_key}"
            )
            author_key = (
                f"author:{canonical_author}"
                if canonical_author
                else f"generic:{work_key}"
            )
            component_id = "component:" + hashlib.sha256(
                component_key.encode("utf-8")
            ).hexdigest()[:16]
        else:
            v7 = v7_rows.get(unit.unit_id)
            if v7 is None or v7.get("include_in_v7") != "true":
                raise ValueError(f"stored sonnet is absent from V7: {unit.unit_id}")
            v7_split = str(v7["v7_split"])
            expected_training = "true" if v7_split == "train" else "false"
            if v7.get("v7_training_eligible") != expected_training:
                raise ValueError(f"V7 training eligibility mismatch: {unit.unit_id}")
            work_key = str(v7["work_group_id"])
            author_key = str(v7["author_group_id"]) or f"generic:{work_key}"
            raw_epoch = str(v7["epoch_bucket"])
            if raw_epoch not in epoch_harmonization:
                raise ValueError(f"unmapped V7 epoch bucket: {raw_epoch}")
            epoch_key = str(epoch_harmonization[raw_epoch])
        counted.append(
            CountedUnit(
                unit=unit,
                text_tokens=text_tokens,
                training_tokens=text_tokens + 1,
                component_key=component_key,
                component_id=component_id,
                v7_split=v7_split,
                author_key=author_key,
                work_key=work_key,
                epoch_key=epoch_key,
            )
        )
        if index % 100 == 0 or index == len(units):
            _report(
                progress,
                f"counted logical units={index:,}/{len(units):,} "
                f"progress={index / len(units):.1%}",
            )
    return counted


def select_broader_validation(
    counted: Sequence[CountedUnit],
    *,
    target_fraction: float,
    maximum_component_fraction: float,
    tolerance: float,
    seed: int,
) -> dict[str, Any]:
    """Select deterministic global author/work components near 1% per role."""

    role_totals = Counter()
    component_roles: dict[str, Counter[str]] = defaultdict(Counter)
    component_ids: dict[str, str] = {}
    for row in counted:
        role = row.unit.final_role
        if role not in BROADER_ROLES:
            raise ValueError(f"unexpected broader role: {role}")
        role_totals[role] += row.training_tokens
        component_roles[row.component_key][role] += row.training_tokens
        component_ids[row.component_key] = row.component_id
    if set(role_totals) != set(BROADER_ROLES):
        raise ValueError("broader validation selection is missing a corpus role")

    oversize: set[str] = set()
    for key, role_counts in component_roles.items():
        if any(
            role_counts[role] / role_totals[role] > maximum_component_fraction
            for role in BROADER_ROLES
        ):
            oversize.add(key)
    targets = {
        role: role_totals[role] * target_fraction for role in BROADER_ROLES
    }
    selected: set[str] = set()
    selected_totals = Counter()
    current_score = _selection_score(selected_totals, targets)
    candidates = sorted(
        set(component_roles) - oversize,
        key=lambda key: (
            hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest(),
            key,
        ),
    )
    while True:
        best_key: str | None = None
        best_score = current_score
        for key in candidates:
            if key in selected:
                continue
            proposed = selected_totals + component_roles[key]
            score = _selection_score(proposed, targets)
            if score < best_score - 1e-15:
                best_key = key
                best_score = score
        if best_key is None:
            break
        selected.add(best_key)
        selected_totals.update(component_roles[best_key])
        current_score = best_score

    role_rows = {}
    for role in BROADER_ROLES:
        fraction = selected_totals[role] / role_totals[role]
        minimum = target_fraction - tolerance
        maximum = target_fraction + tolerance
        role_rows[role] = {
            "total_tokens": role_totals[role],
            "validation_tokens": selected_totals[role],
            "training_tokens": role_totals[role] - selected_totals[role],
            "validation_fraction": fraction,
            "target_fraction": target_fraction,
            "acceptance_minimum": minimum,
            "acceptance_maximum": maximum,
            "passes_tolerance": minimum <= fraction <= maximum,
        }
    if not all(row["passes_tolerance"] for row in role_rows.values()):
        raise ValueError("broader validation selection missed its approved tolerance")

    unit_splits = {
        row.unit.unit_id: ("validation" if row.component_key in selected else "train")
        for row in counted
    }
    train_components = set(component_roles) - selected
    if selected & train_components:
        raise AssertionError("broader validation component crossed the split boundary")
    selected_identity = hashlib.sha256()
    for key in sorted(selected):
        selected_identity.update(component_ids[key].encode("ascii"))
        selected_identity.update(b"\n")
    return {
        "unit_splits": unit_splits,
        "selected_components": selected,
        "oversize_components": oversize,
        "component_ids": component_ids,
        "report": {
            "target_fraction_per_role": target_fraction,
            "maximum_component_fraction_per_role": maximum_component_fraction,
            "acceptance_tolerance_fraction": tolerance,
            "seed": seed,
            "component_count": len(component_roles),
            "validation_component_count": len(selected),
            "oversize_training_component_count": len(oversize),
            "selected_component_identity_sha256": selected_identity.hexdigest(),
            "roles": role_rows,
            "author_work_components_disjoint": True,
        },
    }


def build_broader_split_rows(
    counted: Sequence[CountedUnit], split_result: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Create the public per-unit broader split ledger without corpus text."""

    unit_splits = _mapping(split_result, "unit_splits")
    selected = set(split_result["selected_components"])
    oversize = set(split_result["oversize_components"])
    rows = []
    for row in sorted(counted, key=lambda item: item.unit.unit_id):
        split = str(unit_splits[row.unit.unit_id])
        if row.component_key in selected:
            decision = "selected_global_author_work_validation_component"
        elif row.component_key in oversize:
            decision = "oversize_component_retained_training"
        else:
            decision = "deterministic_training_component"
        rows.append(
            {
                "unit_id": row.unit.unit_id,
                "final_role": row.unit.final_role,
                "source_group": row.unit.source_group,
                "source_id": row.unit.source_id,
                "title": row.unit.title,
                "author": row.unit.author,
                "component_id": row.component_id,
                "component_key": row.component_key,
                "token_count": row.training_tokens,
                "split": split,
                "component_decision": decision,
                "oversize_component": str(row.component_key in oversize).lower(),
            }
        )
    return rows


def write_broader_split_manifest(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write the deterministic public broader train/validation ledger."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=SPLIT_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def build_pool_specs(
    *,
    counted: Sequence[CountedUnit],
    broader_splits: Mapping[str, Any],
    replay_text_path: Path,
    replay_report_path: Path,
    tokenizer: Any,
    eos_token_id: int,
) -> tuple[PoolSpec, ...]:
    """Assign every canonical unit and replay sample to one isolated pool."""

    del eos_token_id
    pools: dict[str, list[EncodedDocument]] = defaultdict(list)
    for row in counted:
        unit = row.unit
        if unit.unit_kind == "broader":
            split = str(broader_splits[unit.unit_id])
            pool_id = f"{split}_{unit.final_role}"
            author_key = row.author_key
            work_key = row.work_key
            epoch_key = row.epoch_key
        else:
            split = row.v7_split
            pool_id = f"sonnets_{split}"
            author_key = row.author_key
            work_key = row.work_key
            epoch_key = row.epoch_key
        pools[pool_id].append(
            EncodedDocument(
                unit_id=unit.unit_id,
                logical_sha256=unit.logical_sha256,
                characters=unit.logical_character_count,
                expected_tokens=row.training_tokens,
                source_group=unit.source_group,
                source_id=unit.source_id,
                author_key=author_key,
                work_key=work_key,
                epoch_key=epoch_key,
                unit=unit,
            )
        )

    replay_report = _read_json(replay_report_path)
    replay_sha = _sha256(replay_text_path)
    if replay_report.get("output_sha256") != replay_sha:
        raise ValueError("modern replay does not match its lineage report")
    replay_text = replay_text_path.read_text(encoding="utf-8")
    replay_tokens = len(_token_ids(tokenizer, replay_text)) + 1
    pools["modern_preservation_replay"].append(
        EncodedDocument(
            unit_id="paisa_even_byte_windows_v1",
            logical_sha256=replay_sha,
            characters=len(replay_text),
            expected_tokens=replay_tokens,
            source_group="paisa_local",
            source_id="paisa_even_byte_windows_v1",
            author_key="",
            work_key="paisa_even_byte_windows_v1",
            epoch_key="modern_preservation",
            text_path=replay_text_path,
        )
    )
    expected_ids = {
        *(f"train_{role}" for role in BROADER_ROLES),
        *(f"validation_{role}" for role in BROADER_ROLES),
        "sonnets_train",
        "sonnets_validation",
        "sonnets_test",
        "modern_preservation_replay",
    }
    if set(pools) != expected_ids:
        raise ValueError(f"encoded pool accounting mismatch: {sorted(set(pools) ^ expected_ids)}")
    specs = []
    for pool_id in sorted(pools):
        if pool_id.startswith("train_") or pool_id.startswith("validation_"):
            split, role = pool_id.split("_", 1)
        elif pool_id.startswith("sonnets_"):
            split = pool_id.removeprefix("sonnets_")
            role = "standard_sonnets"
        else:
            split = "train"
            role = "modern_preservation_replay"
        specs.append(
            PoolSpec(
                pool_id=pool_id,
                corpus_role=role,
                split=split,
                documents=tuple(sorted(pools[pool_id], key=lambda item: item.unit_id)),
            )
        )
    return tuple(specs)


def encode_pool(
    *,
    pool: PoolSpec,
    reader: CanonicalCorpusReader,
    output_dir: Path,
    tokenizer: Any,
    tokenizer_fingerprint: str,
    eos_token_id: int,
    shard_target_tokens: int,
    checkpoint_interval_documents: int,
    progress_interval_documents: int,
    max_documents: int | None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Encode one isolated pool with completed-document crash recovery."""

    if not pool.documents:
        raise ValueError(f"encoded pool is empty: {pool.pool_id}")
    pool_identity = _pool_identity(pool)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / f"{pool.pool_id}.metadata.json"
    completed = _load_completed_pool(
        metadata_path=metadata_path,
        pool=pool,
        pool_identity=pool_identity,
        tokenizer_fingerprint=tokenizer_fingerprint,
        shard_target_tokens=shard_target_tokens,
    )
    if completed is not None:
        _report(progress, f"{pool.pool_id}: verified existing completed output")
        return completed

    checkpoint_path = output_dir / f".{pool.pool_id}.checkpoint.json"
    index_part_path = output_dir / f".{pool.pool_id}.documents.jsonl.part"
    index_path = output_dir / f"{pool.pool_id}.documents.jsonl"
    state = _load_pool_state(
        output_dir=output_dir,
        pool=pool,
        pool_identity=pool_identity,
        checkpoint_path=checkpoint_path,
        index_part_path=index_part_path,
        tokenizer_fingerprint=tokenizer_fingerprint,
        shard_target_tokens=shard_target_tokens,
    )
    writer = _ShardedInt32Writer(
        output_dir=output_dir,
        split_id=pool.pool_id,
        shard_target_tokens=shard_target_tokens,
        state=state,
    )
    invocation_started = time.monotonic()
    invocation_documents = 0
    index_part_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with index_part_path.open(
            "r+b" if index_part_path.exists() else "w+b"
        ) as index_handle:
            index_handle.truncate(state.index_bytes)
            index_handle.seek(state.index_bytes)
            if not checkpoint_path.exists():
                writer.flush()
                index_handle.flush()
                os.fsync(index_handle.fileno())
                _persist_pool_state(
                    pool=pool,
                    pool_identity=pool_identity,
                    state=state,
                    checkpoint_path=checkpoint_path,
                    tokenizer_fingerprint=tokenizer_fingerprint,
                    shard_target_tokens=shard_target_tokens,
                )
            for document in pool.documents[state.documents:]:
                text = _read_encoded_document(reader, document)
                token_ids = _token_ids(tokenizer, text)
                token_ids.append(eos_token_id)
                if len(token_ids) != document.expected_tokens:
                    raise ValueError(
                        f"token count changed during encoding: {document.unit_id}"
                    )
                _validate_token_ids(token_ids, len(tokenizer))
                start, end = writer.write_document(token_ids)
                index_row = {
                    "pool_id": pool.pool_id,
                    "corpus_role": pool.corpus_role,
                    "split": pool.split,
                    "document_index": state.documents,
                    "unit_id": document.unit_id,
                    "logical_sha256": document.logical_sha256,
                    "characters": document.characters,
                    "tokens": len(token_ids),
                    "source_group": document.source_group,
                    "source_id": document.source_id,
                    "author_key": document.author_key,
                    "work_key": document.work_key,
                    "epoch_key": document.epoch_key,
                    "token_start": start,
                    "token_end": end,
                }
                index_handle.write(
                    (json.dumps(index_row, ensure_ascii=False, sort_keys=True) + "\n").encode(
                        "utf-8"
                    )
                )
                state.documents += 1
                state.characters += document.characters
                state.tokens += len(token_ids)
                state.eos_tokens += 1
                state.index_bytes = index_handle.tell()
                invocation_documents += 1

                should_checkpoint = (
                    state.documents % checkpoint_interval_documents == 0
                    or (
                        max_documents is not None
                        and invocation_documents >= max_documents
                    )
                )
                if should_checkpoint:
                    writer.flush()
                    index_handle.flush()
                    os.fsync(index_handle.fileno())
                    _persist_pool_state(
                        pool=pool,
                        pool_identity=pool_identity,
                        state=state,
                        checkpoint_path=checkpoint_path,
                        tokenizer_fingerprint=tokenizer_fingerprint,
                        shard_target_tokens=shard_target_tokens,
                    )
                if (
                    state.documents % progress_interval_documents == 0
                    or state.documents == len(pool.documents)
                ):
                    elapsed = max(time.monotonic() - invocation_started, 1e-9)
                    rate = invocation_documents / elapsed
                    remaining = (len(pool.documents) - state.documents) / rate if rate else 0.0
                    _report(
                        progress,
                        f"{pool.pool_id}: documents={state.documents:,}/"
                        f"{len(pool.documents):,} progress="
                        f"{state.documents / len(pool.documents):.1%} "
                        f"tokens={state.tokens:,} eta={_format_duration(remaining)}",
                    )
                if max_documents is not None and invocation_documents >= max_documents:
                    writer.close_incomplete()
                    return {
                        "pool_id": pool.pool_id,
                        "status": "incomplete",
                        "documents": state.documents,
                        "tokens": state.tokens,
                        "checkpoint_path": _local_artifact_path(checkpoint_path),
                    }
            writer.finalize_current_shard()
            index_handle.flush()
            os.fsync(index_handle.fileno())
    except Exception:
        writer.close_incomplete()
        raise

    if state.documents != len(pool.documents):
        raise ValueError(f"encoded pool document count mismatch: {pool.pool_id}")
    index_part_path.replace(index_path)
    report = _complete_pool_report(
        pool=pool,
        state=state,
        pool_identity=pool_identity,
        index_path=index_path,
        tokenizer_fingerprint=tokenizer_fingerprint,
        shard_target_tokens=shard_target_tokens,
    )
    _write_json(metadata_path, report)
    checkpoint_path.unlink(missing_ok=True)
    _report(
        progress,
        f"{pool.pool_id}: complete documents={state.documents:,} "
        f"tokens={state.tokens:,} shards={len(state.completed_shards)}",
    )
    return report


def build_exact_stage_plan(
    *,
    training_role_tokens: Mapping[str, int],
    v7_train_tokens: int,
    replay_tokens: int,
    composition_policy: Mapping[str, Any],
    data_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Floor one-primary-pass stage budgets to exact ratios and 2,048 windows."""

    composition_stages = _mapping(composition_policy, "stages")
    alignment = int(_mapping(data_policy, "windowing")["budget_alignment_tokens"])
    available_tokens = {
        "historical_general": int(training_role_tokens["historical_general"]),
        "historical_non_sonnet_poetry": int(
            training_role_tokens["historical_non_sonnet_poetry"]
        ),
        "nineteenth_century_bridge": int(
            training_role_tokens["nineteenth_century_bridge"]
        ),
        "modern_preservation_replay": int(replay_tokens),
        "standard_sonnets_v7_train": int(v7_train_tokens),
    }
    available_tokens["stage_1_historical_replay"] = (
        available_tokens["historical_general"]
        + available_tokens["nineteenth_century_bridge"]
    )
    available_tokens["stage_2_historical_replay"] = (
        available_tokens["stage_1_historical_replay"]
        + available_tokens["historical_non_sonnet_poetry"]
    )
    primary_tokens = {
        "stage_1_historical_general": int(training_role_tokens["historical_general"]),
        "stage_2_non_sonnet_poetry": int(
            training_role_tokens["historical_non_sonnet_poetry"]
        ),
        "stage_3_sonnets": int(v7_train_tokens),
    }
    primary_component = {
        "stage_1_historical_general": "historical_general",
        "stage_2_non_sonnet_poetry": "historical_non_sonnet_poetry",
        "stage_3_sonnets": "standard_sonnets_v7_train",
    }
    stages = []
    total_budget = 0
    total_windows = 0
    for stage_id in (
        "stage_1_historical_general",
        "stage_2_non_sonnet_poetry",
        "stage_3_sonnets",
    ):
        components = _mapping(_mapping(composition_stages, stage_id), "components")
        if not math.isclose(
            sum(float(value) for value in components.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"stage component shares do not sum to one: {stage_id}")
        primary = primary_component[stage_id]
        share = float(components[primary])
        maximum_total = primary_tokens[stage_id] / share
        budget = math.floor(maximum_total / alignment) * alignment
        if budget <= 0:
            raise ValueError(f"stage budget is empty: {stage_id}")
        component_rows = []
        allocated = 0
        for name, value in components.items():
            if name not in available_tokens:
                raise ValueError(f"stage component is unavailable: {stage_id} {name}")
            tokens = round(budget * float(value))
            if tokens % 2048:
                raise ValueError(
                    f"stage component is not aligned to whole windows: {stage_id} {name}"
                )
            available = available_tokens[name]
            if available <= 0:
                raise ValueError(f"stage component has no tokens: {stage_id} {name}")
            component_rows.append(
                {
                    "component": name,
                    "target_share": float(value),
                    "available_tokens": available,
                    "draw_tokens": tokens,
                    "draw_windows_2048": tokens // 2048,
                    "draw_to_available_ratio": tokens / available,
                }
            )
            allocated += tokens
        if allocated != budget:
            raise ValueError(f"stage component rounding drifted: {stage_id}")
        primary_draw = next(
            row["draw_tokens"] for row in component_rows if row["component"] == primary
        )
        if primary_draw > primary_tokens[stage_id]:
            raise ValueError(f"stage primary draw exceeds available tokens: {stage_id}")
        windows = budget // 2048
        stages.append(
            {
                "stage_id": stage_id,
                "primary_component": primary,
                "primary_available_tokens": primary_tokens[stage_id],
                "budget_tokens": budget,
                "target_windows_2048": windows,
                "unused_primary_tail_tokens": primary_tokens[stage_id] - primary_draw,
                "components": component_rows,
            }
        )
        total_budget += budget
        total_windows += windows
    return {
        "budget_mode": "one_primary_role_pass_floor_aligned",
        "context_length": 2048,
        "source_span_tokens": 2049,
        "alignment_tokens": alignment,
        "stages": stages,
        "total_budget_tokens": total_budget,
        "total_target_windows_2048": total_windows,
        "modern_replay_available_tokens": replay_tokens,
        "component_available_tokens": available_tokens,
        "sampling_assignment_status": "pending_deterministic_training_sampler",
        "sampling_constraints": {
            "broader_work_maximum_share": float(
                _mapping(composition_policy, "concentration_ceilings")["broader_work"]
            ),
            "broader_author_maximum_share": float(
                _mapping(composition_policy, "concentration_ceilings")["broader_author"]
            ),
            "sonnet_author_maximum_share": float(
                _mapping(composition_policy, "concentration_ceilings")["sonnet_author"]
            ),
            "sonnet_epoch_maximum_share": float(
                _mapping(composition_policy, "concentration_ceilings")["sonnet_epoch"]
            ),
            "enforcement": (
                "the later deterministic sampler must enforce these ceilings; "
                "this checkpoint materializes role/split pools, not sampled windows"
            ),
        },
        "packing_policy": (
            "pack only within one role and split using one EOS per logical document; "
            "extract 2,049-token source spans for 2,048 next-token targets"
        ),
    }


def compare_encoded_builds(
    primary: Mapping[str, Any], reproduction: Mapping[str, Any]
) -> dict[str, Any]:
    """Require independently written pools, indexes, and shards to match by content."""

    primary_pools = {row["pool_id"]: row for row in primary["pools"]}
    reproduction_pools = {row["pool_id"]: row for row in reproduction["pools"]}
    if set(primary_pools) != set(reproduction_pools):
        return {"match": False, "reason": "pool IDs differ"}
    pool_rows = []
    for pool_id in sorted(primary_pools):
        left = primary_pools[pool_id]
        right = reproduction_pools[pool_id]
        fields_match = all(
            left[field] == right[field]
            for field in (
                "documents",
                "characters",
                "tokens",
                "eos_tokens",
                "pool_identity_sha256",
                "content_identity_sha256",
            )
        )
        pool_rows.append({"pool_id": pool_id, "match": fields_match})
    match = (
        all(row["match"] for row in pool_rows)
        and primary["encoded_content_identity_sha256"]
        == reproduction["encoded_content_identity_sha256"]
        and primary["broader_validation"] == reproduction["broader_validation"]
        and primary["stage_plan"] == reproduction["stage_plan"]
    )
    return {
        "match": match,
        "independent_build_count": 2,
        "content_identity_sha256": primary["encoded_content_identity_sha256"],
        "pool_results": pool_rows,
        "primary_and_reproduction_paths_distinct": True,
        "reproduction_artifacts_public": False,
    }


def encoded_content_identity(pool_reports: Sequence[Mapping[str, Any]]) -> str:
    """Hash pool content independently of the physical output directory."""

    digest = hashlib.sha256()
    for pool in sorted(pool_reports, key=lambda row: str(row["pool_id"])):
        digest.update(str(pool["pool_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(pool["content_identity_sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def render_encoded_data_markdown(report: Mapping[str, Any]) -> str:
    """Render the public checkpoint-8C activation and scale summary."""

    totals = _mapping(report, "totals")
    validation = _mapping(report, "broader_validation")
    stage_plan = _mapping(report, "stage_plan")
    lines = [
        "# Minerva 7B V7 Encoded Data And Stage Plan",
        "",
        f"Status: **{str(report['status']).upper()}**.",
        "",
        "Two independent local builds produced the same content identity. The",
        "canonical broader roles and V7 training sonnets are active for local",
        "training-data use; this does not authorize a GPU benchmark or training run.",
        "",
        "## Encoded Scale",
        "",
        "| Measurement | Value |",
        "| --- | ---: |",
        f"| Documents | {int(totals['documents']):,} |",
        f"| Tokens | {int(totals['tokens']):,} |",
        f"| EOS boundaries | {int(totals['eos_tokens']):,} |",
        f"| Shards | {int(totals['shards']):,} |",
        f"| Encoded bytes | {int(totals['encoded_bytes']):,} |",
        "",
        "## Broader Validation",
        "",
        "| Role | Validation tokens | Fraction | Pass |",
        "| --- | ---: | ---: | --- |",
    ]
    for role in BROADER_ROLES:
        row = validation["roles"][role]
        lines.append(
            f"| {role} | {int(row['validation_tokens']):,} | "
            f"{float(row['validation_fraction']):.2%} | "
            f"{row['passes_tolerance']} |"
        )
    lines.extend(
        [
            "",
            "## Exact Initial Stage Budgets",
            "",
            "| Stage | Budget tokens | 2,048-token windows | Unused primary tail |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for stage in stage_plan["stages"]:
        lines.append(
            f"| {stage['stage_id']} | {int(stage['budget_tokens']):,} | "
            f"{int(stage['target_windows_2048']):,} | "
            f"{int(stage['unused_primary_tail_tokens']):,} |"
        )
    lines.extend(
        [
            "",
            "Training windows pack documents only within the same role and split,",
            "with one EOS boundary. Each 2,049-token source span yields 2,048",
            "next-token targets. Validation windows are fixed and non-overlapping.",
            "Document indexes preserve author/work groups and harmonized epoch",
            "labels so the later sampler can enforce the frozen concentration caps.",
            "This checkpoint materializes token pools and exact whole-window budgets,",
            "not sampled window assignments. The later deterministic sampler must",
            "enforce the frozen broader work/author and sonnet author/epoch ceilings.",
            "",
            "## Safety Boundary",
            "",
            "V7 validation/test, broader validation, and protected V6 sonnets remain",
            "outside every training pool. Conditioned material is absent. Token shards,",
            "document indexes, and the PAISÀ replay remain local and ignored. No GPU",
            "work starts and no reusable cache is deleted.",
            "",
        ]
    )
    return "\n".join(lines)


def _complete_pool_report(
    *,
    pool: PoolSpec,
    state: ShardedEncodingState,
    pool_identity: str,
    index_path: Path,
    tokenizer_fingerprint: str,
    shard_target_tokens: int,
) -> dict[str, Any]:
    index_sha = _sha256(index_path)
    content = hashlib.sha256()
    content.update(pool.pool_id.encode("utf-8"))
    content.update(b"\0")
    content.update(pool_identity.encode("ascii"))
    content.update(b"\0")
    content.update(index_sha.encode("ascii"))
    content.update(b"\n")
    for shard in state.completed_shards:
        content.update(str(shard["token_count"]).encode("ascii"))
        content.update(b"\0")
        content.update(str(shard["sha256"]).encode("ascii"))
        content.update(b"\n")
    return {
        "pool_id": pool.pool_id,
        "corpus_role": pool.corpus_role,
        "split": pool.split,
        "status": "complete",
        "pool_identity_sha256": pool_identity,
        "documents": state.documents,
        "characters": state.characters,
        "tokens": state.tokens,
        "eos_tokens": state.eos_tokens,
        "tokenizer_sha256": tokenizer_fingerprint,
        "shard_target_tokens": shard_target_tokens,
        "shards": state.completed_shards,
        "document_index": {
            "path": _local_artifact_path(index_path),
            "bytes": index_path.stat().st_size,
            "sha256": index_sha,
            "public": False,
        },
        "content_identity_sha256": content.hexdigest(),
    }


def _load_completed_pool(
    *,
    metadata_path: Path,
    pool: PoolSpec,
    pool_identity: str,
    tokenizer_fingerprint: str,
    shard_target_tokens: int,
) -> dict[str, Any] | None:
    if not metadata_path.is_file():
        return None
    report = _read_json(metadata_path)
    expected = {
        "pool_id": pool.pool_id,
        "status": "complete",
        "pool_identity_sha256": pool_identity,
        "documents": len(pool.documents),
        "tokenizer_sha256": tokenizer_fingerprint,
        "shard_target_tokens": shard_target_tokens,
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise ValueError(f"completed pool metadata mismatch: {pool.pool_id} {field}")
    shards = report.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"completed pool has no shards: {pool.pool_id}")
    for shard in shards:
        path = Path(str(shard["path"]))
        if not path.is_file() or path.stat().st_size != int(shard["bytes"]):
            raise ValueError(f"completed encoded shard is missing: {path}")
        if _sha256(path) != shard["sha256"]:
            raise ValueError(f"completed encoded shard hash mismatch: {path}")
    index = _mapping(report, "document_index")
    index_path = Path(str(index["path"]))
    if not index_path.is_file() or index_path.stat().st_size != int(index["bytes"]):
        raise ValueError(f"completed pool index is missing: {pool.pool_id}")
    if _sha256(index_path) != index["sha256"]:
        raise ValueError(f"completed pool index hash mismatch: {pool.pool_id}")
    return report


def _load_pool_state(
    *,
    output_dir: Path,
    pool: PoolSpec,
    pool_identity: str,
    checkpoint_path: Path,
    index_part_path: Path,
    tokenizer_fingerprint: str,
    shard_target_tokens: int,
) -> ShardedEncodingState:
    if not checkpoint_path.is_file():
        if index_part_path.exists():
            raise ValueError(f"orphaned encoded index: {index_part_path}")
        return ShardedEncodingState()
    payload = _read_json(checkpoint_path)
    identity = _mapping(payload, "identity")
    expected = {
        "pool_id": pool.pool_id,
        "pool_identity_sha256": pool_identity,
        "expected_documents": len(pool.documents),
        "tokenizer_sha256": tokenizer_fingerprint,
        "shard_target_tokens": shard_target_tokens,
    }
    for field, value in expected.items():
        if identity.get(field) != value:
            raise ValueError(f"encoded checkpoint mismatch: {pool.pool_id} {field}")
    saved = _mapping(payload, "state")
    state = ShardedEncodingState(
        index_bytes=int(saved["index_bytes"]),
        documents=int(saved["documents"]),
        characters=int(saved["characters"]),
        tokens=int(saved["tokens"]),
        eos_tokens=int(saved["eos_tokens"]),
        current_shard_index=int(saved["current_shard_index"]),
        current_shard_tokens=int(saved["current_shard_tokens"]),
        completed_shards=list(saved["completed_shards"]),
    )
    if not 0 <= state.documents <= len(pool.documents):
        raise ValueError(f"encoded checkpoint document offset is invalid: {pool.pool_id}")
    if not index_part_path.is_file() or index_part_path.stat().st_size < state.index_bytes:
        raise ValueError(f"encoded checkpoint index is truncated: {pool.pool_id}")
    for shard in state.completed_shards:
        path = Path(str(shard["path"]))
        if not path.is_file() or path.stat().st_size != int(shard["bytes"]):
            raise ValueError(f"encoded checkpoint shard is missing: {path}")
    current_part = output_dir / (
        f".{pool.pool_id}-{state.current_shard_index:05d}.int32.bin.part"
    )
    current_final = output_dir / (
        f"{pool.pool_id}-{state.current_shard_index:05d}.int32.bin"
    )
    expected_bytes = state.current_shard_tokens * INT32_BYTES
    if not current_part.exists() and current_final.exists():
        if current_final.stat().st_size != expected_bytes:
            raise ValueError(f"uncheckpointed shard rollover is inconsistent: {pool.pool_id}")
        current_final.replace(current_part)
    for stale in output_dir.glob(f".{pool.pool_id}-*.int32.bin.part"):
        if stale != current_part:
            stale.unlink()
    for stale in output_dir.glob(f"{pool.pool_id}-*.int32.bin"):
        index_text = stale.name.removeprefix(f"{pool.pool_id}-").split(".", 1)[0]
        if int(index_text) >= state.current_shard_index:
            stale.unlink()
    if not current_part.is_file() or current_part.stat().st_size < expected_bytes:
        raise ValueError(f"active encoded checkpoint shard is truncated: {pool.pool_id}")
    return state


def _persist_pool_state(
    *,
    pool: PoolSpec,
    pool_identity: str,
    state: ShardedEncodingState,
    checkpoint_path: Path,
    tokenizer_fingerprint: str,
    shard_target_tokens: int,
) -> None:
    _write_json(
        checkpoint_path,
        {
            "identity": {
                "pool_id": pool.pool_id,
                "pool_identity_sha256": pool_identity,
                "expected_documents": len(pool.documents),
                "tokenizer_sha256": tokenizer_fingerprint,
                "shard_target_tokens": shard_target_tokens,
            },
            "state": asdict(state),
        },
    )


def _pool_identity(pool: PoolSpec) -> str:
    digest = hashlib.sha256()
    for document in pool.documents:
        digest.update(document.unit_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(document.logical_sha256.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(document.expected_tokens).encode("ascii"))
        digest.update(b"\0")
        digest.update(document.author_key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(document.work_key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(document.epoch_key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _selection_score(selected: Mapping[str, int], targets: Mapping[str, float]) -> float:
    return sum(
        ((float(selected.get(role, 0)) - target) / target) ** 2
        for role, target in targets.items()
    )


def _validate_count_reproduction(
    *,
    counted: Sequence[CountedUnit],
    composition_report: Mapping[str, Any],
    tokenizer_fingerprint: str,
) -> None:
    totals = _mapping(composition_report, "totals")
    if len(counted) != int(totals["documents"]):
        raise ValueError("encoded-data count did not reproduce logical-unit count")
    if sum(row.training_tokens for row in counted) != int(totals["training_tokens"]):
        raise ValueError("encoded-data count did not reproduce checkpoint-8B tokens")
    tokenizer = _mapping(composition_report, "tokenizer")
    if tokenizer.get("serialized_sha256") != tokenizer_fingerprint:
        raise ValueError("composition report tokenizer fingerprint changed")


def _read_encoded_document(
    reader: CanonicalCorpusReader, document: EncodedDocument
) -> str:
    if document.unit is not None:
        return reader.read_text(document.unit)
    if document.text_path is None:
        raise ValueError(f"encoded document has no text source: {document.unit_id}")
    text = document.text_path.read_text(encoding="utf-8")
    if hashlib.sha256(document.text_path.read_bytes()).hexdigest() != document.logical_sha256:
        raise ValueError(f"local replay content changed: {document.unit_id}")
    return text


def _load_v7_rows(path: Path) -> dict[str, dict[str, str]]:
    required = {
        "unit_id",
        "include_in_v7",
        "v7_split",
        "v7_training_eligible",
        "author_group_id",
        "work_group_id",
        "epoch_bucket",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("V7 manifest is missing encoded-data fields")
        rows = {}
        for row in reader:
            unit_id = row["unit_id"]
            if unit_id in rows:
                raise ValueError(f"duplicate V7 identity: {unit_id}")
            rows[unit_id] = row
    return rows


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    token_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if not isinstance(token_ids, list) or any(
        not isinstance(token_id, int) for token_id in token_ids
    ):
        raise ValueError("Minerva tokenizer must return a list of input_ids")
    return token_ids


def _validate_token_ids(token_ids: Sequence[int], vocab_size: int) -> None:
    if any(token_id < 0 or token_id >= vocab_size for token_id in token_ids):
        raise ValueError("tokenizer returned an out-of-vocabulary token ID")


def _validate_config(config: MinervaV7DataConfig) -> None:
    if config.expected_protected_v6_count < 0:
        raise ValueError("expected protected V6 count must be non-negative")
    if config.max_documents_per_pool_run is not None and (
        config.max_documents_per_pool_run <= 0
    ):
        raise ValueError("max_documents_per_pool_run must be positive")
    inputs = (
        config.policy_path,
        config.composition_policy_path,
        config.composition_report_path,
        config.v7_manifest_path,
        config.replay_text_path,
        config.replay_report_path,
    )
    for path in inputs:
        if not path.resolve().is_relative_to(config.repo_root.resolve()):
            raise ValueError(f"encoded-data input is outside repository: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
    if config.output_dir.resolve() == config.reproduction_output_dir.resolve():
        raise ValueError("primary and reproduction output directories must differ")


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _portable(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    root = repo_root.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path is outside repository: {path}")
    return PurePosixPath(resolved.relative_to(root)).as_posix()


def _local_artifact_path(path: Path) -> str:
    try:
        return PurePosixPath(path.resolve().relative_to(Path.cwd().resolve())).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _report(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)
