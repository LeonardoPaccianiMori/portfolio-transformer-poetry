"""Build deterministic local stage and evaluation indexes for Minerva V7."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SAMPLING_VERSION = "minerva_7b_v7_stage_windows_v1"
ENCODED_DATA_VERSION = "minerva_7b_v7_encoded_data_v1"
Progress = Callable[[str], None]


@dataclass(frozen=True)
class MinervaV7WindowConfig:
    """Pin the encoded inputs, local outputs, and public aggregate reports."""

    repo_root: Path
    policy_path: Path
    encoded_report_path: Path
    primary_encoded_dir: Path
    reproduction_encoded_dir: Path
    primary_output_dir: Path
    reproduction_output_dir: Path
    json_report_path: Path
    markdown_report_path: Path
    max_index_files_per_run: int | None = None


@dataclass(frozen=True)
class DocumentSpan:
    """One encoded document's global token interval and grouping metadata."""

    document_index: int
    unit_id: str
    start: int
    end: int
    author_key: str
    work_key: str
    epoch_key: str


@dataclass(frozen=True)
class ShardSpan:
    """One physical shard's interval in a pool's logical token stream."""

    shard_index: int
    start: int
    end: int
    token_count: int
    sha256: str
    path: Path


@dataclass(frozen=True)
class PoolIndex:
    """Verified metadata needed to reference windows without loading token IDs."""

    pool_id: str
    corpus_role: str
    split: str
    tokens: int
    documents: tuple[DocumentSpan, ...]
    shards: tuple[ShardSpan, ...]
    content_identity_sha256: str


@dataclass(frozen=True)
class TokenContribution:
    """Exact target-token exposure assigned to one encoded document."""

    document_index: int
    unit_id: str
    tokens: int
    author_key: str
    work_key: str
    epoch_key: str


@dataclass(frozen=True)
class WindowCandidate:
    """One consecutive source span plus its exact loss-token accounting."""

    pool_id: str
    pool_window_index: int
    source_start: int
    source_slices: tuple[tuple[int, int, int], ...]
    contributions: tuple[TokenContribution, ...]


@dataclass(frozen=True)
class SampledWindow:
    """A selected candidate and the deterministic cycle that exposed it."""

    candidate: WindowCandidate
    selection_cycle: int
    component_window_index: int


def load_sampling_policy(path: Path, encoded_report_path: Path) -> dict[str, Any]:
    """Load the frozen 8D policy and verify its encoded-data lineage."""

    policy = _read_json(path)
    if policy.get("sampling_version") != SAMPLING_VERSION:
        raise ValueError("unexpected Minerva V7 sampling policy version")
    if policy.get("encoded_data_version") != ENCODED_DATA_VERSION:
        raise ValueError("sampling policy has the wrong encoded-data version")
    if policy.get("encoded_report_sha256") != _sha256(encoded_report_path):
        raise ValueError("encoded report does not match the sampling policy")
    windowing = _mapping(policy, "windowing")
    if (
        int(windowing["context_length"]) != 2048
        or int(windowing["source_span_tokens"]) != 2049
        or int(windowing["target_tokens_per_window"]) != 2048
        or int(windowing["target_stride_tokens"]) != 2048
    ):
        raise ValueError("sampling policy must preserve 2,048-token next-token windows")
    expected = _mapping(policy, "expected")
    if int(expected["training_windows"]) * 2048 != int(
        expected["training_target_tokens"]
    ):
        raise ValueError("sampling policy training budget is not whole-window aligned")
    return policy


def prepare_and_verify_minerva_v7_windows(
    config: MinervaV7WindowConfig,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Build primary and reproduction indexes and publish aggregate evidence."""

    _validate_config(config)
    policy = load_sampling_policy(config.policy_path, config.encoded_report_path)
    encoded_report = _read_json(config.encoded_report_path)
    expected_identity = str(policy["encoded_content_identity_sha256"])
    if encoded_report.get("status") != "active_verified":
        raise ValueError("encoded Minerva V7 data is not active and verified")
    if encoded_report.get("encoded_content_identity_sha256") != expected_identity:
        raise ValueError("encoded content identity changed after sampling approval")

    _report(progress, "verifying primary encoded shards and document indexes")
    primary_pools = load_verified_encoded_build(
        config.primary_encoded_dir, encoded_report, verify_binary_hashes=True
    )
    _report(progress, "verifying independent reproduction encoded build")
    reproduction_pools = load_verified_encoded_build(
        config.reproduction_encoded_dir, encoded_report, verify_binary_hashes=True
    )
    if pool_index_identity(primary_pools) != pool_index_identity(reproduction_pools):
        raise ValueError("primary and reproduction pool indexes differ")

    primary = build_one_window_index(
        output_dir=config.primary_output_dir,
        pools=primary_pools,
        policy=policy,
        encoded_report=encoded_report,
        max_index_files=config.max_index_files_per_run,
        progress=progress,
    )
    if primary["status"] != "complete":
        return primary
    reproduction = build_one_window_index(
        output_dir=config.reproduction_output_dir,
        pools=reproduction_pools,
        policy=policy,
        encoded_report=encoded_report,
        max_index_files=config.max_index_files_per_run,
        progress=progress,
    )
    if reproduction["status"] != "complete":
        return {
            "sampling_version": SAMPLING_VERSION,
            "status": "incomplete_reproduction",
            "primary": primary,
            "reproduction": reproduction,
        }
    comparison = compare_window_indexes(primary, reproduction)
    if not comparison["match"]:
        raise ValueError("independent window indexes do not match")

    report = {
        "sampling_version": SAMPLING_VERSION,
        "build_date": policy["build_date"],
        "status": "active_verified",
        "encoded_data_version": ENCODED_DATA_VERSION,
        "encoded_content_identity_sha256": expected_identity,
        "provenance": {
            "sampling_policy_path": _portable(config.policy_path, config.repo_root),
            "sampling_policy_sha256": _sha256(config.policy_path),
            "encoded_report_path": _portable(
                config.encoded_report_path, config.repo_root
            ),
            "encoded_report_sha256": _sha256(config.encoded_report_path),
        },
        "format": primary["format"],
        "training": primary["training"],
        "evaluation": primary["evaluation"],
        "window_index_content_identity_sha256": primary[
            "window_index_content_identity_sha256"
        ],
        "reproduction": comparison,
        "verification": {
            "independent_window_index_count": 2,
            "independent_window_indexes_match": True,
            "exact_stage_and_component_budgets": True,
            "concentration_caps_enforced_on_target_tokens": True,
            "cross_document_contributions_exact": True,
            "cross_shard_source_slices_exact": True,
            "validation_targets_fixed_sequential_non_overlapping": True,
            "v7_validation_test_training_excluded": True,
            "broader_validation_training_excluded": True,
            "protected_v6_training_excluded": True,
            "conditioned_material_included": False,
            "gpu_work_started": False,
            "cache_deleted": False,
        },
        "publication": {
            "individual_window_indexes_public": False,
            "token_ids_public": False,
            "aggregate_evidence_and_hashes_public": True,
        },
    }
    _write_json(config.json_report_path, report)
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(
        render_window_report_markdown(report), encoding="utf-8"
    )
    _report(progress, f"published aggregate report: {config.json_report_path}")
    return report


def load_verified_encoded_build(
    encoded_dir: Path,
    encoded_report: Mapping[str, Any],
    *,
    verify_binary_hashes: bool,
) -> dict[str, PoolIndex]:
    """Verify one encoded build and load its path-independent pool indexes."""

    expected_pools = encoded_report.get("pools")
    if not isinstance(expected_pools, list):
        raise ValueError("encoded report is missing pools")
    pools: dict[str, PoolIndex] = {}
    for expected in expected_pools:
        if not isinstance(expected, Mapping):
            raise ValueError("encoded report contains an invalid pool")
        pool_id = str(expected["pool_id"])
        metadata_path = encoded_dir / f"{pool_id}.metadata.json"
        metadata = _read_json(metadata_path)
        for key in (
            "pool_id",
            "corpus_role",
            "split",
            "documents",
            "tokens",
            "content_identity_sha256",
        ):
            if metadata.get(key) != expected.get(key):
                raise ValueError(f"encoded pool metadata mismatch: {pool_id} {key}")
        index_path = encoded_dir / f"{pool_id}.documents.jsonl"
        expected_index = _mapping(metadata, "document_index")
        if _sha256(index_path) != expected_index.get("sha256"):
            raise ValueError(f"encoded document index hash mismatch: {pool_id}")
        shards = _load_shards(encoded_dir, metadata, verify_binary_hashes)
        documents = _load_documents(index_path, metadata, shards)
        pools[pool_id] = PoolIndex(
            pool_id=pool_id,
            corpus_role=str(metadata["corpus_role"]),
            split=str(metadata["split"]),
            tokens=int(metadata["tokens"]),
            documents=documents,
            shards=shards,
            content_identity_sha256=str(metadata["content_identity_sha256"]),
        )
    if len(pools) != len(expected_pools):
        raise ValueError("encoded pool IDs are not unique")
    return pools


def enumerate_pool_windows(
    pool: PoolIndex,
    *,
    source_span_tokens: int,
    target_stride_tokens: int,
) -> tuple[WindowCandidate, ...]:
    """Enumerate consecutive next-token spans and exact document contributions."""

    if source_span_tokens != target_stride_tokens + 1:
        raise ValueError("a next-token source span must be exactly stride plus one")
    candidates: list[WindowCandidate] = []
    document_cursor = 0
    for window_index, source_start in enumerate(
        range(0, pool.tokens - source_span_tokens + 1, target_stride_tokens)
    ):
        source_end = source_start + source_span_tokens
        target_start = source_start + 1
        while pool.documents[document_cursor].end <= target_start:
            document_cursor += 1
        contributions: list[TokenContribution] = []
        cursor = document_cursor
        while cursor < len(pool.documents) and pool.documents[cursor].start < source_end:
            document = pool.documents[cursor]
            overlap = min(source_end, document.end) - max(target_start, document.start)
            if overlap > 0:
                contributions.append(
                    TokenContribution(
                        document_index=document.document_index,
                        unit_id=document.unit_id,
                        tokens=overlap,
                        author_key=document.author_key,
                        work_key=document.work_key,
                        epoch_key=document.epoch_key,
                    )
                )
            cursor += 1
        if sum(row.tokens for row in contributions) != target_stride_tokens:
            raise ValueError(f"window target accounting mismatch: {pool.pool_id}")
        candidates.append(
            WindowCandidate(
                pool_id=pool.pool_id,
                pool_window_index=window_index,
                source_start=source_start,
                source_slices=_source_slices(pool.shards, source_start, source_end),
                contributions=tuple(contributions),
            )
        )
    return tuple(candidates)


def sample_component_windows(
    candidates: Sequence[WindowCandidate],
    *,
    count: int,
    target_tokens_per_window: int,
    seed: int,
    stage_id: str,
    component: str,
    ceilings: Mapping[str, float],
) -> tuple[SampledWindow, ...]:
    """Select deterministic candidates while enforcing exact exposure ceilings."""

    if not candidates or count <= 0:
        raise ValueError("component sampling requires candidates and a positive draw")
    total_tokens = count * target_tokens_per_window
    capacities = {
        field: math.floor(total_tokens * float(limit))
        for field, limit in ceilings.items()
    }
    exposures: dict[str, Counter[str]] = {field: Counter() for field in ceilings}
    selected: list[SampledWindow] = []
    cycle = 0
    while len(selected) < count:
        accepted = 0
        ordered = sorted(
            candidates,
            key=lambda row: _candidate_order_key(
                seed, stage_id, component, cycle, row
            ),
        )
        for candidate in ordered:
            additions = {
                field: _candidate_group_counts(candidate, field)
                for field in ceilings
            }
            if any(
                not group
                or exposures[field][group] + tokens > capacities[field]
                for field, groups in additions.items()
                for group, tokens in groups.items()
            ):
                continue
            for field, groups in additions.items():
                exposures[field].update(groups)
            selected.append(
                SampledWindow(
                    candidate=candidate,
                    selection_cycle=cycle,
                    component_window_index=len(selected),
                )
            )
            accepted += 1
            if len(selected) == count:
                break
        if accepted == 0:
            raise ValueError(
                f"concentration caps make component infeasible: {stage_id} {component}"
            )
        cycle += 1
    return tuple(selected)


def interleave_stage_windows(
    sampled: Mapping[str, Sequence[SampledWindow]],
) -> tuple[tuple[str, SampledWindow], ...]:
    """Evenly interleave component queues while preserving their exact quotas."""

    schedule: list[tuple[float, str, int]] = []
    for component, rows in sampled.items():
        if not rows:
            raise ValueError(f"empty sampled stage component: {component}")
        quota = len(rows)
        schedule.extend(
            ((index + 0.5) / quota, component, index) for index in range(quota)
        )
    schedule.sort(key=lambda row: (row[0], row[1], row[2]))
    return tuple((component, sampled[component][index]) for _, component, index in schedule)


def build_one_window_index(
    *,
    output_dir: Path,
    pools: Mapping[str, PoolIndex],
    policy: Mapping[str, Any],
    encoded_report: Mapping[str, Any],
    max_index_files: int | None,
    progress: Progress | None,
) -> dict[str, Any]:
    """Build one resumable local index tree from a verified encoded build."""

    completed = _load_completed_window_build(output_dir, policy)
    if completed is not None:
        _report(progress, f"verified existing window index: {output_dir}")
        return completed
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / ".checkpoint.json"
    checkpoint = _load_window_checkpoint(checkpoint_path, policy)
    completed_files = dict(checkpoint.get("files", {}))
    written_this_run = 0
    windowing = _mapping(policy, "windowing")
    source_span = int(windowing["source_span_tokens"])
    stride = int(windowing["target_stride_tokens"])
    seed = int(policy["seed"])
    candidates = {
        pool_id: enumerate_pool_windows(
            pool, source_span_tokens=source_span, target_stride_tokens=stride
        )
        for pool_id, pool in pools.items()
    }
    stage_plan = _stage_plan(encoded_report)
    component_pools = _mapping(policy, "stage_component_pools")
    training_rows: list[dict[str, Any]] = []
    stage_reports: list[dict[str, Any]] = []
    for stage in stage_plan:
        stage_id = str(stage["stage_id"])
        sampled: dict[str, tuple[SampledWindow, ...]] = {}
        component_reports = []
        for component_row in stage["components"]:
            component = str(component_row["component"])
            pool_ids = component_pools.get(component)
            if not isinstance(pool_ids, list) or not pool_ids:
                raise ValueError(f"sampling policy is missing component pools: {component}")
            component_candidates = tuple(
                candidate
                for pool_id in pool_ids
                for candidate in candidates[str(pool_id)]
            )
            ceilings = _component_ceilings(policy, component)
            draw_windows = int(component_row["draw_windows_2048"])
            selected = sample_component_windows(
                component_candidates,
                count=draw_windows,
                target_tokens_per_window=stride,
                seed=seed,
                stage_id=stage_id,
                component=component,
                ceilings=ceilings,
            )
            sampled[component] = selected
            component_reports.append(
                _component_report(component, selected, stride, ceilings)
            )
        ordered = interleave_stage_windows(sampled)
        path = output_dir / "training" / f"{stage_id}.jsonl"
        rows = [
            _training_row(stage_id, stage_index, component, sampled_row, source_span, stride)
            for stage_index, (component, sampled_row) in enumerate(ordered)
        ]
        file_report, newly_written = _write_or_verify_index_file(
            path, rows, output_dir, completed_files
        )
        written_this_run += int(newly_written)
        training_rows.extend(rows)
        stage_reports.append(
            {
                "stage_id": stage_id,
                "windows": len(rows),
                "target_tokens": len(rows) * stride,
                "components": component_reports,
                "index": file_report,
            }
        )
        _persist_checkpoint(checkpoint_path, policy, completed_files)
        _report(progress, f"indexed training stage {stage_id}: windows={len(rows):,}")
        if max_index_files is not None and written_this_run >= max_index_files:
            return _incomplete_window_report(output_dir, completed_files)

    evaluation_report: dict[str, Any] = {}
    evaluation_rows: list[dict[str, Any]] = []
    evaluation = _mapping(policy, "evaluation")
    for split_name, policy_key in (("validation", "validation_pools"), ("test", "test_pools")):
        split_reports = []
        for pool_id_value in sorted(str(value) for value in evaluation[policy_key]):
            pool_id = str(pool_id_value)
            pool = pools[pool_id]
            fixed = candidates[pool_id]
            rows = [
                _evaluation_row(split_name, index, row, source_span, stride)
                for index, row in enumerate(fixed)
            ]
            path = output_dir / split_name / f"{pool_id}.jsonl"
            file_report, newly_written = _write_or_verify_index_file(
                path, rows, output_dir, completed_files
            )
            written_this_run += int(newly_written)
            evaluation_rows.extend(rows)
            split_reports.append(
                {
                    "pool_id": pool_id,
                    "windows": len(rows),
                    "target_tokens": len(rows) * stride,
                    "dropped_tail_tokens": pool.tokens - (1 + len(rows) * stride),
                    "index": file_report,
                }
            )
            _persist_checkpoint(checkpoint_path, policy, completed_files)
            _report(progress, f"indexed {split_name} pool {pool_id}: windows={len(rows):,}")
            if max_index_files is not None and written_this_run >= max_index_files:
                return _incomplete_window_report(output_dir, completed_files)
        evaluation_report[split_name] = {
            "windows": sum(row["windows"] for row in split_reports),
            "target_tokens": sum(row["target_tokens"] for row in split_reports),
            "pools": split_reports,
        }

    training_pool_ids = {
        contribution["pool_id"]
        for row in training_rows
        for contribution in row["target_contributions"]
    }
    forbidden = set(evaluation["validation_pools"]) | set(evaluation["test_pools"])
    if training_pool_ids & forbidden:
        raise ValueError("held-out pool entered a training window index")
    expected = _mapping(policy, "expected")
    if len(training_rows) != int(expected["training_windows"]):
        raise ValueError("training window count differs from the frozen budget")
    if len(training_rows) * stride != int(expected["training_target_tokens"]):
        raise ValueError("training target-token budget differs from policy")

    identity = _index_content_identity(completed_files)
    report = {
        "sampling_version": SAMPLING_VERSION,
        "sampling_policy_sha256": _sha256_from_json(policy),
        "status": "complete",
        "format": {
            "storage": "local JSON Lines indexes; encoded int32 token IDs remain in checkpoint-8C shards",
            "source_span_tokens": source_span,
            "target_tokens_per_window": stride,
            "target_stride_tokens": stride,
            "cross_document_accounting": "exact per-document target-token contributions",
            "cross_shard_references": "one or more shard_index/token_offset/token_count slices",
            "individual_window_indexes_public": False,
        },
        "training": {
            "windows": len(training_rows),
            "target_tokens": len(training_rows) * stride,
            "stages": stage_reports,
        },
        "evaluation": evaluation_report,
        "files": [completed_files[key] for key in sorted(completed_files)],
        "window_index_content_identity_sha256": identity,
    }
    _write_json(output_dir / "manifest.json", report)
    checkpoint_path.unlink(missing_ok=True)
    return report


def compare_window_indexes(
    primary: Mapping[str, Any], reproduction: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare independently written indexes using path-independent identities."""

    primary_identity = str(primary["window_index_content_identity_sha256"])
    reproduction_identity = str(reproduction["window_index_content_identity_sha256"])
    return {
        "match": primary_identity == reproduction_identity,
        "primary_content_identity_sha256": primary_identity,
        "reproduction_content_identity_sha256": reproduction_identity,
        "training_windows_match": primary["training"]["windows"]
        == reproduction["training"]["windows"],
        "evaluation_windows_match": primary["evaluation"] == reproduction["evaluation"],
    }


def pool_index_identity(pools: Mapping[str, PoolIndex]) -> str:
    """Hash all path-independent pool boundaries and identities."""

    digest = hashlib.sha256()
    for pool_id in sorted(pools):
        pool = pools[pool_id]
        digest.update(pool_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(pool.content_identity_sha256.encode("ascii"))
        digest.update(b"\0")
        for document in pool.documents:
            digest.update(
                f"{document.document_index}:{document.unit_id}:{document.start}:"
                f"{document.end}:{document.author_key}:{document.work_key}:"
                f"{document.epoch_key}\n".encode("utf-8")
            )
    return digest.hexdigest()


def render_window_report_markdown(report: Mapping[str, Any]) -> str:
    """Render public aggregate evidence without exposing individual indexes."""

    training = _mapping(report, "training")
    lines = [
        "# Minerva 7B V7 Deterministic Stage Windows",
        "",
        "Checkpoint 8D materializes local deterministic training, validation, and test",
        "window indexes over the independently reproduced checkpoint-8C token pools.",
        "The individual indexes and token IDs remain local; this report publishes only",
        "the frozen policy, hashes, counts, and aggregate cap evidence.",
        "",
        "## Training curriculum",
        "",
        "| Stage | Windows | Target tokens |",
        "| --- | ---: | ---: |",
    ]
    for stage in training["stages"]:
        lines.append(
            f"| {stage['stage_id']} | {int(stage['windows']):,} | "
            f"{int(stage['target_tokens']):,} |"
        )
    lines.extend(
        [
            f"| **Total** | **{int(training['windows']):,}** | "
            f"**{int(training['target_tokens']):,}** |",
            "",
            "Every source span contains 2,049 tokens and advances by 2,048 target",
            "tokens. Documents are concatenated only within one role/split pool using",
            "their encoded EOS separators. Cross-document windows retain exact",
            "per-document target contributions, and cross-shard windows retain every",
            "physical shard slice.",
            "",
            "## Concentration evidence",
            "",
            "| Stage / component | Group | Maximum | Ceiling | Pass |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for stage in training["stages"]:
        for component in stage["components"]:
            for concentration in component["concentration"]:
                lines.append(
                    f"| {stage['stage_id']} / {component['component']} | "
                    f"{concentration['field']} | "
                    f"{float(concentration['maximum_share']):.4%} | "
                    f"{float(concentration['ceiling']):.2%} | "
                    f"{'yes' if concentration['passes'] else 'no'} |"
                )
    evaluation = _mapping(report, "evaluation")
    lines.extend(
        [
            "",
            "## Fixed held-out windows",
            "",
            "| Split | Windows | Target tokens |",
            "| --- | ---: | ---: |",
            f"| validation | {int(evaluation['validation']['windows']):,} | "
            f"{int(evaluation['validation']['target_tokens']):,} |",
            f"| test | {int(evaluation['test']['windows']):,} | "
            f"{int(evaluation['test']['target_tokens']):,} |",
            "",
            "Held-out targets are sequential and non-overlapping. Their final incomplete",
            "tails are dropped without padding. V7 validation/test and broader validation",
            "pools never enter a training index.",
            "",
            "## Reproduction and boundaries",
            "",
            f"- Window-index content identity: `{report['window_index_content_identity_sha256']}`.",
            f"- Independent reproduction matches: `{str(report['reproduction']['match']).lower()}`.",
            "- Conditioned and protected V6 material included: `false`.",
            "- GPU work started: `false`.",
            "- Local caches deleted: `false`.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_shards(
    encoded_dir: Path,
    metadata: Mapping[str, Any],
    verify_binary_hashes: bool,
) -> tuple[ShardSpan, ...]:
    rows = metadata.get("shards")
    if not isinstance(rows, list) or not rows:
        raise ValueError("encoded pool has no shards")
    shards = []
    expected_start = 0
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("encoded shard row is invalid")
        shard_index = int(row["shard_index"])
        start = int(row["global_token_start"])
        end = int(row["global_token_end"])
        count = int(row["token_count"])
        if shard_index != position or start != expected_start or end - start != count:
            raise ValueError("encoded shard coverage is not contiguous")
        path = encoded_dir / f"{metadata['pool_id']}-{shard_index:05d}.int32.bin"
        if path.stat().st_size != count * 4:
            raise ValueError(f"encoded shard size mismatch: {path}")
        sha = str(row["sha256"])
        if verify_binary_hashes and _sha256(path) != sha:
            raise ValueError(f"encoded shard hash mismatch: {path}")
        shards.append(ShardSpan(shard_index, start, end, count, sha, path))
        expected_start = end
    if expected_start != int(metadata["tokens"]):
        raise ValueError("encoded shard total does not match pool tokens")
    return tuple(shards)


def _load_documents(
    path: Path, metadata: Mapping[str, Any], shards: Sequence[ShardSpan]
) -> tuple[DocumentSpan, ...]:
    documents = []
    start = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if int(row["document_index"]) != line_number - 1:
                raise ValueError("encoded document indexes are not consecutive")
            if row.get("pool_id") != metadata.get("pool_id"):
                raise ValueError("encoded document belongs to the wrong pool")
            tokens = int(row["tokens"])
            end = start + tokens
            if _coordinate(shards, start) != _position_tuple(row["token_start"]):
                raise ValueError("encoded document start coordinate mismatch")
            if _coordinate(shards, end, allow_end=True) != _position_tuple(
                row["token_end"]
            ):
                raise ValueError("encoded document end coordinate mismatch")
            documents.append(
                DocumentSpan(
                    document_index=line_number - 1,
                    unit_id=str(row["unit_id"]),
                    start=start,
                    end=end,
                    author_key=str(row["author_key"]),
                    work_key=str(row["work_key"]),
                    epoch_key=str(row["epoch_key"]),
                )
            )
            start = end
    if len(documents) != int(metadata["documents"]) or start != int(metadata["tokens"]):
        raise ValueError("encoded document accounting does not match pool metadata")
    return tuple(documents)


def _source_slices(
    shards: Sequence[ShardSpan], start: int, end: int
) -> tuple[tuple[int, int, int], ...]:
    slices = []
    for shard in shards:
        if shard.end <= start:
            continue
        if shard.start >= end:
            break
        overlap_start = max(start, shard.start)
        overlap_end = min(end, shard.end)
        slices.append(
            (shard.shard_index, overlap_start - shard.start, overlap_end - overlap_start)
        )
    if sum(row[2] for row in slices) != end - start:
        raise ValueError("source window crosses missing shard coverage")
    return tuple(slices)


def _candidate_order_key(
    seed: int,
    stage_id: str,
    component: str,
    cycle: int,
    candidate: WindowCandidate,
) -> tuple[str, str, int]:
    value = (
        f"{seed}\0{stage_id}\0{component}\0{cycle}\0{candidate.pool_id}\0"
        f"{candidate.pool_window_index}"
    )
    return (
        hashlib.sha256(value.encode("utf-8")).hexdigest(),
        candidate.pool_id,
        candidate.pool_window_index,
    )


def _candidate_group_counts(
    candidate: WindowCandidate, field: str
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in candidate.contributions:
        group = str(getattr(row, field))
        if not group:
            raise ValueError(f"cap-sensitive contribution has blank {field}")
        counts[group] += row.tokens
    return counts


def _component_ceilings(
    policy: Mapping[str, Any], component: str
) -> dict[str, float]:
    ceilings = _mapping(policy, "concentration_ceilings")
    if component == "standard_sonnets_v7_train":
        row = _mapping(ceilings, "standard_sonnets_v7_train")
    elif component == "modern_preservation_replay":
        return {}
    else:
        row = _mapping(ceilings, "broader")
    return {str(field): float(value) for field, value in row.items()}


def _component_report(
    component: str,
    sampled: Sequence[SampledWindow],
    target_tokens_per_window: int,
    ceilings: Mapping[str, float],
) -> dict[str, Any]:
    pool_counts = Counter(row.candidate.pool_id for row in sampled)
    total = len(sampled) * target_tokens_per_window
    concentration = []
    for field, ceiling in ceilings.items():
        exposure: Counter[str] = Counter()
        for sampled_row in sampled:
            exposure.update(_candidate_group_counts(sampled_row.candidate, field))
        maximum = max(exposure.values(), default=0)
        concentration.append(
            {
                "field": field,
                "distinct_groups": len(exposure),
                "maximum_tokens": maximum,
                "maximum_share": maximum / total,
                "ceiling": ceiling,
                "integer_capacity_tokens": math.floor(total * ceiling),
                "passes": maximum <= math.floor(total * ceiling),
            }
        )
    return {
        "component": component,
        "windows": len(sampled),
        "target_tokens": total,
        "selection_cycles": max(row.selection_cycle for row in sampled) + 1,
        "repeated_draws": len(sampled)
        - len(
            {
                (row.candidate.pool_id, row.candidate.pool_window_index)
                for row in sampled
            }
        ),
        "pool_windows": dict(sorted(pool_counts.items())),
        "concentration": concentration,
    }


def _training_row(
    stage_id: str,
    stage_window_index: int,
    component: str,
    sampled: SampledWindow,
    source_span: int,
    target_tokens: int,
) -> dict[str, Any]:
    row = _base_window_row(sampled.candidate, source_span, target_tokens)
    row.update(
        {
            "index_kind": "training",
            "stage_id": stage_id,
            "stage_window_index": stage_window_index,
            "component": component,
            "component_window_index": sampled.component_window_index,
            "selection_cycle": sampled.selection_cycle,
        }
    )
    return row


def _evaluation_row(
    split: str,
    split_window_index: int,
    candidate: WindowCandidate,
    source_span: int,
    target_tokens: int,
) -> dict[str, Any]:
    row = _base_window_row(candidate, source_span, target_tokens)
    row.update(
        {
            "index_kind": split,
            "split_window_index": split_window_index,
            "selection_cycle": 0,
        }
    )
    return row


def _base_window_row(
    candidate: WindowCandidate, source_span: int, target_tokens: int
) -> dict[str, Any]:
    return {
        "sampling_version": SAMPLING_VERSION,
        "pool_id": candidate.pool_id,
        "pool_window_index": candidate.pool_window_index,
        "global_source_start": candidate.source_start,
        "source_span_tokens": source_span,
        "target_tokens": target_tokens,
        "source_slices": [
            {
                "shard_index": shard_index,
                "token_offset": token_offset,
                "token_count": token_count,
            }
            for shard_index, token_offset, token_count in candidate.source_slices
        ],
        "target_contributions": [
            {
                "pool_id": candidate.pool_id,
                "document_index": row.document_index,
                "unit_id": row.unit_id,
                "tokens": row.tokens,
                "author_key": row.author_key,
                "work_key": row.work_key,
                "epoch_key": row.epoch_key,
            }
            for row in candidate.contributions
        ],
    }


def _write_or_verify_index_file(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    completed_files: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    relative = path.relative_to(output_dir).as_posix()
    existing = completed_files.get(relative)
    if isinstance(existing, Mapping) and path.exists():
        if int(existing.get("rows", -1)) == len(rows) and existing.get("sha256") == _sha256(path):
            return dict(existing), False
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    part.replace(path)
    report = {
        "path": relative,
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "public": False,
    }
    completed_files[relative] = report
    return report, True


def _load_completed_window_build(
    output_dir: Path, policy: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = output_dir / "manifest.json"
    if not path.exists():
        return None
    report = _read_json(path)
    if report.get("sampling_version") != policy.get("sampling_version"):
        raise ValueError("existing window manifest uses another policy version")
    if report.get("sampling_policy_sha256") != _sha256_from_json(policy):
        return None
    for row in report.get("files", []):
        file_path = output_dir / str(row["path"])
        if not file_path.exists() or _sha256(file_path) != row.get("sha256"):
            raise ValueError("completed window index file failed verification")
    if _index_content_identity({row["path"]: row for row in report["files"]}) != report.get(
        "window_index_content_identity_sha256"
    ):
        raise ValueError("completed window index identity mismatch")
    return report


def _load_window_checkpoint(path: Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {"sampling_version": SAMPLING_VERSION, "files": {}}
    checkpoint = _read_json(path)
    if checkpoint.get("sampling_policy_sha256") != _sha256_from_json(policy):
        raise ValueError("partial window checkpoint belongs to another policy")
    return checkpoint


def _persist_checkpoint(
    path: Path, policy: Mapping[str, Any], files: Mapping[str, Any]
) -> None:
    _write_json(
        path,
        {
            "sampling_version": SAMPLING_VERSION,
            "sampling_policy_sha256": _sha256_from_json(policy),
            "files": dict(files),
        },
    )


def _incomplete_window_report(
    output_dir: Path, completed_files: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "sampling_version": SAMPLING_VERSION,
        "status": "incomplete",
        "output_dir": str(output_dir),
        "completed_index_files": len(completed_files),
    }


def _index_content_identity(files: Mapping[str, Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(files):
        row = files[relative]
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["rows"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _stage_plan(encoded_report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    stage_plan = _mapping(encoded_report, "stage_plan")
    stages = stage_plan.get("stages")
    if not isinstance(stages, list) or len(stages) != 3:
        raise ValueError("encoded report has an invalid stage plan")
    return stages


def _coordinate(
    shards: Sequence[ShardSpan], position: int, *, allow_end: bool = False
) -> tuple[int, int]:
    for shard in shards:
        if shard.start <= position < shard.end:
            return shard.shard_index, position - shard.start
        if allow_end and position == shard.end:
            return shard.shard_index, shard.token_count
    raise ValueError("global token position is outside shard coverage")


def _position_tuple(value: Any) -> tuple[int, int]:
    if not isinstance(value, Mapping):
        raise ValueError("encoded token coordinate must be an object")
    return int(value["shard_index"]), int(value["token_offset"])


def _validate_config(config: MinervaV7WindowConfig) -> None:
    if config.max_index_files_per_run is not None and config.max_index_files_per_run <= 0:
        raise ValueError("max_index_files_per_run must be positive")
    if config.primary_encoded_dir.resolve() == config.reproduction_encoded_dir.resolve():
        raise ValueError("independent encoded builds must use different directories")
    if config.primary_output_dir.resolve() == config.reproduction_output_dir.resolve():
        raise ValueError("independent window indexes must use different directories")


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"expected object at {key}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    part.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_from_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()


def _portable(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _report(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)
