"""Count the frozen canonical corpus with Minerva and gate staged mixtures.

Checkpoint 8B measures logical documents without flattening or encoding them.
The resulting policy freezes sampling shares and concentration ceilings while
leaving every canonical corpus role inactive until a later encoding checkpoint.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sonnet_corpus.canonical_corpus_reader import CanonicalCorpusReader
from sonnet_corpus.sonnet_v7_split import canonicalize_author_label
from sonnet_training.minerva_7b_qlora import (
    MINERVA_7B_INSTRUCT_MODEL_ID,
    MINERVA_7B_INSTRUCT_REVISION,
)


COMPOSITION_VERSION = "minerva_7b_v7_composition_gate_v1"
TOKEN_COUNT_VERSION = "minerva_7b_v7_token_counts_v1"
Progress = Callable[[int, int, str], None]


@dataclass(frozen=True)
class MinervaV7CompositionConfig:
    """Locate frozen checkpoint inputs and public aggregate outputs."""

    repo_root: Path
    policy_path: Path
    canonical_corpus_dir: Path
    v7_manifest_path: Path
    replay_text_path: Path
    replay_report_path: Path
    json_report_path: Path
    markdown_report_path: Path
    tokenizer_cache_dir: Path
    expected_protected_v6_count: int = 387
    progress_interval: int = 100


@dataclass
class _Aggregate:
    documents: int = 0
    characters: int = 0
    text_tokens: int = 0
    eos_tokens: int = 0

    @property
    def training_tokens(self) -> int:
        return self.text_tokens + self.eos_tokens

    def add(self, *, characters: int, text_tokens: int, eos_tokens: int = 1) -> None:
        self.documents += 1
        self.characters += characters
        self.text_tokens += text_tokens
        self.eos_tokens += eos_tokens

    def as_dict(self) -> dict[str, int | float]:
        return {
            "documents": self.documents,
            "characters": self.characters,
            "text_tokens": self.text_tokens,
            "eos_tokens": self.eos_tokens,
            "training_tokens": self.training_tokens,
            "characters_per_text_token": (
                self.characters / self.text_tokens if self.text_tokens else 0.0
            ),
        }


def load_composition_policy(path: Path) -> dict[str, Any]:
    """Load and validate the exact checkpoint-8B mixture policy."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("composition_version") != COMPOSITION_VERSION:
        raise ValueError("unexpected Minerva V7 composition policy version")
    tokenizer = _mapping(payload, "tokenizer")
    if tokenizer.get("model_id") != MINERVA_7B_INSTRUCT_MODEL_ID:
        raise ValueError("composition policy is not pinned to Minerva 7B Instruct")
    if tokenizer.get("revision") != MINERVA_7B_INSTRUCT_REVISION:
        raise ValueError("composition policy has the wrong Minerva revision")
    accounting = _mapping(payload, "token_accounting")
    if accounting != {
        "add_special_tokens": False,
        "append_eos_per_logical_unit": True,
        "protected_sonnets_audit_only": True,
    }:
        raise ValueError("composition token-accounting policy drifted")

    stages = _mapping(payload, "stages")
    required_stages = {"stage_1_historical_general", "stage_2_non_sonnet_poetry", "stage_3_sonnets"}
    if set(stages) != required_stages:
        raise ValueError("composition policy must define exactly three stages")
    for stage_id, stage in stages.items():
        components = _mapping(stage, "components")
        shares = [float(value) for value in components.values()]
        if not components or any(value <= 0.0 for value in shares):
            raise ValueError(f"{stage_id} mixture shares must be positive")
        if not math.isclose(sum(shares), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{stage_id} mixture shares must sum to one")

    ceilings = _mapping(payload, "concentration_ceilings")
    for key in ("broader_work", "broader_author", "sonnet_author", "sonnet_epoch"):
        value = float(ceilings[key])
        if not 0.0 < value <= 1.0:
            raise ValueError(f"invalid concentration ceiling: {key}")
    stage_1 = _mapping(stages, "stage_1_historical_general")
    stage_1_components = _mapping(stage_1, "components")
    if float(stage_1_components["nineteenth_century_bridge"]) > float(
        ceilings["nineteenth_century_bridge"]
    ):
        raise ValueError("stage-1 bridge share exceeds its approved ceiling")
    return payload


def build_minerva_v7_composition(
    config: MinervaV7CompositionConfig,
    *,
    tokenizer: Any | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Count canonical logical units and build the deterministic composition gate."""

    _validate_config(config)
    policy = load_composition_policy(config.policy_path)
    reader = CanonicalCorpusReader(
        config.repo_root,
        config.canonical_corpus_dir,
        expected_protected_v6_count=config.expected_protected_v6_count,
    )
    v7_rows = _load_v7_rows(config.v7_manifest_path)
    stored_sonnets = {
        unit.unit_id for unit in reader.units if unit.unit_kind == "standard_sonnet"
    }
    included_sonnets = {
        unit_id for unit_id, row in v7_rows.items() if row["include_in_v7"] == "true"
    }
    if stored_sonnets != included_sonnets:
        raise ValueError("V7 included identities do not match stored canonical sonnets")

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            MINERVA_7B_INSTRUCT_MODEL_ID,
            revision=MINERVA_7B_INSTRUCT_REVISION,
            cache_dir=config.tokenizer_cache_dir,
            local_files_only=True,
        )
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token_id, int) or eos_token_id < 0:
        raise ValueError("Minerva tokenizer must define a non-negative EOS token ID")
    tokenizer_fingerprint = tokenizer_sha256(tokenizer)

    aggregates: dict[str, dict[str, _Aggregate]] = {
        "roles": {},
        "v7_splits": {},
        "source_groups": {},
        "broader_works": {},
        "broader_authors": {},
        "sonnet_train_works": {},
        "sonnet_train_authors": {},
        "sonnet_train_epochs": {},
    }
    identity = hashlib.sha256()
    units = reader.units
    for index, unit in enumerate(units, start=1):
        text = reader.read_text(unit)
        text_tokens = _count_text_tokens(tokenizer, text)
        _add(aggregates["roles"], unit.final_role, len(text), text_tokens)
        _add(aggregates["source_groups"], unit.source_group, len(text), text_tokens)

        split = "broader_training"
        if unit.unit_kind == "broader":
            work_key = f"{unit.source_group}:{unit.source_id}"
            author_key = _broader_author_key(unit.author, work_key)
            _add(aggregates["broader_works"], work_key, len(text), text_tokens)
            _add(aggregates["broader_authors"], author_key, len(text), text_tokens)
        else:
            v7 = v7_rows[unit.unit_id]
            split = v7["v7_split"]
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"stored sonnet has invalid V7 split: {unit.unit_id}")
            expected_training = "true" if split == "train" else "false"
            if v7["v7_training_eligible"] != expected_training:
                raise ValueError(
                    f"V7 training eligibility disagrees with split: {unit.unit_id}"
                )
            _add(aggregates["v7_splits"], split, len(text), text_tokens)
            if split == "train":
                work_key = v7["work_group_id"]
                author_key = v7["author_group_id"] or f"generic:{work_key}"
                epoch_key = _harmonized_epoch(v7["epoch_bucket"], policy)
                _add(aggregates["sonnet_train_works"], work_key, len(text), text_tokens)
                _add(aggregates["sonnet_train_authors"], author_key, len(text), text_tokens)
                _add(aggregates["sonnet_train_epochs"], epoch_key, len(text), text_tokens)

        identity.update(unit.unit_id.encode("utf-8"))
        identity.update(b"\0")
        identity.update(split.encode("utf-8"))
        identity.update(b"\0")
        identity.update(str(text_tokens).encode("ascii"))
        identity.update(b"\0")
        identity.update(unit.logical_sha256.encode("ascii"))
        identity.update(b"\n")
        if progress is not None and (
            index % config.progress_interval == 0 or index == len(units)
        ):
            progress(index, len(units) + 1, unit.unit_id)

    replay_source_report = json.loads(
        config.replay_report_path.read_text(encoding="utf-8")
    )
    replay_sha256 = _sha256(config.replay_text_path)
    if replay_source_report.get("sample_version") != "paisa_even_byte_windows_v1":
        raise ValueError("unexpected modern replay sample version")
    if replay_source_report.get("output_sha256") != replay_sha256:
        raise ValueError("modern replay sample SHA-256 does not match its report")
    if replay_source_report.get(
        "output_size_bytes"
    ) != config.replay_text_path.stat().st_size:
        raise ValueError("modern replay sample byte size does not match its report")
    replay_text = config.replay_text_path.read_text(encoding="utf-8")
    replay_tokens = _count_text_tokens(tokenizer, replay_text)
    replay = _Aggregate()
    replay.add(characters=len(replay_text), text_tokens=replay_tokens)
    if progress is not None:
        progress(len(units) + 1, len(units) + 1, "modern_preservation_replay")

    aggregate_report = {
        name: _aggregate_rows(values)
        for name, values in aggregates.items()
    }
    gate = build_staged_composition_gate(
        aggregates=aggregate_report,
        replay=replay.as_dict(),
        policy=policy,
    )
    report = {
        "token_count_version": TOKEN_COUNT_VERSION,
        "composition_version": COMPOSITION_VERSION,
        "build_date": policy["build_date"],
        "status": "pass" if gate["pass"] else "fail",
        "activation_status": "inactive_pending_encoded_mixtures",
        "tokenizer": {
            "model_id": MINERVA_7B_INSTRUCT_MODEL_ID,
            "revision": MINERVA_7B_INSTRUCT_REVISION,
            "class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer),
            "eos_token_id": eos_token_id,
            "serialized_sha256": tokenizer_fingerprint,
        },
        "token_accounting": policy["token_accounting"],
        "provenance": {
            "policy_path": _portable(config.policy_path, config.repo_root),
            "policy_sha256": _sha256(config.policy_path),
            "canonical_build_report_path": _portable(
                reader.build_report_path, config.repo_root
            ),
            "canonical_build_report_sha256": _sha256(reader.build_report_path),
            "v7_manifest_path": _portable(config.v7_manifest_path, config.repo_root),
            "v7_manifest_sha256": _sha256(config.v7_manifest_path),
            "v7_identity_sha256": policy["v7_identity_sha256"],
            "replay_sample_id": replay_source_report.get("sample_version"),
            "replay_sample_sha256": replay_sha256,
            "replay_license_lineage": replay_source_report.get("license_lineage"),
            "replay_public": False,
        },
        "totals": _overall_totals(aggregates["roles"]),
        "logical_unit_token_identity_sha256": identity.hexdigest(),
        "aggregates": aggregate_report,
        "modern_preservation_replay": replay.as_dict(),
        "composition_gate": gate,
        "verification": {
            "all_stored_logical_units_counted": True,
            "all_v7_included_identities_counted": True,
            "v7_validation_test_training_excluded": True,
            "protected_v6_training_excluded": True,
            "conditioned_material_included": False,
            "token_ids_persisted": False,
            "encoded_training_shards_created": False,
            "stage_role_weights_frozen": True,
            "corpus_roles_activated": False,
            "cache_deleted": False,
            "gpu_work_started": False,
        },
    }
    if report["status"] != "pass":
        raise ValueError("Minerva V7 composition gate failed")
    return report


def build_staged_composition_gate(
    *,
    aggregates: Mapping[str, list[dict[str, Any]]],
    replay: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate role availability and the feasibility of approved sampling caps."""

    roles = _rows_by_key(aggregates["roles"])
    v7_splits = _rows_by_key(aggregates["v7_splits"])
    required_roles = {
        "historical_general",
        "historical_non_sonnet_poetry",
        "nineteenth_century_bridge",
        "standard_sonnets",
    }
    missing_roles = sorted(required_roles - set(roles))
    stages = _mapping(policy, "stages")
    ceilings = _mapping(policy, "concentration_ceilings")

    broader_work = _concentration(
        aggregates["broader_works"], float(ceilings["broader_work"])
    )
    broader_author = _concentration(
        aggregates["broader_authors"], float(ceilings["broader_author"])
    )
    sonnet_author = _concentration(
        aggregates["sonnet_train_authors"], float(ceilings["sonnet_author"])
    )
    sonnet_epoch = _concentration(
        aggregates["sonnet_train_epochs"], float(ceilings["sonnet_epoch"])
    )
    concentrations = {
        "broader_work": broader_work,
        "broader_author": broader_author,
        "sonnet_author": sonnet_author,
        "sonnet_epoch": sonnet_epoch,
    }
    infeasible = sorted(
        name for name, result in concentrations.items() if not result["cap_feasible"]
    )
    component_tokens = {role: 0 for role in required_roles}
    component_tokens.update(
        {role: int(row["training_tokens"]) for role, row in roles.items()}
    )
    component_tokens["modern_preservation_replay"] = int(replay["training_tokens"])
    component_tokens["standard_sonnets_v7_train"] = int(
        v7_splits.get("train", {}).get("training_tokens", 0)
    )
    component_tokens["stage_1_historical_replay"] = (
        component_tokens.get("historical_general", 0)
        + component_tokens.get("nineteenth_century_bridge", 0)
    )
    component_tokens["stage_2_historical_replay"] = (
        component_tokens["stage_1_historical_replay"]
        + component_tokens.get("historical_non_sonnet_poetry", 0)
    )
    unavailable = sorted(name for name, count in component_tokens.items() if count <= 0)

    stage_rows = []
    for stage_id, stage in stages.items():
        components = _mapping(stage, "components")
        unknown_components = sorted(set(components) - set(component_tokens))
        if unknown_components:
            raise ValueError(
                f"{stage_id} references unknown components: {unknown_components}"
            )
        stage_rows.append(
            {
                "stage_id": stage_id,
                "purpose": stage["purpose"],
                "components": [
                    {"component": name, "target_share": float(share)}
                    for name, share in components.items()
                ],
                "shares_sum": sum(float(value) for value in components.values()),
            }
        )

    passed = not missing_roles and not unavailable and not infeasible
    return {
        "pass": passed,
        "stage_mixtures": stage_rows,
        "component_available_training_tokens": component_tokens,
        "concentration": concentrations,
        "missing_roles": missing_roles,
        "unavailable_components": unavailable,
        "infeasible_caps": infeasible,
        "sampling_policy": policy["sampling_policy"],
        "bridge_ceiling": float(ceilings["nineteenth_century_bridge"]),
        "modern_replay_semantics": (
            "PAISA modern-language replay limits language drift; instruction-following "
            "preservation is measured by separate fixed evaluation gates, not claimed "
            "as instruction-training data."
        ),
    }


def write_composition_reports(
    report: Mapping[str, Any], json_path: Path, markdown_path: Path
) -> None:
    """Write deterministic public JSON and Markdown aggregate reports."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_composition_markdown(report), encoding="utf-8")


def render_composition_markdown(report: Mapping[str, Any]) -> str:
    """Render the compact human review of the token-count and mixture gate."""

    totals = _mapping(report, "totals")
    replay = _mapping(report, "modern_preservation_replay")
    gate = _mapping(report, "composition_gate")
    roles = _rows_by_key(_mapping(report, "aggregates")["roles"])
    lines = [
        "# Minerva 7B V7 Token Counts And Composition Gate",
        "",
        f"Status: **{str(report['status']).upper()}**.",
        "",
        "The pinned Minerva tokenizer counted each verified logical document without",
        "adding model wrappers, then accounted for exactly one EOS boundary per unit.",
        "No token IDs or encoded training shards were persisted.",
        "",
        "## Scale",
        "",
        "| Measurement | Value |",
        "| --- | ---: |",
        f"| Logical units | {int(totals['documents']):,} |",
        f"| Logical characters | {int(totals['characters']):,} |",
        f"| Text tokens | {int(totals['text_tokens']):,} |",
        f"| EOS boundaries | {int(totals['eos_tokens']):,} |",
        f"| Training-accounting tokens | {int(totals['training_tokens']):,} |",
        f"| Modern replay tokens | {int(replay['training_tokens']):,} |",
        "",
        "## Frozen Logical Roles",
        "",
        "| Role | Documents | Characters | Training tokens |",
        "| --- | ---: | ---: | ---: |",
    ]
    for role in sorted(roles):
        row = roles[role]
        lines.append(
            f"| {role} | {int(row['documents']):,} | "
            f"{int(row['characters']):,} | {int(row['training_tokens']):,} |"
        )
    lines.extend([
        "",
        "## Approved Stage Mixtures",
        "",
        "| Stage | Component | Target share |",
        "| --- | --- | ---: |",
    ])
    for stage in gate["stage_mixtures"]:
        for component in stage["components"]:
            lines.append(
                f"| {stage['stage_id']} | {component['component']} | "
                f"{float(component['target_share']):.0%} |"
            )
    lines.extend([
        "",
        "## Concentration Controls",
        "",
        "| Dimension | Raw largest share | Ceiling | Reweighting required | Feasible |",
        "| --- | ---: | ---: | --- | --- |",
    ])
    for name, result in gate["concentration"].items():
        lines.append(
            f"| {name} | {float(result['largest_raw_share']):.2%} | "
            f"{float(result['ceiling']):.2%} | {result['reweighting_required']} | "
            f"{result['cap_feasible']} |"
        )
    lines.extend([
        "",
        "The 5% replay component is the deterministic PAISA modern-language sample.",
        "Instruction-following preservation remains a separate fixed evaluation gate;",
        "this report does not mislabel PAISA as instruction-tuning data.",
        "",
        "## Safety Boundary",
        "",
        "V7 validation/test and protected V6 sonnets are counted only for audit and",
        "remain unavailable to training. Conditioned material remains absent. Corpus",
        "roles stay inactive; no encoded mixture, GPU job, or cache deletion occurs.",
        "",
    ])
    return "\n".join(lines)


def tokenizer_sha256(tokenizer: Any) -> str:
    """Fingerprint the effective fast-tokenizer serialization."""

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and callable(getattr(backend, "to_str", None)):
        serialized = backend.to_str().encode("utf-8")
    else:
        serialized = json.dumps(
            {
                "class": tokenizer.__class__.__name__,
                "vocab_size": len(tokenizer),
                "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            },
            sort_keys=True,
        ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _count_text_tokens(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    token_ids = encoded["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("tokenizer returned multiple sequences for one document")
        token_ids = token_ids[0]
    if not isinstance(token_ids, list):
        token_ids = list(token_ids)
    if not token_ids:
        raise ValueError("canonical logical unit tokenized to zero tokens")
    return len(token_ids)


def _load_v7_rows(path: Path) -> dict[str, dict[str, str]]:
    required = {
        "unit_id", "include_in_v7", "v7_split", "work_group_id",
        "author_group_id", "epoch_bucket", "v7_training_eligible",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("V7 manifest is missing composition fields")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            unit_id = row["unit_id"]
            if unit_id in rows:
                raise ValueError(f"duplicate V7 identity: {unit_id}")
            rows[unit_id] = row
    return rows


def _harmonized_epoch(raw: str, policy: Mapping[str, Any]) -> str:
    mapping = _mapping(policy, "epoch_harmonization")
    if raw not in mapping:
        raise ValueError(f"unmapped V7 epoch bucket: {raw}")
    return str(mapping[raw])


def _broader_author_key(author: str, work_key: str) -> str:
    canonical = canonicalize_author_label(author)
    return f"author:{canonical}" if canonical else f"generic:{work_key}"


def _add(
    aggregates: dict[str, _Aggregate],
    key: str,
    characters: int,
    text_tokens: int,
) -> None:
    if not key:
        raise ValueError("composition aggregate key must not be empty")
    aggregates.setdefault(key, _Aggregate()).add(
        characters=characters, text_tokens=text_tokens
    )


def _aggregate_rows(values: Mapping[str, _Aggregate]) -> list[dict[str, Any]]:
    rows = [{"key": key, **aggregate.as_dict()} for key, aggregate in values.items()]
    return sorted(rows, key=lambda row: (-int(row["training_tokens"]), str(row["key"])))


def _overall_totals(roles: Mapping[str, _Aggregate]) -> dict[str, Any]:
    total = _Aggregate()
    for aggregate in roles.values():
        total.documents += aggregate.documents
        total.characters += aggregate.characters
        total.text_tokens += aggregate.text_tokens
        total.eos_tokens += aggregate.eos_tokens
    return total.as_dict()


def _rows_by_key(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["key"]): row for row in rows}


def _concentration(rows: list[dict[str, Any]], ceiling: float) -> dict[str, Any]:
    total = sum(int(row["training_tokens"]) for row in rows)
    if total <= 0:
        return {
            "group_count": len(rows),
            "largest_group": None,
            "largest_raw_share": 0.0,
            "ceiling": ceiling,
            "reweighting_required": False,
            "cap_feasible": False,
        }
    largest = max(rows, key=lambda row: int(row["training_tokens"]))
    share = int(largest["training_tokens"]) / total
    return {
        "group_count": len(rows),
        "largest_group": largest["key"],
        "largest_raw_share": share,
        "ceiling": ceiling,
        "reweighting_required": share > ceiling,
        "cap_feasible": len(rows) * ceiling >= 1.0,
    }


def _validate_config(config: MinervaV7CompositionConfig) -> None:
    config.repo_root.resolve()
    if config.progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    if config.expected_protected_v6_count < 0:
        raise ValueError("expected protected V6 count must be non-negative")
    for path in (
        config.policy_path,
        config.v7_manifest_path,
        config.replay_text_path,
        config.replay_report_path,
    ):
        resolved = path.resolve()
        if not resolved.is_relative_to(config.repo_root.resolve()):
            raise ValueError(f"composition input must be inside repository: {path}")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"composition payload is missing mapping: {key}")
    return value


def _portable(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    root = repo_root.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path is outside repository: {path}")
    return PurePosixPath(resolved.relative_to(root)).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
