"""Freeze leakage-aware V7 splits over the canonical standard-sonnet corpus."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


V7_VERSION = "sonnets_expanded_v7"
V7_DATE = "2026-08-11"
V7_SEED = 1337
NEW_SPLIT_TARGETS = {"train": 0.90, "validation": 0.05, "test": 0.05}
HELDOUT_TOLERANCE = 0.005
MAX_HELDOUT_GROUP_TARGET_SHARE = 0.10

V7_ADDED_FIELDS = (
    "canonical_author_key",
    "author_group_id",
    "author_resolution_status",
    "work_group_id",
    "split_group_id",
    "v7_split",
    "v7_split_tier",
    "v7_split_decision",
    "include_in_v7",
    "v7_training_eligible",
)
AUTHOR_GROUP_FIELDS = (
    "author_label_id",
    "raw_author_label",
    "canonical_author_key",
    "author_group_id",
    "representative_author_label",
    "resolution_status",
    "sonnet_universe_count",
    "new_candidate_count",
    "v6_presence",
    "protected_v6_presence",
)

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_TRAILING_UNIT = re.compile(r"(?::sonnet_?\d+|:char\d+-\d+)$")
_NAME_PARTICLES = {"d", "da", "de", "degli", "dei", "del", "della", "di", "il", "lo"}
_NAME_ALIASES = {"iacopo": "jacopo", "iacomo": "giacomo"}
_GENERIC_AUTHOR_LABELS = {
    "",
    "Anonimo",
    "Poesie anonime",
    "Varie Rime degli Arcadi",
    "Non definito",
    "Various",
    "unresolved",
    "unknown",
}

Progress = Callable[[str], None]


@dataclass(frozen=True)
class V7SplitPolicy:
    seed: int = V7_SEED
    train_fraction: float = NEW_SPLIT_TARGETS["train"]
    validation_fraction: float = NEW_SPLIT_TARGETS["validation"]
    test_fraction: float = NEW_SPLIT_TARGETS["test"]
    heldout_tolerance: float = HELDOUT_TOLERANCE
    max_heldout_group_target_share: float = MAX_HELDOUT_GROUP_TARGET_SHARE
    stratum_weight: float = 0.35


@dataclass(frozen=True)
class V7SplitConfig:
    repo_root: Path
    canonical_sonnet_manifest_path: Path
    v6_manifest_path: Path
    author_group_path: Path
    v7_manifest_path: Path
    json_report_path: Path
    markdown_report_path: Path
    policy: V7SplitPolicy = V7SplitPolicy()


def canonicalize_author_label(label: str) -> str:
    """Return an order-insensitive key with bounded historical-name aliases."""

    folded = unicodedata.normalize("NFKD", label.casefold())
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    tokens = []
    for token in _WORD.findall(folded):
        if token in _NAME_PARTICLES:
            continue
        tokens.append(_NAME_ALIASES.get(token, token))
    return " ".join(sorted(tokens))


_GENERIC_AUTHOR_KEYS = frozenset(
    canonicalize_author_label(label) for label in _GENERIC_AUTHOR_LABELS
)


def author_group_id(label: str) -> str:
    """Return a stable resolved-author identity or empty for generic labels."""

    key = canonicalize_author_label(label)
    if key in _GENERIC_AUTHOR_KEYS:
        return ""
    return f"author:{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}"


def derive_work_group_id(source_group: str, source_id: str) -> str:
    """Collapse archive-specific sonnet/range suffixes to their source work."""

    root = _TRAILING_UNIT.sub("", source_id)
    identity = f"{source_group}:{root}"
    return f"work:{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"


def build_v7_sonnet_split(
    config: V7SplitConfig,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Build deterministic V7 identity, author-group, split, and report files."""

    _validate_policy(config.policy)
    source_fields, source_rows = _read_csv(config.canonical_sonnet_manifest_path)
    _, v6_rows = _read_csv(config.v6_manifest_path)
    _require_fields(
        source_fields,
        {
            "unit_id", "source_group", "source_id", "author", "epoch_bucket",
            "original_split", "logical_sha256", "storage_kind", "storage_path",
            "activation_status", "training_eligible",
        },
        "canonical sonnet manifest",
    )
    if len({row["unit_id"] for row in source_rows}) != len(source_rows):
        raise ValueError("canonical sonnet manifest contains duplicate unit IDs")
    _verify_v6_freeze(source_rows, v6_rows)
    if progress:
        progress(f"inventory={len(source_rows)} v6={len(v6_rows)}")

    author_rows, author_metadata = _build_author_group_rows(source_rows)
    v6_author_groups = {
        author_group_id(row["author"])
        for row in source_rows
        if row["source_group"] == "v6_sonnets" and author_group_id(row["author"])
    }
    protected_author_groups = {
        author_group_id(row["author"])
        for row in source_rows
        if row["activation_status"] == "protected_v6_validation_test"
        and author_group_id(row["author"])
    }

    new_rows = [
        row
        for row in source_rows
        if row["training_eligible"] == "true" and not row["original_split"]
    ]
    components, component_for_unit = _build_new_components(new_rows)
    assignments, assignment_metadata = _assign_new_components(
        components,
        v6_author_groups,
        len(new_rows),
        config.policy,
    )
    if progress:
        progress(
            "groups={} forced_train={} validation={} test={}".format(
                len(components),
                assignment_metadata["forced_train_character_group_count"],
                assignment_metadata["new_split_counts"]["validation"],
                assignment_metadata["new_split_counts"]["test"],
            )
        )

    v7_rows = []
    total = len(source_rows)
    for index, row in enumerate(source_rows, 1):
        v7_rows.append(
            _v7_row(
                row,
                component_for_unit,
                assignments,
                v6_author_groups,
                protected_author_groups,
            )
        )
        if progress and (index == 1 or index % 1000 == 0 or index == total):
            progress(f"manifest={index}/{total}")

    report = _build_report(
        config,
        source_rows,
        v7_rows,
        author_rows,
        author_metadata,
        components,
        assignment_metadata,
        v6_author_groups,
        protected_author_groups,
    )
    _validate_v7(source_rows, v7_rows, report, config.policy)
    _write_csv_atomic(config.author_group_path, AUTHOR_GROUP_FIELDS, author_rows)
    _write_csv_atomic(
        config.v7_manifest_path,
        tuple(source_fields) + V7_ADDED_FIELDS,
        v7_rows,
    )
    _write_json_atomic(config.json_report_path, report)
    _write_text_atomic(config.markdown_report_path, render_v7_markdown(report))
    return report


def render_v7_markdown(report: dict[str, Any]) -> str:
    split_lines = "\n".join(
        f"- `{split}`: {count:,} sonnets."
        for split, count in report["v7_split_counts"].items()
    )
    source_lines = "\n".join(
        f"- `{source}`: validation {counts.get('validation', 0):,}; test {counts.get('test', 0):,}."
        for source, counts in report["clean_heldout_source_counts"].items()
    )
    return (
        "# Expanded Standard-Sonnet Corpus V7 Split Freeze\n\n"
        "## Result\n\n"
        f"Checkpoint 8A accounts for all {report['sonnet_universe_count']:,} canonical sonnet identities and "
        f"includes {report['v7_included_count']:,} in the V7 train/validation/test corpus. "
        f"The remaining {report['v7_excluded_count']:,} retain their canonical exclusion decisions.\n\n"
        f"{split_lines}\n\n"
        "All 1,868 V6 assignments are preserved exactly: 1,481 train, 190 validation, and 197 test. "
        "The V6 evaluation tier remains exact-identity/work held out but is explicitly not claimed to be "
        "author-disjoint.\n\n"
        "## Clean V7 Held-Out Cohorts\n\n"
        f"New grouped assignment adds {report['clean_validation_count']:,} validation and "
        f"{report['clean_test_count']:,} test sonnets. Resolved authors are absent from V6 and from V7 "
        "training; generic author labels are grouped by complete source work. Author/work connected "
        "components cannot cross the new train/validation/test boundary.\n\n"
        f"{source_lines}\n\n"
        "The approved revised policy retains all "
        f"{report['approved_new_legacy_author_training_count']:,} new sonnets whose canonical author also "
        "appears in protected V6 evaluation. This does not make the legacy tier author-disjoint; it preserves "
        "valuable training text while the separate clean V7 cohorts measure author-level generalization.\n\n"
        "## Boundary\n\n"
        "This checkpoint freezes split identities only. It copies no corpus text, includes no conditioned "
        "material, performs no Minerva tokenization, assigns no training-mixture weight, starts no GPU work, "
        "and deletes no reusable cache.\n"
    )


def _build_author_group_rows(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_label = Counter(row["author"] for row in rows)
    new_by_label = Counter(
        row["author"]
        for row in rows
        if row["training_eligible"] == "true" and not row["original_split"]
    )
    v6_groups = {
        author_group_id(row["author"])
        for row in rows
        if row["source_group"] == "v6_sonnets" and author_group_id(row["author"])
    }
    protected_groups = {
        author_group_id(row["author"])
        for row in rows
        if row["activation_status"] == "protected_v6_validation_test"
        and author_group_id(row["author"])
    }
    labels_by_group: dict[str, list[str]] = defaultdict(list)
    for label in by_label:
        group = author_group_id(label)
        if group:
            labels_by_group[group].append(label)
    representatives = {
        group: min(labels, key=lambda label: ("," in label, len(label), label.casefold()))
        for group, labels in labels_by_group.items()
    }
    result = []
    for label in sorted(by_label, key=lambda value: (value.casefold(), value)):
        key = canonicalize_author_label(label)
        group = author_group_id(label)
        result.append(
            {
                "author_label_id": f"author_label:{hashlib.sha256(label.encode('utf-8')).hexdigest()[:16]}",
                "raw_author_label": label,
                "canonical_author_key": key if group else "",
                "author_group_id": group,
                "representative_author_label": representatives.get(group, ""),
                "resolution_status": (
                    "deterministic_resolved_author_alias"
                    if group and len(labels_by_group[group]) > 1
                    else "deterministic_resolved_author"
                    if group
                    else "generic_author_work_grouped"
                ),
                "sonnet_universe_count": by_label[label],
                "new_candidate_count": new_by_label[label],
                "v6_presence": str(group in v6_groups if group else False).lower(),
                "protected_v6_presence": str(
                    group in protected_groups if group else False
                ).lower(),
            }
        )
    return result, {
        "raw_author_label_count": len(by_label),
        "resolved_author_group_count": len(labels_by_group),
        "resolved_alias_group_count": sum(len(labels) > 1 for labels in labels_by_group.values()),
        "generic_author_label_count": sum(not author_group_id(label) for label in by_label),
    }


def _build_new_components(
    rows: list[dict[str, str]],
) -> tuple[dict[str, list[dict[str, str]]], dict[str, str]]:
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    unit_node: dict[str, str] = {}
    for row in rows:
        work = derive_work_group_id(row["source_group"], row["source_id"])
        author = author_group_id(row["author"])
        author_node = author or f"generic:{work}"
        union(author_node, work)
        unit_node[row["unit_id"]] = author_node

    root_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    root_nodes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        root = find(unit_node[row["unit_id"]])
        root_rows[root].append(row)
    for node in parent:
        root_nodes[find(node)].add(node)

    components: dict[str, list[dict[str, str]]] = {}
    component_for_unit: dict[str, str] = {}
    for root in sorted(root_rows):
        identity = "\n".join(sorted(root_nodes[root]))
        component = f"split_group:{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"
        components[component] = sorted(root_rows[root], key=lambda row: row["unit_id"])
        for row in root_rows[root]:
            component_for_unit[row["unit_id"]] = component
    return components, component_for_unit


def _assign_new_components(
    components: dict[str, list[dict[str, str]]],
    v6_author_groups: set[str],
    new_count: int,
    policy: V7SplitPolicy,
) -> tuple[dict[str, str], dict[str, Any]]:
    target = round(new_count * policy.validation_fraction)
    max_heldout_size = max(1, math.floor(target * policy.max_heldout_group_target_share))
    assignments: dict[str, str] = {}
    reasons: dict[str, str] = {}
    for component, rows in components.items():
        authors = {author_group_id(row["author"]) for row in rows if author_group_id(row["author"])}
        if authors & v6_author_groups:
            assignments[component] = "train"
            reasons[component] = "legacy_v6_author_overlap_approved_train"
        elif len(rows) > max_heldout_size:
            assignments[component] = "train"
            reasons[component] = "author_work_component_over_heldout_cap_train"

    candidates = {
        component: rows
        for component, rows in components.items()
        if component not in assignments
    }
    stratum_totals = Counter(
        (row["source_group"], row["epoch_bucket"])
        for rows in candidates.values()
        for row in rows
    )
    validation_components = _select_heldout_components(
        candidates,
        target,
        "validation",
        stratum_totals,
        policy,
    )
    for component in validation_components:
        assignments[component] = "validation"
        reasons[component] = "clean_author_work_disjoint_validation"
        candidates.pop(component)
    test_components = _select_heldout_components(
        candidates,
        target,
        "test",
        stratum_totals,
        policy,
    )
    for component in test_components:
        assignments[component] = "test"
        reasons[component] = "clean_author_work_disjoint_test"
        candidates.pop(component)
    for component in candidates:
        assignments[component] = "train"
        reasons[component] = "clean_group_assignment_train"

    counts = Counter()
    for component, split in assignments.items():
        counts[split] += len(components[component])
    return assignments, {
        "target_heldout_count": target,
        "max_heldout_component_size": max_heldout_size,
        "new_split_counts": dict(sorted(counts.items())),
        "component_decisions": reasons,
        "forced_train_character_group_count": sum(
            len(components[component])
            for component, reason in reasons.items()
            if reason == "legacy_v6_author_overlap_approved_train"
        ),
        "oversize_forced_train_count": sum(
            len(components[component])
            for component, reason in reasons.items()
            if reason == "author_work_component_over_heldout_cap_train"
        ),
    }


def _select_heldout_components(
    pool: dict[str, list[dict[str, str]]],
    target: int,
    split: str,
    stratum_totals: Counter[tuple[str, str]],
    policy: V7SplitPolicy,
) -> list[str]:
    available_count = sum(stratum_totals.values())
    stratum_targets = {
        stratum: target * count / available_count
        for stratum, count in stratum_totals.items()
    }
    selected: list[str] = []
    selected_count = 0
    selected_strata: Counter[tuple[str, str]] = Counter()

    def objective(count: int, strata: Counter[tuple[str, str]]) -> float:
        global_error = ((count - target) / max(target, 1)) ** 2
        stratum_error = sum(
            (stratum_totals[stratum] / available_count)
            * ((strata[stratum] - stratum_targets[stratum]) / max(stratum_targets[stratum], 1)) ** 2
            for stratum in stratum_totals
        )
        return global_error + policy.stratum_weight * stratum_error

    current = objective(selected_count, selected_strata)
    while pool:
        best: tuple[float, str, str, Counter[tuple[str, str]]] | None = None
        for component, rows in pool.items():
            added = Counter((row["source_group"], row["epoch_bucket"]) for row in rows)
            score = objective(selected_count + len(rows), selected_strata + added)
            tie = hashlib.sha256(
                f"{policy.seed}:{split}:{component}".encode("utf-8")
            ).hexdigest()
            candidate = (score, tie, component, added)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None or best[0] >= current - 1e-15:
            break
        current, _, component, added = best
        selected.append(component)
        selected_count += len(pool[component])
        selected_strata += added
        pool = {key: value for key, value in pool.items() if key != component}
    return selected


def _v7_row(
    row: dict[str, str],
    component_for_unit: dict[str, str],
    assignments: dict[str, str],
    v6_author_groups: set[str],
    protected_author_groups: set[str],
) -> dict[str, Any]:
    result: dict[str, Any] = dict(row)
    author_key = canonicalize_author_label(row["author"])
    author = author_group_id(row["author"])
    work = derive_work_group_id(row["source_group"], row["source_id"])
    if row["source_group"] == "v6_sonnets":
        split = row["original_split"]
        tier = "legacy_v6_locked"
        decision = f"preserve_v6_{split}"
        split_group = f"legacy_v6:{row['unit_id']}"
        include = True
    elif row["training_eligible"] == "false":
        split = "excluded"
        tier = "canonical_exclusion"
        decision = "preserve_canonical_exclusion"
        split_group = ""
        include = False
    else:
        split_group = component_for_unit[row["unit_id"]]
        split = assignments[split_group]
        tier = "clean_v7_grouped" if split in {"validation", "test"} else "v7_training"
        if split == "train" and author and author in protected_author_groups:
            decision = "approved_legacy_protected_author_overlap_train"
        elif split == "train" and author and author in v6_author_groups:
            decision = "approved_legacy_v6_author_overlap_train"
        elif split == "train":
            decision = "group_assigned_train"
        else:
            decision = f"clean_author_work_disjoint_{split}"
        include = True
    result.update(
        {
            "canonical_author_key": author_key if author else "",
            "author_group_id": author,
            "author_resolution_status": (
                "resolved_author" if author else "generic_author_work_grouped"
            ),
            "work_group_id": work,
            "split_group_id": split_group,
            "v7_split": split,
            "v7_split_tier": tier,
            "v7_split_decision": decision,
            "include_in_v7": str(include).lower(),
            "v7_training_eligible": str(split == "train").lower(),
        }
    )
    return result


def _build_report(
    config: V7SplitConfig,
    source_rows: list[dict[str, str]],
    v7_rows: list[dict[str, Any]],
    author_rows: list[dict[str, Any]],
    author_metadata: dict[str, Any],
    components: dict[str, list[dict[str, str]]],
    assignment_metadata: dict[str, Any],
    v6_author_groups: set[str],
    protected_author_groups: set[str],
) -> dict[str, Any]:
    included = [row for row in v7_rows if row["include_in_v7"] == "true"]
    split_counts = Counter(row["v7_split"] for row in included)
    clean = [row for row in v7_rows if row["v7_split_tier"] == "clean_v7_grouped"]
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    epoch_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in clean:
        source_counts[row["source_group"]][row["v7_split"]] += 1
        epoch_counts[row["epoch_bucket"]][row["v7_split"]] += 1
    v6_train = [
        row for row in v7_rows
        if row["source_group"] == "v6_sonnets" and row["v7_split"] == "train"
    ]
    approved_overlap = [
        row for row in v7_rows
        if row["v7_split_decision"] == "approved_legacy_protected_author_overlap_train"
    ]
    new_count = sum(
        row["source_group"] != "v6_sonnets" and row["training_eligible"] == "true"
        for row in v7_rows
    )
    report = {
        "v7_version": V7_VERSION,
        "build_date": V7_DATE,
        "seed": config.policy.seed,
        "sonnet_universe_count": len(v7_rows),
        "v7_included_count": len(included),
        "v7_excluded_count": sum(row["v7_split"] == "excluded" for row in v7_rows),
        "v7_split_counts": dict(sorted(split_counts.items())),
        "v6_split_counts": dict(sorted(Counter(
            row["v7_split"] for row in v7_rows if row["source_group"] == "v6_sonnets"
        ).items())),
        "new_candidate_count": new_count,
        "new_split_counts": assignment_metadata["new_split_counts"],
        "new_target_fractions": {
            "train": config.policy.train_fraction,
            "validation": config.policy.validation_fraction,
            "test": config.policy.test_fraction,
        },
        "heldout_tolerance": config.policy.heldout_tolerance,
        "target_new_heldout_count": assignment_metadata["target_heldout_count"],
        "max_heldout_component_size": assignment_metadata["max_heldout_component_size"],
        "clean_validation_count": sum(row["v7_split"] == "validation" for row in clean),
        "clean_test_count": sum(row["v7_split"] == "test" for row in clean),
        "clean_heldout_source_counts": {
            key: dict(sorted(value.items())) for key, value in sorted(source_counts.items())
        },
        "clean_heldout_epoch_counts": {
            key: dict(sorted(value.items())) for key, value in sorted(epoch_counts.items())
        },
        "new_author_work_component_count": len(components),
        "approved_new_legacy_author_training_count": len(approved_overlap),
        "oversize_component_training_count": assignment_metadata["oversize_forced_train_count"],
        "v6_author_group_count": len(v6_author_groups),
        "protected_v6_author_group_count": len(protected_author_groups),
        "legacy_v6_train_author_overlap_count": sum(
            row["author_group_id"] in protected_author_groups for row in v6_train
        ),
        **author_metadata,
        "author_group_manifest_row_count": len(author_rows),
        "input_sha256": {
            _portable(config.canonical_sonnet_manifest_path, config.repo_root): _sha_file(config.canonical_sonnet_manifest_path),
            _portable(config.v6_manifest_path, config.repo_root): _sha_file(config.v6_manifest_path),
        },
        "verification": {
            "all_canonical_identities_accounted": True,
            "all_v6_splits_preserved": True,
            "canonical_exclusions_preserved": True,
            "new_author_work_components_single_split": True,
            "clean_heldout_resolved_authors_absent_from_v6_and_training": True,
            "generic_heldout_works_absent_from_training": True,
            "legacy_v6_author_overlap_disclosed": True,
            "conditioned_material_included": False,
            "corpus_text_copied": False,
            "minerva_tokenization_performed": False,
            "mixture_weights_assigned": False,
            "gpu_work_started": False,
            "cache_deleted": False,
        },
    }
    report["v7_identity_sha256"] = _identity_sha(v7_rows)
    return report


def _validate_v7(
    source_rows: list[dict[str, str]],
    v7_rows: list[dict[str, Any]],
    report: dict[str, Any],
    policy: V7SplitPolicy,
) -> None:
    if len(v7_rows) != len(source_rows):
        raise ValueError("V7 manifest does not account for the canonical universe")
    by_source = {row["unit_id"]: row for row in source_rows}
    for row in v7_rows:
        source = by_source[row["unit_id"]]
        for field in source:
            if row[field] != source[field]:
                raise ValueError(f"canonical field changed for {row['unit_id']}: {field}")
        if "conditioned" in " ".join(str(value) for value in row.values()).casefold():
            raise ValueError(f"conditioned sonnet entered V7: {row['unit_id']}")
        if row["storage_path"].startswith("data/local/"):
            raise ValueError(f"local-cache storage entered V7: {row['unit_id']}")

    included = [row for row in v7_rows if row["include_in_v7"] == "true"]
    hashes = [row["logical_sha256"] for row in included]
    if not all(hashes) or len(hashes) != len(set(hashes)):
        raise ValueError("V7 included identities are not exact-text unique")

    new_rows = [
        row
        for row in v7_rows
        if row["source_group"] != "v6_sonnets" and row["training_eligible"] == "true"
    ]
    component_splits: dict[str, set[str]] = defaultdict(set)
    for row in new_rows:
        component_splits[row["split_group_id"]].add(row["v7_split"])
    if any(len(splits) != 1 for splits in component_splits.values()):
        raise ValueError("new author/work component crosses V7 splits")

    train_authors = {
        row["author_group_id"]
        for row in included
        if row["v7_split"] == "train" and row["author_group_id"]
    }
    v6_authors = {
        row["author_group_id"]
        for row in included
        if row["source_group"] == "v6_sonnets" and row["author_group_id"]
    }
    clean_heldout = [row for row in v7_rows if row["v7_split_tier"] == "clean_v7_grouped"]
    heldout_authors = {row["author_group_id"] for row in clean_heldout if row["author_group_id"]}
    if heldout_authors & train_authors or heldout_authors & v6_authors:
        raise ValueError("clean held-out author appears in V6 or V7 training")
    heldout_works = {row["work_group_id"] for row in clean_heldout}
    train_works = {row["work_group_id"] for row in included if row["v7_split"] == "train"}
    if heldout_works & train_works:
        raise ValueError("clean held-out work appears in V7 training")

    new_count = report["new_candidate_count"]
    for split in ("validation", "test"):
        actual = report["new_split_counts"][split] / new_count
        if abs(actual - getattr(policy, f"{split}_fraction")) > policy.heldout_tolerance:
            raise ValueError(f"new {split} fraction exceeds tolerance")
    if report["v6_split_counts"] != {"test": 197, "train": 1481, "validation": 190}:
        raise ValueError("V6 split counts changed")
    if report["approved_new_legacy_author_training_count"] <= 0:
        raise ValueError("approved legacy-author training cohort is unexpectedly empty")


def _verify_v6_freeze(
    canonical_rows: list[dict[str, str]], v6_rows: list[dict[str, str]]
) -> None:
    v6_by_id = {row["poem_id"]: row for row in v6_rows}
    canonical_v6 = {
        row["source_id"]: row for row in canonical_rows if row["source_group"] == "v6_sonnets"
    }
    if len(v6_by_id) != 1868 or len(canonical_v6) != 1868 or set(v6_by_id) != set(canonical_v6):
        raise ValueError("canonical V6 identity set does not match the frozen V6 manifest")
    for poem_id, v6 in v6_by_id.items():
        expected = v6["split_expanded_with_petrarch"]
        canonical = canonical_v6[poem_id]
        if canonical["original_split"] != expected:
            raise ValueError(f"canonical V6 split changed: {poem_id}")


def _validate_policy(policy: V7SplitPolicy) -> None:
    fractions = policy.train_fraction + policy.validation_fraction + policy.test_fraction
    if abs(fractions - 1.0) > 1e-12:
        raise ValueError("V7 split fractions must sum to one")
    if min(policy.train_fraction, policy.validation_fraction, policy.test_fraction) <= 0:
        raise ValueError("V7 split fractions must be positive")
    if policy.heldout_tolerance <= 0 or not 0 < policy.max_heldout_group_target_share <= 1:
        raise ValueError("V7 tolerance and group cap must be positive")
    if policy.stratum_weight < 0:
        raise ValueError("V7 stratum weight must be non-negative")


def _identity_sha(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: value["unit_id"]):
        digest.update(
            "\0".join(
                (
                    row["unit_id"], row["logical_sha256"], row["author_group_id"],
                    row["work_group_id"], row["split_group_id"], row["v7_split"],
                )
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        return fields, list(reader)


def _require_fields(fields: tuple[str, ...], required: set[str], label: str) -> None:
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _write_csv_atomic(
    path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
