#!/usr/bin/env python3
"""Export aggregate, text-free lineage for the four planned HF artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WINDOW_ROOT = REPOSITORY_ROOT / "data/local/minerva_7b_v7/window_indexes/training"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "release/huggingface"
RELEASE_PLAN = DEFAULT_OUTPUT_ROOT / "release_plan.yml"
CANONICAL_UNITS = REPOSITORY_ROOT / "data/metadata/cross_archive_canonical_units_v1.csv"
V6_SONNETS = REPOSITORY_ROOT / "data/metadata/sonnets_expanded_v6_manifest.csv"

STAGE_ORDER = (
    "stage_1_historical_general",
    "stage_2_non_sonnet_poetry",
    "stage_3_sonnets",
)
ARTIFACT_STAGES = {
    "stage1": STAGE_ORDER[:1],
    "stage2": STAGE_ORDER[:2],
    "stage3": STAGE_ORDER,
    "dpo_adapter": STAGE_ORDER,
}
DIRECT_FAMILIES = {
    "bibit": "Biblioteca Italiana",
    "gutenberg": "Project Gutenberg",
    "ilc_ota": "Oxford Text Archive / ILC",
    "liber_liber": "Liber Liber",
    "paisa_even_byte_windows_v1": "PAISA",
    "wikisource": "Italian Wikisource",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_archive(value: str) -> str:
    lowered = value.casefold()
    if "wikisource" in lowered:
        return "Italian Wikisource"
    if "liber liber" in lowered:
        return "Liber Liber"
    if "gutenberg" in lowered:
        return "Project Gutenberg"
    if "biblioteca italiana" in lowered or "bibit" in lowered:
        return "Biblioteca Italiana"
    if "oxford" in lowered or "ilc" in lowered:
        return "Oxford Text Archive / ILC"
    raise ValueError(f"Unrecognized source archive: {value!r}")


def _load_existing_source_families(path: Path = CANONICAL_UNITS) -> dict[str, str]:
    families: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            unit_id = row.get("unit_id") or row.get("canonical_unit_id")
            archive = row.get("source_archive")
            if unit_id and archive:
                families[unit_id] = _normalize_archive(archive)
    if not families:
        raise ValueError(f"No canonical-unit source families found in {path}")
    return families


def _load_v6_source_families(path: Path = V6_SONNETS) -> dict[str, str]:
    families: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            poem_id = row.get("poem_id") or row.get("id")
            archive = row.get("source_archive")
            if poem_id and archive:
                families[poem_id] = _normalize_archive(archive)
    if not families:
        raise ValueError(f"No V6 sonnet source families found in {path}")
    return families


def source_family(
    unit_id: str,
    *,
    existing_families: dict[str, str],
    v6_families: dict[str, str],
) -> str:
    prefix = unit_id.split(":", 1)[0]
    if prefix in DIRECT_FAMILIES:
        return DIRECT_FAMILIES[prefix]
    if prefix == "existing":
        try:
            return existing_families[unit_id]
        except KeyError as exc:
            raise ValueError(f"Missing canonical source family for {unit_id}") from exc
    if prefix == "v6":
        poem_id = unit_id.split(":", 2)[2]
        try:
            return v6_families[poem_id]
        except KeyError as exc:
            raise ValueError(f"Missing V6 source family for {unit_id}") from exc
    raise ValueError(f"Unrecognized training unit prefix in {unit_id}")


def aggregate_stage(
    path: Path,
    *,
    selected_windows: int,
    existing_families: dict[str, str],
    v6_families: dict[str, str],
) -> dict[str, Any]:
    components: Counter[str] = Counter()
    families: Counter[str] = Counter()
    selected_bytes = hashlib.sha256()
    rows = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            if rows >= selected_windows:
                break
            selected_bytes.update(raw_line)
            row = json.loads(raw_line)
            rows += 1
            components[str(row["component"])] += int(row["target_tokens"])
            contribution_total = 0
            for contribution in row["target_contributions"]:
                tokens = int(contribution["tokens"])
                contribution_total += tokens
                family = source_family(
                    str(contribution["unit_id"]),
                    existing_families=existing_families,
                    v6_families=v6_families,
                )
                families[family] += tokens
            if contribution_total != int(row["target_tokens"]):
                raise ValueError(f"Target-token mismatch in {path} row {rows}")
    if rows != selected_windows:
        raise ValueError(f"Expected {selected_windows} rows in {path}, found {rows}")
    target_tokens = selected_windows * 2048
    if sum(families.values()) != target_tokens:
        raise ValueError(f"Source-family total mismatch for {path}")
    return {
        "stage_id": path.stem,
        "selected_windows": selected_windows,
        "target_tokens": target_tokens,
        "selected_window_rows_sha256": selected_bytes.hexdigest(),
        "component_target_tokens": dict(sorted(components.items())),
        "source_family_target_tokens": dict(sorted(families.items())),
    }


def build_lineages(
    *, window_root: Path = DEFAULT_WINDOW_ROOT, plan_path: Path = RELEASE_PLAN
) -> dict[str, dict[str, Any]]:
    plan = _load_json(plan_path)
    artifacts = {row["artifact_id"]: row for row in plan["artifacts"]}
    stage_limits = {
        "stage_1_historical_general": int(artifacts["stage1"]["selected_windows"]),
        "stage_2_non_sonnet_poetry": int(artifacts["stage2"]["selected_windows"]),
        "stage_3_sonnets": int(artifacts["stage3"]["selected_windows"]),
    }
    existing_families = _load_existing_source_families()
    v6_families = _load_v6_source_families()
    stage_rows = {
        stage_id: aggregate_stage(
            window_root / f"{stage_id}.jsonl",
            selected_windows=stage_limits[stage_id],
            existing_families=existing_families,
            v6_families=v6_families,
        )
        for stage_id in STAGE_ORDER
    }

    lineages: dict[str, dict[str, Any]] = {}
    for artifact_id, included_stages in ARTIFACT_STAGES.items():
        cumulative: Counter[str] = Counter()
        for stage_id in included_stages:
            cumulative.update(stage_rows[stage_id]["source_family_target_tokens"])
        total = sum(cumulative.values())
        lineage: dict[str, Any] = {
            "schema_version": "transformer_poetry_hf_lineage_v1",
            "artifact_id": artifact_id,
            "publication_scope": "aggregate_only_no_corpus_text_or_unit_identifiers",
            "artifact": artifacts[artifact_id],
            "parent": plan["parent"],
            "completed_training_stages": [stage_rows[stage] for stage in included_stages],
            "cumulative_target_tokens": total,
            "cumulative_source_family_target_tokens": dict(sorted(cumulative.items())),
            "cumulative_source_family_percent": {
                family: round(tokens * 100 / total, 8)
                for family, tokens in sorted(cumulative.items())
            },
            "qualification": (
                "Counts are exact target-token contributions from the deterministic "
                "sampled windows consumed by the selected endpoints; they are not "
                "document counts, example counts, or corpus-size shares."
            ),
            "rights_boundary": (
                "Source disclosure is not a redistribution grant and does not decide "
                "whether training-data licenses govern model weights."
            ),
        }
        if artifact_id == "dpo_adapter":
            lineage["preference_training"] = {
                "classification": "AI-judged_not_human_aligned",
                "generated_candidates": 4096,
                "preference_pairs": 534,
                "training_pairs": 482,
                "validation_pairs": 52,
                "optimizer_updates": 61,
                "human_ai_calibration_agreement": "12/20_failed_gate",
                "raw_material_public": False,
            }
        lineages[artifact_id] = lineage
    return lineages


def write_lineages(lineages: dict[str, dict[str, Any]], output_root: Path) -> None:
    for artifact_id, payload in sorted(lineages.items()):
        destination = output_root / artifact_id / "lineage.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-root", type=Path, default=DEFAULT_WINDOW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--plan", type=Path, default=RELEASE_PLAN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_lineages(
        build_lineages(window_root=args.window_root, plan_path=args.plan),
        args.output_root,
    )


if __name__ == "__main__":
    main()
