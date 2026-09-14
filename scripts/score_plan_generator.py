#!/usr/bin/env python3
"""Score the plan generator and the composed two-stage pipeline."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.few_shot_context_validation import load_context_records
from sonnet_analysis.minerva_v7_exploratory_prompts import (
    validate_exploratory_prompt_manifest,
)
from sonnet_evaluation.sonnet_prosody_sealed import score_records
from sonnet_training.reasoning_trace_sft import (
    PLAN_ARMS,
    parse_trace,
    validate_trace,
)
from sonnet_training.self_play_rft import self_play_reading


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-dir",
        type=Path,
        default=ROOT / "artifacts/local/plan_generator/validation/generation",
    )
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=ROOT / "artifacts/local/plan_generator/validation/stage2",
    )
    parser.add_argument(
        "--judge-json",
        type=Path,
        default=ROOT / "artifacts/local/plan_generator/coherence_judge_v1.json",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=ROOT / "reports/plan_generator_sft_v1.md",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=ROOT / "reports/plan_generator_sft_v1.json",
    )
    parser.add_argument(
        "--scores-jsonl",
        type=Path,
        default=ROOT / "artifacts/local/plan_generator/validation/scores_v1.jsonl",
    )
    return parser.parse_args()


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def format_value(value: float | None, fmt: str) -> str:
    return "n/a" if value is None else fmt.format(value)


def main() -> None:
    args = parse_args()
    args.stage1_dir = args.stage1_dir.resolve()
    args.stage2_dir = args.stage2_dir.resolve()
    args.judge_json = args.judge_json.resolve()
    args.report_md = args.report_md.resolve()
    args.report_json = args.report_json.resolve()
    args.scores_jsonl = args.scores_jsonl.resolve()
    prompts = validate_exploratory_prompt_manifest(
        ROOT / "configs/minerva_7b_v7_exploratory_prompts.json",
        expected_sha256="2f33aa518aa61c11193831e53b07fd3bd861a72bf68bb23c0e0e5b1a13b1d0c7",
    )["prompts"]
    opening_by_id = {str(prompt["id"]): str(prompt["opening_line"]) for prompt in prompts}
    stage1 = load_context_records(args.stage1_dir)
    plan_stats: dict[str, dict] = {}
    plan_valid_keys: dict[str, set[tuple[str, int]]] = {}
    for arm in PLAN_ARMS:
        rows = [record for record in stage1 if record["system_id"] == arm]
        parsed_count = 0
        schema_count = 0
        valid_count = 0
        schemes = Counter()
        valid_keys = set()
        for record in rows:
            opening = opening_by_id[str(record["prompt_id"])]
            parsed = parse_trace(str(record["text"]), require_poem=False)
            if parsed is None:
                continue
            parsed_count += 1
            trace = validate_trace(parsed, opening)
            if trace["schema_valid"]:
                schema_count += 1
                schemes[str(parsed["schema"])] += 1
            if trace["valid"]:
                valid_count += 1
                valid_keys.add((str(record["prompt_id"]), int(record["seed"])))
        plan_stats[arm] = {
            "outputs": len(rows),
            "parsed": parsed_count,
            "schema_valid": schema_count,
            "valid": valid_count,
            "schemes": schemes.most_common(5),
        }
        plan_valid_keys[arm] = valid_keys

    stage2 = load_context_records(args.stage2_dir)
    stage2_scored = score_records(
        [
            {
                "path": record["path"],
                "prompt_id": record["prompt_id"],
                "seed": record["seed"],
                "system_id": record["system_id"],
                "text": record["text"],
            }
            for record in stage2
        ]
    )
    for row in stage2_scored:
        row["valid_without_repair"] = bool(row["quatrain_ok"] and row["tercet_ok"])
    stage2_valid_keys: dict[str, set[tuple[str, int]]] = {}
    stage2_stats: dict[str, dict] = {}
    for arm in PLAN_ARMS:
        condition = f"stage2_{arm}"
        rows = [row for row in stage2_scored if row["system_id"] == condition]
        keys = {
            (str(row["prompt_id"]), int(row["seed"]))
            for row in rows
            if row["valid_without_repair"]
        }
        stage2_valid_keys[arm] = keys
        stage2_stats[arm] = {
            "poems": len(rows),
            "valid_without_repair": mean(
                [float(row["valid_without_repair"]) for row in rows]
            ),
            "accepted_lines": mean(
                [float(row["hendecasyllable_lines"]) for row in rows]
            )
            if rows
            else None,
            "failed_lines": mean([float(row["failed_lines"]) for row in rows])
            if rows
            else None,
        }

    composed: dict[str, float] = {}
    for arm in PLAN_ARMS:
        grid = {
            (str(record["prompt_id"]), int(record["seed"]))
            for record in stage1
            if record["system_id"] == arm
        }
        wins = len(plan_valid_keys[arm] & stage2_valid_keys[arm])
        composed[arm] = wins / len(grid) if grid else 0.0
        stage2_stats[arm]["composed"] = composed[arm]

    reading = self_play_reading(
        baseline_valid=composed[PLAN_ARMS[0]],
        rft_valid=composed[PLAN_ARMS[1]],
        baseline_accepted=float(stage2_stats[PLAN_ARMS[0]]["accepted_lines"] or 0.0),
        rft_accepted=float(stage2_stats[PLAN_ARMS[1]]["accepted_lines"] or 0.0),
        judge_gain=None,
    )

    with args.scores_jsonl.open("w", encoding="utf-8") as handle:
        for row in stage2_scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Plan Generator and Composed Pipeline v1",
        "",
        f"Date: {datetime.date.today().isoformat()}",
        "",
        "Stage one writes the rhyme plan only. Stage two writes the poem with",
        "the existing plan-following model, using every valid stage-one plan.",
        "The composed rate counts the full opening-and-seed grid where the plan",
        "is valid and the poem is scheme-valid without repair.",
        "",
        "## Stage one: plan validity",
        "",
        "| Metric | Baseline | Plan SFT |",
        "|---|---:|---:|",
    ]
    for key, label in (
        ("outputs", "Outputs"),
        ("parsed", "Parsed plans"),
        ("schema_valid", "Valid scheme patterns"),
        ("valid", "Valid plans (true rhymes)"),
    ):
        values = [plan_stats[arm][key] for arm in PLAN_ARMS]
        lines.append(
            "| " + label + " | " + " | ".join(str(value) for value in values) + " |"
        )
    lines.extend(["", "## Stage two and composed", "", "| Metric | Baseline | Plan SFT |", "|---|---:|---:|"])
    for key, label, fmt in (
        ("poems", "Stage-two poems", "{:d}"),
        ("valid_without_repair", "Poems scheme-valid", "{:.4f}"),
        ("accepted_lines", "Accepted lines", "{:.3f}"),
        ("failed_lines", "Failed lines", "{:.3f}"),
        ("composed", "Composed pipeline rate", "{:.4f}"),
    ):
        values = [stage2_stats[arm][key] for arm in PLAN_ARMS]
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(format_value(value, fmt) for value in values)
            + " |"
        )
    lines.extend(["", "## Pre-registered reading", "", f"- Result: {reading['result']}"])
    for reason in reading["reasons"]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Stage two uses the previously trained plan-following model; the",
            "  composed rate is the honest end-to-end number for this pipeline.",
            "- Plans come from corpus poems and lexicon augmentation, so the",
            "  planning skill is imitation of valid plans, not free invention.",
            "",
            "Verification: `python3 scripts/score_plan_generator.py`",
            "",
        ]
    )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "schema_version": 1,
        "plan_stats": {
            arm: {**plan_stats[arm], "schemes": [list(row) for row in plan_stats[arm]["schemes"]]}
            for arm in PLAN_ARMS
        },
        "stage2_stats": stage2_stats,
        "composed": composed,
        "reading": reading,
        "scores_jsonl": str(args.scores_jsonl.relative_to(ROOT)),
    }
    args.report_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "plan-generator-score | "
        f"plan_valid_baseline={plan_stats[PLAN_ARMS[0]]['valid']}/{plan_stats[PLAN_ARMS[0]]['outputs']} "
        f"plan_valid_sft={plan_stats[PLAN_ARMS[1]]['valid']}/{plan_stats[PLAN_ARMS[1]]['outputs']} "
        f"composed_baseline={composed[PLAN_ARMS[0]]:.4f} "
        f"composed_sft={composed[PLAN_ARMS[1]]:.4f} reading={reading['result']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
