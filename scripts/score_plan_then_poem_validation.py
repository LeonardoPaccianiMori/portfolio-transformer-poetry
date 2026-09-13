#!/usr/bin/env python3
"""Score the plan-following evaluation over planned, mismatched, and control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.form_targeted_validation import load_generation_records
from sonnet_analysis.plan_then_poem_validation import (
    DEFAULT_SCHEME,
    adherence,
    load_plan_records,
)
from sonnet_evaluation.sonnet_prosody_sealed import (
    aggregate,
    mcnemar,
    paired_comparison,
    pair_records,
    score_records,
)

PLANNED = "planned"
MISMATCHED = "mismatched"
CONTROL = "control"
BASELINE = "dpo"
PLAN_CONDITIONS = (PLANNED, MISMATCHED, CONTROL)
SYSTEMS = PLAN_CONDITIONS + (BASELINE,)
CONTINUOUS_FIELDS = (
    "hendecasyllable_lines",
    "failed_lines",
    "uncertain_lines",
    "rhyme_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-dir",
        type=Path,
        default=ROOT / "artifacts/local/plan_following_sft/validation/generation",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=ROOT
        / "artifacts/local/verifier_labelled_dpo/validation/generation",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=ROOT / "reports/plan_following_sft_v1.md",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=ROOT / "reports/plan_following_sft_v1.json",
    )
    parser.add_argument(
        "--scores-jsonl",
        type=Path,
        default=ROOT / "artifacts/local/plan_following_sft/validation/scores_v1.jsonl",
    )
    return parser.parse_args()


def repository_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return "<outside repository>"


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def format_report(
    *,
    adherence_metrics: dict,
    metrics: dict,
    comparisons: list[tuple[str, str, dict]],
    per_position: list[float],
    reading: dict,
) -> str:
    lines = [
        "# Plan-Following SFT v1",
        "",
        "Date: 2026-09-13",
        "",
        "The planned condition shows the model the pre-committed line endings.",
        "The mismatched condition shows the plan of a different opening, which",
        "separates plan reading from a shifted ending distribution. The format",
        "control shows non-rhyming placeholder endings. The baseline is the",
        "verifier-DPO no-plan output. All conditions use the same openings,",
        "seeds, prompt builder, and recipe; the sealed test set was not",
        "accessed. Echo outputs are excluded from adherence and reported.",
        "",
        "## Adherence",
        "",
        "| Metric | Planned | Mismatched | Control |",
        "|---|---:|---:|---:|",
        f"| Word match rate | {adherence_metrics['word_rate'][PLANNED]:.4f} | {adherence_metrics['word_rate'][MISMATCHED]:.4f} | {adherence_metrics['word_rate'][CONTROL]:.4f} |",
        f"| Key match rate | {adherence_metrics['key_rate'][PLANNED]:.4f} | {adherence_metrics['key_rate'][MISMATCHED]:.4f} | {adherence_metrics['key_rate'][CONTROL]:.4f} |",
        f"| Planned scheme compliance | {adherence_metrics['scheme_rate'][PLANNED]:.4f} | {adherence_metrics['scheme_rate'][MISMATCHED]:.4f} | {adherence_metrics['scheme_rate'][CONTROL]:.4f} |",
        f"| Echo outputs | {adherence_metrics['echo_count'][PLANNED]} | {adherence_metrics['echo_count'][MISMATCHED]} | {adherence_metrics['echo_count'][CONTROL]} |",
        "",
        "## Form metrics",
        "",
        "| Metric | Planned | Mismatched | Control | Baseline |",
        "|---|---:|---:|---:|---:|",
        f"| Accepted lines (mean) | {metrics[PLANNED]['mean_hendecasyllable_lines']:.3f} | {metrics[MISMATCHED]['mean_hendecasyllable_lines']:.3f} | {metrics[CONTROL]['mean_hendecasyllable_lines']:.3f} | {metrics[BASELINE]['mean_hendecasyllable_lines']:.3f} |",
        f"| Failed lines (mean) | {metrics[PLANNED]['mean_failed_lines']:.3f} | {metrics[MISMATCHED]['mean_failed_lines']:.3f} | {metrics[CONTROL]['mean_failed_lines']:.3f} | {metrics[BASELINE]['mean_failed_lines']:.3f} |",
        f"| Uncertain lines (mean) | {metrics[PLANNED]['mean_uncertain_lines']:.3f} | {metrics[MISMATCHED]['mean_uncertain_lines']:.3f} | {metrics[CONTROL]['mean_uncertain_lines']:.3f} | {metrics[BASELINE]['mean_uncertain_lines']:.3f} |",
        f"| Rhyme score (mean) | {metrics[PLANNED]['mean_rhyme_score']:.4f} | {metrics[MISMATCHED]['mean_rhyme_score']:.4f} | {metrics[CONTROL]['mean_rhyme_score']:.4f} | {metrics[BASELINE]['mean_rhyme_score']:.4f} |",
        "",
        "## Planned key-match adherence by line position",
        "",
        "| " + " | ".join(str(index + 1) for index in range(14)) + " |",
        "|" + "---:|" * 14,
        "| " + " | ".join(f"{value:.3f}" for value in per_position) + " |",
        "",
        "## Paired comparisons",
        "",
        "| Comparison | Field | Mean difference | 95% CI |",
        "|---|---|---:|---|",
    ]
    for label, field, row in comparisons:
        lines.append(
            f"| {label} | {field} | "
            f"{row['mean_difference_stage_3_minus_dpo']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Pre-registered reading",
            "",
            f"- Result: {reading['result']}",
        ]
    )
    for reason in reading["reasons"]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Training uses each sonnet's own endings; reproduction is rewarded",
            "  and prospective planning is not tested.",
            "- Adherence can be met with odd or archaic endings and does not",
            "  measure coherence or literary quality.",
            "- The checker's definite coverage limit (87.1% on the ground truth)",
            "  applies to the metre counts.",
            "",
            "Verification: `python3 scripts/score_plan_then_poem_validation.py`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    plan_records = load_plan_records(args.generation_dir)
    plan_seeds = {int(record["seed"]) for record in plan_records}
    baseline_records = [
        record
        for record in load_generation_records(args.baseline_dir)
        if record["system_id"] == BASELINE and int(record["seed"]) in plan_seeds
    ]
    for record in plan_records:
        metrics = adherence(str(record["text"]), record["planned_words"])
        if record["echo"]:
            metrics = {**metrics, "word_match_rate": 0.0, "key_match_rate": 0.0}
        record.update(metrics)
    scored = score_records(plan_records + baseline_records)
    for record in scored:
        if record["system_id"] in PLAN_CONDITIONS:
            record["plan_scheme_ok"] = bool(
                record["rhyme_scheme"] == DEFAULT_SCHEME
                and record["quatrain_ok"]
                and record["tercet_ok"]
            )

    def values(system: str, field: str) -> list[float]:
        return [
            float(record[field])
            for record in scored
            if record["system_id"] == system and record.get(field) is not None
        ]

    adherence_metrics = {
        "word_rate": {
            system: mean(values(system, "word_match_rate")) or 0.0
            for system in PLAN_CONDITIONS
        },
        "key_rate": {
            system: mean(values(system, "key_match_rate")) or 0.0
            for system in PLAN_CONDITIONS
        },
        "scheme_rate": {
            system: mean(values(system, "plan_scheme_ok")) or 0.0
            for system in PLAN_CONDITIONS
        },
        "echo_count": {
            system: sum(
                1 for record in scored
                if record["system_id"] == system and record.get("echo")
            )
            for system in PLAN_CONDITIONS
        },
    }
    per_position = [
        mean(
            [
                float(record["key_by_line"][index])
                for record in scored
                if record["system_id"] == PLANNED
                and record.get("key_by_line")
                and index < len(record["key_by_line"])
            ]
        )
        or 0.0
        for index in range(14)
    ]
    metrics = {system: aggregate(scored, system) for system in SYSTEMS}
    comparisons = []
    for label, systems in (
        ("planned - baseline", (PLANNED, BASELINE)),
        ("mismatched - baseline", (MISMATCHED, BASELINE)),
        ("control - baseline", (CONTROL, BASELINE)),
        ("planned - mismatched", (PLANNED, MISMATCHED)),
        ("planned - control", (PLANNED, CONTROL)),
    ):
        pairs = pair_records(scored, systems=systems)
        for field in CONTINUOUS_FIELDS:
            comparisons.append((label, field, paired_comparison(pairs, field)))
    pm_pairs = pair_records(scored, systems=(PLANNED, MISMATCHED))
    adherence_gap = paired_comparison(pm_pairs, "key_match_rate")

    reasons = []
    key_rate = adherence_metrics["key_rate"][PLANNED]
    scheme_rate = adherence_metrics["scheme_rate"][PLANNED]
    accepted = next(
        row for label, field, row in comparisons
        if label == "planned - baseline" and field == "hendecasyllable_lines"
    )
    if (
        key_rate >= 0.40
        and scheme_rate > 0
        and adherence_gap["ci95_low"] > 0
        and accepted["ci95_high"] > -0.5
    ):
        result = "PROCEED_TO_PLAN_THEN_POEM"
        reasons.append(
            "Adherence reached 40% with scheme compliance above zero, clearly "
            "above the mismatched condition, without a material metre drop."
        )
    elif key_rate < 0.20 or adherence_gap["ci95_high"] < 0:
        result = "STOP_OR_REFRAME"
        reasons.append(
            "Adherence stayed below 20% or failed to exceed the mismatched "
            "condition."
        )
    else:
        result = "ONE_MORE_EPOCH_THEN_REVIEW"
        reasons.append(
            "Adherence is in the partial band; one identical epoch is "
            "pre-registered, then a single re-evaluation."
        )
    reasons.append(
        f"Planned key-match {key_rate:.4f}; mismatched "
        f"{adherence_metrics['key_rate'][MISMATCHED]:.4f}; gap 95% CI "
        f"[{adherence_gap['ci95_low']:.4f}, {adherence_gap['ci95_high']:.4f}]; "
        f"planned scheme compliance {scheme_rate:.4f}."
    )

    args.scores_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_jsonl.open("w", encoding="utf-8") as handle:
        for record in scored:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report_markdown = format_report(
        adherence_metrics=adherence_metrics,
        metrics=metrics,
        comparisons=comparisons,
        per_position=per_position,
        reading={"result": result, "reasons": reasons},
    )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(report_markdown, encoding="utf-8")

    payload = {
        "schema_version": 1,
        "generation_dir": repository_relative(args.generation_dir),
        "baseline_dir": repository_relative(args.baseline_dir),
        "adherence": adherence_metrics,
        "adherence_gap_planned_minus_mismatched": adherence_gap,
        "per_position_key_match": per_position,
        "system_metrics": metrics,
        "paired_comparisons": [
            {"comparison": label, "field": field, **row}
            for label, field, row in comparisons
        ],
        "reading": {"result": result, "reasons": reasons},
        "scores_jsonl": repository_relative(args.scores_jsonl),
        "checker_status": "reviewed_conservative_2026-09-13",
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "plan-following-score | "
        f"planned={key_rate:.4f} mismatched="
        f"{adherence_metrics['key_rate'][MISMATCHED]:.4f} "
        f"control={adherence_metrics['key_rate'][CONTROL]:.4f} "
        f"scheme={scheme_rate:.4f} gap_ci="
        f"[{adherence_gap['ci95_low']:.4f}, {adherence_gap['ci95_high']:.4f}] "
        f"reading={result}"
    )


if __name__ == "__main__":
    main()
