#!/usr/bin/env python3
"""Score the opening-anchored plan grid and its repaired variant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.constrained_plan_validation import load_anchored_records
from sonnet_analysis.form_targeted_validation import load_generation_records
from sonnet_analysis.plan_then_poem_validation import DEFAULT_SCHEME, adherence
from sonnet_evaluation.sonnet_prosody_sealed import (
    aggregate,
    paired_comparison,
    pair_records,
    score_records,
)

ANCHORED = "anchored"
REPAIRED = "anchored_repaired"
BASELINE = "dpo"
SYSTEMS = (ANCHORED, REPAIRED, BASELINE)
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
        default=ROOT / "artifacts/local/constrained_plan/validation/generation",
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
        default=ROOT / "reports/constrained_plan_validation_v1.md",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=ROOT / "reports/constrained_plan_validation_v1.json",
    )
    parser.add_argument(
        "--scores-jsonl",
        type=Path,
        default=ROOT / "artifacts/local/constrained_plan/validation/scores_v1.jsonl",
    )
    return parser.parse_args()


def repository_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return "<outside repository>"


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def main() -> None:
    args = parse_args()
    plan_records = load_anchored_records(args.generation_dir)
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
        if record["system_id"] in (ANCHORED, REPAIRED):
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

    scheme_rate = {
        system: mean(values(system, "plan_scheme_ok")) or 0.0
        for system in (ANCHORED, REPAIRED)
    }
    key_rate = {
        system: mean(values(system, "key_match_rate")) or 0.0
        for system in (ANCHORED, REPAIRED)
    }
    repair_distance = mean(values(REPAIRED, "repair_distance")) or 0.0
    metrics = {system: aggregate(scored, system) for system in SYSTEMS}
    comparisons = []
    for label, systems in (
        ("anchored - baseline", (ANCHORED, BASELINE)),
        ("repaired - baseline", (REPAIRED, BASELINE)),
        ("repaired - anchored", (REPAIRED, ANCHORED)),
    ):
        pairs = pair_records(scored, systems=systems)
        for field in CONTINUOUS_FIELDS:
            comparisons.append((label, field, paired_comparison(pairs, field)))

    accepted = next(
        row for label, field, row in comparisons
        if label == "repaired - anchored" and field == "hendecasyllable_lines"
    )
    reasons = []
    if scheme_rate[ANCHORED] >= 0.95:
        result = "SKIP_REPAIR_FORM_MET"
        reasons.append(
            "The no-repair anchored condition already reaches 95% scheme "
            "compliance."
        )
    elif scheme_rate[REPAIRED] >= 0.95 and accepted["ci95_low"] > -0.5:
        result = "FORM_OBJECTIVE_MET"
        reasons.append(
            "Repaired scheme compliance reaches 95% without a material "
            "accepted-line drop."
        )
    elif scheme_rate[REPAIRED] >= 0.95 and (
        accepted["ci95_high"] < -0.5 or repair_distance > 4.0
    ):
        result = "TRADE_OFF_TRY_DECODE_CONSTRAINTS"
        reasons.append(
            "Repair fixes the scheme but costs accepted lines or repairs more "
            "than four lines per output."
        )
    else:
        result = "FORM_STILL_UNMET"
        reasons.append("Repaired scheme compliance stays below 95%.")
    reasons.append(
        f"Anchored scheme {scheme_rate[ANCHORED]:.4f}, repaired "
        f"{scheme_rate[REPAIRED]:.4f}, repaired-minus-anchored accepted lines "
        f"{accepted['mean_difference_stage_3_minus_dpo']:.3f} "
        f"[{accepted['ci95_low']:.3f}, {accepted['ci95_high']:.3f}], mean "
        f"repair distance {repair_distance:.3f}."
    )

    args.scores_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_jsonl.open("w", encoding="utf-8") as handle:
        for record in scored:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    lines = [
        "# Constrained Plan Validation v1",
        "",
        "Date: 2026-09-13",
        "",
        "Opening-anchored plans fix line 1 to the opening's own ending and",
        "build a feasible scheme. The repaired condition replaces line-final",
        "words with the planned words. The baseline is the verifier-DPO no-plan",
        "output. Form only; no literary-quality claim.",
        "",
        "## Metrics",
        "",
        "| Metric | Anchored | Repaired | Baseline |",
        "|---|---:|---:|---:|",
        f"| Scheme compliance | {scheme_rate[ANCHORED]:.4f} | {scheme_rate[REPAIRED]:.4f} | 0.0000 |",
        f"| Key-match adherence | {key_rate[ANCHORED]:.4f} | {key_rate[REPAIRED]:.4f} | n/a |",
        f"| Mean repair distance | n/a | {repair_distance:.3f} | n/a |",
        f"| Accepted lines (mean) | {metrics[ANCHORED]['mean_hendecasyllable_lines']:.3f} | {metrics[REPAIRED]['mean_hendecasyllable_lines']:.3f} | {metrics[BASELINE]['mean_hendecasyllable_lines']:.3f} |",
        f"| Failed lines (mean) | {metrics[ANCHORED]['mean_failed_lines']:.3f} | {metrics[REPAIRED]['mean_failed_lines']:.3f} | {metrics[BASELINE]['mean_failed_lines']:.3f} |",
        f"| Rhyme score (mean) | {metrics[ANCHORED]['mean_rhyme_score']:.4f} | {metrics[REPAIRED]['mean_rhyme_score']:.4f} | {metrics[BASELINE]['mean_rhyme_score']:.4f} |",
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
            f"- Result: {result}",
        ]
    )
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Repair can break grammar and meaning; all claims are form-only.",
            "- The coherence proxies are structural, not a literary judgment.",
            "",
            "Verification: `python3 scripts/score_constrained_plan_validation.py`",
            "",
        ]
    )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "schema_version": 1,
        "generation_dir": repository_relative(args.generation_dir),
        "baseline_dir": repository_relative(args.baseline_dir),
        "scheme_rate": scheme_rate,
        "key_rate": key_rate,
        "repair_distance": repair_distance,
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
        "constrained-plan-score | "
        f"anchored_scheme={scheme_rate[ANCHORED]:.4f} "
        f"repaired_scheme={scheme_rate[REPAIRED]:.4f} "
        f"repair_distance={repair_distance:.3f} "
        f"accepted_gap={accepted['mean_difference_stage_3_minus_dpo']:.3f} "
        f"[{accepted['ci95_low']:.3f}, {accepted['ci95_high']:.3f}] "
        f"reading={result}"
    )


if __name__ == "__main__":
    main()
