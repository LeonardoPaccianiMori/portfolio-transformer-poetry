#!/usr/bin/env python3
"""Score the plan-then-poem planned and control grids against the baseline."""

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
CONTROL = "control"
BASELINE = "dpo"
SYSTEMS = (PLANNED, CONTROL, BASELINE)
CONTINUOUS_FIELDS = (
    "hendecasyllable_lines",
    "failed_lines",
    "uncertain_lines",
    "rhyme_score",
)
BINARY_FIELDS = ("quatrain_ok", "tercet_ok", "all_lines_valid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-dir",
        type=Path,
        default=ROOT / "artifacts/local/plan_then_poem/validation/generation",
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
        default=ROOT / "reports/plan_then_poem_validation_v1.md",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=ROOT / "reports/plan_then_poem_validation_v1.json",
    )
    parser.add_argument(
        "--scores-jsonl",
        type=Path,
        default=ROOT / "artifacts/local/plan_then_poem/validation/scores_v1.jsonl",
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
    comparisons: dict,
    reading: dict,
) -> str:
    lines = [
        "# Plan-Then-Poem Inference Test v1",
        "",
        "Date: 2026-09-13",
        "",
        "The planned condition appends a numbered list of pre-committed line",
        "endings to the frozen prompt. The format control appends the same list",
        "shape with non-rhyming placeholder endings. The baseline is the",
        "verifier-DPO no-plan output. All systems use the same openings, seeds,",
        "prompt builder, and recipe; the sealed test set was not accessed.",
        "The checker is reviewed and measures form only.",
        "",
        "## Plan adherence",
        "",
        "| Metric | Planned | Control |",
        "|---|---:|---:|",
        f"| Word match rate | {adherence_metrics['planned_word_rate']:.4f} | {adherence_metrics['control_word_rate']:.4f} |",
        f"| Key match rate | {adherence_metrics['planned_key_rate']:.4f} | {adherence_metrics['control_key_rate']:.4f} |",
        f"| Planned scheme compliance | {adherence_metrics['planned_scheme_rate']:.4f} | {adherence_metrics['control_scheme_rate']:.4f} |",
        "",
        "## Form metrics",
        "",
        "| Metric | Planned | Control | Baseline |",
        "|---|---:|---:|---:|",
        f"| Accepted lines (mean) | {metrics[PLANNED]['mean_hendecasyllable_lines']:.3f} | {metrics[CONTROL]['mean_hendecasyllable_lines']:.3f} | {metrics[BASELINE]['mean_hendecasyllable_lines']:.3f} |",
        f"| Failed lines (mean) | {metrics[PLANNED]['mean_failed_lines']:.3f} | {metrics[CONTROL]['mean_failed_lines']:.3f} | {metrics[BASELINE]['mean_failed_lines']:.3f} |",
        f"| Uncertain lines (mean) | {metrics[PLANNED]['mean_uncertain_lines']:.3f} | {metrics[CONTROL]['mean_uncertain_lines']:.3f} | {metrics[BASELINE]['mean_uncertain_lines']:.3f} |",
        f"| Rhyme score (mean) | {metrics[PLANNED]['mean_rhyme_score']:.4f} | {metrics[CONTROL]['mean_rhyme_score']:.4f} | {metrics[BASELINE]['mean_rhyme_score']:.4f} |",
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
            "- Form and adherence only. No coherence or literary-quality claim.",
            "- Adherence can be met with odd or archaic endings; the adherence",
            "  rate does not measure meaning.",
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
        record.update(adherence(str(record["text"]), record["planned_words"]))
        record["plan_scheme_ok"] = None
    scored = score_records(plan_records + baseline_records)
    for record in scored:
        if record["system_id"] in (PLANNED, CONTROL):
            record["plan_scheme_ok"] = bool(
                record["rhyme_scheme"] == DEFAULT_SCHEME
                and record["quatrain_ok"]
                and record["tercet_ok"]
            )

    def rates(system: str, field: str) -> list[float]:
        return [
            float(record[field])
            for record in scored
            if record["system_id"] == system and record.get(field) is not None
        ]

    adherence_metrics = {
        "planned_word_rate": mean(rates(PLANNED, "word_match_rate")) or 0.0,
        "planned_key_rate": mean(rates(PLANNED, "key_match_rate")) or 0.0,
        "control_word_rate": mean(rates(CONTROL, "word_match_rate")) or 0.0,
        "control_key_rate": mean(rates(CONTROL, "key_match_rate")) or 0.0,
        "planned_scheme_rate": mean(rates(PLANNED, "plan_scheme_ok")) or 0.0,
        "control_scheme_rate": mean(rates(CONTROL, "plan_scheme_ok")) or 0.0,
    }
    metrics = {system: aggregate(scored, system) for system in SYSTEMS}
    comparisons = []
    for label, systems in (
        ("planned - baseline", (PLANNED, BASELINE)),
        ("control - baseline", (CONTROL, BASELINE)),
        ("planned - control", (PLANNED, CONTROL)),
    ):
        pairs = pair_records(scored, systems=systems)
        for field in CONTINUOUS_FIELDS:
            comparisons.append((label, field, paired_comparison(pairs, field)))

    reading_reasons = []
    key_rate = adherence_metrics["planned_key_rate"]
    scheme_rate = adherence_metrics["planned_scheme_rate"]
    if key_rate >= 0.40 and scheme_rate > 0:
        result = "PROCEED_TO_SFT_PROPOSAL"
        reading_reasons.append(
            "Key-match adherence is at least 40% and planned scheme compliance "
            "exceeds zero."
        )
    elif key_rate < 0.20:
        result = "TRAIN_PLAN_FOLLOWING_FIRST"
        reading_reasons.append(
            "Key-match adherence is below 20%; the model cannot follow plans "
            "under this prompt."
        )
    else:
        result = "MECHANICAL_AUDIT_AND_SMALL_SFT"
        reading_reasons.append(
            "Adherence is between the gates or high without scheme compliance; "
            "audit the format and propose a small plan-following SFT arm."
        )
    reading_reasons.append(
        f"Observed key-match adherence {key_rate:.4f}; planned scheme "
        f"compliance {scheme_rate:.4f}."
    )

    args.scores_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_jsonl.open("w", encoding="utf-8") as handle:
        for record in scored:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    report_markdown = format_report(
        adherence_metrics=adherence_metrics,
        metrics=metrics,
        comparisons=comparisons,
        reading={"result": result, "reasons": reading_reasons},
    )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(report_markdown, encoding="utf-8")

    payload = {
        "schema_version": 1,
        "generation_dir": repository_relative(args.generation_dir),
        "baseline_dir": repository_relative(args.baseline_dir),
        "adherence": adherence_metrics,
        "system_metrics": metrics,
        "paired_comparisons": [
            {"comparison": label, "field": field, **row}
            for label, field, row in comparisons
        ],
        "reading": {"result": result, "reasons": reading_reasons},
        "scores_jsonl": repository_relative(args.scores_jsonl),
        "checker_status": "reviewed_conservative_2026-09-13",
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "plan-then-poem-score | "
        f"planned_key_rate={adherence_metrics['planned_key_rate']:.4f} "
        f"control_key_rate={adherence_metrics['control_key_rate']:.4f} "
        f"planned_scheme_rate={adherence_metrics['planned_scheme_rate']:.4f} "
        f"reading={result}"
    )


if __name__ == "__main__":
    main()
