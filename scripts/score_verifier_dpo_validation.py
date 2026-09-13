#!/usr/bin/env python3
"""Score the verifier-DPO matched validation grid against Stage-3."""

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
from sonnet_evaluation.sonnet_prosody_sealed import (
    aggregate,
    mcnemar,
    paired_comparison,
    pair_records,
    score_records,
)

PAIR_ORDER = ("dpo", "stage_3")
BASELINE = "stage_3"
CANDIDATE = "dpo"
CONTINUOUS_FIELDS = (
    "hendecasyllable_lines",
    "failed_lines",
    "uncertain_lines",
    "perfect_rhyme_pairs",
    "soft_rhyme_pairs",
    "rhyme_score",
)
BINARY_FIELDS = (
    "structure_ok",
    "stanza_pattern_ok",
    "all_lines_valid",
    "quatrain_ok",
    "tercet_ok",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-dir",
        type=Path,
        default=ROOT / "artifacts/local/verifier_labelled_dpo/validation/generation",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=ROOT / "reports/verifier_labelled_dpo_v1.md",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=ROOT / "reports/verifier_labelled_dpo_v1.json",
    )
    parser.add_argument(
        "--scores-jsonl",
        type=Path,
        default=ROOT / "artifacts/local/verifier_labelled_dpo/validation/scores_v1.jsonl",
    )
    return parser.parse_args()


def repository_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return "<outside repository>"


def full_form_valid_flags(records: list[dict]) -> None:
    for record in records:
        record["full_form_valid"] = bool(
            record["all_lines_valid"]
            and record["quatrain_ok"]
            and record["tercet_ok"]
        )


def count_true(records: list[dict], system_id: str, field: str) -> int:
    return sum(
        1 for record in records if record["system_id"] == system_id and record[field]
    )


def format_report(
    *,
    baseline: dict,
    candidate: dict,
    comparisons: list[dict],
    mcnemar_rows: list[dict],
    baseline_valid: int,
    candidate_valid: int,
    reading: dict,
) -> str:
    lines = [
        "# Verifier-Labelled Form DPO v1",
        "",
        "Date: 2026-09-13",
        "",
        "Matched validation comparison of the published Stage-3 baseline and the",
        "verifier-labelled form DPO adapter. The 120 validation openings, seeds,",
        "prompt builder, and recipe are identical across systems; the sealed test",
        "set was not accessed. The checker is reviewed and measures form only,",
        "not literary quality.",
        "",
        "## Per-system form metrics",
        "",
        "| Metric | Stage 3 | Verifier DPO |",
        "|---|---:|---:|",
        f"| Outputs | {baseline['output_count']} | {candidate['output_count']} |",
        f"| Structure OK rate | {baseline['structure_ok_rate']:.4f} | {candidate['structure_ok_rate']:.4f} |",
        f"| Stanza pattern OK rate | {baseline['stanza_pattern_ok_rate']:.4f} | {candidate['stanza_pattern_ok_rate']:.4f} |",
        f"| All 14 lines valid rate | {baseline['all_lines_valid_rate']:.4f} | {candidate['all_lines_valid_rate']:.4f} |",
        f"| Quatrain scheme OK rate | {baseline['quatrain_ok_rate']:.4f} | {candidate['quatrain_ok_rate']:.4f} |",
        f"| Tercet scheme OK rate | {baseline['tercet_ok_rate']:.4f} | {candidate['tercet_ok_rate']:.4f} |",
        f"| Mean hendecasyllable lines | {baseline['mean_hendecasyllable_lines']:.3f} | {candidate['mean_hendecasyllable_lines']:.3f} |",
        f"| Mean failed lines | {baseline['mean_failed_lines']:.3f} | {candidate['mean_failed_lines']:.3f} |",
        f"| Mean uncertain lines | {baseline['mean_uncertain_lines']:.3f} | {candidate['mean_uncertain_lines']:.3f} |",
        f"| Mean rhyme score | {baseline['mean_rhyme_score']:.4f} | {candidate['mean_rhyme_score']:.4f} |",
        "",
        f"- Full-form valid outputs: Stage 3 {baseline_valid}, verifier DPO {candidate_valid}",
        "",
        "## Paired comparisons (verifier DPO minus Stage 3)",
        "",
        "| Field | Mean difference | 95% CI | + pairs | - pairs | = pairs |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['field']} | {row['mean_difference_stage_3_minus_dpo']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
            f"{row['positive_pairs']} | {row['negative_pairs']} | {row['equal_pairs']} |"
        )
    lines.extend(
        [
            "",
            "The shared comparison helper labels the first paired system `stage_3`",
            "and the second `dpo`; here the first is the verifier DPO adapter and",
            "the second is the Stage-3 baseline, so the table is DPO minus Stage 3.",
            "",
            "## Discordant binary outcomes (McNemar normal approximation)",
            "",
            "| Field | DPO only | Stage 3 only | Both | Neither | p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in mcnemar_rows:
        lines.append(
            f"| {row['field']} | {row['stage_3_only']} | {row['dpo_only']} | "
            f"{row['both']} | {row['neither']} | {row['p_value_normal_approximation']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Pre-registered reading",
            "",
            f"- Result: {'SIGNAL' if reading['signal'] else 'NULL'}",
        ]
    )
    for reason in reading["reasons"]:
        lines.append(f"- {reason}")
    failed = next(
        (row for row in comparisons if row["field"] == "failed_lines"), None
    )
    uncertain = next(
        (row for row in comparisons if row["field"] == "uncertain_lines"), None
    )
    if failed is not None and uncertain is not None:
        lines.append(
            f"- Secondary movement: failed lines "
            f"{failed['mean_difference_stage_3_minus_dpo']:+.3f} "
            f"[{failed['ci95_low']:.3f}, {failed['ci95_high']:.3f}], uncertain "
            f"lines {uncertain['mean_difference_stage_3_minus_dpo']:+.3f} "
            f"[{uncertain['ci95_low']:.3f}, {uncertain['ci95_high']:.3f}]."
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Form only. No literary-quality or selection-for-quality claim.",
            "- The checker's definite coverage limit (87.1% on the ground truth)",
            "  applies to the metre counts.",
            "- DPO on surface form can trade against coherence; this report does",
            "  not measure coherence or grammar.",
            "",
            "Verification: `python3 scripts/score_verifier_dpo_validation.py`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    records = load_generation_records(args.generation_dir)
    scored = score_records(records)
    full_form_valid_flags(scored)
    pairs = pair_records(scored, systems=PAIR_ORDER)
    baseline = aggregate(scored, BASELINE)
    candidate = aggregate(scored, CANDIDATE)
    comparisons = [paired_comparison(pairs, field) for field in CONTINUOUS_FIELDS]
    mcnemar_rows = [mcnemar(pairs, field) for field in BINARY_FIELDS]
    baseline_valid = count_true(scored, BASELINE, "full_form_valid")
    candidate_valid = count_true(scored, CANDIDATE, "full_form_valid")

    accepted = next(
        row for row in comparisons if row["field"] == "hendecasyllable_lines"
    )
    rhyme = next(row for row in comparisons if row["field"] == "rhyme_score")
    reasons = []
    signal = False
    if candidate_valid > 0 and baseline_valid == 0:
        signal = True
        reasons.append(
            "The verifier DPO produced full-form valid outputs where Stage 3 had none."
        )
    if (
        accepted["ci95_low"] > 0
        and accepted["mean_difference_stage_3_minus_dpo"] >= 0.23
        and rhyme["ci95_high"] >= 0
    ):
        signal = True
        reasons.append(
            "Paired accepted-line increase excludes zero and reaches the sealed DPO "
            "effect, with no significant rhyme-score drop."
        )
    if not signal:
        reasons.append(
            "No pre-registered signal: no valid DPO output and no accepted-line "
            "increase that excludes zero and reaches 0.23 without a significant "
            "rhyme-score drop."
        )

    args.scores_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_jsonl.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report_markdown = format_report(
        baseline=baseline,
        candidate=candidate,
        comparisons=comparisons,
        mcnemar_rows=mcnemar_rows,
        baseline_valid=baseline_valid,
        candidate_valid=candidate_valid,
        reading={"signal": signal, "reasons": reasons},
    )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(report_markdown, encoding="utf-8")

    payload = {
        "schema_version": 1,
        "generation_dir": repository_relative(args.generation_dir),
        "paired_order": list(PAIR_ORDER),
        "system_metrics": {BASELINE: baseline, CANDIDATE: candidate},
        "full_form_valid": {BASELINE: baseline_valid, CANDIDATE: candidate_valid},
        "paired_comparisons": comparisons,
        "mcnemar": mcnemar_rows,
        "reading": {"signal": signal, "reasons": reasons},
        "scores_jsonl": repository_relative(args.scores_jsonl),
        "checker_status": "reviewed_conservative_2026-09-13",
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "verifier-dpo-score | "
        f"pairs={len(pairs)} stage_3_valid={baseline_valid} "
        f"dpo_valid={candidate_valid} "
        f"accepted_diff={accepted['mean_difference_stage_3_minus_dpo']:.3f} "
        f"[{accepted['ci95_low']:.3f}, {accepted['ci95_high']:.3f}] "
        f"rhyme_diff={rhyme['mean_difference_stage_3_minus_dpo']:.4f} "
        f"[{rhyme['ci95_low']:.4f}, {rhyme['ci95_high']:.4f}] "
        f"reading={'SIGNAL' if signal else 'NULL'}"
    )


if __name__ == "__main__":
    main()
