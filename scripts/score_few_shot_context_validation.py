#!/usr/bin/env python3
"""Score the few-shot context A/B with form, proxies, and judge results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.constrained_plan_validation import repair_endings
from sonnet_analysis.few_shot_context_validation import (
    load_context_records,
    memorization_screen,
)
from sonnet_analysis.plan_then_poem_validation import DEFAULT_SCHEME, adherence
from sonnet_evaluation.coherence import coherence_proxies
from sonnet_evaluation.sonnet_prosody_sealed import (
    aggregate,
    paired_comparison,
    pair_records,
    score_records,
)
from sonnet_training.form_targeted_data import load_train_rows, read_sonnet_text

ZERO = "zeroshot"
CONDITIONS = (ZERO, "fewshot1", "fewshot3")
CONTINUOUS_FIELDS = ("hendecasyllable_lines", "failed_lines", "uncertain_lines", "rhyme_score")
PROXY_FIELDS = (
    "type_token_ratio",
    "repeated_line_ratio",
    "repeated_bigram_ratio",
    "final_word_repetition",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-dir",
        type=Path,
        default=ROOT / "artifacts/local/few_shot_context/validation/generation",
    )
    parser.add_argument(
        "--judge-json",
        type=Path,
        default=ROOT / "artifacts/local/few_shot_context/coherence_judge_v1.json",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=ROOT / "reports/few_shot_context_validation_v1.md",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=ROOT / "reports/few_shot_context_validation_v1.json",
    )
    parser.add_argument(
        "--scores-jsonl",
        type=Path,
        default=ROOT / "artifacts/local/few_shot_context/validation/scores_v1.jsonl",
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=ROOT / "data/processed/sonnets_expanded_v8/sonnets_manifest.csv",
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
    records = load_context_records(args.generation_dir)
    original_records = []
    repaired_records = []
    for record in records:
        metrics = adherence(str(record["text"]), record["planned_words"])
        record.update(metrics)
        repaired_text, repair = repair_endings(
            str(record["text"]), record["planned_words"]
        )
        record["repair_distance"] = repair["repair_distance"]
        record.update(
            {
                f"proxy_{key}": value
                for key, value in coherence_proxies(str(record["text"])).items()
            }
        )
        original_records.append(record)
        repaired_records.append({**record, "text": repaired_text})

    original_scored = score_records(
        [
            {
                "path": record["path"],
                "prompt_id": record["prompt_id"],
                "seed": record["seed"],
                "system_id": record["system_id"],
                "text": record["text"],
            }
            for record in original_records
        ]
    )
    repaired_scored = score_records(
        [
            {
                "path": record["path"],
                "prompt_id": record["prompt_id"],
                "seed": record["seed"],
                "system_id": record["system_id"],
                "text": record["text"],
            }
            for record in repaired_records
        ]
    )
    for rows in (original_scored, repaired_scored):
        for row in rows:
            row["plan_scheme_ok"] = bool(
                row["rhyme_scheme"] == DEFAULT_SCHEME
                and row["quatrain_ok"]
                and row["tercet_ok"]
            )

    def values(rows, system: str, field: str) -> list[float]:
        return [
            float(row[field])
            for row in rows
            if row["system_id"] == system and row.get(field) is not None
        ]

    form = {}
    for condition in CONDITIONS:
        form[condition] = {
            "original_scheme": mean(values(original_scored, condition, "plan_scheme_ok")) or 0.0,
            "repaired_scheme": mean(values(repaired_scored, condition, "plan_scheme_ok")) or 0.0,
            "key_match": mean(values(original_records, condition, "key_match_rate")) or 0.0,
            "repair_distance": mean(values(original_records, condition, "repair_distance")) or 0.0,
            "metrics": aggregate(repaired_scored, condition),
        }
        form[condition]["proxies"] = {
            field: mean(values(original_records, condition, f"proxy_{field}")) or 0.0
            for field in PROXY_FIELDS
        }

    judge_summary = None
    if args.judge_json.is_file():
        payload = json.loads(args.judge_json.read_text(encoding="utf-8"))
        judge_summary = {}
        for judge, data in payload["judges"].items():
            condition_means = {}
            for condition, rows in data["conditions"].items():
                scores = [
                    row["scores"]["mean"]
                    for row in rows
                    if row.get("scores") is not None
                ]
                condition_means[condition] = mean(scores)
            judge_summary[judge] = {
                "condition_means": condition_means,
                "calibration_separation": data["calibration"]["separation"],
            }

    corpus_rows = load_train_rows(args.corpus_manifest)
    screen = memorization_screen(
        records,
        (read_sonnet_text(ROOT, row) for row in corpus_rows),
    )

    best_condition = max(
        CONDITIONS[1:], key=lambda condition: form[condition]["repaired_scheme"]
    )
    reference = form[ZERO]
    accepted_gap = (
        form[best_condition]["metrics"]["mean_hendecasyllable_lines"]
        - reference["metrics"]["mean_hendecasyllable_lines"]
    )
    judge_gain = None
    if judge_summary:
        gains = []
        for judge, data in judge_summary.items():
            zero = data["condition_means"].get(ZERO)
            best = data["condition_means"].get(best_condition)
            if zero is not None and best is not None:
                gains.append(best - zero)
        judge_gain = mean(gains)
    reasons = []
    if (
        form[best_condition]["repaired_scheme"] >= 0.95
        and accepted_gap >= -0.5
        and judge_gain is not None
        and judge_gain >= 0.3
    ):
        reading = "CONTEXT_HELPS"
        reasons.append(
            "The best few-shot condition keeps the repaired scheme at 95% or "
            "more and raises the judge mean by at least 0.3."
        )
    else:
        reading = "CONTEXT_NULL"
        reasons.append(
            "No few-shot condition improves the judge mean by 0.3 or more "
            "while keeping form."
        )
    reasons.append(
        f"Best condition {best_condition}; repaired scheme "
        f"{form[best_condition]['repaired_scheme']:.4f}; accepted gap "
        f"{accepted_gap:+.3f}; judge gain {judge_gain}."
    )
    reasons.append(
        "Memorization screen: max copied corpus lines "
        f"{max(screen['conditions'][c]['max_exact_lines'] for c in CONDITIONS)}; "
        "max mean 5-gram overlap "
        f"{max(screen['conditions'][c]['mean_shingle_hit'] for c in CONDITIONS):.4f}."
    )

    args.scores_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_jsonl.open("w", encoding="utf-8") as handle:
        for row in repaired_scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Few-Shot Context A/B v1",
        "",
        "Date: 2026-09-13",
        "",
        "Zero, one, and three complete plan-plus-sonnet examples in the chat",
        "context. Same anchored plans, opening, seed, and recipe. Repaired",
        "scheme is the guaranteed-form condition. Judge means come from the",
        "calibrated panel; form-only claims otherwise.",
        "",
        "## Form",
        "",
        "| Metric | Zero | One | Three |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("original_scheme", "Scheme before repair"),
        ("repaired_scheme", "Scheme after repair"),
        ("key_match", "Key-match adherence"),
        ("repair_distance", "Mean repair distance"),
    ):
        lines.append(
            f"| {label} | {form[CONDITIONS[0]][key]:.4f} | "
            f"{form[CONDITIONS[1]][key]:.4f} | {form[CONDITIONS[2]][key]:.4f} |"
        )
    lines.extend(
        [
            f"| Accepted lines | {form[CONDITIONS[0]]['metrics']['mean_hendecasyllable_lines']:.3f} | "
            f"{form[CONDITIONS[1]]['metrics']['mean_hendecasyllable_lines']:.3f} | "
            f"{form[CONDITIONS[2]]['metrics']['mean_hendecasyllable_lines']:.3f} |",
            "",
            "## Coherence proxies",
            "",
            "| Metric | Zero | One | Three |",
            "|---|---:|---:|---:|",
        ]
    )
    for field in PROXY_FIELDS:
        lines.append(
            f"| {field} | {form[CONDITIONS[0]]['proxies'][field]:.4f} | "
            f"{form[CONDITIONS[1]]['proxies'][field]:.4f} | "
            f"{form[CONDITIONS[2]]['proxies'][field]:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Memorization screen",
            "",
            f"Verbatim overlap with {screen['corpus_documents']:,} training "
            f"sonnets ({screen['corpus_lines']:,} lines, "
            f"{screen['shingle_size']}-word shingles). Opening line excluded.",
            "",
            "| Metric | Zero | One | Three |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, key, fmt in (
        ("Outputs with a copied line", "outputs_with_exact_line", "{:d}"),
        ("Outputs with 4+ copied lines", "outputs_with_four_exact_lines", "{:d}"),
        ("Max copied lines in one output", "max_exact_lines", "{:d}"),
        ("Mean 5-gram overlap", "mean_shingle_hit", "{:.4f}"),
    ):
        values = [screen["conditions"][condition][key] for condition in CONDITIONS]
        lines.append("| " + label + " | " + " | ".join(fmt.format(v) for v in values) + " |")
    if judge_summary:
        lines.extend(["", "## Judge panel", "", "| Judge | Zero | One | Three | Separation |", "|---|---:|---:|---:|---:|"])
        for judge, data in judge_summary.items():
            means = data["condition_means"]
            lines.append(
                f"| {judge} | {means.get(ZERO)} | {means.get('fewshot1')} | "
                f"{means.get('fewshot3')} | {data['calibration_separation']} |"
            )
    lines.extend(["", "## Pre-registered reading", "", f"- Result: {reading}"])
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Repair can break grammar and meaning; the judge panel is the",
            "  coherence evidence, not the checker.",
            "- Judge calibration uses deterministic corruptions of corpus",
            "  sonnets; separation near zero weakens the panel.",
            "",
            "Verification: `python3 scripts/score_few_shot_context_validation.py`",
            "",
        ]
    )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "schema_version": 1,
        "generation_dir": repository_relative(args.generation_dir),
        "form": form,
        "judge_summary": judge_summary,
        "memorization_screen": screen,
        "reading": {"result": reading, "reasons": reasons},
        "scores_jsonl": repository_relative(args.scores_jsonl),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"few-shot-score | best={best_condition} "
        f"repaired_scheme={form[best_condition]['repaired_scheme']:.4f} "
        f"accepted_gap={accepted_gap:+.3f} judge_gain={judge_gain} "
        f"reading={reading}"
    )


if __name__ == "__main__":
    main()
