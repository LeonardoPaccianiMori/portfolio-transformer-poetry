#!/usr/bin/env python3
"""Score the reasoning-trace arms: trace validity and poem form."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.few_shot_context_validation import load_context_records
from sonnet_analysis.minerva_v7_exploratory_prompts import (
    validate_exploratory_prompt_manifest,
)
from sonnet_evaluation.coherence import coherence_proxies
from sonnet_evaluation.corpus_memorization import memorization_screen
from sonnet_evaluation.sonnet_prosody_sealed import score_records
from sonnet_training.form_targeted_data import load_train_rows, read_sonnet_text
from sonnet_training.reasoning_trace_sft import (
    TRACE_ARMS,
    parse_trace,
    validate_trace,
)
from sonnet_training.self_play_rft import self_play_reading

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
        default=ROOT / "artifacts/local/reasoning_trace/validation/generation",
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=ROOT / "data/processed/sonnets_expanded_v8/sonnets_manifest.csv",
    )
    parser.add_argument(
        "--judge-json",
        type=Path,
        default=ROOT / "artifacts/local/reasoning_trace/coherence_judge_v1.json",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=ROOT / "reports/reasoning_trace_sft_v1.md",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=ROOT / "reports/reasoning_trace_sft_v1.json",
    )
    parser.add_argument(
        "--scores-jsonl",
        type=Path,
        default=ROOT / "artifacts/local/reasoning_trace/validation/scores_v1.jsonl",
    )
    return parser.parse_args()


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def format_value(value: float | None, fmt: str) -> str:
    return "n/a" if value is None else fmt.format(value)


def main() -> None:
    args = parse_args()
    args.generation_dir = args.generation_dir.resolve()
    args.corpus_manifest = args.corpus_manifest.resolve()
    args.judge_json = args.judge_json.resolve()
    args.report_md = args.report_md.resolve()
    args.report_json = args.report_json.resolve()
    args.scores_jsonl = args.scores_jsonl.resolve()
    prompts = validate_exploratory_prompt_manifest(
        ROOT / "configs/minerva_7b_v7_exploratory_prompts.json",
        expected_sha256="2f33aa518aa61c11193831e53b07fd3bd861a72bf68bb23c0e0e5b1a13b1d0c7",
    )["prompts"]
    opening_by_id = {str(prompt["id"]): str(prompt["opening_line"]) for prompt in prompts}
    records = load_context_records(args.generation_dir)
    poems = []
    trace_rows = []
    for record in records:
        opening = opening_by_id[str(record["prompt_id"])]
        parsed = parse_trace(str(record["text"]))
        if parsed is None:
            poem_text = str(record["text"])
            trace = {
                "well_formed": False,
                "schema_valid": False,
                "rhyme_consistent": False,
                "valid": False,
                "canonical_scheme": None,
            }
        else:
            poem_text = str(parsed["poem_text"])
            trace = validate_trace(parsed, opening)
        poems.append(
            {
                "path": record["path"],
                "prompt_id": record["prompt_id"],
                "seed": record["seed"],
                "system_id": record["system_id"],
                "text": poem_text,
            }
        )
        trace_rows.append(
            {
                "path": record["path"],
                "system_id": record["system_id"],
                "parsed": parsed is not None,
                "trace_valid": bool(trace["valid"]),
                "trace_schema_valid": bool(trace["schema_valid"]),
                "trace_scheme": trace["canonical_scheme"],
            }
        )
    scored = score_records(poems)
    for row in scored:
        row["valid_without_repair"] = bool(row["quatrain_ok"] and row["tercet_ok"])
    trace_by_path = {row["path"]: row for row in trace_rows}
    for row in scored:
        trace = trace_by_path[row["path"]]
        row.update(trace)
        row["scheme_matches_trace"] = bool(
            trace["parsed"]
            and trace["trace_scheme"] is not None
            and row["rhyme_scheme"] == trace["trace_scheme"]
        )
    for record in poems:
        record.update(
            {
                f"proxy_{key}": value
                for key, value in coherence_proxies(str(record["text"])).items()
            }
        )

    def arm_rows(arm: str) -> list[dict]:
        return [row for row in scored if row["system_id"] == arm]

    def record_rows(arm: str) -> list[dict]:
        return [row for row in poems if row["system_id"] == arm]

    def field_values(rows: list[dict], field: str) -> list[float]:
        return [float(row[field]) for row in rows if row.get(field) is not None]

    form = {}
    for arm in TRACE_ARMS:
        rows = arm_rows(arm)
        if not rows:
            form[arm] = {
                "outputs": 0,
                "trace_parsed": 0.0,
                "trace_valid": 0.0,
                "valid_without_repair": 0.0,
                "scheme_matches_trace": 0.0,
                "accepted_lines": 0.0,
                "failed_lines": 0.0,
                "uncertain_lines": 0.0,
                "stanza_pattern_ok": 0.0,
                "rhyme_score": 0.0,
                "proxies": {field: 0.0 for field in PROXY_FIELDS},
            }
            continue
        form[arm] = {
            "outputs": len(rows),
            "trace_parsed": mean([float(row["parsed"]) for row in rows]),
            "trace_valid": mean([float(row["trace_valid"]) for row in rows]),
            "valid_without_repair": mean(
                [float(row["valid_without_repair"]) for row in rows]
            ),
            "scheme_matches_trace": mean(
                [float(row["scheme_matches_trace"]) for row in rows]
            ),
            "accepted_lines": mean(field_values(rows, "hendecasyllable_lines")),
            "failed_lines": mean(field_values(rows, "failed_lines")),
            "uncertain_lines": mean(field_values(rows, "uncertain_lines")),
            "stanza_pattern_ok": mean(field_values(rows, "stanza_pattern_ok")),
            "rhyme_score": mean(field_values(rows, "rhyme_score")),
            "proxies": {
                field: mean(
                    [
                        float(record[f"proxy_{field}"])
                        for record in record_rows(arm)
                        if record.get(f"proxy_{field}") is not None
                    ]
                )
                for field in PROXY_FIELDS
            },
        }

    corpus_rows = load_train_rows(args.corpus_manifest)
    screen = memorization_screen(
        [
            {
                "system_id": record["system_id"],
                "text": record["text"],
            }
            for record in poems
        ],
        (read_sonnet_text(ROOT, row) for row in corpus_rows),
    )

    judge_summary = None
    judge_gain = None
    if args.judge_json.is_file():
        payload = json.loads(args.judge_json.read_text(encoding="utf-8"))
        judge_summary = {}
        for judge, data in payload["judges"].items():
            condition_means = {}
            for arm, arm_scores in data["conditions"].items():
                scores = [
                    row["scores"]["mean"]
                    for row in arm_scores
                    if row.get("scores") is not None
                ]
                condition_means[arm] = mean(scores)
            judge_summary[judge] = {
                "condition_means": condition_means,
                "calibration_separation": data["calibration"]["separation"],
            }
        gains = []
        for data in judge_summary.values():
            baseline = data["condition_means"].get(TRACE_ARMS[0])
            trained = data["condition_means"].get(TRACE_ARMS[1])
            if baseline is not None and trained is not None:
                gains.append(trained - baseline)
        judge_gain = mean(gains)

    reading = self_play_reading(
        baseline_valid=float(form[TRACE_ARMS[0]]["valid_without_repair"]),
        rft_valid=float(form[TRACE_ARMS[1]]["valid_without_repair"]),
        baseline_accepted=float(form[TRACE_ARMS[0]]["accepted_lines"]),
        rft_accepted=float(form[TRACE_ARMS[1]]["accepted_lines"]),
        judge_gain=judge_gain,
    )

    args.scores_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_jsonl.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Reasoning-Trace SFT v1",
        "",
        f"Date: {datetime.date.today().isoformat()}",
        "",
        "The model writes a rhyme plan (scheme plus one ending per line) and",
        "then the sonnet. Traces are derived from corpus poems; the evaluation",
        "arms use the same openings, seeds, and recipe. Metrics exclude repair.",
        "",
        "## Trace and form",
        "",
        "| Metric | Baseline | Trace SFT |",
        "|---|---:|---:|",
    ]
    for key, label, fmt in (
        ("trace_parsed", "Outputs with a parsed trace", "{:.4f}"),
        ("trace_valid", "Valid traces (scheme and true rhymes)", "{:.4f}"),
        ("valid_without_repair", "Scheme valid without repair", "{:.4f}"),
        ("scheme_matches_trace", "Poem scheme equals traced scheme", "{:.4f}"),
        ("accepted_lines", "Accepted lines", "{:.3f}"),
        ("failed_lines", "Failed lines", "{:.3f}"),
        ("uncertain_lines", "Uncertain lines", "{:.3f}"),
        ("stanza_pattern_ok", "4+4+3+3 stanza pattern", "{:.4f}"),
        ("rhyme_score", "Rhyme score", "{:.4f}"),
    ):
        values = [form[arm][key] for arm in TRACE_ARMS]
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(format_value(value, fmt) for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Coherence proxies",
            "",
            "| Metric | Baseline | Trace SFT |",
            "|---|---:|---:|",
        ]
    )
    for field in PROXY_FIELDS:
        values = [form[arm]["proxies"][field] for arm in TRACE_ARMS]
        lines.append(
            f"| {field} | "
            + " | ".join(format_value(value, "{:.4f}") for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Memorization screen",
            "",
            f"Verbatim overlap with {screen['corpus_documents']:,} training "
            f"sonnets ({screen['corpus_lines']:,} lines, "
            f"{screen['shingle_size']}-word shingles). Poem text only.",
            "",
            "| Metric | Baseline | Trace SFT |",
            "|---|---:|---:|",
        ]
    )
    baseline_stats = screen["conditions"].get(TRACE_ARMS[0], {})
    trained_stats = screen["conditions"].get(TRACE_ARMS[1], {})
    for label, key, fmt in (
        ("Outputs with a copied line", "outputs_with_exact_line", "{:d}"),
        ("Max copied lines in one output", "max_exact_lines", "{:d}"),
        ("Mean 5-gram overlap", "mean_shingle_hit", "{:.4f}"),
    ):
        values = [baseline_stats.get(key, 0), trained_stats.get(key, 0)]
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(format_value(value, fmt) for value in values)
            + " |"
        )
    if judge_summary:
        lines.extend(
            [
                "",
                "## Judge panel",
                "",
                "| Judge | Baseline | Trace SFT | Separation |",
                "|---|---:|---:|---:|",
            ]
        )
        for judge, data in judge_summary.items():
            means = data["condition_means"]
            lines.append(
                f"| {judge} | {means.get(TRACE_ARMS[0])} | "
                f"{means.get(TRACE_ARMS[1])} | {data['calibration_separation']} |"
            )
    lines.extend(["", "## Pre-registered reading", "", f"- Result: {reading['result']}"])
    for reason in reading["reasons"]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Traces come from corpus poems, so the poems are learned imitations",
            "  of real texts, not new compositions.",
            "- The gate uses the poem scheme without repair; trace validity is",
            "  reported separately.",
            "",
            "Verification: `python3 scripts/score_reasoning_trace_sft.py`",
            "",
        ]
    )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "schema_version": 1,
        "generation_dir": str(args.generation_dir.relative_to(ROOT)),
        "form": form,
        "memorization_screen": screen,
        "judge_summary": judge_summary,
        "judge_gain": judge_gain,
        "reading": reading,
        "scores_jsonl": str(args.scores_jsonl.relative_to(ROOT)),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "reasoning-trace-score | "
        f"baseline_valid={format_value(form[TRACE_ARMS[0]]['valid_without_repair'], '{:.4f}')} "
        f"trace_valid={format_value(form[TRACE_ARMS[1]]['valid_without_repair'], '{:.4f}')} "
        f"trace_valid_rate={format_value(form[TRACE_ARMS[1]]['trace_valid'], '{:.4f}')} "
        f"judge_gain={judge_gain} reading={reading['result']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
