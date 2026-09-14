#!/usr/bin/env python3
"""Score the self-play RFT arms without repair and read the gate."""

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
from sonnet_evaluation.coherence import coherence_proxies
from sonnet_evaluation.corpus_memorization import memorization_screen
from sonnet_evaluation.sonnet_prosody_sealed import score_records
from sonnet_training.form_targeted_data import load_train_rows, read_sonnet_text
from sonnet_training.self_play_rft import ARMS, self_play_reading

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
        default=ROOT / "artifacts/local/self_play_rft/validation/generation",
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=ROOT / "data/processed/sonnets_expanded_v8/sonnets_manifest.csv",
    )
    parser.add_argument(
        "--judge-json",
        type=Path,
        default=ROOT / "artifacts/local/self_play_rft/coherence_judge_v1.json",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=ROOT / "reports/self_play_rft_v1.md",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=ROOT / "reports/self_play_rft_v1.json",
    )
    parser.add_argument(
        "--scores-jsonl",
        type=Path,
        default=ROOT / "artifacts/local/self_play_rft/validation/scores_v1.jsonl",
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
    records = load_context_records(args.generation_dir)
    scored = score_records(
        [
            {
                "path": record["path"],
                "prompt_id": record["prompt_id"],
                "seed": record["seed"],
                "system_id": record["system_id"],
                "text": record["text"],
            }
            for record in records
        ]
    )
    for row in scored:
        row["valid_without_repair"] = bool(row["quatrain_ok"] and row["tercet_ok"])
    for record in records:
        record.update(
            {
                f"proxy_{key}": value
                for key, value in coherence_proxies(str(record["text"])).items()
            }
        )

    def arm_rows(arm: str) -> list[dict]:
        return [row for row in scored if row["system_id"] == arm]

    def record_rows(arm: str) -> list[dict]:
        return [row for row in records if row["system_id"] == arm]

    def field_values(rows: list[dict], field: str) -> list[float]:
        return [
            float(row[field]) for row in rows if row.get(field) is not None
        ]

    form = {}
    for arm in ARMS:
        rows = arm_rows(arm)
        form[arm] = {
            "outputs": len(rows),
            "valid_without_repair": mean(
                [float(row["valid_without_repair"]) for row in rows]
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

    rows = load_train_rows(args.corpus_manifest)
    screen = memorization_screen(
        records,
        (read_sonnet_text(ROOT, row) for row in rows),
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
            baseline = data["condition_means"].get(ARMS[0])
            rft = data["condition_means"].get(ARMS[1])
            if baseline is not None and rft is not None:
                gains.append(rft - baseline)
        judge_gain = mean(gains)

    reading = self_play_reading(
        baseline_valid=float(form[ARMS[0]]["valid_without_repair"]),
        rft_valid=float(form[ARMS[1]]["valid_without_repair"]),
        baseline_accepted=float(form[ARMS[0]]["accepted_lines"]),
        rft_accepted=float(form[ARMS[1]]["accepted_lines"]),
        judge_gain=judge_gain,
    )

    args.scores_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_jsonl.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Self-Play Rejection SFT v1",
        "",
        f"Date: {datetime.date.today().isoformat()}",
        "",
        "Rejection fine-tuning on anchored candidates that already satisfy the",
        "scheme without repair. Generation uses the no-plan prompt with the same",
        "openings, seeds, and recipe for both arms. Metrics exclude repair.",
        "",
        "## Form",
        "",
        "| Metric | Baseline | RFT |",
        "|---|---:|---:|",
    ]
    for key, label, fmt in (
        ("valid_without_repair", "Scheme valid without repair", "{:.4f}"),
        ("accepted_lines", "Accepted lines", "{:.3f}"),
        ("failed_lines", "Failed lines", "{:.3f}"),
        ("uncertain_lines", "Uncertain lines", "{:.3f}"),
        ("stanza_pattern_ok", "4+4+3+3 stanza pattern", "{:.4f}"),
        ("rhyme_score", "Rhyme score", "{:.4f}"),
    ):
        values = [form[arm][key] for arm in ARMS]
        lines.append(
            "| " + label + " | "
            + " | ".join(format_value(v, fmt) for v in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Coherence proxies",
            "",
            "| Metric | Baseline | RFT |",
            "|---|---:|---:|",
        ]
    )
    for field in PROXY_FIELDS:
        values = [form[arm]["proxies"][field] for arm in ARMS]
        lines.append(
            f"| {field} | "
            + " | ".join(format_value(v, "{:.4f}") for v in values)
            + " |"
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
            "| Metric | Baseline | RFT |",
            "|---|---:|---:|",
        ]
    )
    screen_conditions = screen["conditions"]
    baseline_stats = screen_conditions.get(ARMS[0], {})
    rft_stats = screen_conditions.get(ARMS[1], {})
    for label, key, fmt in (
        ("Outputs with a copied line", "outputs_with_exact_line", "{:d}"),
        ("Max copied lines in one output", "max_exact_lines", "{:d}"),
        ("Mean 5-gram overlap", "mean_shingle_hit", "{:.4f}"),
    ):
        values = [baseline_stats.get(key, 0), rft_stats.get(key, 0)]
        lines.append(
            "| " + label + " | "
            + " | ".join(format_value(v, fmt) for v in values)
            + " |"
        )
    if judge_summary:
        lines.extend(
            [
                "",
                "## Judge panel",
                "",
                "| Judge | Baseline | RFT | Separation |",
                "|---|---:|---:|---:|",
            ]
        )
        for judge, data in judge_summary.items():
            means = data["condition_means"]
            lines.append(
                f"| {judge} | {means.get(ARMS[0])} | {means.get(ARMS[1])} | "
                f"{data['calibration_separation']} |"
            )
    lines.extend(["", "## Pre-registered reading", "", f"- Result: {reading['result']}"])
    for reason in reading["reasons"]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Training examples come from model outputs on the same 120 openings",
            "  with different seeds, so the measurement covers form, not new",
            "  openings.",
            "- The judge panel is secondary evidence; the gate uses judge gain only",
            "  as a guard against coherence loss.",
            "",
            "Verification: `python3 scripts/score_self_play_rft.py`",
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
        "self-play-score | "
        f"baseline_valid={form[ARMS[0]]['valid_without_repair']:.4f} "
        f"rft_valid={form[ARMS[1]]['valid_without_repair']:.4f} "
        f"accepted_gap={form[ARMS[1]]['accepted_lines'] - form[ARMS[0]]['accepted_lines']:+.3f} "
        f"judge_gain={judge_gain} reading={reading['result']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
