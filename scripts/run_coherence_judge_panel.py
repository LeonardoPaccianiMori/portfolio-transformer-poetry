#!/usr/bin/env python3
"""Run the calibrated model-judge coherence panel on the context A/B grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.few_shot_context_validation import load_context_records
from sonnet_evaluation.coherence import calibrate_judge, judge_texts
from sonnet_evaluation.rhyme_lexicon import line_final_word
from sonnet_training.form_targeted_data import load_train_rows, read_sonnet_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-dir",
        type=Path,
        default=ROOT / "artifacts/local/few_shot_context/validation/generation",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["zeroshot", "fewshot1", "fewshot3"],
    )
    parser.add_argument("--sample-per-condition", type=int, default=10)
    parser.add_argument(
        "--judges",
        nargs="+",
        default=["opencode-go/glm-5.2", "opencode-go/qwen3.6-plus"],
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "artifacts/local/few_shot_context/coherence_judge_v1.json",
    )
    parser.add_argument("--calibration-count", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=240)
    return parser.parse_args()


def calibration_originals(count: int) -> list[str]:
    rows = load_train_rows(
        ROOT / "data/processed/sonnets_expanded_v8/sonnets_manifest.csv"
    )
    originals = []
    for row in rows:
        text = read_sonnet_text(ROOT, row)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) != 14:
            continue
        if any(line_final_word(line) is None for line in lines):
            continue
        originals.append(text)
        if len(originals) >= count:
            break
    if len(originals) < count:
        raise ValueError("not enough calibration originals")
    return originals


def main() -> None:
    args = parse_args()
    records = load_context_records(args.generation_dir)
    sampled: dict[str, list[dict]] = {}
    for condition in args.conditions:
        rows = [record for record in records if record["system_id"] == condition]
        if not rows:
            raise ValueError(f"no records for condition {condition}")
        step = max(1, len(rows) // args.sample_per_condition)
        sampled[condition] = rows[::step][: args.sample_per_condition]
    originals = calibration_originals(args.calibration_count)
    output: dict = {
        "generation_dir": str(args.generation_dir),
        "conditions": list(args.conditions),
        "sample_per_condition": args.sample_per_condition,
        "judges": {},
    }
    for judge in args.judges:
        print(f"coherence-panel | calibration {judge}", flush=True)
        calibration = calibrate_judge(
            judge,
            originals,
            timeout_seconds=args.timeout,
            progress=lambda message: print(f"coherence-panel | {judge} {message}", flush=True),
        )
        condition_scores = {}
        for condition, rows in sampled.items():
            print(
                f"coherence-panel | judging {judge} {condition} "
                f"({len(rows)} outputs)",
                flush=True,
            )
            results = judge_texts(
                judge,
                [str(row["text"]) for row in rows],
                timeout_seconds=args.timeout,
                progress=lambda message: print(
                    f"coherence-panel | {judge} {condition} {message}", flush=True
                ),
            )
            condition_scores[condition] = [
                {"path": row["path"], "scores": result}
                for row, result in zip(rows, results, strict=True)
            ]
        output["judges"][judge] = {
            "calibration": calibration,
            "conditions": condition_scores,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"coherence-panel | wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
