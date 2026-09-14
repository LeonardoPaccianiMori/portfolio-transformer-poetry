#!/usr/bin/env python3
"""Build joint trace cards: corpus traces plus valid pipeline pairs."""

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
from sonnet_analysis.minerva_v7_exploratory_prompts import (
    validate_exploratory_prompt_manifest,
)
from sonnet_evaluation.sonnet_prosody_sealed import score_records
from sonnet_training.reasoning_trace_sft import (
    derive_trace,
    parse_trace,
    planned_words_from_trace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-traces",
        type=Path,
        default=ROOT / "artifacts/local/reasoning_trace/data/traces_v1.jsonl",
    )
    parser.add_argument(
        "--stage1-dir",
        type=Path,
        default=ROOT / "artifacts/local/plan_follower_v2/validation/plans_temp04",
    )
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=ROOT / "artifacts/local/plan_follower_v2/validation/stage2_temp04",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/local/joint_trace/data/cards_v1.jsonl",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=ROOT / "artifacts/local/joint_trace/data/card_stats_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    args.stats_json = args.stats_json.resolve()
    prompts = validate_exploratory_prompt_manifest(
        ROOT / "configs/minerva_7b_v7_exploratory_prompts.json",
        expected_sha256="2f33aa518aa61c11193831e53b07fd3bd861a72bf68bb23c0e0e5b1a13b1d0c7",
    )["prompts"]
    opening_by_id = {str(prompt["id"]): str(prompt["opening_line"]) for prompt in prompts}
    cards = []
    with args.corpus_traces.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                card = json.loads(line)
                cards.append(
                    {
                        "unit_id": f"corpus::{card['unit_id']}",
                        "opening_line": str(card["opening_line"]),
                        "scheme": str(card["scheme"]),
                        "trace_text": str(card["trace_text"]),
                        "poem_text": str(card["poem_text"]),
                        "source": "corpus",
                    }
                )
    stats = {"corpus_cards": len(cards), "pipeline_cards": 0, "pipeline_skipped": 0}
    stage1 = load_context_records(args.stage1_dir)
    plan_by_key = {}
    for record in stage1:
        opening = opening_by_id[str(record["prompt_id"])]
        parsed = parse_trace(str(record["text"]), require_poem=False)
        if parsed is None:
            continue
        words = planned_words_from_trace(parsed, opening)
        if words is None:
            continue
        plan_by_key[(str(record["prompt_id"]), int(record["seed"]))] = (
            str(parsed["schema"]),
            words,
        )
    stage2 = load_context_records(args.stage2_dir)
    text_by_path = {record["path"]: str(record["text"]) for record in stage2}
    scored = score_records(
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
    for row in scored:
        if not (row["quatrain_ok"] and row["tercet_ok"]):
            stats["pipeline_skipped"] += 1
            continue
        opening = opening_by_id[str(row["prompt_id"])]
        plan = plan_by_key.get((str(row["prompt_id"]), int(row["seed"])))
        if plan is None:
            stats["pipeline_skipped"] += 1
            continue
        scheme, words = plan
        trace_text = derive_trace(scheme, words)
        cards.append(
            {
                "unit_id": f"pipeline::{row['path']}",
                "opening_line": opening,
                "scheme": scheme,
                "trace_text": trace_text,
                "poem_text": text_by_path[str(row["path"])],
                "source": "pipeline",
            }
        )
        stats["pipeline_cards"] += 1
    if not cards:
        raise ValueError("no joint cards were built")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for card in cards:
            handle.write(json.dumps(card, ensure_ascii=False) + "\n")
    stats["total_cards"] = len(cards)
    stats["output"] = str(args.output.relative_to(ROOT))
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "joint-trace-data | corpus={corpus_cards} pipeline={pipeline_cards} "
        "skipped={pipeline_skipped} total={total_cards}".format(**stats),
        flush=True,
    )


if __name__ == "__main__":
    main()
