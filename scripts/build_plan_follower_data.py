#!/usr/bin/env python3
"""Build plan-follower training cards from valid generated-plan poems."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=ROOT / "artifacts/local/plan_generator/validation/stage2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/local/plan_follower_v2/data/cards_v1.jsonl",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=ROOT / "artifacts/local/plan_follower_v2/data/card_stats_v1.json",
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
    records = load_context_records(args.stage2_dir)
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
    score_by_path = {row["path"]: row for row in scored}
    cards = []
    stats = {"records": len(records), "valid": 0, "kept": 0}
    for record in records:
        row = score_by_path[record["path"]]
        if not (row["quatrain_ok"] and row["tercet_ok"]):
            continue
        stats["valid"] += 1
        if int(row["failed_lines"]) > 1:
            continue
        cards.append(
            {
                "unit_id": str(record["path"]),
                "opening_line": opening_by_id[str(record["prompt_id"])],
                "planned_words": [str(word) for word in record["planned_words"]],
                "poem_text": str(record["text"]),
            }
        )
        stats["kept"] += 1
    if not cards:
        raise ValueError("no plan-follower cards were built")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for card in cards:
            handle.write(json.dumps(card, ensure_ascii=False) + "\n")
    stats["output"] = str(args.output.relative_to(ROOT))
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "plan-follower-data | records={records} valid={valid} kept={kept}".format(
            **stats
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
