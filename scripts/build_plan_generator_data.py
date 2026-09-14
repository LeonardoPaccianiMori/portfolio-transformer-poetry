#!/usr/bin/env python3
"""Build plan-only training data: corpus traces plus augmented valid plans."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.constrained_plan_validation import plan_from_anchored
from sonnet_evaluation.rhyme_lexicon import load_lexicon
from sonnet_training.reasoning_trace_sft import derive_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traces",
        type=Path,
        default=ROOT / "artifacts/local/reasoning_trace/data/traces_v1.jsonl",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=ROOT / "data/metadata/rhyme_lexicon_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/local/plan_generator/data/plans_v1.jsonl",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=ROOT / "artifacts/local/plan_generator/data/plan_stats_v1.json",
    )
    parser.add_argument("--augmented-per-card", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    args.stats_json = args.stats_json.resolve()
    lexicon = load_lexicon(args.lexicon)
    cards = []
    with args.traces.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cards.append(json.loads(line))
    stats = {
        "corpus_cards": 0,
        "augmented_cards": 0,
        "augmented_skipped": 0,
    }
    output_cards = []
    for index, card in enumerate(cards):
        output_cards.append(
            {
                "unit_id": str(card["unit_id"]),
                "opening_line": str(card["opening_line"]),
                "scheme": str(card["scheme"]),
                "trace_text": str(card["trace_text"]),
                "source": "corpus",
            }
        )
        stats["corpus_cards"] += 1
        for variant in range(args.augmented_per_card):
            try:
                plan = plan_from_anchored(
                    str(card["opening_line"]),
                    lexicon,
                    seed=args.seed + index * 17 + variant,
                )
            except ValueError:
                stats["augmented_skipped"] += 1
                continue
            scheme = str(plan["scheme"])
            words = [str(word) for word in plan["words"]]
            output_cards.append(
                {
                    "unit_id": f"{card['unit_id']}::aug{variant}",
                    "opening_line": str(card["opening_line"]),
                    "scheme": scheme,
                    "trace_text": derive_trace(scheme, words),
                    "source": "augmented",
                }
            )
            stats["augmented_cards"] += 1
    if not output_cards:
        raise ValueError("no plan cards were built")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for card in output_cards:
            handle.write(json.dumps(card, ensure_ascii=False) + "\n")
    stats["total_cards"] = len(output_cards)
    stats["output"] = str(args.output.relative_to(ROOT))
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "plan-generator-data | corpus={corpus_cards} augmented={augmented_cards} "
        "skipped={augmented_skipped} total={total_cards}".format(**stats),
        flush=True,
    )


if __name__ == "__main__":
    main()
