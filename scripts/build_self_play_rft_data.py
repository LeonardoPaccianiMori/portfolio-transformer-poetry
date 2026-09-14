#!/usr/bin/env python3
"""Filter anchored self-play candidates into the rejection dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.coherence import coherence_proxies
from sonnet_evaluation.corpus_memorization import build_corpus_line_index
from sonnet_evaluation.sonnet_prosody_sealed import score_records
from sonnet_training.form_targeted_data import load_train_rows, read_sonnet_text
from sonnet_training.self_play_rft import (
    collect_candidates,
    filter_cards,
    load_anchored_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchored-dir",
        type=Path,
        default=ROOT / "artifacts/local/self_play/validation/generation",
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=ROOT / "data/processed/sonnets_expanded_v8/sonnets_manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/local/self_play_rft/data/candidates_v1.jsonl",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=ROOT / "artifacts/local/self_play_rft/data/filter_stats_v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_anchored_candidates(args.anchored_dir)
    scores = score_records(
        [
            {
                "path": record["path"],
                "prompt_id": record["prompt_id"],
                "seed": record["seed"],
                "system_id": "anchored",
                "text": record["text"],
            }
            for record in records
        ]
    )
    proxies = {
        record["path"]: coherence_proxies(record["text"]) for record in records
    }
    rows = load_train_rows(args.corpus_manifest)
    line_index = build_corpus_line_index(
        read_sonnet_text(ROOT, row) for row in rows
    )
    cards = collect_candidates(records, scores, proxies, line_index)
    kept, rejected = filter_cards(cards)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for card in sorted(kept, key=lambda row: (row["prompt_id"], row["seed"])):
            handle.write(json.dumps(card, ensure_ascii=False) + "\n")
    stats = {
        "candidate_count": len(cards),
        "kept_count": len(kept),
        "rejected": rejected,
        "corpus_lines_indexed": len(line_index),
        "candidate_dir": str(args.anchored_dir.relative_to(ROOT)),
        "output": str(args.output.relative_to(ROOT)),
    }
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "self-play-data | candidates={candidate_count} kept={kept_count} "
        "rejected={rejected}".format(**stats),
        flush=True,
    )


if __name__ == "__main__":
    main()
