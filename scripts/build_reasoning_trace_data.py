#!/usr/bin/env python3
"""Derive reasoning traces and poem targets from the V8 corpus sonnets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_exploratory_prompts import (
    validate_exploratory_prompt_manifest,
)
from sonnet_evaluation.rhyme_lexicon import line_final_word
from sonnet_evaluation.sonnet_prosody_sealed import score_output
from sonnet_training.form_targeted_data import load_train_rows, read_sonnet_text
from sonnet_training.reasoning_trace_sft import canonical_scheme, derive_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/processed/sonnets_expanded_v8/sonnets_manifest.csv",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=ROOT / "configs/minerva_7b_v7_exploratory_prompts.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/local/reasoning_trace/data/traces_v1.jsonl",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=ROOT / "artifacts/local/reasoning_trace/data/trace_stats_v1.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = validate_exploratory_prompt_manifest(
        args.prompts,
        expected_sha256="2f33aa518aa61c11193831e53b07fd3bd861a72bf68bb23c0e0e5b1a13b1d0c7",
    )["prompts"]
    prompt_hashes = {
        str(prompt.get("source_logical_sha256", "")) for prompt in prompts
    }
    rows = load_train_rows(args.manifest)
    excluded = {
        row["unit_id"] for row in rows if row["logical_sha256"] in prompt_hashes
    }
    stats = {
        "rows": 0,
        "excluded": 0,
        "malformed": 0,
        "scheme_rejected": 0,
        "kept": 0,
    }
    cards = []
    for row in rows:
        stats["rows"] += 1
        if row["unit_id"] in excluded:
            stats["excluded"] += 1
            continue
        text = read_sonnet_text(ROOT, row)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) != 14:
            stats["malformed"] += 1
            continue
        words = [line_final_word(line) for line in lines]
        if any(word is None for word in words):
            stats["malformed"] += 1
            continue
        metrics = score_output(text)
        if not (metrics["quatrain_ok"] and metrics["tercet_ok"]):
            stats["scheme_rejected"] += 1
            continue
        scheme = canonical_scheme(metrics["rhyme_scheme"])
        if scheme is None:
            stats["scheme_rejected"] += 1
            continue
        trace_text = derive_trace(scheme, [str(word) for word in words])
        cards.append(
            {
                "unit_id": str(row["unit_id"]),
                "opening_line": lines[0],
                "scheme": scheme,
                "planned_words": [str(word) for word in words],
                "trace_text": trace_text,
                "poem_text": text.strip("\n"),
            }
        )
        stats["kept"] += 1
        if args.limit is not None and len(cards) >= args.limit:
            break
    if not cards:
        raise ValueError("no reasoning-trace cards were built")
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
        "reasoning-trace-data | rows={rows} kept={kept} excluded={excluded} "
        "malformed={malformed} scheme_rejected={scheme_rejected}".format(**stats),
        flush=True,
    )


if __name__ == "__main__":
    main()
