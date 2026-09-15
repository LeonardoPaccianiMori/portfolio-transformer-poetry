#!/usr/bin/env python3
"""Build plan-following cards from checker-valid synthetic teacher sonnets.

Each kept sonnet is reduced to a plan (its own rhyme scheme and ending words)
so the poem model can be fine-tuned on the teacher's poems in the same
plan-following format used for the pipeline data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.rhyme_lexicon import line_final_word
from sonnet_evaluation.sonnet_prosody_sealed import score_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-dir",
        type=Path,
        default=ROOT / "artifacts/local/synthetic_pilot/generation",
    )
    parser.add_argument(
        "--judge-json",
        type=Path,
        default=None,
        help="optional judge payload; keeps only outputs whose mean meets the threshold",
    )
    parser.add_argument("--judge-threshold", type=float, default=3.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/local/synthetic_pilot/data/teacher_cards_v1.jsonl",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=ROOT / "artifacts/local/synthetic_pilot/data/teacher_stats_v1.json",
    )
    parser.add_argument("--max-failed-lines", type=int, default=1)
    return parser.parse_args()


def judge_means(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    means: dict[str, list[float]] = {}
    for data in payload["judges"].values():
        for condition, rows in data["conditions"].items():
            for row in rows:
                if row.get("scores") is None:
                    continue
                means.setdefault(str(row["path"]), []).append(
                    float(row["scores"]["mean"])
                )
    return {path: sum(values) / len(values) for path, values in means.items()}


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    args.stats_json = args.stats_json.resolve()
    judged = (
        judge_means(args.judge_json) if args.judge_json is not None else {}
    )
    cards = []
    stats = {
        "records": 0,
        "malformed": 0,
        "scheme_rejected": 0,
        "failed_line_rejected": 0,
        "judge_rejected": 0,
        "kept": 0,
    }
    by_model: dict[str, int] = {}
    for path in sorted(args.generation_dir.glob("*.json")):
        if path.name == "complete.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        stats["records"] += 1
        text = str(payload.get("text", ""))
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) != 14:
            stats["malformed"] += 1
            continue
        metrics = score_output(text)
        if not (metrics["quatrain_ok"] and metrics["tercet_ok"]):
            stats["scheme_rejected"] += 1
            continue
        if int(metrics["failed_lines"]) > args.max_failed_lines:
            stats["failed_line_rejected"] += 1
            continue
        if judged:
            mean = judged.get(path.name)
            if mean is None or mean < args.judge_threshold:
                stats["judge_rejected"] += 1
                continue
        planned = [str(word) for word in payload.get("planned_words", [])]
        if len(planned) == 14:
            words = planned
        else:
            words = [line_final_word(line) for line in lines]
        if any(word is None for word in words) or len(words) != 14:
            stats["malformed"] += 1
            continue
        model = str(payload.get("teacher_model", payload.get("condition", "unknown")))
        cards.append(
            {
                "unit_id": f"teacher::{model}::{payload.get('prompt_id', path.stem)}",
                "opening_line": lines[0],
                "planned_words": [str(word) for word in words],
                "poem_text": text,
                "teacher_model": model,
                "scheme": str(metrics["rhyme_scheme"]),
            }
        )
        stats["kept"] += 1
        by_model[model] = by_model.get(model, 0) + 1
    if not cards:
        raise ValueError("no teacher cards were built")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for card in cards:
            handle.write(json.dumps(card, ensure_ascii=False) + "\n")
    stats["by_model"] = by_model
    stats["output"] = str(args.output.relative_to(ROOT))
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "teacher-cards | records={records} kept={kept} scheme_rejected={scheme_rejected} "
        "failed_line_rejected={failed_line_rejected} judge_rejected={judge_rejected} "
        "malformed={malformed}".format(**stats),
        flush=True,
    )


if __name__ == "__main__":
    main()
