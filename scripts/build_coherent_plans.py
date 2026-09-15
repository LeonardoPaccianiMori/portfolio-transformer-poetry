#!/usr/bin/env python3
"""Build coherent plan records from valid free-mode teacher sonnets.

Teacher sonnets that pass the checker provide a rhyme plan that is coherent by
construction: their own ending words. Evaluating the project's poem writer on
these plans isolates plan quality from writer capability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.rhyme_lexicon import line_final_word
from sonnet_evaluation.sonnet_prosody_sealed import score_output
from sonnet_training.reasoning_trace_sft import derive_trace

VERSION = "coherent_plan_from_teacher_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        type=Path,
        nargs="+",
        default=[
            ROOT / "artifacts/local/api_teacher/free_generation",
            ROOT / "artifacts/local/api_teacher/generation",
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/local/api_teacher/coherent_plans",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    for source in args.sources:
        for path in sorted(source.glob("*.json")):
            if path.name == "complete.json":
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            text = str(payload.get("text", "")).strip()
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if len(lines) != 14:
                skipped += 1
                continue
            metrics = score_output(text)
            if not (metrics["quatrain_ok"] and metrics["tercet_ok"]):
                skipped += 1
                continue
            words = [line_final_word(line) for line in lines]
            if any(word is None for word in words):
                skipped += 1
                continue
            opening = str(payload["opening_line"])
            scheme = str(metrics["rhyme_scheme"])
            record = {
                "generation_version": VERSION,
                "analysis_role": "coherent_plan_from_teacher",
                "condition": "plan_sft",
                "prompt": {"id": str(payload["prompt_id"]), "opening_line": opening},
                "prompt_id": str(payload["prompt_id"]),
                "opening_line": opening,
                "seed": int(payload.get("seed", 0)),
                "planned_words": [str(word) for word in words],
                "text": derive_trace(scheme, [str(word) for word in words]),
                "teacher_model": str(payload.get("teacher_model", "")),
                "origin": str(source.name),
            }
            digest = hashlib.sha256(
                f"{VERSION}|{record['prompt_id']}|{record['seed']}".encode()
            ).hexdigest()[:20]
            target = output_dir / f"{digest}.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(temporary, target)
            kept += 1
    outputs = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name == "complete.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        outputs.append(
            {
                "path": path.name,
                "condition": str(payload["condition"]),
                "prompt_id": str(payload["prompt_id"]),
                "seed": int(payload["seed"]),
            }
        )
    complete = {
        "generation_version": VERSION,
        "analysis_role": "coherent_plan_from_teacher",
        "arms": ["plan_sft"],
        "completed_output_count": len(outputs),
        "outputs": outputs,
        "v7_test_accessed": False,
    }
    temporary = (output_dir / "complete.json").with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output_dir / "complete.json")
    print(
        f"coherent-plans | kept={kept} skipped={skipped} -> {output_dir}", flush=True
    )


if __name__ == "__main__":
    main()
