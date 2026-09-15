#!/usr/bin/env python3
"""Grammar-repair teacher generation.

Takes poems from the project's own pipeline that already satisfy the scheme
but contain grammar errors, and asks a permissively licensed API teacher to
repair grammar while keeping every line-ending word. The pair becomes targeted
training data for the coherence weakness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.few_shot_context_validation import load_context_records
from sonnet_analysis.minerva_v7_exploratory_prompts import (
    validate_exploratory_prompt_manifest,
)
from sonnet_evaluation.sonnet_prosody_sealed import score_output
from sonnet_training.teacher_api import PROVIDERS, call_api

REPAIR_VERSION = "grammar_repair_teacher_v1"
REPAIR_INSTRUCTION = (
    "Correggi la grammatica e la sintassi di questo sonetto italiano classico. "
    "Mantieni esattamente le stesse parole di rima alla fine di ogni verso e lo "
    "stesso schema. Non cambiare il senso, non aggiungere commenti. Restituisci "
    "solo i quattordici versi corretti, senza numeri, etichette o prosa.\n\n"
    "Parole di rima, nell'ordine dei versi:\n{words}\n\n"
    "Sonetto:\n{poem}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "artifacts/local/plan_follower_v2/validation/stage2_temp04",
    )
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/local/grammar_repair/generation",
    )
    return parser.parse_args()


def output_name(model: str, key: str) -> str:
    digest = hashlib.sha256(f"{REPAIR_VERSION}|{model}|{key}".encode()).hexdigest()[:20]
    return digest + ".json"


def clean_sonnet(raw: str, opening_line: str) -> tuple[str, bool]:
    lines = [line.strip() for line in str(raw).splitlines() if line.strip()]
    normalized = " ".join(opening_line.split()).casefold()
    for index, line in enumerate(lines):
        if " ".join(line.split()).casefold() == normalized:
            window = lines[index : index + 14]
            if len(window) == 14:
                return "\n".join(window), True
    verse_lines = [
        line
        for line in lines
        if not line.endswith(":")
        and not line.lower().startswith(("ecco", "nota", "versi", "spiegazione"))
    ]
    if len(verse_lines) >= 14:
        return "\n".join(verse_lines[:14]), False
    return "\n".join(verse_lines), False


def main() -> None:
    args = parse_args()
    key = os.environ.get(PROVIDERS[args.provider]["env"], "")
    if not key:
        raise SystemExit(f"set {PROVIDERS[args.provider]['env']} first")
    prompts = validate_exploratory_prompt_manifest(
        ROOT / "configs/minerva_7b_v7_exploratory_prompts.json",
        expected_sha256="2f33aa518aa61c11193831e53b07fd3bd861a72bf68bb23c0e0e5b1a13b1d0c7",
    )["prompts"]
    opening_by_id = {str(prompt["id"]): str(prompt["opening_line"]) for prompt in prompts}
    records = load_context_records(args.source_dir)
    candidates = []
    for index, record in enumerate(records):
        if args.stride > 0 and index % args.stride != args.offset:
            continue
        metrics = score_output(str(record["text"]))
        if not (metrics["quatrain_ok"] and metrics["tercet_ok"]):
            continue
        candidates.append(record)
        if len(candidates) >= args.limit:
            break
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"grammar-repair | provider={args.provider} model={args.model} "
        f"candidates={len(candidates)}",
        flush=True,
    )
    started = time.monotonic()
    completed = 0
    errors = 0
    prompt_tokens = 0
    completion_tokens = 0
    for record in candidates:
        opening = opening_by_id[str(record["prompt_id"])]
        words = [str(word) for word in record["planned_words"]]
        job_key = f"{record['prompt_id']}|{record['seed']}"
        path = output_dir / output_name(args.model, job_key)
        if path.is_file():
            continue
        prompt = REPAIR_INSTRUCTION.format(
            words="\n".join(f"{i + 1}. {word}" for i, word in enumerate(words)),
            poem=str(record["text"]).strip(),
        )
        try:
            raw, usage = call_api(
                args.provider,
                args.model,
                key,
                prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            errors += 1
            print(f"grammar-repair | error={exc}", flush=True)
            continue
        text, opening_found = clean_sonnet(raw, opening)
        payload = {
            "generation_version": REPAIR_VERSION,
            "analysis_role": "grammar_repair_teacher",
            "condition": f"{args.provider}/{args.model}",
            "teacher_model": f"{args.provider}/{args.model}",
            "prompt": {"id": str(record["prompt_id"]), "opening_line": opening},
            "prompt_id": str(record["prompt_id"]),
            "opening_line": opening,
            "seed": int(record["seed"]),
            "planned_words": words,
            "source_text": str(record["text"]),
            "text": text,
            "raw": raw,
            "opening_found": opening_found,
            "usage": usage,
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
        completed += 1
        prompt_tokens += int(usage.get("prompt_tokens", 0))
        completion_tokens += int(usage.get("completion_tokens", 0))
        if completed % 25 == 0:
            print(
                f"grammar-repair | completed={completed} errors={errors} "
                f"elapsed={time.monotonic() - started:.0f}s",
                flush=True,
            )
    complete = {
        "generation_version": REPAIR_VERSION,
        "analysis_role": "grammar_repair_teacher",
        "models": [f"{args.provider}/{args.model}"],
        "completed_output_count": completed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "v7_test_accessed": False,
    }
    temporary = (output_dir / "complete.json").with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output_dir / "complete.json")
    print(
        f"grammar-repair | complete outputs={completed} errors={errors} "
        f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens}",
        flush=True,
    )


if __name__ == "__main__":
    main()
