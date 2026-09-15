#!/usr/bin/env python3
"""Synthetic sonnet pilot: ask large models for classical Italian sonnets.

The pilot measures whether a large language model can produce sonnets that
pass the checker on our evaluation openings. Outputs are stored per model and
opening with the raw text, so the checker, the memorization screen, and the
judge panel can score them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_exploratory_prompts import (
    validate_exploratory_prompt_manifest,
)

PILOT_VERSION = "synthetic_sonnet_pilot_v1"
INSTRUCTION = (
    "Scrivi un sonetto in italiano classico di esattamente quattordici versi "
    "endecasillabi, con rime secondo uno schema regolare (per esempio "
    "ABBA ABBA CDE CDE). Usa come primo verso esattamente questo:\n"
    "{opening}\n"
    "Restituisci solo i quattordici versi, senza titolo, introduzione, "
    "spiegazione, numeri, etichette o testo in prosa."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=ROOT / "configs/minerva_7b_v7_exploratory_prompts.json",
    )
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/local/synthetic_pilot/generation",
    )
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def output_name(model: str, prompt_id: str) -> str:
    key = f"{PILOT_VERSION}|{model}|{prompt_id}"
    return hashlib.sha256(key.encode()).hexdigest()[:20] + ".json"


def clean_sonnet(raw: str, opening_line: str) -> tuple[str, bool]:
    lines = [line.strip() for line in str(raw).splitlines() if line.strip()]
    normalized_opening = " ".join(opening_line.split()).casefold()
    for index, line in enumerate(lines):
        if " ".join(line.split()).casefold() == normalized_opening:
            window = lines[index : index + 14]
            if len(window) == 14:
                return "\n".join(window), True
    verse_lines = [
        line
        for line in lines
        if not line.endswith(":")
        and not line.lower().startswith(
            ("ecco", "nota", "ecco il", "sonetto", "versi", "spiegazione")
        )
    ]
    if len(verse_lines) >= 14:
        return "\n".join(verse_lines[:14]), False
    return "\n".join(verse_lines), False


def main() -> None:
    args = parse_args()
    prompts = validate_exploratory_prompt_manifest(
        args.prompts,
        expected_sha256="2f33aa518aa61c11193831e53b07fd3bd861a72bf68bb23c0e0e5b1a13b1d0c7",
    )["prompts"][: args.limit]
    prompts = [
        prompt
        for index, prompt in enumerate(prompts)
        if args.stride > 0 and index % args.stride == args.offset
    ]
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed = 0
    skipped = 0
    for model in args.models:
        for prompt in prompts:
            path = output_dir / output_name(model, str(prompt["id"]))
            if path.is_file():
                skipped += 1
                continue
            instruction = INSTRUCTION.format(opening=str(prompt["opening_line"]))
            command = [
                "opencode",
                "run",
                "-m",
                model,
                instruction,
            ]
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                    check=False,
                )
                raw = result.stdout.strip()
            except subprocess.TimeoutExpired:
                raw = ""
            text, opening_found = clean_sonnet(raw, str(prompt["opening_line"]))
            payload = {
                "generation_version": PILOT_VERSION,
                "analysis_role": "synthetic_sonnet_pilot",
                "condition": model,
                "prompt": dict(prompt),
                "prompt_id": str(prompt["id"]),
                "opening_line": str(prompt["opening_line"]),
                "seed": 0,
                "planned_words": [],
                "shots": 0,
                "text": text,
                "raw": raw,
                "opening_found": opening_found,
                "teacher_model": model,
                "instruction": instruction,
            }
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
            completed += 1
            print(
                f"synthetic-pilot | model={model} completed={completed} "
                f"skipped={skipped} elapsed={time.monotonic() - started:.0f}s",
                flush=True,
            )
            if args.sleep:
                time.sleep(args.sleep)
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
                "seed": 0,
            }
        )
    complete = {
        "generation_version": PILOT_VERSION,
        "analysis_role": "synthetic_sonnet_pilot",
        "models": list(args.models),
        "prompt_count": len(prompts),
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
        f"synthetic-pilot | complete outputs={len(outputs)} "
        f"elapsed={time.monotonic() - started:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
