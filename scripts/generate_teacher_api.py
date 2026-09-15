#!/usr/bin/env python3
"""Teacher generation through a provider API with clean output-training terms.

Uses DeepSeek only: its model licence is MIT and its platform terms
explicitly permit using outputs to train other models. Keys are read from the
environment and never stored in the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.constrained_plan_validation import plan_from_anchored
from sonnet_analysis.minerva_v7_prompt_intervention import intervention_user_content
from sonnet_analysis.plan_then_poem_validation import PROMPT_ARM, plan_instruction
from sonnet_evaluation.rhyme_lexicon import load_lexicon

API_VERSION = "api_teacher_pilot_v1"
FREE_INSTRUCTION = (
    "Scrivi un sonetto in italiano classico di esattamente quattordici versi "
    "endecasillabi, con rime secondo uno schema regolare (per esempio "
    "ABBA ABBA CDE CDE). Usa come primo verso esattamente questo:\n"
    "{opening}\n"
    "Restituisci solo i quattordici versi, senza titolo, introduzione, "
    "spiegazione, numeri, etichette o testo in prosa."
)
PROVIDERS = {
    "deepseek": {
        "url": "https://api.deepseek.com/chat/completions",
        "env": "DEEPSEEK_API_KEY",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "mistral": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "env": "MISTRAL_API_KEY",
        "models": ["mistral-large-latest", "mistral-medium-latest"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(PROVIDERS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["planned", "free"], default="planned")
    parser.add_argument("--openings-file", type=Path, required=True)
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=ROOT / "data/metadata/rhyme_lexicon_v1.json",
    )
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7500])
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/local/api_teacher/generation",
    )
    return parser.parse_args()


def output_name(model: str, prompt_id: str, seed: int) -> str:
    key = f"{API_VERSION}|{model}|{prompt_id}|{seed}"
    return hashlib.sha256(key.encode()).hexdigest()[:20] + ".json"


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


def call_api(provider: str, model: str, key: str, prompt: str, args) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        PROVIDERS[provider]["url"],
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.loads(response.read().decode("utf-8"))
    usage = body.get("usage", {}) or {}
    return str(body["choices"][0]["message"]["content"]).strip(), {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
    }


def main() -> None:
    args = parse_args()
    key = os.environ.get(PROVIDERS[args.provider]["env"], "")
    if not key:
        raise SystemExit(
            f"set {PROVIDERS[args.provider]['env']} in the environment first"
        )
    lexicon = load_lexicon(args.lexicon)
    openings = [
        json.loads(line)
        for line in args.openings_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    openings = [
        opening
        for index, opening in enumerate(openings)
        if args.stride > 0 and index % args.stride == args.offset
    ]
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    skipped = 0
    for index, opening in enumerate(openings):
        opening_line = str(opening["opening_line"])
        if args.mode == "planned":
            try:
                plan = plan_from_anchored(
                    opening_line, lexicon, seed=int(args.seeds[0]) + index * 13
                )
            except ValueError:
                skipped += 1
                continue
            words = [str(word) for word in plan["words"]]
            prompt = (
                intervention_user_content(
                    opening_line, PROMPT_ARM, extra_instruction=plan_instruction(words)
                )
                + f"\n{opening_line}\n"
            )
            scheme = str(plan["scheme"])
        else:
            words = []
            scheme = ""
            prompt = FREE_INSTRUCTION.format(opening=opening_line)
        seed = int(args.seeds[0])
        path = output_dir / output_name(args.model, str(opening["id"]), seed)
        if path.is_file():
            continue
        jobs.append(
            {
                "opening": opening,
                "opening_line": opening_line,
                "prompt": prompt,
                "words": words,
                "scheme": scheme,
                "seed": seed,
                "path": path,
            }
        )
    print(
        f"api-teacher | provider={args.provider} model={args.model} mode={args.mode} "
        f"jobs={len(jobs)} skipped_plans={skipped}",
        flush=True,
    )
    started = time.monotonic()
    completed = 0
    errors = 0
    prompt_tokens = 0
    completion_tokens = 0

    def run_job(job):
        raw, usage = call_api(args.provider, args.model, key, job["prompt"], args)
        text, opening_found = clean_sonnet(raw, job["opening_line"])
        payload = {
            "generation_version": API_VERSION,
            "analysis_role": "api_teacher_pilot",
            "condition": f"{args.provider}/{args.model}",
            "teacher_model": f"{args.provider}/{args.model}",
            "mode": args.mode,
            "prompt": {"id": str(job["opening"]["id"]), "opening_line": job["opening_line"]},
            "prompt_id": str(job["opening"]["id"]),
            "opening_line": job["opening_line"],
            "seed": job["seed"],
            "scheme": job["scheme"],
            "planned_words": job["words"],
            "shots": 0,
            "instruction": job["prompt"],
            "text": text,
            "raw": raw,
            "opening_found": opening_found,
            "usage": usage,
        }
        temporary = job["path"].with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, job["path"])
        return usage

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                usage = future.result()
            except Exception as exc:  # noqa: BLE001 - report and continue
                errors += 1
                print(f"api-teacher | error={exc}", flush=True)
                continue
            completed += 1
            prompt_tokens += int(usage.get("prompt_tokens", 0))
            completion_tokens += int(usage.get("completion_tokens", 0))
            if completed % 25 == 0 or completed == len(jobs):
                print(
                    f"api-teacher | completed={completed}/{len(jobs)} errors={errors} "
                    f"elapsed={time.monotonic() - started:.0f}s",
                    flush=True,
                )
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
        "generation_version": API_VERSION,
        "analysis_role": "api_teacher_pilot",
        "models": sorted({row["condition"] for row in outputs}),
        "completed_output_count": len(outputs),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "outputs": outputs,
        "v7_test_accessed": False,
    }
    temporary = (output_dir / "complete.json").with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(complete, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output_dir / "complete.json")
    print(
        f"api-teacher | complete outputs={len(outputs)} errors={errors} "
        f"elapsed={time.monotonic() - started:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
