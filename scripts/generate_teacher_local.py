#!/usr/bin/env python3
"""Self-hosted synthetic teacher generation with a permissively licensed model.

Runs on the GPU instance. Loads an open-weight model (optionally 4-bit NF4),
generates classical Italian sonnets on the given openings, and stores them in
the synthetic-pilot payload schema so the checker, the memorization screen, the
judge panel, and the teacher-card builder all work unchanged.
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
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--quantization", choices=["none", "nf4"], default="nf4")
    parser.add_argument("--openings-file", type=Path, default=None)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=ROOT / "configs/minerva_7b_v7_exploratory_prompts.json",
    )
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=7300)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/local/permissive_teacher/generation",
    )
    return parser.parse_args()


def output_name(model_id: str, prompt_id: str) -> str:
    key = f"{PILOT_VERSION}|{model_id}|{prompt_id}"
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


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("self-hosted teacher generation requires one CUDA GPU")
    if args.openings_file is not None:
        prompts = [
            json.loads(line)
            for line in args.openings_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        from sonnet_analysis.minerva_v7_exploratory_prompts import (
            validate_exploratory_prompt_manifest,
        )

        prompts = validate_exploratory_prompt_manifest(
            args.prompts,
            expected_sha256="2f33aa518aa61c11193831e53b07fd3bd861a72bf68bb23c0e0e5b1a13b1d0c7",
        )["prompts"]
    prompts = prompts[: args.limit]
    prompts = [
        prompt
        for index, prompt in enumerate(prompts)
        if args.stride > 0 and index % args.stride == args.offset
    ]
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for prompt in prompts:
        path = output_dir / output_name(args.model_id, str(prompt["id"]))
        if path.is_file():
            continue
        jobs.append((prompt, path))
    print(
        f"permissive-teacher | model={args.model_id} jobs={len(jobs)} "
        f"quantization={args.quantization}",
        flush=True,
    )
    if not jobs:
        return
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.revision, local_files_only=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = None
    if args.quantization == "nf4":
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map={"": 0},
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )
    model.eval()
    previous_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    started = time.monotonic()
    torch.manual_seed(args.seed)
    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start : start + args.batch_size]
        rendered = []
        for prompt, _ in batch:
            instruction = INSTRUCTION.format(opening=str(prompt["opening_line"]))
            try:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            rendered.append(text)
        inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        for (prompt, path), output_ids, prompt_ids in zip(
            batch, generated, inputs["input_ids"], strict=True
        ):
            completion = output_ids[len(prompt_ids) :]
            raw = tokenizer.decode(completion, skip_special_tokens=True).strip()
            text, opening_found = clean_sonnet(raw, str(prompt["opening_line"]))
            instruction = INSTRUCTION.format(opening=str(prompt["opening_line"]))
            payload = {
                "generation_version": PILOT_VERSION,
                "analysis_role": "synthetic_sonnet_pilot",
                "condition": args.model_id,
                "teacher_model": args.model_id,
                "teacher_revision": args.revision,
                "prompt": dict(prompt),
                "prompt_id": str(prompt["id"]),
                "opening_line": str(prompt["opening_line"]),
                "seed": args.seed,
                "planned_words": [],
                "shots": 0,
                "instruction": instruction,
                "text": text,
                "raw": raw,
                "opening_found": opening_found,
            }
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        print(
            f"permissive-teacher | completed={min(start + args.batch_size, len(jobs))}"
            f"/{len(jobs)} elapsed={time.monotonic() - started:.0f}s",
            flush=True,
        )
    tokenizer.padding_side = previous_side
    outputs = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name == "complete.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        outputs.append(
            {
                "path": path.name,
                "condition": str(payload.get("condition", args.model_id)),
                "prompt_id": str(payload.get("prompt_id", "")),
                "seed": int(payload.get("seed", 0)),
            }
        )
    complete = {
        "generation_version": PILOT_VERSION,
        "analysis_role": "synthetic_sonnet_pilot",
        "models": sorted({row["condition"] for row in outputs}),
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
        f"permissive-teacher | complete outputs={len(outputs)} "
        f"elapsed={time.monotonic() - started:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
