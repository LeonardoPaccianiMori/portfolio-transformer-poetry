#!/usr/bin/env python3
"""Self-hosted teacher generation with a lexicon rhyme plan.

The teacher writes the poem to a plan we supply, so the endings (and therefore
the scheme) are guaranteed if the teacher follows the plan. This plays to the
teacher's strengths: grammatical, coherent verse. Only permissively licensed
self-hosted weights are used, so the resulting corpus has clean provenance.
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

from sonnet_analysis.constrained_plan_validation import plan_from_anchored
from sonnet_analysis.minerva_v7_prompt_intervention import intervention_user_content
from sonnet_analysis.plan_then_poem_validation import PROMPT_ARM, plan_instruction
from sonnet_evaluation.rhyme_lexicon import load_lexicon

TEACHER_VERSION = "planned_teacher_pilot_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--quantization", choices=["none", "nf4"], default="none")
    parser.add_argument("--openings-file", type=Path, required=True)
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=ROOT / "data/metadata/rhyme_lexicon_v1.json",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7400])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=420)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/local/permissive_teacher/planned_generation",
    )
    return parser.parse_args()


def render_planned_prompt(tokenizer, opening_line: str, words) -> str:
    """Render the plan-following prompt, disabling thinking when supported."""

    user_content = intervention_user_content(
        opening_line, PROMPT_ARM, extra_instruction=plan_instruction(list(words))
    )
    messages = [{"role": "user", "content": user_content}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"{rendered}{opening_line}\n"


def output_name(model_id: str, prompt_id: str, seed: int) -> str:
    key = f"{TEACHER_VERSION}|{model_id}|{prompt_id}|{seed}"
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
        if not line.endswith(":") and not line.lower().startswith(("ecco", "nota", "spiegazione"))
    ]
    if len(verse_lines) >= 14:
        return "\n".join(verse_lines[:14]), False
    return "\n".join(verse_lines), False


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("planned teacher generation requires one CUDA GPU")
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
    plans = []
    skipped_plans = 0
    for index, opening in enumerate(openings):
        for seed in args.seeds:
            try:
                plan = plan_from_anchored(
                    str(opening["opening_line"]),
                    lexicon,
                    seed=int(seed) + index * 13,
                )
            except ValueError:
                skipped_plans += 1
                continue
            plans.append(
                {
                    "opening": opening,
                    "seed": int(seed),
                    "scheme": str(plan["scheme"]),
                    "words": [str(word) for word in plan["words"]],
                }
            )
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
    tokenizer.padding_side = "left"
    jobs = []
    for plan in plans:
        prompt_id = str(plan["opening"]["id"])
        path = output_dir / output_name(args.model_id, prompt_id, plan["seed"])
        if path.is_file():
            continue
        rendered = render_planned_prompt(tokenizer, str(plan["opening"]["opening_line"]), plan["words"])
        jobs.append((plan, path, rendered))
    print(
        f"planned-teacher | model={args.model_id} jobs={len(jobs)} "
        f"plans={len(plans)} skipped_plans={skipped_plans}",
        flush=True,
    )
    started = time.monotonic()
    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start : start + args.batch_size]
        inputs = tokenizer(
            [rendered for _, _, rendered in batch], return_tensors="pt", padding=True
        ).to(model.device)
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
        for (plan, path, rendered), output_ids, prompt_ids in zip(
            batch, generated, inputs["input_ids"], strict=True
        ):
            completion = output_ids[len(prompt_ids) :]
            raw = tokenizer.decode(completion, skip_special_tokens=True).strip()
            opening_line = str(plan["opening"]["opening_line"])
            text, opening_found = clean_sonnet(raw, opening_line)
            payload = {
                "generation_version": TEACHER_VERSION,
                "analysis_role": "planned_teacher_pilot",
                "condition": args.model_id,
                "teacher_model": args.model_id,
                "teacher_revision": args.revision,
                "prompt": {
                    "id": str(plan["opening"]["id"]),
                    "opening_line": opening_line,
                },
                "prompt_id": str(plan["opening"]["id"]),
                "opening_line": opening_line,
                "seed": int(plan["seed"]),
                "scheme": str(plan["scheme"]),
                "planned_words": list(plan["words"]),
                "shots": 0,
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
            f"planned-teacher | completed={min(start + args.batch_size, len(jobs))}"
            f"/{len(jobs)} elapsed={time.monotonic() - started:.0f}s",
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
        "generation_version": TEACHER_VERSION,
        "analysis_role": "planned_teacher_pilot",
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
        f"planned-teacher | complete outputs={len(outputs)} "
        f"elapsed={time.monotonic() - started:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
