#!/usr/bin/env python3
"""Generate stage-two poems from the valid stage-one plans."""

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

from sonnet_analysis.few_shot_context_validation import load_context_records
from sonnet_analysis.minerva_v7_exploratory_prompts import (
    validate_exploratory_prompt_manifest,
)
from sonnet_analysis.minerva_v7_high_volume_generation import generate_batch
from sonnet_analysis.plan_then_poem_validation import planned_prompt
from sonnet_training.reasoning_trace_sft import (
    parse_trace,
    planned_words_from_trace,
    validate_trace,
)
from sonnet_training.self_play_rft import output_name

STAGE2_VERSION = "plan_generator_stage2_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-dir",
        type=Path,
        default=ROOT / "artifacts/local/plan_generator/validation/generation",
    )
    parser.add_argument(
        "--source-arms",
        nargs="+",
        default=["plan_baseline", "plan_sft"],
    )
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/merged_sft"))
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/plan_generator_sft.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/local/plan_generator/validation/stage2",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("stage-two generation requires one CUDA GPU")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    recipe = dict(config["generation"]["stage2_recipe"])
    prompts = validate_exploratory_prompt_manifest(
        ROOT / "configs/minerva_7b_v7_exploratory_prompts.json",
        expected_sha256="2f33aa518aa61c11193831e53b07fd3bd861a72bf68bb23c0e0e5b1a13b1d0c7",
    )["prompts"]
    prompt_by_id = {str(prompt["id"]): dict(prompt) for prompt in prompts}
    records = load_context_records(args.stage1_dir)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_dir), local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    identity = hashlib.sha256()
    for path in sorted(args.model_dir.glob("model-*.safetensors")):
        identity.update(hashlib.sha256(path.read_bytes()).digest())
    jobs = []
    stats = {
        "source_records": 0,
        "plan_invalid": 0,
        "plan_valid": 0,
        "jobs": 0,
    }
    for record in records:
        if str(record["system_id"]) not in args.source_arms:
            continue
        stats["source_records"] += 1
        prompt = prompt_by_id[str(record["prompt_id"])]
        opening = str(prompt["opening_line"])
        parsed = parse_trace(str(record["text"]))
        if parsed is None:
            stats["plan_invalid"] += 1
            continue
        trace = validate_trace(parsed, opening)
        words = planned_words_from_trace(parsed, opening)
        if not trace["valid"] or words is None:
            stats["plan_invalid"] += 1
            continue
        stats["plan_valid"] += 1
        condition = f"stage2_{record['system_id']}"
        rendered = planned_prompt(tokenizer, opening, words)
        path = output_dir / output_name(
            condition, str(record["prompt_id"]), int(record["seed"]), version=STAGE2_VERSION
        )
        if path.is_file():
            continue
        jobs.append(
            {
                "prompt": prompt,
                "seed": int(record["seed"]),
                "rendered_prompt": rendered,
                "planned_words": words,
                "condition": condition,
            }
        )
    stats["jobs"] = len(jobs)
    print(f"plan-stage2 | jobs={len(jobs)} stats={stats}", flush=True)
    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start : start + args.batch_size]
        results = generate_batch(
            model=model,
            tokenizer=tokenizer,
            jobs=batch,
            recipe=recipe,
            device=device,
        )
        for job, result in zip(batch, results, strict=True):
            payload = {
                "generation_version": STAGE2_VERSION,
                "analysis_role": "plan_generator_stage2",
                "condition": job["condition"],
                "prompt": job["prompt"],
                "planned_words": job["planned_words"],
                "recipe": dict(recipe),
                "model_identity_sha256": identity.hexdigest() or None,
                **result,
                "v7_test_accessed": False,
            }
            path = output_dir / output_name(
                job["condition"],
                str(job["prompt"]["id"]),
                int(job["seed"]),
                version=STAGE2_VERSION,
            )
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        print(
            f"plan-stage2 | completed={min(start + args.batch_size, len(jobs))}/{len(jobs)}",
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
                "prompt_id": str(payload["prompt"]["id"]),
                "seed": int(payload["seed"]),
            }
        )
    complete = {
        "generation_version": STAGE2_VERSION,
        "analysis_role": "plan_generator_stage2",
        "arms": sorted({row["condition"] for row in outputs}),
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
        f"plan-stage2 | complete outputs={len(outputs)} "
        f"model={identity.hexdigest()[:12]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
