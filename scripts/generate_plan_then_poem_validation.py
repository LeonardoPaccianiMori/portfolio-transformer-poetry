#!/usr/bin/env python3
"""Generate the planned and control validation grids for the plan test."""

from __future__ import annotations

import argparse
import hashlib
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
from sonnet_analysis.minerva_v7_runtime import (
    gpu_preflight,
    load_bf16_model_and_tokenizer,
    load_verified_state,
)
from sonnet_analysis.plan_then_poem_validation import (
    build_plans,
    generate_plan_validation,
)
from sonnet_evaluation.rhyme_lexicon import load_lexicon
from sonnet_training.minerva_v7_ai_dpo import TARGET_MODULES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-audit", type=Path, default=None)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="load a merged model directly instead of Stage-3 plus an adapter",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=ROOT / "artifacts/local/verifier_labelled_dpo/training/best_adapter.pt",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=ROOT / "data/metadata/rhyme_lexicon_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/local/plan_then_poem/validation/generation",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[5200, 5201])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hourly-rate", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.state_audit is None and args.model_dir is None:
        raise SystemExit("pass either --state-audit or --model-dir")
    prompts = validate_exploratory_prompt_manifest(
        ROOT / "configs/minerva_7b_v7_exploratory_prompts.json",
        expected_sha256="2f33aa518aa61c11193831e53b07fd3bd861a72bf68bb23c0e0e5b1a13b1d0c7",
    )["prompts"]
    lexicon = load_lexicon(args.lexicon)
    plans, control_words = build_plans(prompts, lexicon)
    device = torch.device("cuda:0")
    if args.model_dir is not None:
        model_dir = args.model_dir
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        adapter_identity = None
        preflight = {"gpu_name": torch.cuda.get_device_name(device)}
    else:
        state = load_verified_state(args.state_audit, "stage_3_selected")
        preflight = gpu_preflight(
            output_root=ROOT / args.output_dir,
            required_output_bytes=1024**3,
            hourly_rate=args.hourly_rate,
        )
        adapter_path = (
            ROOT / args.adapter if not args.adapter.is_absolute() else args.adapter
        )
        adapter_identity = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
        model, tokenizer = load_bf16_model_and_tokenizer(
            state=state, config={}, device=device
        )
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                target_modules=list(TARGET_MODULES),
            ),
        )
        checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("parent_state_identity_sha256")
            != state["state_identity_sha256"]
        ):
            raise ValueError("verifier DPO adapter parent mismatch")
        set_peft_model_state_dict(model, checkpoint["adapter_state_dict"])
        model.eval()
    recipe = {
        "temperature": 0.85,
        "top_k": None,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 4,
        "max_new_tokens": 512,
        "continuation_line_target": 13,
    }
    result = generate_plan_validation(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        plans=plans,
        control_words=control_words,
        seeds=args.seeds,
        recipe=recipe,
        output_dir=ROOT / args.output_dir,
        device=device,
        batch_size=args.batch_size,
        adapter_identity=adapter_identity,
        progress=lambda message: print(
            f"plan-then-poem | {message}", flush=True
        ),
    )
    print(
        "plan-then-poem | complete "
        f"outputs={result['completed_output_count']}/"
        f"{result['planned_output_count']} "
        f"gpu={preflight['gpu_name']} adapter={adapter_identity[:12]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
