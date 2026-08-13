#!/usr/bin/env python3
"""Generate the matched Stage-3 versus DPO validation grid."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_dpo_validation import generate_matched_validation
from sonnet_analysis.minerva_v7_exploratory_prompts import (
    validate_exploratory_prompt_manifest,
)
from sonnet_analysis.minerva_v7_runtime import (
    gpu_preflight, load_bf16_model_and_tokenizer, load_verified_state,
)
from sonnet_training.minerva_v7_ai_dpo import (
    TARGET_MODULES, load_dpo_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument(
        "--adapter", type=Path,
        default=Path("artifacts/local/minerva_7b_v7_dpo/training/best_adapter.pt"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/local/minerva_7b_v7_dpo/validation/generation"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hourly-rate", type=float, required=True)
    args = parser.parse_args()

    config = load_dpo_config(ROOT / "configs/minerva_7b_v7_ai_judged_dpo.json")
    prompts = validate_exploratory_prompt_manifest(
        ROOT / "configs/minerva_7b_v7_exploratory_prompts.json",
        expected_sha256="2f33aa518aa61c11193831e53b07fd3bd861a72bf68bb23c0e0e5b1a13b1d0c7",
    )["prompts"]
    state = load_verified_state(ROOT / args.state_audit, "stage_3_selected")
    adapter_path = ROOT / args.adapter
    adapter_identity = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    print(
        "minerva-v7-dpo | start job=matched_validation device=1xh100-80gb "
        "total_steps=960 progress_interval=one_batch prompts=120 seeds=4 systems=2",
        flush=True,
    )
    preflight = gpu_preflight(
        output_root=ROOT / args.output_dir,
        required_output_bytes=2 * 1024**3,
        hourly_rate=args.hourly_rate,
    )
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

    model, tokenizer = load_bf16_model_and_tokenizer(
        state=state, config=config, device=torch.device("cuda:0")
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05,
            bias="none", target_modules=list(TARGET_MODULES),
        ),
    )
    checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=True)
    if checkpoint.get("parent_state_identity_sha256") != state["state_identity_sha256"]:
        raise ValueError("DPO validation adapter parent mismatch")
    set_peft_model_state_dict(model, checkpoint["adapter_state_dict"])
    recipe = {
        "recipe_id": "no_labels_creative", "temperature": 0.85,
        "top_p": 0.95, "top_k": None, "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 4, "max_new_tokens": 512,
        "continuation_line_target": 13,
    }
    started = time.monotonic()
    result = generate_matched_validation(
        model=model, tokenizer=tokenizer, prompts=prompts,
        seeds=(5200, 5201, 5202, 5203), recipe=recipe,
        output_dir=ROOT / args.output_dir,
        state_identity=str(state["state_identity_sha256"]),
        adapter_identity=adapter_identity, device="cuda:0",
        batch_size=args.batch_size,
        progress=lambda message: print(f"minerva-v7-dpo | {message}", flush=True),
    )
    elapsed = time.monotonic() - started
    print(
        "minerva-v7-dpo | complete job=matched_validation "
        f"outputs={result['completed_output_count']} elapsed={elapsed:.1f}s "
        f"cost_usd={elapsed / 3600 * args.hourly_rate:.3f} "
        f"gpu={preflight['gpu_name']} v7_test_accessed=False",
        flush=True,
    )


if __name__ == "__main__":
    main()
