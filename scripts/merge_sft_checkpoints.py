#!/usr/bin/env python3
"""Rebuild the merged SFT model from Stage-3 and the two saved adapters."""

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

from sonnet_analysis.minerva_v7_runtime import load_verified_state
from sonnet_training.plan_following_sft import TARGET_MODULES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--verifier-adapter", type=Path, required=True)
    parser.add_argument("--plan-adapter", type=Path, required=True)
    parser.add_argument("--state-audit", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model_dir),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    verifier = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=list(TARGET_MODULES),
    )
    model = get_peft_model(model, verifier)
    checkpoint = torch.load(
        args.verifier_adapter, map_location="cpu", weights_only=True
    )
    if args.state_audit is not None:
        state = load_verified_state(args.state_audit, "stage_3_selected")
        if (
            checkpoint.get("parent_state_identity_sha256")
            != state["state_identity_sha256"]
        ):
            raise ValueError("verifier adapter parent mismatch")
    set_peft_model_state_dict(model, checkpoint["adapter_state_dict"])
    model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, str(args.plan_adapter))
    model = model.merge_and_unload()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    report = {
        "base_model_dir": str(args.base_model_dir),
        "verifier_adapter": str(args.verifier_adapter),
        "plan_adapter": str(args.plan_adapter),
        "output_dir": str(args.output_dir),
        "merged_files": sorted(
            path.name for path in args.output_dir.glob("*.safetensors")
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
