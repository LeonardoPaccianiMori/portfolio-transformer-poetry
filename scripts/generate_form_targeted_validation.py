#!/usr/bin/env python3
"""Generate the matched baseline and candidate validation grid for the pilot."""

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

from sonnet_analysis.form_targeted_validation import generate_form_targeted_validation
from sonnet_analysis.minerva_v7_exploratory_prompts import (
    validate_exploratory_prompt_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/form_targeted_lora_pilot.json",
    )
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads(args.config.read_text(encoding="utf-8"))
    evaluation = config["evaluation"]
    prompts = validate_exploratory_prompt_manifest(
        ROOT / evaluation["prompts"],
        expected_sha256=evaluation["prompts_sha256"],
    )["prompts"]
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("pilot validation requires exactly one CUDA GPU")
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.base_model_dir), local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model_dir),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    model = PeftModel.from_pretrained(model, str(args.adapter_dir))
    model.eval()
    adapter_file = args.adapter_dir / "adapter_model.safetensors"
    adapter_identity = hashlib.sha256(adapter_file.read_bytes()).hexdigest()
    output_dir = args.output_dir or ROOT / evaluation["output_dir"]
    result = generate_form_targeted_validation(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        seeds=evaluation["seeds"],
        recipe=evaluation["recipe"],
        output_dir=output_dir,
        adapter_identity=adapter_identity,
        device=device,
        batch_size=args.batch_size,
        progress=lambda message: print(f"form-targeted-generate | {message}", flush=True),
    )
    print(
        "form-targeted-generate | complete "
        f"outputs={result['completed_output_count']}/{result['planned_output_count']} "
        f"adapter={adapter_identity[:12]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
