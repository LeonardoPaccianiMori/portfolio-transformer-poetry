#!/usr/bin/env python3
"""Generate one reasoning-trace arm for evaluation."""

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
from sonnet_training.reasoning_trace_sft import (
    TRACE_ARMS,
    generate_trace_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=TRACE_ARMS, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/reasoning_trace_sft.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/local/reasoning_trace/validation/generation",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("reasoning-trace generation requires one CUDA GPU")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = args.seeds or [int(seed) for seed in config["generation"]["seeds"]]
    recipe = dict(config["generation"]["recipe"])
    prompts = validate_exploratory_prompt_manifest(
        ROOT / "configs/minerva_7b_v7_exploratory_prompts.json",
        expected_sha256="2f33aa518aa61c11193831e53b07fd3bd861a72bf68bb23c0e0e5b1a13b1d0c7",
    )["prompts"]
    device = torch.device("cuda:0")
    tokenizer_dir = args.tokenizer_dir or args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True
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
    result = generate_trace_validation(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        arm=args.arm,
        seeds=seeds,
        recipe=recipe,
        output_dir=ROOT / args.output_dir,
        device=device,
        batch_size=args.batch_size,
        model_identity=identity.hexdigest() or None,
        progress=lambda message: print(f"trace-gen | {message}", flush=True),
    )
    print(
        f"trace-gen | complete arm={args.arm} "
        f"outputs={result['completed_output_count']} "
        f"model={identity.hexdigest()[:12]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
