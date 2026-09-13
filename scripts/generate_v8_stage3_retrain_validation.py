#!/usr/bin/env python3
"""Generate and merge the V8 Stage-3 retrain matched validation outputs."""

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

from sonnet_analysis.form_targeted_validation import (
    generate_single_system_validation,
    merge_matched_generation,
)
from sonnet_analysis.minerva_v7_exploratory_prompts import (
    validate_exploratory_prompt_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/v8_stage3_retrain.json",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, default=None)
    parser.add_argument("--merged-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def model_identity(model_dir: Path) -> str | None:
    candidate = model_dir / "model.safetensors"
    if not candidate.is_file():
        candidates = sorted(model_dir.glob("model-*.safetensors"))
        if not candidates:
            return None
        digest = hashlib.sha256()
        for path in candidates:
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()
    return hashlib.sha256(candidate.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads(args.config.read_text(encoding="utf-8"))
    evaluation = config["evaluation"]
    prompts = validate_exploratory_prompt_manifest(
        ROOT / evaluation["prompts"],
        expected_sha256=evaluation["prompts_sha256"],
    )["prompts"]
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("candidate generation requires exactly one CUDA GPU")
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir), local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    candidate_dir = args.candidate_dir or ROOT / evaluation["candidate_output_dir"]
    result = generate_single_system_validation(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        seeds=evaluation["seeds"],
        recipe=evaluation["recipe"],
        output_dir=candidate_dir,
        device=device,
        batch_size=args.batch_size,
        identity_sha256=model_identity(args.model_dir),
        progress=lambda message: print(
            f"v8-stage3-generate | {message}", flush=True
        ),
    )
    merged_dir = args.merged_dir or ROOT / evaluation["merged_output_dir"]
    merged = merge_matched_generation(
        baseline_dir=ROOT / evaluation["baseline_generation_dir"],
        candidate_dir=candidate_dir,
        output_dir=merged_dir,
    )
    print(
        "v8-stage3-generate | complete "
        f"candidate={result['completed_output_count']} "
        f"merged={merged['completed_output_count']} dir={merged_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
