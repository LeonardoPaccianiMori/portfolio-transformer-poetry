#!/usr/bin/env python3
"""Generate the zero, one, and three shot context A/B grids."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.constrained_plan_validation import plan_from_anchored
from sonnet_analysis.few_shot_context_validation import (
    build_context_examples,
    generate_context_validation,
)
from sonnet_analysis.minerva_v7_exploratory_prompts import (
    validate_exploratory_prompt_manifest,
)
from sonnet_evaluation.rhyme_lexicon import load_lexicon
from sonnet_training.form_targeted_data import load_train_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=ROOT / "data/metadata/rhyme_lexicon_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/local/few_shot_context/validation/generation",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[5200])
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("context A/B generation requires one CUDA GPU")
    prompts = validate_exploratory_prompt_manifest(
        ROOT / "configs/minerva_7b_v7_exploratory_prompts.json",
        expected_sha256="2f33aa518aa61c11193831e53b07fd3bd861a72bf68bb23c0e0e5b1a13b1d0c7",
    )["prompts"]
    lexicon = load_lexicon(args.lexicon)
    anchored_plans = {}
    skipped = 0
    for index, prompt in enumerate(prompts):
        try:
            anchored_plans[str(prompt["id"])] = plan_from_anchored(
                str(prompt["opening_line"]), lexicon, seed=6200 + index
            )
        except ValueError:
            skipped += 1
    rows = load_train_rows(
        ROOT / "data/processed/sonnets_expanded_v8/sonnets_manifest.csv"
    )
    prompt_hashes = {
        str(prompt.get("source_logical_sha256", "")) for prompt in prompts
    }
    examples = build_context_examples(
        rows, ROOT, limit=3, excluded_logical_sha256=prompt_hashes
    )
    device = torch.device("cuda:0")
    tokenizer_dir = args.tokenizer_dir or args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True
    )
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
    recipe = {
        "temperature": 0.85,
        "top_k": None,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 4,
        "max_new_tokens": 512,
        "continuation_line_target": 13,
    }
    print(
        f"context-ab | plans={len(anchored_plans)} skipped={skipped} "
        f"examples={[example['unit_id'] for example in examples]}",
        flush=True,
    )
    result = generate_context_validation(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompt for prompt in prompts if str(prompt["id"]) in anchored_plans],
        plans=anchored_plans,
        examples=examples,
        seeds=args.seeds,
        recipe=recipe,
        output_dir=ROOT / args.output_dir,
        device=device,
        batch_size=args.batch_size,
        model_identity=identity.hexdigest() or None,
        progress=lambda message: print(f"context-ab | {message}", flush=True),
    )
    print(
        "context-ab | complete "
        f"outputs={result['completed_output_count']} "
        f"model={identity.hexdigest()[:12]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
