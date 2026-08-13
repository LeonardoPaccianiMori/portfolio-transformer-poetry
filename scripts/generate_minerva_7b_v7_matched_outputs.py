#!/usr/bin/env python3
"""Dry-run or manually generate one verified V7 state's matched BF16 outputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_generation import generate_state_outputs
from sonnet_analysis.minerva_v7_runtime import (
    gpu_preflight, load_bf16_model_and_tokenizer, load_research_config,
    load_verified_state, sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--research-config", type=Path, default=Path("configs/minerva_7b_v7_research.json"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/local/minerva_7b_v7_analysis/matched_generation"))
    parser.add_argument("--hourly-rate", type=float, required=True)
    parser.add_argument("--qualification-one-output", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_research_config(args.research_config)
    state = load_verified_state(args.state_audit, args.state_id)
    prompt_path = Path(config["prompt_path"])
    if sha256_file(prompt_path) != config["prompt_sha256"]:
        raise ValueError("matched-generation prompt hash mismatch")
    prompts = json.loads(prompt_path.read_text(encoding="utf-8"))
    seeds = [config["generation"]["confirmatory_seed"], *config["generation"]["exploratory_replication_seeds"]]
    count = 1 if args.qualification_one_output else len(prompts) * len(seeds)
    print(
        f"minerva-v7-research | start job=matched_generation state={args.state_id} "
        f"execute={args.execute} outputs={count} seeds={seeds}", flush=True,
    )
    if not args.execute:
        print("minerva-v7-research | dry_run_complete no_model_loaded=True", flush=True)
        return
    # Generation JSON/text is small; the 40-GiB minimum is chiefly atomic model-study headroom.
    preflight = gpu_preflight(output_root=args.output_root, required_output_bytes=1024**3, hourly_rate=args.hourly_rate)
    import torch

    model, tokenizer = load_bf16_model_and_tokenizer(state=state, config=config, device=torch.device("cuda:0"))
    if args.qualification_one_output:
        prompts = prompts[:1]
        seeds = seeds[:1]
        destination = args.output_root / "qualification" / args.state_id
    else:
        destination = args.output_root / args.state_id
    started = time.monotonic()
    completion = generate_state_outputs(
        model=model, tokenizer=tokenizer, state_id=args.state_id,
        state_identity_sha256=str(state["state_identity_sha256"]),
        prompts=prompts, seeds=seeds,
        recipe={**config["generation"], "top_k": config["generation"]["top_k"]},
        output_dir=destination, device="cuda:0",
        progress=lambda message: print(f"minerva-v7-research | state={args.state_id} {message}", flush=True),
    )
    elapsed = time.monotonic() - started
    print(
        f"minerva-v7-research | complete state={args.state_id} outputs={completion['output_count']} "
        f"elapsed={elapsed:.1f}s cost_usd={elapsed/3600*args.hourly_rate:.2f} gpu={preflight['gpu_name']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
