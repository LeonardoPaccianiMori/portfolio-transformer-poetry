#!/usr/bin/env python3
"""Dry-run, qualify, or execute the Stage-3 prompt/stopping experiment."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_prompt_intervention import (
    generate_prompt_intervention,
    load_experiment_prompts,
    load_prompt_intervention_config,
)
from sonnet_analysis.minerva_v7_runtime import (
    gpu_preflight,
    load_bf16_model_and_tokenizer,
    load_verified_state,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/minerva_7b_v7_stage_3_prompt_intervention.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/local/minerva_7b_v7_stage_3_prompt_intervention"),
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hourly-rate", type=float, required=True)
    parser.add_argument("--qualification-one-batch-per-arm", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = load_prompt_intervention_config(args.config)
    prompts = load_experiment_prompts(config)
    state = load_verified_state(args.state_audit, str(config["state_id"]))
    batch_size = args.batch_size or int(config["default_batch_size"])
    scope = "qualification" if args.qualification_one_batch_per_arm else "full"
    print(
        "minerva-v7-research | start job=stage_3_prompt_intervention "
        f"state={config['state_id']} execute={args.execute} scope={scope} "
        f"final_outputs={config['final_outputs']} maximum_attempts={config['maximum_model_attempts']} "
        f"batch_size={batch_size}",
        flush=True,
    )
    if not args.execute:
        print(
            "minerva-v7-research | dry_run_complete no_model_loaded=True "
            "frozen_high_volume_grid_unchanged=True v7_test_accessed=False",
            flush=True,
        )
        return

    destination = (
        args.output_root / "qualification"
        if args.qualification_one_batch_per_arm
        else args.output_root / "authoritative"
    )
    preflight = gpu_preflight(
        output_root=destination,
        required_output_bytes=2 * 1024**3,
        hourly_rate=args.hourly_rate,
    )
    import torch

    model, tokenizer = load_bf16_model_and_tokenizer(
        state=state,
        config=config,
        device=torch.device("cuda:0"),
    )
    started = time.monotonic()
    result = generate_prompt_intervention(
        model=model,
        tokenizer=tokenizer,
        state_identity_sha256=str(state["state_identity_sha256"]),
        prompts=prompts,
        config=config,
        output_dir=destination,
        device="cuda:0",
        batch_size=batch_size,
        maximum_batches_per_arm=(
            int(config["qualification_batches_per_arm"])
            if args.qualification_one_batch_per_arm
            else None
        ),
        progress=lambda message: print(
            f"minerva-v7-research | state=stage_3_selected {message}", flush=True
        ),
    )
    elapsed = time.monotonic() - started
    print(
        "minerva-v7-research | complete "
        f"final_outputs={result['completed_final_output_count']} elapsed={elapsed:.1f}s "
        f"cost_usd={elapsed / 3600 * args.hourly_rate:.2f} gpu={preflight['gpu_name']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
