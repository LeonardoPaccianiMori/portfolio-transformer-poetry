#!/usr/bin/env python3
"""Dry-run or manually execute one state's high-volume BF16 generation grid."""

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

from sonnet_analysis.minerva_v7_exploratory_prompts import validate_exploratory_prompt_manifest
from sonnet_analysis.minerva_v7_high_volume_generation import (
    generate_high_volume_state, load_high_volume_config,
)
from sonnet_analysis.minerva_v7_runtime import (
    gpu_preflight, load_bf16_model_and_tokenizer, load_verified_state,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/minerva_7b_v7_high_volume_generation.json"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/local/minerva_7b_v7_analysis/high_volume_generation"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hourly-rate", type=float, required=True)
    parser.add_argument("--qualification-one-batch", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_high_volume_config(args.config)
    prompt_path = Path(config["prompt_manifest_path"])
    prompts = validate_exploratory_prompt_manifest(
        prompt_path, expected_sha256=config["prompt_manifest_sha256"]
    )["prompts"]
    state = load_verified_state(args.state_audit, args.state_id)
    batch_size = args.batch_size or int(config["default_batch_size"])
    planned = batch_size if args.qualification_one_batch else int(config["outputs_per_state"])
    print(
        f"minerva-v7-research | start job=high_volume_generation state={args.state_id} "
        f"execute={args.execute} outputs={planned} batch_size={batch_size} "
        f"scope={'qualification' if args.qualification_one_batch else 'full'}",
        flush=True,
    )
    if not args.execute:
        print(
            "minerva-v7-research | dry_run_complete no_model_loaded=True "
            "confirmatory_grid_unchanged=True",
            flush=True,
        )
        return
    destination = (
        args.output_root / "qualification" / args.state_id
        if args.qualification_one_batch else args.output_root / args.state_id
    )
    preflight = gpu_preflight(
        output_root=destination, required_output_bytes=2 * 1024**3,
        hourly_rate=args.hourly_rate,
    )
    import torch

    model, tokenizer = load_bf16_model_and_tokenizer(
        state=state, config={"research_version": "high_volume"},
        device=torch.device("cuda:0"),
    )
    started = time.monotonic()
    result = generate_high_volume_state(
        model=model, tokenizer=tokenizer, state_id=args.state_id,
        state_identity_sha256=str(state["state_identity_sha256"]),
        prompts=prompts, seeds=config["seeds"], recipes=config["recipes"],
        output_dir=destination, device="cuda:0", batch_size=batch_size,
        maximum_batches=1 if args.qualification_one_batch else None,
        progress=lambda message: print(
            f"minerva-v7-research | state={args.state_id} {message}", flush=True
        ),
    )
    elapsed = time.monotonic() - started
    completed = int(result["completed_output_count"])
    per_output = elapsed / max(completed, 1)
    projected = per_output * int(config["outputs_per_state"])
    print(
        f"minerva-v7-research | complete completed={completed} elapsed={elapsed:.1f}s "
        f"measured_seconds_per_output={per_output:.3f} "
        f"projected_state_hours={projected/3600:.2f} "
        f"projected_state_cost_usd={projected/3600*args.hourly_rate:.2f} "
        f"gpu={preflight['gpu_name']}", flush=True,
    )


if __name__ == "__main__":
    main()
