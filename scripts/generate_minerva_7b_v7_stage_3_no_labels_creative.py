#!/usr/bin/env python3
"""Dry-run, qualify, or execute the Stage-3 no-labels/creative cell."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_no_labels_creative import (
    generate_no_labels_creative,
    load_no_labels_creative_config,
    load_no_labels_creative_prompts,
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
        default=Path("configs/minerva_7b_v7_stage_3_no_labels_creative.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/local/minerva_7b_v7_stage_3_no_labels_creative"),
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hourly-rate", type=float, required=True)
    parser.add_argument("--qualification-eight-outputs", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = load_no_labels_creative_config(args.config)
    prompts = load_no_labels_creative_prompts(config)
    state = load_verified_state(args.state_audit, str(config["state_id"]))
    batch_size = args.batch_size or int(config["default_batch_size"])
    qualification = args.qualification_eight_outputs
    planned = int(config["qualification_output_count"] if qualification else config["final_outputs"])
    print(
        "minerva-v7-research | start job=stage_3_no_labels_creative "
        f"state={config['state_id']} execute={args.execute} "
        f"scope={'qualification' if qualification else 'full'} "
        f"outputs={planned} batch_size={batch_size} progress_interval=one_batch",
        flush=True,
    )
    if not args.execute:
        print(
            "minerva-v7-research | dry_run_complete no_model_loaded=True "
            "v7_test_accessed=False training_performed=False",
            flush=True,
        )
        return

    destination = args.output_root / ("qualification" if qualification else "authoritative")
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
    result = generate_no_labels_creative(
        model=model,
        tokenizer=tokenizer,
        state_identity_sha256=str(state["state_identity_sha256"]),
        prompts=prompts,
        config=config,
        output_dir=destination,
        device="cuda:0",
        batch_size=batch_size,
        maximum_outputs=planned if qualification else None,
        progress=lambda message: print(
            f"minerva-v7-research | state=stage_3_selected {message}", flush=True
        ),
    )
    elapsed = time.monotonic() - started
    completed = int(result["completed_output_count"])
    projected_seconds = elapsed / max(completed, 1) * int(config["final_outputs"])
    print(
        "minerva-v7-research | complete "
        f"completed={completed} elapsed={elapsed:.1f}s "
        f"projected_full_minutes={projected_seconds / 60:.1f} "
        f"projected_full_cost_usd={projected_seconds / 3600 * args.hourly_rate:.2f} "
        f"gpu={preflight['gpu_name']} v7_test_accessed=False training_performed=False",
        flush=True,
    )


if __name__ == "__main__":
    main()
