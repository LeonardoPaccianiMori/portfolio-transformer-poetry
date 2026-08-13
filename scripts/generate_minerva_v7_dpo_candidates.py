#!/usr/bin/env python3
"""Dry-run, qualify, or generate the bounded training-only DPO candidates."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_dpo_preferences import (
    generate_preference_candidates,
    load_preference_config,
    validate_training_prompt_manifest,
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
        "--config", type=Path,
        default=Path("configs/minerva_7b_v7_dpo_preferences.json"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("artifacts/local/minerva_7b_v7_dpo/candidates"),
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hourly-rate", type=float, required=True)
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = load_preference_config(args.config)
    prompt_path = ROOT / str(config["prompt_manifest_path"])
    manifest = validate_training_prompt_manifest(
        prompt_path, expected_sha256=str(config["prompt_manifest_sha256"])
    )
    state = load_verified_state(args.state_audit, str(config["state_id"]))
    if state.get("state_identity_sha256") != config["state_identity_sha256"]:
        raise ValueError("DPO generation state identity mismatch")
    planned = int(
        config["qualification_candidate_count"]
        if args.qualification else config["candidate_count"]
    )
    batch_size = args.batch_size or int(config["default_batch_size"])
    print(
        "minerva-v7-dpo | start job=generate_candidates device=1xh100-80gb "
        f"scope={'qualification' if args.qualification else 'full'} "
        f"total_steps={planned} progress_interval={batch_size} execute={args.execute}",
        flush=True,
    )
    if not args.execute:
        print(
            "minerva-v7-dpo | dry_run_complete no_model_loaded=True "
            "source_split=train v7_test_accessed=False training_performed=False",
            flush=True,
        )
        return

    destination = args.output_root / (
        "qualification" if args.qualification else "authoritative"
    )
    preflight = gpu_preflight(
        output_root=destination,
        required_output_bytes=4 * 1024**3,
        hourly_rate=args.hourly_rate,
    )
    import torch

    model, tokenizer = load_bf16_model_and_tokenizer(
        state=state, config=config, device=torch.device("cuda:0")
    )
    started = time.monotonic()
    result = generate_preference_candidates(
        model=model, tokenizer=tokenizer, prompts=manifest["prompts"],
        config=config, output_dir=destination, device="cuda:0",
        batch_size=batch_size,
        maximum_candidates=planned if args.qualification else None,
        progress=lambda message: print(f"minerva-v7-dpo | {message}", flush=True),
    )
    elapsed = time.monotonic() - started
    completed = int(result["completed_candidate_count"])
    projected_seconds = elapsed / max(completed, 1) * int(config["candidate_count"])
    print(
        "minerva-v7-dpo | complete "
        f"completed={completed} elapsed={elapsed:.1f}s "
        f"projected_full_minutes={projected_seconds / 60:.1f} "
        f"projected_full_cost_usd={projected_seconds / 3600 * args.hourly_rate:.2f} "
        f"gpu={preflight['gpu_name']} source_split=train "
        "v7_test_accessed=False training_performed=False",
        flush=True,
    )


if __name__ == "__main__":
    main()
