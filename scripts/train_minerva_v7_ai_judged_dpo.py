#!/usr/bin/env python3
"""Qualify or train the bounded Minerva V7 AI-judged LoRA-DPO adapter."""

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

from sonnet_analysis.minerva_v7_runtime import load_verified_state
from sonnet_training.minerva_v7_ai_dpo import (
    PARENT_IDENTITY,
    build_training_plan,
    load_ai_majority_examples,
    load_dpo_config,
    train_ai_judged_dpo,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/minerva_7b_v7_ai_judged_dpo.json"),
    )
    parser.add_argument(
        "--preferences", type=Path,
        default=Path(
            "artifacts/local/minerva_7b_v7_dpo/review/authoritative/"
            "ai_majority_preferences.frozen.json"
        ),
    )
    parser.add_argument(
        "--state-audit", type=Path,
        default=Path(
            "artifacts/local/minerva_7b_v7_analysis/state_audit/"
            "seven_state_audit.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/local/minerva_7b_v7_dpo/training"),
    )
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_dpo_config(ROOT / args.config)
    examples = load_ai_majority_examples(ROOT / args.preferences)
    state = load_verified_state(ROOT / args.state_audit, "stage_3_selected")
    if state["state_identity_sha256"] != PARENT_IDENTITY:
        raise ValueError("DPO parent state identity mismatch")
    plan = build_training_plan(examples, config=config)
    mode = "qualification" if args.qualification else "authoritative"
    print(
        "minerva-v7-dpo | start job=train_ai_judged_lora_dpo "
        f"mode={mode} device=1xh100-80gb total_steps="
        f"{1 if args.qualification else plan['total_steps']} "
        f"progress_interval={config['progress_interval']} "
        f"train_pairs={len(plan['train'])} validation_pairs={len(plan['validation'])} "
        f"context={config['context_length']} accumulation="
        f"{config['gradient_accumulation_steps']} dry_run={args.dry_run}",
        flush=True,
    )
    if args.dry_run:
        print(
            "minerva-v7-dpo | dry_run_complete model_loaded=False training_started=False "
            "v7_test_accessed=False",
            flush=True,
        )
        return
    started = time.monotonic()
    output_dir = ROOT / args.output_dir
    if args.qualification:
        output_dir = output_dir / "qualification"
    result = train_ai_judged_dpo(
        repo_root=ROOT,
        config=config,
        examples=examples,
        state=state,
        output_dir=output_dir,
        qualification=args.qualification,
        resume_from=(ROOT / args.resume_from if args.resume_from else None),
        progress=lambda message: print(f"minerva-v7-dpo | {message}", flush=True),
    )
    print(
        "minerva-v7-dpo | complete "
        f"mode={mode} elapsed={time.monotonic() - started:.1f}s "
        f"cost_usd={result['cost_usd']:.3f} "
        f"peak_gpu_memory_gib={result['peak_gpu_memory_bytes'] / 1024**3:.2f} "
        "v7_test_accessed=False",
        flush=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
