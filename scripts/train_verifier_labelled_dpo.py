#!/usr/bin/env python3
"""Train the verifier-labelled form LoRA-DPO adapter on Stage-3."""

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
from sonnet_training.minerva_v7_ai_dpo import PARENT_IDENTITY, train_ai_judged_dpo
from sonnet_training.verifier_labelled_dpo import (
    CHECKPOINT_PREFIX,
    EXPERIMENT_VERSION,
    load_verifier_dpo_config,
    load_verifier_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/verifier_labelled_dpo.json",
    )
    parser.add_argument(
        "--preferences",
        type=Path,
        default=ROOT / "artifacts/local/verifier_labelled_dpo/preferences.frozen.json",
    )
    parser.add_argument(
        "--state-audit",
        type=Path,
        default=ROOT / "artifacts/local/verifier_labelled_dpo/state_audit.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/local/verifier_labelled_dpo/training",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_verifier_dpo_config(args.config)
    examples = load_verifier_examples(args.preferences)
    state = load_verified_state(args.state_audit, "stage_3_selected")
    if state["state_identity_sha256"] != PARENT_IDENTITY:
        raise ValueError("verifier DPO parent is not the exact Stage-3 state")
    print(
        "minerva-verifier-dpo | start job=train_verifier_form_lora_dpo "
        f"pairs={len(examples)} device=1xh100-80gb dry_run={args.dry_run}",
        flush=True,
    )
    if args.dry_run:
        print(
            "minerva-verifier-dpo | dry_run_complete training_started=False "
            "v7_test_accessed=False",
            flush=True,
        )
        return
    started = time.monotonic()
    result = train_ai_judged_dpo(
        repo_root=ROOT,
        config=config,
        examples=examples,
        state=state,
        output_dir=args.output_dir,
        qualification=False,
        experiment_version=EXPERIMENT_VERSION,
        checkpoint_prefix=CHECKPOINT_PREFIX,
        progress=lambda message: print(
            f"minerva-verifier-dpo | {message}", flush=True
        ),
    )
    print(
        "minerva-verifier-dpo | complete "
        f"elapsed={time.monotonic() - started:.1f}s "
        f"cost_usd={result['cost_usd']:.3f} "
        f"peak_gpu_memory_gib={result['peak_gpu_memory_bytes'] / 1024**3:.2f} "
        "v7_test_accessed=False",
        flush=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
