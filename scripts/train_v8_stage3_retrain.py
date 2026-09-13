#!/usr/bin/env python3
"""Run the V8 Stage-3 full-weight retrain on one H100."""

from __future__ import annotations

import argparse
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
from sonnet_training.v8_stage3_retrain import train_v8_stage3_retrain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/v8_stage3_retrain.json",
    )
    parser.add_argument("--stage2-dir", type=Path, default=None)
    parser.add_argument("--token-shard", type=Path, default=None)
    parser.add_argument("--replay-shard", type=Path, default=None)
    parser.add_argument("--validation-shard", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = config["data"]
    training = config["training"]
    evaluation = config["evaluation"]
    prompts = validate_exploratory_prompt_manifest(
        ROOT / evaluation["prompts"],
        expected_sha256=evaluation["prompts_sha256"],
    )["prompts"]
    report = train_v8_stage3_retrain(
        stage2_dir=args.stage2_dir or ROOT / training["stage2_dir"],
        token_shard_path=args.token_shard or ROOT / data["v8_train_shard"],
        replay_shard_path=args.replay_shard or ROOT / data["replay_shard"],
        validation_shard_path=(
            args.validation_shard or ROOT / data["v8_validation_shard"]
        ),
        output_dir=args.output_dir or ROOT / training["output_dir"],
        config=config,
        prompts=prompts,
        progress=lambda message: print(
            f"v8-stage3-retrain | {message}", flush=True
        ),
    )
    print(
        "v8-stage3-retrain | complete "
        f"updates={report['updates_completed']}/{report['updates_planned']} "
        f"aborted={report['aborted']} "
        f"retention_gate_passed={report['retention_gate_passed']} "
        f"cost_usd={report['estimated_cost_usd']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
