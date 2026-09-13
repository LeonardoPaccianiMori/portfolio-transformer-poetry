#!/usr/bin/env python3
"""Print the frozen V8 Stage-3 retrain window plan without a GPU."""

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

from sonnet_training.v8_stage3_retrain_data import load_plan_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/v8_stage3_retrain.json",
    )
    parser.add_argument("--train-shard", type=Path, default=None)
    parser.add_argument("--replay-shard", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = config["data"]
    train_shard = args.train_shard or ROOT / data["v8_train_shard"]
    replay_shard = args.replay_shard or ROOT / data["replay_shard"]
    for path in (train_shard, replay_shard):
        if not path.is_file():
            raise FileNotFoundError(path)
    train, replay, plan, summary = load_plan_inputs(
        train_shard=train_shard,
        replay_shard=replay_shard,
        config=config,
    )
    output = {
        **summary,
        "train_shard": str(train_shard),
        "train_shard_sha256": hashlib.sha256(train_shard.read_bytes()).hexdigest(),
        "replay_shard": str(replay_shard),
        "replay_shard_sha256": hashlib.sha256(replay_shard.read_bytes()).hexdigest(),
        "train_tokens": int(train.size),
        "replay_tokens": int(replay.size),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
