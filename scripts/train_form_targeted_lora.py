#!/usr/bin/env python3
"""Run the one-arm form-targeted LoRA pilot on the V8 train sonnets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.form_targeted_lora import train_form_targeted_lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/form_targeted_lora_pilot.json",
    )
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--token-shard", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_config = config["data"]
    token_shard = (
        args.token_shard
        or ROOT / data_config["output_dir"] / "tokens-00000.int32.bin"
    )
    output_dir = args.output_dir or ROOT / config["training"]["output_dir"]
    print(
        "form-targeted-train | start "
        f"base={args.base_model_dir} shard={token_shard}",
        flush=True,
    )
    report = train_form_targeted_lora(
        base_model_dir=args.base_model_dir,
        token_shard_path=token_shard,
        output_dir=output_dir,
        config=config,
        progress=lambda message: print(f"form-targeted-train | {message}", flush=True),
    )
    print(
        "form-targeted-train | complete "
        f"steps={report['completed_steps']}/{report['planned_steps']} "
        f"tokens={report['tokens_seen']} loss={report['final_loss']} "
        f"adapter={report['adapter_dir']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
