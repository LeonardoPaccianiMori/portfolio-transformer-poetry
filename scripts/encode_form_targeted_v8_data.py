#!/usr/bin/env python3
"""Encode the V8 corrected sonnet train split for the form-targeted LoRA pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_runtime import tokenizer_sha256
from sonnet_training.form_targeted_data import (
    encode_train_split,
    load_train_rows,
    read_sonnet_text,
    write_token_shard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/form_targeted_lora_pilot.json",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        required=True,
        help="local directory with the frozen Minerva tokenizer",
    )
    parser.add_argument(
        "--check-hashes",
        action="store_true",
        help="verify every selected record hash without tokenizing",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="manifest split to encode; defaults to the config value",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_config = config["data"]
    split = args.split or data_config["split"]
    manifest_path = ROOT / data_config["manifest"]
    rows = load_train_rows(manifest_path, split=split)
    print(f"form-targeted-encode | selected {len(rows)} train sonnets", flush=True)

    if args.check_hashes:
        for index, row in enumerate(rows, start=1):
            read_sonnet_text(ROOT, row)
            if index % 4000 == 0:
                print(f"form-targeted-encode | verified {index}", flush=True)
        print("form-targeted-encode | hash check complete", flush=True)
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_dir), local_files_only=True
    )
    fingerprint = tokenizer_sha256(tokenizer)
    if fingerprint != config.get("tokenizer_sha256", fingerprint):
        raise ValueError("tokenizer fingerprint differs from the frozen value")
    tokens, records = encode_train_split(
        rows, ROOT, tokenizer=tokenizer, progress=print
    )
    output_dir = args.output_dir or ROOT / data_config["output_dir"]
    report = write_token_shard(
        output_dir,
        tokens,
        records,
        manifest_path=manifest_path,
        tokenizer_sha256=fingerprint,
        max_sequence_tokens=int(data_config["max_sequence_tokens"]),
        split=split,
    )
    print(
        "form-targeted-encode | complete "
        f"sonnets={report['sonnet_count']} tokens={report['token_count']} "
        f"shard={report['shard_path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
