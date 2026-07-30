#!/usr/bin/env python3
"""Fit the fresh BPE tokenizer for the final PAISÀ-historical rescue."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.paisa_historical_tokenizer import PaisaHistoricalTokenizerConfig
from sonnet_corpus.paisa_historical_tokenizer import train_paisa_historical_rescue_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curriculum-config",
        type=Path,
        default=ROOT / "configs/paisa_historical_rescue_v1.json",
    )
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--local-report-path", type=Path)
    parser.add_argument("--public-report-path", type=Path)
    parser.add_argument("--training-checkpoint-path", type=Path)
    parser.add_argument("--merge-progress-interval", type=int, default=500)
    parser.add_argument("--max-merges-per-run", type=int)
    parser.add_argument(
        "--resume-until-complete",
        action="store_true",
        help="Repeat checkpointed merge chunks until the requested vocabulary is complete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume_until_complete and args.max_merges_per_run is None:
        raise ValueError("--resume-until-complete requires --max-merges-per-run")
    config = PaisaHistoricalTokenizerConfig(
        curriculum_config_path=args.curriculum_config,
        tokenizer_path=args.tokenizer_path,
        local_report_path=args.local_report_path,
        public_report_path=args.public_report_path,
        training_checkpoint_path=args.training_checkpoint_path,
        merge_progress_interval=args.merge_progress_interval,
        max_merges_per_run=args.max_merges_per_run,
    )
    print(
        "rescue-tokenizer | start "
        f"curriculum={args.curriculum_config} "
        f"merge_progress_interval={args.merge_progress_interval}",
        flush=True,
    )
    while True:
        report = train_paisa_historical_rescue_tokenizer(
            config,
            progress=lambda message: print(f"rescue-tokenizer | {message}", flush=True),
        )
        print(
            "rescue-tokenizer | status="
            f"{report['status']} vocabulary="
            f"{report['tokenizer']['actual_vocab_size']}/"
            f"{report['tokenizer']['target_vocab_size']}",
            flush=True,
        )
        if report["status"] == "complete":
            break
        if not args.resume_until_complete:
            return
        print("rescue-tokenizer | resuming from checkpoint", flush=True)
    print(
        "rescue-tokenizer | wrote tokenizer: "
        f"{report['local_artifacts']['tokenizer_path']}",
        flush=True,
    )
    print(
        "rescue-tokenizer | wrote public report: "
        f"{args.public_report_path or 'derived from curriculum report'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
