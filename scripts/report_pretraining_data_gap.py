#!/usr/bin/env python3
"""Write a public pretraining corpus-scale and exposure-budget report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.pretraining_data_gap import (
    PretrainingDataGapConfig,
    build_pretraining_data_gap_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    local_root = ROOT / "data/local/pretraining/expanded_italian_1200_1800_v1"
    parser.add_argument(
        "--run-config-path",
        type=Path,
        default=ROOT / "runs/pretraining_quality_swiglu_larger_200k_001/config.json",
    )
    parser.add_argument(
        "--encoding-report-path",
        type=Path,
        default=local_root / "encoded/bpe_8000_report.json",
    )
    parser.add_argument(
        "--tokenizer-report-path",
        type=Path,
        default=local_root / "tokenizers/bpe_8000_report.json",
    )
    parser.add_argument(
        "--balance-report-path",
        type=Path,
        default=local_root / "balance_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/pretraining_data_gap.md",
    )
    parser.add_argument("--target-unique-tokens", type=int, default=75_000_000)
    parser.add_argument("--max-train-steps", type=int, default=650_000)
    parser.add_argument("--heuristic-tokens-per-parameter", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "data-gap | start "
        f"target_unique_tokens={args.target_unique_tokens:,} "
        f"max_train_steps={args.max_train_steps:,}",
        flush=True,
    )
    report = build_pretraining_data_gap_report(
        PretrainingDataGapConfig(
            run_config_path=args.run_config_path,
            encoding_report_path=args.encoding_report_path,
            tokenizer_report_path=args.tokenizer_report_path,
            balance_report_path=args.balance_report_path,
            markdown_report_path=args.output,
            target_unique_tokens=args.target_unique_tokens,
            max_train_steps=args.max_train_steps,
            heuristic_tokens_per_parameter=args.heuristic_tokens_per_parameter,
        )
    )
    print(f"data-gap | wrote report: {args.output}", flush=True)
    print(
        "data-gap | complete "
        f"current_tokens={report['current_total_tokens']:,} "
        f"additional_tokens_needed={report['additional_unique_tokens_needed']:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
