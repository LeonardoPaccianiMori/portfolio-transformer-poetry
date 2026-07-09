#!/usr/bin/env python3
"""Encode the local broader pretraining corpus into PyTorch tensors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.pretraining_dataset import PretrainingDatasetConfig
from sonnet_corpus.pretraining_dataset import build_pretraining_token_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-sources-dir",
        type=Path,
        default=ROOT / "data/local/pretraining/processed/sources",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=ROOT / "data/local/pretraining/tokenizers/bpe_8000.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/local/pretraining/encoded",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=ROOT / "data/local/pretraining/encoded/bpe_8000_report.json",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--document-separator", default="<|endoftext|>")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PretrainingDatasetConfig(
        processed_sources_dir=args.processed_sources_dir,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        report_path=args.report_path,
        validation_fraction=args.validation_fraction,
        document_separator=args.document_separator,
    )
    report = build_pretraining_token_dataset(config)
    print(f"wrote train tokens: {report['train_path']}")
    print(f"wrote validation tokens: {report['validation_path']}")
    print(f"wrote report: {args.report_path}")
    print(
        "tokens: "
        f"train={report['train_tokens']}, "
        f"validation={report['validation_tokens']}, "
        f"total={report['total_tokens']}"
    )


if __name__ == "__main__":
    main()
