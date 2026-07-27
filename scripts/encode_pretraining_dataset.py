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
        default=ROOT / "data/processed/expanded_italian_1200_1800_v1/sources",
    )
    parser.add_argument(
        "--mixture-report-path",
        type=Path,
        default=ROOT / "reports/pretraining_historical_italian_v2_mixture_report.json",
    )
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=ROOT
        / "data/metadata/pretraining_tokenizers/pretraining_historical_italian_v2_bpe_16000.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/local/pretraining/pretraining_historical_italian_v2/encoded",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=ROOT / "reports/pretraining_historical_italian_v2_encoded_report.json",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--document-separator", default="<|endoftext|>")
    parser.add_argument("--train-filename", default="bpe_16000_train.pt")
    parser.add_argument("--validation-filename", default="bpe_16000_validation.pt")
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
        manifest_path=args.manifest_path,
        mixture_report_path=args.mixture_report_path,
        train_filename=args.train_filename,
        validation_filename=args.validation_filename,
    )
    report = build_pretraining_token_dataset(
        config,
        progress=lambda message: print(f"encode | {message}", flush=True),
    )
    print(f"encode | wrote train tokens: {report['train_path']}", flush=True)
    print(f"encode | wrote validation tokens: {report['validation_path']}", flush=True)
    print(f"encode | wrote report: {args.report_path}", flush=True)
    print(
        "encode | tokens: "
        f"train={report['train_tokens']}, "
        f"validation={report['validation_tokens']}, "
        f"total={report['total_tokens']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
