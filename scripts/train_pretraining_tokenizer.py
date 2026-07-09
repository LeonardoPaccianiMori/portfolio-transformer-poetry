#!/usr/bin/env python3
"""Train the local broader-corpus BPE tokenizer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.pretraining_tokenizer import PretrainingTokenizerConfig
from sonnet_corpus.pretraining_tokenizer import train_pretraining_bpe_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=ROOT / "data/local/pretraining/processed/corpus.txt",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=ROOT / "data/local/pretraining/tokenizers/bpe_8000.json",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=ROOT / "data/local/pretraining/tokenizers/bpe_8000_report.json",
    )
    parser.add_argument(
        "--build-report-path",
        type=Path,
        default=ROOT / "data/local/pretraining/build_report.json",
    )
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    parser.add_argument("--training-character-limit", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PretrainingTokenizerConfig(
        corpus_path=args.corpus_path,
        tokenizer_path=args.tokenizer_path,
        report_path=args.report_path,
        build_report_path=args.build_report_path,
        vocab_size=args.vocab_size,
        special_tokens=tuple(args.special_token),
        training_character_limit=args.training_character_limit,
    )
    report = train_pretraining_bpe_tokenizer(config)
    print(f"wrote tokenizer: {args.tokenizer_path}")
    print(f"wrote report: {args.report_path}")
    print(
        "corpus tokens: "
        f"{report['token_count']} "
        f"({report['characters_per_token']:.2f} characters/token)"
    )
    print(f"boundary warnings: {len(report['boundary_warnings'])}")


if __name__ == "__main__":
    main()
