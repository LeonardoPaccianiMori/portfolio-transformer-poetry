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
        default=ROOT / "data/processed/pretraining_historical_italian_v2/corpus.txt",
    )
    parser.add_argument(
        "--mixture-report-path",
        type=Path,
        default=ROOT / "reports/pretraining_historical_italian_v2_mixture_report.json",
    )
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=ROOT
        / "data/metadata/pretraining_tokenizers/pretraining_historical_italian_v2_bpe_16000.json",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=ROOT / "reports/pretraining_historical_italian_v2_bpe_16000_report.json",
    )
    parser.add_argument(
        "--build-report-path",
        type=Path,
        default=ROOT / "reports/pretraining_historical_italian_v2_mixture_report.json",
    )
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    parser.add_argument("--training-character-limit", type=int, default=4_000_000)
    parser.add_argument("--minimum-source-characters", type=int, default=20_000)
    parser.add_argument("--merge-progress-interval", type=int, default=500)
    parser.add_argument(
        "--training-checkpoint-path",
        type=Path,
        default=ROOT
        / "data/local/pretraining/tokenizers/pretraining_historical_italian_v2_bpe_16000_training_state.json",
    )
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
    config = PretrainingTokenizerConfig(
        corpus_path=args.corpus_path,
        tokenizer_path=args.tokenizer_path,
        report_path=args.report_path,
        build_report_path=args.build_report_path,
        vocab_size=args.vocab_size,
        special_tokens=tuple(args.special_token),
        training_character_limit=args.training_character_limit,
        manifest_path=args.manifest_path,
        source_dir=args.source_dir,
        mixture_report_path=args.mixture_report_path,
        minimum_source_characters=args.minimum_source_characters,
        merge_progress_interval=args.merge_progress_interval,
        training_checkpoint_path=args.training_checkpoint_path,
        max_merges_per_run=args.max_merges_per_run,
    )
    while True:
        report = train_pretraining_bpe_tokenizer(
            config,
            progress=lambda message: print(f"tokenizer | {message}", flush=True),
        )
        print(f"tokenizer | wrote report: {args.report_path}", flush=True)
        if report["status"] == "complete":
            break
        print(
            "tokenizer | checkpointed "
            f"vocabulary={report['actual_vocab_size']}/{report['target_vocab_size']} "
            f"merges={report['merge_count']}",
            flush=True,
        )
        if not args.resume_until_complete:
            return
        print("tokenizer | resuming from checkpoint", flush=True)

    print(f"tokenizer | wrote tokenizer: {args.tokenizer_path}", flush=True)
    print(
        "tokenizer | corpus tokens: "
        f"{report['token_count']} "
        f"({report['characters_per_token']:.2f} characters/token)",
        flush=True,
    )
    print(
        f"tokenizer | boundary warnings: {len(report['boundary_warnings'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
