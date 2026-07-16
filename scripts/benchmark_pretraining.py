#!/usr/bin/env python3
"""Benchmark broader-pretraining model candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.pretraining_benchmark import PretrainingBenchmarkConfig
from sonnet_training.pretraining_benchmark import benchmark_pretraining_candidates
from sonnet_training.pretraining_benchmark import pretraining_candidates_for_set


CORPUS_DIR = Path("data/local/pretraining/expanded_italian_1200_1800_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-tokens-path",
        type=Path,
        default=CORPUS_DIR / "encoded" / "bpe_8000_train.pt",
    )
    parser.add_argument(
        "--validation-tokens-path",
        type=Path,
        default=CORPUS_DIR / "encoded" / "bpe_8000_validation.pt",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=CORPUS_DIR / "tokenizers" / "bpe_8000.json",
    )
    parser.add_argument(
        "--json-report-path",
        type=Path,
        default=Path("data/local/pretraining/benchmarks/pretraining_benchmark.json"),
    )
    parser.add_argument(
        "--markdown-report-path",
        type=Path,
        default=Path("reports/pretraining_hardware_benchmark.md"),
    )
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--benchmark-steps", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--candidate-set",
        choices=["baseline_relu", "quality_swiglu"],
        default="quality_swiglu",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PretrainingBenchmarkConfig(
        train_tokens_path=args.train_tokens_path,
        validation_tokens_path=args.validation_tokens_path,
        tokenizer_path=args.tokenizer_path,
        json_report_path=args.json_report_path,
        markdown_report_path=args.markdown_report_path,
        context_length=args.context_length,
        warmup_steps=args.warmup_steps,
        benchmark_steps=args.benchmark_steps,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        candidate_set_name=args.candidate_set,
    )
    report = benchmark_pretraining_candidates(
        repo_root=ROOT,
        config=config,
        candidates=pretraining_candidates_for_set(args.candidate_set),
        progress=lambda message: print(f"benchmark | {message}", flush=True),
    )
    print(f"wrote JSON report: {args.json_report_path}")
    print(f"wrote Markdown report: {args.markdown_report_path}")
    for result in report["results"]:
        if result["status"] == "ok":
            print(
                "{name}: {tokens_per_second:.1f} tokens/s, {memory} MiB peak".format(
                    name=result["name"],
                    tokens_per_second=result["tokens_per_second"],
                    memory=(
                        "n/a"
                        if result["peak_cuda_memory_mib"] is None
                        else f"{result['peak_cuda_memory_mib']:.1f}"
                    ),
                )
            )
        else:
            print(f"{result['name']}: error: {result['error']}")


if __name__ == "__main__":
    main()
