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

from sonnet_training.pretraining_benchmark import PRETRAINING_DATASET_REPORT_PATH
from sonnet_training.pretraining_benchmark import PRETRAINING_TOKENIZER_PATH
from sonnet_training.pretraining_benchmark import PRETRAINING_TRAIN_TOKENS_PATH
from sonnet_training.pretraining_benchmark import PRETRAINING_VALIDATION_TOKENS_PATH
from sonnet_training.pretraining_benchmark import PretrainingBenchmarkConfig
from sonnet_training.pretraining_benchmark import benchmark_pretraining_candidates
from sonnet_training.pretraining_benchmark import pretraining_candidates_for_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-tokens-path",
        type=Path,
        default=Path(PRETRAINING_TRAIN_TOKENS_PATH),
    )
    parser.add_argument(
        "--validation-tokens-path",
        type=Path,
        default=Path(PRETRAINING_VALIDATION_TOKENS_PATH),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path(PRETRAINING_TOKENIZER_PATH),
    )
    parser.add_argument(
        "--dataset-report-path",
        type=Path,
        default=Path(PRETRAINING_DATASET_REPORT_PATH),
    )
    parser.add_argument(
        "--json-report-path",
        type=Path,
        default=Path(
            "data/local/pretraining/benchmarks/"
            "pretraining_historical_italian_v2_benchmark.json"
        ),
    )
    parser.add_argument(
        "--markdown-report-path",
        type=Path,
        default=Path("reports/pretraining_historical_italian_v2_hardware_benchmark.md"),
    )
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--benchmark-steps", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--candidate-set",
        choices=["baseline_relu", "quality_swiglu", "historical_v2_quality_swiglu"],
        default="historical_v2_quality_swiglu",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--normalization-type",
        choices=["layer_norm", "rms_norm"],
        default="layer_norm",
    )
    parser.add_argument("--normalization-eps", type=float, default=1e-5)
    parser.add_argument(
        "--position-encoding-type",
        choices=["learned_absolute", "rope"],
        default="learned_absolute",
    )
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--tie-token-embeddings", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PretrainingBenchmarkConfig(
        train_tokens_path=args.train_tokens_path,
        validation_tokens_path=args.validation_tokens_path,
        tokenizer_path=args.tokenizer_path,
        dataset_report_path=args.dataset_report_path,
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
        normalization_type=args.normalization_type,
        normalization_eps=args.normalization_eps,
        position_encoding_type=args.position_encoding_type,
        rope_theta=args.rope_theta,
        tie_token_embeddings=args.tie_token_embeddings,
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
