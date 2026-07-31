#!/usr/bin/env python3
"""GPU-calibrate the one approved PAISA-historical rescue architecture."""

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
from sonnet_training.pretraining_benchmark import paisa_historical_rescue_candidates
from sonnet_training.pretraining_benchmark import resolve_required_cuda_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--benchmark-steps", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = str(resolve_required_cuda_device(args.device))
    config = PretrainingBenchmarkConfig(
        dataset_version="paisa_historical_rescue_v1",
        train_tokens_path=Path(
            "data/local/pretraining/paisa_historical_rescue_v1/encoded/"
            "paisa_train.uint16.bin"
        ),
        validation_tokens_path=Path(
            "data/local/pretraining/paisa_historical_rescue_v1/encoded/"
            "paisa_validation.uint16.bin"
        ),
        tokenizer_path=Path(
            "data/local/pretraining/paisa_historical_rescue_v1/tokenizer.json"
        ),
        dataset_report_path=Path(
            "reports/paisa_historical_rescue_v1_encoded_report.json"
        ),
        train_split_id="paisa_train",
        validation_split_id="paisa_validation",
        json_report_path=Path(
            "data/local/pretraining/benchmarks/"
            "paisa_historical_rescue_v1_hardware_benchmark.json"
        ),
        markdown_report_path=Path(
            "reports/paisa_historical_rescue_v1_hardware_benchmark.md"
        ),
        context_length=512,
        warmup_steps=args.warmup_steps,
        benchmark_steps=args.benchmark_steps,
        eval_batches=args.eval_batches,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
        candidate_set_name="paisa_historical_rescue",
        normalization_type="layer_norm",
        position_encoding_type="learned_absolute",
        tie_token_embeddings=False,
    )
    report = benchmark_pretraining_candidates(
        repo_root=ROOT,
        config=config,
        candidates=paisa_historical_rescue_candidates(),
        progress=lambda message: print(f"rescue-benchmark | {message}", flush=True),
    )
    print(
        "rescue-benchmark | wrote public report: "
        "reports/paisa_historical_rescue_v1_hardware_benchmark.md",
        flush=True,
    )
    for result in report["results"]:
        if result["status"] == "ok":
            peak_memory_mib = result["peak_cuda_memory_mib"]
            memory_text = (
                "peak VRAM unavailable"
                if peak_memory_mib is None
                else f"{peak_memory_mib:.1f} MiB peak"
            )
            print(
                "rescue-benchmark | {name}: {tokens_per_second:.1f} tokens/s, "
                "{memory}".format(
                    name=result["name"],
                    tokens_per_second=result["tokens_per_second"],
                    memory=memory_text,
                ),
                flush=True,
            )
        else:
            print(
                f"rescue-benchmark | {result['name']}: error: {result['error']}",
                flush=True,
            )
if __name__ == "__main__":
    main()
