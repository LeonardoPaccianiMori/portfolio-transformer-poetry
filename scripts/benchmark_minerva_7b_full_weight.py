#!/usr/bin/env python3
"""Benchmark fixed-effective-batch Minerva 7B full-weight H100 recipes."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_full_weight_benchmark import (
    Minerva7BFullWeightBenchmarkConfig,
    benchmark_minerva_7b_full_weight,
)


def main() -> None:
    config = Minerva7BFullWeightBenchmarkConfig()
    started_at = time.monotonic()

    def progress(message: str) -> None:
        print(
            f"minerva-full-benchmark | {message} | "
            f"elapsed={_format_duration(time.monotonic() - started_at)}",
            flush=True,
        )

    print(
        "minerva-full-benchmark | start candidates=8 microbatches=1,2,4,8 "
        "checkpointing=on,off effective_tokens_per_update=4096 "
        "warmup_updates=1 timed_updates=5 estimated_runtime=5m-15m_cached",
        flush=True,
    )
    report = benchmark_minerva_7b_full_weight(
        repo_root=ROOT,
        config=config,
        progress=progress,
    )
    selected = report["selected_candidate"]
    print(
        "minerva-full-benchmark | complete selected={candidate} "
        "throughput={speed:.1f}tokens/s projected_hours={hours:.1f} "
        "projected_cost=${cost:.2f} checkpoint_retained=false output={output}".format(
            candidate=selected["candidate_id"],
            speed=selected["tokens_per_second"],
            hours=selected["projection"]["projected_hours_with_overhead"],
            cost=selected["projection"]["projected_cost_usd"],
            output=ROOT / config.output_path,
        ),
        flush=True,
    )


def _format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


if __name__ == "__main__":
    main()
