#!/usr/bin/env python3
"""Benchmark full-weight Minerva 7B on two NVLink-connected H100 GPUs."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_full_weight_ddp_benchmark import (
    Minerva7BDualH100DdpBenchmarkConfig,
    benchmark_minerva_7b_full_weight_ddp,
)


def main() -> None:
    config = Minerva7BDualH100DdpBenchmarkConfig()
    rank = int(os.environ.get("RANK", "0"))
    started_at = time.monotonic()

    def progress(message: str) -> None:
        if rank == 0:
            print(
                f"minerva-h100-ddp-benchmark | {message} | "
                f"elapsed={_format_duration(time.monotonic() - started_at)}",
                flush=True,
            )

    if rank == 0:
        print(
            "minerva-h100-ddp-benchmark | start ranks=2 candidates=6 "
            "global_tokens=4096,8192 buckets_mib=25,100,250 "
            "warmup_updates=1 timed_updates=5 "
            "estimated_runtime=5m-15m_cached",
            flush=True,
        )
    report = benchmark_minerva_7b_full_weight_ddp(
        repo_root=ROOT,
        config=config,
        progress=progress,
    )
    if rank == 0:
        selected = report["selected_candidate"]
        print(
            "minerva-h100-ddp-benchmark | complete selected={candidate} "
            "throughput={speed:.1f}tokens/s projected_hours={hours:.1f} "
            "projected_cost=${cost:.2f} checkpoint_retained=false "
            "output={output}".format(
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
