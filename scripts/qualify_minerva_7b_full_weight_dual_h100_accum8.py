#!/usr/bin/env python3
"""Endurance-qualify the fastest eager dual-H100 full-weight recipe."""

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
    Minerva7BDualH100Accum8EnduranceConfig,
    benchmark_minerva_7b_full_weight_ddp,
)


def main() -> None:
    config = Minerva7BDualH100Accum8EnduranceConfig()
    rank = int(os.environ.get("RANK", "0"))
    started_at = time.monotonic()

    def progress(message: str) -> None:
        if rank == 0:
            print(
                f"minerva-h100-endurance | {message} | "
                f"job_elapsed={_format_duration(time.monotonic() - started_at)}",
                flush=True,
            )

    if rank == 0:
        print(
            "minerva-h100-endurance | start ranks=2 local_micro=8 accumulation=8 "
            "global_tokens=65536 bucket_mib=25 warmup_updates=5 "
            "timed_updates=100 progress_interval=10 validation_transition=true "
            "estimated_runtime=8m-15m_cached",
            flush=True,
        )
    report = benchmark_minerva_7b_full_weight_ddp(
        repo_root=ROOT,
        config=config,
        progress=progress,
    )
    if rank == 0:
        selected = report["selected_candidate"]
        validation = report["validation_transition"]
        print(
            "minerva-h100-endurance | complete status={status} "
            "throughput={speed:.1f}tokens/s free_after_cache_release={free:.1f}MiB "
            "validation={initial:.4f}->{final:.4f} output={output}".format(
                status=selected["fit_decision"],
                speed=selected["tokens_per_second"],
                free=selected["minimum_free_memory_after_mib"],
                initial=validation["initial"]["mean_loss"],
                final=validation["final"]["mean_loss"],
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
