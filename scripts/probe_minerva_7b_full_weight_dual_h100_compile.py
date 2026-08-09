#!/usr/bin/env python3
"""Probe torch.compile on the fastest dual-H100 accumulation recipe."""

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
    Minerva7BDualH100Accum8CompileProbeConfig,
    benchmark_minerva_7b_full_weight_ddp,
)


EAGER_REFERENCE_TOKENS_PER_SECOND = 14_934.107617301936


def main() -> None:
    config = Minerva7BDualH100Accum8CompileProbeConfig()
    rank = int(os.environ.get("RANK", "0"))
    started_at = time.monotonic()

    def progress(message: str) -> None:
        if rank == 0:
            print(
                f"minerva-h100-compile | {message} | "
                f"job_elapsed={_format_duration(time.monotonic() - started_at)}",
                flush=True,
            )

    if rank == 0:
        print(
            "minerva-h100-compile | start ranks=2 local_micro=8 accumulation=8 "
            "global_tokens=65536 bucket_mib=25 compile_mode=default "
            "warmup_updates=2 timed_updates=10 estimated_runtime=3m-10m_cached",
            flush=True,
        )
    report = benchmark_minerva_7b_full_weight_ddp(
        repo_root=ROOT,
        config=config,
        progress=progress,
    )
    if rank == 0:
        selected = report["selected_candidate"]
        speed = selected["tokens_per_second"]
        speedup = speed / EAGER_REFERENCE_TOKENS_PER_SECOND
        decision = "adopt" if speedup > 1.0 else "keep_eager"
        print(
            "minerva-h100-compile | complete status={status} "
            "throughput={speed:.1f}tokens/s eager_ratio={ratio:.3f} "
            "decision={decision} output={output}".format(
                status=selected["fit_decision"],
                speed=speed,
                ratio=speedup,
                decision=decision,
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
