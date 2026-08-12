#!/usr/bin/env python3
"""Verify checkpoint 8E and freeze its local modern-preservation index."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_v7_protocol import (
    FullWeightProtocolConfig,
    prepare_full_weight_protocol,
)


def main() -> None:
    os.chdir(ROOT)
    config = FullWeightProtocolConfig(
        repo_root=ROOT,
        protocol_path=ROOT / "configs/minerva_7b_v7_full_weight_protocol.json",
        modern_encoded_dir=ROOT / "data/local/minerva_7b_full_weight/encoded",
        preservation_index_path=ROOT
        / "data/local/minerva_7b_v7/modern_preservation_validation_v1.jsonl",
        json_report_path=ROOT / "reports/minerva_7b_v7_full_weight_protocol_v1.json",
        markdown_report_path=ROOT / "reports/minerva_7b_v7_full_weight_protocol_v1.md",
    )
    started = time.monotonic()

    def progress(message: str) -> None:
        print(
            f"minerva-v7-protocol | {message} | "
            f"elapsed={_format_duration(time.monotonic() - started)}",
            flush=True,
        )

    print(
        "minerva-v7-protocol | start device=cpu context=2048 stages=3 "
        "modern_validation_windows=128 progress_interval=phase "
        "estimated_runtime=5s-20s gpu_authorized=false",
        flush=True,
    )
    report = prepare_full_weight_protocol(config, progress=progress)
    print(
        "minerva-v7-protocol | complete status={status} updates={updates:,} "
        "hardware_candidates={candidates} elapsed={elapsed} report={report}".format(
            status=report["status"],
            updates=report["training"]["total_optimizer_updates"],
            candidates=report["hardware_qualification"]["candidate_count"],
            elapsed=_format_duration(time.monotonic() - started),
            report=config.json_report_path.relative_to(ROOT),
        ),
        flush=True,
    )


def _format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


if __name__ == "__main__":
    main()
