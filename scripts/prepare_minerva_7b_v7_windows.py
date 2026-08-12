#!/usr/bin/env python3
"""Build independently reproduced deterministic Minerva V7 window indexes."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_v7_windows import (
    MinervaV7WindowConfig,
    prepare_and_verify_minerva_v7_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-index-files-per-run",
        type=int,
        default=None,
        help="Testing/resume control; omit to complete every index file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    config = MinervaV7WindowConfig(
        repo_root=ROOT,
        policy_path=ROOT / "data/metadata/minerva_7b_v7_sampling_policy_v1.json",
        encoded_report_path=ROOT / "reports/minerva_7b_v7_encoded_data_v1.json",
        primary_encoded_dir=ROOT / "data/local/minerva_7b_v7/encoded",
        reproduction_encoded_dir=ROOT
        / "data/local/minerva_7b_v7/encoded_reproduction",
        primary_output_dir=ROOT / "data/local/minerva_7b_v7/window_indexes",
        reproduction_output_dir=ROOT
        / "data/local/minerva_7b_v7/window_indexes_reproduction",
        json_report_path=ROOT / "reports/minerva_7b_v7_stage_windows_v1.json",
        markdown_report_path=ROOT / "reports/minerva_7b_v7_stage_windows_v1.md",
        max_index_files_per_run=args.max_index_files_per_run,
    )
    started = time.monotonic()

    def progress(message: str) -> None:
        print(
            f"minerva-v7-windows | {message} | "
            f"elapsed={_format_duration(time.monotonic() - started)}",
            flush=True,
        )

    print(
        "minerva-v7-windows | start device=cpu independent_indexes=2 "
        "training_windows=47,360 context=2048 source_span=2049 "
        "progress_interval=index_file estimated_runtime=1m-4m resumable=true",
        flush=True,
    )
    report = prepare_and_verify_minerva_v7_windows(config, progress=progress)
    if report["status"] != "active_verified":
        print(
            "minerva-v7-windows | paused after an atomic index file; rerun the "
            "same command to resume",
            flush=True,
        )
        return
    print(
        "minerva-v7-windows | complete status={status} training_windows={windows:,} "
        "validation_windows={validation:,} test_windows={test:,} elapsed={elapsed} "
        "report={report}".format(
            status=report["status"],
            windows=report["training"]["windows"],
            validation=report["evaluation"]["validation"]["windows"],
            test=report["evaluation"]["test"]["windows"],
            elapsed=_format_duration(time.monotonic() - started),
            report=config.json_report_path.relative_to(ROOT),
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
