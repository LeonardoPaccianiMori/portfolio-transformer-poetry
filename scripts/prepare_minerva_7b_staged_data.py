#!/usr/bin/env python3
"""Tokenize local staged Minerva 7B training and preservation data."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_staged_data import prepare_minerva_7b_staged_data


def main() -> None:
    started_at = time.monotonic()

    def progress(message: str) -> None:
        print(
            f"minerva-data | {message} | elapsed={time.monotonic() - started_at:.1f}s",
            flush=True,
        )

    print(
        "minerva-data | start historical_sources=36 context=512 "
        "estimated_runtime=2m-10m",
        flush=True,
    )
    report = prepare_minerva_7b_staged_data(repo_root=ROOT, progress=progress)
    print(
        "minerva-data | complete historical_train={historical:,} "
        "historical_validation={validation:,} replay={replay:,} ".format(
            historical=report["historical_train_tokens"],
            validation=report["historical_validation_tokens"],
            replay=report["modern_replay_train_tokens"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
