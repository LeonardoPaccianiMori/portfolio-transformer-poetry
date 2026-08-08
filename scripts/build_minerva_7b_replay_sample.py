#!/usr/bin/env python3
"""Build the deterministic local PAISÀ replay sample for Minerva 7B."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_staged_data import build_replay_text_sample


def main() -> None:
    started_at = time.monotonic()

    def progress(message: str) -> None:
        print(
            f"minerva-replay | {message} | elapsed={time.monotonic() - started_at:.1f}s",
            flush=True,
        )

    source_path = (
        ROOT
        / "data/local/pretraining/paisa/paisa_modern_italian_v1/train.txt"
    )
    output_root = ROOT / "data/local/minerva_7b_staged"
    print(
        "minerva-replay | start strategy=256_even_windows target=8MiB "
        "estimated_runtime=1m-3m",
        flush=True,
    )
    report = build_replay_text_sample(
        source_path=source_path,
        output_path=output_root / "replay_train.txt",
        report_path=output_root / "replay_sample_report.json",
        progress=progress,
    )
    print(
        "minerva-replay | complete bytes={size} sha256={sha}".format(
            size=report["output_size_bytes"],
            sha=report["output_sha256"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
