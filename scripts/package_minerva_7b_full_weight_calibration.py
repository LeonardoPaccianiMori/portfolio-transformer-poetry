#!/usr/bin/env python3
"""Package the verified Minerva full-weight H100 calibration payload."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_full_weight_bundle import (
    package_minerva_7b_full_weight_calibration,
)


def main() -> None:
    output_path = ROOT / "artifacts/minerva_7b_full_weight_h100_calibration.tar.gz"
    started_at = time.monotonic()
    print(
        "minerva-full-bundle | start scope=calibration-windows-only "
        "estimated_runtime=under-1m",
        flush=True,
    )
    report = package_minerva_7b_full_weight_calibration(
        repo_root=ROOT,
        output_path=output_path,
    )
    print(
        "minerva-full-bundle | complete bytes={bytes:,} sha256={sha256} "
        "elapsed={elapsed:.1f}s output={output}".format(
            bytes=report["output_bytes"],
            sha256=report["output_sha256"],
            elapsed=time.monotonic() - started_at,
            output=output_path,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
