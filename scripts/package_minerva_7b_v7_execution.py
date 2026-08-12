#!/usr/bin/env python3
"""Build and verify the private Minerva V7 remote execution bundle."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_training.minerva_7b_v7_bundle import (
    package_v7_execution_bundle,
    verify_v7_execution_bundle,
)


def main() -> None:
    output = ROOT / "artifacts/minerva_7b_v7_execution_bundle_v1.tar.gz"
    started = time.monotonic()
    print(
        "minerva-v7-bundle | start job=private-transfer-bundle device=cpu "
        "total_steps=2 progress_interval=1 estimated_runtime=5m-15m",
        flush=True,
    )
    report = package_v7_execution_bundle(repo_root=ROOT, output_path=output)
    print(
        f"minerva-v7-bundle | step=1/2 progress=50.0% bytes={report['output_bytes']:,} "
        f"sha256={report['output_sha256']}",
        flush=True,
    )
    verified = verify_v7_execution_bundle(output)
    print(
        "minerva-v7-bundle | step=2/2 progress=100.0% files={files} "
        "test_material=false elapsed={elapsed:.1f}s output={output}".format(
            files=len(verified["files"]),
            elapsed=time.monotonic() - started,
            output=output,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
