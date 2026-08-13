#!/usr/bin/env python3
"""Plan or run one memory-bounded Minerva V7 SafeTensors comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_weights import compare_model_weights, plan_weight_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-model-dir", type=Path, required=True)
    parser.add_argument("--right-model-dir", type=Path, required=True)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--execute", action="store_true", help="Scan tensors; default is a metadata-only dry run.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    chunk_bytes = args.chunk_mib * 1024 * 1024
    print(
        f"minerva-v7-analysis | start job=weight_change execute={args.execute} "
        f"chunk_mib={args.chunk_mib}", flush=True,
    )
    plan = plan_weight_comparison(
        left_model_dir=args.left_model_dir, right_model_dir=args.right_model_dir,
        chunk_bytes=chunk_bytes,
    )
    print(
        f"minerva-v7-analysis | tensors={plan['tensor_count']} "
        f"scan_gib={plan['input_bytes_to_scan'] / 1024**3:.2f} "
        f"working_mib={plan['maximum_projected_working_bytes'] / 1024**2:.1f}", flush=True,
    )
    if args.execute:
        def progress(index: int, total: int, name: str, elapsed: float) -> None:
            print(
                f"minerva-v7-analysis | tensor={index}/{total} "
                f"progress={100 * index / total:.1f}% name={name} elapsed={elapsed:.1f}s",
                flush=True,
            )
        report = compare_model_weights(
            left_model_dir=args.left_model_dir, right_model_dir=args.right_model_dir,
            chunk_bytes=chunk_bytes, progress=progress,
        )
    else:
        report = plan
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"minerva-v7-analysis | complete output={args.output}", flush=True)


if __name__ == "__main__":
    main()
