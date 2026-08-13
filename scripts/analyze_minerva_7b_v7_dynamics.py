#!/usr/bin/env python3
"""Validate and export Minerva V7 loss, LR, gate, memory, and cost evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_dynamics import build_dynamics_report, write_dynamics_exports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/minerva_7b_v7_full_weight_001"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/minerva_7b_v7_full_weight_protocol.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/local/minerva_7b_v7_analysis/dynamics"))
    args = parser.parse_args()
    print("minerva-v7-analysis | start job=training_dynamics expected_updates=2960", flush=True)
    report = build_dynamics_report(run_dir=args.run_dir, protocol_path=args.protocol)
    paths = write_dynamics_exports(report, args.output_dir)
    for row in report["stages"]:
        print(
            f"minerva-v7-analysis | stage={row['stage_id']} "
            f"updates={row['observed_updates']}/{row['expected_updates']} "
            f"complete={row['complete']}", flush=True,
        )
    print(
        f"minerva-v7-analysis | complete status={report['status']} "
        f"issues={len(report['issues'])} report={paths['report']}", flush=True,
    )


if __name__ == "__main__":
    main()
