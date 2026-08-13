#!/usr/bin/env python3
"""Audit the seven post-training research states without modifying them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_registry import audit_research_states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/minerva_7b_v7_full_weight_001"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/minerva_7b_v7_full_weight_protocol.json"))
    parser.add_argument("--parent-model-dir", type=Path)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(
        "minerva-v7-analysis | start job=state_audit states=7 "
        f"verify_hashes={args.verify_hashes}", flush=True,
    )
    report = audit_research_states(
        run_dir=args.run_dir,
        protocol_path=args.protocol,
        parent_model_dir=args.parent_model_dir,
        verify_hashes=args.verify_hashes,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for row in report["states"]:
        print(
            f"minerva-v7-analysis | state={row['state_id']} status={row['status']} "
            f"issues={len(row['issues'])}", flush=True,
        )
    print(
        f"minerva-v7-analysis | complete ready={report['status_counts']['complete']}/7 "
        f"all_ready={report['all_seven_states_complete']} output={args.output}", flush=True,
    )


if __name__ == "__main__":
    main()
