#!/usr/bin/env python3
"""Analyze a verified pair of completed V7 probe extractions on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_representation import analyze_state_pair
from sonnet_analysis.minerva_v7_runtime import load_verified_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--left-state-id", required=True)
    parser.add_argument("--right-state-id", required=True)
    parser.add_argument("--extraction-root", type=Path, default=Path("artifacts/local/minerva_7b_v7_analysis/gpu_extraction"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left, right, comparison_id = load_verified_comparison(
        args.state_audit, args.left_state_id, args.right_state_id
    )
    print(f"minerva-v7-research | start job=representation comparison={comparison_id}", flush=True)
    report = analyze_state_pair(
        left_state_dir=args.extraction_root / args.left_state_id,
        right_state_dir=args.extraction_root / args.right_state_id,
        comparison_id=comparison_id,
        expected_state_identities={
            "left": str(left["state_identity_sha256"]),
            "right": str(right["state_identity_sha256"]),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"minerva-v7-research | complete probes={report['probe_count']} "
        f"streams={report['stream_count']} output={args.output}", flush=True,
    )


if __name__ == "__main__":
    main()
