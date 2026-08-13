#!/usr/bin/env python3
"""Score matched V7 generations and create a deterministic blinded review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_behavior import analyze_matched_generations, build_blinded_review
from sonnet_analysis.minerva_v7_memorization import load_verified_sonnet_train_reference
from sonnet_analysis.minerva_v7_registry import MODEL_STATES
from sonnet_analysis.minerva_v7_runtime import load_verified_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, default=Path("artifacts/local/minerva_7b_v7_analysis/matched_generation"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirmatory-seed", type=int, default=4099)
    parser.add_argument("--memorization-reference", type=Path, required=True, help="Hash-verified private sonnets_train reference manifest.")
    args = parser.parse_args()
    verified = {
        state.state_id: load_verified_state(args.state_audit, state.state_id)
        for state in MODEL_STATES
    }
    states = {
        state_id: args.generation_root / state_id for state_id in verified
    }
    print(f"minerva-v7-research | start job=behavior states={len(states)}", flush=True)
    memorization_records, reference_manifest = load_verified_sonnet_train_reference(
        args.memorization_reference
    )
    report = analyze_matched_generations(
        state_directories=states,
        confirmatory_seed=args.confirmatory_seed,
        memorization_records=memorization_records,
        progress=lambda message: print(f"minerva-v7-research | memorization {message}", flush=True),
        expected_state_identities={
            state_id: str(row["state_identity_sha256"])
            for state_id, row in verified.items()
        },
    )
    report["memorization_reference_manifest_sha256"] = reference_manifest["manifest_sha256"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "behavior.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    review = build_blinded_review(
        behavior_report=report,
        mapping_path=args.output_dir / "blind_mapping.private.json",
        review_path=args.output_dir / "blinded_review.md",
    )
    print(
        f"minerva-v7-research | complete outputs={len(report['rows'])} "
        f"report={report_path} review={review['review_path']}", flush=True,
    )


if __name__ == "__main__":
    main()
