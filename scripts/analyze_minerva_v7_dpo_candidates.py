#!/usr/bin/env python3
"""Verify and screen Minerva V7 DPO candidates without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_dpo_preferences import (
    analyze_preference_candidates,
    load_verified_candidates,
    write_json_atomic,
)
from sonnet_analysis.minerva_v7_memorization import (
    load_verified_sonnet_train_reference,
    score_texts_against_reference,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--memorization-reference", type=Path,
        default=Path(
            "artifacts/local/minerva_7b_v7_analysis/memorization_reference/manifest.json"
        ),
    )
    parser.add_argument("--hourly-rate", type=float, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    candidates, completion = load_verified_candidates(
        args.candidate_dir, require_complete=args.require_complete
    )
    print(
        "minerva-v7-dpo | start job=analyze_candidates device=cpu "
        f"total_steps={len(candidates)} progress_interval=1000_training_records",
        flush=True,
    )
    records, reference = load_verified_sonnet_train_reference(
        args.memorization_reference
    )
    memorization = score_texts_against_reference(
        [str(row["text"]) for row in candidates], records,
        progress=lambda message: print(f"minerva-v7-dpo | {message}", flush=True),
    )
    report, assessments, public_pairs, private_mapping = analyze_preference_candidates(
        candidates=candidates,
        memorization_scores=memorization,
        generation_elapsed_seconds=float(completion["generation_elapsed_seconds"]),
        hourly_rate_usd=args.hourly_rate,
    )
    report["memorization_reference_manifest_sha256"] = reference["manifest_sha256"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        args.output_dir / "candidate_assessments.json",
        {
            "preference_version": report["preference_version"],
            "candidate_count": len(assessments),
            "assessments": assessments,
            "v7_test_accessed": False,
        },
    )
    write_json_atomic(args.output_dir / "blinded_pairs.json", public_pairs)
    write_json_atomic(args.output_dir / "blind_mapping.private.json", private_mapping)
    completion_pairs = report.pop("completion_contrast_pairs")
    completion_mapping = report.pop("completion_contrast_mapping_private")
    write_json_atomic(
        args.output_dir / "completion_contrast_blinded_pairs.json",
        completion_pairs,
    )
    write_json_atomic(
        args.output_dir / "completion_contrast_mapping.private.json",
        completion_mapping,
    )
    write_json_atomic(args.output_dir / "analysis.json", report)
    print(
        "minerva-v7-dpo | complete "
        f"candidates={report['candidate_count']} eligible={report['eligible_candidate_count']} "
        f"yield={report['eligible_candidate_yield']:.3f} "
        f"projected_pairs={report['projected_full_pair_count']} "
        f"projected_minutes={report['projected_full_generation_minutes']:.1f} "
        f"projected_cost_usd={report['projected_full_generation_cost_usd']:.2f} "
        "v7_test_accessed=False training_performed=False",
        flush=True,
    )


if __name__ == "__main__":
    main()
