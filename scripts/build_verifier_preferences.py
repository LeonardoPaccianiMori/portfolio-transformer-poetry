#!/usr/bin/env python3
"""Build verifier-labelled form preferences from the frozen DPO candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.verifier_preferences import (
    build_verifier_pairs,
    load_candidate_records,
    score_candidates,
    write_verifier_preferences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "artifacts/local/minerva_7b_v7_dpo/candidates/authoritative",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/local/verifier_labelled_dpo/preferences.frozen.json",
    )
    parser.add_argument("--min-gap", type=float, default=2.0)
    parser.add_argument("--min-chosen-score", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_candidate_records(args.candidates)
    scored = score_candidates(records)
    pairs = build_verifier_pairs(
        scored,
        min_gap=args.min_gap,
        min_chosen_score=args.min_chosen_score,
    )
    payload = write_verifier_preferences(
        args.output,
        scored=scored,
        pairs=pairs,
        candidates_dir=args.candidates,
        min_gap=args.min_gap,
        min_chosen_score=args.min_chosen_score,
    )
    chosen_scores = [pair["chosen_score"] for pair in pairs]
    rejected_scores = [pair["rejected_score"] for pair in pairs]
    print(f"candidates: {payload['candidate_count']}")
    print(f"degenerate candidates: {payload['degenerate_candidate_count']}")
    print(f"pairs: {payload['pair_count']}")
    print(
        "chosen score mean/min/max: "
        f"{sum(chosen_scores) / len(chosen_scores):.2f} "
        f"{min(chosen_scores):.2f} {max(chosen_scores):.2f}"
    )
    print(
        "rejected score mean/min/max: "
        f"{sum(rejected_scores) / len(rejected_scores):.2f} "
        f"{min(rejected_scores):.2f} {max(rejected_scores):.2f}"
    )
    print(f"wrote preferences: {args.output}")


if __name__ == "__main__":
    main()
