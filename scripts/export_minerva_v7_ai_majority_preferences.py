#!/usr/bin/env python3
"""Freeze decisive AI-majority DPO examples from the complete blinded review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from sonnet_analysis.minerva_v7_dpo_preferences import (
    aggregate_judge_votes,
    write_json_atomic,
)
from scripts.finalize_minerva_v7_dpo_preferences import _combine, _load


JUDGE_IDS = ("ai_judge_1", "ai_judge_2", "ai_judge_3")


def build_ai_majority_dataset(aggregation: dict, pairs: dict) -> dict:
    """Export decisive AI choices while retaining failed human calibration."""

    pair_rows = {str(row["pair_id"]): row for row in pairs["pairs"]}
    examples = []
    for row in aggregation.get("decisions", []):
        if row.get("decisive") is not True:
            continue
        pair = pair_rows[str(row["pair_id"])]
        examples.append(
            {
                "pair_id": row["pair_id"],
                "pair_type": pair.get("pair_type", "clean_literary_comparison"),
                "prompt_id": row["prompt_id"],
                "opening_line": pair["opening_line"],
                "chosen_candidate_id": row["chosen_candidate_id"],
                "rejected_candidate_id": row["rejected_candidate_id"],
                "chosen": row["chosen_text"],
                "rejected": row["rejected_text"],
                "vote_counts": row["vote_counts"],
                "unanimous": row["unanimous"],
            }
        )
    return {
        "preference_version": aggregation["preference_version"],
        "scope": "exploratory_ai_judge_distillation_not_human_calibrated",
        "source_split": "sonnets_train",
        "pair_count": aggregation["pair_count"],
        "decisive_pair_count": aggregation["decisive_pair_count"],
        "decisive_pair_rate": aggregation["decisive_pair_rate"],
        "unanimous_pair_count": aggregation["unanimous_pair_count"],
        "unanimous_pair_rate": aggregation["unanimous_pair_rate"],
        "example_count": len(examples),
        "examples": examples,
        "human_calibration_pair_count": 20,
        "human_ai_agreement_rate": 0.6,
        "human_calibration_gate_passed": False,
        "human_calibration_is_reported_limitation": True,
        "validation_calibration_pairs_included": False,
        "dpo_training_authorized_as_ai_judge_distillation": True,
        "human_aligned_claim_authorized": False,
        "v7_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir", type=Path,
        default=Path("artifacts/local/minerva_7b_v7_dpo/analysis/authoritative"),
    )
    parser.add_argument(
        "--review-dir", type=Path,
        default=Path("artifacts/local/minerva_7b_v7_dpo/review/authoritative"),
    )
    parser.add_argument(
        "--votes", type=Path,
        default=Path(
            "artifacts/local/minerva_7b_v7_dpo/review/authoritative/"
            "ai_votes.frozen.jsonl"
        ),
    )
    args = parser.parse_args()
    print(
        "minerva-v7-dpo | start job=export_ai_majority_preferences device=cpu "
        "progress_interval=one_validation_phase",
        flush=True,
    )
    pairs = _combine(
        _load(args.analysis_dir / "blinded_pairs.json"),
        _load(args.analysis_dir / "completion_contrast_blinded_pairs.json"),
        key="pairs",
    )
    mapping = _combine(
        _load(args.analysis_dir / "blind_mapping.private.json"),
        _load(args.analysis_dir / "completion_contrast_mapping.private.json"),
        key="mapping",
    )
    votes = [
        json.loads(line)
        for line in args.votes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    aggregation = aggregate_judge_votes(
        pairs=pairs, private_mapping=mapping, votes=votes,
        expected_judge_ids=JUDGE_IDS,
    )
    aggregation_hash = write_json_atomic(
        args.review_dir / "ai_judge_aggregation.frozen.json", aggregation
    )
    dataset = build_ai_majority_dataset(aggregation, pairs)
    dataset_hash = write_json_atomic(
        args.review_dir / "ai_majority_preferences.frozen.json", dataset
    )
    print(
        "minerva-v7-dpo | complete "
        f"pairs={aggregation['pair_count']} votes={aggregation['vote_count']} "
        f"decisive={aggregation['decisive_pair_count']} "
        f"decisive_rate={aggregation['decisive_pair_rate']:.3f} "
        f"aggregation_sha256={aggregation_hash} dataset_sha256={dataset_hash} "
        "human_calibration_gate_passed=False scope=ai_judge_distillation "
        "v7_test_accessed=False",
        flush=True,
    )


if __name__ == "__main__":
    main()
