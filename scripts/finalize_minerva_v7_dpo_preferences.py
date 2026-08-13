#!/usr/bin/env python3
"""Aggregate frozen DPO votes and, after calibration, export preferences."""

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
    aggregate_judge_votes,
    build_chosen_rejected_dataset,
    score_user_calibration,
    write_json_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--votes", type=Path, required=True)
    parser.add_argument("--user-votes", type=Path)
    parser.add_argument(
        "--judge-ids", nargs=3,
        default=("ai_judge_1", "ai_judge_2", "ai_judge_3"),
    )
    args = parser.parse_args()
    print(
        "minerva-v7-dpo | start job=finalize_preferences device=cpu "
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
    votes = _jsonl(args.votes)
    aggregation = aggregate_judge_votes(
        pairs=pairs, private_mapping=mapping, votes=votes,
        expected_judge_ids=tuple(args.judge_ids),
    )
    args.review_dir.mkdir(parents=True, exist_ok=True)
    aggregate_hash = write_json_atomic(
        args.review_dir / "judge_aggregation.frozen.json", aggregation
    )
    if args.user_votes is None:
        print(
            "minerva-v7-dpo | complete phase=ai_vote_aggregation "
            f"pairs={aggregation['pair_count']} votes={aggregation['vote_count']} "
            f"decisive_rate={aggregation['decisive_pair_rate']:.3f} "
            f"aggregation_sha256={aggregate_hash} "
            "awaiting_human_ai_calibration=True dpo_training_authorized=False",
            flush=True,
        )
        return

    calibration_packet = _load(
        args.review_dir / "user_calibration_packet.json"
    )
    calibration = score_user_calibration(
        aggregation=aggregation,
        calibration_packet=calibration_packet,
        user_votes=_jsonl(args.user_votes),
    )
    calibration_hash = write_json_atomic(
        args.review_dir / "human_ai_calibration.frozen.json", calibration
    )
    dataset = build_chosen_rejected_dataset(
        aggregation=aggregation, calibration=calibration
    )
    dataset_hash = write_json_atomic(
        args.review_dir / "chosen_rejected.frozen.json", dataset
    )
    print(
        "minerva-v7-dpo | complete phase=calibrated_preference_export "
        f"examples={dataset['example_count']} "
        f"human_ai_agreement={calibration['agreement_rate']:.3f} "
        f"calibration_sha256={calibration_hash} dataset_sha256={dataset_hash} "
        "dpo_training_authorized=False v7_test_accessed=False",
        flush=True,
    )


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _combine(left: dict, right: dict, *, key: str) -> dict:
    rows = list(left.get(key, [])) + list(right.get(key, []))
    identities = [str(row["pair_id"]) for row in rows]
    if not rows or len(identities) != len(set(identities)):
        raise ValueError("DPO ordinary/completion evidence is empty or overlaps")
    count_key = "pair_count"
    return {
        "preference_version": left["preference_version"],
        count_key: len(rows),
        key: sorted(rows, key=lambda row: str(row["pair_id"])),
        "v7_test_accessed": False,
    }


if __name__ == "__main__":
    main()
