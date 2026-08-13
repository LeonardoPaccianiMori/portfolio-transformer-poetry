#!/usr/bin/env python3
"""Freeze three blinded AI votes for the validation-only calibration packet."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_dpo_preferences import write_json_atomic


JUDGE_IDS = ("ai_judge_1", "ai_judge_2", "ai_judge_3")
SCORE_KEYS = (
    "grammar",
    "coherence",
    "historical_register",
    "poetic_force",
    "form",
    "volta_closure",
)


def aggregate_validation_calibration_votes(
    *, packet: Mapping, votes: Sequence[Mapping]
) -> dict:
    """Aggregate validation-only votes without constructing DPO examples."""

    pairs = {str(row["pair_id"]): row for row in packet.get("pairs", [])}
    if len(pairs) != 20 or int(packet.get("pair_count", -1)) != len(pairs):
        raise ValueError("validation calibration packet must contain 20 unique pairs")
    grouped: dict[str, list[dict]] = defaultdict(list)
    assignments: set[tuple[str, str]] = set()
    for raw in votes:
        row = dict(raw)
        pair_id = str(row.get("pair_id", ""))
        judge_id = str(row.get("judge_id", ""))
        if pair_id not in pairs or judge_id not in JUDGE_IDS:
            raise ValueError("validation calibration vote has unknown identity")
        assignment = (pair_id, judge_id)
        if assignment in assignments:
            raise ValueError("duplicate validation calibration vote")
        assignments.add(assignment)
        if row.get("preference") not in {"A", "B", "tie"}:
            raise ValueError("invalid validation calibration preference")
        scores = row.get("scores")
        if not isinstance(scores, Mapping):
            raise ValueError("validation calibration vote lacks scores")
        for side in ("A", "B"):
            values = scores.get(side)
            if not isinstance(values, Mapping) or set(values) != set(SCORE_KEYS):
                raise ValueError("validation calibration score schema mismatch")
            if any(type(values[key]) is not int or not 1 <= values[key] <= 5 for key in SCORE_KEYS):
                raise ValueError("validation calibration score is outside 1--5")
        completion = row.get("terminal_syntax_complete")
        if not isinstance(completion, Mapping) or set(completion) != {"A", "B"}:
            raise ValueError("validation calibration completion schema mismatch")
        if any(type(completion[side]) is not bool for side in ("A", "B")):
            raise ValueError("validation calibration completion must be boolean")
        if not str(row.get("evidence", "")).strip():
            raise ValueError("validation calibration vote requires evidence")
        grouped[pair_id].append(row)
    expected = {(pair_id, judge_id) for pair_id in pairs for judge_id in JUDGE_IDS}
    if assignments != expected:
        raise ValueError("validation calibration vote set is incomplete")
    decisions = []
    for pair_id in sorted(pairs):
        pair_votes = sorted(grouped[pair_id], key=lambda row: row["judge_id"])
        counts = Counter(str(row["preference"]) for row in pair_votes)
        majority = "A" if counts["A"] >= 2 else "B" if counts["B"] >= 2 else "no_majority"
        decisions.append(
            {
                "pair_id": pair_id,
                "pair_type": pairs[pair_id]["pair_type"],
                "vote_counts": {key: counts[key] for key in ("A", "B", "tie")},
                "majority_preference": majority,
                "decisive": majority in {"A", "B"},
                "unanimous": counts[majority] == 3 if majority in {"A", "B"} else False,
                "votes": pair_votes,
            }
        )
    decisive = sum(row["decisive"] for row in decisions)
    unanimous = sum(row["unanimous"] for row in decisions)
    return {
        "calibration_version": packet["calibration_version"],
        "source_role": "validation_only_human_ai_calibration_not_dpo_training",
        "pair_count": len(decisions),
        "vote_count": len(votes),
        "judge_ids": list(JUDGE_IDS),
        "decisive_pair_count": decisive,
        "decisive_pair_rate": decisive / len(decisions),
        "unanimous_pair_count": unanimous,
        "unanimous_pair_rate": unanimous / len(decisions),
        "decisions": decisions,
        "correlated_ai_judges_are_not_independent_replicates": True,
        "eligible_for_dpo_training": False,
        "v7_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("artifacts/local/minerva_7b_v7_dpo/human_ai_calibration"),
    )
    args = parser.parse_args()
    root = args.calibration_dir
    print(
        "minerva-v7-dpo | start job=freeze_validation_calibration_ai_votes "
        "device=cpu total_steps=60 progress_interval=final-only",
        flush=True,
    )
    packet = json.loads((root / "packet.json").read_text(encoding="utf-8"))
    votes = [
        json.loads(line)
        for judge_id in JUDGE_IDS
        for line in (root / f"{judge_id}.votes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    aggregation = aggregate_validation_calibration_votes(
        packet=packet,
        votes=votes,
    )
    aggregation_hash = write_json_atomic(
        root / "ai_aggregation.frozen.json", aggregation
    )
    print(
        "minerva-v7-dpo | complete "
        f"pairs={aggregation['pair_count']} votes={aggregation['vote_count']} "
        f"decisive_rate={aggregation['decisive_pair_rate']:.3f} "
        f"unanimous_rate={aggregation['unanimous_pair_rate']:.3f} "
        f"aggregation_sha256={aggregation_hash} "
        "eligible_for_dpo_training=False v7_test_accessed=False",
        flush=True,
    )


if __name__ == "__main__":
    main()
