import pytest

from scripts.freeze_minerva_v7_validation_calibration_ai_votes import (
    JUDGE_IDS,
    SCORE_KEYS,
    aggregate_validation_calibration_votes,
)


def test_validation_calibration_uses_three_distinct_recorded_judges():
    assert JUDGE_IDS == ("ai_judge_1", "ai_judge_2", "ai_judge_3")
    assert len(set(JUDGE_IDS)) == 3


def test_validation_calibration_aggregation_cannot_export_dpo_examples():
    packet = {
        "calibration_version": "v",
        "pair_count": 20,
        "pairs": [
            {"pair_id": f"calibration_pair_{index}", "pair_type": "literary"}
            for index in range(20)
        ],
    }
    side = {key: 3 for key in SCORE_KEYS}
    votes = [
        {
            "pair_id": pair["pair_id"],
            "judge_id": judge,
            "preference": "A",
            "scores": {"A": side, "B": side},
            "terminal_syntax_complete": {"A": True, "B": False},
            "evidence": "A has the more coherent ending.",
        }
        for pair in packet["pairs"]
        for judge in JUDGE_IDS
    ]
    result = aggregate_validation_calibration_votes(packet=packet, votes=votes)
    assert result["pair_count"] == 20
    assert result["vote_count"] == 60
    assert result["decisive_pair_rate"] == 1.0
    assert result["eligible_for_dpo_training"] is False
    assert all("chosen_text" not in row for row in result["decisions"])


def test_validation_calibration_aggregation_rejects_missing_vote():
    packet = {
        "calibration_version": "v",
        "pair_count": 20,
        "pairs": [
            {"pair_id": f"calibration_pair_{index}", "pair_type": "literary"}
            for index in range(20)
        ],
    }
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_validation_calibration_votes(packet=packet, votes=[])
