from scripts.score_minerva_v7_validation_human_ai_calibration import compare_human_ai


def test_human_ai_comparison_reports_pair_type_rates_and_never_trains():
    human = [
        {"pair_id": "p1", "preference": "A", "scores": {}, "terminal_syntax_complete": {}, "reason": "r"},
        {"pair_id": "p2", "preference": "tie", "scores": {}, "terminal_syntax_complete": {}, "reason": "r"},
    ]
    aggregation = {
        "calibration_version": "v",
        "decisions": [
            {"pair_id": "p1", "pair_type": "literary", "majority_preference": "A", "vote_counts": {"A": 3, "B": 0, "tie": 0}},
            {"pair_id": "p2", "pair_type": "completion", "majority_preference": "B", "vote_counts": {"A": 0, "B": 3, "tie": 0}},
        ],
    }
    result = compare_human_ai(human_rows=human, aggregation=aggregation)
    assert result["agreement_count"] == 1
    assert result["agreement_rate"] == 0.5
    assert result["agreement_gate_at_least_080"] is False
    assert result["eligible_for_dpo_training"] is False
