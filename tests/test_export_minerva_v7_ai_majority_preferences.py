from scripts.export_minerva_v7_ai_majority_preferences import build_ai_majority_dataset


def test_ai_majority_export_is_explicitly_not_human_calibrated():
    aggregation = {
        "preference_version": "v", "pair_count": 1, "decisive_pair_count": 1,
        "decisive_pair_rate": 1.0, "unanimous_pair_count": 0,
        "unanimous_pair_rate": 0.0,
        "decisions": [{
            "pair_id": "pair_1", "prompt_id": "p", "decisive": True,
            "chosen_candidate_id": "a", "rejected_candidate_id": "b",
            "chosen_text": "chosen", "rejected_text": "rejected",
            "vote_counts": {"A": 2, "B": 1, "tie": 0}, "unanimous": False,
        }],
    }
    pairs = {"pairs": [{
        "pair_id": "pair_1", "pair_type": "literary", "opening_line": "Prima",
    }]}
    result = build_ai_majority_dataset(aggregation, pairs)
    assert result["example_count"] == 1
    assert result["human_ai_agreement_rate"] == 0.6
    assert result["human_calibration_gate_passed"] is False
    assert result["validation_calibration_pairs_included"] is False
    assert result["dpo_training_authorized_as_ai_judge_distillation"] is True
    assert result["human_aligned_claim_authorized"] is False
