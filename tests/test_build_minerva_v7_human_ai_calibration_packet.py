from pathlib import Path

from scripts.build_minerva_v7_human_ai_calibration_packet import (
    build_validation_calibration_packet,
)
from scripts.prepare_minerva_v7_dpo_review import _calibration_markdown


ROOT = Path(__file__).resolve().parents[1]


def test_validation_calibration_packet_is_balanced_blinded_and_reproducible():
    kwargs = {
        "analysis_path": ROOT / "artifacts/local/minerva_7b_v7_stage_3_no_labels_creative/analysis/analysis.json",
        "prompt_path": ROOT / "configs/minerva_7b_v7_exploratory_prompts.json",
    }
    first = build_validation_calibration_packet(**kwargs)
    second = build_validation_calibration_packet(**kwargs)
    assert first == second
    packet, mapping = first
    assert packet["pair_count"] == mapping["pair_count"] == 20
    assert packet["pair_type_counts"] == {
        "matched_literary_comparison": 10,
        "terminal_completion_contrast": 10,
    }
    assert packet["eligible_for_dpo_training"] is False
    private_metadata_keys = {
        "cell_id",
        "candidate_a_cell",
        "candidate_b_cell",
    }
    assert all(
        private_metadata_keys.isdisjoint(row)
        for row in packet["pairs"]
    )
    assert all(row["candidate_a_cell"] for row in mapping["mapping"])
    markdown = _calibration_markdown(packet)
    assert "no_labels_balanced" not in markdown
    assert "no_labels_creative" not in markdown
    assert markdown.count("Preference (A/B/tie)") == 20
