import json

import pytest

from scripts.prepare_minerva_v7_dpo_review import (
    _calibration_markdown,
    _combine_pair_packets,
)


def _packet(prefix, pair_type=None):
    row = {
        "pair_id": f"{prefix}_1", "prompt_id": "p", "opening_line": "o",
        "candidate_a": "poem A", "candidate_b": "poem B",
    }
    if pair_type:
        row["pair_type"] = pair_type
    return {
        "preference_version": "minerva_7b_v7_dpo_preferences_v1",
        "pair_count": 1, "pairs": [row], "v7_test_accessed": False,
    }


def test_review_packet_combines_pair_types_without_revealing_answers():
    combined = _combine_pair_packets(
        _packet("pair"),
        _packet("completion_pair", "terminal_completion_contrast"),
    )
    assert combined["pair_count"] == 2
    assert combined["component_pair_counts"] == {
        "clean_literary_comparison": 1,
        "terminal_completion_contrast": 1,
    }
    markdown = _calibration_markdown({"pairs": combined["pairs"]})
    assert "poem A" in markdown and "poem B" in markdown
    assert "AI majority answers are intentionally absent" in markdown
    assert "expected_complete_side" not in markdown
    assert markdown.count("Scores for A (1--5)") == 2
    assert markdown.count("Terminal syntax genuinely complete") == 4


def test_review_packet_rejects_duplicate_pair_ids():
    packet = _packet("pair")
    with pytest.raises(ValueError, match="overlap"):
        _combine_pair_packets(packet, json.loads(json.dumps(packet)))
