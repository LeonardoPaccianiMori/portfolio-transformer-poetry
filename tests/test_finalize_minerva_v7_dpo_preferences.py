import pytest

from scripts.finalize_minerva_v7_dpo_preferences import _combine


def test_finalize_combines_public_or_private_rows_deterministically():
    left = {
        "preference_version": "v", "pair_count": 1,
        "pairs": [{"pair_id": "pair_b"}],
    }
    right = {
        "preference_version": "v", "pair_count": 1,
        "pairs": [{"pair_id": "completion_pair_a"}],
    }
    combined = _combine(left, right, key="pairs")
    assert combined["pair_count"] == 2
    assert [row["pair_id"] for row in combined["pairs"]] == [
        "completion_pair_a", "pair_b"
    ]


def test_finalize_rejects_overlapping_pairs():
    packet = {
        "preference_version": "v", "pair_count": 1,
        "mapping": [{"pair_id": "pair_1"}],
    }
    with pytest.raises(ValueError, match="overlaps"):
        _combine(packet, packet, key="mapping")
