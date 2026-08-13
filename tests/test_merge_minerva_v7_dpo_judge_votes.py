import json

import pytest

from scripts.merge_minerva_v7_dpo_judge_votes import merge_vote_chunks


def _vote(pair_id, judge_id):
    scores = {
        key: 3
        for key in (
            "grammar", "coherence", "historical_register", "poetic_force",
            "form", "volta_closure",
        )
    }
    return {
        "pair_id": pair_id,
        "judge_id": judge_id,
        "preference": "A",
        "scores": {"A": scores, "B": scores},
        "evidence": "A is more coherent.",
    }


def test_merge_vote_chunks_requires_exact_frozen_assignment_set(tmp_path):
    manifest = {
        "assignment_count": 2,
        "assignments": [
            {"judge_id": "j1", "pair": {"pair_id": "pair_1"}},
            {"judge_id": "j2", "pair": {"pair_id": "pair_1"}},
        ],
    }
    path = tmp_path / "votes.jsonl"
    path.write_text(
        "\n".join(json.dumps(_vote("pair_1", judge)) for judge in ("j1", "j2")),
        encoding="utf-8",
    )
    rows = merge_vote_chunks(assignment_manifest=manifest, vote_paths=[path])
    assert [(row["pair_id"], row["judge_id"]) for row in rows] == [
        ("pair_1", "j1"), ("pair_1", "j2")
    ]


def test_merge_vote_chunks_rejects_incomplete_set(tmp_path):
    manifest = {
        "assignment_count": 1,
        "assignments": [
            {"judge_id": "j1", "pair": {"pair_id": "pair_1"}},
        ],
    }
    with pytest.raises(ValueError, match="incomplete"):
        merge_vote_chunks(assignment_manifest=manifest, vote_paths=[])
