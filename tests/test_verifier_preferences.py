import json
from pathlib import Path

import pytest

from sonnet_evaluation.verifier_preferences import (
    build_verifier_pairs,
    form_score,
    load_candidate_records,
    load_verifier_preferences,
    text_degeneracy,
    write_verifier_preferences,
)


def metrics(**overrides) -> dict:
    base = {
        "hendecasyllable_lines": 10,
        "quatrain_ok": False,
        "tercet_ok": False,
        "all_lines_valid": False,
        "rhyme_score": 0.5,
    }
    base.update(overrides)
    return base


def test_form_score_formula():
    assert form_score(metrics(rhyme_score=0.5)) == 10.5
    assert (
        form_score(
            metrics(
                hendecasyllable_lines=14,
                quatrain_ok=True,
                tercet_ok=True,
                all_lines_valid=True,
                rhyme_score=1.0,
            )
        )
        == 21.0
    )


def test_text_degeneracy_flags_broken_text():
    good = "Nel mezzo del cammin di nostra vita\nmi ritrovai per una selva oscura"
    broken = "a b c, d' e? f,, g h ' i j k"
    assert text_degeneracy(good)["degenerate"] is False
    assert text_degeneracy(broken)["degenerate"] is True


def test_build_verifier_pairs_picks_strongest_and_weakest():
    scored = [
        {"candidate_id": "c1", "prompt_id": "p1", "opening_line": "x", "text": "t1", "recipe_id": "r", "form_score": 3.0},
        {"candidate_id": "c2", "prompt_id": "p1", "opening_line": "x", "text": "t2", "recipe_id": "r", "form_score": 12.0},
        {"candidate_id": "c3", "prompt_id": "p1", "opening_line": "x", "text": "t3", "recipe_id": "r", "form_score": 9.0},
        {"candidate_id": "c4", "prompt_id": "p2", "opening_line": "y", "text": "t4", "recipe_id": "r", "form_score": 12.0},
    ]
    pairs = build_verifier_pairs(scored, min_gap=2.0, min_chosen_score=8.0)
    assert len(pairs) == 1
    assert pairs[0]["chosen_candidate_id"] == "c2"
    assert pairs[0]["rejected_candidate_id"] == "c1"


def test_build_verifier_pairs_skips_small_gaps():
    scored = [
        {"candidate_id": "c1", "prompt_id": "p1", "opening_line": "x", "text": "t1", "recipe_id": "r", "form_score": 9.0},
        {"candidate_id": "c2", "prompt_id": "p1", "opening_line": "x", "text": "t2", "recipe_id": "r", "form_score": 10.0},
    ]
    with pytest.raises(ValueError):
        build_verifier_pairs(scored, min_gap=2.0, min_chosen_score=8.0)


def test_write_and_load_verifier_preferences_round_trip(tmp_path):
    scored = [
        {
            "candidate_id": "c1",
            "prompt_id": "p1",
            "opening_line": "x",
            "text": "t1",
            "recipe_id": "r",
            "form_score": 3.0,
            "degeneracy": {"degenerate": False},
            "metrics": {},
        }
    ]
    pairs = [
        {
            "pair_id": "pair_1",
            "pair_type": "verifier_form_contrast",
            "prompt_id": "p1",
            "opening_line": "x",
            "chosen": "a",
            "rejected": "b",
            "chosen_score": 9.0,
            "rejected_score": 3.0,
            "chosen_candidate_id": "c2",
            "rejected_candidate_id": "c1",
        }
    ]
    path = tmp_path / "prefs.json"
    write_verifier_preferences(
        path,
        scored=scored,
        pairs=pairs,
        candidates_dir=tmp_path,
        min_gap=2.0,
        min_chosen_score=8.0,
    )
    payload = load_verifier_preferences(path)
    assert payload["pair_count"] == 1
    payload["preference_version"] = "bad"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_verifier_preferences(path)


def test_load_candidate_records_validates_lineage(tmp_path):
    payload = {
        "candidate_id": "cand_1",
        "prompt_id": "p1",
        "opening_line": "x",
        "recipe_id": "r",
        "seed": 1,
        "text": "abc",
        "source_split": "sonnets_train",
        "v7_test_accessed": False,
    }
    path = tmp_path / "candidate_1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    records = load_candidate_records(tmp_path)
    assert records[0]["candidate_id"] == "cand_1"
    payload["v7_test_accessed"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_candidate_records(tmp_path)
