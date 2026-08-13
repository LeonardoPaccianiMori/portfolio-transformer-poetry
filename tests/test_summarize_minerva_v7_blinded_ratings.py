from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/summarize_minerva_v7_blinded_ratings.py"
SPEC = importlib.util.spec_from_file_location("blinded_summary", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def rating(**overrides):
    row = {
        "blind_id": "a",
        "grammar": 4,
        "historical_register": 3,
        "poetic_quality": 4,
        "sonnet_form_coherence": 4,
        "volta_argument": 3,
        "meta_text": "no",
        "truncation": "no",
        "evidence": "complete",
    }
    row.update(overrides)
    return row


def test_quality_thresholds_are_explicit():
    assert MODULE.qualifies_moderate(rating())
    assert MODULE.qualifies_strict(rating())
    assert MODULE.qualifies_moderate(rating(grammar=3, poetic_quality=3, sonnet_form_coherence=3))
    assert not MODULE.qualifies_strict(rating(grammar=3))
    assert not MODULE.qualifies_moderate(rating(meta_text="yes"))
    assert not MODULE.qualifies_moderate(rating(truncation="yes"))
    assert not MODULE.qualifies_moderate(rating(collapse=True))


def test_load_ratings_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "ratings.jsonl"
    path.write_text("\n".join(json.dumps(rating()) for _ in range(2)) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.load_ratings(path)


def test_summarize_counts_scores_and_quality_gates():
    rows = [
        rating(),
        rating(blind_id="b", grammar=3, poetic_quality=3, sonnet_form_coherence=3),
        rating(blind_id="c", meta_text="yes", grammar=1, poetic_quality=1),
    ]
    summary = MODULE.summarize(rows)
    assert summary["rows"] == 3
    assert summary["grammar"]["counts"] == {"1": 1, "3": 1, "4": 1}
    assert summary["moderate_clean_count"] == 2
    assert summary["strict_good_count"] == 1


def test_paired_comparison_matches_prompt_and_reports_right_minus_left():
    rows = [
        rating(blind_id="a1", system_id="stage_3", prompt_id="p1", seed=1, grammar=2),
        rating(blind_id="a2", system_id="dpo", prompt_id="p1", seed=1, grammar=4),
        rating(blind_id="b1", system_id="stage_3", prompt_id="p2", seed=2, grammar=3),
        rating(blind_id="b2", system_id="dpo", prompt_id="p2", seed=2, grammar=4),
    ]
    result = MODULE.paired_comparison(
        rows, left="stage_3", right="dpo", resamples=20, seed=7
    )
    assert result["paired_prompts"] == 2
    assert result["grammar"]["mean_paired_change"] == 1.5
    assert result["meta_text_free"]["mean_paired_change"] == 0.0
