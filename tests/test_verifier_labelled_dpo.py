import json
from pathlib import Path

import pytest

from sonnet_training.verifier_labelled_dpo import (
    EXPERIMENT_VERSION,
    load_verifier_dpo_config,
    load_verifier_examples,
)

ROOT = Path(__file__).resolve().parents[1]


def test_load_verifier_dpo_config_accepts_the_repository_config():
    config = load_verifier_dpo_config(ROOT / "configs/verifier_labelled_dpo.json")
    assert config["experiment_version"] == EXPERIMENT_VERSION
    assert config["authorization"]["verifier_labelled_dpo_authorized"] is True


def test_config_mismatch_raises(tmp_path):
    payload = json.loads(
        (ROOT / "configs/verifier_labelled_dpo.json").read_text(encoding="utf-8")
    )
    payload["lora_rank"] = 64
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_verifier_dpo_config(path)


def test_load_verifier_examples_maps_pairs(tmp_path):
    payload = {
        "preference_version": "minerva_7b_v7_verifier_form_preferences_v1",
        "scope": "exploratory_verifier_labelled_form_dpo",
        "source_split": "sonnets_train",
        "pair_type": "verifier_form_contrast",
        "v7_test_accessed": False,
        "pairs": [
            {
                "pair_id": "pair_1",
                "pair_type": "verifier_form_contrast",
                "prompt_id": "p1",
                "opening_line": "x",
                "chosen": "a",
                "rejected": "b",
            }
        ],
    }
    path = tmp_path / "prefs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    examples = load_verifier_examples(path)
    assert len(examples) == 1
    assert examples[0].vote_counts == {}
    assert examples[0].chosen == "a"
