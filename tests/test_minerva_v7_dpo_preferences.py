import json
from pathlib import Path

import pytest
import torch
from types import SimpleNamespace

from sonnet_analysis.minerva_v7_dpo_preferences import (
    PROMPT_COUNT,
    build_blinded_pairs,
    analyze_preference_candidates,
    build_candidate_jobs,
    build_chosen_rejected_dataset,
    build_completion_contrast_pairs,
    build_judge_assignments,
    build_training_prompt_manifest,
    build_user_calibration_packet,
    deterministic_candidate_screen,
    evaluate_preference_gates,
    generate_preference_candidates,
    load_preference_config,
    load_verified_candidates,
    aggregate_judge_votes,
    score_user_calibration,
    validate_training_prompt_manifest,
    validate_vote,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[1]


class BatchTokenizer:
    all_special_ids = []
    eos_token_id = 0
    pad_token_id = None
    padding_side = "right"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return "P:"

    def __call__(self, texts, *, add_special_tokens, return_tensors, padding):
        return {
            "input_ids": torch.tensor([[1, 2] for _ in texts]),
            "attention_mask": torch.ones((len(texts), 2), dtype=torch.long),
        }

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        return "".join(chr(value) for value in ids)


class BatchModel:
    def eval(self):
        return self

    def __call__(self, *, input_ids, **kwargs):
        logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 256))
        return SimpleNamespace(logits=logits, past_key_values=object())


@pytest.fixture(scope="module")
def training_manifest():
    return build_training_prompt_manifest(
        document_index_path=ROOT / "data/local/minerva_7b_v7/encoded/sonnets_train.documents.jsonl",
        reference_manifest_path=ROOT / "artifacts/local/minerva_7b_v7_analysis/memorization_reference/manifest.json",
        validation_prompt_path=ROOT / "configs/minerva_7b_v7_exploratory_prompts.json",
    )


@pytest.mark.local_artifact
def test_training_prompt_manifest_is_balanced_training_only_and_reproducible(
    training_manifest, tmp_path
):
    second = build_training_prompt_manifest(
        document_index_path=ROOT / "data/local/minerva_7b_v7/encoded/sonnets_train.documents.jsonl",
        reference_manifest_path=ROOT / "artifacts/local/minerva_7b_v7_analysis/memorization_reference/manifest.json",
        validation_prompt_path=ROOT / "configs/minerva_7b_v7_exploratory_prompts.json",
    )
    assert training_manifest == second
    assert len(training_manifest["prompts"]) == PROMPT_COUNT
    assert training_manifest["excluded_validation_opening_count"] == 2
    assert all(row["source_split"] == "sonnets_train" for row in training_manifest["prompts"])
    path = tmp_path / "prompts.json"
    identity = write_json_atomic(path, training_manifest)
    assert validate_training_prompt_manifest(path, expected_sha256=identity) == training_manifest
    assert len(build_candidate_jobs(training_manifest["prompts"])) == 4096


@pytest.mark.local_artifact
def test_prompt_validator_rejects_held_out_lineage(training_manifest, tmp_path):
    broken = json.loads(json.dumps(training_manifest))
    broken["prompts"][0]["source_split"] = "sonnets_validation"
    path = tmp_path / "broken.json"
    write_json_atomic(path, broken)
    with pytest.raises(ValueError, match="training-only|held-out"):
        validate_training_prompt_manifest(path)


@pytest.mark.local_artifact
def test_candidate_config_is_bounded_and_generation_is_resumable(
    training_manifest, tmp_path
):
    config = load_preference_config(
        ROOT / "configs/minerva_7b_v7_dpo_preferences.json"
    )
    assert config["authorization"]["dpo_training_authorized"] is False
    config = {
        **config,
        "recipes": [
            {**recipe, "max_new_tokens": 1} for recipe in config["recipes"]
        ],
    }
    first = generate_preference_candidates(
        model=BatchModel(), tokenizer=BatchTokenizer(),
        prompts=training_manifest["prompts"], config=config,
        output_dir=tmp_path, device="cpu", batch_size=4, maximum_candidates=8,
    )
    second = generate_preference_candidates(
        model=BatchModel(), tokenizer=BatchTokenizer(),
        prompts=training_manifest["prompts"], config=config,
        output_dir=tmp_path, device="cpu", batch_size=4, maximum_candidates=8,
    )
    assert first["completed_candidate_count"] == 8
    assert second["completed_candidate_count"] == 16
    rows, marker = load_verified_candidates(tmp_path)
    assert len(rows) == marker["completed_candidate_count"] == 16
    assert all(row["source_split"] == "sonnets_train" for row in rows)


def _candidate(candidate_id, prompt_id, recipe_id, text):
    opening = "Nel chiaro lume nasce il mio pensiero,"
    return {
        "candidate_id": candidate_id,
        "prompt_id": prompt_id,
        "recipe_id": recipe_id,
        "seed": 7300,
        "opening_line": opening,
        "text": f"{opening}\n{text}",
    }


def test_deterministic_screen_records_every_failure_without_deleting_evidence():
    good_lines = "\n".join(
        [
            "ardente il core cerca sua ragione",
            "fra stelle antiche veglia la memoria",
            "e il vento reca un nome alla vittoria",
            "mentre la notte tace al mio sentiero",
            "la fonte pura specchia il cielo intero",
            "nel bosco trema ancora questa storia",
            "ma torna il giorno a sciogliere la gloria",
            "che chiuse il duolo in carcere severo",
            "ora si desta il canto alla marina",
            "e muta il pianto in limpida speranza",
            "quando la luce il nostro passo inchina",
            "così nel petto vive la costanza",
            "e trova pace al termine: abbastanza.",
        ]
    )
    good = _candidate("c1", "p1", "no_labels_balanced", good_lines)
    screen = deterministic_candidate_screen(good, memorization={"risk_level": "low"})
    assert screen["eligible_for_blind_pairing"]
    bad = {**good, "candidate_id": "c2", "text": "testo incompleto,"}
    rejected = deterministic_candidate_screen(bad, memorization={"risk_level": "high"})
    assert not rejected["eligible_for_blind_pairing"]
    assert "exact_opening" in rejected["rejection_reasons"]
    assert "no_high_risk_memorization" in rejected["rejection_reasons"]


def test_blind_pairs_are_same_prompt_order_invariant_and_hide_recipe_labels():
    opening = "Nel chiaro lume nasce il mio pensiero,"
    candidates = [
        {
            "candidate_id": f"c{index}", "prompt_id": "p1",
            "recipe_id": "no_labels_balanced" if index % 2 == 0 else "no_labels_creative",
            "seed": 7300 + index, "opening_line": opening,
            "text": f"{opening}\npoema {index}",
        }
        for index in range(4)
    ]
    assessments = [
        {"candidate_id": row["candidate_id"], "prompt_id": "p1", "eligible_for_blind_pairing": True}
        for row in candidates
    ]
    first = build_blinded_pairs(candidates=candidates, assessments=assessments)
    second = build_blinded_pairs(
        candidates=list(reversed(candidates)), assessments=list(reversed(assessments))
    )
    assert first == second
    public, private = first
    assert public["pair_count"] == 1
    assert "recipe" not in json.dumps(public).lower()
    assert private["mapping"][0]["candidate_a_recipe_id"]


def test_vote_calibration_and_abort_gates_are_explicit():
    scores = {
        side: {
            metric: 3
            for metric in (
                "grammar", "coherence", "historical_register", "poetic_force", "form", "volta_closure"
            )
        }
        for side in ("A", "B")
    }
    vote = validate_vote(
        {"pair_id": "pair_123", "judge_id": "judge_1", "preference": "A", "scores": scores, "evidence": "A closes its syntax."}
    )
    votes = [{**vote, "judge_id": f"judge_{index}"} for index in range(3)]
    gates = evaluate_preference_gates(
        completed_candidates=64, eligible_candidates=8, pair_count=100,
        high_risk_memorization_count=0, projected_full_minutes=35,
        projected_full_cost_usd=1.4, votes=votes,
        user_calibration_accuracy=0.85,
    )
    assert gates["dpo_training_gate_passed"]
    assert gates["dpo_training_authorized"] is False
    pairs = {
        "pairs": [
            {"pair_id": f"pair_{index}", "candidate_a": "a", "candidate_b": "b"}
            for index in range(20)
        ]
    }
    assert build_user_calibration_packet(pairs)["pair_count"] == 20


def test_candidate_analysis_projects_yield_pairs_cost_and_preserves_rejections():
    opening = "Nel chiaro lume nasce il mio pensiero,"
    lines = "\n".join(
        [
            "ardente il core cerca sua ragione",
            "fra stelle antiche veglia la memoria",
            "e il vento reca un nome alla vittoria",
            "mentre la notte tace al mio sentiero",
            "la fonte pura specchia il cielo intero",
            "nel bosco trema ancora questa storia",
            "ma torna il giorno a sciogliere la gloria",
            "che chiuse il duolo in carcere severo",
            "ora si desta il canto alla marina",
            "e muta il pianto in limpida speranza",
            "quando la luce il nostro passo inchina",
            "così nel petto vive la costanza",
            "e trova pace al termine: abbastanza.",
        ]
    )
    candidates = []
    for prompt_index in range(8):
        for recipe_index, recipe_id in enumerate(
            ("no_labels_balanced", "no_labels_creative")
        ):
            candidates.append(
                {
                    "candidate_id": f"c_{prompt_index}_{recipe_index}",
                    "prompt_id": f"p_{prompt_index}", "recipe_id": recipe_id,
                    "seed": 7300, "opening_line": opening,
                    "text": f"{opening}\n{lines}",
                }
            )
    memorization = [{"risk_level": "low"} for _ in candidates]
    report, assessments, pairs, mapping = analyze_preference_candidates(
        candidates=candidates, memorization_scores=memorization,
        generation_elapsed_seconds=8, hourly_rate_usd=2.384,
    )
    assert report["eligible_candidate_count"] == 16
    assert report["projected_full_pair_count"] == 512
    assert len(assessments) == 16
    assert pairs["pair_count"] == mapping["pair_count"] == 8
    assert report["gates"]["dpo_training_authorized"] is False


def test_three_judge_majority_and_human_ai_calibration_gate():
    pairs = {
        "pair_count": 20,
        "pairs": [
            {
                "pair_id": f"pair_{index:03d}", "prompt_id": f"p{index}",
                "opening_line": "Apertura", "candidate_a": f"A {index}",
                "candidate_b": f"B {index}",
            }
            for index in range(20)
        ],
    }
    mapping = {
        "mapping": [
            {
                "pair_id": f"pair_{index:03d}", "prompt_id": f"p{index}",
                "candidate_a_id": f"a{index}", "candidate_b_id": f"b{index}",
                "candidate_a_recipe_id": "balanced",
                "candidate_b_recipe_id": "creative",
            }
            for index in range(20)
        ]
    }
    assignments = build_judge_assignments(
        pairs, judge_ids=("judge_1", "judge_2", "judge_3")
    )
    assert assignments["assignment_count"] == 60
    score = {
        metric: 3
        for metric in (
            "grammar", "coherence", "historical_register", "poetic_force",
            "form", "volta_closure",
        )
    }
    votes = [
        {
            "pair_id": assignment["pair"]["pair_id"],
            "judge_id": assignment["judge_id"], "preference": "A",
            "scores": {"A": score, "B": score}, "evidence": "A is preferred.",
        }
        for assignment in assignments["assignments"]
    ]
    aggregation = aggregate_judge_votes(
        pairs=pairs, private_mapping=mapping, votes=votes,
        expected_judge_ids=("judge_1", "judge_2", "judge_3"),
    )
    assert aggregation["decisive_pair_rate"] == 1.0
    packet = build_user_calibration_packet(pairs, count=20)
    calibration = score_user_calibration(
        aggregation=aggregation, calibration_packet=packet,
        user_votes=[
            {"pair_id": row["pair_id"], "preference": "A"}
            for row in packet["pairs"]
        ],
    )
    assert calibration["agreement_rate"] == 1.0
    dataset = build_chosen_rejected_dataset(
        aggregation=aggregation, calibration=calibration
    )
    assert dataset["example_count"] == 20
    assert dataset["dpo_training_authorized"] is False


def test_incomplete_votes_and_failed_human_calibration_block_export():
    pairs = {
        "pair_count": 1,
        "pairs": [{
            "pair_id": "pair_1", "prompt_id": "p", "opening_line": "o",
            "candidate_a": "a", "candidate_b": "b",
        }],
    }
    mapping = {"mapping": [{
        "pair_id": "pair_1", "prompt_id": "p", "candidate_a_id": "a",
        "candidate_b_id": "b", "candidate_a_recipe_id": "x",
        "candidate_b_recipe_id": "y",
    }]}
    score = {
        metric: 3
        for metric in (
            "grammar", "coherence", "historical_register", "poetic_force",
            "form", "volta_closure",
        )
    }
    one_vote = [{
        "pair_id": "pair_1", "judge_id": "j1", "preference": "A",
        "scores": {"A": score, "B": score}, "evidence": "evidence",
    }]
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_judge_votes(
            pairs=pairs, private_mapping=mapping, votes=one_vote,
            expected_judge_ids=("j1", "j2", "j3"),
        )
    with pytest.raises(PermissionError, match="calibration"):
        build_chosen_rejected_dataset(
            aggregation={"decisive_pair_rate": 1.0, "decisions": []},
            calibration={"agreement_gate_at_least_080": False},
        )


def test_completion_contrasts_pair_complete_with_otherwise_safe_incomplete_text():
    opening = "Nel chiaro lume nasce il mio pensiero,"
    complete_candidate = {
        "candidate_id": "complete", "prompt_id": "p",
        "recipe_id": "balanced", "seed": 1, "opening_line": opening,
        "text": f"{opening}\nchiusa.",
    }
    incomplete_candidate = {
        "candidate_id": "incomplete", "prompt_id": "p",
        "recipe_id": "creative", "seed": 2, "opening_line": opening,
        "text": f"{opening}\nmentre",
    }
    base_checks = {
        "exact_opening": True, "exact_fourteen_lines": True,
        "meta_text_free": True, "no_repeated_line_collapse": True,
        "no_very_long_line": True, "below_repetition_threshold": True,
        "no_high_risk_memorization": True,
    }
    assessments = [
        {
            "candidate_id": "complete", "prompt_id": "p",
            "checks": {**base_checks, "complete_terminal_syntax": True},
        },
        {
            "candidate_id": "incomplete", "prompt_id": "p",
            "checks": {**base_checks, "complete_terminal_syntax": False},
        },
    ]
    first = build_completion_contrast_pairs(
        candidates=[complete_candidate, incomplete_candidate],
        assessments=assessments,
    )
    second = build_completion_contrast_pairs(
        candidates=[incomplete_candidate, complete_candidate],
        assessments=list(reversed(assessments)),
    )
    assert first == second
    public, private = first
    assert public["pair_count"] == private["pair_count"] == 1
    assert "expected_complete_side" not in json.dumps(public)
    assert private["mapping"][0]["expected_complete_side"] in {"A", "B"}
    assert "punctuation alone" in public["pairs"][0]["judge_instruction"]
