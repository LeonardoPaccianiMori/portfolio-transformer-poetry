import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sonnet_analysis.minerva_v7_exploratory_prompts import (
    PROMPT_COUNT, PROMPT_VERSION, validate_exploratory_prompt_manifest,
)
from sonnet_analysis.minerva_v7_high_volume_analysis import (
    analyze_high_volume_outputs, blinded_review_markdown,
    build_high_volume_blinded_sample,
)
from sonnet_analysis.minerva_v7_high_volume_generation import (
    EXPECTED_RECIPE_IDS, EXPECTED_SEEDS, generate_batch,
    generate_high_volume_state, load_high_volume_config,
)
from sonnet_analysis.minerva_v7_registry import MODEL_STATES


ROOT = Path(__file__).resolve().parents[1]


class BatchTokenizer:
    all_special_ids = []
    eos_token_id = 0
    pad_token_id = None
    padding_side = "right"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return "P:"

    def __call__(self, texts, *, add_special_tokens, return_tensors, padding):
        assert padding
        rows = [[1, 2] for _ in texts]
        return {
            "input_ids": torch.tensor(rows),
            "attention_mask": torch.ones((len(rows), 2), dtype=torch.long),
        }

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        return "".join(chr(value) for value in ids)


class BatchModel:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False
        return self

    def __call__(self, *, input_ids, **kwargs):
        logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], 256))
        return SimpleNamespace(logits=logits, past_key_values=object())


def _prompts(count=120):
    return [
        {
            "id": f"p{index:03d}", "opening_line": f"Prima {index}",
            "period": "13th_century", "source_split": "sonnets_validation",
        }
        for index in range(count)
    ]


def _recipes(max_new_tokens=1):
    return [
        {
            "recipe_id": recipe_id, "temperature": 0.7, "top_p": 0.92,
            "top_k": None, "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 4, "max_new_tokens": max_new_tokens,
            "continuation_line_target": 13,
        }
        for recipe_id in EXPECTED_RECIPE_IDS
    ]


def test_public_prompt_and_generation_contracts_are_frozen_and_test_free():
    prompts = validate_exploratory_prompt_manifest(
        ROOT / "configs/minerva_7b_v7_exploratory_prompts.json",
        expected_sha256="2f33aa518aa61c11193831e53b07fd3bd861a72bf68bb23c0e0e5b1a13b1d0c7",
    )
    config = load_high_volume_config(
        ROOT / "configs/minerva_7b_v7_high_volume_generation.json"
    )
    assert prompts["prompt_version"] == PROMPT_VERSION
    assert len(prompts["prompts"]) == PROMPT_COUNT
    assert config["outputs_per_state"] == 120 * 8 * 3
    assert config["all_seven_states_outputs"] == 20_160
    assert config["authorization"]["v7_test_access_authorized"] is False
    assert all(row["source_split"] == "sonnets_validation" for row in prompts["prompts"])


def test_batched_generation_is_deterministic_per_seed_and_keeps_token_evidence():
    jobs = [
        {"prompt": prompt, "seed": seed, "recipe": _recipes()[0]}
        for prompt, seed in zip(_prompts(2), EXPECTED_SEEDS[:2])
    ]
    first = generate_batch(
        model=BatchModel(), tokenizer=BatchTokenizer(), jobs=jobs,
        recipe=_recipes()[0], device="cpu",
    )
    second = generate_batch(
        model=BatchModel(), tokenizer=BatchTokenizer(), jobs=jobs,
        recipe=_recipes()[0], device="cpu",
    )
    assert [row["generated_token_ids"] for row in first] == [row["generated_token_ids"] for row in second]
    assert all(row["conditioning_input_ids"] == [1, 2] for row in first)
    assert all(row["batch_size"] == 2 for row in first)


def test_one_batch_qualification_is_separate_and_resumable(tmp_path):
    first = generate_high_volume_state(
        model=BatchModel(), tokenizer=BatchTokenizer(), state_id="untouched_parent",
        state_identity_sha256="a" * 64, prompts=_prompts(), seeds=EXPECTED_SEEDS,
        recipes=_recipes(), output_dir=tmp_path, device="cpu", batch_size=4,
        maximum_batches=1,
    )
    second = generate_high_volume_state(
        model=BatchModel(), tokenizer=BatchTokenizer(), state_id="untouched_parent",
        state_identity_sha256="a" * 64, prompts=_prompts(), seeds=EXPECTED_SEEDS,
        recipes=_recipes(), output_dir=tmp_path, device="cpu", batch_size=4,
        maximum_batches=1,
    )
    assert first["completed_output_count"] == 4
    assert second["completed_output_count"] == 8
    assert first["completion_scope"] == "qualification_or_incomplete_prefix"
    assert not (tmp_path / "complete.json").exists()
    assert (tmp_path / "qualification_or_progress.json").is_file()


def _write_authoritative_state(path: Path, state_id: str, identity: str) -> None:
    path.mkdir(parents=True)
    outputs = []
    for recipe_id in EXPECTED_RECIPE_IDS:
        for prompt in _prompts():
            for seed in EXPECTED_SEEDS:
                payload = {
                    "generation_version": "minerva_7b_v7_high_volume_generation_v1",
                    "analysis_role": "exploratory_high_volume",
                    "state_id": state_id, "state_identity_sha256": identity,
                    "prompt": prompt, "seed": seed,
                    "recipe": {"recipe_id": recipe_id},
                    "text": f"{prompt['opening_line']}\nSeconda\n" + "\n".join(["verso"] * 12),
                    "opening_line": prompt["opening_line"],
                    "v7_test_accessed": False,
                }
                name = hashlib.sha256(f"{state_id}|{prompt['id']}|{seed}|{recipe_id}".encode()).hexdigest()[:20] + ".json"
                target = path / name
                target.write_text(json.dumps(payload), encoding="utf-8")
                outputs.append(
                    {
                        "path": name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                        "prompt_id": prompt["id"], "seed": seed, "recipe_id": recipe_id,
                    }
                )
    completion = {
        "generation_version": "minerva_7b_v7_high_volume_generation_v1",
        "completion_scope": "authoritative_120_prompt_8_seed_3_recipe_grid",
        "completed_output_count": 2880, "outputs": outputs,
        "state_identity_sha256": identity, "v7_test_accessed": False,
    }
    (path / "complete.json").write_text(json.dumps(completion), encoding="utf-8")


def test_high_volume_analysis_covers_all_stages_bootstraps_prompts_and_blinds(tmp_path):
    directories = {}
    identities = {}
    for index, state in enumerate(MODEL_STATES):
        identity = f"{index:x}" * 64
        directory = tmp_path / state.state_id
        _write_authoritative_state(directory, state.state_id, identity)
        directories[state.state_id] = directory
        identities[state.state_id] = identity
    report = analyze_high_volume_outputs(
        state_directories=directories, expected_state_identities=identities,
        bootstrap_resamples=10, bootstrap_seed=8311, confidence_level=0.95,
    )
    assert len(report["rows"]) == 20_160
    assert len(report["summaries"]) == 7 * 3
    assert len(report["comparison_summaries"]) == 7 * 3
    blind = build_high_volume_blinded_sample(
        analysis_report=report, selected_prompt_count=24, selection_seed=8317
    )
    assert blind["sample_rows"] == 504
    markdown = blinded_review_markdown(blind)
    assert "untouched_parent" not in markdown
    assert "conservative" not in markdown


def test_high_volume_analysis_rejects_missing_state_and_tampered_output(tmp_path):
    with pytest.raises(ValueError, match="all seven"):
        analyze_high_volume_outputs(
            state_directories={}, expected_state_identities={},
            bootstrap_resamples=10, bootstrap_seed=1, confidence_level=0.95,
        )
