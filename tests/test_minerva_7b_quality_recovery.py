import copy
import json
from pathlib import Path

import pytest

from sonnet_evaluation.minerva_7b_quality_recovery import (
    RECOVERY_CONDITIONS,
    RECOVERY_OUTPUT_COUNT,
    _load_complete_condition,
    load_recovery_config,
    validate_recovery_config,
    validate_recovery_prompts,
    validate_stage_a_checkpoint,
)
from sonnet_evaluation.minerva_judge_gate import sha256_file
from sonnet_training.minerva_7b_qlora import (
    MINERVA_7B_INSTRUCT_MODEL_ID,
    MINERVA_7B_INSTRUCT_REVISION,
)


ROOT = Path(__file__).resolve().parents[1]


def test_recovery_config_and_validation_prompt_lock_are_consistent():
    config_path = ROOT / "configs/minerva_7b_quality_recovery.json"
    config = load_recovery_config(config_path)
    prompts = validate_recovery_prompts(config=config, repo_root=ROOT)

    assert len(RECOVERY_CONDITIONS) == 7
    assert len(prompts) == 12
    assert len(prompts) * len(RECOVERY_CONDITIONS) == RECOVERY_OUTPUT_COUNT
    assert len({prompt["author"] for prompt in prompts}) == 12
    assert config["prompt_sha256"] == sha256_file(
        ROOT / config["prompt_path"]
    )
    assert config["manifest_sha256"] == sha256_file(
        ROOT / config["manifest_path"]
    )


def test_recovery_prompts_are_disjoint_from_prior_and_final_sets():
    config = load_recovery_config(
        ROOT / "configs/minerva_7b_quality_recovery.json"
    )
    recovery = {
        row["poem_id"]
        for row in json.loads((ROOT / config["prompt_path"]).read_text())
    }
    excluded = set()
    for prompt_set in config["excluded_prompt_sets"]:
        excluded.update(
            row["poem_id"]
            for row in json.loads((ROOT / prompt_set["path"]).read_text())
        )
    assert recovery.isdisjoint(excluded)


def test_recovery_config_rejects_decoder_or_judge_threshold_changes():
    config = load_recovery_config(
        ROOT / "configs/minerva_7b_quality_recovery.json"
    )
    decoder_change = copy.deepcopy(config)
    decoder_change["generation"]["conditions"][0]["temperature"] = 0.1
    with pytest.raises(ValueError, match="generation recipe"):
        validate_recovery_config(decoder_change)

    judge_change = copy.deepcopy(config)
    judge_change["judge"]["thresholds"]["grammar_auroc"] = 0.1
    with pytest.raises(ValueError, match="thresholds"):
        validate_recovery_config(judge_change)

    artifact_change = copy.deepcopy(config)
    artifact_change["prompt_path"] = "configs/other.json"
    with pytest.raises(ValueError, match="prompt_path"):
        validate_recovery_config(artifact_change)


def test_stage_a_validation_requires_selected_preservation_checkpoint():
    checkpoint = {
        "checkpoint_type": "minerva_7b_historical_lora_adapter",
        "model_id": MINERVA_7B_INSTRUCT_MODEL_ID,
        "revision": MINERVA_7B_INSTRUCT_REVISION,
        "row": {"step": 4000, "preservation_gate_passed": True},
    }
    validate_stage_a_checkpoint(checkpoint)

    checkpoint["row"]["preservation_gate_passed"] = False
    with pytest.raises(ValueError, match="qualifying"):
        validate_stage_a_checkpoint(checkpoint)


def test_completed_recovery_condition_requires_matching_model_lineage(tmp_path):
    output_dir = tmp_path / "condition"
    output_dir.mkdir()
    output_path = output_dir / "prompt__seed_2029.txt"
    output_path.write_text("Opening\nContinuation\n", encoding="utf-8")
    checkpoint_path = tmp_path / "adapter.pt"
    prompt_path = tmp_path / "prompts.json"
    condition = RECOVERY_CONDITIONS[1]
    metadata = {
        "model_variant": "minerva_7b_recovery_stage_a_control",
        "model_id": MINERVA_7B_INSTRUCT_MODEL_ID,
        "revision": MINERVA_7B_INSTRUCT_REVISION,
        "adapter_checkpoint_path": str(checkpoint_path),
        "prompt_config_path": str(prompt_path),
        "conditioning_format": "minerva_chat_complete_sonnet_v1",
        "seeds": [2029],
        "max_new_tokens": 512,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "continuation_line_target": 13,
        "generated_files": [{
            "source_prompt_id": "prompt",
            "path": str(output_path),
        }],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    loaded = _load_complete_condition(
        output_dir=output_dir,
        condition=condition,
        prompts=[{"id": "prompt"}],
        checkpoint_path=checkpoint_path,
        prompt_config_path=prompt_path,
        model_id=MINERVA_7B_INSTRUCT_MODEL_ID,
        revision=MINERVA_7B_INSTRUCT_REVISION,
        conditioning_format="minerva_chat_complete_sonnet_v1",
    )
    assert loaded == metadata

    metadata["model_variant"] = "wrong"
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="model_variant"):
        _load_complete_condition(
            output_dir=output_dir,
            condition=condition,
            prompts=[{"id": "prompt"}],
            checkpoint_path=checkpoint_path,
            prompt_config_path=prompt_path,
            model_id=MINERVA_7B_INSTRUCT_MODEL_ID,
            revision=MINERVA_7B_INSTRUCT_REVISION,
            conditioning_format="minerva_chat_complete_sonnet_v1",
        )
