import copy
import json
from pathlib import Path

import pytest

from sonnet_evaluation.minerva_7b_parent_confirmation import (
    CONFIRMATION_CONDITIONS,
    CONFIRMATION_PROMPT_COUNT,
    _load_complete_condition,
    generate_parent_decoding_confirmation,
    load_confirmation_config,
    validate_confirmation_artifacts,
    validate_confirmation_config,
    validate_confirmation_prompts,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/minerva_7b_parent_decoding_confirmation.json"


@pytest.mark.local_artifact
def test_parent_confirmation_config_and_artifacts_are_frozen():
    config = load_confirmation_config(CONFIG_PATH)

    validate_confirmation_artifacts(config=config, repo_root=ROOT)
    prompts = validate_confirmation_prompts(config=config, repo_root=ROOT)

    assert len(prompts) == CONFIRMATION_PROMPT_COUNT
    assert len({prompt["author"] for prompt in prompts}) == 12


def test_parent_confirmation_rejects_recipe_change():
    config = load_confirmation_config(CONFIG_PATH)
    changed = copy.deepcopy(config)
    changed["generation"]["conditions"][1]["repetition_penalty"] = 1.05

    with pytest.raises(ValueError, match="generation recipe"):
        validate_confirmation_config(changed)


def test_parent_confirmation_requires_cuda(tmp_path: Path):
    with pytest.raises(ValueError, match="requires CUDA"):
        generate_parent_decoding_confirmation(
            repo_root=ROOT,
            config_path=CONFIG_PATH,
            output_root=tmp_path / "outputs",
            device="cpu",
            cache_dir=tmp_path / "cache",
        )


def test_parent_confirmation_resume_locks_lineage(tmp_path: Path):
    condition = CONFIRMATION_CONDITIONS[0]
    output_dir = tmp_path / condition["condition_id"]
    output_dir.mkdir()
    prompts = [
        {"id": f"prompt_{index}"} for index in range(CONFIRMATION_PROMPT_COUNT)
    ]
    generated = []
    for prompt in prompts:
        path = output_dir / f"{prompt['id']}__seed_4099.txt"
        path.write_text("opening\ncontinuation\n", encoding="utf-8")
        generated.append({
            "path": str(path),
            "source_prompt_id": prompt["id"],
        })
    metadata = {
        "model_variant": "minerva_7b_parent_confirmation_untouched_default",
        "model_id": "model",
        "revision": "revision",
        "adapter_checkpoint_path": None,
        "prompt_config_path": str(tmp_path / "prompts.json"),
        "conditioning_format": "format",
        "seeds": [4099],
        "max_new_tokens": 512,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "continuation_line_target": 13,
        "generated_files": generated,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    loaded = _load_complete_condition(
        output_dir=output_dir,
        condition=condition,
        prompts=prompts,
        checkpoint_path=None,
        prompt_config_path=tmp_path / "prompts.json",
        model_id="model",
        revision="revision",
        conditioning_format="format",
    )
    assert loaded == metadata

    metadata["revision"] = "changed"
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="revision"):
        _load_complete_condition(
            output_dir=output_dir,
            condition=condition,
            prompts=prompts,
            checkpoint_path=None,
            prompt_config_path=tmp_path / "prompts.json",
            model_id="model",
            revision="revision",
            conditioning_format="format",
        )
