import json
from pathlib import Path

import pytest
import torch

from sonnet_evaluation.checkpoint_neighborhood import (
    generate_checkpoint_neighborhoods,
    load_checkpoint_neighborhood_plan,
)
from sonnet_model.transformer import CausalTransformerLanguageModel


def write_tiny_run(repo_root: Path) -> Path:
    run_dir = repo_root / "runs" / "tiny"
    run_dir.mkdir(parents=True)
    model = CausalTransformerLanguageModel(
        vocab_size=4,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        feed_forward_dim=16,
        max_context_length=8,
    )
    (run_dir / "config.json").write_text(
        json.dumps({
            "vocab_size": 4,
            "embedding_dim": 8,
            "num_layers": 1,
            "num_heads": 2,
            "head_dim": 4,
            "feed_forward_dim": 16,
            "max_context_length": 8,
        }),
        encoding="utf-8",
    )
    (run_dir / "tokenizer.json").write_text(
        json.dumps({
            "type": "character",
            "char_to_id": {"A": 0, "m": 1, "o": 2, "r": 3},
        }),
        encoding="utf-8",
    )
    torch.save({"model_state_dict": model.state_dict()}, run_dir / "model.pt")
    return run_dir


def write_plan(path: Path) -> None:
    path.write_text(
        json.dumps({
            "runs": [
                {
                    "id": "tiny",
                    "run_dir": "runs/tiny",
                    "selected_checkpoint_id": "best",
                    "checkpoints": [
                        {
                            "id": "best",
                            "checkpoint_path": "runs/tiny/model.pt",
                            "step": 1,
                            "validation_loss": 1.0,
                        },
                    ],
                },
            ],
        }),
        encoding="utf-8",
    )


def test_generate_checkpoint_neighborhoods_writes_prompt_outputs_and_metadata(
    tmp_path: Path,
):
    write_tiny_run(tmp_path)
    plan_path = tmp_path / "plan.json"
    prompts_path = tmp_path / "prompts.json"
    write_plan(plan_path)
    prompts_path.write_text(
        json.dumps([{"id": "amor", "text": "Amor"}]),
        encoding="utf-8",
    )
    progress_messages = []

    metadata = generate_checkpoint_neighborhoods(
        repo_root=tmp_path,
        plan_path=plan_path,
        prompts_path=prompts_path,
        output_root=tmp_path / "outputs",
        max_new_tokens=2,
        seed=123,
        device=torch.device("cpu"),
        progress=progress_messages.append,
    )

    output_path = tmp_path / "outputs" / "tiny" / "best" / "amor.txt"
    assert output_path.is_file()
    assert metadata["runs"][0]["selected_checkpoint_id"] == "best"
    assert metadata["runs"][0]["checkpoints"][0]["step"] == 1
    assert (tmp_path / "outputs" / "metadata.json").is_file()
    assert "run 1/1: tiny" in progress_messages
    assert "tiny checkpoint 1/1: best (step 1)" in progress_messages
    assert any(message.startswith("tiny/best | generating prompt") for message in progress_messages)


def test_load_checkpoint_neighborhood_plan_rejects_unknown_selected_checkpoint(
    tmp_path: Path,
):
    plan_path = tmp_path / "plan.json"
    write_plan(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["runs"][0]["selected_checkpoint_id"] = "missing"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="selected_checkpoint_id"):
        load_checkpoint_neighborhood_plan(plan_path)
