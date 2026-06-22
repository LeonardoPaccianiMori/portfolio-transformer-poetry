import json
from pathlib import Path

import pytest
import torch

from sonnet_evaluation.generation import (
    generate_for_prompts,
    generate_text,
    load_char_tokenizer,
    load_prompts,
    load_transformer_from_checkpoint,
    safe_prompt_filename,
)
from sonnet_model.transformer import CausalTransformerLanguageModel


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def build_tiny_model() -> CausalTransformerLanguageModel:
    return CausalTransformerLanguageModel(
        vocab_size=4,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        feed_forward_dim=16,
        max_context_length=8,
    )


def write_tiny_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    model = build_tiny_model()
    write_json(
        run_dir / "config.json",
        {
            "vocab_size": 4,
            "embedding_dim": 8,
            "num_layers": 1,
            "num_heads": 2,
            "head_dim": 4,
            "feed_forward_dim": 16,
            "max_context_length": 8,
        },
    )
    write_json(
        run_dir / "tokenizer.json",
        {
            "type": "character",
            "char_to_id": {
                "A": 0,
                "m": 1,
                "o": 2,
                "r": 3,
            },
        },
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": 4,
        },
        run_dir / "model.pt",
    )


def test_load_char_tokenizer_round_trips_text(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    write_json(
        tokenizer_path,
        {
            "type": "character",
            "char_to_id": {
                "A": 0,
                "m": 1,
            },
        },
    )

    tokenizer = load_char_tokenizer(tokenizer_path)

    assert tokenizer.decode(tokenizer.encode("Am")) == "Am"


def test_load_char_tokenizer_rejects_non_character_tokenizer(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    write_json(
        tokenizer_path,
        {
            "type": "bpe",
            "char_to_id": {},
        },
    )

    with pytest.raises(ValueError, match="character"):
        load_char_tokenizer(tokenizer_path)


def test_load_prompts_reads_prompt_list(tmp_path):
    prompts_path = tmp_path / "prompts.json"
    write_json(
        prompts_path,
        [
            {
                "id": "amor",
                "text": "Amor",
            },
        ],
    )

    prompts = load_prompts(prompts_path)

    assert prompts == [{"id": "amor", "text": "Amor"}]


def test_load_prompts_rejects_missing_fields(tmp_path):
    prompts_path = tmp_path / "prompts.json"
    write_json(
        prompts_path,
        [
            {
                "id": "amor",
            },
        ],
    )

    with pytest.raises(ValueError, match="id and text"):
        load_prompts(prompts_path)


def test_load_transformer_from_checkpoint_restores_model(tmp_path):
    run_dir = tmp_path / "run"
    write_tiny_run(run_dir)

    model = load_transformer_from_checkpoint(
        checkpoint_path=run_dir / "model.pt",
        config_path=run_dir / "config.json",
        device=torch.device("cpu"),
    )

    assert not model.training
    assert model.output_projection.out_features == 4


def test_generate_text_returns_prompt_plus_generated_text(tmp_path):
    run_dir = tmp_path / "run"
    write_tiny_run(run_dir)
    tokenizer = load_char_tokenizer(run_dir / "tokenizer.json")
    model = load_transformer_from_checkpoint(
        checkpoint_path=run_dir / "model.pt",
        config_path=run_dir / "config.json",
        device=torch.device("cpu"),
    )

    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="Amor",
        max_new_tokens=3,
        device=torch.device("cpu"),
        seed=123,
    )

    assert text.startswith("Amor")
    assert len(text) == 7


def test_safe_prompt_filename_sanitizes_prompt_id():
    assert safe_prompt_filename("solo et pensoso") == "solo_et_pensoso.txt"


def test_safe_prompt_filename_rejects_empty_safe_name():
    with pytest.raises(ValueError, match="prompt id"):
        safe_prompt_filename("!!!")


def test_generate_for_prompts_writes_outputs_and_metadata(tmp_path):
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "outputs"
    write_tiny_run(run_dir)

    metadata = generate_for_prompts(
        run_dir=run_dir,
        prompts=[
            {
                "id": "amor",
                "text": "Amor",
            },
            {
                "id": "line start",
                "text": "A",
            },
        ],
        output_dir=output_dir,
        max_new_tokens=3,
        seed=123,
        device=torch.device("cpu"),
    )

    metadata_path = output_dir / "metadata.json"
    first_output = output_dir / "amor.txt"
    second_output = output_dir / "line_start.txt"

    assert metadata_path.is_file()
    assert first_output.is_file()
    assert second_output.is_file()
    assert first_output.read_text(encoding="utf-8").startswith("Amor")
    assert metadata["base_seed"] == 123
    assert metadata["max_new_tokens"] == 3
    assert len(metadata["generated_files"]) == 2
