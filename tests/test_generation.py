import json
from pathlib import Path

import pytest
import torch

from sonnet_corpus.bpe import train_bpe_tokenizer
from sonnet_evaluation.generation import (
    completed_non_empty_line_count,
    generate_for_prompts,
    generate_text,
    generate_text_result,
    line_count_stop_reached,
    load_char_tokenizer,
    load_prompts,
    load_tokenizer,
    load_transformer_from_checkpoint,
    non_empty_line_count,
    safe_prompt_filename,
    single_token_id_for_text,
    stop_reason_for_text,
    validate_stop_text,
)
from sonnet_model.normalization import RMSNorm
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


def write_biased_tiny_run(
    run_dir: Path,
    favored_token_id: int,
) -> None:
    run_dir.mkdir(parents=True)
    model = build_tiny_model()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

        model.output_projection.bias[favored_token_id] = 10.0

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


def write_tiny_bpe_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    model = build_tiny_model()
    tokenizer = train_bpe_tokenizer(
        texts=["Amor"],
        vocab_size=4,
    )
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
    tokenizer.save(run_dir / "tokenizer.json")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": 4,
        },
        run_dir / "model.pt",
    )


def build_tiny_model_with_vocab(vocab_size: int) -> CausalTransformerLanguageModel:
    return CausalTransformerLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        feed_forward_dim=16,
        max_context_length=8,
    )


def write_stop_biased_tiny_bpe_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    tokenizer = train_bpe_tokenizer(
        texts=["A\n<|poem_end|>"],
        vocab_size=4,
        special_tokens=["<|poem_end|>"],
    )
    model = build_tiny_model_with_vocab(tokenizer.vocab_size)
    stop_token_id = tokenizer.encode("<|poem_end|>")[0]
    newline_token_id = tokenizer.encode("\n")[0]

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

        model.output_projection.bias[stop_token_id] = 10.0
        model.output_projection.bias[newline_token_id] = 9.0

    write_json(
        run_dir / "config.json",
        {
            "vocab_size": tokenizer.vocab_size,
            "embedding_dim": 8,
            "num_layers": 1,
            "num_heads": 2,
            "head_dim": 4,
            "feed_forward_dim": 16,
            "max_context_length": 8,
        },
    )
    tokenizer.save(run_dir / "tokenizer.json")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": tokenizer.vocab_size,
        },
        run_dir / "model.pt",
    )


def write_line_biased_tiny_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    model = build_tiny_model()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

        model.output_projection.bias[1] = 10.0

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
                "\n": 1,
                "m": 2,
                "o": 3,
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


def test_load_tokenizer_loads_unicode_bpe_tokenizer(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = train_bpe_tokenizer(
        texts=["Amor"],
        vocab_size=4,
    )
    tokenizer.save(tokenizer_path)

    loaded = load_tokenizer(tokenizer_path)

    assert loaded.decode(loaded.encode("Amor")) == "Amor"


def test_load_tokenizer_rejects_unknown_tokenizer_type(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    write_json(
        tokenizer_path,
        {
            "type": "unknown",
        },
    )

    with pytest.raises(ValueError, match="unsupported tokenizer"):
        load_tokenizer(tokenizer_path)


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


def test_load_transformer_from_checkpoint_restores_rms_norm_architecture(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    config_path = tmp_path / "config.json"
    model = CausalTransformerLanguageModel(
        vocab_size=4,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        feed_forward_dim=16,
        max_context_length=8,
        normalization_type="rms_norm",
    )
    write_json(
        config_path,
        {
            "vocab_size": 4,
            "embedding_dim": 8,
            "num_layers": 1,
            "num_heads": 2,
            "head_dim": 4,
            "feed_forward_dim": 16,
            "max_context_length": 8,
            "normalization_type": "rms_norm",
            "normalization_eps": 1e-5,
        },
    )
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    restored = load_transformer_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=torch.device("cpu"),
    )

    assert restored.normalization_type == "rms_norm"
    assert isinstance(restored.final_layer_norm, RMSNorm)


def test_load_transformer_from_checkpoint_restores_rope_architecture(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    config_path = tmp_path / "config.json"
    model = CausalTransformerLanguageModel(
        vocab_size=4,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        feed_forward_dim=16,
        max_context_length=8,
        position_encoding_type="rope",
        rope_theta=10_000.0,
    )
    write_json(
        config_path,
        {
            "vocab_size": 4,
            "embedding_dim": 8,
            "num_layers": 1,
            "num_heads": 2,
            "head_dim": 4,
            "feed_forward_dim": 16,
            "max_context_length": 8,
            "position_encoding_type": "rope",
            "rope_theta": 10_000.0,
        },
    )
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    restored = load_transformer_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=torch.device("cpu"),
    )

    assert restored.position_encoding_type == "rope"
    assert restored.rope_theta == 10_000.0
    assert restored.embedding.position_embedding is None


def test_load_transformer_from_checkpoint_restores_swiglu_architecture(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    config_path = tmp_path / "config.json"
    model = CausalTransformerLanguageModel(
        vocab_size=4,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        feed_forward_dim=5,
        max_context_length=8,
        feed_forward_type="swiglu",
    )
    write_json(
        config_path,
        {
            "vocab_size": 4,
            "embedding_dim": 8,
            "num_layers": 1,
            "num_heads": 2,
            "head_dim": 4,
            "feed_forward_dim": 5,
            "max_context_length": 8,
            "feed_forward_type": "swiglu",
        },
    )
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    restored = load_transformer_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=torch.device("cpu"),
    )

    assert restored.feed_forward_type == "swiglu"
    assert restored.blocks[0].feed_forward.feed_forward_type == "swiglu"


def test_load_transformer_from_checkpoint_restores_tied_token_embeddings(tmp_path):
    checkpoint_path = tmp_path / "model.pt"
    config_path = tmp_path / "config.json"
    model = CausalTransformerLanguageModel(
        vocab_size=4,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        feed_forward_dim=16,
        max_context_length=8,
        tie_token_embeddings=True,
    )
    write_json(
        config_path,
        {
            "vocab_size": 4,
            "embedding_dim": 8,
            "num_layers": 1,
            "num_heads": 2,
            "head_dim": 4,
            "feed_forward_dim": 16,
            "max_context_length": 8,
            "tie_token_embeddings": True,
        },
    )
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    restored = load_transformer_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=torch.device("cpu"),
    )

    assert restored.tie_token_embeddings is True
    assert restored.embedding.token_embedding.weight is restored.output_projection.weight


def test_load_transformer_from_checkpoint_supports_selection_architecture(tmp_path):
    run_dir = tmp_path / "run"
    selection_path = tmp_path / "selected_checkpoint.json"
    write_tiny_run(run_dir)
    write_json(
        selection_path,
        {
            "model_architecture": {
                "vocab_size": 4,
                "embedding_dim": 8,
                "num_layers": 1,
                "num_heads": 2,
                "head_dim": 4,
                "feed_forward_dim": 16,
                "max_context_length": 8,
            },
        },
    )

    model = load_transformer_from_checkpoint(
        checkpoint_path=run_dir / "model.pt",
        config_path=selection_path,
        device=torch.device("cpu"),
    )

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


def test_non_empty_line_count_ignores_blank_lines():
    assert non_empty_line_count("A\n\nm\n") == 2


def test_completed_non_empty_line_count_ignores_unfinished_last_line():
    assert completed_non_empty_line_count("A\nm") == 1
    assert completed_non_empty_line_count("A\nm\n") == 2


def test_line_count_stop_reached_detects_target_line_count():
    assert line_count_stop_reached("A\nm\n", target_lines=2)
    assert not line_count_stop_reached("A\n", target_lines=2)
    assert not line_count_stop_reached("A\nm", target_lines=2)
    assert not line_count_stop_reached("A\n", target_lines=None)


def test_line_count_stop_reached_rejects_invalid_target_lines():
    with pytest.raises(ValueError, match="target_lines"):
        line_count_stop_reached("A\n", target_lines=0)


def test_validate_stop_text_rejects_empty_stop_text(tmp_path):
    run_dir = tmp_path / "run"
    write_tiny_run(run_dir)
    tokenizer = load_char_tokenizer(run_dir / "tokenizer.json")

    with pytest.raises(ValueError, match="stop_text"):
        validate_stop_text(tokenizer, "")


def test_single_token_id_for_text_rejects_multi_token_text(tmp_path):
    run_dir = tmp_path / "run"
    write_tiny_run(run_dir)
    tokenizer = load_char_tokenizer(run_dir / "tokenizer.json")

    with pytest.raises(ValueError, match="exactly one token"):
        single_token_id_for_text(tokenizer, "Am")


def test_stop_reason_for_text_prefers_stop_text_over_target_lines():
    reason = stop_reason_for_text(
        text="A\nm",
        stop_text="m",
        target_lines=2,
    )

    assert reason == "stop_text"


def test_generate_text_result_stops_on_stop_text(tmp_path):
    run_dir = tmp_path / "run"
    write_biased_tiny_run(run_dir, favored_token_id=1)
    tokenizer = load_char_tokenizer(run_dir / "tokenizer.json")
    model = load_transformer_from_checkpoint(
        checkpoint_path=run_dir / "model.pt",
        config_path=run_dir / "config.json",
        device=torch.device("cpu"),
    )

    result = generate_text_result(
        model=model,
        tokenizer=tokenizer,
        prompt="A",
        max_new_tokens=5,
        device=torch.device("cpu"),
        seed=123,
        temperature=1.0,
        top_k=1,
        stop_text="m",
    )

    assert result["text"] == "Am"
    assert result["stop_reason"] == "stop_text"
    assert result["generated_new_tokens"] == 1


def test_generate_text_result_stops_after_target_line_is_completed(tmp_path):
    run_dir = tmp_path / "run"
    write_line_biased_tiny_run(run_dir)
    tokenizer = load_char_tokenizer(run_dir / "tokenizer.json")
    model = load_transformer_from_checkpoint(
        checkpoint_path=run_dir / "model.pt",
        config_path=run_dir / "config.json",
        device=torch.device("cpu"),
    )

    result = generate_text_result(
        model=model,
        tokenizer=tokenizer,
        prompt="A",
        max_new_tokens=5,
        device=torch.device("cpu"),
        seed=123,
        top_k=1,
        target_lines=1,
    )

    assert result["text"] == "A\n"
    assert result["stop_reason"] == "target_lines"
    assert result["generated_new_tokens"] == 1


def test_generate_text_result_suppresses_one_token_stop_text_until_target_lines(tmp_path):
    run_dir = tmp_path / "run"
    write_stop_biased_tiny_bpe_run(run_dir)
    tokenizer = load_tokenizer(run_dir / "tokenizer.json")
    model = load_transformer_from_checkpoint(
        checkpoint_path=run_dir / "model.pt",
        config_path=run_dir / "config.json",
        device=torch.device("cpu"),
    )

    result = generate_text_result(
        model=model,
        tokenizer=tokenizer,
        prompt="A",
        max_new_tokens=5,
        device=torch.device("cpu"),
        seed=123,
        top_k=1,
        stop_text="<|poem_end|>",
        target_lines=1,
        suppress_stop_text_until_target_lines=True,
    )

    assert result["text"] == "A\n"
    assert result["stop_reason"] == "target_lines"
    assert "<|poem_end|>" not in result["text"]


def test_generate_text_result_requires_stop_text_for_stop_suppression(tmp_path):
    run_dir = tmp_path / "run"
    write_tiny_run(run_dir)
    tokenizer = load_char_tokenizer(run_dir / "tokenizer.json")
    model = load_transformer_from_checkpoint(
        checkpoint_path=run_dir / "model.pt",
        config_path=run_dir / "config.json",
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="stop_text"):
        generate_text_result(
            model=model,
            tokenizer=tokenizer,
            prompt="A",
            max_new_tokens=1,
            device=torch.device("cpu"),
            seed=123,
            target_lines=1,
            suppress_stop_text_until_target_lines=True,
        )


def test_generate_text_result_requires_single_token_stop_text_for_stop_suppression(tmp_path):
    run_dir = tmp_path / "run"
    write_tiny_run(run_dir)
    tokenizer = load_char_tokenizer(run_dir / "tokenizer.json")
    model = load_transformer_from_checkpoint(
        checkpoint_path=run_dir / "model.pt",
        config_path=run_dir / "config.json",
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="exactly one token"):
        generate_text_result(
            model=model,
            tokenizer=tokenizer,
            prompt="A",
            max_new_tokens=1,
            device=torch.device("cpu"),
            seed=123,
            stop_text="Am",
            target_lines=1,
            suppress_stop_text_until_target_lines=True,
        )


def test_safe_prompt_filename_sanitizes_prompt_id():
    assert safe_prompt_filename("solo et pensoso") == "solo_et_pensoso.txt"


def test_safe_prompt_filename_rejects_empty_safe_name():
    with pytest.raises(ValueError, match="prompt id"):
        safe_prompt_filename("!!!")


def test_generate_for_prompts_writes_outputs_and_metadata(tmp_path):
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "outputs"
    write_tiny_run(run_dir)

    progress_messages = []
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
        progress=progress_messages.append,
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
    assert metadata["temperature"] == 1.0
    assert metadata["top_k"] is None
    assert metadata["stop_text"] is None
    assert metadata["target_lines"] is None
    assert len(metadata["generated_files"]) == 2
    assert f"loading tokenizer from {run_dir / 'tokenizer.json'}" in progress_messages
    assert "generating prompt 1/2: amor" in progress_messages
    assert f"wrote prompt 2/2: {second_output}" in progress_messages


def test_generate_for_prompts_supports_bpe_tokenizer(tmp_path):
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "outputs"
    write_tiny_bpe_run(run_dir)

    metadata = generate_for_prompts(
        run_dir=run_dir,
        prompts=[
            {
                "id": "amor",
                "text": "Amor",
            },
        ],
        output_dir=output_dir,
        max_new_tokens=1,
        seed=123,
        device=torch.device("cpu"),
    )

    first_output = output_dir / "amor.txt"

    assert first_output.is_file()
    assert first_output.read_text(encoding="utf-8").startswith("Amor")
    assert len(metadata["generated_files"]) == 1


def test_generate_for_prompts_records_explicit_checkpoint_and_config_paths(tmp_path):
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "outputs"
    selection_path = tmp_path / "selected_checkpoint.json"
    write_tiny_run(run_dir)
    write_json(
        selection_path,
        {
            "model_architecture": {
                "vocab_size": 4,
                "embedding_dim": 8,
                "num_layers": 1,
                "num_heads": 2,
                "head_dim": 4,
                "feed_forward_dim": 16,
                "max_context_length": 8,
            },
        },
    )

    metadata = generate_for_prompts(
        run_dir=run_dir,
        prompts=[{"id": "amor", "text": "Amor"}],
        output_dir=output_dir,
        max_new_tokens=1,
        seed=123,
        device=torch.device("cpu"),
        checkpoint_path=run_dir / "model.pt",
        model_config_path=selection_path,
    )

    assert metadata["checkpoint_path"] == str(run_dir / "model.pt")
    assert metadata["model_config_path"] == str(selection_path)


def test_generate_for_prompts_records_controlled_decoding_metadata(tmp_path):
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "outputs"
    write_biased_tiny_run(run_dir, favored_token_id=1)

    metadata = generate_for_prompts(
        run_dir=run_dir,
        prompts=[
            {
                "id": "amor",
                "text": "A",
            },
        ],
        output_dir=output_dir,
        max_new_tokens=5,
        seed=123,
        device=torch.device("cpu"),
        temperature=0.8,
        top_k=1,
        stop_text="m",
        target_lines=14,
        suppress_stop_text_until_target_lines=False,
    )

    assert metadata["temperature"] == 0.8
    assert metadata["top_k"] == 1
    assert metadata["stop_text"] == "m"
    assert metadata["target_lines"] == 14
    assert metadata["suppress_stop_text_until_target_lines"] is False
    assert metadata["generated_files"][0]["stop_reason"] == "stop_text"
    assert metadata["generated_files"][0]["generated_new_tokens"] == 1
