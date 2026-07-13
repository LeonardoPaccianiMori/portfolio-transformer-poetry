from pathlib import Path

import pytest
import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer, train_bpe_tokenizer
from sonnet_model.normalization import RMSNorm
from sonnet_model.transformer import CausalTransformerLanguageModel
from sonnet_training.rmsnorm_conversion import (
    initialize_rms_norm_conversion_from_parent,
)


def write_layer_norm_parent(tmp_path: Path) -> tuple[Path, BytePairEncodingTokenizer]:
    tokenizer = train_bpe_tokenizer(
        texts=["Amor antico\n"],
        vocab_size=20,
        special_tokens=["<|endoftext|>"],
    )
    model = CausalTransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        feed_forward_dim=16,
        max_context_length=8,
    )
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_(float(index + 1))

    checkpoint_path = tmp_path / "parent.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "config": {
                "embedding_dim": 8,
                "num_layers": 1,
                "num_heads": 2,
                "head_dim": 4,
                "feed_forward_dim": 16,
                "max_context_length": 8,
            },
            "vocab_size": tokenizer.vocab_size,
        },
        checkpoint_path,
    )
    return checkpoint_path, tokenizer


def extended_tokenizer(tokenizer: BytePairEncodingTokenizer) -> BytePairEncodingTokenizer:
    return BytePairEncodingTokenizer(
        token_to_id={**tokenizer.token_to_id, "ü": tokenizer.vocab_size},
        merges=tokenizer.merges,
        special_tokens=tokenizer.special_tokens,
    )


def test_rms_norm_conversion_copies_shared_weights_and_scales(tmp_path):
    checkpoint_path, tokenizer = write_layer_norm_parent(tmp_path)
    parent = torch.load(checkpoint_path, map_location="cpu")
    target_tokenizer = extended_tokenizer(tokenizer)

    model, optimizer, checkpoint, metadata = initialize_rms_norm_conversion_from_parent(
        checkpoint_path=checkpoint_path,
        tokenizer=target_tokenizer,
        learning_rate=3e-5,
        device=torch.device("cpu"),
    )

    assert checkpoint["vocab_size"] == tokenizer.vocab_size
    assert optimizer.state == {}
    assert isinstance(model.final_layer_norm, RMSNorm)
    assert torch.equal(
        model.embedding.token_embedding.weight[: tokenizer.vocab_size],
        parent["model_state_dict"]["embedding.token_embedding.weight"],
    )
    assert torch.equal(
        model.blocks[0].attention_layer_norm.weight,
        parent["model_state_dict"]["blocks.0.attention_layer_norm.weight"],
    )
    assert torch.equal(
        model.final_layer_norm.weight,
        parent["model_state_dict"]["final_layer_norm.weight"],
    )
    assert metadata["conversion_type"] == "layer_norm_to_rms_norm"
    assert metadata["target_vocab_size"] == target_tokenizer.vocab_size
    assert metadata["optimizer_state_restored"] is False
    assert set(metadata["dropped_layer_norm_bias_keys"]) == {
        "blocks.0.attention_layer_norm.bias",
        "blocks.0.feed_forward_layer_norm.bias",
        "final_layer_norm.bias",
    }


def test_rms_norm_conversion_rejects_non_layer_norm_parent(tmp_path):
    checkpoint_path, tokenizer = write_layer_norm_parent(tmp_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["config"]["normalization_type"] = "rms_norm"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="LayerNorm parent"):
        initialize_rms_norm_conversion_from_parent(
            checkpoint_path=checkpoint_path,
            tokenizer=tokenizer,
            learning_rate=3e-5,
            device=torch.device("cpu"),
        )


def test_rms_norm_conversion_rejects_incompatible_parent_weight_shape(tmp_path):
    checkpoint_path, tokenizer = write_layer_norm_parent(tmp_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["model_state_dict"]["output_projection.weight"] = torch.zeros(3, 8)
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="incompatible"):
        initialize_rms_norm_conversion_from_parent(
            checkpoint_path=checkpoint_path,
            tokenizer=tokenizer,
            learning_rate=3e-5,
            device=torch.device("cpu"),
        )


def test_rms_norm_conversion_rejects_nonpositive_learning_rate(tmp_path):
    checkpoint_path, tokenizer = write_layer_norm_parent(tmp_path)

    with pytest.raises(ValueError, match="learning_rate"):
        initialize_rms_norm_conversion_from_parent(
            checkpoint_path=checkpoint_path,
            tokenizer=tokenizer,
            learning_rate=0.0,
            device=torch.device("cpu"),
        )
