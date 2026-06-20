import pytest
import torch

from sonnet_model.transformer import TokenAndPositionEmbedding


def test_token_and_position_embedding_returns_expected_shape():
    vocab_size = 20
    batch_size = 4
    context_length = 8
    embedding_dim = 32
    max_context_length = 128

    module = TokenAndPositionEmbedding(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_context_length=max_context_length,
    )
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
    )

    output = module(input_ids)

    assert output.shape == (batch_size, context_length, embedding_dim)


def test_token_and_position_embedding_rejects_non_batched_input_ids():
    module = TokenAndPositionEmbedding(
        vocab_size=20,
        embedding_dim=32,
        max_context_length=128,
    )
    input_ids = torch.tensor(
        [1, 2, 3],
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="shape"):
        module(input_ids)


def test_token_and_position_embedding_rejects_too_long_context():
    module = TokenAndPositionEmbedding(
        vocab_size=20,
        embedding_dim=32,
        max_context_length=4,
    )
    input_ids = torch.randint(
        low=0,
        high=20,
        size=(2, 5),
    )

    with pytest.raises(ValueError, match="max_context_length"):
        module(input_ids)


def test_token_and_position_embedding_preserves_input_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module = TokenAndPositionEmbedding(
        vocab_size=20,
        embedding_dim=32,
        max_context_length=128,
    ).to(device)
    input_ids = torch.randint(
        low=0,
        high=20,
        size=(4, 8),
        device=device,
    )

    output = module(input_ids)

    assert output.device == input_ids.device


def test_token_and_position_embedding_backpropagates_to_both_embedding_tables():
    module = TokenAndPositionEmbedding(
        vocab_size=20,
        embedding_dim=32,
        max_context_length=128,
    )
    input_ids = torch.randint(
        low=0,
        high=20,
        size=(4, 8),
    )

    output = module(input_ids)
    loss = output.sum()

    loss.backward()

    assert module.token_embedding.weight.grad is not None
    assert module.position_embedding.weight.grad is not None
    assert module.token_embedding.weight.grad.shape == module.token_embedding.weight.shape
    assert (
        module.position_embedding.weight.grad.shape
        == module.position_embedding.weight.shape
    )


def test_token_and_position_embedding_rejects_invalid_init_arguments():
    with pytest.raises(ValueError, match="vocab_size"):
        TokenAndPositionEmbedding(
            vocab_size=0,
            embedding_dim=32,
            max_context_length=128,
        )

    with pytest.raises(ValueError, match="embedding_dim"):
        TokenAndPositionEmbedding(
            vocab_size=20,
            embedding_dim=0,
            max_context_length=128,
        )

    with pytest.raises(ValueError, match="max_context_length"):
        TokenAndPositionEmbedding(
            vocab_size=20,
            embedding_dim=32,
            max_context_length=0,
        )
