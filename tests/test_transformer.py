import pytest
import torch

from sonnet_model.transformer import (
    CausalSelfAttentionHead,
    FeedForward,
    MultiHeadCausalSelfAttention,
    TokenAndPositionEmbedding,
)


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


def test_causal_self_attention_head_returns_expected_shape():
    batch_size = 4
    context_length = 8
    embedding_dim = 32
    head_dim = 16

    head = CausalSelfAttentionHead(
        embedding_dim=embedding_dim,
        head_dim=head_dim,
        max_context_length=128,
    )
    x = torch.randn(batch_size, context_length, embedding_dim)

    output = head(x)

    assert output.shape == (batch_size, context_length, head_dim)


def test_causal_self_attention_head_can_return_attention_weights():
    batch_size = 4
    context_length = 8
    embedding_dim = 32
    head_dim = 16

    head = CausalSelfAttentionHead(
        embedding_dim=embedding_dim,
        head_dim=head_dim,
        max_context_length=128,
    )
    x = torch.randn(batch_size, context_length, embedding_dim)

    output, attention_weights = head(
        x,
        return_attention_weights=True,
    )

    assert output.shape == (batch_size, context_length, head_dim)
    assert attention_weights.shape == (batch_size, context_length, context_length)


def test_causal_self_attention_head_rejects_non_embedding_input():
    head = CausalSelfAttentionHead(
        embedding_dim=32,
        head_dim=16,
        max_context_length=128,
    )
    x = torch.randn(4, 32)

    with pytest.raises(ValueError, match="shape"):
        head(x)


def test_causal_self_attention_head_rejects_too_long_context():
    head = CausalSelfAttentionHead(
        embedding_dim=32,
        head_dim=16,
        max_context_length=4,
    )
    x = torch.randn(2, 5, 32)

    with pytest.raises(ValueError, match="max_context_length"):
        head(x)


def test_causal_self_attention_head_masks_future_positions():
    batch_size = 2
    context_length = 5
    embedding_dim = 32
    head_dim = 16

    head = CausalSelfAttentionHead(
        embedding_dim=embedding_dim,
        head_dim=head_dim,
        max_context_length=128,
    )
    x = torch.randn(batch_size, context_length, embedding_dim)

    _, attention_weights = head(
        x,
        return_attention_weights=True,
    )

    future_positions = torch.triu(
        torch.ones(context_length, context_length, dtype=torch.bool),
        diagonal=1,
    )

    assert torch.all(attention_weights[:, future_positions] == 0.0)


def test_causal_self_attention_head_attention_rows_sum_to_one():
    batch_size = 2
    context_length = 5
    embedding_dim = 32
    head_dim = 16

    head = CausalSelfAttentionHead(
        embedding_dim=embedding_dim,
        head_dim=head_dim,
        max_context_length=128,
    )
    x = torch.randn(batch_size, context_length, embedding_dim)

    _, attention_weights = head(
        x,
        return_attention_weights=True,
    )
    row_sums = attention_weights.sum(dim=-1)

    assert torch.allclose(row_sums, torch.ones_like(row_sums))


def test_causal_self_attention_head_buffer_moves_to_input_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    head = CausalSelfAttentionHead(
        embedding_dim=32,
        head_dim=16,
        max_context_length=128,
    ).to(device)
    x = torch.randn(4, 8, 32, device=device)

    output = head(x)

    assert output.device == x.device
    assert head.causal_mask.device == x.device


def test_causal_self_attention_head_backpropagates_to_qkv_projections():
    head = CausalSelfAttentionHead(
        embedding_dim=32,
        head_dim=16,
        max_context_length=128,
    )
    x = torch.randn(4, 8, 32)

    output = head(x)
    loss = output.sum()

    loss.backward()

    assert head.query.weight.grad is not None
    assert head.key.weight.grad is not None
    assert head.value.weight.grad is not None
    assert head.query.weight.grad.shape == head.query.weight.shape
    assert head.key.weight.grad.shape == head.key.weight.shape
    assert head.value.weight.grad.shape == head.value.weight.shape


def test_causal_self_attention_head_rejects_invalid_init_arguments():
    with pytest.raises(ValueError, match="embedding_dim"):
        CausalSelfAttentionHead(
            embedding_dim=0,
            head_dim=16,
            max_context_length=128,
        )

    with pytest.raises(ValueError, match="head_dim"):
        CausalSelfAttentionHead(
            embedding_dim=32,
            head_dim=0,
            max_context_length=128,
        )

    with pytest.raises(ValueError, match="max_context_length"):
        CausalSelfAttentionHead(
            embedding_dim=32,
            head_dim=16,
            max_context_length=0,
        )


def test_multi_head_causal_self_attention_returns_expected_shape():
    batch_size = 4
    context_length = 8
    embedding_dim = 32
    num_heads = 2
    head_dim = 16

    attention = MultiHeadCausalSelfAttention(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        max_context_length=128,
    )
    x = torch.randn(batch_size, context_length, embedding_dim)

    output = attention(x)

    assert output.shape == (batch_size, context_length, embedding_dim)


def test_multi_head_causal_self_attention_can_return_attention_weights():
    batch_size = 4
    context_length = 8
    embedding_dim = 32
    num_heads = 2
    head_dim = 16

    attention = MultiHeadCausalSelfAttention(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        max_context_length=128,
    )
    x = torch.randn(batch_size, context_length, embedding_dim)

    output, attention_weights = attention(
        x,
        return_attention_weights=True,
    )

    assert output.shape == (batch_size, context_length, embedding_dim)
    assert attention_weights.shape == (
        batch_size,
        num_heads,
        context_length,
        context_length,
    )


def test_multi_head_causal_self_attention_rejects_non_embedding_input():
    attention = MultiHeadCausalSelfAttention(
        embedding_dim=32,
        num_heads=2,
        head_dim=16,
        max_context_length=128,
    )
    x = torch.randn(4, 32)

    with pytest.raises(ValueError, match="shape"):
        attention(x)


def test_multi_head_causal_self_attention_rejects_too_long_context():
    attention = MultiHeadCausalSelfAttention(
        embedding_dim=32,
        num_heads=2,
        head_dim=16,
        max_context_length=4,
    )
    x = torch.randn(2, 5, 32)

    with pytest.raises(ValueError, match="max_context_length"):
        attention(x)


def test_multi_head_causal_self_attention_masks_future_positions_in_every_head():
    batch_size = 2
    context_length = 5
    embedding_dim = 32
    num_heads = 2
    head_dim = 16

    attention = MultiHeadCausalSelfAttention(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        max_context_length=128,
    )
    x = torch.randn(batch_size, context_length, embedding_dim)

    _, attention_weights = attention(
        x,
        return_attention_weights=True,
    )
    future_positions = torch.triu(
        torch.ones(context_length, context_length, dtype=torch.bool),
        diagonal=1,
    )

    assert torch.all(attention_weights[:, :, future_positions] == 0.0)


def test_multi_head_causal_self_attention_attention_rows_sum_to_one():
    batch_size = 2
    context_length = 5
    embedding_dim = 32
    num_heads = 2
    head_dim = 16

    attention = MultiHeadCausalSelfAttention(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        max_context_length=128,
    )
    x = torch.randn(batch_size, context_length, embedding_dim)

    _, attention_weights = attention(
        x,
        return_attention_weights=True,
    )
    row_sums = attention_weights.sum(dim=-1)

    assert torch.allclose(row_sums, torch.ones_like(row_sums))


def test_multi_head_causal_self_attention_backpropagates_to_all_heads_and_projection():
    attention = MultiHeadCausalSelfAttention(
        embedding_dim=32,
        num_heads=2,
        head_dim=16,
        max_context_length=128,
    )
    x = torch.randn(4, 8, 32)

    output = attention(x)
    loss = output.sum()

    loss.backward()

    for head in attention.heads:
        assert head.query.weight.grad is not None
        assert head.key.weight.grad is not None
        assert head.value.weight.grad is not None

    assert attention.output_projection.weight.grad is not None
    assert attention.output_projection.bias.grad is not None


def test_multi_head_causal_self_attention_rejects_invalid_init_arguments():
    with pytest.raises(ValueError, match="embedding_dim"):
        MultiHeadCausalSelfAttention(
            embedding_dim=0,
            num_heads=2,
            head_dim=16,
            max_context_length=128,
        )

    with pytest.raises(ValueError, match="num_heads"):
        MultiHeadCausalSelfAttention(
            embedding_dim=32,
            num_heads=0,
            head_dim=16,
            max_context_length=128,
        )

    with pytest.raises(ValueError, match="head_dim"):
        MultiHeadCausalSelfAttention(
            embedding_dim=32,
            num_heads=2,
            head_dim=0,
            max_context_length=128,
        )

    with pytest.raises(ValueError, match="max_context_length"):
        MultiHeadCausalSelfAttention(
            embedding_dim=32,
            num_heads=2,
            head_dim=16,
            max_context_length=0,
        )


def test_feed_forward_returns_expected_shape():
    batch_size = 4
    context_length = 8
    embedding_dim = 32
    feed_forward_dim = 128

    feed_forward = FeedForward(
        embedding_dim=embedding_dim,
        feed_forward_dim=feed_forward_dim,
    )
    x = torch.randn(batch_size, context_length, embedding_dim)

    output = feed_forward(x)

    assert output.shape == (batch_size, context_length, embedding_dim)


def test_feed_forward_rejects_non_embedding_input():
    feed_forward = FeedForward(
        embedding_dim=32,
        feed_forward_dim=128,
    )
    x = torch.randn(4, 32)

    with pytest.raises(ValueError, match="shape"):
        feed_forward(x)


def test_feed_forward_preserves_input_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feed_forward = FeedForward(
        embedding_dim=32,
        feed_forward_dim=128,
    ).to(device)
    x = torch.randn(4, 8, 32, device=device)

    output = feed_forward(x)

    assert output.device == x.device


def test_feed_forward_backpropagates_to_both_linear_layers():
    feed_forward = FeedForward(
        embedding_dim=32,
        feed_forward_dim=128,
    )
    x = torch.randn(4, 8, 32)

    output = feed_forward(x)
    loss = output.sum()

    loss.backward()

    first_linear = feed_forward.network[0]
    second_linear = feed_forward.network[2]

    assert first_linear.weight.grad is not None
    assert first_linear.bias.grad is not None
    assert second_linear.weight.grad is not None
    assert second_linear.bias.grad is not None


def test_feed_forward_rejects_invalid_init_arguments():
    with pytest.raises(ValueError, match="embedding_dim"):
        FeedForward(
            embedding_dim=0,
            feed_forward_dim=128,
        )

    with pytest.raises(ValueError, match="feed_forward_dim"):
        FeedForward(
            embedding_dim=32,
            feed_forward_dim=0,
        )
