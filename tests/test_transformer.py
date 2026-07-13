import pytest
import torch

from sonnet_model.normalization import RMSNorm
from sonnet_model.transformer import (
    CausalSelfAttentionHead,
    CausalTransformerLanguageModel,
    FeedForward,
    MultiHeadCausalSelfAttention,
    TokenAndPositionEmbedding,
    TransformerBlock,
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


def test_transformer_block_returns_expected_shape():
    batch_size = 4
    context_length = 8
    embedding_dim = 32

    block = TransformerBlock(
        embedding_dim=embedding_dim,
        num_heads=2,
        head_dim=16,
        feed_forward_dim=128,
        max_context_length=128,
    )
    x = torch.randn(batch_size, context_length, embedding_dim)

    output = block(x)

    assert output.shape == (batch_size, context_length, embedding_dim)


def test_transformer_block_rejects_non_embedding_input():
    block = TransformerBlock(
        embedding_dim=32,
        num_heads=2,
        head_dim=16,
        feed_forward_dim=128,
        max_context_length=128,
    )
    x = torch.randn(4, 32)

    with pytest.raises(ValueError, match="shape"):
        block(x)


def test_transformer_block_rejects_too_long_context():
    block = TransformerBlock(
        embedding_dim=32,
        num_heads=2,
        head_dim=16,
        feed_forward_dim=128,
        max_context_length=4,
    )
    x = torch.randn(2, 5, 32)

    with pytest.raises(ValueError, match="max_context_length"):
        block(x)


def test_transformer_block_output_is_not_trivially_identical_to_input():
    block = TransformerBlock(
        embedding_dim=32,
        num_heads=2,
        head_dim=16,
        feed_forward_dim=128,
        max_context_length=128,
    )
    x = torch.randn(4, 8, 32)

    output = block(x)

    assert not torch.equal(output, x)


def test_transformer_block_preserves_input_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    block = TransformerBlock(
        embedding_dim=32,
        num_heads=2,
        head_dim=16,
        feed_forward_dim=128,
        max_context_length=128,
    ).to(device)
    x = torch.randn(4, 8, 32, device=device)

    output = block(x)

    assert output.device == x.device


def test_transformer_block_backpropagates_to_attention_feed_forward_and_layer_norms():
    block = TransformerBlock(
        embedding_dim=32,
        num_heads=2,
        head_dim=16,
        feed_forward_dim=128,
        max_context_length=128,
    )
    x = torch.randn(4, 8, 32)

    output = block(x)
    loss = output.sum()

    loss.backward()

    assert block.attention_layer_norm.weight.grad is not None
    assert block.attention_layer_norm.bias.grad is not None
    assert block.feed_forward_layer_norm.weight.grad is not None
    assert block.feed_forward_layer_norm.bias.grad is not None
    assert block.attention.output_projection.weight.grad is not None
    assert block.feed_forward.network[0].weight.grad is not None
    assert block.feed_forward.network[2].weight.grad is not None


def test_transformer_block_uses_rms_norm_when_requested():
    block = TransformerBlock(
        embedding_dim=32,
        num_heads=2,
        head_dim=16,
        feed_forward_dim=128,
        max_context_length=128,
        normalization_type="rms_norm",
    )
    x = torch.randn(4, 8, 32)

    output = block(x)
    output.sum().backward()

    assert isinstance(block.attention_layer_norm, RMSNorm)
    assert isinstance(block.feed_forward_layer_norm, RMSNorm)
    assert block.attention_layer_norm.weight.grad is not None
    assert block.feed_forward_layer_norm.weight.grad is not None
    assert not hasattr(block.attention_layer_norm, "bias")


def test_transformer_block_rejects_invalid_init_arguments():
    with pytest.raises(ValueError, match="embedding_dim"):
        TransformerBlock(
            embedding_dim=0,
            num_heads=2,
            head_dim=16,
            feed_forward_dim=128,
            max_context_length=128,
        )

    with pytest.raises(ValueError, match="num_heads"):
        TransformerBlock(
            embedding_dim=32,
            num_heads=0,
            head_dim=16,
            feed_forward_dim=128,
            max_context_length=128,
        )

    with pytest.raises(ValueError, match="head_dim"):
        TransformerBlock(
            embedding_dim=32,
            num_heads=2,
            head_dim=0,
            feed_forward_dim=128,
            max_context_length=128,
        )

    with pytest.raises(ValueError, match="feed_forward_dim"):
        TransformerBlock(
            embedding_dim=32,
            num_heads=2,
            head_dim=16,
            feed_forward_dim=0,
            max_context_length=128,
        )

    with pytest.raises(ValueError, match="max_context_length"):
        TransformerBlock(
            embedding_dim=32,
            num_heads=2,
            head_dim=16,
            feed_forward_dim=128,
            max_context_length=0,
        )


def build_test_transformer_model() -> CausalTransformerLanguageModel:
    return CausalTransformerLanguageModel(
        vocab_size=20,
        embedding_dim=32,
        num_layers=2,
        num_heads=2,
        head_dim=16,
        feed_forward_dim=128,
        max_context_length=128,
    )


def test_causal_transformer_language_model_returns_logits_without_targets():
    vocab_size = 20
    batch_size = 4
    context_length = 8

    model = build_test_transformer_model()
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
    )

    logits, loss = model(input_ids)

    assert logits.shape == (batch_size, context_length, vocab_size)
    assert loss is None


def test_causal_transformer_language_model_returns_logits_and_scalar_loss_with_targets():
    vocab_size = 20
    batch_size = 4
    context_length = 8

    model = build_test_transformer_model()
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
    )
    target_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
    )

    logits, loss = model(input_ids, target_ids)

    assert logits.shape == (batch_size, context_length, vocab_size)
    assert loss is not None
    assert loss.ndim == 0


def test_causal_transformer_language_model_rejects_non_batched_input_ids():
    model = build_test_transformer_model()
    input_ids = torch.tensor(
        [1, 2, 3],
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="shape"):
        model(input_ids)


def test_causal_transformer_language_model_rejects_mismatched_target_shape():
    vocab_size = 20
    model = build_test_transformer_model()
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(4, 8),
    )
    target_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(4, 7),
    )

    with pytest.raises(ValueError, match="same shape"):
        model(input_ids, target_ids)


def test_causal_transformer_language_model_rejects_too_long_context():
    vocab_size = 20
    model = CausalTransformerLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=32,
        num_layers=2,
        num_heads=2,
        head_dim=16,
        feed_forward_dim=128,
        max_context_length=4,
    )
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(2, 5),
    )

    with pytest.raises(ValueError, match="max_context_length"):
        model(input_ids)


def test_causal_transformer_language_model_preserves_input_device():
    vocab_size = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_test_transformer_model().to(device)
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(4, 8),
        device=device,
    )

    logits, _ = model(input_ids)

    assert logits.device == input_ids.device


def test_causal_transformer_language_model_backpropagates_through_full_model():
    vocab_size = 20
    model = build_test_transformer_model()
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(4, 8),
    )
    target_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(4, 8),
    )

    _, loss = model(input_ids, target_ids)

    assert loss is not None

    loss.backward()

    assert model.embedding.token_embedding.weight.grad is not None
    assert model.embedding.position_embedding.weight.grad is not None
    assert model.blocks[0].attention.output_projection.weight.grad is not None
    assert model.blocks[0].feed_forward.network[0].weight.grad is not None
    assert model.final_layer_norm.weight.grad is not None
    assert model.output_projection.weight.grad is not None


def test_causal_transformer_language_model_uses_rms_norm_when_requested():
    model = CausalTransformerLanguageModel(
        vocab_size=20,
        embedding_dim=32,
        num_layers=2,
        num_heads=2,
        head_dim=16,
        feed_forward_dim=128,
        max_context_length=128,
        normalization_type="rms_norm",
    )
    input_ids = torch.randint(0, 20, (2, 8))
    target_ids = torch.randint(0, 20, (2, 8))

    _, loss = model(input_ids, target_ids)

    assert isinstance(model.final_layer_norm, RMSNorm)
    assert model.normalization_type == "rms_norm"
    assert loss is not None


def test_causal_transformer_language_model_optimizer_step_updates_weights():
    vocab_size = 20
    model = build_test_transformer_model()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(4, 8),
    )
    target_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(4, 8),
    )

    weights_before = model.output_projection.weight.detach().clone()

    _, loss = model(input_ids, target_ids)

    assert loss is not None

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    weights_after = model.output_projection.weight.detach().clone()

    assert not torch.equal(weights_before, weights_after)


def test_causal_transformer_language_model_rejects_invalid_init_arguments():
    valid_arguments = {
        "vocab_size": 20,
        "embedding_dim": 32,
        "num_layers": 2,
        "num_heads": 2,
        "head_dim": 16,
        "feed_forward_dim": 128,
        "max_context_length": 128,
    }

    for argument_name in valid_arguments:
        invalid_arguments = {
            **valid_arguments,
            argument_name: 0,
        }

        with pytest.raises(ValueError, match=argument_name):
            CausalTransformerLanguageModel(**invalid_arguments)


def test_causal_transformer_language_model_generate_appends_requested_tokens():
    vocab_size = 20
    model = build_test_transformer_model()
    input_ids = torch.tensor(
        [[1, 2, 3]],
        dtype=torch.long,
    )
    generator = torch.Generator().manual_seed(0)

    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        generator=generator,
    )

    assert generated_ids.shape == (1, 8)
    assert torch.equal(generated_ids[:, :3], input_ids)
    assert generated_ids.min().item() >= 0
    assert generated_ids.max().item() < vocab_size


def test_causal_transformer_language_model_generate_returns_prompt_when_no_tokens_requested():
    model = build_test_transformer_model()
    input_ids = torch.tensor(
        [[1, 2, 3]],
        dtype=torch.long,
    )

    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=0,
    )

    assert torch.equal(generated_ids, input_ids)


def test_causal_transformer_language_model_generate_rejects_non_batched_input_ids():
    model = build_test_transformer_model()
    input_ids = torch.tensor(
        [1, 2, 3],
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="shape"):
        model.generate(
            input_ids=input_ids,
            max_new_tokens=5,
        )


def test_causal_transformer_language_model_generate_rejects_negative_max_new_tokens():
    model = build_test_transformer_model()
    input_ids = torch.tensor(
        [[1, 2, 3]],
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="max_new_tokens"):
        model.generate(
            input_ids=input_ids,
            max_new_tokens=-1,
        )


def test_causal_transformer_language_model_generate_crops_long_context():
    vocab_size = 20
    model = CausalTransformerLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=32,
        num_layers=2,
        num_heads=2,
        head_dim=16,
        feed_forward_dim=128,
        max_context_length=4,
    )
    input_ids = torch.tensor(
        [[1, 2, 3, 4]],
        dtype=torch.long,
    )
    generator = torch.Generator().manual_seed(0)

    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        generator=generator,
    )

    assert generated_ids.shape == (1, 9)
    assert torch.equal(generated_ids[:, :4], input_ids)
    assert generated_ids.min().item() >= 0
    assert generated_ids.max().item() < vocab_size


def test_causal_transformer_language_model_generate_preserves_input_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_test_transformer_model().to(device)
    input_ids = torch.tensor(
        [[1, 2, 3]],
        dtype=torch.long,
        device=device,
    )

    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
    )

    assert generated_ids.device == input_ids.device


def build_biased_generation_model(
    favored_token_id: int,
) -> CausalTransformerLanguageModel:
    model = build_test_transformer_model()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

        model.output_projection.bias[favored_token_id] = 10.0

    return model


def test_causal_transformer_language_model_generate_supports_top_k_sampling():
    model = build_biased_generation_model(favored_token_id=2)
    input_ids = torch.tensor(
        [[1, 1]],
        dtype=torch.long,
    )

    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=3,
        top_k=1,
    )

    assert generated_ids.tolist() == [[1, 1, 2, 2, 2]]


def test_causal_transformer_language_model_generate_stops_on_stop_token_id():
    model = build_biased_generation_model(favored_token_id=2)
    input_ids = torch.tensor(
        [[1, 1]],
        dtype=torch.long,
    )

    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        top_k=1,
        stop_token_id=2,
    )

    assert generated_ids.tolist() == [[1, 1, 2]]


def test_causal_transformer_language_model_generate_forbids_token_ids():
    model = build_biased_generation_model(favored_token_id=2)
    input_ids = torch.tensor(
        [[1, 1]],
        dtype=torch.long,
    )

    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=3,
        top_k=1,
        forbidden_token_ids={2},
    )

    assert 2 not in generated_ids[:, 2:].tolist()[0]


def test_causal_transformer_language_model_generate_rejects_invalid_forbidden_ids():
    model = build_test_transformer_model()
    input_ids = torch.tensor(
        [[1, 2, 3]],
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="forbidden_token_ids"):
        model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            forbidden_token_ids={20},
        )

    with pytest.raises(ValueError, match="forbidden_token_ids"):
        model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            forbidden_token_ids=set(range(20)),
        )


def test_causal_transformer_language_model_generate_rejects_invalid_temperature():
    model = build_test_transformer_model()
    input_ids = torch.tensor(
        [[1, 2, 3]],
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="temperature"):
        model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            temperature=0.0,
        )


def test_causal_transformer_language_model_generate_rejects_invalid_top_k():
    model = build_test_transformer_model()
    input_ids = torch.tensor(
        [[1, 2, 3]],
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="top_k"):
        model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            top_k=0,
        )

    with pytest.raises(ValueError, match="vocab_size"):
        model.generate(
            input_ids=input_ids,
            max_new_tokens=1,
            top_k=21,
        )
