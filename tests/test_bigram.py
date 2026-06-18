import torch

from sonnet_model.bigram import BigramLanguageModel


def test_bigram_forward_returns_logits_without_targets():
    vocab_size = 10
    batch_size = 4
    context_length = 8

    model = BigramLanguageModel(vocab_size=vocab_size)

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
    )

    logits, loss = model(input_ids)

    assert logits.shape == (batch_size, context_length, vocab_size)
    assert loss is None


def test_bigram_forward_returns_logits_and_scalar_loss_with_targets():
    vocab_size = 10
    batch_size = 4
    context_length = 8

    model = BigramLanguageModel(vocab_size=vocab_size)

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


def test_bigram_loss_backpropagates_to_embedding_table():
    vocab_size = 10
    batch_size = 4
    context_length = 8

    model = BigramLanguageModel(vocab_size=vocab_size)

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

    _, loss = model(input_ids, target_ids)

    assert loss is not None

    loss.backward()

    gradients = model.token_embedding_table.weight.grad

    assert gradients is not None
    assert gradients.shape == model.token_embedding_table.weight.shape


def test_bigram_optimizer_step_updates_embedding_weights():
    vocab_size = 10
    batch_size = 4
    context_length = 8

    model = BigramLanguageModel(vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
    )

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

    weights_before = model.token_embedding_table.weight.detach().clone()

    _, loss = model(input_ids, target_ids)

    assert loss is not None

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    weights_after = model.token_embedding_table.weight.detach().clone()

    assert not torch.equal(weights_before, weights_after)
