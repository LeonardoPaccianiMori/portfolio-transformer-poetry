import pytest
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


def test_bigram_generate_appends_requested_number_of_tokens():
    vocab_size = 10
    model = BigramLanguageModel(vocab_size=vocab_size)

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


def test_bigram_generate_returns_prompt_when_no_new_tokens_requested():
    vocab_size = 10
    model = BigramLanguageModel(vocab_size=vocab_size)

    input_ids = torch.tensor(
        [[1, 2, 3]],
        dtype=torch.long,
    )

    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=0,
    )

    assert torch.equal(generated_ids, input_ids)


def test_bigram_generate_rejects_non_batched_input_ids():
    vocab_size = 10
    model = BigramLanguageModel(vocab_size=vocab_size)

    input_ids = torch.tensor(
        [1, 2, 3],
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="shape"):
        model.generate(
            input_ids=input_ids,
            max_new_tokens=5,
        )


def test_bigram_generate_rejects_negative_max_new_tokens():
    vocab_size = 10
    model = BigramLanguageModel(vocab_size=vocab_size)

    input_ids = torch.tensor(
        [[1, 2, 3]],
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="max_new_tokens"):
        model.generate(
            input_ids=input_ids,
            max_new_tokens=-1,
        )
