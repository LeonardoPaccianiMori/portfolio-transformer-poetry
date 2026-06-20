import pytest
import torch

from sonnet_model.bigram import BigramLanguageModel
from sonnet_training.steps import (
    estimate_next_token_loss,
    train_next_token_step,
)

def test_train_next_token_step_returns_float_loss():
    vocab_size = 10
    token_ids = torch.arange(100, dtype=torch.long) % vocab_size
    model = BigramLanguageModel(vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
    )

    loss = train_next_token_step(
        model=model,
        optimizer=optimizer,
        token_ids=token_ids,
        batch_size=4,
        context_length=8,
        device=torch.device("cpu"),
    )

    assert isinstance(loss, float)
    assert loss > 0.0


def test_train_next_token_step_updates_model_weights():
    vocab_size = 10
    token_ids = torch.arange(100, dtype=torch.long) % vocab_size
    model = BigramLanguageModel(vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
    )

    weights_before = model.token_embedding_table.weight.detach().clone()

    train_next_token_step(
        model=model,
        optimizer=optimizer,
        token_ids=token_ids,
        batch_size=4,
        context_length=8,
        device=torch.device("cpu"),
    )

    weights_after = model.token_embedding_table.weight.detach().clone()

    assert not torch.equal(weights_before, weights_after)


def test_train_next_token_step_sets_model_to_train_mode():
    vocab_size = 10
    token_ids = torch.arange(100, dtype=torch.long) % vocab_size
    model = BigramLanguageModel(vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
    )

    model.eval()

    train_next_token_step(
        model=model,
        optimizer=optimizer,
        token_ids=token_ids,
        batch_size=4,
        context_length=8,
        device=torch.device("cpu"),
    )

    assert model.training


def test_estimate_next_token_loss_returns_float_loss():
    vocab_size = 10
    token_ids = torch.arange(100, dtype=torch.long) % vocab_size
    model = BigramLanguageModel(vocab_size=vocab_size)

    loss = estimate_next_token_loss(
        model=model,
        token_ids=token_ids,
        batch_size=4,
        context_length=8,
        eval_batches=3,
        device=torch.device("cpu"),
    )

    assert isinstance(loss, float)
    assert loss > 0.0


def test_estimate_next_token_loss_does_not_update_weights():
    vocab_size = 10
    token_ids = torch.arange(100, dtype=torch.long) % vocab_size
    model = BigramLanguageModel(vocab_size=vocab_size)

    weights_before = model.token_embedding_table.weight.detach().clone()

    estimate_next_token_loss(
        model=model,
        token_ids=token_ids,
        batch_size=4,
        context_length=8,
        eval_batches=3,
        device=torch.device("cpu"),
    )

    weights_after = model.token_embedding_table.weight.detach().clone()

    assert torch.equal(weights_before, weights_after)


def test_estimate_next_token_loss_rejects_invalid_eval_batches():
    vocab_size = 10
    token_ids = torch.arange(100, dtype=torch.long) % vocab_size
    model = BigramLanguageModel(vocab_size=vocab_size)

    with pytest.raises(ValueError, match="eval_batches"):
        estimate_next_token_loss(
            model=model,
            token_ids=token_ids,
            batch_size=4,
            context_length=8,
            eval_batches=0,
            device=torch.device("cpu"),
        )
