import pytest
import torch

from sonnet_model.bigram import BigramLanguageModel
from sonnet_training.steps import (
    estimate_next_token_loss,
    train_next_token_model,
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


def test_train_next_token_model_returns_loss_history():
    vocab_size = 10
    train_token_ids = torch.arange(100, dtype=torch.long) % vocab_size
    validation_token_ids = torch.arange(100, dtype=torch.long) % vocab_size

    model = BigramLanguageModel(vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
    )

    history = train_next_token_model(
        model=model,
        optimizer=optimizer,
        train_token_ids=train_token_ids,
        validation_token_ids=validation_token_ids,
        batch_size=4,
        context_length=8,
        train_steps=5,
        eval_interval=2,
        eval_batches=2,
        device=torch.device("cpu"),
    )

    assert history[0]["step"] == 1
    assert history[-1]["step"] == 5
    assert all("train_loss" in row for row in history)
    assert all("validation_loss" in row for row in history)
    assert all(row["train_loss"] > 0.0 for row in history)
    assert all(row["validation_loss"] > 0.0 for row in history)


def test_train_next_token_model_updates_model_weights():
    vocab_size = 10
    train_token_ids = torch.arange(100, dtype=torch.long) % vocab_size
    validation_token_ids = torch.arange(100, dtype=torch.long) % vocab_size

    model = BigramLanguageModel(vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
    )

    weights_before = model.token_embedding_table.weight.detach().clone()

    train_next_token_model(
        model=model,
        optimizer=optimizer,
        train_token_ids=train_token_ids,
        validation_token_ids=validation_token_ids,
        batch_size=4,
        context_length=8,
        train_steps=5,
        eval_interval=2,
        eval_batches=2,
        device=torch.device("cpu"),
    )

    weights_after = model.token_embedding_table.weight.detach().clone()

    assert not torch.equal(weights_before, weights_after)


def test_train_next_token_model_rejects_invalid_train_steps():
    vocab_size = 10
    token_ids = torch.arange(100, dtype=torch.long) % vocab_size
    model = BigramLanguageModel(vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
    )

    with pytest.raises(ValueError, match="train_steps"):
        train_next_token_model(
            model=model,
            optimizer=optimizer,
            train_token_ids=token_ids,
            validation_token_ids=token_ids,
            batch_size=4,
            context_length=8,
            train_steps=0,
            eval_interval=2,
            eval_batches=2,
            device=torch.device("cpu"),
        )


def test_train_next_token_model_rejects_invalid_eval_interval():
    vocab_size = 10
    token_ids = torch.arange(100, dtype=torch.long) % vocab_size
    model = BigramLanguageModel(vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
    )

    with pytest.raises(ValueError, match="eval_interval"):
        train_next_token_model(
            model=model,
            optimizer=optimizer,
            train_token_ids=token_ids,
            validation_token_ids=token_ids,
            batch_size=4,
            context_length=8,
            train_steps=5,
            eval_interval=0,
            eval_batches=2,
            device=torch.device("cpu"),
        )
