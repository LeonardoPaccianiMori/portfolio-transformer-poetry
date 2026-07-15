import pytest
import torch

from sonnet_model.bigram import BigramLanguageModel
from sonnet_training.steps import (
    estimate_next_token_loss,
    estimate_next_token_loss_on_sequential_windows,
    sequential_next_token_window_count,
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


def test_train_next_token_step_clips_gradients_and_reports_pre_clipping_norm():
    torch.manual_seed(0)
    vocab_size = 10
    token_ids = torch.arange(100, dtype=torch.long) % vocab_size
    model = BigramLanguageModel(vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    loss, pre_clipping_gradient_norm = train_next_token_step(
        model=model,
        optimizer=optimizer,
        token_ids=token_ids,
        batch_size=4,
        context_length=8,
        device=torch.device("cpu"),
        max_gradient_norm=0.01,
        return_gradient_norm=True,
    )
    post_clipping_gradient_norm = torch.linalg.vector_norm(
        torch.stack([
            parameter.grad.detach().norm()
            for parameter in model.parameters()
            if parameter.grad is not None
        ])
    )

    assert loss > 0.0
    assert pre_clipping_gradient_norm is not None
    assert pre_clipping_gradient_norm > 0.01
    assert post_clipping_gradient_norm <= 0.01 + 1e-6


def test_train_next_token_step_rejects_nonpositive_gradient_norm_limit():
    model = BigramLanguageModel(vocab_size=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    token_ids = torch.arange(100, dtype=torch.long) % 10

    with pytest.raises(ValueError, match="max_gradient_norm"):
        train_next_token_step(
            model=model,
            optimizer=optimizer,
            token_ids=token_ids,
            batch_size=4,
            context_length=8,
            device=torch.device("cpu"),
            max_gradient_norm=0.0,
        )


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


def test_sequential_validation_uses_all_complete_non_overlapping_windows():
    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_batches: list[torch.Tensor] = []

        def forward(self, input_ids, target_ids):
            self.input_batches.append(input_ids.cpu())
            logits = torch.zeros(
                (*input_ids.shape, 20),
                device=input_ids.device,
            )
            return logits, torch.tensor(2.0, device=input_ids.device)

    token_ids = torch.arange(17, dtype=torch.long)
    model = RecordingModel()

    loss = estimate_next_token_loss_on_sequential_windows(
        model=model,
        token_ids=token_ids,
        batch_size=2,
        context_length=4,
        device=torch.device("cpu"),
    )

    assert sequential_next_token_window_count(token_ids, context_length=4) == 4
    assert loss == 2.0
    assert torch.equal(
        model.input_batches[0],
        torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
    )
    assert torch.equal(
        model.input_batches[1],
        torch.tensor([[8, 9, 10, 11], [12, 13, 14, 15]]),
    )


def test_sequential_validation_is_deterministic_when_rng_state_changes():
    token_ids = torch.arange(49, dtype=torch.long) % 10
    model = BigramLanguageModel(vocab_size=10)

    first_loss = estimate_next_token_loss_on_sequential_windows(
        model=model,
        token_ids=token_ids,
        batch_size=2,
        context_length=8,
        device=torch.device("cpu"),
    )
    torch.rand(100)
    second_loss = estimate_next_token_loss_on_sequential_windows(
        model=model,
        token_ids=token_ids,
        batch_size=2,
        context_length=8,
        device=torch.device("cpu"),
    )

    assert first_loss == second_loss


def test_sequential_validation_rejects_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        estimate_next_token_loss_on_sequential_windows(
            model=BigramLanguageModel(vocab_size=10),
            token_ids=torch.arange(20, dtype=torch.long) % 10,
            batch_size=0,
            context_length=8,
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
