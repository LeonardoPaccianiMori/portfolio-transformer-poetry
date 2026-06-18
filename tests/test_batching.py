import pytest
import torch

from sonnet_corpus.batching import sample_next_token_batch


def test_sample_next_token_batch_returns_shifted_batches():
    token_ids = torch.arange(20, dtype=torch.long)

    torch.manual_seed(42)

    x_batch, y_batch = sample_next_token_batch(
        token_ids=token_ids,
        batch_size=4,
        context_length=5,
    )

    assert x_batch.shape == (4, 5)
    assert y_batch.shape == (4, 5)

    assert x_batch.dtype == torch.long
    assert y_batch.dtype == torch.long

    assert torch.equal(x_batch[:, 1:], y_batch[:, :-1])


def test_sample_next_token_batch_moves_batches_to_requested_device():
    token_ids = torch.arange(20, dtype=torch.long)
    device = torch.device("cpu")

    x_batch, y_batch = sample_next_token_batch(
        token_ids=token_ids,
        batch_size=4,
        context_length=5,
        device=device,
    )

    assert x_batch.device == device
    assert y_batch.device == device
    assert token_ids.device.type == "cpu"


def test_sample_next_token_batch_rejects_non_1d_token_ids():
    token_ids = torch.arange(20, dtype=torch.long).reshape(4, 5)

    with pytest.raises(ValueError, match="1D tensor"):
        sample_next_token_batch(
            token_ids=token_ids,
            batch_size=4,
            context_length=5,
        )


def test_sample_next_token_batch_rejects_non_long_token_ids():
    token_ids = torch.arange(20, dtype=torch.float32)

    with pytest.raises(ValueError, match="dtype torch.long"):
        sample_next_token_batch(
            token_ids=token_ids,
            batch_size=4,
            context_length=5,
        )


def test_sample_next_token_batch_rejects_invalid_batch_size():
    token_ids = torch.arange(20, dtype=torch.long)

    with pytest.raises(ValueError, match="batch_size"):
        sample_next_token_batch(
            token_ids=token_ids,
            batch_size=0,
            context_length=5,
        )


def test_sample_next_token_batch_rejects_invalid_context_length():
    token_ids = torch.arange(20, dtype=torch.long)

    with pytest.raises(ValueError, match="context_length"):
        sample_next_token_batch(
            token_ids=token_ids,
            batch_size=4,
            context_length=0,
        )


def test_sample_next_token_batch_rejects_too_short_token_stream():
    token_ids = torch.arange(5, dtype=torch.long)

    with pytest.raises(ValueError, match="longer than context_length"):
        sample_next_token_batch(
            token_ids=token_ids,
            batch_size=4,
            context_length=5,
        )
