import numpy as np
import pytest

from sonnet_training.form_targeted_lora import (
    batch_starts,
    cosine_learning_rate,
    token_tensor,
    window_count,
)


def test_window_count():
    assert window_count(100, 50) == 1
    assert window_count(101, 50) == 2
    assert window_count(50, 50) == 0


def test_cosine_learning_rate_warms_up_and_decays():
    base = 1e-4
    assert cosine_learning_rate(
        0, total_steps=100, warmup_steps=10, base_learning_rate=base
    ) == pytest.approx(base / 10)
    assert cosine_learning_rate(
        9, total_steps=100, warmup_steps=10, base_learning_rate=base
    ) == pytest.approx(base)
    assert cosine_learning_rate(
        99, total_steps=100, warmup_steps=10, base_learning_rate=base
    ) < base * 0.01


def test_batch_starts_partition_all_windows():
    batches = batch_starts(10, 4, seed=1, epoch=0)
    flattened = [index for batch in batches for index in batch]
    assert sorted(flattened) == list(range(10))
    assert all(len(batch) <= 4 for batch in batches)


def test_batch_starts_are_deterministic_per_seed_and_epoch():
    first = batch_starts(20, 5, seed=7, epoch=0)
    second = batch_starts(20, 5, seed=7, epoch=0)
    third = batch_starts(20, 5, seed=7, epoch=1)
    assert first == second
    assert first != third


def test_token_tensor_shape_and_bounds():
    tokens = np.arange(20, dtype=np.int32)
    window = token_tensor(tokens, 3, 5)
    assert window.tolist() == [3, 4, 5, 6, 7, 8]
    with pytest.raises(ValueError, match="outside"):
        token_tensor(tokens, 15, 5)
