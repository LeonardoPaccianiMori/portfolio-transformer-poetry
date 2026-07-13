import pytest
import torch

from sonnet_model.positional_encoding import RotaryPositionEmbedding


def test_rope_preserves_shape_and_vector_norms():
    rope = RotaryPositionEmbedding(
        head_dim=8,
        max_context_length=16,
    )
    x = torch.randn(2, 6, 8)

    rotated = rope(x)

    assert rotated.shape == x.shape
    assert torch.allclose(
        rotated.norm(dim=-1),
        x.norm(dim=-1),
        atol=1e-6,
    )


def test_rope_leaves_position_zero_unchanged_and_changes_later_positions():
    rope = RotaryPositionEmbedding(
        head_dim=4,
        max_context_length=8,
    )
    repeated_vector = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]).repeat(1, 3, 1)

    rotated = rope(repeated_vector)

    assert torch.allclose(rotated[:, 0], repeated_vector[:, 0])
    assert not torch.allclose(rotated[:, 1], repeated_vector[:, 1])


def test_rope_backpropagates_to_its_input():
    rope = RotaryPositionEmbedding(
        head_dim=4,
        max_context_length=8,
    )
    x = torch.randn(2, 3, 4, requires_grad=True)

    rope(x).sum().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


@pytest.mark.parametrize(
    ("head_dim", "max_context_length", "theta", "message"),
    [
        (0, 8, 10_000.0, "head_dim"),
        (3, 8, 10_000.0, "even"),
        (4, 0, 10_000.0, "max_context_length"),
        (4, 8, 0.0, "theta"),
    ],
)
def test_rope_rejects_invalid_initialization(
    head_dim: int,
    max_context_length: int,
    theta: float,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        RotaryPositionEmbedding(
            head_dim=head_dim,
            max_context_length=max_context_length,
            theta=theta,
        )


def test_rope_rejects_invalid_input_shape_and_context_length():
    rope = RotaryPositionEmbedding(head_dim=4, max_context_length=2)

    with pytest.raises(ValueError, match="shape"):
        rope(torch.randn(2, 4))
    with pytest.raises(ValueError, match="head_dim"):
        rope(torch.randn(2, 2, 6))
    with pytest.raises(ValueError, match="max_context_length"):
        rope(torch.randn(2, 3, 4))
