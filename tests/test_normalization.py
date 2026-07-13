import pytest
import torch
from torch import nn

from sonnet_model.normalization import RMSNorm, build_normalization_layer


def test_rms_norm_matches_root_mean_square_formula():
    layer = RMSNorm(embedding_dim=3, eps=1e-5)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([1.0, 2.0, 3.0]))
    x = torch.tensor([[3.0, 4.0, 0.0]])

    output = layer(x)
    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    expected = expected * layer.weight

    assert torch.allclose(output, expected)


def test_rms_norm_backpropagates_to_input_and_scale_weight():
    layer = RMSNorm(embedding_dim=4)
    x = torch.randn(2, 3, 4, requires_grad=True)

    layer(x).sum().backward()

    assert x.grad is not None
    assert layer.weight.grad is not None
    assert not hasattr(layer, "bias")


@pytest.mark.parametrize(
    ("embedding_dim", "eps", "message"),
    [
        (0, 1e-5, "embedding_dim"),
        (4, 0.0, "eps"),
    ],
)
def test_rms_norm_rejects_invalid_initialization(
    embedding_dim: int,
    eps: float,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        RMSNorm(embedding_dim=embedding_dim, eps=eps)


def test_rms_norm_rejects_wrong_final_dimension():
    layer = RMSNorm(embedding_dim=4)

    with pytest.raises(ValueError, match="final dimension"):
        layer(torch.randn(2, 3, 5))


def test_build_normalization_layer_returns_requested_module():
    assert isinstance(build_normalization_layer(4, "layer_norm"), nn.LayerNorm)
    assert isinstance(build_normalization_layer(4, "rms_norm"), RMSNorm)

    with pytest.raises(ValueError, match="normalization_type"):
        build_normalization_layer(4, "unknown")  # type: ignore[arg-type]
