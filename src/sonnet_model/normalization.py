"""Normalization layers used by the causal transformer."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


NormalizationType = Literal["layer_norm", "rms_norm"]


class RMSNorm(nn.Module):
    """Scale vectors by their root-mean-square magnitude."""

    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0")
        if eps <= 0:
            raise ValueError("eps must be greater than 0")

        self.embedding_dim = embedding_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0 or x.shape[-1] != self.embedding_dim:
            raise ValueError(
                "x must have embedding_dim as its final dimension"
            )

        inverse_root_mean_square = torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return x * inverse_root_mean_square * self.weight


def build_normalization_layer(
    embedding_dim: int,
    normalization_type: NormalizationType,
    eps: float = 1e-5,
) -> nn.Module:
    """Construct the requested normalization layer with a shared epsilon."""
    if normalization_type == "layer_norm":
        return nn.LayerNorm(embedding_dim, eps=eps)
    if normalization_type == "rms_norm":
        return RMSNorm(embedding_dim=embedding_dim, eps=eps)
    raise ValueError("normalization_type must be 'layer_norm' or 'rms_norm'")
