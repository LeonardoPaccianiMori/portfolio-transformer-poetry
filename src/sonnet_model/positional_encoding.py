"""Position-encoding components for the causal transformer."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


PositionEncodingType = Literal["learned_absolute", "rope"]


def validate_position_encoding_type(position_encoding_type: str) -> None:
    """Reject unsupported position-encoding names consistently."""
    if position_encoding_type not in {"learned_absolute", "rope"}:
        raise ValueError(
            "position_encoding_type must be 'learned_absolute' or 'rope'"
        )


class RotaryPositionEmbedding(nn.Module):
    """Apply RoPE rotations to query or key vectors with shape (B, T, H)."""

    def __init__(
        self,
        head_dim: int,
        max_context_length: int,
        theta: float = 10_000.0,
    ):
        super().__init__()

        if head_dim <= 0:
            raise ValueError("head_dim must be greater than 0")
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if max_context_length <= 0:
            raise ValueError("max_context_length must be greater than 0")
        if theta <= 0:
            raise ValueError("theta must be greater than 0")

        self.head_dim = head_dim
        self.max_context_length = max_context_length
        self.theta = theta
        inverse_frequencies = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_context_length, dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequencies)

        # These deterministic tables are recreated from configuration, not trained.
        self.register_buffer("cosine", angles.cos(), persistent=False)
        self.register_buffer("sine", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch_size, context_length, head_dim)")
        if x.shape[-1] != self.head_dim:
            raise ValueError("x must have head_dim as its final dimension")

        _, context_length, _ = x.shape
        if context_length > self.max_context_length:
            raise ValueError("context_length exceeds max_context_length")

        cosine = self.cosine[:context_length].to(dtype=x.dtype).unsqueeze(0)
        sine = self.sine[:context_length].to(dtype=x.dtype).unsqueeze(0)
        even_coordinates = x[..., 0::2]
        odd_coordinates = x[..., 1::2]
        rotated_pairs = torch.stack(
            (
                even_coordinates * cosine - odd_coordinates * sine,
                even_coordinates * sine + odd_coordinates * cosine,
            ),
            dim=-1,
        )
        return rotated_pairs.flatten(start_dim=-2)
