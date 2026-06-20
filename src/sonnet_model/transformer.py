import torch
from torch import nn


class TokenAndPositionEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_context_length: int,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than 0")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0")

        if max_context_length <= 0:
            raise ValueError("max_context_length must be greater than 0")

        self.max_context_length = max_context_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_context_length, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch_size, context_length)")

        _, context_length = input_ids.shape

        if context_length > self.max_context_length:
            raise ValueError("context_length exceeds max_context_length")

        position_ids = torch.arange(
            context_length,
            device=input_ids.device,
        )

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)

        return token_embeddings + position_embeddings
