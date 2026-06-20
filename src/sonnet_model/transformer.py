import torch
import torch.nn.functional as F
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


class CausalSelfAttentionHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        head_dim: int,
        max_context_length: int,
    ):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0")

        if head_dim <= 0:
            raise ValueError("head_dim must be greater than 0")

        if max_context_length <= 0:
            raise ValueError("max_context_length must be greater than 0")

        self.max_context_length = max_context_length
        self.head_dim = head_dim
        self.query = nn.Linear(embedding_dim, head_dim, bias=False)
        self.key = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, head_dim, bias=False)

        causal_mask = torch.tril(torch.ones(max_context_length, max_context_length))
        self.register_buffer("causal_mask", causal_mask)

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(
                "x must have shape (batch_size, context_length, embedding_dim)"
            )

        _, context_length, _ = x.shape

        if context_length > self.max_context_length:
            raise ValueError("context_length exceeds max_context_length")

        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores * (self.head_dim ** -0.5)

        mask = self.causal_mask[:context_length, :context_length]
        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(attention_scores, dim=-1)
        output = attention_weights @ values

        if return_attention_weights:
            return output, attention_weights

        return output
