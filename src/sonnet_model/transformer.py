import torch
import torch.nn.functional as F
from torch import nn

from sonnet_model.normalization import NormalizationType, build_normalization_layer
from sonnet_model.positional_encoding import (
    PositionEncodingType,
    RotaryPositionEmbedding,
    validate_position_encoding_type,
)


class TokenAndPositionEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_context_length: int,
        position_encoding_type: PositionEncodingType = "learned_absolute",
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than 0")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0")

        if max_context_length <= 0:
            raise ValueError("max_context_length must be greater than 0")
        validate_position_encoding_type(position_encoding_type)

        self.max_context_length = max_context_length
        self.position_encoding_type = position_encoding_type
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = (
            nn.Embedding(max_context_length, embedding_dim)
            if position_encoding_type == "learned_absolute"
            else None
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch_size, context_length)")

        _, context_length = input_ids.shape

        if context_length > self.max_context_length:
            raise ValueError("context_length exceeds max_context_length")

        token_embeddings = self.token_embedding(input_ids)
        if self.position_embedding is None:
            return token_embeddings

        position_ids = torch.arange(context_length, device=input_ids.device)
        position_embeddings = self.position_embedding(position_ids)

        return token_embeddings + position_embeddings


class CausalSelfAttentionHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        head_dim: int,
        max_context_length: int,
        position_encoding_type: PositionEncodingType = "learned_absolute",
        rope_theta: float = 10_000.0,
    ):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0")

        if head_dim <= 0:
            raise ValueError("head_dim must be greater than 0")

        if max_context_length <= 0:
            raise ValueError("max_context_length must be greater than 0")
        validate_position_encoding_type(position_encoding_type)
        if rope_theta <= 0:
            raise ValueError("rope_theta must be greater than 0")

        self.max_context_length = max_context_length
        self.head_dim = head_dim
        self.position_encoding_type = position_encoding_type
        self.query = nn.Linear(embedding_dim, head_dim, bias=False)
        self.key = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, head_dim, bias=False)
        self.rotary_embedding = (
            RotaryPositionEmbedding(
                head_dim=head_dim,
                max_context_length=max_context_length,
                theta=rope_theta,
            )
            if position_encoding_type == "rope"
            else None
        )

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
        if self.rotary_embedding is not None:
            queries = self.rotary_embedding(queries)
            keys = self.rotary_embedding(keys)

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores * (self.head_dim ** -0.5)

        mask = self.causal_mask[:context_length, :context_length]
        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(attention_scores, dim=-1)
        output = attention_weights @ values

        if return_attention_weights:
            return output, attention_weights

        return output


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        head_dim: int,
        max_context_length: int,
        position_encoding_type: PositionEncodingType = "learned_absolute",
        rope_theta: float = 10_000.0,
    ):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0")

        if num_heads <= 0:
            raise ValueError("num_heads must be greater than 0")

        if head_dim <= 0:
            raise ValueError("head_dim must be greater than 0")

        if max_context_length <= 0:
            raise ValueError("max_context_length must be greater than 0")
        validate_position_encoding_type(position_encoding_type)

        self.heads = nn.ModuleList([
            CausalSelfAttentionHead(
                embedding_dim=embedding_dim,
                head_dim=head_dim,
                max_context_length=max_context_length,
                position_encoding_type=position_encoding_type,
                rope_theta=rope_theta,
            )
            for _ in range(num_heads)
        ])
        self.output_projection = nn.Linear(
            num_heads * head_dim,
            embedding_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(
                "x must have shape (batch_size, context_length, embedding_dim)"
            )

        if return_attention_weights:
            head_outputs = []
            head_attention_weights = []

            for head in self.heads:
                head_output, attention_weights = head(
                    x,
                    return_attention_weights=True,
                )
                head_outputs.append(head_output)
                head_attention_weights.append(attention_weights)

            concatenated = torch.cat(head_outputs, dim=-1)
            output = self.output_projection(concatenated)
            stacked_attention_weights = torch.stack(
                head_attention_weights,
                dim=1,
            )

            return output, stacked_attention_weights

        head_outputs = [
            head(x)
            for head in self.heads
        ]
        concatenated = torch.cat(head_outputs, dim=-1)

        return self.output_projection(concatenated)


class FeedForward(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        feed_forward_dim: int,
    ):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0")

        if feed_forward_dim <= 0:
            raise ValueError("feed_forward_dim must be greater than 0")

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Linear(feed_forward_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "x must have shape (batch_size, context_length, embedding_dim)"
            )

        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        head_dim: int,
        feed_forward_dim: int,
        max_context_length: int,
        normalization_type: NormalizationType = "layer_norm",
        normalization_eps: float = 1e-5,
        position_encoding_type: PositionEncodingType = "learned_absolute",
        rope_theta: float = 10_000.0,
    ):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0")

        if num_heads <= 0:
            raise ValueError("num_heads must be greater than 0")

        if head_dim <= 0:
            raise ValueError("head_dim must be greater than 0")

        if feed_forward_dim <= 0:
            raise ValueError("feed_forward_dim must be greater than 0")

        if max_context_length <= 0:
            raise ValueError("max_context_length must be greater than 0")
        validate_position_encoding_type(position_encoding_type)

        self.attention_layer_norm = build_normalization_layer(
            embedding_dim=embedding_dim,
            normalization_type=normalization_type,
            eps=normalization_eps,
        )
        self.attention = MultiHeadCausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            max_context_length=max_context_length,
            position_encoding_type=position_encoding_type,
            rope_theta=rope_theta,
        )
        self.feed_forward_layer_norm = build_normalization_layer(
            embedding_dim=embedding_dim,
            normalization_type=normalization_type,
            eps=normalization_eps,
        )
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            feed_forward_dim=feed_forward_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "x must have shape (batch_size, context_length, embedding_dim)"
            )

        x = x + self.attention(self.attention_layer_norm(x))
        x = x + self.feed_forward(self.feed_forward_layer_norm(x))

        return x


class CausalTransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        feed_forward_dim: int,
        max_context_length: int,
        normalization_type: NormalizationType = "layer_norm",
        normalization_eps: float = 1e-5,
        position_encoding_type: PositionEncodingType = "learned_absolute",
        rope_theta: float = 10_000.0,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than 0")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0")

        if num_layers <= 0:
            raise ValueError("num_layers must be greater than 0")

        if num_heads <= 0:
            raise ValueError("num_heads must be greater than 0")

        if head_dim <= 0:
            raise ValueError("head_dim must be greater than 0")

        if feed_forward_dim <= 0:
            raise ValueError("feed_forward_dim must be greater than 0")

        if max_context_length <= 0:
            raise ValueError("max_context_length must be greater than 0")
        validate_position_encoding_type(position_encoding_type)

        self.max_context_length = max_context_length
        self.normalization_type = normalization_type
        self.normalization_eps = normalization_eps
        self.position_encoding_type = position_encoding_type
        self.rope_theta = rope_theta
        self.embedding = TokenAndPositionEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_context_length=max_context_length,
            position_encoding_type=position_encoding_type,
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                feed_forward_dim=feed_forward_dim,
                max_context_length=max_context_length,
                normalization_type=normalization_type,
                normalization_eps=normalization_eps,
                position_encoding_type=position_encoding_type,
                rope_theta=rope_theta,
            )
            for _ in range(num_layers)
        ])
        self.final_layer_norm = build_normalization_layer(
            embedding_dim=embedding_dim,
            normalization_type=normalization_type,
            eps=normalization_eps,
        )
        self.output_projection = nn.Linear(embedding_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch_size, context_length)")

        if target_ids is not None and target_ids.shape != input_ids.shape:
            raise ValueError("target_ids must have the same shape as input_ids")

        hidden_states = self.embedding(input_ids)

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.output_projection(hidden_states)

        loss = None

        if target_ids is not None:
            batch_size, context_length, vocab_size = logits.shape
            logits_flat = logits.view(
                batch_size * context_length,
                vocab_size,
            )
            targets_flat = target_ids.view(
                batch_size * context_length,
            )
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        generator: torch.Generator | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        stop_token_id: int | None = None,
        forbidden_token_ids: set[int] | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch_size, context_length)")

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be greater than or equal to 0")

        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if forbidden_token_ids is not None:
            vocab_size = self.output_projection.out_features

            if len(forbidden_token_ids) >= vocab_size:
                raise ValueError("forbidden_token_ids cannot contain every token")

            invalid_token_ids = [
                token_id
                for token_id in forbidden_token_ids
                if token_id < 0 or token_id >= vocab_size
            ]

            if invalid_token_ids:
                raise ValueError("forbidden_token_ids contains an invalid token id")

        generated_ids = input_ids

        for _ in range(max_new_tokens):
            cropped_input_ids = generated_ids[:, -self.max_context_length:]
            logits, _ = self(cropped_input_ids)
            next_token_logits = logits[:, -1, :]
            next_token_logits = next_token_logits / temperature

            if forbidden_token_ids is not None:
                next_token_logits[:, list(forbidden_token_ids)] = float("-inf")

            if top_k is not None:
                vocab_size = next_token_logits.shape[-1]

                if top_k > vocab_size:
                    raise ValueError("top_k must be less than or equal to vocab_size")

                top_values, _ = torch.topk(next_token_logits, k=top_k, dim=-1)
                minimum_top_value = top_values[:, [-1]]
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < minimum_top_value,
                    float("-inf"),
                )

            next_token_probabilities = F.softmax(next_token_logits, dim=-1)
            next_token_ids = torch.multinomial(
                next_token_probabilities,
                num_samples=1,
                generator=generator,
            )
            generated_ids = torch.cat(
                (generated_ids, next_token_ids),
                dim=1,
            )

            if (
                stop_token_id is not None
                and torch.all(next_token_ids == stop_token_id)
            ):
                break

        return generated_ids
