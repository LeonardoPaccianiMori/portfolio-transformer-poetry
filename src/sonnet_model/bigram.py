import torch
import torch.nn.functional as F
from torch import nn


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.token_embedding_table(input_ids)

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
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch_size, context_length)")

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be greater than or equal to 0")

        generated_ids = input_ids

        for _ in range(max_new_tokens):
            logits, _ = self(generated_ids)
            next_token_logits = logits[:, -1, :]
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

        return generated_ids
