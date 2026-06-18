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
