import torch
from torch import nn

from sonnet_corpus.batching import sample_next_token_batch


def train_next_token_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    token_ids: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device | str,
) -> float:
    model.train()

    input_ids, target_ids = sample_next_token_batch(
        token_ids=token_ids,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
    )

    _, loss = model(input_ids, target_ids)

    if loss is None:
        raise RuntimeError("model did not return a training loss")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return float(loss.item())


def estimate_next_token_loss(
    model: nn.Module,
    token_ids: torch.Tensor,
    batch_size: int,
    context_length: int,
    eval_batches: int,
    device: torch.device | str,
) -> float:
    if eval_batches <= 0:
        raise ValueError("eval_batches must be greater than 0")

    model.eval()

    losses = []

    with torch.no_grad():
        for _ in range(eval_batches):
            input_ids, target_ids = sample_next_token_batch(
                token_ids=token_ids,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
            )

            _, loss = model(input_ids, target_ids)

            if loss is None:
                raise RuntimeError("model did not return an evaluation loss")

            losses.append(loss.item())

    model.train()

    return float(sum(losses) / len(losses))
