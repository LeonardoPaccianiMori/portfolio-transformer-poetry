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


def train_next_token_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_token_ids: torch.Tensor,
    validation_token_ids: torch.Tensor,
    batch_size: int,
    context_length: int,
    train_steps: int,
    eval_interval: int,
    eval_batches: int,
    device: torch.device | str,
) -> list[dict[str, float | int]]:
    if train_steps <= 0:
        raise ValueError("train_steps must be greater than 0")

    if eval_interval <= 0:
        raise ValueError("eval_interval must be greater than 0")

    history = []

    for step in range(1, train_steps + 1):
        train_loss = train_next_token_step(
            model=model,
            optimizer=optimizer,
            token_ids=train_token_ids,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        if step == 1 or step % eval_interval == 0 or step == train_steps:
            validation_loss = estimate_next_token_loss(
                model=model,
                token_ids=validation_token_ids,
                batch_size=batch_size,
                context_length=context_length,
                eval_batches=eval_batches,
                device=device,
            )

            history.append({
                "step": step,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            })

    return history
