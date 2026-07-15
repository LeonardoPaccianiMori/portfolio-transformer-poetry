import torch
from torch import nn

from sonnet_corpus.batching import sample_next_token_batch
from sonnet_training.progress import TrainingProgressReporter


def train_next_token_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    token_ids: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device | str,
    max_gradient_norm: float | None = None,
    return_gradient_norm: bool = False,
) -> float | tuple[float, float | None]:
    """Run one update, optionally clipping and reporting the gradient norm."""
    if max_gradient_norm is not None and max_gradient_norm <= 0:
        raise ValueError("max_gradient_norm must be greater than 0 when provided")

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
    pre_clipping_gradient_norm = None
    if max_gradient_norm is not None:
        pre_clipping_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_gradient_norm,
            ).item()
        )
    optimizer.step()

    loss_value = float(loss.item())
    if return_gradient_norm:
        return loss_value, pre_clipping_gradient_norm
    return loss_value


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
    progress_interval: int = 0,
) -> list[dict[str, float | int]]:
    if train_steps <= 0:
        raise ValueError("train_steps must be greater than 0")

    if eval_interval <= 0:
        raise ValueError("eval_interval must be greater than 0")
    if progress_interval < 0:
        raise ValueError("progress_interval must be greater than or equal to 0")

    history = []
    progress = None
    if progress_interval:
        progress = TrainingProgressReporter(
            total_steps=train_steps,
            progress_interval=progress_interval,
        )
        progress.write_start(label="transformer training", device=str(device))

    for step in range(1, train_steps + 1):
        train_loss = train_next_token_step(
            model=model,
            optimizer=optimizer,
            token_ids=train_token_ids,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        should_evaluate = (
            step == 1
            or step % eval_interval == 0
            or step == train_steps
        )
        if should_evaluate:
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

        if progress is not None and progress.should_report(
            step,
            force=should_evaluate,
        ):
            progress.write_progress(
                step=step,
                train_loss=train_loss,
                validation_loss=(validation_loss if should_evaluate else None),
            )

    return history
