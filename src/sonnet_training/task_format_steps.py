"""Batching, masked updates, and deterministic validation for task-format data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from sonnet_corpus.task_format import (
    IGNORE_INDEX,
    EncodedSonnetContinuationExample,
)


@dataclass(frozen=True)
class SonnetContinuationBatch:
    """One right-padded task-format batch and its supervised-token count."""

    input_ids: torch.Tensor
    target_ids: torch.Tensor
    supervised_target_count: int


def collate_sonnet_continuation_examples(
    examples: Sequence[EncodedSonnetContinuationExample],
    pad_token_id: int,
    max_context_length: int,
    device: torch.device | str | None = None,
) -> SonnetContinuationBatch:
    """Right-pad full sonnet examples while ignoring all padded labels.

    Padding uses the existing end-of-text token. Because padding is appended on
    the right, causal attention prevents it from affecting predictions at any
    valid earlier position, and its labels are ignored by cross-entropy.
    """
    if not examples:
        raise ValueError("examples must contain at least one task example")
    if pad_token_id < 0:
        raise ValueError("pad_token_id must be greater than or equal to 0")
    if max_context_length <= 0:
        raise ValueError("max_context_length must be greater than 0")

    _validate_encoded_examples(examples)
    longest_sequence_length = max(example.input_ids.numel() for example in examples)
    if longest_sequence_length > max_context_length:
        raise ValueError("task example exceeds max_context_length")

    batch_size = len(examples)
    input_ids = torch.full(
        (batch_size, longest_sequence_length),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    target_ids = torch.full(
        (batch_size, longest_sequence_length),
        fill_value=IGNORE_INDEX,
        dtype=torch.long,
    )

    for index, example in enumerate(examples):
        sequence_length = example.input_ids.numel()
        input_ids[index, :sequence_length] = example.input_ids
        target_ids[index, :sequence_length] = example.target_ids

    supervised_target_count = int((target_ids != IGNORE_INDEX).sum().item())
    if supervised_target_count == 0:
        raise ValueError("task batch must contain at least one supervised target")

    if device is not None:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

    return SonnetContinuationBatch(
        input_ids=input_ids,
        target_ids=target_ids,
        supervised_target_count=supervised_target_count,
    )


def sample_sonnet_continuation_batch(
    examples: Sequence[EncodedSonnetContinuationExample],
    batch_size: int,
    pad_token_id: int,
    max_context_length: int,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> SonnetContinuationBatch:
    """Sample full sonnets uniformly by document, with replacement."""
    if not examples:
        raise ValueError("examples must contain at least one task example")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    indices = torch.randint(
        low=0,
        high=len(examples),
        size=(batch_size,),
        generator=generator,
    )
    selected_examples = [examples[index.item()] for index in indices]
    return collate_sonnet_continuation_examples(
        examples=selected_examples,
        pad_token_id=pad_token_id,
        max_context_length=max_context_length,
        device=device,
    )


def train_sonnet_continuation_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    examples: Sequence[EncodedSonnetContinuationExample],
    batch_size: int,
    pad_token_id: int,
    max_context_length: int,
    device: torch.device | str,
    max_gradient_norm: float | None = None,
    return_gradient_norm: bool = False,
    gradient_accumulation_steps: int = 1,
) -> float | tuple[float, float | None]:
    """Run one masked optimizer update over sampled full-sonnet batches."""
    if max_gradient_norm is not None and max_gradient_norm <= 0:
        raise ValueError("max_gradient_norm must be greater than 0 when provided")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be greater than 0")

    microbatches = [
        sample_sonnet_continuation_batch(
            examples=examples,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
            max_context_length=max_context_length,
            device=device,
        )
        for _ in range(gradient_accumulation_steps)
    ]
    total_supervised_targets = sum(
        batch.supervised_target_count
        for batch in microbatches
    )

    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    for batch in microbatches:
        _, loss = model(batch.input_ids, batch.target_ids)
        if loss is None:
            raise RuntimeError("model did not return a training loss")

        target_weight = batch.supervised_target_count / total_supervised_targets
        (loss * target_weight).backward()
        total_loss += float(loss.item()) * batch.supervised_target_count

    pre_clipping_gradient_norm = None
    if max_gradient_norm is not None:
        pre_clipping_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_gradient_norm,
            ).item()
        )
    optimizer.step()

    loss_value = total_loss / total_supervised_targets
    if return_gradient_norm:
        return loss_value, pre_clipping_gradient_norm
    return loss_value


def estimate_sonnet_continuation_loss(
    model: nn.Module,
    examples: Sequence[EncodedSonnetContinuationExample],
    batch_size: int,
    pad_token_id: int,
    max_context_length: int,
    device: torch.device | str,
) -> float:
    """Score every task-format example once with token-weighted loss."""
    if not examples:
        raise ValueError("examples must contain at least one task example")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_supervised_targets = 0

    with torch.no_grad():
        for first_index in range(0, len(examples), batch_size):
            batch = collate_sonnet_continuation_examples(
                examples=examples[first_index:first_index + batch_size],
                pad_token_id=pad_token_id,
                max_context_length=max_context_length,
                device=device,
            )
            _, loss = model(batch.input_ids, batch.target_ids)
            if loss is None:
                raise RuntimeError("model did not return an evaluation loss")

            total_loss += float(loss.item()) * batch.supervised_target_count
            total_supervised_targets += batch.supervised_target_count

    model.train(was_training)
    return total_loss / total_supervised_targets


def _validate_encoded_examples(
    examples: Sequence[EncodedSonnetContinuationExample],
) -> None:
    for example in examples:
        if example.input_ids.ndim != 1 or example.target_ids.ndim != 1:
            raise ValueError("task example tensors must be 1D")
        if example.input_ids.dtype != torch.long or example.target_ids.dtype != torch.long:
            raise ValueError("task example tensors must have dtype torch.long")
        if example.input_ids.shape != example.target_ids.shape:
            raise ValueError("task example input_ids and target_ids must match")
        if example.input_ids.numel() == 0:
            raise ValueError("task example tensors must not be empty")
