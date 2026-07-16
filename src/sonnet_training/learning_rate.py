"""Learning-rate schedules shared by the project's training runners."""

from __future__ import annotations

import math
from typing import Literal, Protocol

import torch


LearningRateSchedule = Literal["constant", "warmup_cosine"]


class LearningRateScheduleConfig(Protocol):
    """The configuration fields needed to calculate one learning rate."""

    train_steps: int
    learning_rate: float
    learning_rate_schedule: LearningRateSchedule
    warmup_steps: int
    min_learning_rate: float


def learning_rate_for_step(config: LearningRateScheduleConfig, step: int) -> float:
    """Return the configured learning rate for one one-indexed training step."""

    if step <= 0 or step > config.train_steps:
        raise ValueError("step must be between 1 and train_steps")
    if config.learning_rate_schedule == "constant":
        return config.learning_rate

    if config.learning_rate_schedule == "warmup_cosine":
        if config.warmup_steps and step <= config.warmup_steps:
            return config.learning_rate * step / config.warmup_steps

        decay_steps = config.train_steps - config.warmup_steps
        decay_progress = (step - config.warmup_steps) / decay_steps
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return config.min_learning_rate + cosine_factor * (
            config.learning_rate - config.min_learning_rate
        )

    raise ValueError("unsupported learning_rate_schedule")


def set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    """Apply one learning rate to every optimizer parameter group."""

    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
