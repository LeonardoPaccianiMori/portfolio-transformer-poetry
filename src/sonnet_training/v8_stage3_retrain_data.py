"""Frozen window plan for the full-weight V8 Stage-3 retrain.

The plan packs the encoded V8 train stream into fixed-length windows and
inserts one preservation-replay window after every fixed number of train
windows, so the replay supplies about five percent of target-token exposure.
All counts are derived from the shard sizes, never estimated.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from sonnet_training.form_targeted_data import load_token_shard

WINDOW_KIND_TRAIN = "train"
WINDOW_KIND_REPLAY = "replay"


def replay_token_target(train_tokens: int, fraction: float) -> int:
    if train_tokens <= 0:
        raise ValueError("train token count must be positive")
    if not 0 < fraction < 1:
        raise ValueError("replay fraction must be between zero and one")
    return math.ceil(train_tokens * fraction / (1.0 - fraction))


def replay_every(train_windows: int, fraction: float) -> int:
    if train_windows <= 0:
        raise ValueError("train window count must be positive")
    return max(1, round((1.0 - fraction) / fraction))


def build_window_plan(
    train_tokens: int,
    replay_tokens: int,
    *,
    context_length: int,
    fraction: float,
) -> list[dict[str, Any]]:
    train_windows = train_tokens // context_length
    if train_windows == 0:
        raise ValueError("train stream is shorter than one window")
    every = replay_every(train_windows, fraction)
    plan: list[dict[str, Any]] = []
    replay_start = 0
    for index in range(train_windows):
        if index * context_length + context_length + 1 > train_tokens:
            break
        plan.append(
            {
                "kind": WINDOW_KIND_TRAIN,
                "token_start": index * context_length,
            }
        )
        if (index + 1) % every == 0:
            if replay_start + context_length + 1 <= replay_tokens:
                plan.append(
                    {
                        "kind": WINDOW_KIND_REPLAY,
                        "token_start": replay_start,
                    }
                )
                replay_start += context_length
    if not plan:
        raise ValueError("window plan is empty")
    return plan


def plan_summary(plan: list[dict[str, Any]], *, windows_per_update: int) -> dict[str, Any]:
    train_windows = sum(1 for row in plan if row["kind"] == WINDOW_KIND_TRAIN)
    replay_windows = len(plan) - train_windows
    updates = len(plan) // windows_per_update
    if updates == 0:
        raise ValueError("window plan is shorter than one optimizer update")
    return {
        "total_windows": len(plan),
        "train_windows": train_windows,
        "replay_windows": replay_windows,
        "replay_share": replay_windows / len(plan),
        "optimizer_updates": updates,
        "trailing_windows_dropped": len(plan) - updates * windows_per_update,
    }


def load_plan_inputs(
    *, train_shard: Path, replay_shard: Path, config: Mapping[str, Any]
) -> tuple[Any, Any, list[dict[str, Any]], dict[str, Any]]:
    train = load_token_shard(train_shard)
    replay = load_token_shard(replay_shard)
    data = config["data"]
    context_length = int(data["context_length"])
    fraction = float(data["replay_fraction"])
    target = replay_token_target(int(train.size), fraction)
    if target > int(replay.size):
        raise ValueError("replay shard is too small for the requested fraction")
    plan = build_window_plan(
        int(train.size),
        int(replay.size),
        context_length=context_length,
        fraction=fraction,
    )
    summary = plan_summary(
        plan, windows_per_update=int(data["windows_per_update"])
    )
    summary["replay_token_target"] = target
    summary["replay_tokens_used"] = summary["replay_windows"] * context_length
    summary["context_length"] = context_length
    return train, replay, plan, summary
