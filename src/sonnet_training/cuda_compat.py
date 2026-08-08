"""CUDA memory helpers for runtimes that reject explicit device arguments."""

from __future__ import annotations

from typing import Any

import torch


def activate_cuda_device(device: torch.device | str) -> int:
    """Make one CUDA device current and return its concrete index."""
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("device must be a CUDA device")
    device_index = 0 if resolved.index is None else resolved.index
    torch.cuda.set_device(device_index)
    return device_index


def prepare_cuda_memory_measurement(device: torch.device | str) -> int:
    """Clear cached blocks and reset peaks on the active CUDA device."""
    device_index = activate_cuda_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return device_index


def max_cuda_memory_allocated(device: torch.device | str) -> int:
    activate_cuda_device(device)
    return int(torch.cuda.max_memory_allocated())


def max_cuda_memory_reserved(device: torch.device | str) -> int:
    activate_cuda_device(device)
    return int(torch.cuda.max_memory_reserved())


def cuda_memory_info(device: torch.device | str) -> tuple[int, int]:
    activate_cuda_device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return int(free_bytes), int(total_bytes)


def synchronize_cuda(device: torch.device | str) -> None:
    activate_cuda_device(device)
    torch.cuda.synchronize()


def cuda_device_name(device: torch.device | str) -> str:
    activate_cuda_device(device)
    return str(torch.cuda.get_device_name())


def cuda_device_properties(device: torch.device | str) -> Any:
    activate_cuda_device(device)
    return torch.cuda.get_device_properties()

