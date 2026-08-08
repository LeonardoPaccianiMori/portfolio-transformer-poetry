from types import SimpleNamespace

import torch

from sonnet_training.cuda_compat import (
    activate_cuda_device,
    cuda_device_name,
    cuda_device_properties,
    cuda_memory_info,
    max_cuda_memory_allocated,
    max_cuda_memory_reserved,
    prepare_cuda_memory_measurement,
    synchronize_cuda,
)


def test_cuda_compat_uses_active_device_and_no_argument_memory_apis(monkeypatch):
    calls = []

    monkeypatch.setattr(torch.cuda, "set_device", lambda index: calls.append(("set", index)))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(("empty",)))
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda: calls.append(("reset",)),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 200)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (300, 400))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append(("sync",)))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "Test GPU")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda: SimpleNamespace(total_memory=400),
    )

    assert activate_cuda_device("cuda:0") == 0
    assert prepare_cuda_memory_measurement("cuda:0") == 0
    assert max_cuda_memory_allocated("cuda:0") == 100
    assert max_cuda_memory_reserved("cuda:0") == 200
    assert cuda_memory_info("cuda:0") == (300, 400)
    synchronize_cuda("cuda:0")
    assert cuda_device_name("cuda:0") == "Test GPU"
    assert cuda_device_properties("cuda:0").total_memory == 400
    assert ("reset",) in calls
    assert ("sync",) in calls

