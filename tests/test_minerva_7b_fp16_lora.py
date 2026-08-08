from dataclasses import replace

import pytest
import torch

from sonnet_training.minerva_7b_fp16_lora import (
    MINERVA_7B_FP16_MINIMUM_HEADROOM_MIB,
    Minerva7BFP16LoRACalibrationConfig,
    build_minerva_7b_fp16_lora_report,
    validate_minerva_7b_fp16_lora_config,
)


def test_minerva_7b_fp16_lora_recipe_is_locked_and_unquantized():
    config = Minerva7BFP16LoRACalibrationConfig()

    validate_minerva_7b_fp16_lora_config(config)

    assert config.context_length == 512
    assert config.batch_size == 1
    assert config.lora_rank == 8
    assert config.target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert MINERVA_7B_FP16_MINIMUM_HEADROOM_MIB == 4096.0

    with pytest.raises(ValueError, match="locked"):
        validate_minerva_7b_fp16_lora_config(replace(config, lora_rank=4))


def test_minerva_7b_fp16_lora_report_applies_remote_headroom_gate():
    report = build_minerva_7b_fp16_lora_report(
        config=Minerva7BFP16LoRACalibrationConfig(),
        status="ok",
        device=torch.device("cuda:0"),
        gpu_name="Test GPU",
        total_gpu_memory_mib=49152.0,
        peak_allocated_mib=24000.0,
        peak_reserved_mib=25000.0,
        free_memory_after_mib=23000.0,
        loss=2.5,
        total_parameter_count=1000,
        trainable_parameter_count=10,
        optimizer_update_seconds=2.0,
        processed_tokens=400,
        package_versions={"torch": "test"},
    )

    assert report["remote_training_fit_decision"] == "pass"
    assert report["weight_loading"]["quantized"] is False
    assert report["tokens_per_second"] == 200.0

    rejected = build_minerva_7b_fp16_lora_report(
        config=Minerva7BFP16LoRACalibrationConfig(),
        status="ok",
        device=torch.device("cuda:0"),
        gpu_name="Test GPU",
        total_gpu_memory_mib=49152.0,
        peak_allocated_mib=46000.0,
        peak_reserved_mib=47000.0,
        free_memory_after_mib=2000.0,
        loss=2.5,
        total_parameter_count=1000,
        trainable_parameter_count=10,
        optimizer_update_seconds=2.0,
        processed_tokens=400,
        package_versions={"torch": "test"},
    )

    assert rejected["remote_training_fit_decision"] == "reject"
