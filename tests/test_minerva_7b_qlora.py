from dataclasses import replace
from pathlib import Path

import pytest
import torch

from sonnet_training.minerva_7b_qlora import (
    MINERVA_7B_QLORA_TARGET_MODULES,
    Minerva7BQLoRACalibrationConfig,
    build_minerva_7b_calibration_report,
    validate_minerva_7b_calibration_config,
    write_minerva_7b_calibration_report,
)


def test_minerva_7b_calibration_recipe_is_locked():
    config = Minerva7BQLoRACalibrationConfig()

    validate_minerva_7b_calibration_config(config)

    assert config.context_length == 512
    assert config.batch_size == 1
    assert config.lora_rank == 8
    assert config.lora_alpha == 16
    assert config.target_modules == MINERVA_7B_QLORA_TARGET_MODULES

    with pytest.raises(ValueError, match="locked"):
        validate_minerva_7b_calibration_config(replace(config, lora_rank=4))


def test_minerva_7b_calibration_report_applies_headroom_gate(tmp_path: Path):
    report = build_minerva_7b_calibration_report(
        config=Minerva7BQLoRACalibrationConfig(),
        status="ok",
        device=torch.device("cuda:0"),
        gpu_name="Test GPU",
        total_gpu_memory_mib=6144.0,
        peak_allocated_mib=5000.0,
        peak_reserved_mib=5400.0,
        free_memory_after_mib=600.0,
        loss=2.5,
        total_parameter_count=1000,
        trainable_parameter_count=10,
        package_versions={"torch": "test"},
    )
    output_path = tmp_path / "calibration.json"
    write_minerva_7b_calibration_report(output_path, report)

    assert report["local_training_fit_decision"] == "pass"
    assert report["trainable_parameter_fraction"] == 0.01
    assert '"status": "ok"' in output_path.read_text(encoding="utf-8")


def test_minerva_7b_oom_is_a_completed_rejection():
    report = build_minerva_7b_calibration_report(
        config=Minerva7BQLoRACalibrationConfig(),
        status="out_of_memory",
        device=torch.device("cuda:0"),
        gpu_name="Test GPU",
        total_gpu_memory_mib=6144.0,
        peak_allocated_mib=5900.0,
        peak_reserved_mib=6100.0,
        free_memory_after_mib=100.0,
        loss=None,
        total_parameter_count=None,
        trainable_parameter_count=None,
        package_versions={"torch": "test"},
        error="CUDA out of memory",
    )

    assert report["status"] == "out_of_memory"
    assert report["local_training_fit_decision"] == "reject"
