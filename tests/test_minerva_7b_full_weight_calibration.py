from dataclasses import replace

import pytest
import torch

from sonnet_training.minerva_7b_full_weight_calibration import (
    MINIMUM_POST_OPTIMIZER_HEADROOM_MIB,
    Minerva7BFullWeightCalibrationConfig,
    audit_full_weight_model,
    build_full_weight_calibration_report,
    validate_full_weight_calibration_config,
)


def _model_audit(parameter_count=1000):
    return {
        "total_parameter_count": parameter_count,
        "trainable_parameter_count": parameter_count,
        "trainable_parameter_fraction": 1.0,
        "frozen_parameter_names": [],
        "adapter_parameter_names": [],
        "quantized": False,
        "parameter_dtype_counts": {"bfloat16": parameter_count},
        "all_weights_trainable": True,
        "adapter_free": True,
        "quantization_free": True,
    }


def _update_rows():
    return [
        {
            "update": index + 1,
            "source_split": "paisa_train",
            "loss": 2.0 - index * 0.01,
            "gradient_norm": 0.5,
            "learning_rate": 1e-6,
            "update_seconds": 1.0,
            "free_memory_after_optimizer_mib": 20_000.0,
        }
        for index in range(5)
    ]


def test_full_weight_calibration_recipe_is_locked():
    config = Minerva7BFullWeightCalibrationConfig()

    validate_full_weight_calibration_config(config)

    assert config.context_length == 512
    assert config.optimizer_updates == 5
    assert config.optimizer == "PagedAdamW8bit"
    assert config.parameter_dtype == "bfloat16"
    assert MINIMUM_POST_OPTIMIZER_HEADROOM_MIB == 8192
    with pytest.raises(ValueError, match="locked"):
        validate_full_weight_calibration_config(
            replace(config, learning_rate=2e-6)
        )


def test_full_weight_model_audit_detects_frozen_parameters():
    model = torch.nn.Linear(4, 4, bias=False).to(torch.bfloat16)

    accepted = audit_full_weight_model(model)

    assert accepted["all_weights_trainable"] is True
    assert accepted["parameter_dtype_counts"] == {"bfloat16": 16}

    model.weight.requires_grad = False
    rejected = audit_full_weight_model(model)

    assert rejected["all_weights_trainable"] is False
    assert rejected["trainable_parameter_fraction"] == 0.0


def test_full_weight_report_requires_numerics_model_and_eight_gib_headroom():
    config = Minerva7BFullWeightCalibrationConfig()
    report = build_full_weight_calibration_report(
        config=config,
        status="ok",
        device=torch.device("cuda:0"),
        gpu_name="NVIDIA H100 80GB HBM3",
        total_gpu_memory_mib=81_000.0,
        native_bf16_supported=True,
        model_audit=_model_audit(),
        update_rows=_update_rows(),
        validation={"initial": {}, "final": {}},
        peak_allocated_mib=50_000.0,
        peak_reserved_mib=60_000.0,
        minimum_free_after_optimizer_mib=19_000.0,
        elapsed_seconds=5.0,
        package_versions={"torch": "test"},
    )

    assert report["full_weight_training_fit_decision"] == "pass"
    assert report["processed_tokens"] == 5 * 512
    assert report["retained_model_checkpoint"] is False
    assert report["long_training_authorized"] is False

    rejected = build_full_weight_calibration_report(
        config=config,
        status="ok",
        device=torch.device("cuda:0"),
        gpu_name="NVIDIA H100 80GB HBM3",
        total_gpu_memory_mib=81_000.0,
        native_bf16_supported=True,
        model_audit=_model_audit(),
        update_rows=_update_rows(),
        validation={"initial": {}, "final": {}},
        peak_allocated_mib=70_000.0,
        peak_reserved_mib=75_000.0,
        minimum_free_after_optimizer_mib=6_000.0,
        elapsed_seconds=5.0,
        package_versions={"torch": "test"},
    )

    assert rejected["memory_gate_passed"] is False
    assert rejected["full_weight_training_fit_decision"] == "reject"
