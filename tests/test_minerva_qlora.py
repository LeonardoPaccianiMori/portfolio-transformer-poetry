from pathlib import Path

import pytest
import torch

from sonnet_training.minerva_qlora import MINERVA_3B_MODEL_ID
from sonnet_training.minerva_qlora import MINERVA_3B_REVISION
from sonnet_training.minerva_qlora import MINERVA_QLORA_TARGET_MODULES
from sonnet_training.minerva_qlora import MinervaQLoRACalibrationConfig
from sonnet_training.minerva_qlora import build_calibration_report
from sonnet_training.minerva_qlora import validate_calibration_config
from sonnet_training.minerva_qlora import write_calibration_report


def test_minerva_calibration_configuration_locks_the_hardware_gate():
    config = MinervaQLoRACalibrationConfig()

    validate_calibration_config(config)

    assert config.model_id == MINERVA_3B_MODEL_ID
    assert config.revision == MINERVA_3B_REVISION
    assert config.context_length == 512
    assert config.batch_size == 1
    assert config.lora_rank == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.05
    assert config.target_modules == MINERVA_QLORA_TARGET_MODULES


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("model_id", "sapienzanlp/Minerva-7B-base-v1.0", "locked to Minerva"),
        ("context_length", 1024, "locked to 512"),
        ("batch_size", 2, "locked to 1"),
        ("lora_rank", 8, "rank and alpha"),
        ("lora_alpha", 16, "rank and alpha"),
        ("lora_dropout", 0.0, "dropout"),
    ],
)
def test_minerva_calibration_rejects_recipe_drift(field_name, value, message):
    config = MinervaQLoRACalibrationConfig()
    changed = {**config.__dict__, field_name: value}

    with pytest.raises(ValueError, match=message):
        validate_calibration_config(MinervaQLoRACalibrationConfig(**changed))


def test_calibration_report_records_memory_and_adapter_scope(tmp_path: Path):
    report = build_calibration_report(
        config=MinervaQLoRACalibrationConfig(),
        device=torch.device("cuda:0"),
        gpu_name="Test GPU",
        loss=1.25,
        total_parameter_count=1000,
        trainable_parameter_count=20,
        peak_allocated_mib=1234.5,
        peak_reserved_mib=1400.0,
        package_versions={"torch": "test"},
    )
    output_path = tmp_path / "calibration.json"

    write_calibration_report(output_path, report)

    assert report["quantization"]["quant_type"] == "nf4"
    assert report["gradient_checkpointing"] is True
    assert report["optimizer"] == "PagedAdamW8bit"
    assert report["trainable_parameter_fraction"] == 0.02
    assert '"peak_allocated_mib": 1234.5' in output_path.read_text(encoding="utf-8")
