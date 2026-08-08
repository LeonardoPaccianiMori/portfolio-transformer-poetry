import pytest

from sonnet_training.minerva_7b_historical_calibration import (
    HistoricalCalibrationConfig,
    validate_calibration_config,
)


def test_historical_calibration_recipe_is_locked():
    validate_calibration_config(HistoricalCalibrationConfig())

    with pytest.raises(ValueError, match="locked"):
        validate_calibration_config(HistoricalCalibrationConfig(timed_updates=2))
