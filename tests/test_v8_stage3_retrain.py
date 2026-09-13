import pytest

from sonnet_training.v8_stage3_retrain import (
    abort_reason,
    retention_ratio,
    stage3_learning_rate,
)


def test_stage3_learning_rate_warms_up_decays_and_floors():
    kwargs = {
        "total_updates": 93,
        "warmup_updates": 7,
        "peak_learning_rate": 1e-6,
        "minimum_learning_rate": 1e-7,
    }
    assert stage3_learning_rate(0, **kwargs) == pytest.approx(1e-6 / 7)
    assert stage3_learning_rate(6, **kwargs) == pytest.approx(1e-6)
    assert stage3_learning_rate(50, **kwargs) < stage3_learning_rate(10, **kwargs)
    assert stage3_learning_rate(92, **kwargs) < 4e-7
    assert stage3_learning_rate(93, **kwargs) == pytest.approx(1e-7)
    assert stage3_learning_rate(1000, **kwargs) == pytest.approx(1e-7)


def test_retention_ratio():
    assert retention_ratio(1.0, 1.05) == pytest.approx(1.05)
    with pytest.raises(ValueError):
        retention_ratio(0.0, 1.0)


def test_abort_reason_covers_each_rule():
    common = {
        "elapsed_seconds": 10.0,
        "hourly_rate_usd": 2.5,
        "spend_ceiling_usd": 8.0,
        "max_preclip_gradient_norm": 100.0,
    }
    assert (
        abort_reason(loss=float("nan"), preclip_gradient_norm=1.0, **common)
        == "non_finite_loss"
    )
    assert (
        abort_reason(loss=1.0, preclip_gradient_norm=101.0, **common)
        == "gradient_spike"
    )
    assert (
        abort_reason(
            loss=1.0,
            preclip_gradient_norm=1.0,
            elapsed_seconds=20000.0,
            hourly_rate_usd=2.5,
            spend_ceiling_usd=8.0,
            max_preclip_gradient_norm=100.0,
        )
        == "spend_ceiling"
    )
    assert abort_reason(loss=1.0, preclip_gradient_norm=1.0, **common) is None
