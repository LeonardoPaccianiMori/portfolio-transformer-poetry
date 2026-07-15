import pytest

from sonnet_training.progress import TrainingProgressReporter, format_duration


def test_progress_reporter_writes_start_and_validation_progress(capsys):
    times = iter([10.0, 20.0])
    reporter = TrainingProgressReporter(
        total_steps=100,
        progress_interval=10,
        clock=lambda: next(times),
    )

    reporter.write_start(label="pretraining", device="cuda:0")
    reporter.write_progress(
        step=10,
        train_loss=2.5,
        validation_loss=2.7,
        learning_rate=3e-4,
        checkpoint_written=True,
        best_validation=True,
    )

    output = capsys.readouterr().out
    assert "pretraining started | steps=0/100 | device=cuda:0" in output
    assert "step=10/100" in output
    assert "progress=10.0%" in output
    assert "train_loss=2.5000" in output
    assert "validation_loss=2.7000" in output
    assert "learning_rate=3.00e-04" in output
    assert "elapsed=10s" in output
    assert "eta=1m30s" in output
    assert "checkpoint=saved" in output
    assert "best_validation=updated" in output


def test_progress_reporter_selects_first_interval_and_final_steps():
    reporter = TrainingProgressReporter(
        total_steps=25,
        progress_interval=10,
    )

    assert reporter.should_report(1)
    assert not reporter.should_report(9)
    assert reporter.should_report(10)
    assert reporter.should_report(25)
    assert reporter.should_report(9, force=True)


@pytest.mark.parametrize(
    ("total_steps", "progress_interval", "start_step", "message"),
    [
        (0, 10, 0, "total_steps"),
        (10, 0, 0, "progress_interval"),
        (10, 10, -1, "start_step"),
        (10, 10, 10, "start_step"),
    ],
)
def test_progress_reporter_rejects_invalid_initialization(
    total_steps: int,
    progress_interval: int,
    start_step: int,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        TrainingProgressReporter(
            total_steps=total_steps,
            progress_interval=progress_interval,
            start_step=start_step,
        )


def test_format_duration_formats_seconds_minutes_and_hours():
    assert format_duration(3.4) == "3s"
    assert format_duration(90.0) == "1m30s"
    assert format_duration(3_661.0) == "1h01m01s"
