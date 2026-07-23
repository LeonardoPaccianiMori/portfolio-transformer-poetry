import json

import pytest

from sonnet_corpus.pretraining_data_gap import (
    PretrainingDataGapConfig,
    build_pretraining_data_gap_report,
    render_pretraining_data_gap_markdown,
)


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_config(tmp_path):
    run_config_path = tmp_path / "run.json"
    encoding_report_path = tmp_path / "encoding.json"
    tokenizer_report_path = tmp_path / "tokenizer.json"
    balance_report_path = tmp_path / "balance.json"
    output_path = tmp_path / "report.md"
    write_json(
        run_config_path,
        {
            "batch_size": 2,
            "context_length": 512,
            "completed_steps": 100_000,
            "parameter_count": 10_000_000,
        },
    )
    write_json(
        encoding_report_path,
        {
            "total_tokens": 20_000_000,
            "train_tokens": 19_800_000,
            "validation_tokens": 200_000,
            "source_count": 12,
        },
    )
    write_json(tokenizer_report_path, {"characters_per_token": 2.5})
    write_json(
        balance_report_path,
        {
            "total_cleaned_characters": 50_000_000,
            "source_entries": [
                {"name": "largest_work", "character_share": 0.25},
            ],
            "author_entries": [
                {"name": "largest_author", "character_share": 0.30},
            ],
        },
    )
    return PretrainingDataGapConfig(
        run_config_path=run_config_path,
        encoding_report_path=encoding_report_path,
        tokenizer_report_path=tokenizer_report_path,
        balance_report_path=balance_report_path,
        markdown_report_path=output_path,
        target_unique_tokens=75_000_000,
        max_train_steps=650_000,
    )


def test_build_pretraining_data_gap_report_calculates_scale_and_exposure(tmp_path):
    config = make_config(tmp_path)

    report = build_pretraining_data_gap_report(config)

    assert report["additional_unique_tokens_needed"] == 55_000_000
    assert report["target_completion_fraction"] == pytest.approx(20_000_000 / 75_000_000)
    assert report["tokens_per_step"] == 1024
    assert report["completed_exposures"] == 102_400_000
    assert report["completed_passes_over_train_stream"] == pytest.approx(
        102_400_000 / 19_800_000
    )
    assert report["max_exposures"] == 665_600_000
    assert report["max_passes_over_target_corpus"] == pytest.approx(
        665_600_000 / 75_000_000
    )
    assert config.markdown_report_path.is_file()


def test_render_pretraining_data_gap_markdown_records_scale_and_no_cap_decision(tmp_path):
    report = build_pretraining_data_gap_report(make_config(tmp_path))

    markdown = render_pretraining_data_gap_markdown(report)

    assert "# Pretraining Data-Gap Report" in markdown
    assert "55,000,000" in markdown
    assert "largest_work" in markdown
    assert "does not propose capping it" in markdown


def test_build_pretraining_data_gap_report_rejects_missing_required_fields(tmp_path):
    config = make_config(tmp_path)
    write_json(config.encoding_report_path, {"total_tokens": 20_000_000})

    with pytest.raises(ValueError, match="train_tokens"):
        build_pretraining_data_gap_report(config)


def test_build_pretraining_data_gap_report_rejects_non_positive_target(tmp_path):
    config = make_config(tmp_path)
    invalid_config = PretrainingDataGapConfig(
        **{**config.__dict__, "target_unique_tokens": 0}
    )

    with pytest.raises(ValueError, match="target_unique_tokens"):
        build_pretraining_data_gap_report(invalid_config)
