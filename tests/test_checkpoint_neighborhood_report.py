import json
from pathlib import Path

import pytest

from sonnet_evaluation.checkpoint_neighborhood_report import (
    build_checkpoint_neighborhood_report,
    summarize_checkpoint_neighborhoods,
    write_checkpoint_neighborhood_report,
)


def write_generation_batch(output_dir: Path, prompt_text: str) -> None:
    output_dir.mkdir(parents=True)
    generated_path = output_dir / "amor.txt"
    generated_path.write_text(
        f"{prompt_text} che move il sole\n",
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps({
            "stop_text": None,
            "generated_files": [{
                "prompt_id": "amor",
                "prompt_text": prompt_text,
                "path": str(generated_path),
                "seed": 1337,
            }],
        }),
        encoding="utf-8",
    )


def write_neighborhood_metadata(tmp_path: Path) -> Path:
    before_dir = tmp_path / "outputs" / "tiny" / "before"
    best_dir = tmp_path / "outputs" / "tiny" / "best"
    write_generation_batch(before_dir, "Amor")
    write_generation_batch(best_dir, "Amor")
    metadata_path = tmp_path / "outputs" / "metadata.json"
    metadata_path.write_text(
        json.dumps({
            "runs": [{
                "id": "tiny",
                "selected_checkpoint_id": "best",
                "checkpoints": [
                    {
                        "id": "before",
                        "step": 10,
                        "validation_loss": 2.1,
                        "output_dir": str(before_dir),
                    },
                    {
                        "id": "best",
                        "step": 20,
                        "validation_loss": 2.0,
                        "output_dir": str(best_dir),
                    },
                ],
            }],
        }),
        encoding="utf-8",
    )
    return metadata_path


def test_summarize_checkpoint_neighborhoods_scores_all_batches(tmp_path: Path):
    metadata_path = write_neighborhood_metadata(tmp_path)

    summaries = summarize_checkpoint_neighborhoods(metadata_path)

    assert len(summaries) == 2
    assert summaries[0]["run_id"] == "tiny"
    assert summaries[0]["checkpoint_id"] == "before"
    assert not summaries[0]["selected_by_validation"]
    assert summaries[1]["selected_by_validation"]
    assert summaries[1]["all_prompts_preserved"]
    assert summaries[1]["average_character_count"] > 0


def test_write_checkpoint_neighborhood_report_writes_interpretation(tmp_path: Path):
    metadata_path = write_neighborhood_metadata(tmp_path)
    output_path = tmp_path / "reports" / "neighborhood.md"

    summaries = write_checkpoint_neighborhood_report(
        metadata_path=metadata_path,
        output_path=output_path,
    )

    report = output_path.read_text(encoding="utf-8")
    assert len(summaries) == 2
    assert "# Pretraining Checkpoint-Neighborhood Evaluation" in report
    assert "| Parent Run | Checkpoint |" in report
    assert "not a basis for cherry-picking" in report


def test_summarize_checkpoint_neighborhoods_rejects_missing_selected_batch(
    tmp_path: Path,
):
    metadata_path = write_neighborhood_metadata(tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["runs"][0]["selected_checkpoint_id"] = "missing"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="selects a checkpoint"):
        summarize_checkpoint_neighborhoods(metadata_path)


def test_build_checkpoint_neighborhood_report_rejects_empty_summaries():
    with pytest.raises(ValueError, match="must not be empty"):
        build_checkpoint_neighborhood_report([])
