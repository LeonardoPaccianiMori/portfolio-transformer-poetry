import json
from pathlib import Path

from sonnet_evaluation.minerva_7b_parent_confirmation import (
    CONFIRMATION_CONDITIONS,
    CONFIRMATION_OUTPUT_COUNT,
    CONFIRMATION_VERSION,
)
from sonnet_evaluation.minerva_7b_parent_confirmation_report import (
    build_parent_confirmation_review_artifacts,
    parse_parent_confirmation_review,
)


def test_parent_confirmation_report_blinds_then_applies_gates(
    tmp_path: Path, monkeypatch
):
    output_root = tmp_path / "generation"
    condition_summaries = []
    for condition in CONFIRMATION_CONDITIONS:
        condition_id = condition["condition_id"]
        generation_dir = output_root / condition_id
        generation_dir.mkdir(parents=True)
        generated_files = []
        for index in range(24):
            prompt_id = f"prompt_{index}__seed_4099"
            output_path = generation_dir / f"{prompt_id}.txt"
            opening = f"Opening {index}"
            output_path.write_text(
                opening
                + "\n"
                + "\n".join(
                    f"Verso {line} della prova{'  ' if line == 1 else ''}"
                    for line in range(1, 14)
                )
                + "\n",
                encoding="utf-8",
            )
            generated_files.append({
                "prompt_id": prompt_id,
                "source_prompt_id": f"prompt_{index}",
                "poem_id": f"poem_{index}",
                "author": f"Author {index}",
                "prompt_text": opening,
                "opening_line": opening,
                "path": str(output_path),
                "seed": 4099,
                "stop_reason": "target_lines",
                "generated_new_tokens": 100,
                "completed_continuation_lines": 13,
            })
        (generation_dir / "metadata.json").write_text(
            json.dumps({
                "generation_format": "task_format_opening_line_continuation",
                "total_line_target": 14,
                "stop_text": "<eos>",
                "generated_files": generated_files,
            }),
            encoding="utf-8",
        )
        condition_summaries.append({
            "condition_id": condition_id,
            "model_state": condition["model_state"],
            "output_count": 24,
        })
    (output_root / "confirmation_summary.json").write_text(
        json.dumps({
            "confirmation_version": CONFIRMATION_VERSION,
            "condition_count": len(CONFIRMATION_CONDITIONS),
            "output_count": CONFIRMATION_OUTPUT_COUNT,
            "conditions": condition_summaries,
            "final_test_used": False,
            "training_used": False,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sonnet_evaluation.minerva_7b_parent_confirmation_report."
        "load_training_records",
        lambda **kwargs: [{
            "poem_id": "training",
            "title_or_first_line": "Different training poem",
            "author": "Training Author",
            "clean_text_path": "training.txt",
            "text": "Completely unrelated training material.",
        }],
    )
    mapping_path = output_root / "blind_mapping.json"
    review_path = tmp_path / "review.md"
    automatic_path = tmp_path / "automatic.md"
    result_path = tmp_path / "result.md"

    pending = build_parent_confirmation_review_artifacts(
        output_root=output_root,
        mapping_path=mapping_path,
        review_path=review_path,
        automatic_report_path=automatic_path,
        result_report_path=result_path,
        repo_root=tmp_path,
        manifest_path=tmp_path / "manifest.csv",
    )

    assert pending["review_complete"] is False
    assert len(json.loads(mapping_path.read_text())) == CONFIRMATION_OUTPUT_COUNT
    review_text = review_path.read_text(encoding="utf-8")
    assert review_text.count("| TODO | TODO | TODO | TODO |") == 72
    assert "untouched_default" not in review_text
    assert not any(line.endswith(" ") for line in review_text.splitlines())
    automatic_text = automatic_path.read_text(encoding="utf-8")
    assert "High-risk overlap" in automatic_text

    completed_lines = []
    for line in review_text.splitlines():
        if line.startswith("| `"):
            blind_id = line.split("|", maxsplit=2)[1].strip()
            completed_lines.append(f"| {blind_id} | yes | yes | no | reviewed |")
        else:
            completed_lines.append(line)
    review_path.write_text("\n".join(completed_lines) + "\n", encoding="utf-8")

    completed = build_parent_confirmation_review_artifacts(
        output_root=output_root,
        mapping_path=mapping_path,
        review_path=review_path,
        automatic_report_path=automatic_path,
        result_report_path=result_path,
        repo_root=tmp_path,
        manifest_path=tmp_path / "manifest.csv",
    )

    assert completed["review_complete"] is True
    assert len(parse_parent_confirmation_review(review_path)) == 72
    result_text = result_path.read_text(encoding="utf-8")
    assert result_text.count("| pass |") == len(CONFIRMATION_CONDITIONS)
    assert "Validation candidate selected" in result_text
