import json
from pathlib import Path

from sonnet_evaluation.minerva_7b_quality_recovery import (
    RECOVERY_CONDITIONS,
    RECOVERY_OUTPUT_COUNT,
    RECOVERY_VERSION,
)
from sonnet_evaluation.minerva_7b_recovery_report import (
    build_recovery_review_artifacts,
    parse_recovery_review,
)


def test_recovery_report_builds_blinded_scaffold_then_final_result(tmp_path: Path):
    output_root = tmp_path / "generation"
    condition_summaries = []
    for condition in RECOVERY_CONDITIONS:
        condition_id = condition["condition_id"]
        generation_dir = output_root / condition_id
        generation_dir.mkdir(parents=True)
        generated_files = []
        for index in range(12):
            prompt_id = f"prompt_{index}__seed_2029"
            output_path = generation_dir / f"{prompt_id}.txt"
            opening = f"Opening {index}"
            output_path.write_text(
                opening + "\n" + "\n".join(
                    f"Verso {line} della prova{'  ' if line == 1 else ''}"
                    for line in range(1, 14)
                ) + "\n",
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
                "seed": 2029,
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
            "output_count": 12,
        })
    output_root.mkdir(exist_ok=True)
    (output_root / "recovery_summary.json").write_text(
        json.dumps({
            "recovery_version": RECOVERY_VERSION,
            "condition_count": len(RECOVERY_CONDITIONS),
            "output_count": RECOVERY_OUTPUT_COUNT,
            "conditions": condition_summaries,
            "final_test_used": False,
            "training_used": False,
        }),
        encoding="utf-8",
    )
    mapping_path = output_root / "blind_mapping.json"
    review_path = tmp_path / "review.md"
    automatic_path = tmp_path / "automatic.md"
    result_path = tmp_path / "result.md"

    pending = build_recovery_review_artifacts(
        output_root=output_root,
        mapping_path=mapping_path,
        review_path=review_path,
        automatic_report_path=automatic_path,
        result_report_path=result_path,
    )

    assert pending["review_complete"] is False
    assert pending["output_count"] == RECOVERY_OUTPUT_COUNT
    assert len(json.loads(mapping_path.read_text())) == RECOVERY_OUTPUT_COUNT
    review_text = review_path.read_text(encoding="utf-8")
    assert review_text.count("| TODO | TODO | TODO | TODO |") == RECOVERY_OUTPUT_COUNT
    assert not any(line.endswith(" ") for line in review_text.splitlines())
    assert "stage_b_control" not in review_text
    assert "Controlled form" in automatic_path.read_text(encoding="utf-8")

    completed_lines = []
    for line in review_text.splitlines():
        if line.startswith("| `"):
            blind_id = line.split("|", maxsplit=2)[1].strip()
            completed_lines.append(
                f"| {blind_id} | yes | yes | no | reviewed |"
            )
        else:
            completed_lines.append(line)
    review_path.write_text("\n".join(completed_lines) + "\n", encoding="utf-8")

    completed = build_recovery_review_artifacts(
        output_root=output_root,
        mapping_path=mapping_path,
        review_path=review_path,
        automatic_report_path=automatic_path,
        result_report_path=result_path,
    )

    assert completed["review_complete"] is True
    assert len(parse_recovery_review(review_path)) == RECOVERY_OUTPUT_COUNT
    result_text = result_path.read_text(encoding="utf-8")
    assert "Lineage:" in result_text
    assert "Stage B decoding:" in result_text
