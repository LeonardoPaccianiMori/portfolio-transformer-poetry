import json
from pathlib import Path

import pytest

from sonnet_evaluation.minerva_sanity_audit import (
    MINERVA_SANITY_CONDITION_IDS,
    MINERVA_SANITY_SELECTABLE_CONDITIONS,
    build_minerva_sanity_automatic_report,
    build_minerva_sanity_blinded_review,
    build_minerva_sonnet_instruction_prompt,
    minerva_sanity_conditions,
    score_minerva_sanity_audit,
    validate_minerva_sanity_prompts,
    _reusable_condition_metadata,
)


def _prompts():
    periods = [
        "XIII secolo",
        "XIII secolo",
        "XIII secolo",
        "XIV secolo",
        "XVI secolo",
        "XVI secolo",
        "XVII secolo",
        "XVIII secolo",
    ]
    return [
        {
            "id": f"prompt_{index}",
            "poem_id": f"validation_{index}",
            "author": f"Author {index}",
            "period": period,
            "opening_line": f"Opening {index}",
        }
        for index, period in enumerate(periods)
    ]


def _write_condition(output_root: Path, condition: dict, index: int) -> None:
    output_dir = output_root / condition["condition_id"]
    output_dir.mkdir(parents=True)
    generated_files = []
    for prompt_index in range(8):
        prompt_id = f"prompt_{prompt_index}__seed_4242"
        output_path = output_dir / f"{prompt_id}.txt"
        output_path.write_text(
            f"Opening {prompt_index}\n" + "\n".join(["Verso"] * 13) + "\n",
            encoding="utf-8",
        )
        generated_files.append({
            "prompt_id": prompt_id,
            "source_prompt_id": f"prompt_{prompt_index}",
            "poem_id": f"validation_{prompt_index}",
            "author": f"Author {prompt_index}",
            "prompt_text": f"Opening {prompt_index}",
            "opening_line": f"Opening {prompt_index}",
            "path": str(output_path),
            "seed": 4242,
            "stop_reason": "target_lines",
            "completed_continuation_lines": 13,
        })
    (output_dir / "metadata.json").write_text(
        json.dumps({
            "generation_format": "task_format_opening_line_continuation",
            "model_variant": condition["condition_id"],
            "total_line_target": 14,
            "stop_text": "</s>",
            "generated_files": generated_files,
        }),
        encoding="utf-8",
    )


def _write_audit(output_root: Path) -> list[dict]:
    conditions = minerva_sanity_conditions(
        best_checkpoint_path=Path("best.pt"),
        final_checkpoint_path=Path("final.pt"),
    )
    for index, condition in enumerate(conditions):
        _write_condition(output_root, condition, index)
    (output_root / "audit_metadata.json").write_text(
        json.dumps({
            "audit_version": "minerva_3b_validation_sanity_v1",
            "conditions": conditions,
        }),
        encoding="utf-8",
    )
    return conditions


def test_instruction_prompt_ends_with_exact_visible_opening():
    prompt = build_minerva_sonnet_instruction_prompt("Primo verso")

    assert "esattamente quattordici versi" in prompt
    assert prompt.endswith("Sonetto:\nPrimo verso\n")


def test_sanity_prompt_validation_rejects_final_test_overlap():
    prompts = _prompts()
    validate_minerva_sanity_prompts(prompts, [{"poem_id": "test_poem"}])

    with pytest.raises(ValueError, match="overlap"):
        validate_minerva_sanity_prompts(
            prompts,
            [{"poem_id": "validation_3"}],
        )


def test_sanity_conditions_lock_selectable_scales_and_diagnostic_controls():
    conditions = minerva_sanity_conditions(
        best_checkpoint_path=Path("best.pt"),
        final_checkpoint_path=Path("final.pt"),
    )

    assert tuple(row["condition_id"] for row in conditions) == (
        MINERVA_SANITY_CONDITION_IDS
    )
    assert tuple(
        row["condition_id"] for row in conditions if row["selectable"]
    ) == MINERVA_SANITY_SELECTABLE_CONDITIONS
    assert conditions[1]["conditioning_format"] == (
        "explicit_italian_sonnet_instruction_v1"
    )


def test_sanity_scoring_and_report_keep_automatic_evidence_non_decisive(tmp_path):
    output_root = tmp_path / "audit"
    _write_audit(output_root)

    rows = score_minerva_sanity_audit(output_root)
    report = build_minerva_sanity_automatic_report(output_root, rows)

    assert len(rows) == 7
    assert all(row["controlled_forms"] == 8 for row in rows)
    assert "best_scale_050" in report
    assert "Human judgments come from the separately blinded review" in report


def test_blinded_review_hides_condition_identity_and_writes_mapping(tmp_path):
    output_root = tmp_path / "audit"
    conditions = _write_audit(output_root)

    review, mapping = build_minerva_sanity_blinded_review(output_root)

    assert len(mapping) == 56
    assert "Generally grammatical Italian: TODO yes/no" in review
    assert all(condition["condition_id"] not in review for condition in conditions)
    assert {row["condition_id"] for row in mapping.values()} == set(
        MINERVA_SANITY_CONDITION_IDS
    )


def test_condition_reuse_requires_exact_adapter_identity(tmp_path):
    output_dir = tmp_path / "best_scale_050"
    output_dir.mkdir()
    generated_path = output_dir / "sample.txt"
    generated_path.write_text("Prima\nSeconda\n", encoding="utf-8")
    condition = {
        "condition_id": "best_scale_050",
        "checkpoint_path": "best.pt",
        "adapter_epoch": 3,
        "adapter_scale": 0.5,
        "conditioning_format": "opening_line_newline",
    }
    metadata = {
        "model_variant": "best_scale_050",
        "adapter_checkpoint_path": "best.pt",
        "adapter_epoch": 3,
        "adapter_scale": 0.5,
        "conditioning_format": "opening_line_newline",
        "max_new_tokens": 512,
        "seeds": [4242],
        "generated_files": [{"path": str(generated_path)}],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    assert _reusable_condition_metadata(
        output_dir=output_dir,
        condition=condition,
        prompt_count=1,
    ) == metadata

    condition["checkpoint_path"] = "other.pt"
    assert _reusable_condition_metadata(
        output_dir=output_dir,
        condition=condition,
        prompt_count=1,
    ) is None
