import json
from contextlib import nullcontext
from pathlib import Path

from sonnet_analysis.minerva_v7_dpo_validation import (
    SYSTEM_IDS,
    generate_matched_validation,
)
from sonnet_evaluation.metrics import score_generated_text


class FakeModel:
    def disable_adapter(self):
        return nullcontext()


def test_matched_validation_writes_exact_paired_grid(monkeypatch, tmp_path):
    def fake_batch(**kwargs):
        return [
            {
                "text": f"{job['prompt']['opening_line']}\ncontinuazione",
                "opening_line": job["prompt"]["opening_line"],
                "conditioning_prompt": "p", "conditioning_input_ids": [1],
                "generated_token_ids": [2], "seed": job["seed"],
                "stop_reason": "target_lines", "generated_new_tokens": 1,
                "completed_continuation_lines": 1, "batch_elapsed_seconds": 0.1,
                "batch_size": len(kwargs["jobs"]),
            }
            for job in kwargs["jobs"]
        ]

    monkeypatch.setattr(
        "sonnet_analysis.minerva_v7_dpo_validation.generate_batch", fake_batch
    )
    result = generate_matched_validation(
        model=FakeModel(), tokenizer=object(),
        prompts=[{"id": "p1", "opening_line": "Prima"}],
        seeds=(7, 8), recipe={"recipe_id": "r"}, output_dir=tmp_path,
        state_identity="s", adapter_identity="a", device="cpu", batch_size=2,
    )
    assert result["completed_output_count"] == 4
    assert {row["system_id"] for row in result["outputs"]} == set(SYSTEM_IDS)
    assert result["v7_test_accessed"] is False
    for row in result["outputs"]:
        assert json.loads((tmp_path / row["path"]).read_text())["seed"] in {7, 8}


def test_validation_analysis_uses_public_prompt_preservation_metric_key():
    """Guard the analysis boundary against the removed starts_with_prompt key."""

    metrics = score_generated_text("Prima\ncontinuazione", "Prima")
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/analyze_minerva_v7_dpo_validation.py"
    ).read_text(encoding="utf-8")

    assert metrics["prompt_preserved"] is True
    assert 'metrics["prompt_preserved"]' in script
    assert 'metrics["starts_with_prompt"]' not in script
