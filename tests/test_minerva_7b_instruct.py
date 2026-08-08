import json
from pathlib import Path

import pytest

from sonnet_evaluation.minerva_7b_instruct import (
    build_minerva_7b_instruct_prompt,
    minerva_7b_load_metadata,
    write_minerva_7b_instruct_baseline_scaffolds,
)
from sonnet_training.minerva_7b_qlora import (
    MINERVA_7B_INSTRUCT_MODEL_ID,
    MINERVA_7B_INSTRUCT_REVISION,
)


class ChatTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        assert messages[0]["role"] == "user"
        return "<chat>assistant\n"


def test_minerva_7b_prompt_uses_chat_template_and_exact_opening_prefill():
    prompt = build_minerva_7b_instruct_prompt(ChatTokenizer(), "Primo verso")

    assert prompt.startswith("<chat>assistant")
    assert prompt.endswith("Primo verso\n")

    with pytest.raises(ValueError, match="exactly one"):
        build_minerva_7b_instruct_prompt(ChatTokenizer(), "Primo\nverso")


def test_minerva_7b_load_modes_distinguish_nf4_from_unquantized_fp16():
    nf4 = minerva_7b_load_metadata("nf4")
    fp16 = minerva_7b_load_metadata("fp16")

    assert nf4["weight_loading"]["quantized"] is True
    assert nf4["weight_loading"]["quant_type"] == "nf4"
    assert fp16["weight_loading"] == {
        "quantized": False,
        "parameter_dtype": "float16",
        "compute_dtype": "float16",
    }
    assert fp16["model_variant"] == "minerva_7b_instruct_fp16"

    with pytest.raises(ValueError, match="unsupported"):
        minerva_7b_load_metadata("int8")


def test_minerva_7b_scaffolds_score_exact_eight_outputs(tmp_path: Path):
    generation_dir = tmp_path / "generation"
    generation_dir.mkdir()
    generated_files = []
    for index in range(8):
        output_path = generation_dir / f"output_{index}.txt"
        output_path.write_text(
            f"Opening {index}\n" + "\n".join(["Verso"] * 13) + "\n",
            encoding="utf-8",
        )
        generated_files.append({
            "prompt_id": f"prompt_{index}__seed_4242",
            "source_prompt_id": f"prompt_{index}",
            "poem_id": f"poem_{index}",
            "author": f"Author {index}",
            "prompt_text": f"Opening {index}",
            "opening_line": f"Opening {index}",
            "path": str(output_path),
            "seed": 4242,
            "stop_reason": "target_lines",
            "completed_continuation_lines": 13,
        })
    (generation_dir / "metadata.json").write_text(
        json.dumps({
            "generation_format": "task_format_opening_line_continuation",
            "model_id": MINERVA_7B_INSTRUCT_MODEL_ID,
            "revision": MINERVA_7B_INSTRUCT_REVISION,
            "total_line_target": 14,
            "stop_text": "<eot>",
            "generated_files": generated_files,
        }),
        encoding="utf-8",
    )
    automatic_path = tmp_path / "automatic.md"
    review_path = tmp_path / "review.md"

    rows = write_minerva_7b_instruct_baseline_scaffolds(
        generation_dir=generation_dir,
        automatic_report_path=automatic_path,
        review_path=review_path,
    )

    assert len(rows) == 8
    assert all(row["automatic_control_pass"] for row in rows)
    assert "Exact-opening controlled forms: **8/8**" in automatic_path.read_text()
    assert "Generally grammatical Italian: TODO yes/no" in review_path.read_text()
