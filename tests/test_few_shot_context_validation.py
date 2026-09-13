import csv
import hashlib
from pathlib import Path

from sonnet_analysis.few_shot_context_validation import (
    build_context_examples,
    context_prompt,
    output_name,
)
from sonnet_training.form_targeted_data import load_train_rows


class FakeTokenizer:
    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True
    ) -> str:
        rendered = "|".join(
            f"{message['role']}:{message['content']}" for message in messages
        )
        return f"{rendered}|assistant:"


def write_fixture(tmp_path: Path, count: int = 2) -> Path:
    lines = []
    for index in range(count):
        lines.extend([f"sonetto {index} verso {line} vita" for line in range(14)])
    text = ("\n".join(lines) + "\n").encode("utf-8")
    storage = tmp_path / "sonnets.txt"
    storage.write_bytes(text)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "unit_id",
                "v8_split",
                "v8_rendering_policy",
                "storage_path",
                "byte_start",
                "byte_end",
                "logical_sha256",
                "line_count",
            ],
        )
        writer.writeheader()
        for index in range(count):
            start = index * len(text) // count
            end = (index + 1) * len(text) // count
            chunk = text[start:end]
            writer.writerow(
                {
                    "unit_id": f"train-{index}",
                    "v8_split": "train",
                    "v8_rendering_policy": "derived_source_lines_4+4+3+3_v1",
                    "storage_path": "sonnets.txt",
                    "byte_start": start,
                    "byte_end": end,
                    "logical_sha256": hashlib.sha256(chunk).hexdigest(),
                    "line_count": 14,
                }
            )
    return manifest


def test_build_context_examples_builds_plan_and_text(tmp_path):
    manifest = write_fixture(tmp_path, count=2)
    rows = load_train_rows(manifest)
    examples = build_context_examples(rows, tmp_path, limit=2)
    assert len(examples) == 2
    assert examples[0]["words"][0] == "vita"
    assert len(examples[0]["words"]) == 14


def test_context_prompt_places_examples_before_the_final_user_turn():
    examples = [
        {
            "opening_line": "Primo verso di prova vita",
            "words": ["vita"] * 14,
            "text": "Primo verso di prova vita",
        }
    ]
    prompt = context_prompt(
        FakeTokenizer(),
        "Chi vuole aver gioiosa vita intera",
        ["segno"] * 14,
        examples,
    )
    assert prompt.count("user:") == 2
    assert prompt.count("assistant:") == 2
    assert "Chi vuole aver gioiosa vita intera" in prompt


def test_output_name_is_condition_specific():
    assert output_name("zeroshot", "p1", 5200) != output_name("fewshot1", "p1", 5200)
    assert output_name("zeroshot", "p1", 5200) == output_name("zeroshot", "p1", 5200)
