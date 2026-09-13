import csv
import hashlib
from pathlib import Path

from sonnet_training.form_targeted_data import load_train_rows
from sonnet_training.plan_following_sft import (
    build_examples,
    pad_batch,
    split_examples,
)


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 2

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True
    ) -> str:
        return f"U:{messages[0]['content']}|A:"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) % 1000 for character in text]}


def write_fixture(tmp_path: Path) -> Path:
    lines = [f"verso numero {index} vita" for index in range(14)]
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
        writer.writerow(
            {
                "unit_id": "train-1",
                "v8_split": "train",
                "v8_rendering_policy": "derived_source_lines_4+4+3+3_v1",
                "storage_path": "sonnets.txt",
                "byte_start": 0,
                "byte_end": len(text),
                "logical_sha256": hashlib.sha256(text).hexdigest(),
                "line_count": 14,
            }
        )
    return manifest


def test_build_examples_keeps_targets_and_eos(tmp_path):
    manifest = write_fixture(tmp_path)
    rows = load_train_rows(manifest)
    examples, skipped = build_examples(
        rows, tmp_path, tokenizer=FakeTokenizer(), max_sequence_tokens=100000
    )
    assert len(examples) == 1
    example = examples[0]
    assert example["planned_words"][0] == "vita"
    assert example["target_ids"][-1] == 2
    assert skipped["malformed"] == 0


def test_build_examples_honours_exclusions(tmp_path):
    manifest = write_fixture(tmp_path)
    with manifest.open("a", encoding="utf-8", newline="") as handle:
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
        writer.writerow(
            {
                "unit_id": "train-2",
                "v8_split": "train",
                "v8_rendering_policy": "derived_source_lines_4+4+3+3_v1",
                "storage_path": "sonnets.txt",
                "byte_start": 0,
                "byte_end": 651,
                "logical_sha256": hashlib.sha256(
                    (tmp_path / "sonnets.txt").read_bytes()
                ).hexdigest(),
                "line_count": 14,
            }
        )
    rows = load_train_rows(manifest)
    examples, skipped = build_examples(
        rows,
        tmp_path,
        tokenizer=FakeTokenizer(),
        max_sequence_tokens=100000,
        excluded_unit_ids=["train-1"],
    )
    assert len(examples) == 1
    assert examples[0]["unit_id"] == "train-2"
    assert skipped["excluded"] == 1


def test_pad_batch_masks_prompt_and_padding():
    examples = [
        {"prompt_ids": [1, 2], "target_ids": [3, 4]},
        {"prompt_ids": [5], "target_ids": [6]},
    ]
    input_ids, attention_mask, labels = pad_batch(examples, pad_token_id=0)
    assert labels[0].tolist() == [-100, -100, 3, 4]
    assert labels[1].tolist() == [-100, 6, -100, -100]
    assert attention_mask[1].tolist() == [1, 1, 0, 0]
    assert input_ids[1].tolist() == [5, 6, 0, 0]


def test_split_examples_is_disjoint_and_deterministic():
    examples = [{"unit_id": f"u{index}"} for index in range(100)]
    first_train, first_validation = split_examples(
        examples, validation_fraction=0.1, seed=7
    )
    second_train, second_validation = split_examples(
        examples, validation_fraction=0.1, seed=7
    )
    assert [row["unit_id"] for row in first_validation] == [
        row["unit_id"] for row in second_validation
    ]
    assert len(first_validation) == 10
    assert {row["unit_id"] for row in first_train}.isdisjoint(
        {row["unit_id"] for row in first_validation}
    )
