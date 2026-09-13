import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest

from sonnet_training.form_targeted_data import (
    encode_train_split,
    load_token_shard,
    load_train_rows,
    read_sonnet_text,
    write_token_shard,
)


class FakeTokenizer:
    eos_token_id = 7

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in text]


def write_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    storage = tmp_path / "sonnets.txt"
    train_text = b"First line\nsecond line\n"
    quarantine_text = b"Quarantined text\n"
    storage.write_bytes(train_text + quarantine_text)
    train_hash = hashlib.sha256(train_text).hexdigest()
    quarantine_hash = hashlib.sha256(quarantine_text).hexdigest()
    manifest = tmp_path / "sonnets_manifest.csv"
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
                "byte_end": len(train_text),
                "logical_sha256": train_hash,
                "line_count": 2,
            }
        )
        writer.writerow(
            {
                "unit_id": "quarantine-1",
                "v8_split": "quarantine",
                "v8_rendering_policy": "derived_source_lines_4+4+3+3_v1",
                "storage_path": "sonnets.txt",
                "byte_start": len(train_text),
                "byte_end": len(train_text) + len(quarantine_text),
                "logical_sha256": quarantine_hash,
                "line_count": 1,
            }
        )
    return manifest, storage, {"train_text": train_text}


def test_load_train_rows_filters_split(tmp_path):
    manifest, _, _ = write_fixture(tmp_path)
    rows = load_train_rows(manifest)
    assert [row["unit_id"] for row in rows] == ["train-1"]


def test_load_train_rows_rejects_missing_columns(tmp_path):
    manifest = tmp_path / "bad.csv"
    manifest.write_text("unit_id,v8_split\nx,train\n", encoding="utf-8")
    with pytest.raises(ValueError, match="required columns"):
        load_train_rows(manifest)


def test_read_sonnet_text_verifies_hash(tmp_path):
    manifest, _, fixtures = write_fixture(tmp_path)
    row = load_train_rows(manifest)[0]
    assert read_sonnet_text(tmp_path, row).encode("utf-8") == fixtures["train_text"]
    row["logical_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        read_sonnet_text(tmp_path, row)


def test_encode_train_split_appends_eos(tmp_path):
    manifest, _, fixtures = write_fixture(tmp_path)
    rows = load_train_rows(manifest)
    tokens, records = encode_train_split(rows, tmp_path, tokenizer=FakeTokenizer())
    expected = [ord(char) for char in fixtures["train_text"].decode("utf-8")]
    assert tokens.tolist() == expected + [7]
    assert records[0]["token_count"] == len(expected) + 1
    assert records[0]["unit_id"] == "train-1"


def test_write_token_shard_round_trip(tmp_path):
    manifest, _, _ = write_fixture(tmp_path)
    rows = load_train_rows(manifest)
    tokens, records = encode_train_split(rows, tmp_path, tokenizer=FakeTokenizer())
    output_dir = tmp_path / "encoded"
    report = write_token_shard(
        output_dir,
        tokens,
        records,
        manifest_path=manifest,
        tokenizer_sha256="a" * 64,
        max_sequence_tokens=512,
    )
    assert report["sonnet_count"] == 1
    assert report["token_count"] == int(tokens.size)
    reloaded = load_token_shard(Path(report["shard_path"]))
    assert np.array_equal(reloaded, tokens)
