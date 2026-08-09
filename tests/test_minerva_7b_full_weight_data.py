import json
from dataclasses import replace
from pathlib import Path

import torch

from sonnet_training.minerva_7b_full_weight_data import (
    Minerva7BFullWeightDataConfig,
    load_full_weight_calibration_windows,
    load_int32_shard,
    prepare_minerva_7b_full_weight_data,
)


class FakeTokenizer:
    eos_token_id = 2

    def __len__(self):
        return 1024

    def __call__(self, text, **kwargs):
        return {"input_ids": [3 + ord(character) % 100 for character in text]}


def _write_document_split(path: Path, documents: list[str]) -> None:
    path.write_text(
        "".join(f"{document}\n<|endoftext|>\n" for document in documents),
        encoding="utf-8",
    )


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_inputs(tmp_path: Path) -> Minerva7BFullWeightDataConfig:
    paisa_dir = tmp_path / "paisa"
    staged_dir = tmp_path / "staged"
    paisa_dir.mkdir()
    staged_dir.mkdir()
    paisa_train = ["a" * 400, "b" * 400]
    paisa_validation = ["c" * 700]
    historical_train = ["d" * 700, "e" * 700]
    historical_validation = ["f" * 700, "g" * 700]
    _write_document_split(paisa_dir / "train.txt", paisa_train)
    _write_document_split(paisa_dir / "validation.txt", paisa_validation)
    _write_document_split(staged_dir / "historical_train.txt", historical_train)
    _write_document_split(
        staged_dir / "historical_validation.txt", historical_validation
    )

    attribution_rows = [
        {
            "document_id": "paisa-train-1",
            "split": "train",
            "status": "retained",
            "text_sha256": _sha256_text(paisa_train[0]),
        },
        {
            "document_id": "paisa-validation-1",
            "split": "validation",
            "status": "retained",
            "text_sha256": _sha256_text(paisa_validation[0]),
        },
        {
            "document_id": "paisa-train-2",
            "split": "train",
            "status": "retained",
            "text_sha256": _sha256_text(paisa_train[1]),
        },
    ]
    attribution = paisa_dir / "document_attribution.jsonl"
    attribution.write_text(
        "".join(json.dumps(row) + "\n" for row in attribution_rows),
        encoding="utf-8",
    )
    paisa_report = tmp_path / "paisa_report.json"
    paisa_report.write_text(json.dumps({
        "local_artifacts": {
            "train_text_path": str(paisa_dir / "train.txt"),
            "validation_text_path": str(paisa_dir / "validation.txt"),
            "document_attribution_inventory_path": str(attribution),
        }
    }))
    curriculum = tmp_path / "curriculum.json"
    curriculum.write_text(json.dumps({
        "paisa": {"train_documents": 2, "validation_documents": 1},
        "historical": {
            "source_count": 2,
            "sources": [
                {"source_id": "historical-1"},
                {"source_id": "historical-2"},
            ],
        },
        "local_artifacts": {
            "historical_train_path": str(staged_dir / "historical_train.txt"),
            "historical_validation_path": str(
                staged_dir / "historical_validation.txt"
            ),
        },
    }))
    return Minerva7BFullWeightDataConfig(
        curriculum_report_path=str(curriculum),
        paisa_build_report_path=str(paisa_report),
        output_dir=str(tmp_path / "encoded"),
        public_report_path=str(tmp_path / "public_report.json"),
        shard_target_tokens=600,
        progress_interval_documents=1,
        checkpoint_interval_documents=1,
    )


def test_full_weight_data_streams_all_splits_and_preserves_eos(tmp_path):
    config = _write_inputs(tmp_path)

    report = prepare_minerva_7b_full_weight_data(
        repo_root=tmp_path,
        config=config,
        tokenizer=FakeTokenizer(),
    )

    assert report["status"] == "complete"
    assert report["totals"]["documents"] == 7
    assert report["totals"]["eos_tokens"] == 7
    assert len(report["splits"]) == 4
    train = next(row for row in report["splits"] if row["split_id"] == "paisa_train")
    assert len(train["shards"]) == 2
    assert train["historical_source_ids"] == []
    index_rows = [
        json.loads(line)
        for line in Path(train["document_index"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["source_id"] for row in index_rows] == [
        "paisa-train-1",
        "paisa-train-2",
    ]
    mapped = load_int32_shard(
        Path(train["shards"][0]["path"]),
        token_count=train["shards"][0]["token_count"],
    )
    assert mapped.dtype == torch.int32
    assert mapped[-1].item() == FakeTokenizer.eos_token_id

    calibration = load_full_weight_calibration_windows(
        Path(report["calibration_windows"]["path"])
    )
    assert calibration["training_windows"].shape == (5, 512)
    assert calibration["validation_windows"].shape == (2, 512)


def test_full_weight_data_resumes_at_completed_document_boundary(tmp_path):
    config = _write_inputs(tmp_path)
    interrupted = prepare_minerva_7b_full_weight_data(
        repo_root=tmp_path,
        config=replace(config, max_documents_per_split_run=1),
        tokenizer=FakeTokenizer(),
    )

    assert interrupted["status"] == "incomplete"
    assert interrupted["splits"][0]["documents"] == 1
    assert (tmp_path / "encoded/.paisa_train.checkpoint.json").is_file()

    report = prepare_minerva_7b_full_weight_data(
        repo_root=tmp_path,
        config=config,
        tokenizer=FakeTokenizer(),
    )

    assert report["status"] == "complete"
    assert report["totals"]["documents"] == 7
    assert not (tmp_path / "encoded/.paisa_train.checkpoint.json").exists()


def test_full_weight_data_recovers_an_uncheckpointed_shard_rollover(tmp_path):
    config = _write_inputs(tmp_path)
    prepare_minerva_7b_full_weight_data(
        repo_root=tmp_path,
        config=replace(config, max_documents_per_split_run=1),
        tokenizer=FakeTokenizer(),
    )
    output_dir = tmp_path / "encoded"
    active = output_dir / ".paisa_train-00000.int32.bin.part"
    active.replace(output_dir / "paisa_train-00000.int32.bin")
    (output_dir / ".paisa_train-00001.int32.bin.part").write_bytes(b"stale")

    report = prepare_minerva_7b_full_weight_data(
        repo_root=tmp_path,
        config=config,
        tokenizer=FakeTokenizer(),
    )

    assert report["status"] == "complete"
    train = next(row for row in report["splits"] if row["split_id"] == "paisa_train")
    assert train["documents"] == 2
    assert train["eos_tokens"] == 2
