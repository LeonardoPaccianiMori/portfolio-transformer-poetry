import json
from collections.abc import Mapping
from pathlib import Path

import torch

from sonnet_training.minerva_7b_staged_data import (
    Minerva7BStagedDataConfig,
    ReplaySampleConfig,
    build_replay_text_sample,
    prepare_minerva_7b_staged_data,
    select_even_windows,
)


class FakeTokenizer:
    eos_token_id = 2

    def __len__(self):
        return 51203

    def __call__(self, text, **kwargs):
        return {"input_ids": [3 + ord(character) % 100 for character in text]}


class FakeBatchEncoding(Mapping):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __getitem__(self, key):
        if key != "input_ids":
            raise KeyError(key)
        return self.input_ids

    def __iter__(self):
        return iter(("input_ids",))

    def __len__(self):
        return 1


class FakeBatchEncodingTokenizer(FakeTokenizer):
    def __call__(self, text, **kwargs):
        return FakeBatchEncoding([3 + ord(character) % 100 for character in text])


def test_select_even_windows_covers_stream_endpoints():
    windows = select_even_windows(range(30), context_length=4, window_count=3)

    assert windows == [list(range(5)), list(range(12, 17)), list(range(25, 30))]


def test_build_replay_text_sample_is_deterministic(tmp_path):
    source = tmp_path / "train.txt"
    source.write_text("".join(f"document {index:05d}\n" for index in range(2000)))
    config = ReplaySampleConfig(target_bytes=2048, chunk_count=8)

    first = build_replay_text_sample(
        source_path=source,
        output_path=tmp_path / "first.txt",
        report_path=tmp_path / "first.json",
        config=config,
    )
    second = build_replay_text_sample(
        source_path=source,
        output_path=tmp_path / "second.txt",
        report_path=tmp_path / "second.json",
        config=config,
    )

    assert first["output_sha256"] == second["output_sha256"]
    assert (tmp_path / "first.txt").read_bytes() == (tmp_path / "second.txt").read_bytes()


def test_prepare_staged_data_preserves_source_splits_and_writes_int32(tmp_path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    sources = []
    for index, text in enumerate(("a" * 120, "b" * 220), start=1):
        path = source_dir / f"source_{index}.txt"
        path.write_text(text)
        sources.append({"source_id": f"source_{index}", "source_path": str(path)})
    report_path = tmp_path / "mixture.json"
    report_path.write_text(json.dumps({"sources": sources}))
    replay_path = tmp_path / "replay.txt"
    replay_path.write_text("modern replay " * 100)
    preservation_path = tmp_path / "preservation.txt"
    preservation_path.write_text("held out modern validation " * 200)
    output_dir = tmp_path / "encoded"
    config = Minerva7BStagedDataConfig(
        mixture_report_path=str(report_path),
        replay_text_path=str(replay_path),
        preservation_text_path=str(preservation_path),
        output_dir=str(output_dir),
        context_length=512,
        preservation_window_count=2,
    )

    report = prepare_minerva_7b_staged_data(
        repo_root=tmp_path,
        config=config,
        tokenizer=FakeTokenizer(),
    )

    train = torch.load(output_dir / "historical_train.pt", weights_only=True)
    validation = torch.load(
        output_dir / "historical_validation.pt", weights_only=True
    )
    preservation = torch.load(
        output_dir / "modern_preservation_validation.pt", weights_only=True
    )
    assert report["source_count"] == 2
    assert train.dtype == torch.int32 and validation.dtype == torch.int32
    assert train.tolist().count(2) == 2
    assert validation.tolist().count(2) == 2
    assert preservation.shape == (2, 513)


def test_prepare_staged_data_accepts_hugging_face_style_mapping(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("historical text " * 60)
    mixture = tmp_path / "mixture.json"
    mixture.write_text(json.dumps({
        "sources": [{"source_id": "source", "source_path": str(source)}]
    }))
    replay = tmp_path / "replay.txt"
    replay.write_text("modern replay " * 60)
    preservation = tmp_path / "preservation.txt"
    preservation.write_text("modern validation " * 120)

    report = prepare_minerva_7b_staged_data(
        repo_root=tmp_path,
        config=Minerva7BStagedDataConfig(
            mixture_report_path=str(mixture),
            replay_text_path=str(replay),
            preservation_text_path=str(preservation),
            output_dir=str(tmp_path / "encoded"),
            preservation_window_count=2,
        ),
        tokenizer=FakeBatchEncodingTokenizer(),
    )

    assert report["source_count"] == 1
