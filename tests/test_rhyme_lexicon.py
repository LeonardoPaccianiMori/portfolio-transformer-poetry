import csv
import hashlib
import json
from pathlib import Path

import pytest

from sonnet_evaluation.rhyme_lexicon import (
    build_lexicon,
    choose_endings,
    line_final_word,
    load_lexicon,
    sonnet_endings,
    write_lexicon,
)
from sonnet_training.form_targeted_data import load_train_rows


def make_lexicon() -> dict:
    entries = []
    for index, key in enumerate(
        ["ita", "ore", "are", "ente", "ando", "ita2", "ore2", "are2"]
    ):
        entries.append(
            {
                "key": key,
                "soft_key": key,
                "count": 100 - index,
                "words": [f"w{key}{n}" for n in range(6)],
            }
        )
    return {"lexicon_version": "sonnet_rhyme_lexicon_v1", "entries": entries}


def test_line_final_word_handles_punctuation_and_apostrophes():
    assert line_final_word("Nel mezzo del cammin di nostra vita") == "vita"
    assert line_final_word("soave pien d'amore,") == "d'amore"
    assert line_final_word("   ") is None


def test_sonnet_endings_extracts_fourteen_words():
    text = "\n".join([f"verso numero {index} vita" for index in range(14)])
    endings = sonnet_endings(text)
    assert len(endings) == 14
    assert set(endings) == {"vita"}


def test_build_lexicon_counts_keys(tmp_path):
    storage = tmp_path / "sonnets.txt"
    lines = [f"verso {index} core" for index in range(7)] + [
        f"verso {index} vita" for index in range(7)
    ]
    text = ("\n".join(lines) + "\n").encode("utf-8")
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
    rows = load_train_rows(manifest)
    payload = build_lexicon(rows, tmp_path, manifest_path=manifest)
    keys = {entry["key"]: entry for entry in payload["entries"]}
    assert keys["ore"]["count"] == 7
    assert keys["ita"]["count"] == 7
    assert "core" in keys["ore"]["words"]


def test_choose_endings_is_deterministic_and_class_consistent():
    lexicon = make_lexicon()
    scheme = "ABBAABBACDECDE"
    first = choose_endings(scheme, lexicon, seed=6200)
    second = choose_endings(scheme, lexicon, seed=6200)
    assert first == second
    by_class = {}
    for row in first:
        by_class.setdefault(row["class"], set()).add((row["key"], row["word"]))
    assert all(len(values) == 1 for values in by_class.values())
    assert len({row["key"] for row in first}) == len(set(scheme))
    assert len({row["word"] for row in first}) == len(set(scheme))


def test_write_and_load_lexicon_round_trip(tmp_path):
    lexicon = make_lexicon()
    path = tmp_path / "lexicon.json"
    write_lexicon(path, lexicon)
    loaded = load_lexicon(path)
    assert loaded["entries"][0]["key"] == "ita"
    loaded["lexicon_version"] = "bad"
    path.write_text(json.dumps(loaded), encoding="utf-8")
    with pytest.raises(ValueError):
        load_lexicon(path)
