"""Encode the V8 corrected sonnet train split for the form-targeted pilot.

The encoder reads the 4+4+3+3 derived view by byte range, verifies the frozen
logical hash of every selected record, appends the tokenizer EOS token after
each sonnet, and writes one int32 token shard plus an auditable report.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

DATA_VERSION = "form_targeted_v8_train_v1"
INT32_BYTES = 4
REQUIRED_COLUMNS = (
    "unit_id",
    "v8_split",
    "v8_rendering_policy",
    "storage_path",
    "byte_start",
    "byte_end",
    "logical_sha256",
    "line_count",
)
Progress = Callable[[str], None]


def load_train_rows(
    manifest_path: Path, *, split: str = "train"
) -> list[dict[str, str]]:
    import csv

    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("manifest has no header")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"manifest lacks required columns: {missing}")
        rows = [row for row in reader if row["v8_split"] == split]
    if not rows:
        raise ValueError(f"manifest has no rows for split {split!r}")
    return rows


def read_sonnet_text(root: Path, row: Mapping[str, str]) -> str:
    path = root / row["storage_path"]
    start = int(row["byte_start"])
    end = int(row["byte_end"])
    if start < 0 or end <= start:
        raise ValueError(f"invalid byte range for {row['unit_id']}")
    with path.open("rb") as handle:
        handle.seek(start)
        data = handle.read(end - start)
    if hashlib.sha256(data).hexdigest() != row["logical_sha256"]:
        raise ValueError(f"logical hash mismatch for {row['unit_id']}")
    return data.decode("utf-8")


def encode_train_split(
    rows: Iterable[Mapping[str, str]],
    root: Path,
    *,
    tokenizer: Any,
    progress: Progress | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    eos_token_id = int(tokenizer.eos_token_id)
    tokens: list[int] = []
    records: list[dict[str, Any]] = []
    row_list = list(rows)
    for index, row in enumerate(row_list, start=1):
        text = read_sonnet_text(root, row)
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        if not ids:
            raise ValueError(f"empty tokenization for {row['unit_id']}")
        tokens.extend(ids)
        tokens.append(eos_token_id)
        records.append(
            {
                "unit_id": row["unit_id"],
                "token_count": len(ids) + 1,
                "character_count": len(text),
                "line_count": int(row["line_count"]),
                "sha256": row["logical_sha256"],
            }
        )
        if progress and index % 2000 == 0:
            progress(f"encoded {index}/{len(row_list)} sonnets")
    return np.asarray(tokens, dtype=np.int32), records


def write_token_shard(
    output_dir: Path,
    tokens: np.ndarray,
    records: list[dict[str, Any]],
    *,
    manifest_path: Path,
    tokenizer_sha256: str,
    max_sequence_tokens: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / "tokens-00000.int32.bin"
    tokens.tofile(shard_path)
    shard_sha256 = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    report = {
        "data_version": DATA_VERSION,
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "tokenizer_sha256": tokenizer_sha256,
        "max_sequence_tokens": max_sequence_tokens,
        "sonnet_count": len(records),
        "token_count": int(tokens.size),
        "eos_per_sonnet": True,
        "shard_path": str(shard_path),
        "shard_sha256": shard_sha256,
        "records": records,
    }
    (output_dir / "encode_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def load_token_shard(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) % INT32_BYTES:
        raise ValueError("token shard is not a multiple of four bytes")
    count = len(data) // INT32_BYTES
    return np.asarray(struct.unpack(f"<{count}i", data), dtype=np.int32)
