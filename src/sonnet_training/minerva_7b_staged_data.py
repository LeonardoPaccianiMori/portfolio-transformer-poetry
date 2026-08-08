"""Prepare deterministic local data for staged Minerva 7B LoRA adaptation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from sonnet_training.minerva_7b_qlora import (
    MINERVA_7B_INSTRUCT_MODEL_ID,
    MINERVA_7B_INSTRUCT_REVISION,
)


STAGED_DATA_VERSION = "minerva_7b_historical_dapt_v1"


@dataclass(frozen=True)
class Minerva7BStagedDataConfig:
    """Freeze the source and split policy used by the two-stage run."""

    model_id: str = MINERVA_7B_INSTRUCT_MODEL_ID
    revision: str = MINERVA_7B_INSTRUCT_REVISION
    mixture_report_path: str = (
        "reports/pretraining_historical_italian_v2_mixture_report.json"
    )
    replay_text_path: str = "data/local/minerva_7b_staged/replay_train.txt"
    preservation_text_path: str = (
        "data/local/pretraining/paisa_historical_rescue_v1/"
        "paisa_validation_tokenizable.txt"
    )
    cache_dir: str = "data/local/minerva_qlora/huggingface"
    output_dir: str = "data/local/minerva_7b_staged/encoded"
    context_length: int = 512
    historical_validation_fraction: float = 0.01
    preservation_window_count: int = 256
    seed: int = 1337


@dataclass(frozen=True)
class ReplaySampleConfig:
    """Describe deterministic byte-window sampling from PAISÀ training text."""

    target_bytes: int = 8 * 1024 * 1024
    chunk_count: int = 256


def build_replay_text_sample(
    *,
    source_path: Path,
    output_path: Path,
    report_path: Path,
    config: ReplaySampleConfig = ReplaySampleConfig(),
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Sample evenly spaced, newline-aligned PAISÀ training windows."""
    if config.target_bytes <= 0 or config.chunk_count <= 0:
        raise ValueError("replay sample sizes must be greater than zero")
    if not source_path.is_file():
        raise FileNotFoundError(f"PAISÀ training text does not exist: {source_path}")
    source_size = source_path.stat().st_size
    if source_size <= config.target_bytes:
        raise ValueError("PAISÀ source must be larger than the replay sample")

    chunk_size = math.ceil(config.target_bytes / config.chunk_count)
    maximum_offset = source_size - chunk_size - 1
    chunks: list[bytes] = []
    with source_path.open("rb") as handle:
        for index in range(config.chunk_count):
            offset = round(index * maximum_offset / max(1, config.chunk_count - 1))
            handle.seek(offset)
            if offset:
                handle.readline()
            chunk = handle.read(chunk_size)
            chunk += handle.readline()
            chunk.decode("utf-8")
            chunks.append(chunk.rstrip() + b"\n")
            if (index + 1) % 32 == 0 or index + 1 == config.chunk_count:
                _report(progress, f"sampled replay chunk {index + 1}/{config.chunk_count}")

    output = b"\n".join(chunks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output)
    report = {
        "sample_version": "paisa_even_byte_windows_v1",
        "source_path": str(source_path),
        "source_size_bytes": source_size,
        "source_build_report_path": (
            "data/local/pretraining/paisa/paisa_modern_italian_v1/build_report.json"
        ),
        "output_path": str(output_path),
        "output_size_bytes": len(output),
        "output_sha256": _sha256(output_path),
        "config": asdict(config),
        "license_lineage": "PAISÀ CC BY-NC-SA; local non-commercial training data",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def prepare_minerva_7b_staged_data(
    *,
    repo_root: Path,
    config: Minerva7BStagedDataConfig = Minerva7BStagedDataConfig(),
    progress: Callable[[str], None] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Tokenize historical, replay, and preservation text with Minerva."""
    validate_staged_data_config(config)
    mixture_path = _resolve(repo_root, config.mixture_report_path)
    replay_path = _resolve(repo_root, config.replay_text_path)
    preservation_path = _resolve(repo_root, config.preservation_text_path)
    for path in (mixture_path, replay_path, preservation_path):
        if not path.is_file():
            raise FileNotFoundError(f"required staged-data input does not exist: {path}")

    if tokenizer is None:
        from transformers import AutoTokenizer

        _report(progress, "loading pinned Minerva tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            cache_dir=_resolve(repo_root, config.cache_dir),
        )
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token_id, int) or eos_token_id < 0:
        raise ValueError("Minerva tokenizer must define a non-negative eos_token_id")

    mixture = json.loads(mixture_path.read_text(encoding="utf-8"))
    sources = mixture.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("historical mixture report must contain sources")

    historical_train: list[int] = []
    historical_validation: list[int] = []
    source_rows: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        source_path_value = source.get("source_path") if isinstance(source, dict) else None
        source_id = source.get("source_id") if isinstance(source, dict) else None
        if not isinstance(source_path_value, str) or not isinstance(source_id, str):
            raise ValueError("historical source row is missing source_path or source_id")
        source_path = _resolve(repo_root, source_path_value)
        text = source_path.read_text(encoding="utf-8")
        token_ids = _token_ids(tokenizer, text)
        if len(token_ids) < 2:
            raise ValueError(f"historical source tokenized too short: {source_id}")
        split_index = max(
            1,
            min(
                len(token_ids) - 1,
                math.floor(len(token_ids) * (1.0 - config.historical_validation_fraction)),
            ),
        )
        train_ids = token_ids[:split_index]
        validation_ids = token_ids[split_index:]
        historical_train.extend([*train_ids, eos_token_id])
        historical_validation.extend([*validation_ids, eos_token_id])
        source_rows.append({
            "source_id": source_id,
            "source_path": source_path_value,
            "source_sha256": _sha256(source_path),
            "total_tokens": len(token_ids),
            "train_tokens": len(train_ids) + 1,
            "validation_tokens": len(validation_ids) + 1,
        })
        _report(progress, f"tokenized historical source {index}/{len(sources)}: {source_id}")

    _report(progress, "tokenizing deterministic PAISÀ replay sample")
    replay_ids = [*_token_ids(tokenizer, replay_path.read_text(encoding="utf-8")), eos_token_id]
    _report(progress, "tokenizing PAISÀ preservation validation text")
    preservation_ids = _token_ids(
        tokenizer,
        preservation_path.read_text(encoding="utf-8"),
    )
    preservation_windows = select_even_windows(
        preservation_ids,
        context_length=config.context_length,
        window_count=config.preservation_window_count,
    )

    output_dir = _resolve(repo_root, config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "historical_train": output_dir / "historical_train.pt",
        "historical_validation": output_dir / "historical_validation.pt",
        "modern_replay_train": output_dir / "modern_replay_train.pt",
        "modern_preservation_validation": (
            output_dir / "modern_preservation_validation.pt"
        ),
    }
    torch.save(torch.tensor(historical_train, dtype=torch.int32), artifacts["historical_train"])
    torch.save(
        torch.tensor(historical_validation, dtype=torch.int32),
        artifacts["historical_validation"],
    )
    torch.save(torch.tensor(replay_ids, dtype=torch.int32), artifacts["modern_replay_train"])
    torch.save(
        torch.tensor(preservation_windows, dtype=torch.int32),
        artifacts["modern_preservation_validation"],
    )

    report = {
        "data_version": STAGED_DATA_VERSION,
        "config": asdict(config),
        "model_id": config.model_id,
        "revision": config.revision,
        "tokenizer_size": len(tokenizer),
        "eos_token_id": eos_token_id,
        "mixture_report_path": config.mixture_report_path,
        "mixture_report_sha256": _sha256(mixture_path),
        "source_count": len(source_rows),
        "sources": source_rows,
        "historical_train_tokens": len(historical_train),
        "historical_validation_tokens": len(historical_validation),
        "modern_replay_train_tokens": len(replay_ids),
        "modern_preservation_window_count": len(preservation_windows),
        "modern_preservation_tokens": sum(len(row) for row in preservation_windows),
        "artifacts": {
            name: {
                "path": str(path.relative_to(repo_root)),
                "sha256": _sha256(path),
            }
            for name, path in artifacts.items()
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _report(progress, f"wrote staged-data report: {report_path}")
    return report


def validate_staged_data_config(config: Minerva7BStagedDataConfig) -> None:
    if config.model_id != MINERVA_7B_INSTRUCT_MODEL_ID:
        raise ValueError("staged data is locked to Minerva 7B Instruct")
    if config.revision != MINERVA_7B_INSTRUCT_REVISION:
        raise ValueError("staged data revision does not match the pinned model")
    if config.context_length != 512:
        raise ValueError("staged data context length is locked to 512")
    if not 0.0 < config.historical_validation_fraction < 0.5:
        raise ValueError("historical_validation_fraction must be between zero and 0.5")
    if config.preservation_window_count <= 0:
        raise ValueError("preservation_window_count must be greater than zero")


def select_even_windows(
    token_ids: Sequence[int], *, context_length: int, window_count: int
) -> list[list[int]]:
    """Select fixed input-plus-target windows across a held-out token stream."""
    window_length = context_length + 1
    if context_length <= 0 or window_count <= 0:
        raise ValueError("window dimensions must be greater than zero")
    if len(token_ids) < window_length:
        raise ValueError("token stream is too short for one preservation window")
    maximum_start = len(token_ids) - window_length
    starts = [
        round(index * maximum_start / max(1, window_count - 1))
        for index in range(window_count)
    ]
    if len(set(starts)) != len(starts):
        raise ValueError("token stream is too short for distinct preservation windows")
    return [list(token_ids[start:start + window_length]) for start in starts]


def load_staged_tensor(path: Path, *, dimensions: int) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != dimensions:
        raise ValueError(f"staged tensor must have {dimensions} dimensions: {path}")
    if tensor.dtype != torch.int32:
        raise ValueError(f"staged tensor must use torch.int32: {path}")
    return tensor


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    token_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if not isinstance(token_ids, list) or any(not isinstance(item, int) for item in token_ids):
        raise ValueError("Minerva tokenizer must return a list of input_ids")
    return token_ids


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _report(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
