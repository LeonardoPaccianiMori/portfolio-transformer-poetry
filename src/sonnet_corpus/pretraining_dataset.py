"""Encode the broader Italian pretraining corpus into PyTorch token tensors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from .bpe import BytePairEncodingTokenizer
from .pretraining_tokenizer import encode_text_by_pretoken


@dataclass(frozen=True)
class PretrainingDatasetConfig:
    """Configuration for building local pretraining token tensors."""

    processed_sources_dir: Path = Path("data/local/pretraining/processed/sources")
    tokenizer_path: Path = Path("data/local/pretraining/tokenizers/bpe_8000.json")
    output_dir: Path = Path("data/local/pretraining/encoded")
    report_path: Path = Path("data/local/pretraining/encoded/bpe_8000_report.json")
    validation_fraction: float = 0.01
    document_separator: str = "<|endoftext|>"
    train_filename: str = "bpe_8000_train.pt"
    validation_filename: str = "bpe_8000_validation.pt"


def build_pretraining_token_dataset(
    config: PretrainingDatasetConfig,
) -> dict[str, Any]:
    """Encode processed broader-corpus sources and save train/validation tensors."""

    started_at = _utc_now()
    _validate_validation_fraction(config.validation_fraction)
    source_paths = list_processed_source_paths(config.processed_sources_dir)
    tokenizer = BytePairEncodingTokenizer.load(config.tokenizer_path)
    separator_ids = tokenizer.encode(config.document_separator)
    if len(separator_ids) != 1:
        raise ValueError(
            "document_separator must encode to exactly one token for this dataset"
        )

    train_token_ids: list[int] = []
    validation_token_ids: list[int] = []
    source_reports: list[dict[str, Any]] = []
    for source_index, source_path in enumerate(source_paths):
        text = _read_non_empty_source(source_path)
        token_ids = encode_text_by_pretoken(text, tokenizer)
        train_ids, validation_ids = split_source_token_ids(
            token_ids,
            validation_fraction=config.validation_fraction,
            source_id=source_path.stem,
        )

        if source_index > 0:
            train_token_ids.extend(separator_ids)
            validation_token_ids.extend(separator_ids)
        train_token_ids.extend(train_ids)
        validation_token_ids.extend(validation_ids)
        source_reports.append({
            "source_id": source_path.stem,
            "source_path": _portable_path(source_path),
            "total_tokens": len(token_ids),
            "train_tokens": len(train_ids),
            "validation_tokens": len(validation_ids),
        })

    train_tensor = torch.tensor(train_token_ids, dtype=torch.long)
    validation_tensor = torch.tensor(validation_token_ids, dtype=torch.long)
    train_path = config.output_dir / config.train_filename
    validation_path = config.output_dir / config.validation_filename
    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_tensor, train_path)
    torch.save(validation_tensor, validation_path)

    report = {
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "processed_sources_dir": _portable_path(config.processed_sources_dir),
        "tokenizer_path": _portable_path(config.tokenizer_path),
        "output_dir": _portable_path(config.output_dir),
        "train_path": _portable_path(train_path),
        "validation_path": _portable_path(validation_path),
        "validation_fraction": config.validation_fraction,
        "document_separator": config.document_separator,
        "source_count": len(source_paths),
        "train_tokens": int(train_tensor.numel()),
        "validation_tokens": int(validation_tensor.numel()),
        "total_tokens": int(train_tensor.numel() + validation_tensor.numel()),
        "train_dtype": str(train_tensor.dtype),
        "validation_dtype": str(validation_tensor.dtype),
        "sources": source_reports,
    }
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def load_pretraining_token_splits(
    output_dir: Path,
    *,
    train_filename: str = "bpe_8000_train.pt",
    validation_filename: str = "bpe_8000_validation.pt",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load local pretraining train/validation token tensors."""

    train_tokens = torch.load(output_dir / train_filename)
    validation_tokens = torch.load(output_dir / validation_filename)
    _validate_loaded_tensor(train_tokens, name="train")
    _validate_loaded_tensor(validation_tokens, name="validation")
    return train_tokens, validation_tokens


def list_processed_source_paths(processed_sources_dir: Path) -> list[Path]:
    """Return deterministic processed source paths for encoding."""

    if not processed_sources_dir.is_dir():
        raise FileNotFoundError(
            f"processed sources directory does not exist: {processed_sources_dir}"
        )

    source_paths = sorted(processed_sources_dir.glob("*.txt"))
    if not source_paths:
        raise ValueError(
            f"processed sources directory has no .txt files: {processed_sources_dir}"
        )
    return source_paths


def split_source_token_ids(
    token_ids: list[int],
    *,
    validation_fraction: float,
    source_id: str,
) -> tuple[list[int], list[int]]:
    """Split one source into train tokens and final validation tokens."""

    _validate_validation_fraction(validation_fraction)
    if len(token_ids) < 2:
        raise ValueError(f"source has fewer than two tokens: {source_id}")

    validation_count = max(1, int(len(token_ids) * validation_fraction))
    validation_count = min(validation_count, len(token_ids) - 1)
    split_index = len(token_ids) - validation_count
    return token_ids[:split_index], token_ids[split_index:]


def _read_non_empty_source(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"processed source file is empty: {path}")
    return text


def _validate_validation_fraction(validation_fraction: float) -> None:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be greater than 0 and less than 1")


def _validate_loaded_tensor(tensor: object, *, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} tokens file did not contain a torch.Tensor")
    if tensor.ndim != 1:
        raise ValueError(f"{name} tokens must be a 1D tensor")
    if tensor.dtype != torch.long:
        raise ValueError(f"{name} tokens must use dtype torch.long")


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)
