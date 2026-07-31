"""Stream the locked PAISA-historical curriculum into compact token files."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from array import array
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from .bpe import BytePairEncodingTokenizer, TokenPair, merge_token_pair
from .paisa_build import PAISA_DOCUMENT_SEPARATOR
from .pretraining_tokenizer import PRETOKEN_PATTERN


ProgressCallback = Callable[[str], None]
UINT16_BYTES = 2
UINT16_MAX = 65_535
ENCODING_FORMAT = "little_endian_uint16"
_HASH_CHUNK_SIZE = 1_048_576


@dataclass(frozen=True)
class PaisaHistoricalEncodingConfig:
    """Paths and bounded-memory settings for the rescue split encoder."""

    tokenizer_report_path: Path = Path(
        "reports/paisa_historical_rescue_v1_tokenizer_report.json"
    )
    curriculum_report_path: Path = Path(
        "reports/paisa_historical_rescue_v1_curriculum_report.json"
    )
    output_dir: Path = Path(
        "data/local/pretraining/paisa_historical_rescue_v1/encoded"
    )
    local_report_path: Path = Path(
        "data/local/pretraining/paisa_historical_rescue_v1/encoded_report.json"
    )
    public_report_path: Path = Path(
        "reports/paisa_historical_rescue_v1_encoded_report.json"
    )
    progress_interval_documents: int = 5_000
    checkpoint_interval_documents: int = 1_000
    pretoken_cache_entries: int = 250_000
    max_documents_per_split_run: int | None = None


@dataclass(frozen=True)
class SplitEncodingSpec:
    """One already-fixed text split and its expected document count."""

    split_id: str
    source_path: Path
    expected_documents: int
    output_path: Path
    metadata_path: Path
    checkpoint_path: Path


@dataclass
class SplitEncodingProgress:
    """Counters persisted at a safe output-file boundary."""

    source_offset: int = 0
    output_bytes: int = 0
    documents: int = 0
    characters: int = 0
    tokens: int = 0


class BoundedPretokenEncoder:
    """Encode repeated whitespace/non-whitespace spans with a bounded LRU cache."""

    def __init__(
        self,
        tokenizer: BytePairEncodingTokenizer,
        *,
        max_cache_entries: int,
    ) -> None:
        if max_cache_entries <= 0:
            raise ValueError("max_cache_entries must be greater than 0")
        self.tokenizer = tokenizer
        self.max_cache_entries = max_cache_entries
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(tokenizer.merges)
        }
        self.cache: OrderedDict[str, tuple[int, ...]] = OrderedDict()

    def encode(self, text: str) -> list[int]:
        """Encode ordinary text while reusing common word and whitespace spans."""

        if any(token in text for token in self.tokenizer.special_tokens):
            raise ValueError("document text contains a reserved special token")

        token_ids: list[int] = []
        for pretoken in PRETOKEN_PATTERN.findall(text):
            encoded = self.cache.get(pretoken)
            if encoded is None:
                encoded = tuple(
                    self.tokenizer.token_to_id[token]
                    for token in _encode_pretoken(pretoken, self.merge_ranks)
                )
                self.cache[pretoken] = encoded
                if len(self.cache) > self.max_cache_entries:
                    self.cache.popitem(last=False)
            else:
                self.cache.move_to_end(pretoken)
            token_ids.extend(encoded)
        return token_ids


def encode_paisa_historical_splits(
    config: PaisaHistoricalEncodingConfig,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Encode the four fixed curriculum splits without loading them into memory."""

    _validate_config(config)
    started_at = _utc_now()
    tokenizer_report = _read_json(config.tokenizer_report_path)
    curriculum_report = _read_json(config.curriculum_report_path)
    tokenizer_path = _tokenizer_path(tokenizer_report)
    tokenizer = BytePairEncodingTokenizer.load(tokenizer_path)
    _validate_tokenizer(tokenizer, tokenizer_report)
    tokenizer_sha256 = _file_sha256(tokenizer_path)
    specs = _build_split_specs(
        config=config,
        tokenizer_report=tokenizer_report,
        curriculum_report=curriculum_report,
    )

    _write_progress(
        progress,
        "start "
        f"splits={len(specs)} format={ENCODING_FORMAT} "
        f"cache_entries={config.pretoken_cache_entries}",
    )
    split_reports: list[dict[str, object]] = []
    for index, spec in enumerate(specs, start=1):
        _write_progress(
            progress,
            f"split {index}/{len(specs)} start: {spec.split_id}",
        )
        split_report = encode_split_to_uint16(
            spec=spec,
            tokenizer=tokenizer,
            tokenizer_sha256=tokenizer_sha256,
            progress_interval_documents=config.progress_interval_documents,
            checkpoint_interval_documents=config.checkpoint_interval_documents,
            pretoken_cache_entries=config.pretoken_cache_entries,
            max_documents=config.max_documents_per_split_run,
            progress=progress,
        )
        split_reports.append(split_report)
        if split_report["status"] != "complete":
            _write_progress(
                progress,
                f"paused after checkpoint: {spec.split_id}",
            )
            return {
                "curriculum_id": str(tokenizer_report["curriculum_id"]),
                "status": "incomplete",
                "started_at_utc": started_at,
                "finished_at_utc": _utc_now(),
                "splits": split_reports,
            }

    report = _make_report(
        config=config,
        started_at=started_at,
        tokenizer_report=tokenizer_report,
        tokenizer_path=tokenizer_path,
        tokenizer_sha256=tokenizer_sha256,
        split_reports=split_reports,
    )
    _write_json(config.local_report_path, report)
    _write_json(config.public_report_path, report)
    _write_progress(progress, f"wrote local report: {config.local_report_path}")
    _write_progress(progress, f"wrote public report: {config.public_report_path}")
    return report


def encode_split_to_uint16(
    *,
    spec: SplitEncodingSpec,
    tokenizer: BytePairEncodingTokenizer,
    tokenizer_sha256: str,
    progress_interval_documents: int,
    checkpoint_interval_documents: int,
    pretoken_cache_entries: int,
    max_documents: int | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Encode one separator-delimited split into an atomic uint16 artifact."""

    if spec.expected_documents <= 0:
        raise ValueError("expected_documents must be greater than 0")
    if progress_interval_documents <= 0:
        raise ValueError("progress_interval_documents must be greater than 0")
    if checkpoint_interval_documents <= 0:
        raise ValueError("checkpoint_interval_documents must be greater than 0")
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be greater than 0 when provided")
    if not spec.source_path.is_file():
        raise FileNotFoundError(f"split source file does not exist: {spec.source_path}")

    source_stat = spec.source_path.stat()
    existing = _load_completed_split(
        spec=spec,
        tokenizer_sha256=tokenizer_sha256,
        source_stat=source_stat,
    )
    if existing is not None:
        _write_progress(progress, f"{spec.split_id}: verified existing output")
        return existing

    part_path = spec.output_path.with_suffix(spec.output_path.suffix + ".part")
    state = _load_or_initialize_progress(
        spec=spec,
        part_path=part_path,
        tokenizer_sha256=tokenizer_sha256,
        source_stat=source_stat,
    )
    encoder = BoundedPretokenEncoder(
        tokenizer,
        max_cache_entries=pretoken_cache_entries,
    )
    separator_id = tokenizer.token_to_id[PAISA_DOCUMENT_SEPARATOR]
    invocation_documents = 0
    invocation_started = time.monotonic()
    invocation_start_offset = state.source_offset
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        spec.source_path.open("rb") as source,
        part_path.open("r+b" if part_path.exists() else "w+b") as output,
    ):
        source.seek(state.source_offset)
        output.truncate(state.output_bytes)
        output.seek(state.output_bytes)
        for document, next_source_offset in _iter_documents(source, spec.source_path):
            token_ids = encoder.encode(document)
            token_ids.append(separator_id)
            _validate_token_ids(token_ids, tokenizer.vocab_size)
            _write_uint16(output, token_ids)
            state.source_offset = next_source_offset
            state.output_bytes += len(token_ids) * UINT16_BYTES
            state.documents += 1
            state.characters += len(document)
            state.tokens += len(token_ids)
            invocation_documents += 1

            should_checkpoint = (
                state.documents % checkpoint_interval_documents == 0
                or (
                    max_documents is not None
                    and invocation_documents >= max_documents
                )
            )
            if should_checkpoint:
                _persist_progress(
                    spec=spec,
                    state=state,
                    output=output,
                    tokenizer_sha256=tokenizer_sha256,
                    source_stat=source_stat,
                )
            if (
                state.documents % progress_interval_documents == 0
                or state.documents == spec.expected_documents
            ):
                _report_split_progress(
                    spec=spec,
                    state=state,
                    source_size=source_stat.st_size,
                    invocation_started=invocation_started,
                    invocation_start_offset=invocation_start_offset,
                    invocation_documents=invocation_documents,
                    progress=progress,
                )
            if max_documents is not None and invocation_documents >= max_documents:
                return _incomplete_split_report(spec, state)

        _persist_progress(
            spec=spec,
            state=state,
            output=output,
            tokenizer_sha256=tokenizer_sha256,
            source_stat=source_stat,
        )

    if state.documents != spec.expected_documents:
        raise ValueError(
            f"{spec.split_id} document count mismatch: "
            f"expected {spec.expected_documents}, found {state.documents}"
        )
    if state.source_offset != source_stat.st_size:
        raise ValueError(f"{spec.split_id} did not consume its complete source file")
    if state.output_bytes != state.tokens * UINT16_BYTES:
        raise ValueError(f"{spec.split_id} output byte count is inconsistent")

    part_path.replace(spec.output_path)
    split_report = _complete_split_report(
        spec=spec,
        state=state,
        tokenizer_sha256=tokenizer_sha256,
        source_stat=source_stat,
    )
    _write_json(spec.metadata_path, split_report)
    spec.checkpoint_path.unlink(missing_ok=True)
    _write_progress(
        progress,
        f"{spec.split_id}: complete documents={state.documents:,} "
        f"tokens={state.tokens:,} output={state.output_bytes / (1024 ** 2):.1f} MiB",
    )
    return split_report


def load_memory_mapped_token_ids(
    path: Path,
    *,
    token_count: int,
) -> torch.Tensor:
    """Map a little-endian uint16 token file without reading it into RAM."""

    if token_count <= 0:
        raise ValueError("token_count must be greater than 0")
    if not path.is_file():
        raise FileNotFoundError(f"token file does not exist: {path}")
    expected_bytes = token_count * UINT16_BYTES
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"token file size mismatch: expected {expected_bytes}, found {actual_bytes}"
        )
    if sys.byteorder != "little":
        raise RuntimeError("little-endian token files require a little-endian host")
    return torch.from_file(
        str(path),
        shared=False,
        size=token_count,
        dtype=torch.uint16,
    )


def _build_split_specs(
    *,
    config: PaisaHistoricalEncodingConfig,
    tokenizer_report: dict[str, object],
    curriculum_report: dict[str, object],
) -> list[SplitEncodingSpec]:
    artifacts = _required_mapping(tokenizer_report, "local_artifacts")
    paisa = _required_mapping(curriculum_report, "paisa")
    historical = _required_mapping(curriculum_report, "historical")
    sanitizer = _required_mapping(
        tokenizer_report,
        "paisa_validation_sanitization",
    )
    split_inputs = (
        (
            "paisa_train",
            Path(str(artifacts["paisa_train_path"])),
            int(paisa["train_documents"]),
        ),
        (
            "paisa_validation",
            Path(str(artifacts["paisa_validation_tokenizable_path"])),
            int(sanitizer["retained_documents"]),
        ),
        (
            "historical_train",
            Path(str(artifacts["historical_train_path"])),
            int(historical["source_count"]),
        ),
        (
            "historical_validation",
            Path(str(artifacts["historical_validation_path"])),
            int(historical["source_count"]),
        ),
    )
    specs = []
    for split_id, source_path, expected_documents in split_inputs:
        specs.append(
            SplitEncodingSpec(
                split_id=split_id,
                source_path=source_path,
                expected_documents=expected_documents,
                output_path=config.output_dir / f"{split_id}.uint16.bin",
                metadata_path=config.output_dir / f"{split_id}.metadata.json",
                checkpoint_path=config.output_dir / f".{split_id}.checkpoint.json",
            )
        )
    return specs


def _iter_documents(
    source,
    source_path: Path,
) -> Iterator[tuple[str, int]]:
    document_bytes = bytearray()
    separator = PAISA_DOCUMENT_SEPARATOR.encode("utf-8")
    while line := source.readline():
        if line.rstrip(b"\r\n") == separator:
            try:
                document = document_bytes.decode("utf-8").strip()
            except UnicodeDecodeError as error:
                raise ValueError(f"split is not valid UTF-8: {source_path}") from error
            if not document:
                raise ValueError(f"split contains an empty document: {source_path}")
            yield document, source.tell()
            document_bytes.clear()
        else:
            document_bytes.extend(line)
    if document_bytes.strip():
        raise ValueError(f"split has an unterminated document: {source_path}")


def _encode_pretoken(
    pretoken: str,
    merge_ranks: dict[TokenPair, int],
) -> tuple[str, ...]:
    sequence = tuple(pretoken)
    while len(sequence) > 1:
        ranked_pairs = [
            (merge_ranks[pair], pair)
            for pair in zip(sequence, sequence[1:])
            if pair in merge_ranks
        ]
        if not ranked_pairs:
            break
        _, best_pair = min(ranked_pairs)
        sequence = tuple(
            merge_token_pair(
                token_sequence=list(sequence),
                pair=best_pair,
                merged_token="".join(best_pair),
            )
        )
    return sequence


def _write_uint16(output, token_ids: list[int]) -> None:
    values = array("H", token_ids)
    if sys.byteorder != "little":
        values.byteswap()
    values.tofile(output)


def _validate_token_ids(token_ids: list[int], vocab_size: int) -> None:
    if vocab_size > UINT16_MAX + 1:
        raise ValueError("tokenizer vocabulary does not fit in uint16")
    if not token_ids:
        raise ValueError("encoded document produced no token IDs")
    minimum = min(token_ids)
    maximum = max(token_ids)
    if minimum < 0 or maximum >= vocab_size:
        raise ValueError(
            f"encoded token ID range [{minimum}, {maximum}] "
            f"is outside vocabulary size {vocab_size}"
        )


def _load_or_initialize_progress(
    *,
    spec: SplitEncodingSpec,
    part_path: Path,
    tokenizer_sha256: str,
    source_stat: os.stat_result,
) -> SplitEncodingProgress:
    if not spec.checkpoint_path.is_file():
        part_path.unlink(missing_ok=True)
        return SplitEncodingProgress()

    payload = _read_json(spec.checkpoint_path)
    _validate_artifact_identity(
        payload=payload,
        spec=spec,
        tokenizer_sha256=tokenizer_sha256,
        source_stat=source_stat,
    )
    if not part_path.is_file():
        raise FileNotFoundError(
            f"checkpoint exists without partial token file: {part_path}"
        )
    state = SplitEncodingProgress(
        source_offset=int(payload["source_offset"]),
        output_bytes=int(payload["output_bytes"]),
        documents=int(payload["documents"]),
        characters=int(payload["characters"]),
        tokens=int(payload["tokens"]),
    )
    if part_path.stat().st_size < state.output_bytes:
        raise ValueError("partial token file is shorter than its checkpoint")
    return state


def _persist_progress(
    *,
    spec: SplitEncodingSpec,
    state: SplitEncodingProgress,
    output,
    tokenizer_sha256: str,
    source_stat: os.stat_result,
) -> None:
    output.flush()
    os.fsync(output.fileno())
    _write_json_atomic(
        spec.checkpoint_path,
        {
            "split_id": spec.split_id,
            "source_path": _portable_path(spec.source_path),
            "source_size": source_stat.st_size,
            "source_mtime_ns": source_stat.st_mtime_ns,
            "tokenizer_sha256": tokenizer_sha256,
            "format": ENCODING_FORMAT,
            "source_offset": state.source_offset,
            "output_bytes": state.output_bytes,
            "documents": state.documents,
            "characters": state.characters,
            "tokens": state.tokens,
        },
    )


def _load_completed_split(
    *,
    spec: SplitEncodingSpec,
    tokenizer_sha256: str,
    source_stat: os.stat_result,
) -> dict[str, object] | None:
    if not spec.output_path.is_file() and not spec.metadata_path.is_file():
        return None
    if spec.output_path.is_file() and not spec.metadata_path.is_file():
        recovered = _recover_completed_split(
            spec=spec,
            tokenizer_sha256=tokenizer_sha256,
            source_stat=source_stat,
        )
        if recovered is not None:
            return recovered
    if not spec.output_path.is_file() or not spec.metadata_path.is_file():
        raise ValueError(
            f"completed split requires both token and metadata files: {spec.split_id}"
        )
    payload = _read_json(spec.metadata_path)
    _validate_artifact_identity(
        payload=payload,
        spec=spec,
        tokenizer_sha256=tokenizer_sha256,
        source_stat=source_stat,
    )
    if payload.get("status") != "complete":
        raise ValueError(f"split metadata is not complete: {spec.metadata_path}")
    expected_bytes = int(payload["tokens"]) * UINT16_BYTES
    if spec.output_path.stat().st_size != expected_bytes:
        raise ValueError(
            "completed token file size does not match metadata: "
            f"{spec.output_path}"
        )
    if _file_sha256(spec.output_path) != payload.get("output_sha256"):
        raise ValueError(
            "completed token file SHA-256 does not match metadata: "
            f"{spec.output_path}"
        )
    spec.checkpoint_path.unlink(missing_ok=True)
    return payload


def _recover_completed_split(
    *,
    spec: SplitEncodingSpec,
    tokenizer_sha256: str,
    source_stat: os.stat_result,
) -> dict[str, object] | None:
    """Finish metadata publication after a crash following the binary rename."""

    if not spec.checkpoint_path.is_file():
        return None
    checkpoint = _read_json(spec.checkpoint_path)
    _validate_artifact_identity(
        payload=checkpoint,
        spec=spec,
        tokenizer_sha256=tokenizer_sha256,
        source_stat=source_stat,
    )
    state = SplitEncodingProgress(
        source_offset=int(checkpoint["source_offset"]),
        output_bytes=int(checkpoint["output_bytes"]),
        documents=int(checkpoint["documents"]),
        characters=int(checkpoint["characters"]),
        tokens=int(checkpoint["tokens"]),
    )
    if state.source_offset != source_stat.st_size:
        return None
    if state.documents != spec.expected_documents:
        return None
    if spec.output_path.stat().st_size != state.output_bytes:
        return None
    payload = _complete_split_report(
        spec=spec,
        state=state,
        tokenizer_sha256=tokenizer_sha256,
        source_stat=source_stat,
    )
    _write_json(spec.metadata_path, payload)
    spec.checkpoint_path.unlink()
    return payload


def _validate_artifact_identity(
    *,
    payload: dict[str, object],
    spec: SplitEncodingSpec,
    tokenizer_sha256: str,
    source_stat: os.stat_result,
) -> None:
    expected = {
        "split_id": spec.split_id,
        "source_path": _portable_path(spec.source_path),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "tokenizer_sha256": tokenizer_sha256,
        "format": ENCODING_FORMAT,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"{spec.split_id} artifact identity mismatch for {field}"
            )


def _complete_split_report(
    *,
    spec: SplitEncodingSpec,
    state: SplitEncodingProgress,
    tokenizer_sha256: str,
    source_stat: os.stat_result,
) -> dict[str, object]:
    return {
        "split_id": spec.split_id,
        "status": "complete",
        "source_path": _portable_path(spec.source_path),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_sha256": _file_sha256(spec.source_path),
        "output_path": _portable_path(spec.output_path),
        "output_bytes": state.output_bytes,
        "output_sha256": _file_sha256(spec.output_path),
        "metadata_path": _portable_path(spec.metadata_path),
        "tokenizer_sha256": tokenizer_sha256,
        "format": ENCODING_FORMAT,
        "dtype": "torch.uint16",
        "documents": state.documents,
        "characters": state.characters,
        "tokens": state.tokens,
        "characters_per_token": state.characters / state.tokens,
        "document_separator_token_count": state.documents,
    }


def _incomplete_split_report(
    spec: SplitEncodingSpec,
    state: SplitEncodingProgress,
) -> dict[str, object]:
    return {
        "split_id": spec.split_id,
        "status": "incomplete",
        "source_path": _portable_path(spec.source_path),
        "output_path": _portable_path(spec.output_path),
        "checkpoint_path": _portable_path(spec.checkpoint_path),
        "documents": state.documents,
        "characters": state.characters,
        "tokens": state.tokens,
    }


def _make_report(
    *,
    config: PaisaHistoricalEncodingConfig,
    started_at: str,
    tokenizer_report: dict[str, object],
    tokenizer_path: Path,
    tokenizer_sha256: str,
    split_reports: list[dict[str, object]],
) -> dict[str, object]:
    total_tokens = sum(int(split["tokens"]) for split in split_reports)
    total_characters = sum(int(split["characters"]) for split in split_reports)
    return {
        "curriculum_id": tokenizer_report["curriculum_id"],
        "status": "complete",
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "split_policy": (
            "encode the four existing curriculum splits independently; "
            "no new random or suffix split"
        ),
        "document_encoding_policy": (
            "strip transport whitespace from each existing document and append "
            "one atomic <|endoftext|> token"
        ),
        "format": {
            "name": ENCODING_FORMAT,
            "dtype": "torch.uint16",
            "bytes_per_token": UINT16_BYTES,
            "loader": "torch.from_file",
        },
        "tokenizer": {
            "path": _portable_path(tokenizer_path),
            "sha256": tokenizer_sha256,
            "vocab_size": int(
                _required_mapping(tokenizer_report, "tokenizer")["actual_vocab_size"]
            ),
            "document_separator": PAISA_DOCUMENT_SEPARATOR,
            "document_separator_token_id": 0,
        },
        "totals": {
            "documents": sum(int(split["documents"]) for split in split_reports),
            "characters": total_characters,
            "tokens": total_tokens,
            "characters_per_token": total_characters / total_tokens,
            "output_bytes": sum(
                int(split["output_bytes"])
                for split in split_reports
            ),
        },
        "splits": split_reports,
        "resumption_policy": (
            "checkpoint complete document boundaries; truncate uncheckpointed "
            "partial output before resuming"
        ),
        "public_repository_policy": (
            "This aggregate report contains no PAISA document text or document URLs. "
            "PAISA-derived token files and local metadata remain under data/local."
        ),
    }


def _validate_tokenizer(
    tokenizer: BytePairEncodingTokenizer,
    tokenizer_report: dict[str, object],
) -> None:
    report_tokenizer = _required_mapping(tokenizer_report, "tokenizer")
    if tokenizer.vocab_size != int(report_tokenizer["actual_vocab_size"]):
        raise ValueError("tokenizer vocabulary size does not match tokenizer report")
    if tokenizer.vocab_size > UINT16_MAX + 1:
        raise ValueError("tokenizer vocabulary does not fit in uint16")
    if tokenizer.special_tokens != [PAISA_DOCUMENT_SEPARATOR]:
        raise ValueError(
            "rescue tokenizer must contain only <|endoftext|> as special token"
        )
    separator_ids = tokenizer.encode(PAISA_DOCUMENT_SEPARATOR)
    if separator_ids != [0]:
        raise ValueError("document separator must be atomic token ID 0")


def _tokenizer_path(tokenizer_report: dict[str, object]) -> Path:
    artifacts = _required_mapping(tokenizer_report, "local_artifacts")
    path = Path(str(artifacts["tokenizer_path"]))
    if not path.is_file():
        raise FileNotFoundError(f"tokenizer file does not exist: {path}")
    return path


def _validate_config(config: PaisaHistoricalEncodingConfig) -> None:
    if config.progress_interval_documents <= 0:
        raise ValueError("progress_interval_documents must be greater than 0")
    if config.checkpoint_interval_documents <= 0:
        raise ValueError("checkpoint_interval_documents must be greater than 0")
    if config.pretoken_cache_entries <= 0:
        raise ValueError("pretoken_cache_entries must be greater than 0")
    if (
        config.max_documents_per_split_run is not None
        and config.max_documents_per_split_run <= 0
    ):
        raise ValueError("max_documents_per_split_run must be greater than 0")


def _report_split_progress(
    *,
    spec: SplitEncodingSpec,
    state: SplitEncodingProgress,
    source_size: int,
    invocation_started: float,
    invocation_start_offset: int,
    invocation_documents: int,
    progress: ProgressCallback | None,
) -> None:
    elapsed = max(time.monotonic() - invocation_started, 1e-9)
    fraction = state.source_offset / source_size
    bytes_this_invocation = state.source_offset - invocation_start_offset
    rate = bytes_this_invocation / elapsed
    remaining_seconds = (
        (source_size - state.source_offset) / rate
        if rate > 0
        else 0.0
    )
    _write_progress(
        progress,
        f"{spec.split_id}: documents={state.documents:,}/"
        f"{spec.expected_documents:,} progress={fraction:.1%} "
        f"tokens={state.tokens:,} elapsed={_format_duration(elapsed)} "
        f"eta={_format_duration(remaining_seconds)} "
        f"run_documents={invocation_documents:,}",
    )


def _required_mapping(
    payload: dict[str, object],
    field: str,
) -> dict[str, object]:
    value = payload.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object is missing mapping field: {field}")
    return value


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary_path, payload)
    temporary_path.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3_600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _write_progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
