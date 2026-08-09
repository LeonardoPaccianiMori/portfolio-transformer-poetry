"""Stream complete PAISA and historical splits through Minerva's tokenizer."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from array import array
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

import torch

from sonnet_corpus.paisa_build import PAISA_DOCUMENT_SEPARATOR
from sonnet_training.minerva_7b_qlora import (
    MINERVA_7B_INSTRUCT_MODEL_ID,
    MINERVA_7B_INSTRUCT_REVISION,
)


FULL_WEIGHT_DATA_VERSION = "minerva_7b_full_weight_mixed_corpus_v1"
ENCODING_FORMAT = "little_endian_int32_shards"
INT32_BYTES = 4
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class Minerva7BFullWeightDataConfig:
    """Freeze inputs and bounded-memory settings for full-corpus tokenization."""

    model_id: str = MINERVA_7B_INSTRUCT_MODEL_ID
    revision: str = MINERVA_7B_INSTRUCT_REVISION
    curriculum_report_path: str = (
        "data/local/pretraining/paisa_historical_rescue_v1/curriculum_report.json"
    )
    paisa_build_report_path: str = (
        "data/local/pretraining/paisa/paisa_modern_italian_v1/build_report.json"
    )
    tokenizer_cache_dir: str = "data/local/minerva_qlora/huggingface"
    output_dir: str = "data/local/minerva_7b_full_weight/encoded"
    public_report_path: str = "reports/minerva_7b_full_weight_data_report.json"
    context_length: int = 512
    shard_target_tokens: int = 8_388_608
    progress_interval_documents: int = 5_000
    checkpoint_interval_documents: int = 500
    max_documents_per_split_run: int | None = None
    seed: int = 1337


@dataclass(frozen=True)
class FullWeightSplitSpec:
    """Describe one preserved source split and its identity stream."""

    split_id: str
    corpus_role: str
    source_path: Path
    expected_documents: int
    attribution_path: Path | None = None
    attribution_split: str | None = None
    historical_source_ids: tuple[str, ...] = ()


@dataclass
class ShardedEncodingState:
    """Persist only completed-document boundaries for interruption recovery."""

    source_offset: int = 0
    attribution_offset: int = 0
    index_bytes: int = 0
    documents: int = 0
    characters: int = 0
    tokens: int = 0
    eos_tokens: int = 0
    current_shard_index: int = 0
    current_shard_tokens: int = 0
    completed_shards: list[dict[str, Any]] = field(default_factory=list)


class _ShardedInt32Writer:
    """Append token IDs to bounded shards while preserving a resumable tail."""

    def __init__(
        self,
        *,
        output_dir: Path,
        split_id: str,
        shard_target_tokens: int,
        state: ShardedEncodingState,
    ) -> None:
        self.output_dir = output_dir
        self.split_id = split_id
        self.shard_target_tokens = shard_target_tokens
        self.state = state
        self.handle: BinaryIO | None = None
        self._open_current_shard()

    def write_document(self, token_ids: Sequence[int]) -> tuple[dict[str, int], dict[str, int]]:
        if not token_ids:
            raise ValueError("cannot write an empty tokenized document")
        if (
            self.state.current_shard_tokens > 0
            and self.state.current_shard_tokens + len(token_ids)
            > self.shard_target_tokens
        ):
            self.finalize_current_shard()
            self._open_current_shard()
        start = {
            "shard_index": self.state.current_shard_index,
            "token_offset": self.state.current_shard_tokens,
        }
        assert self.handle is not None
        _write_int32(self.handle, token_ids)
        self.state.current_shard_tokens += len(token_ids)
        end = {
            "shard_index": self.state.current_shard_index,
            "token_offset": self.state.current_shard_tokens,
        }
        return start, end

    def flush(self) -> None:
        assert self.handle is not None
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def finalize_current_shard(self) -> None:
        if self.handle is None:
            return
        self.flush()
        self.handle.close()
        self.handle = None
        if self.state.current_shard_tokens == 0:
            self._part_path().unlink(missing_ok=True)
            return
        part_path = self._part_path()
        final_path = self._final_path()
        expected_bytes = self.state.current_shard_tokens * INT32_BYTES
        if part_path.stat().st_size != expected_bytes:
            raise ValueError("current token shard has an inconsistent byte size")
        part_path.replace(final_path)
        first_token = sum(
            int(row["token_count"]) for row in self.state.completed_shards
        )
        self.state.completed_shards.append({
            "shard_index": self.state.current_shard_index,
            "path": _portable_path(final_path),
            "token_count": self.state.current_shard_tokens,
            "bytes": expected_bytes,
            "sha256": _sha256(final_path),
            "global_token_start": first_token,
            "global_token_end": first_token + self.state.current_shard_tokens,
        })
        self.state.current_shard_index += 1
        self.state.current_shard_tokens = 0

    def close_incomplete(self) -> None:
        if self.handle is not None:
            self.flush()
            self.handle.close()
            self.handle = None

    def _open_current_shard(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        part_path = self._part_path()
        expected_bytes = self.state.current_shard_tokens * INT32_BYTES
        mode = "r+b" if part_path.exists() else "w+b"
        self.handle = part_path.open(mode)
        self.handle.truncate(expected_bytes)
        self.handle.seek(expected_bytes)

    def _part_path(self) -> Path:
        return self.output_dir / (
            f".{self.split_id}-{self.state.current_shard_index:05d}.int32.bin.part"
        )

    def _final_path(self) -> Path:
        return self.output_dir / (
            f"{self.split_id}-{self.state.current_shard_index:05d}.int32.bin"
        )


def prepare_minerva_7b_full_weight_data(
    *,
    repo_root: Path,
    config: Minerva7BFullWeightDataConfig = Minerva7BFullWeightDataConfig(),
    tokenizer: Any | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Encode all four existing corpus splits without loading a corpus into RAM."""
    validate_full_weight_data_config(config)
    curriculum_path = _resolve(repo_root, config.curriculum_report_path)
    paisa_report_path = _resolve(repo_root, config.paisa_build_report_path)
    curriculum = _read_json(curriculum_path)
    paisa_report = _read_json(paisa_report_path)
    specs = build_full_weight_split_specs(
        repo_root=repo_root,
        curriculum=curriculum,
        paisa_report=paisa_report,
    )
    if tokenizer is None:
        from transformers import AutoTokenizer

        _report(progress, "loading pinned Minerva 7B tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            cache_dir=_resolve(repo_root, config.tokenizer_cache_dir),
        )
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token_id, int) or eos_token_id < 0:
        raise ValueError("Minerva tokenizer must define a non-negative eos_token_id")
    tokenizer_fingerprint = tokenizer_sha256(tokenizer)
    tokenizer_size = len(tokenizer)
    if tokenizer_size > 2**31:
        raise ValueError("Minerva tokenizer vocabulary does not fit signed int32")

    output_dir = _resolve(repo_root, config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    split_reports: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        _report(progress, f"split {index}/{len(specs)} start: {spec.split_id}")
        split_report = encode_full_weight_split(
            spec=spec,
            output_dir=output_dir,
            tokenizer=tokenizer,
            tokenizer_fingerprint=tokenizer_fingerprint,
            eos_token_id=eos_token_id,
            shard_target_tokens=config.shard_target_tokens,
            progress_interval_documents=config.progress_interval_documents,
            checkpoint_interval_documents=config.checkpoint_interval_documents,
            max_documents=config.max_documents_per_split_run,
            progress=progress,
        )
        split_reports.append(split_report)
        if split_report["status"] != "complete":
            return {
                "data_version": FULL_WEIGHT_DATA_VERSION,
                "status": "incomplete",
                "started_at_utc": started_at,
                "finished_at_utc": _utc_now(),
                "splits": split_reports,
            }

    report = _build_data_report(
        repo_root=repo_root,
        config=config,
        curriculum_path=curriculum_path,
        paisa_report_path=paisa_report_path,
        tokenizer=tokenizer,
        tokenizer_fingerprint=tokenizer_fingerprint,
        eos_token_id=eos_token_id,
        tokenizer_size=tokenizer_size,
        split_reports=split_reports,
        started_at=started_at,
    )
    calibration_path = output_dir / "calibration_windows.pt"
    build_full_weight_calibration_windows(
        repo_root=repo_root,
        report=report,
        output_path=calibration_path,
        context_length=config.context_length,
        seed=config.seed,
    )
    report["calibration_windows"] = {
        "path": _portable_path(calibration_path),
        "sha256": _sha256(calibration_path),
        "training_sources": [
            "paisa_train",
            "historical_train",
            "paisa_train",
            "historical_train",
            "paisa_train",
        ],
        "validation_sources": ["paisa_validation", "historical_validation"],
    }
    local_report_path = output_dir / "report.json"
    public_report_path = _resolve(repo_root, config.public_report_path)
    _write_json(local_report_path, report)
    _write_json(public_report_path, report)
    _report(progress, f"wrote local full-weight data report: {local_report_path}")
    _report(progress, f"wrote public aggregate report: {public_report_path}")
    return report


def build_full_weight_split_specs(
    *,
    repo_root: Path,
    curriculum: Mapping[str, Any],
    paisa_report: Mapping[str, Any],
) -> tuple[FullWeightSplitSpec, ...]:
    """Resolve the four previously frozen train/validation splits."""
    local_artifacts = _required_mapping(curriculum, "local_artifacts")
    paisa = _required_mapping(curriculum, "paisa")
    historical = _required_mapping(curriculum, "historical")
    paisa_artifacts = _required_mapping(paisa_report, "local_artifacts")
    historical_sources = historical.get("sources")
    if not isinstance(historical_sources, list) or not historical_sources:
        raise ValueError("curriculum report is missing historical source identities")
    source_ids_list: list[str] = []
    for row in historical_sources:
        if not isinstance(row, Mapping) or not isinstance(row.get("source_id"), str):
            raise ValueError("curriculum contains an invalid historical source row")
        source_ids_list.append(str(row["source_id"]))
    source_ids = tuple(source_ids_list)
    attribution_path = _resolve(
        repo_root,
        str(paisa_artifacts["document_attribution_inventory_path"]),
    )
    specs = (
        FullWeightSplitSpec(
            split_id="paisa_train",
            corpus_role="modern_italian",
            source_path=_resolve(repo_root, str(paisa_artifacts["train_text_path"])),
            expected_documents=int(paisa["train_documents"]),
            attribution_path=attribution_path,
            attribution_split="train",
        ),
        FullWeightSplitSpec(
            split_id="paisa_validation",
            corpus_role="modern_italian",
            source_path=_resolve(
                repo_root,
                str(paisa_artifacts["validation_text_path"]),
            ),
            expected_documents=int(paisa["validation_documents"]),
            attribution_path=attribution_path,
            attribution_split="validation",
        ),
        FullWeightSplitSpec(
            split_id="historical_train",
            corpus_role="historical_italian",
            source_path=_resolve(
                repo_root,
                str(local_artifacts["historical_train_path"]),
            ),
            expected_documents=int(historical["source_count"]),
            historical_source_ids=source_ids,
        ),
        FullWeightSplitSpec(
            split_id="historical_validation",
            corpus_role="historical_italian",
            source_path=_resolve(
                repo_root,
                str(local_artifacts["historical_validation_path"]),
            ),
            expected_documents=int(historical["source_count"]),
            historical_source_ids=source_ids,
        ),
    )
    for spec in specs:
        if not spec.source_path.is_file():
            raise FileNotFoundError(f"full-weight split does not exist: {spec.source_path}")
        if spec.attribution_path is not None and not spec.attribution_path.is_file():
            raise FileNotFoundError(
                f"PAISA attribution inventory does not exist: {spec.attribution_path}"
            )
    return specs


def encode_full_weight_split(
    *,
    spec: FullWeightSplitSpec,
    output_dir: Path,
    tokenizer: Any,
    tokenizer_fingerprint: str,
    eos_token_id: int,
    shard_target_tokens: int,
    progress_interval_documents: int,
    checkpoint_interval_documents: int,
    max_documents: int | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Tokenize one split into resumable int32 shards and a local identity index."""
    _validate_encoding_arguments(
        spec=spec,
        shard_target_tokens=shard_target_tokens,
        progress_interval_documents=progress_interval_documents,
        checkpoint_interval_documents=checkpoint_interval_documents,
        max_documents=max_documents,
    )
    metadata_path = output_dir / f"{spec.split_id}.metadata.json"
    completed = _load_completed_split(
        spec=spec,
        metadata_path=metadata_path,
        tokenizer_fingerprint=tokenizer_fingerprint,
        shard_target_tokens=shard_target_tokens,
    )
    if completed is not None:
        _report(progress, f"{spec.split_id}: verified existing output")
        return completed

    checkpoint_path = output_dir / f".{spec.split_id}.checkpoint.json"
    index_part_path = output_dir / f".{spec.split_id}.documents.jsonl.part"
    index_path = output_dir / f"{spec.split_id}.documents.jsonl"
    state = _load_or_initialize_state(
        spec=spec,
        checkpoint_path=checkpoint_path,
        index_part_path=index_part_path,
        tokenizer_fingerprint=tokenizer_fingerprint,
        shard_target_tokens=shard_target_tokens,
    )
    writer = _ShardedInt32Writer(
        output_dir=output_dir,
        split_id=spec.split_id,
        shard_target_tokens=shard_target_tokens,
        state=state,
    )
    invocation_started = time.monotonic()
    invocation_start_offset = state.source_offset
    invocation_documents = 0
    source_size = spec.source_path.stat().st_size
    index_part_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (
            spec.source_path.open("rb") as source_handle,
            index_part_path.open("r+b" if index_part_path.exists() else "w+b") as index_handle,
        ):
            source_handle.seek(state.source_offset)
            index_handle.truncate(state.index_bytes)
            index_handle.seek(state.index_bytes)
            identity_iterator = _identity_iterator(spec=spec, state=state)
            for document, next_source_offset in _iter_documents(
                source_handle,
                spec.source_path,
            ):
                identity, next_attribution_offset = next(identity_iterator)
                _validate_document_identity(spec, document, identity)
                token_ids = _token_ids(tokenizer, document)
                token_ids.append(eos_token_id)
                _validate_token_ids(token_ids, len(tokenizer))
                start, end = writer.write_document(token_ids)
                index_row = {
                    "split_id": spec.split_id,
                    "corpus_role": spec.corpus_role,
                    "document_index": state.documents,
                    "source_id": identity["source_id"],
                    "text_sha256": identity.get("text_sha256"),
                    "characters": len(document),
                    "tokens": len(token_ids),
                    "token_start": start,
                    "token_end": end,
                }
                index_handle.write(
                    (json.dumps(index_row, ensure_ascii=False) + "\n").encode("utf-8")
                )
                state.source_offset = next_source_offset
                state.attribution_offset = next_attribution_offset
                state.documents += 1
                state.characters += len(document)
                state.tokens += len(token_ids)
                state.eos_tokens += 1
                state.index_bytes = index_handle.tell()
                invocation_documents += 1

                should_checkpoint = (
                    state.documents % checkpoint_interval_documents == 0
                    or (max_documents is not None and invocation_documents >= max_documents)
                )
                if should_checkpoint:
                    writer.flush()
                    index_handle.flush()
                    os.fsync(index_handle.fileno())
                    _persist_state(
                        spec=spec,
                        state=state,
                        checkpoint_path=checkpoint_path,
                        tokenizer_fingerprint=tokenizer_fingerprint,
                        shard_target_tokens=shard_target_tokens,
                    )
                if (
                    state.documents % progress_interval_documents == 0
                    or state.documents == spec.expected_documents
                ):
                    _report_split_progress(
                        spec=spec,
                        state=state,
                        source_size=source_size,
                        invocation_started=invocation_started,
                        invocation_start_offset=invocation_start_offset,
                        invocation_documents=invocation_documents,
                        progress=progress,
                    )
                if max_documents is not None and invocation_documents >= max_documents:
                    writer.close_incomplete()
                    return _incomplete_split_report(spec, state, checkpoint_path)

            try:
                next(identity_iterator)
            except StopIteration:
                pass
            else:
                raise ValueError(
                    f"{spec.split_id} identity inventory contains extra documents"
                )
            writer.finalize_current_shard()
            index_handle.flush()
            os.fsync(index_handle.fileno())
    except Exception:
        writer.close_incomplete()
        raise

    if state.documents != spec.expected_documents:
        raise ValueError(
            f"{spec.split_id} document count mismatch: expected "
            f"{spec.expected_documents}, found {state.documents}"
        )
    if state.source_offset != source_size:
        raise ValueError(f"{spec.split_id} did not consume its complete source")
    index_part_path.replace(index_path)
    report = _complete_split_report(
        spec=spec,
        state=state,
        index_path=index_path,
        tokenizer_fingerprint=tokenizer_fingerprint,
        shard_target_tokens=shard_target_tokens,
    )
    _write_json(metadata_path, report)
    checkpoint_path.unlink(missing_ok=True)
    _report(
        progress,
        f"{spec.split_id}: complete documents={state.documents:,} "
        f"tokens={state.tokens:,} shards={len(state.completed_shards)}",
    )
    return report


def build_full_weight_calibration_windows(
    *,
    repo_root: Path,
    report: Mapping[str, Any],
    output_path: Path,
    context_length: int,
    seed: int,
) -> dict[str, Any]:
    """Extract deterministic mixed-corpus windows for the five-update GPU probe."""
    split_rows = report.get("splits")
    if not isinstance(split_rows, list):
        raise ValueError("full-weight data report is missing splits")
    by_id = {
        str(row["split_id"]): row
        for row in split_rows
        if isinstance(row, Mapping)
    }
    training_sources = (
        "paisa_train",
        "historical_train",
        "paisa_train",
        "historical_train",
        "paisa_train",
    )
    validation_sources = ("paisa_validation", "historical_validation")
    training_rows = []
    training_metadata = []
    for index, split_id in enumerate(training_sources):
        row, start = _select_window(
            repo_root=repo_root,
            split=by_id[split_id],
            context_length=context_length,
            selection_key=f"{seed}:train:{index}:{split_id}",
        )
        training_rows.append(row)
        training_metadata.append({"split_id": split_id, "global_token_start": start})
    validation_rows = []
    validation_metadata = []
    for index, split_id in enumerate(validation_sources):
        row, start = _select_window(
            repo_root=repo_root,
            split=by_id[split_id],
            context_length=context_length,
            selection_key=f"{seed}:validation:{index}:{split_id}",
        )
        validation_rows.append(row)
        validation_metadata.append({"split_id": split_id, "global_token_start": start})
    payload = {
        "data_version": FULL_WEIGHT_DATA_VERSION,
        "model_id": report["model_id"],
        "revision": report["revision"],
        "context_length": context_length,
        "training_sources": list(training_sources),
        "validation_sources": list(validation_sources),
        "training_windows": torch.stack(training_rows).to(torch.int32),
        "validation_windows": torch.stack(validation_rows).to(torch.int32),
        "training_metadata": training_metadata,
        "validation_metadata": validation_metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return payload


def load_full_weight_calibration_windows(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("full-weight calibration windows must contain a dictionary")
    context_length = payload.get("context_length")
    training = payload.get("training_windows")
    validation = payload.get("validation_windows")
    if context_length != 512:
        raise ValueError("full-weight calibration context length must be 512")
    if not isinstance(training, torch.Tensor) or training.shape != (5, 512):
        raise ValueError("full-weight calibration must contain five training windows")
    if not isinstance(validation, torch.Tensor) or validation.shape != (2, 512):
        raise ValueError("full-weight calibration must contain two validation windows")
    if training.dtype != torch.int32 or validation.dtype != torch.int32:
        raise ValueError("full-weight calibration windows must use torch.int32")
    return payload


def load_int32_shard(path: Path, *, token_count: int) -> torch.Tensor:
    """Memory-map one signed little-endian int32 token shard."""
    if token_count <= 0:
        raise ValueError("token_count must be greater than zero")
    if sys.byteorder != "little":
        raise RuntimeError("little-endian token shards require a little-endian host")
    if path.stat().st_size != token_count * INT32_BYTES:
        raise ValueError("int32 token shard byte size does not match its metadata")
    return torch.from_file(
        str(path),
        shared=False,
        size=token_count,
        dtype=torch.int32,
    )


def tokenizer_sha256(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and callable(getattr(backend, "to_str", None)):
        serialized = backend.to_str().encode("utf-8")
    else:
        serialized = json.dumps(
            {
                "class": tokenizer.__class__.__name__,
                "vocab_size": len(tokenizer),
                "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            },
            sort_keys=True,
        ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def validate_full_weight_data_config(config: Minerva7BFullWeightDataConfig) -> None:
    if config.model_id != MINERVA_7B_INSTRUCT_MODEL_ID:
        raise ValueError("full-weight data is locked to Minerva 7B Instruct")
    if config.revision != MINERVA_7B_INSTRUCT_REVISION:
        raise ValueError("full-weight data revision does not match the pinned model")
    if config.context_length != 512:
        raise ValueError("full-weight data context length is locked to 512")
    if config.shard_target_tokens <= config.context_length:
        raise ValueError("shard_target_tokens must exceed context_length")
    if config.progress_interval_documents <= 0:
        raise ValueError("progress_interval_documents must be positive")
    if config.checkpoint_interval_documents <= 0:
        raise ValueError("checkpoint_interval_documents must be positive")
    if (
        config.max_documents_per_split_run is not None
        and config.max_documents_per_split_run <= 0
    ):
        raise ValueError("max_documents_per_split_run must be positive when provided")


def _identity_iterator(
    *, spec: FullWeightSplitSpec, state: ShardedEncodingState
) -> Iterator[tuple[dict[str, Any], int]]:
    if spec.attribution_path is not None:
        assert spec.attribution_split is not None
        with spec.attribution_path.open("rb") as handle:
            handle.seek(state.attribution_offset)
            while line := handle.readline():
                next_offset = handle.tell()
                row = json.loads(line)
                if row.get("status") != "retained":
                    continue
                if row.get("split") != spec.attribution_split:
                    continue
                yield {
                    "source_id": str(row["document_id"]),
                    "text_sha256": str(row["text_sha256"]),
                }, next_offset
        return
    for source_id in spec.historical_source_ids[state.documents:]:
        yield {"source_id": source_id, "text_sha256": None}, 0


def _validate_document_identity(
    spec: FullWeightSplitSpec, document: str, identity: Mapping[str, Any]
) -> None:
    expected_sha = identity.get("text_sha256")
    if expected_sha is not None:
        actual_sha = hashlib.sha256(document.encode("utf-8")).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(
                f"{spec.split_id} text no longer matches PAISA attribution identity "
                f"{identity['source_id']}"
            )


def _iter_documents(
    source: BinaryIO, source_path: Path
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


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    token_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if not isinstance(token_ids, list) or any(
        not isinstance(token_id, int) for token_id in token_ids
    ):
        raise ValueError("Minerva tokenizer must return a list of input_ids")
    return token_ids


def _validate_token_ids(token_ids: Sequence[int], vocab_size: int) -> None:
    if any(token_id < 0 or token_id >= vocab_size for token_id in token_ids):
        raise ValueError("tokenizer returned an out-of-vocabulary token ID")


def _write_int32(handle: BinaryIO, token_ids: Sequence[int]) -> None:
    values = array("i", token_ids)
    if values.itemsize != INT32_BYTES:
        raise RuntimeError("native signed integer storage is not 32-bit")
    if sys.byteorder != "little":
        values.byteswap()
    values.tofile(handle)


def _select_window(
    *,
    repo_root: Path,
    split: Mapping[str, Any],
    context_length: int,
    selection_key: str,
) -> tuple[torch.Tensor, int]:
    total_tokens = int(split["tokens"])
    if total_tokens < context_length:
        raise ValueError(f"split is too short for calibration: {split['split_id']}")
    span = total_tokens - context_length + 1
    start = int.from_bytes(
        hashlib.sha256(selection_key.encode("utf-8")).digest()[:8],
        byteorder="big",
    ) % span
    remaining = context_length
    position = start
    pieces: list[torch.Tensor] = []
    shards = split.get("shards")
    if not isinstance(shards, list):
        raise ValueError("split report is missing token shards")
    for shard in shards:
        shard_start = int(shard["global_token_start"])
        shard_end = int(shard["global_token_end"])
        if position >= shard_end or position < shard_start and pieces:
            continue
        if position < shard_start:
            continue
        mapped = load_int32_shard(
            _resolve(repo_root, str(shard["path"])),
            token_count=int(shard["token_count"]),
        )
        local_start = position - shard_start
        take = min(remaining, shard_end - position)
        pieces.append(mapped[local_start:local_start + take].clone())
        remaining -= take
        position += take
        if remaining == 0:
            break
    if remaining:
        raise ValueError("calibration window crosses missing shard coverage")
    return torch.cat(pieces), start


def _build_data_report(
    *,
    repo_root: Path,
    config: Minerva7BFullWeightDataConfig,
    curriculum_path: Path,
    paisa_report_path: Path,
    tokenizer: Any,
    tokenizer_fingerprint: str,
    eos_token_id: int,
    tokenizer_size: int,
    split_reports: list[dict[str, Any]],
    started_at: str,
) -> dict[str, Any]:
    return {
        "data_version": FULL_WEIGHT_DATA_VERSION,
        "status": "complete",
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "model_id": config.model_id,
        "revision": config.revision,
        "config": asdict(config),
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "vocab_size": tokenizer_size,
            "eos_token_id": eos_token_id,
            "serialized_sha256": tokenizer_fingerprint,
        },
        "provenance": {
            "curriculum_report_path": _portable_path(curriculum_path),
            "curriculum_report_sha256": _sha256(curriculum_path),
            "paisa_build_report_path": _portable_path(paisa_report_path),
            "paisa_build_report_sha256": _sha256(paisa_report_path),
            "license_lineage": (
                "PAISA CC BY-NC-SA plus the historical source-specific public-domain "
                "or licensed lineage recorded by the curriculum and attribution index"
            ),
        },
        "split_policy": (
            "preserve the existing PAISA fingerprint split and all 36 existing "
            "historical source-suffix splits; no sonnet or final-test material"
        ),
        "document_encoding_policy": (
            "tokenize one existing document at a time without special-token insertion, "
            "then append exactly one Minerva EOS token"
        ),
        "format": {
            "name": ENCODING_FORMAT,
            "dtype": "torch.int32",
            "bytes_per_token": INT32_BYTES,
            "loader": "torch.from_file",
            "shard_target_tokens": config.shard_target_tokens,
        },
        "totals": {
            "documents": sum(int(row["documents"]) for row in split_reports),
            "characters": sum(int(row["characters"]) for row in split_reports),
            "tokens": sum(int(row["tokens"]) for row in split_reports),
            "eos_tokens": sum(int(row["eos_tokens"]) for row in split_reports),
            "shards": sum(len(row["shards"]) for row in split_reports),
        },
        "splits": split_reports,
        "resumption_policy": (
            "checkpoint only completed document boundaries; truncate the active shard "
            "and identity index to the last durable checkpoint when resuming"
        ),
        "public_repository_policy": (
            "This report contains aggregate metadata and hashes only. PAISA text, "
            "document identities, Minerva-token shards, and calibration windows stay local."
        ),
    }


def _complete_split_report(
    *,
    spec: FullWeightSplitSpec,
    state: ShardedEncodingState,
    index_path: Path,
    tokenizer_fingerprint: str,
    shard_target_tokens: int,
) -> dict[str, Any]:
    return {
        "split_id": spec.split_id,
        "corpus_role": spec.corpus_role,
        "status": "complete",
        "source_path": _portable_path(spec.source_path),
        "source_size": spec.source_path.stat().st_size,
        "source_mtime_ns": spec.source_path.stat().st_mtime_ns,
        "source_sha256": _sha256(spec.source_path),
        "expected_documents": spec.expected_documents,
        "documents": state.documents,
        "characters": state.characters,
        "tokens": state.tokens,
        "eos_tokens": state.eos_tokens,
        "characters_per_token": state.characters / state.tokens,
        "tokenizer_sha256": tokenizer_fingerprint,
        "shard_target_tokens": shard_target_tokens,
        "shards": state.completed_shards,
        "document_index": {
            "path": _portable_path(index_path),
            "sha256": _sha256(index_path),
            "bytes": index_path.stat().st_size,
            "contains_paisa_document_ids": spec.attribution_path is not None,
            "public": False,
        },
        "attribution_inventory_path": (
            _portable_path(spec.attribution_path)
            if spec.attribution_path is not None
            else None
        ),
        "historical_source_ids": list(spec.historical_source_ids),
    }


def _load_completed_split(
    *,
    spec: FullWeightSplitSpec,
    metadata_path: Path,
    tokenizer_fingerprint: str,
    shard_target_tokens: int,
) -> dict[str, Any] | None:
    if not metadata_path.is_file():
        return None
    report = _read_json(metadata_path)
    expected = {
        "split_id": spec.split_id,
        "status": "complete",
        "source_size": spec.source_path.stat().st_size,
        "source_mtime_ns": spec.source_path.stat().st_mtime_ns,
        "expected_documents": spec.expected_documents,
        "tokenizer_sha256": tokenizer_fingerprint,
        "shard_target_tokens": shard_target_tokens,
    }
    for field_name, value in expected.items():
        if report.get(field_name) != value:
            raise ValueError(f"{spec.split_id} completed artifact mismatch: {field_name}")
    shards = report.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"{spec.split_id} completed report has no shards")
    for shard in shards:
        path = Path(str(shard["path"]))
        if not path.is_file() or path.stat().st_size != int(shard["bytes"]):
            raise ValueError(f"{spec.split_id} completed shard is missing or truncated")
    index = _required_mapping(report, "document_index")
    index_path = Path(str(index["path"]))
    if not index_path.is_file() or index_path.stat().st_size != int(index["bytes"]):
        raise ValueError(f"{spec.split_id} document index is missing or truncated")
    return report


def _load_or_initialize_state(
    *,
    spec: FullWeightSplitSpec,
    checkpoint_path: Path,
    index_part_path: Path,
    tokenizer_fingerprint: str,
    shard_target_tokens: int,
) -> ShardedEncodingState:
    output_dir = checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.is_file():
        if index_part_path.exists():
            raise ValueError(f"orphaned identity index without checkpoint: {index_part_path}")
        return ShardedEncodingState()
    payload = _read_json(checkpoint_path)
    identity = _required_mapping(payload, "identity")
    expected = {
        "split_id": spec.split_id,
        "source_size": spec.source_path.stat().st_size,
        "source_mtime_ns": spec.source_path.stat().st_mtime_ns,
        "expected_documents": spec.expected_documents,
        "tokenizer_sha256": tokenizer_fingerprint,
        "shard_target_tokens": shard_target_tokens,
    }
    for field_name, value in expected.items():
        if identity.get(field_name) != value:
            raise ValueError(f"{spec.split_id} checkpoint mismatch: {field_name}")
    state_payload = _required_mapping(payload, "state")
    state = ShardedEncodingState(
        source_offset=int(state_payload["source_offset"]),
        attribution_offset=int(state_payload["attribution_offset"]),
        index_bytes=int(state_payload["index_bytes"]),
        documents=int(state_payload["documents"]),
        characters=int(state_payload["characters"]),
        tokens=int(state_payload["tokens"]),
        eos_tokens=int(state_payload["eos_tokens"]),
        current_shard_index=int(state_payload["current_shard_index"]),
        current_shard_tokens=int(state_payload["current_shard_tokens"]),
        completed_shards=list(state_payload["completed_shards"]),
    )
    if not index_part_path.is_file() or index_part_path.stat().st_size < state.index_bytes:
        raise ValueError(f"{spec.split_id} checkpoint identity index is truncated")
    for shard in state.completed_shards:
        path = Path(str(shard["path"]))
        if not path.is_file() or path.stat().st_size != int(shard["bytes"]):
            raise ValueError(f"{spec.split_id} checkpoint shard is missing or truncated")
    current_part = output_dir / (
        f".{spec.split_id}-{state.current_shard_index:05d}.int32.bin.part"
    )
    current_final = output_dir / (
        f"{spec.split_id}-{state.current_shard_index:05d}.int32.bin"
    )
    expected_current_bytes = state.current_shard_tokens * INT32_BYTES
    # A crash can land after an uncheckpointed shard rollover. Restore that
    # finalized file to the active tail and discard only later uncheckpointed tails.
    if not current_part.exists() and current_final.exists():
        if current_final.stat().st_size != expected_current_bytes:
            raise ValueError(f"{spec.split_id} uncheckpointed rollover is inconsistent")
        current_final.replace(current_part)
    for stale_path in output_dir.glob(f".{spec.split_id}-*.int32.bin.part"):
        if stale_path != current_part:
            stale_path.unlink()
    for stale_path in output_dir.glob(f"{spec.split_id}-*.int32.bin"):
        index_text = stale_path.name.removeprefix(f"{spec.split_id}-").split(".", 1)[0]
        if int(index_text) >= state.current_shard_index:
            stale_path.unlink()
    if not current_part.is_file() or current_part.stat().st_size < (
        expected_current_bytes
    ):
        raise ValueError(f"{spec.split_id} active checkpoint shard is truncated")
    return state


def _persist_state(
    *,
    spec: FullWeightSplitSpec,
    state: ShardedEncodingState,
    checkpoint_path: Path,
    tokenizer_fingerprint: str,
    shard_target_tokens: int,
) -> None:
    _write_json(checkpoint_path, {
        "identity": {
            "split_id": spec.split_id,
            "source_size": spec.source_path.stat().st_size,
            "source_mtime_ns": spec.source_path.stat().st_mtime_ns,
            "expected_documents": spec.expected_documents,
            "tokenizer_sha256": tokenizer_fingerprint,
            "shard_target_tokens": shard_target_tokens,
        },
        "state": asdict(state),
    })


def _validate_encoding_arguments(
    *,
    spec: FullWeightSplitSpec,
    shard_target_tokens: int,
    progress_interval_documents: int,
    checkpoint_interval_documents: int,
    max_documents: int | None,
) -> None:
    if spec.expected_documents <= 0:
        raise ValueError("expected_documents must be positive")
    if shard_target_tokens <= 0:
        raise ValueError("shard_target_tokens must be positive")
    if progress_interval_documents <= 0 or checkpoint_interval_documents <= 0:
        raise ValueError("progress and checkpoint intervals must be positive")
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be positive when provided")


def _incomplete_split_report(
    spec: FullWeightSplitSpec,
    state: ShardedEncodingState,
    checkpoint_path: Path,
) -> dict[str, Any]:
    return {
        "split_id": spec.split_id,
        "corpus_role": spec.corpus_role,
        "status": "incomplete",
        "checkpoint_path": _portable_path(checkpoint_path),
        "documents": state.documents,
        "characters": state.characters,
        "tokens": state.tokens,
    }


def _report_split_progress(
    *,
    spec: FullWeightSplitSpec,
    state: ShardedEncodingState,
    source_size: int,
    invocation_started: float,
    invocation_start_offset: int,
    invocation_documents: int,
    progress: ProgressCallback | None,
) -> None:
    elapsed = max(time.monotonic() - invocation_started, 1e-9)
    fraction = state.source_offset / source_size
    bytes_this_run = state.source_offset - invocation_start_offset
    bytes_per_second = bytes_this_run / elapsed
    remaining_seconds = (
        (source_size - state.source_offset) / bytes_per_second
        if bytes_per_second > 0
        else math.inf
    )
    eta = "unknown" if not math.isfinite(remaining_seconds) else _format_duration(remaining_seconds)
    _report(
        progress,
        f"{spec.split_id}: documents={state.documents:,}/{spec.expected_documents:,} "
        f"progress={fraction:.1%} tokens={state.tokens:,} "
        f"run_documents={invocation_documents:,} eta={eta}",
    )


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSON file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _report(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
