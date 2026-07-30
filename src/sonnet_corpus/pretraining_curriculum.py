"""Prepare deterministic PAISÀ-to-historical rescue curriculum inputs."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .paisa_build import PAISA_DOCUMENT_SEPARATOR


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class PaisaHistoricalCurriculumConfig:
    """Fixed source, split, sampling, and stage policy for the final rescue."""

    curriculum_id: str
    paisa_build_report_path: Path
    historical_mixture_report_path: Path
    expected_paisa_build_report_sha256: str
    expected_historical_mixture_report_sha256: str
    expected_paisa_release_sha256: str
    local_output_dir: Path
    report_path: Path
    historical_source_validation_fraction: float
    tokenizer_vocab_size: int
    tokenizer_special_tokens: tuple[str, ...]
    paisa_train_sample_characters: int
    historical_train_sample_characters: int
    sample_seed: str
    stages: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class HistoricalSourceSplit:
    """One historical source split before it is concatenated into stage inputs."""

    source_id: str
    source_path: Path
    train_text: str
    validation_text: str


def load_paisa_historical_curriculum_config(path: Path) -> PaisaHistoricalCurriculumConfig:
    """Read a checked curriculum policy from a public JSON configuration file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    tokenizer = _required_mapping(payload, "tokenizer")
    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("curriculum config must contain non-empty stages")
    return PaisaHistoricalCurriculumConfig(
        curriculum_id=str(payload["curriculum_id"]),
        paisa_build_report_path=Path(str(payload["paisa_build_report_path"])),
        historical_mixture_report_path=Path(str(payload["historical_mixture_report_path"])),
        expected_paisa_build_report_sha256=str(payload["expected_paisa_build_report_sha256"]),
        expected_historical_mixture_report_sha256=str(
            payload["expected_historical_mixture_report_sha256"]
        ),
        expected_paisa_release_sha256=str(payload["expected_paisa_release_sha256"]),
        local_output_dir=Path(str(payload["local_output_dir"])),
        report_path=Path(str(payload["report_path"])),
        historical_source_validation_fraction=float(
            payload["historical_source_validation_fraction"]
        ),
        tokenizer_vocab_size=int(tokenizer["vocab_size"]),
        tokenizer_special_tokens=tuple(str(token) for token in tokenizer["special_tokens"]),
        paisa_train_sample_characters=int(tokenizer["paisa_train_sample_characters"]),
        historical_train_sample_characters=int(tokenizer["historical_train_sample_characters"]),
        sample_seed=str(tokenizer["sample_seed"]),
        stages=tuple(dict(stage) for stage in stages if isinstance(stage, dict)),
    )


def prepare_paisa_historical_curriculum(
    config: PaisaHistoricalCurriculumConfig,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Create local historical splits and a train-only tokenizer sample."""

    _validate_config(config)
    started_at = _utc_now()
    paisa_report = _read_json(config.paisa_build_report_path)
    historical_report = _read_json(config.historical_mixture_report_path)
    _validate_input_provenance(config, paisa_report, historical_report)
    paisa_train_path, paisa_validation_path = _paisa_split_paths(paisa_report)
    _require_file(paisa_train_path, label="PAISÀ train text")
    _require_file(paisa_validation_path, label="PAISÀ validation text")

    stage_dir = config.local_output_dir.parent / f".{config.local_output_dir.name}.stage"
    _replace_directory(stage_dir)
    historical_train_path = stage_dir / "historical_train.txt"
    historical_validation_path = stage_dir / "historical_validation.txt"
    tokenizer_sample_path = stage_dir / "tokenizer_training_sample.txt"

    _write_progress(progress, "splitting historical sources without using their validation suffixes")
    historical_splits = _build_historical_splits(
        historical_report=historical_report,
        validation_fraction=config.historical_source_validation_fraction,
        train_path=historical_train_path,
        validation_path=historical_validation_path,
        progress=progress,
    )
    _write_progress(progress, "building PAISÀ train-only tokenizer sample")
    with tokenizer_sample_path.open("w", encoding="utf-8") as sample_handle:
        paisa_sample = _write_paisa_training_sample(
            paisa_train_path,
            sample_handle,
            target_characters=config.paisa_train_sample_characters,
            sample_seed=config.sample_seed,
            progress=progress,
        )
        _write_progress(progress, "building stratified historical train-only tokenizer sample")
        historical_sample = _write_historical_training_sample(
            historical_splits,
            sample_handle,
            target_characters=config.historical_train_sample_characters,
        )

    report = _make_report(
        config=config,
        started_at=started_at,
        paisa_report=paisa_report,
        historical_splits=historical_splits,
        paisa_sample=paisa_sample,
        historical_sample=historical_sample,
        historical_train_path=historical_train_path,
        historical_validation_path=historical_validation_path,
        tokenizer_sample_path=tokenizer_sample_path,
    )
    _write_json(stage_dir / "curriculum_report.json", report)
    _publish_directory(stage_dir, config.local_output_dir)
    _write_json(config.report_path, report)
    _write_progress(progress, f"wrote local curriculum inputs: {config.local_output_dir}")
    _write_progress(progress, f"wrote public curriculum report: {config.report_path}")
    return report


def split_historical_source_text(text: str, *, validation_fraction: float) -> tuple[str, str]:
    """Hold out a source's final text fraction at a newline boundary."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be greater than 0 and less than 1")
    normalized = text.strip()
    if not normalized:
        raise ValueError("historical source text is empty")
    target_index = max(1, int(len(normalized) * (1 - validation_fraction)))
    boundary = normalized.find("\n", target_index)
    if boundary == -1:
        boundary = normalized.rfind("\n", 0, target_index)
    if boundary == -1:
        raise ValueError("historical source lacks a newline for its validation split")
    train_text = normalized[:boundary].rstrip()
    validation_text = normalized[boundary:].lstrip()
    if not train_text or not validation_text:
        raise ValueError("historical source split produced an empty partition")
    return train_text, validation_text


def _build_historical_splits(
    *,
    historical_report: dict[str, object],
    validation_fraction: float,
    train_path: Path,
    validation_path: Path,
    progress: ProgressCallback | None,
) -> list[HistoricalSourceSplit]:
    sources = _required_list(historical_report, "sources")
    splits: list[HistoricalSourceSplit] = []
    with (
        train_path.open("w", encoding="utf-8") as train_handle,
        validation_path.open("w", encoding="utf-8") as validation_handle,
    ):
        for index, source in enumerate(sources, start=1):
            source_mapping = _as_mapping(source, label="historical source")
            source_id = str(source_mapping["source_id"])
            source_path = Path(str(source_mapping["source_path"]))
            _require_file(source_path, label=f"historical source {source_id}")
            source_text = source_path.read_text(encoding="utf-8")
            train_text, validation_text = split_historical_source_text(
                source_text,
                validation_fraction=validation_fraction,
            )
            _write_document(train_handle, train_text)
            _write_document(validation_handle, validation_text)
            splits.append(
                HistoricalSourceSplit(
                    source_id=source_id,
                    source_path=source_path,
                    train_text=train_text,
                    validation_text=validation_text,
                )
            )
            _write_progress(progress, f"split historical source {index}/{len(sources)}: {source_id}")
    return splits


def _write_paisa_training_sample(
    paisa_train_path: Path,
    handle,
    *,
    target_characters: int,
    sample_seed: str,
    progress: ProgressCallback | None,
) -> dict[str, int]:
    total_characters = 0
    document_count = 0
    for document in _iter_separator_delimited_documents(paisa_train_path):
        total_characters += len(document)
        document_count += 1
        if document_count % 50_000 == 0:
            _write_progress(progress, f"scanned PAISÀ train documents={document_count}")
    if total_characters == 0:
        raise ValueError("PAISÀ train text has no documents")

    threshold = min(1.0, target_characters / total_characters)
    selected_documents = 0
    selected_characters = 0
    for document in _iter_separator_delimited_documents(paisa_train_path):
        if _sample_document(document, sample_seed=sample_seed, threshold=threshold):
            _write_document(handle, document)
            selected_documents += 1
            selected_characters += len(document)
    if selected_documents == 0:
        raise ValueError("PAISÀ tokenizer sample selected no training documents")
    return {
        "available_documents": document_count,
        "available_characters": total_characters,
        "target_characters": target_characters,
        "selected_documents": selected_documents,
        "selected_characters": selected_characters,
    }


def _write_historical_training_sample(
    splits: list[HistoricalSourceSplit],
    handle,
    *,
    target_characters: int,
) -> dict[str, object]:
    allocations = _allocate_historical_sample_characters(splits, target_characters)
    sampled_sources: list[dict[str, int | str]] = []
    selected_characters = 0
    for split, allocation in zip(splits, allocations, strict=True):
        sampled_text = _sample_evenly_spaced_text(split.train_text, allocation)
        _write_document(handle, sampled_text)
        selected_characters += len(sampled_text)
        sampled_sources.append(
            {
                "source_id": split.source_id,
                "available_train_characters": len(split.train_text),
                "allocated_characters": allocation,
                "selected_characters": len(sampled_text),
            }
        )
    return {
        "target_characters": target_characters,
        "selected_characters": selected_characters,
        "sources": sampled_sources,
    }


def _allocate_historical_sample_characters(
    splits: list[HistoricalSourceSplit], target_characters: int
) -> list[int]:
    available = [len(split.train_text) for split in splits]
    total_available = sum(available)
    if target_characters <= 0:
        raise ValueError("historical tokenizer sample target must be positive")
    if target_characters >= total_available:
        return available

    per_source_minimum = min(20_000, target_characters // len(splits))
    allocations = [min(size, per_source_minimum) for size in available]
    remaining_budget = target_characters - sum(allocations)
    capacities = [size - allocation for size, allocation in zip(available, allocations, strict=True)]
    total_capacity = sum(capacities)
    for index, capacity in enumerate(capacities):
        share = int(remaining_budget * capacity / total_capacity) if total_capacity else 0
        allocations[index] += min(capacity, share)
    remainder = target_characters - sum(allocations)
    for index in sorted(range(len(splits)), key=lambda value: splits[value].source_id):
        if remainder == 0:
            break
        capacity = available[index] - allocations[index]
        if capacity:
            increment = min(capacity, remainder)
            allocations[index] += increment
            remainder -= increment
    if sum(allocations) != target_characters:
        raise ValueError("could not allocate the historical tokenizer sample budget")
    return allocations


def _sample_evenly_spaced_text(text: str, target_characters: int) -> str:
    if target_characters >= len(text):
        return text
    span_count = min(16, target_characters)
    span_size = target_characters // span_count
    remainder = target_characters % span_count
    spans: list[str] = []
    for index in range(span_count):
        length = span_size + (1 if index < remainder else 0)
        maximum_start = len(text) - length
        start = int(maximum_start * index / (span_count - 1)) if span_count > 1 else 0
        spans.append(text[start : start + length])
    return "\n".join(spans)


def _iter_separator_delimited_documents(path: Path) -> Iterator[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.rstrip("\n") == PAISA_DOCUMENT_SEPARATOR:
                document = "".join(lines).strip()
                if document:
                    yield document
                lines = []
            else:
                lines.append(line)
    if lines:
        raise ValueError(f"PAISÀ train text has an unterminated document: {path}")


def _sample_document(document: str, *, sample_seed: str, threshold: float) -> bool:
    fingerprint = hashlib.sha256(document.encode("utf-8")).hexdigest()
    digest = hashlib.sha256(f"{sample_seed}:{fingerprint}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") < int(threshold * (1 << 64))


def _write_document(handle, text: str) -> None:
    if PAISA_DOCUMENT_SEPARATOR in text:
        raise ValueError("training text contains the reserved document separator")
    handle.write(text.strip())
    handle.write(f"\n{PAISA_DOCUMENT_SEPARATOR}\n")


def _make_report(
    *,
    config: PaisaHistoricalCurriculumConfig,
    started_at: str,
    paisa_report: dict[str, object],
    historical_splits: list[HistoricalSourceSplit],
    paisa_sample: dict[str, int],
    historical_sample: dict[str, object],
    historical_train_path: Path,
    historical_validation_path: Path,
    tokenizer_sample_path: Path,
) -> dict[str, object]:
    historical_train_characters = sum(len(split.train_text) for split in historical_splits)
    historical_validation_characters = sum(
        len(split.validation_text) for split in historical_splits
    )
    document_counts = _required_mapping(paisa_report, "document_counts")
    text_counts = _required_mapping(paisa_report, "text_counts")
    return {
        "curriculum_id": config.curriculum_id,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "provenance": {
            "paisa_build_report_path": str(config.paisa_build_report_path),
            "paisa_build_report_sha256": _file_sha256(config.paisa_build_report_path),
            "historical_mixture_report_path": str(config.historical_mixture_report_path),
            "historical_mixture_report_sha256": _file_sha256(
                config.historical_mixture_report_path
            ),
            "paisa_release_sha256": _paisa_release_sha256(paisa_report),
        },
        "paisa": {
            "train_documents": int(document_counts["train"]),
            "validation_documents": int(document_counts["validation"]),
            "train_characters": int(text_counts["train_characters"]),
            "validation_characters": int(text_counts["validation_characters"]),
            "validation_policy": "existing one-percent document-level fingerprint split",
        },
        "historical": {
            "source_count": len(historical_splits),
            "train_characters": historical_train_characters,
            "validation_characters": historical_validation_characters,
            "validation_fraction_target": config.historical_source_validation_fraction,
            "validation_policy": (
                "final source suffix split at a newline boundary; weaker than PAISÀ's "
                "document-level isolation and reported as such"
            ),
            "sources": [
                {
                    "source_id": split.source_id,
                    "source_path": str(split.source_path),
                    "train_characters": len(split.train_text),
                    "validation_characters": len(split.validation_text),
                }
                for split in historical_splits
            ],
        },
        "tokenizer": {
            "vocab_size": config.tokenizer_vocab_size,
            "special_tokens": list(config.tokenizer_special_tokens),
            "training_policy": "PAISÀ train plus historical train only; no validation text",
            "paisa_sample": paisa_sample,
            "historical_sample": historical_sample,
        },
        "stages": list(config.stages),
        "local_artifacts": {
            "historical_train_path": _final_local_path(config, historical_train_path),
            "historical_validation_path": _final_local_path(config, historical_validation_path),
            "tokenizer_training_sample_path": _final_local_path(config, tokenizer_sample_path),
        },
        "public_repository_policy": (
            "This report contains no PAISÀ document text or document URLs. All PAISÀ "
            "text-bearing curriculum artifacts remain local."
        ),
    }


def _validate_input_provenance(
    config: PaisaHistoricalCurriculumConfig,
    paisa_report: dict[str, object],
    historical_report: dict[str, object],
) -> None:
    _require_equal_sha256(
        config.paisa_build_report_path,
        config.expected_paisa_build_report_sha256,
        label="PAISÀ build report",
    )
    _require_equal_sha256(
        config.historical_mixture_report_path,
        config.expected_historical_mixture_report_sha256,
        label="historical mixture report",
    )
    if _paisa_release_sha256(paisa_report) != config.expected_paisa_release_sha256:
        raise ValueError("PAISÀ release SHA-256 does not match the curriculum configuration")
    if str(historical_report.get("corpus_version")) != "pretraining_historical_italian_v2":
        raise ValueError("curriculum requires pretraining_historical_italian_v2")


def _paisa_split_paths(paisa_report: dict[str, object]) -> tuple[Path, Path]:
    artifacts = _required_mapping(paisa_report, "local_artifacts")
    return Path(str(artifacts["train_text_path"])), Path(str(artifacts["validation_text_path"]))


def _paisa_release_sha256(paisa_report: dict[str, object]) -> str:
    source = _required_mapping(paisa_report, "source")
    release = _required_mapping(source, "release")
    return str(release["sha256"])


def _validate_config(config: PaisaHistoricalCurriculumConfig) -> None:
    if not 0 < config.historical_source_validation_fraction < 1:
        raise ValueError("historical_source_validation_fraction must be between zero and one")
    if config.tokenizer_vocab_size <= 0:
        raise ValueError("tokenizer_vocab_size must be positive")
    if config.tokenizer_special_tokens != (PAISA_DOCUMENT_SEPARATOR,):
        raise ValueError("the rescue tokenizer must use only <|endoftext|> as its special token")
    if config.paisa_train_sample_characters <= 0:
        raise ValueError("paisa_train_sample_characters must be positive")
    if config.historical_train_sample_characters <= 0:
        raise ValueError("historical_train_sample_characters must be positive")
    if not config.sample_seed:
        raise ValueError("sample_seed must not be empty")
    _validate_stage_policy(config.stages)


def _validate_stage_policy(stages: tuple[dict[str, object], ...]) -> None:
    expected = [
        ("modern_italian_pretraining", "paisa_train", 3),
        ("historical_italian_annealing", "historical_train", 12),
    ]
    actual = [
        (str(stage.get("stage_id")), str(stage.get("dataset")), stage.get("max_passes"))
        for stage in stages[:2]
    ]
    if actual != expected:
        raise ValueError("curriculum must begin with the approved three-pass PAISÀ and twelve-pass historical stages")


def _replace_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _publish_directory(stage_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(stage_dir), str(output_dir))


def _final_local_path(config: PaisaHistoricalCurriculumConfig, stage_path: Path) -> str:
    return str(config.local_output_dir / stage_path.name)


def _require_equal_sha256(path: Path, expected: str, *, label: str) -> None:
    actual = _file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 does not match the curriculum configuration")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    _require_file(path, label="JSON report")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _require_file(path: Path, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")


def _required_mapping(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"expected object field: {key}")
    return value


def _required_list(payload: dict[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"expected non-empty list field: {key}")
    return value


def _as_mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
