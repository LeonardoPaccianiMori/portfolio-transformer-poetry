"""Fit the locked train-only BPE tokenizer for the PAISÀ rescue curriculum."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .paisa_build import PAISA_DOCUMENT_SEPARATOR
from .pretraining_curriculum import load_paisa_historical_curriculum_config
from .pretraining_tokenizer import PretrainingTokenizerConfig
from .pretraining_tokenizer import train_pretraining_bpe_tokenizer


ProgressCallback = Callable[[str], None]
_READ_CHUNK_SIZE = 1_048_576


@dataclass(frozen=True)
class PaisaHistoricalTokenizerConfig:
    """Paths and resumability settings for one fixed rescue tokenizer fit."""

    curriculum_config_path: Path
    tokenizer_path: Path | None = None
    local_report_path: Path | None = None
    public_report_path: Path | None = None
    training_checkpoint_path: Path | None = None
    merge_progress_interval: int = 500
    max_merges_per_run: int | None = None


def train_paisa_historical_rescue_tokenizer(
    config: PaisaHistoricalTokenizerConfig,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Fit or resume the curriculum's BPE tokenizer without using validation text."""

    curriculum = load_paisa_historical_curriculum_config(config.curriculum_config_path)
    paths = _resolve_curriculum_paths(curriculum)
    resolved = _resolve_output_paths(config, curriculum.local_output_dir, curriculum.report_path)
    _validate_runtime_config(config)

    _write_progress(progress, "validating locked curriculum provenance")
    provenance = _validate_provenance(curriculum, paths)
    _write_progress(progress, "scanning PAISÀ and historical training characters")
    training_characters = _collect_characters(
        [paths["paisa_train"], paths["historical_train"]],
        progress=progress,
    )
    base_vocabulary_tokens = _build_base_vocabulary_tokens(
        training_characters,
        special_tokens=curriculum.tokenizer_special_tokens,
    )
    tokenizable_paisa_validation_path = (
        curriculum.local_output_dir / "paisa_validation_tokenizable.txt"
    )
    _write_progress(progress, "deriving tokenizable PAISÀ validation documents")
    paisa_validation_sanitization = _write_tokenizable_paisa_validation(
        source_path=paths["paisa_validation_original"],
        output_path=tokenizable_paisa_validation_path,
        training_characters=training_characters,
        progress=progress,
    )
    paths["paisa_validation_tokenizable"] = tokenizable_paisa_validation_path
    _write_progress(progress, "checking held-out character coverage")
    coverage = _validate_validation_character_coverage(
        training_characters=training_characters,
        validation_paths=[
            paths["paisa_validation_tokenizable"],
            paths["historical_validation"],
        ],
        progress=progress,
    )
    sample_path = paths["tokenizer_sample"]
    sample_text = _read_non_empty_text(sample_path)
    _validate_sample_character_coverage(
        training_characters=training_characters,
        sample_text=sample_text,
    )

    tokenizer_report_path = resolved["local_report_path"].with_name(
        "tokenizer_training_detail.json"
    )
    tokenizer_config = PretrainingTokenizerConfig(
        corpus_path=sample_path,
        tokenizer_path=resolved["tokenizer_path"],
        report_path=tokenizer_report_path,
        build_report_path=curriculum.report_path,
        vocab_size=curriculum.tokenizer_vocab_size,
        special_tokens=curriculum.tokenizer_special_tokens,
        training_character_limit=len(sample_text),
        merge_progress_interval=config.merge_progress_interval,
        training_checkpoint_path=resolved["training_checkpoint_path"],
        max_merges_per_run=config.max_merges_per_run,
        base_vocabulary_tokens=base_vocabulary_tokens,
        reuse_completed_tokenizer=True,
    )
    _write_progress(progress, "training BPE merges from the fixed train-only sample")
    detail_report = train_pretraining_bpe_tokenizer(
        tokenizer_config,
        progress=progress,
    )
    report = _make_report(
        curriculum_id=curriculum.curriculum_id,
        curriculum_config_path=config.curriculum_config_path,
        provenance=provenance,
        paths=paths,
        resolved=resolved,
        detail_report=detail_report,
        base_vocabulary_tokens=base_vocabulary_tokens,
        coverage=coverage,
        paisa_validation_sanitization=paisa_validation_sanitization,
    )
    _write_json(resolved["local_report_path"], report)
    _write_json(resolved["public_report_path"], report)
    _write_progress(progress, f"wrote local tokenizer report: {resolved['local_report_path']}")
    _write_progress(progress, f"wrote public tokenizer report: {resolved['public_report_path']}")
    return report


def _resolve_curriculum_paths(curriculum) -> dict[str, Path]:
    paisa_report = _read_json(curriculum.paisa_build_report_path)
    curriculum_report = _read_json(curriculum.report_path)
    paisa_artifacts = _required_mapping(paisa_report, "local_artifacts")
    local_artifacts = _required_mapping(curriculum_report, "local_artifacts")
    paths = {
        "paisa_train": Path(str(paisa_artifacts["train_text_path"])),
        "paisa_validation_original": Path(str(paisa_artifacts["validation_text_path"])),
        "historical_train": Path(str(local_artifacts["historical_train_path"])),
        "historical_validation": Path(str(local_artifacts["historical_validation_path"])),
        "tokenizer_sample": Path(str(local_artifacts["tokenizer_training_sample_path"])),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {label.replace('_', ' ')}: {path}")
    return paths


def _resolve_output_paths(
    config: PaisaHistoricalTokenizerConfig,
    local_output_dir: Path,
    curriculum_report_path: Path,
) -> dict[str, Path]:
    report_stem = curriculum_report_path.stem.removesuffix("_curriculum_report")
    return {
        "tokenizer_path": config.tokenizer_path or local_output_dir / "tokenizer.json",
        "local_report_path": config.local_report_path or local_output_dir / "tokenizer_report.json",
        "public_report_path": config.public_report_path
        or curriculum_report_path.with_name(f"{report_stem}_tokenizer_report.json"),
        "training_checkpoint_path": config.training_checkpoint_path
        or local_output_dir / "tokenizer_training_state.json",
    }


def _validate_runtime_config(config: PaisaHistoricalTokenizerConfig) -> None:
    if config.merge_progress_interval <= 0:
        raise ValueError("merge_progress_interval must be positive")
    if config.max_merges_per_run is not None and config.max_merges_per_run <= 0:
        raise ValueError("max_merges_per_run must be positive")


def _validate_provenance(curriculum, paths: dict[str, Path]) -> dict[str, str]:
    paisa_report = _read_json(curriculum.paisa_build_report_path)
    _require_sha256(
        curriculum.paisa_build_report_path,
        curriculum.expected_paisa_build_report_sha256,
        label="PAISÀ build report",
    )
    _require_sha256(
        curriculum.historical_mixture_report_path,
        curriculum.expected_historical_mixture_report_sha256,
        label="historical mixture report",
    )
    release_sha256 = str(
        _required_mapping(_required_mapping(paisa_report, "source"), "release")["sha256"]
    )
    if release_sha256 != curriculum.expected_paisa_release_sha256:
        raise ValueError("PAISÀ release SHA-256 does not match the curriculum configuration")

    sample_sha256 = _file_sha256(paths["tokenizer_sample"])
    if not sample_sha256:
        raise ValueError("tokenizer training sample has no SHA-256")
    return {
        "paisa_build_report_sha256": _file_sha256(curriculum.paisa_build_report_path),
        "historical_mixture_report_sha256": _file_sha256(
            curriculum.historical_mixture_report_path
        ),
        "paisa_release_sha256": release_sha256,
        "tokenizer_sample_sha256": sample_sha256,
    }


def _collect_characters(
    paths: list[Path],
    *,
    progress: ProgressCallback | None,
) -> set[str]:
    characters: set[str] = set()
    for index, path in enumerate(paths, start=1):
        _write_progress(progress, f"scanning train split {index}/{len(paths)}: {path.name}")
        with path.open("r", encoding="utf-8") as handle:
            while chunk := handle.read(_READ_CHUNK_SIZE):
                characters.update(chunk)
    if not characters:
        raise ValueError("training splits contain no characters")
    return characters


def _build_base_vocabulary_tokens(
    training_characters: set[str],
    *,
    special_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    if special_tokens != (PAISA_DOCUMENT_SEPARATOR,):
        raise ValueError("rescue tokenizer must use only <|endoftext|> as its special token")
    return (*special_tokens, *sorted(training_characters))


def _validate_validation_character_coverage(
    *,
    training_characters: set[str],
    validation_paths: list[Path],
    progress: ProgressCallback | None,
) -> dict[str, object]:
    validation_characters: set[str] = set()
    for index, path in enumerate(validation_paths, start=1):
        _write_progress(progress, f"scanning validation split {index}/{len(validation_paths)}: {path.name}")
        with path.open("r", encoding="utf-8") as handle:
            while chunk := handle.read(_READ_CHUNK_SIZE):
                validation_characters.update(chunk)
    uncovered = sorted(validation_characters - training_characters)
    if uncovered:
        formatted = ", ".join(f"U+{ord(character):04X}" for character in uncovered)
        raise ValueError(
            "validation contains characters absent from training vocabulary: " + formatted
        )
    return {
        "training_unique_characters": len(training_characters),
        "validation_unique_characters": len(validation_characters),
        "uncovered_validation_character_count": 0,
    }


def _write_tokenizable_paisa_validation(
    *,
    source_path: Path,
    output_path: Path,
    training_characters: set[str],
    progress: ProgressCallback | None,
) -> dict[str, object]:
    """Exclude only validation documents with characters absent from training."""

    input_documents = 0
    retained_documents = 0
    excluded_documents = 0
    retained_characters = 0
    excluded_characters = 0
    excluded_codepoints: set[str] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as output:
        lines: list[str] = []
        for line in source:
            if line.rstrip("\n") != PAISA_DOCUMENT_SEPARATOR:
                lines.append(line)
                continue
            document = "".join(lines).strip()
            if not document:
                raise ValueError(f"PAISÀ validation has an empty document: {source_path}")
            input_documents += 1
            uncovered = set(document) - training_characters
            if uncovered:
                excluded_documents += 1
                excluded_characters += len(document)
                excluded_codepoints.update(f"U+{ord(character):04X}" for character in uncovered)
            else:
                output.write(document)
                output.write(f"\n{PAISA_DOCUMENT_SEPARATOR}\n")
                retained_documents += 1
                retained_characters += len(document)
            lines = []
            if input_documents % 1_000 == 0:
                _write_progress(
                    progress,
                    "processed PAISÀ validation documents="
                    f"{input_documents} excluded={excluded_documents}",
                )
    if lines:
        raise ValueError(f"PAISÀ validation has an unterminated document: {source_path}")
    if retained_documents == 0:
        raise ValueError("PAISÀ validation sanitizer excluded every document")
    return {
        "policy": (
            "exclude only PAISÀ validation documents containing a character absent from "
            "both fixed training splits; original validation text remains unchanged"
        ),
        "input_documents": input_documents,
        "retained_documents": retained_documents,
        "excluded_documents": excluded_documents,
        "retained_characters": retained_characters,
        "excluded_characters": excluded_characters,
        "excluded_codepoints": sorted(excluded_codepoints),
    }


def _validate_sample_character_coverage(
    *,
    training_characters: set[str],
    sample_text: str,
) -> None:
    uncovered = sorted(set(sample_text) - training_characters)
    if uncovered:
        formatted = ", ".join(f"U+{ord(character):04X}" for character in uncovered)
        raise ValueError(
            "tokenizer sample contains characters absent from training vocabulary: " + formatted
        )


def _make_report(
    *,
    curriculum_id: str,
    curriculum_config_path: Path,
    provenance: dict[str, str],
    paths: dict[str, Path],
    resolved: dict[str, Path],
    detail_report: dict[str, Any],
    base_vocabulary_tokens: tuple[str, ...],
    coverage: dict[str, object],
    paisa_validation_sanitization: dict[str, object],
) -> dict[str, object]:
    return {
        "curriculum_id": curriculum_id,
        "status": detail_report["status"],
        "finished_at_utc": _utc_now(),
        "curriculum_config_path": _portable_path(curriculum_config_path),
        "provenance": provenance,
        "tokenizer": {
            "target_vocab_size": detail_report["target_vocab_size"],
            "actual_vocab_size": detail_report["actual_vocab_size"],
            "base_vocabulary_size": len(base_vocabulary_tokens),
            "merge_count": detail_report["merge_count"],
            "special_tokens": [PAISA_DOCUMENT_SEPARATOR],
            "merge_statistics_policy": "exact fixed PAISÀ-plus-historical train-only sample",
            "base_vocabulary_policy": "all PAISÀ and historical training characters only",
        },
        "sample": {
            "path": _portable_path(paths["tokenizer_sample"]),
            "characters": detail_report["training_character_count"],
            "sha256": provenance["tokenizer_sample_sha256"],
        },
        "character_coverage": coverage,
        "paisa_validation_sanitization": paisa_validation_sanitization,
        "local_artifacts": {
            "tokenizer_path": _portable_path(resolved["tokenizer_path"]),
            "training_detail_report_path": _portable_path(
                resolved["local_report_path"].with_name("tokenizer_training_detail.json")
            ),
            "checkpoint_path": _portable_path(resolved["training_checkpoint_path"]),
            "paisa_train_path": _portable_path(paths["paisa_train"]),
            "paisa_validation_original_path": _portable_path(
                paths["paisa_validation_original"]
            ),
            "paisa_validation_tokenizable_path": _portable_path(
                paths["paisa_validation_tokenizable"]
            ),
            "historical_train_path": _portable_path(paths["historical_train"]),
            "historical_validation_path": _portable_path(paths["historical_validation"]),
        },
        "public_repository_policy": (
            "This aggregate report contains no PAISÀ document text or document URLs. "
            "The tokenizer and PAISÀ-derived artifacts remain local under the source terms."
        ),
    }


def _read_non_empty_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"tokenizer training sample is empty: {path}")
    return text


def _require_sha256(path: Path, expected: str, *, label: str) -> None:
    if _file_sha256(path) != expected:
        raise ValueError(f"{label} SHA-256 does not match the curriculum configuration")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _required_mapping(payload: dict[str, object], field: str) -> dict[str, object]:
    value = payload.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object is missing mapping field: {field}")
    return value


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _write_progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
