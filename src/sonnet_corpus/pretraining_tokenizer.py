"""Train and report the broader-corpus BPE tokenizer."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .bpe import (
    BytePairEncodingTokenizer,
    TokenPair,
    build_base_vocabulary,
    choose_best_pair,
    merge_token_pair,
)
from .pretraining_build import select_pretraining_build_rows
from .pretraining_manifest import PretrainingSourceRow, read_pretraining_manifest


DEFAULT_PRETRAINING_SPECIAL_TOKENS = ("<|endoftext|>",)
PRETOKEN_PATTERN = re.compile(r"\S+|\s+")
BOUNDARY_WARNING_PATTERNS = (
    "project gutenberg",
    "liber liber",
    "progetto manuzio",
    "nota di trascrizione",
    "indice generale",
)
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class PretrainingTokenizerConfig:
    """Configuration for broader-corpus BPE tokenizer training."""

    corpus_path: Path = Path("data/local/pretraining/processed/corpus.txt")
    tokenizer_path: Path = Path("data/local/pretraining/tokenizers/bpe_8000.json")
    report_path: Path = Path("data/local/pretraining/tokenizers/bpe_8000_report.json")
    build_report_path: Path = Path("data/local/pretraining/build_report.json")
    vocab_size: int = 8000
    special_tokens: tuple[str, ...] = DEFAULT_PRETRAINING_SPECIAL_TOKENS
    training_character_limit: int = 100_000
    manifest_path: Path | None = None
    source_dir: Path | None = None
    minimum_source_characters: int = 0
    merge_progress_interval: int = 500


def train_pretraining_bpe_tokenizer(
    config: PretrainingTokenizerConfig,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Train the first broader-corpus BPE tokenizer and write a JSON report."""

    started_at = _utc_now()
    text = _read_non_empty_text(config.corpus_path)
    training_text, sample_sources, sampling_strategy = _select_training_text(
        config,
        base_text=text,
        progress=progress,
    )
    if not training_text.strip():
        raise ValueError("training text sample is empty")

    _write_progress(progress, "training BPE merges")
    tokenizer = train_weighted_pretoken_bpe_tokenizer(
        training_text=training_text,
        base_text=text,
        vocab_size=config.vocab_size,
        special_tokens=list(config.special_tokens),
        progress=progress,
        progress_interval=config.merge_progress_interval,
    )
    _write_progress(progress, "counting BPE tokens across the full corpus")
    token_count = count_bpe_tokens_by_pretoken(text, tokenizer, progress=progress)
    _write_progress(progress, "checking tokenizer round trips")
    sample_results = _round_trip_samples(text, tokenizer)
    report = {
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "corpus_path": _portable_path(config.corpus_path),
        "tokenizer_path": _portable_path(config.tokenizer_path),
        "build_report_path": _portable_path(config.build_report_path),
        "target_vocab_size": config.vocab_size,
        "actual_vocab_size": tokenizer.vocab_size,
        "merge_count": len(tokenizer.merges),
        "special_tokens": list(config.special_tokens),
        "corpus_character_count": len(text),
        "training_character_limit": config.training_character_limit,
        "training_character_count": len(training_text),
        "sampling_strategy": sampling_strategy,
        "minimum_source_characters": config.minimum_source_characters,
        "sample_sources": sample_sources,
        "token_count": token_count,
        "characters_per_token": len(text) / token_count,
        "boundary_warnings": inspect_build_report_boundaries(config.build_report_path),
        "round_trip_samples": sample_results,
    }

    if not all(item["round_trip_ok"] for item in sample_results):
        raise ValueError("tokenizer failed a round-trip sample check")

    _write_progress(progress, "writing tokenizer and report")
    config.tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(config.tokenizer_path)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _select_training_text(
    config: PretrainingTokenizerConfig,
    *,
    base_text: str,
    progress: ProgressCallback | None,
) -> tuple[str, list[dict[str, int | str]], str]:
    has_manifest = config.manifest_path is not None
    has_source_dir = config.source_dir is not None
    if has_manifest != has_source_dir:
        raise ValueError("manifest_path and source_dir must be provided together")

    if not has_manifest:
        return base_text[: config.training_character_limit], [], "corpus_prefix"

    assert config.manifest_path is not None
    assert config.source_dir is not None
    return _build_stratified_training_sample(
        manifest_path=config.manifest_path,
        source_dir=config.source_dir,
        training_character_limit=config.training_character_limit,
        minimum_source_characters=config.minimum_source_characters,
        progress=progress,
    )


def _build_stratified_training_sample(
    *,
    manifest_path: Path,
    source_dir: Path,
    training_character_limit: int,
    minimum_source_characters: int,
    progress: ProgressCallback | None,
) -> tuple[str, list[dict[str, int | str]], str]:
    if training_character_limit <= 0:
        raise ValueError("training_character_limit must be greater than 0")
    if minimum_source_characters < 0:
        raise ValueError("minimum_source_characters cannot be negative")

    rows = select_pretraining_build_rows(read_pretraining_manifest(manifest_path))
    if not rows:
        raise ValueError("no active broader pretraining prose rows selected")
    _validate_source_files(source_dir, rows)

    source_texts: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        _write_progress(progress, f"reading sample source {index}/{len(rows)}: {row.source_id}")
        source_texts[row.source_id] = (source_dir / f"{row.source_id}.txt").read_text(
            encoding="utf-8"
        )

    available = {source_id: len(text) for source_id, text in source_texts.items()}
    total_available = sum(available.values())
    if training_character_limit > total_available:
        raise ValueError("training_character_limit exceeds available source characters")

    allocations = _allocate_source_characters(
        available,
        training_character_limit=training_character_limit,
        minimum_source_characters=minimum_source_characters,
    )
    sampled_parts: list[str] = []
    sample_sources: list[dict[str, int | str]] = []
    for index, row in enumerate(rows, start=1):
        source_id = row.source_id
        sampled_text = _sample_text_evenly(source_texts[source_id], allocations[source_id])
        sampled_parts.append(sampled_text)
        sample_sources.append(
            {
                "source_id": source_id,
                "available_character_count": available[source_id],
                "allocated_character_count": allocations[source_id],
                "sampled_character_count": len(sampled_text),
            }
        )
        _write_progress(
            progress,
            f"sampled source {index}/{len(rows)}: {source_id} ({len(sampled_text)} characters)",
        )

    return "\n".join(sampled_parts), sample_sources, "stratified_sources"


def _allocate_source_characters(
    available: dict[str, int],
    *,
    training_character_limit: int,
    minimum_source_characters: int,
) -> dict[str, int]:
    if training_character_limit < len(available):
        raise ValueError("training_character_limit is too small to represent every source")

    allocations = {
        source_id: min(minimum_source_characters, character_count)
        for source_id, character_count in available.items()
    }
    remaining = training_character_limit - sum(allocations.values())
    if remaining < 0:
        raise ValueError(
            "training_character_limit is smaller than the requested source minimums"
        )

    capacity = {
        source_id: available[source_id] - allocations[source_id]
        for source_id in available
    }
    total_capacity = sum(capacity.values())
    if remaining > total_capacity:
        raise ValueError("training_character_limit exceeds available source characters")
    if remaining == 0:
        return allocations

    exact_additions = {
        source_id: remaining * source_capacity / total_capacity
        for source_id, source_capacity in capacity.items()
    }
    for source_id, exact_addition in exact_additions.items():
        allocations[source_id] += int(exact_addition)

    unassigned = training_character_limit - sum(allocations.values())
    ranked_remainders = sorted(
        capacity,
        key=lambda source_id: (
            -(exact_additions[source_id] - int(exact_additions[source_id])),
            source_id,
        ),
    )
    for source_id in ranked_remainders[:unassigned]:
        allocations[source_id] += 1
    return allocations


def _sample_text_evenly(text: str, character_count: int) -> str:
    if character_count <= 0:
        return ""
    if character_count >= len(text):
        return text

    chunk_count = min(16, max(1, character_count // 2_000))
    chunk_sizes = _split_character_count(character_count, chunk_count)
    chunks: list[str] = []
    for index, chunk_size in enumerate(chunk_sizes):
        if len(chunk_sizes) == 1:
            start = (len(text) - chunk_size) // 2
        else:
            start = (len(text) - chunk_size) * index // (len(chunk_sizes) - 1)
        chunks.append(_sample_word_aligned_chunk(text, start, chunk_size))
    return "\n".join(chunks)


def _split_character_count(character_count: int, chunk_count: int) -> list[int]:
    base_size, remainder = divmod(character_count, chunk_count)
    return [base_size + (index < remainder) for index in range(chunk_count)]


def _sample_word_aligned_chunk(text: str, start: int, character_count: int) -> str:
    end = min(start + character_count, len(text))
    if start > 0:
        next_boundary = _next_whitespace(text, start)
        if next_boundary is not None and next_boundary < end:
            start = next_boundary
    if end < len(text):
        previous_boundary = _previous_whitespace(text, end)
        if previous_boundary is not None and previous_boundary > start:
            end = previous_boundary
    return text[start:end]


def _next_whitespace(text: str, start: int) -> int | None:
    match = re.search(r"\s", text[start:])
    return start + match.start() if match is not None else None


def _previous_whitespace(text: str, end: int) -> int | None:
    match = re.search(r"\s(?=\S*$)", text[:end])
    return match.start() if match is not None else None


def _validate_source_files(source_dir: Path, rows: list[PretrainingSourceRow]) -> None:
    if not source_dir.is_dir():
        raise ValueError(f"processed source directory does not exist: {source_dir}")

    expected_ids = {row.source_id for row in rows}
    actual_ids = {path.stem for path in source_dir.glob("*.txt")}
    missing_ids = sorted(expected_ids - actual_ids)
    unexpected_ids = sorted(actual_ids - expected_ids)
    if missing_ids or unexpected_ids:
        details = []
        if missing_ids:
            details.append(f"missing={','.join(missing_ids)}")
        if unexpected_ids:
            details.append(f"unexpected={','.join(unexpected_ids)}")
        raise ValueError("processed source files do not match active manifest rows: " + "; ".join(details))


def train_weighted_pretoken_bpe_tokenizer(
    *,
    training_text: str,
    base_text: str,
    vocab_size: int,
    special_tokens: list[str],
    progress: ProgressCallback | None = None,
    progress_interval: int = 500,
) -> BytePairEncodingTokenizer:
    """Train BPE merges from weighted word/whitespace pretoken counts."""

    if vocab_size <= 0:
        raise ValueError("vocab_size must be greater than 0")
    if progress_interval <= 0:
        raise ValueError("progress_interval must be greater than 0")

    vocabulary_tokens = build_base_vocabulary(
        texts=[base_text],
        special_tokens=special_tokens,
    )
    if vocab_size < len(vocabulary_tokens):
        raise ValueError("vocab_size must be at least the base vocabulary size")

    sequence_counts = _count_initial_pretoken_sequences(training_text)
    merges: list[TokenPair] = []
    vocabulary_set = set(vocabulary_tokens)
    _write_progress(
        progress,
        "BPE initial state: "
        f"base_vocabulary={len(vocabulary_tokens)} "
        f"unique_pretokens={len(sequence_counts)}",
    )

    while len(vocabulary_tokens) < vocab_size:
        pair_counts = _weighted_pair_counts(sequence_counts)
        best_pair = choose_best_pair(pair_counts)
        if best_pair is None:
            break

        merged_token = "".join(best_pair)
        if merged_token in special_tokens:
            break

        sequence_counts = _merge_sequence_counts(
            sequence_counts=sequence_counts,
            pair=best_pair,
            merged_token=merged_token,
        )
        merges.append(best_pair)

        if merged_token not in vocabulary_set:
            vocabulary_tokens.append(merged_token)
            vocabulary_set.add(merged_token)

        if len(merges) % progress_interval == 0 or len(vocabulary_tokens) == vocab_size:
            _write_progress(
                progress,
                "BPE merges: "
                f"vocabulary={len(vocabulary_tokens)}/{vocab_size} "
                f"merges={len(merges)}",
            )

    token_to_id = {
        token: token_id
        for token_id, token in enumerate(vocabulary_tokens)
    }
    return BytePairEncodingTokenizer(
        token_to_id=token_to_id,
        merges=merges,
        special_tokens=special_tokens,
    )


def count_bpe_tokens_by_pretoken(
    text: str,
    tokenizer: BytePairEncodingTokenizer,
    *,
    progress: ProgressCallback | None = None,
    progress_interval: int = 10_000,
) -> int:
    """Count BPE tokens exactly using repeated pretoken strings as a cache."""

    if progress_interval <= 0:
        raise ValueError("progress_interval must be greater than 0")
    pretoken_counts = Counter(_iter_pretokens(text))
    merge_ranks = _merge_ranks(tokenizer)
    total = 0
    item_count = len(pretoken_counts)
    for index, (pretoken, count) in enumerate(pretoken_counts.items(), start=1):
        total += len(_encode_pretoken(pretoken, merge_ranks)) * count
        if index % progress_interval == 0 or index == item_count:
            _write_progress(
                progress,
                f"token count cache entries: {index}/{item_count}",
            )
    return total


def encode_text_by_pretoken(
    text: str,
    tokenizer: BytePairEncodingTokenizer,
) -> list[int]:
    """Encode text with the same pretoken cache strategy used in reports."""

    merge_ranks = _merge_ranks(tokenizer)
    encoded: list[int] = []
    encoded_pretokens: dict[str, list[int]] = {}
    for pretoken in _iter_pretokens(text):
        if pretoken not in encoded_pretokens:
            encoded_pretokens[pretoken] = [
                tokenizer.token_to_id[token]
                for token in _encode_pretoken(pretoken, merge_ranks)
            ]
        encoded.extend(encoded_pretokens[pretoken])
    return encoded


def inspect_build_report_boundaries(build_report_path: Path) -> list[dict[str, str]]:
    """Return boundary-sample warnings from the broader-corpus build report."""

    if not build_report_path.is_file():
        return []

    payload = json.loads(build_report_path.read_text(encoding="utf-8"))
    warnings: list[dict[str, str]] = []
    for source in payload.get("sources", []):
        for field in ("first_characters", "last_characters"):
            sample = str(source.get(field, ""))
            matched_pattern = _first_boundary_warning_pattern(sample)
            if matched_pattern is not None:
                warnings.append({
                    "source_id": str(source.get("source_id", "")),
                    "field": field,
                    "pattern": matched_pattern,
                    "sample": sample,
                })
    return warnings


def _read_non_empty_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"corpus file does not exist: {path}")

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"corpus file is empty: {path}")
    return text


def _count_initial_pretoken_sequences(text: str) -> Counter[tuple[str, ...]]:
    return Counter(tuple(pretoken) for pretoken in _iter_pretokens(text))


def _iter_pretokens(text: str) -> list[str]:
    return PRETOKEN_PATTERN.findall(text)


def _weighted_pair_counts(
    sequence_counts: Counter[tuple[str, ...]],
) -> Counter[TokenPair]:
    pair_counts: Counter[TokenPair] = Counter()
    for sequence, count in sequence_counts.items():
        for left_token, right_token in zip(sequence, sequence[1:]):
            pair_counts[(left_token, right_token)] += count
    return pair_counts


def _merge_sequence_counts(
    *,
    sequence_counts: Counter[tuple[str, ...]],
    pair: TokenPair,
    merged_token: str,
) -> Counter[tuple[str, ...]]:
    updated_counts: Counter[tuple[str, ...]] = Counter()
    for sequence, count in sequence_counts.items():
        merged_sequence = tuple(
            merge_token_pair(
                token_sequence=list(sequence),
                pair=pair,
                merged_token=merged_token,
            )
        )
        updated_counts[merged_sequence] += count
    return updated_counts


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


def _merge_ranks(tokenizer: BytePairEncodingTokenizer) -> dict[TokenPair, int]:
    return {
        pair: rank
        for rank, pair in enumerate(tokenizer.merges)
    }


def _round_trip_samples(
    text: str,
    tokenizer: BytePairEncodingTokenizer,
    *,
    sample_size: int = 500,
) -> list[dict[str, Any]]:
    samples = {
        "start": text[:sample_size],
        "middle": _middle_sample(text, sample_size),
        "end": text[-sample_size:],
    }
    results = []
    for name, sample in samples.items():
        decoded = tokenizer.decode(tokenizer.encode(sample))
        results.append({
            "name": name,
            "character_count": len(sample),
            "token_count": len(tokenizer.encode(sample)),
            "round_trip_ok": decoded == sample,
        })
    return results


def _middle_sample(text: str, sample_size: int) -> str:
    start = max((len(text) - sample_size) // 2, 0)
    return text[start : start + sample_size]


def _first_boundary_warning_pattern(sample: str) -> str | None:
    normalized = sample.casefold()
    for pattern in BOUNDARY_WARNING_PATTERNS:
        if pattern in normalized:
            return pattern
    return None


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _write_progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
