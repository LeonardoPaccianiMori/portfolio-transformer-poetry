"""Train and report the broader-corpus BPE tokenizer."""

from __future__ import annotations

import json
import re
from collections import Counter
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


DEFAULT_PRETRAINING_SPECIAL_TOKENS = ("<|endoftext|>",)
PRETOKEN_PATTERN = re.compile(r"\S+|\s+")
BOUNDARY_WARNING_PATTERNS = (
    "project gutenberg",
    "liber liber",
    "progetto manuzio",
    "nota di trascrizione",
    "indice generale",
)


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


def train_pretraining_bpe_tokenizer(
    config: PretrainingTokenizerConfig,
) -> dict[str, Any]:
    """Train the first broader-corpus BPE tokenizer and write a JSON report."""

    started_at = _utc_now()
    text = _read_non_empty_text(config.corpus_path)
    training_text = text[: config.training_character_limit]
    if not training_text.strip():
        raise ValueError("training text sample is empty")

    tokenizer = train_weighted_pretoken_bpe_tokenizer(
        training_text=training_text,
        base_text=text,
        vocab_size=config.vocab_size,
        special_tokens=list(config.special_tokens),
    )
    token_count = count_bpe_tokens_by_pretoken(text, tokenizer)
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
        "token_count": token_count,
        "characters_per_token": len(text) / token_count,
        "boundary_warnings": inspect_build_report_boundaries(config.build_report_path),
        "round_trip_samples": sample_results,
    }

    if not all(item["round_trip_ok"] for item in sample_results):
        raise ValueError("tokenizer failed a round-trip sample check")

    config.tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(config.tokenizer_path)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def train_weighted_pretoken_bpe_tokenizer(
    *,
    training_text: str,
    base_text: str,
    vocab_size: int,
    special_tokens: list[str],
) -> BytePairEncodingTokenizer:
    """Train BPE merges from weighted word/whitespace pretoken counts."""

    if vocab_size <= 0:
        raise ValueError("vocab_size must be greater than 0")

    vocabulary_tokens = build_base_vocabulary(
        texts=[base_text],
        special_tokens=special_tokens,
    )
    if vocab_size < len(vocabulary_tokens):
        raise ValueError("vocab_size must be at least the base vocabulary size")

    sequence_counts = _count_initial_pretoken_sequences(training_text)
    merges: list[TokenPair] = []
    vocabulary_set = set(vocabulary_tokens)

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
) -> int:
    """Count BPE tokens exactly using repeated pretoken strings as a cache."""

    pretoken_counts = Counter(_iter_pretokens(text))
    merge_ranks = _merge_ranks(tokenizer)
    return sum(
        len(_encode_pretoken(pretoken, merge_ranks)) * count
        for pretoken, count in pretoken_counts.items()
    )


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
