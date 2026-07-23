"""Compare versioned sonnet corpora under one fixed fine-tuning tokenizer."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

from sonnet_training.steps import sequential_next_token_window_count

from .bpe import BytePairEncodingTokenizer
from .dataset_text import (
    encode_poem_texts_with_pretraining_tokenizer,
    extend_tokenizer_for_character_coverage,
    load_poem_texts,
    read_manifest_rows,
    select_manifest_rows,
    validate_manifest_rows,
)


ProgressCallback = Callable[[str], None]
SPLITS = ("train", "validation", "test")


def write_sonnet_corpus_scaling_report(
    *,
    repo_root: Path,
    manifest_paths: Mapping[str, Path],
    tokenizer_path: Path,
    output_path: Path,
    dataset: str = "expanded_with_petrarch",
    context_length: int = 512,
    parent_checkpoint_path: str = (
        "runs/pretraining_quality_swiglu_larger_200k_001/best_validation.pt"
    ),
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Measure corpus sizes and write the controlled-scaling experiment report."""
    if not manifest_paths:
        raise ValueError("manifest_paths must contain at least one corpus version")
    if len(set(manifest_paths)) != len(manifest_paths):
        raise ValueError("corpus version labels must be unique")
    if context_length <= 0:
        raise ValueError("context_length must be greater than 0")

    resolved_tokenizer_path = _resolve_path(repo_root, tokenizer_path)
    if not resolved_tokenizer_path.is_file():
        raise FileNotFoundError(
            f"pretraining tokenizer does not exist: {resolved_tokenizer_path}"
        )
    parent_tokenizer = BytePairEncodingTokenizer.load(resolved_tokenizer_path)
    version_summaries = []
    test_ids_by_version: dict[str, set[str]] = {}

    for label, manifest_path in manifest_paths.items():
        _write_progress(progress, f"measuring corpus version: {label}")
        resolved_manifest_path = _resolve_path(repo_root, manifest_path)
        if not resolved_manifest_path.is_file():
            raise FileNotFoundError(
                f"sonnet manifest does not exist: {resolved_manifest_path}"
            )
        summary, test_ids = _summarize_version(
            label=label,
            manifest_path=resolved_manifest_path,
            repo_root=repo_root,
            dataset=dataset,
            context_length=context_length,
            tokenizer_path=resolved_tokenizer_path,
        )
        version_summaries.append(summary)
        test_ids_by_version[label] = test_ids

    overlap_summaries = _build_test_overlap_summaries(test_ids_by_version)
    report = {
        "dataset": dataset,
        "context_length": context_length,
        "parent_checkpoint_path": parent_checkpoint_path,
        "tokenizer_path": _portable_path(resolved_tokenizer_path, repo_root),
        "tokenizer_sha256": _file_sha256(resolved_tokenizer_path),
        "parent_vocab_size": parent_tokenizer.vocab_size,
        "versions": version_summaries,
        "test_set_overlaps": overlap_summaries,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_markdown(report), encoding="utf-8")
    _write_progress(progress, f"wrote scaling report: {output_path}")
    return report


def _summarize_version(
    *,
    label: str,
    manifest_path: Path,
    repo_root: Path,
    dataset: str,
    context_length: int,
    tokenizer_path: Path,
) -> tuple[dict[str, Any], set[str]]:
    rows = read_manifest_rows(manifest_path)
    validate_manifest_rows(rows, dataset)
    split_rows = {
        split: select_manifest_rows(rows, dataset=dataset, split=split)
        for split in SPLITS
    }
    for split, selected_rows in split_rows.items():
        if not selected_rows:
            raise ValueError(f"{label} has no selected {split} poems")

    selected_rows = [
        row
        for split in SPLITS
        for row in split_rows[split]
    ]
    poem_ids = [row["poem_id"] for row in selected_rows]
    if len(poem_ids) != len(set(poem_ids)):
        raise ValueError(f"{label} contains duplicate selected poem IDs")

    split_texts = {
        split: load_poem_texts(split_rows[split], repo_root=repo_root)
        for split in SPLITS
    }
    tokenizer = BytePairEncodingTokenizer.load(tokenizer_path)
    parent_vocab_size = tokenizer.vocab_size
    tokenizer, added_characters = extend_tokenizer_for_character_coverage(
        tokenizer=tokenizer,
        texts=[
            text
            for split in SPLITS
            for text in split_texts[split]
        ],
    )
    encoded_splits = {
        split: encode_poem_texts_with_pretraining_tokenizer(
            split_texts[split],
            tokenizer,
        )
        for split in SPLITS
    }
    split_token_counts = {
        split: int(encoded_splits[split].numel())
        for split in SPLITS
    }
    split_poem_counts = {
        split: len(split_rows[split])
        for split in SPLITS
    }
    summary = {
        "label": label,
        "manifest_path": _portable_path(manifest_path, repo_root),
        "manifest_sha256": _file_sha256(manifest_path),
        "poem_count": len(selected_rows),
        "split_poem_counts": split_poem_counts,
        "split_token_counts": split_token_counts,
        "total_token_count": sum(split_token_counts.values()),
        "parent_vocab_size": parent_vocab_size,
        "finetuning_vocab_size": tokenizer.vocab_size,
        "added_character_count": len(added_characters),
        "validation_window_count": sequential_next_token_window_count(
            encoded_splits["validation"],
            context_length,
        ),
    }
    test_ids = {row["poem_id"] for row in split_rows["test"]}
    return summary, test_ids


def _build_test_overlap_summaries(
    test_ids_by_version: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    summaries = []
    for left_label, right_label in combinations(test_ids_by_version, 2):
        shared_ids = sorted(
            test_ids_by_version[left_label] & test_ids_by_version[right_label]
        )
        summaries.append({
            "versions": [left_label, right_label],
            "shared_test_poem_count": len(shared_ids),
        })

    if len(test_ids_by_version) > 2:
        all_shared_ids = set.intersection(*test_ids_by_version.values())
        summaries.append({
            "versions": list(test_ids_by_version),
            "shared_test_poem_count": len(all_shared_ids),
        })
    return summaries


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Sonnet Corpus Scaling Summary",
        "",
        "## Locked Comparison Protocol",
        "",
        f"- Dataset selector: `{report['dataset']}`",
        f"- Parent checkpoint: `{report['parent_checkpoint_path']}`",
        f"- Parent tokenizer: `{report['tokenizer_path']}`",
        f"- Parent tokenizer SHA-256: `{report['tokenizer_sha256']}`",
        f"- Context length: {report['context_length']}",
        "- Batch size: 2",
        "- Maximum training steps: 20,000",
        "- Evaluation interval: 250 steps",
        "- Validation mode: fixed sequential windows",
        "- Early-stopping patience: 8 evaluations",
        "- Minimum validation improvement: 0.01",
        "- Learning rate: 3e-5",
        "- Seed: 1337",
        "",
        "## Corpus Measurements",
        "",
        "| Version | Poems | Train / val / test poems | "
        "Train / val / test BPE tokens | Total BPE tokens | "
        "Vocabulary | Added characters | Validation windows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for version in report["versions"]:
        poem_counts = version["split_poem_counts"]
        token_counts = version["split_token_counts"]
        lines.append(
            f"| {version['label']} | {version['poem_count']:,} | "
            f"{poem_counts['train']:,} / {poem_counts['validation']:,} / "
            f"{poem_counts['test']:,} | "
            f"{token_counts['train']:,} / {token_counts['validation']:,} / "
            f"{token_counts['test']:,} | {version['total_token_count']:,} | "
            f"{version['parent_vocab_size']:,} -> "
            f"{version['finetuning_vocab_size']:,} | "
            f"{version['added_character_count']:,} | "
            f"{version['validation_window_count']:,} |"
        )

    lines.extend([
        "",
        "## Manifest Provenance",
        "",
    ])
    for version in report["versions"]:
        lines.append(
            f"- {version['label']}: `{version['manifest_path']}` "
            f"(SHA-256 `{version['manifest_sha256']}`)"
        )

    lines.extend([
        "",
        "## Shared Test Records",
        "",
        "| Versions | Shared test poems |",
        "| --- | ---: |",
    ])
    for overlap in report["test_set_overlaps"]:
        lines.append(
            f"| {' + '.join(overlap['versions'])} | "
            f"{overlap['shared_test_poem_count']:,} |"
        )

    lines.extend([
        "",
        "## Interpretation Rules",
        "",
        "- Select each run's checkpoint using only that run's own validation split.",
        "- Do not compare validation-loss values directly across corpus versions, "
        "because their validation records and token counts differ.",
        "- Compare generated outputs with identical prompts, seeds, and decoding "
        "settings.",
        "- Compare held-out behavior on the shared test subset when a record-level "
        "cross-version metric is needed.",
        "- Treat tokenizer vocabulary growth as part of the corpus change: parent "
        "token IDs stay fixed and only missing literal characters are appended.",
        "",
    ])
    return "\n".join(lines)


def _resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _portable_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_progress(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
