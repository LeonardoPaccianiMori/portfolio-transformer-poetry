#!/usr/bin/env python3
"""Compare character and BPE tokenization on the sonnet corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_corpus.bpe import train_bpe_tokenizer
from sonnet_corpus.dataset_text import (
    load_poem_texts,
    read_manifest_rows,
    select_manifest_rows,
)
from sonnet_corpus.tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "metadata" / "poems_manifest.csv",
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--dataset", default="expanded_with_petrarch")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument(
        "--tokenizer-output",
        type=Path,
        default=ROOT / "data" / "metadata" / "bpe_tokenizer.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "reports" / "tokenizer_comparison.md",
    )
    return parser.parse_args()


def load_split_poems(
    manifest_path: Path,
    repo_root: Path,
    dataset: str,
    split: str,
) -> list[str]:
    rows = read_manifest_rows(manifest_path)
    selected_rows = select_manifest_rows(
        rows=rows,
        dataset=dataset,
        split=split,
    )

    return load_poem_texts(
        rows=selected_rows,
        repo_root=repo_root,
    )


def token_count(texts: list[str], tokenizer: CharTokenizer) -> int:
    return sum(
        len(tokenizer.encode(text))
        for text in texts
    )


def bpe_token_count(texts: list[str], tokenizer) -> int:
    return sum(
        len(tokenizer.encode(text))
        for text in texts
    )


def average_token_count(texts: list[str], tokenizer) -> float:
    return mean(
        len(tokenizer.encode(text))
        for text in texts
    )


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def build_tokenizer_comparison_report(
    dataset: str,
    vocab_size: int,
    tokenizer_path: str,
    rows: list[dict[str, object]],
    char_vocab_size: int,
    bpe_vocab_size: int,
    merge_count: int,
) -> str:
    table_lines = [
        "| Split | Poems | Char Tokens | BPE Tokens | Compression Ratio | Avg Char Tokens/Poem | Avg BPE Tokens/Poem |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        table_lines.append(
            "| {split} | {poems} | {char_tokens} | {bpe_tokens} | {ratio:.4f} | {avg_char:.1f} | {avg_bpe:.1f} |".format(
                split=row["split"],
                poems=row["poems"],
                char_tokens=row["char_tokens"],
                bpe_tokens=row["bpe_tokens"],
                ratio=row["compression_ratio"],
                avg_char=row["avg_char_tokens_per_poem"],
                avg_bpe=row["avg_bpe_tokens_per_poem"],
            )
        )

    return "\n\n".join([
        "# Tokenizer Comparison",
        f"Dataset: `{dataset}`",
        f"BPE tokenizer path: `{tokenizer_path}`",
        "## Configuration",
        f"- Character tokenizer vocabulary size: `{char_vocab_size}`",
        f"- BPE target vocabulary size: `{vocab_size}`",
        f"- BPE actual vocabulary size: `{bpe_vocab_size}`",
        f"- BPE learned merge count: `{merge_count}`",
        "- BPE special tokens: `<|poem_end|>`",
        "- BPE merge rules were learned from the training split only.",
        "- Base character coverage used train, validation, and test splits to avoid unknown held-out characters.",
        "## Split Token Counts",
        "\n".join(table_lines),
        "## Interpretation",
        "BPE reduces the number of tokens the transformer must process. A lower compression ratio means shorter sequences relative to character tokenization.",
        "Shorter sequences should make full-sonnet context easier for the model, but this report does not prove better generation quality by itself. The next step is to encode BPE train/validation/test tensors and train a comparable BPE transformer.",
        "",
    ])


def main() -> None:
    args = parse_args()
    splits = ["train", "validation", "test"]
    split_texts = {
        split: load_split_poems(
            manifest_path=args.manifest,
            repo_root=args.repo_root,
            dataset=args.dataset,
            split=split,
        )
        for split in splits
    }
    all_texts = [
        text
        for split in splits
        for text in split_texts[split]
    ]
    train_texts = split_texts["train"]

    char_tokenizer = CharTokenizer.from_texts(all_texts)
    bpe_tokenizer = train_bpe_tokenizer(
        texts=train_texts,
        base_texts=all_texts,
        vocab_size=args.vocab_size,
        special_tokens=["<|poem_end|>"],
    )
    bpe_tokenizer.save(args.tokenizer_output)

    rows = []

    for split in splits:
        texts = split_texts[split]
        char_tokens = token_count(texts, char_tokenizer)
        bpe_tokens = bpe_token_count(texts, bpe_tokenizer)
        rows.append({
            "split": split,
            "poems": len(texts),
            "char_tokens": char_tokens,
            "bpe_tokens": bpe_tokens,
            "compression_ratio": bpe_tokens / char_tokens,
            "avg_char_tokens_per_poem": average_token_count(texts, char_tokenizer),
            "avg_bpe_tokens_per_poem": average_token_count(texts, bpe_tokenizer),
        })

    report = build_tokenizer_comparison_report(
        dataset=args.dataset,
        vocab_size=args.vocab_size,
        tokenizer_path=display_path(args.tokenizer_output, args.repo_root),
        rows=rows,
        char_vocab_size=char_tokenizer.vocab_size,
        bpe_vocab_size=bpe_tokenizer.vocab_size,
        merge_count=len(bpe_tokenizer.merges),
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(report, encoding="utf-8")

    print(f"wrote tokenizer: {args.tokenizer_output}")
    print(f"wrote report: {args.report_output}")


if __name__ == "__main__":
    main()
