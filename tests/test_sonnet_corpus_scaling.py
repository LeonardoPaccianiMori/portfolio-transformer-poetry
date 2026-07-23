import csv
from pathlib import Path

import pytest

from sonnet_corpus.bpe import train_bpe_tokenizer
from sonnet_corpus.dataset_text import PRETRAINING_DOCUMENT_SEPARATOR
from sonnet_corpus.sonnet_corpus_scaling import (
    write_sonnet_corpus_scaling_report,
)


def write_manifest(
    path: Path,
    records: list[tuple[str, str, str]],
) -> None:
    rows = [
        {
            "poem_id": poem_id,
            "clean_text_path": f"data/processed/poems/{poem_id}.txt",
            "include_in_expanded_with_petrarch": "True",
            "split_expanded_with_petrarch": split,
        }
        for poem_id, split, _ in records
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_fixture(
    repo_root: Path,
) -> tuple[dict[str, Path], Path]:
    records_by_version = {
        "v1": [
            ("train_a", "train", "Amor antico"),
            ("validation_a", "validation", "Donna gentile"),
            ("shared_test", "test", "Core nel canto"),
        ],
        "v2": [
            ("train_a", "train", "Amor antico"),
            ("train_b", "train", "Virtute nuova Ω"),
            ("validation_b", "validation", "Chiara memoria"),
            ("shared_test", "test", "Core nel canto"),
            ("new_test", "test", "Novo pensiero"),
        ],
    }
    poems_dir = repo_root / "data" / "processed" / "poems"
    poems_dir.mkdir(parents=True)
    for records in records_by_version.values():
        for poem_id, _, text in records:
            (poems_dir / f"{poem_id}.txt").write_text(text, encoding="utf-8")

    manifest_paths = {}
    for label, records in records_by_version.items():
        manifest_path = repo_root / "data" / "metadata" / f"{label}.csv"
        write_manifest(manifest_path, records)
        manifest_paths[label] = manifest_path

    tokenizer = train_bpe_tokenizer(
        texts=["Amor antico Donna gentile Core nel canto"],
        base_texts=["Amor antico Donna gentile Core nel canto"],
        vocab_size=45,
        special_tokens=[PRETRAINING_DOCUMENT_SEPARATOR],
    )
    tokenizer_path = repo_root / "runs" / "parent" / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    return manifest_paths, tokenizer_path


def test_scaling_report_counts_tokens_extensions_and_shared_test_records(tmp_path):
    manifest_paths, tokenizer_path = write_fixture(tmp_path)
    output_path = tmp_path / "reports" / "scaling.md"

    report = write_sonnet_corpus_scaling_report(
        repo_root=tmp_path,
        manifest_paths=manifest_paths,
        tokenizer_path=tokenizer_path,
        output_path=output_path,
        context_length=2,
    )

    versions = {version["label"]: version for version in report["versions"]}
    assert versions["v1"]["poem_count"] == 3
    assert versions["v2"]["poem_count"] == 5
    assert versions["v2"]["split_poem_counts"] == {
        "train": 2,
        "validation": 1,
        "test": 2,
    }
    assert versions["v2"]["total_token_count"] > versions["v1"]["total_token_count"]
    assert versions["v2"]["added_character_count"] > 0
    assert versions["v2"]["validation_window_count"] > 0
    assert report["test_set_overlaps"] == [{
        "versions": ["v1", "v2"],
        "shared_test_poem_count": 1,
    }]

    markdown = output_path.read_text(encoding="utf-8")
    assert "# Sonnet Corpus Scaling Summary" in markdown
    assert "| v1 | 3 |" in markdown
    assert "| v2 | 5 |" in markdown
    assert "Do not compare validation-loss values directly" in markdown


def test_scaling_report_rejects_a_missing_manifest(tmp_path):
    _, tokenizer_path = write_fixture(tmp_path)

    with pytest.raises(FileNotFoundError, match="sonnet manifest does not exist"):
        write_sonnet_corpus_scaling_report(
            repo_root=tmp_path,
            manifest_paths={"missing": Path("data/metadata/missing.csv")},
            tokenizer_path=tokenizer_path,
            output_path=tmp_path / "report.md",
        )
