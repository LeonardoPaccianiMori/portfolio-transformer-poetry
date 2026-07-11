import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from sonnet_corpus.dataset_text import (
    load_poem_text,
    read_manifest_rows,
    select_manifest_rows,
)
from sonnet_evaluation.metrics import resolve_generated_path


DEFAULT_NGRAM_SIZE = 40
MEDIUM_CONTAINMENT_THRESHOLD = 0.15
HIGH_CONTAINMENT_THRESHOLD = 0.30
MEDIUM_SUBSTRING_THRESHOLD = 80
HIGH_SUBSTRING_THRESHOLD = 160


def normalize_for_memorization(text: str) -> str:
    return " ".join(text.lower().split())


def character_ngram_set(text: str, ngram_size: int) -> set[str]:
    if ngram_size <= 0:
        raise ValueError("ngram_size must be greater than 0")

    if len(text) < ngram_size:
        return set()

    return {
        text[index:index + ngram_size]
        for index in range(len(text) - ngram_size + 1)
    }


def ngram_containment(
    generated_text: str,
    reference_text: str,
    ngram_size: int = DEFAULT_NGRAM_SIZE,
) -> float:
    generated_ngrams = character_ngram_set(
        normalize_for_memorization(generated_text),
        ngram_size=ngram_size,
    )
    reference_ngrams = character_ngram_set(
        normalize_for_memorization(reference_text),
        ngram_size=ngram_size,
    )

    if not generated_ngrams:
        return 0.0

    return len(generated_ngrams & reference_ngrams) / len(generated_ngrams)


def longest_common_substring_length(left: str, right: str) -> int:
    normalized_left = normalize_for_memorization(left)
    normalized_right = normalize_for_memorization(right)

    if normalized_left == "" or normalized_right == "":
        return 0

    match = SequenceMatcher(
        None,
        normalized_left,
        normalized_right,
        autojunk=False,
    ).find_longest_match(
        0,
        len(normalized_left),
        0,
        len(normalized_right),
    )

    return match.size


def memorization_risk_level(
    containment: float,
    longest_common_substring_chars: int,
) -> str:
    if (
        containment >= HIGH_CONTAINMENT_THRESHOLD
        or longest_common_substring_chars >= HIGH_SUBSTRING_THRESHOLD
    ):
        return "high"

    if (
        containment >= MEDIUM_CONTAINMENT_THRESHOLD
        or longest_common_substring_chars >= MEDIUM_SUBSTRING_THRESHOLD
    ):
        return "medium"

    return "low"


def load_training_records(
    manifest_path: Path,
    repo_root: Path,
    dataset: str,
    split: str = "train",
) -> list[dict[str, str]]:
    rows = read_manifest_rows(manifest_path)
    selected_rows = select_manifest_rows(
        rows=rows,
        dataset=dataset,
        split=split,
    )

    return [
        {
            "poem_id": row["poem_id"],
            "title_or_first_line": row["title_or_first_line"],
            "author": row["author"],
            "clean_text_path": row["clean_text_path"],
            "text": load_poem_text(row, repo_root=repo_root),
        }
        for row in selected_rows
    ]


def find_nearest_training_record(
    generated_text: str,
    training_records: list[dict[str, str]],
    ngram_size: int = DEFAULT_NGRAM_SIZE,
) -> dict[str, Any]:
    if not training_records:
        raise ValueError("training_records must contain at least one record")

    best_row: dict[str, Any] | None = None

    for record in training_records:
        containment = ngram_containment(
            generated_text=generated_text,
            reference_text=record["text"],
            ngram_size=ngram_size,
        )
        longest_substring = longest_common_substring_length(
            generated_text,
            record["text"],
        )
        row = {
            "nearest_poem_id": record["poem_id"],
            "nearest_title_or_first_line": record["title_or_first_line"],
            "nearest_author": record["author"],
            "nearest_clean_text_path": record["clean_text_path"],
            "ngram_containment": containment,
            "longest_common_substring_chars": longest_substring,
            "risk_level": memorization_risk_level(
                containment=containment,
                longest_common_substring_chars=longest_substring,
            ),
        }

        if best_row is None:
            best_row = row
            continue

        current_score = (
            row["ngram_containment"],
            row["longest_common_substring_chars"],
        )
        best_score = (
            best_row["ngram_containment"],
            best_row["longest_common_substring_chars"],
        )

        if current_score > best_score:
            best_row = row

    if best_row is None:
        raise RuntimeError("nearest-record search failed unexpectedly")

    return best_row


def read_generation_metadata(generation_dir: Path) -> dict[str, Any]:
    metadata_path = generation_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if not isinstance(metadata, dict):
        raise ValueError("generation metadata must contain a JSON object")

    if "generated_files" not in metadata:
        raise ValueError("generation metadata must contain generated_files")

    return metadata


def score_generation_memorization(
    generation_dir: Path,
    training_records: list[dict[str, str]],
    ngram_size: int = DEFAULT_NGRAM_SIZE,
) -> list[dict[str, Any]]:
    metadata_path = generation_dir / "metadata.json"
    metadata = read_generation_metadata(generation_dir)
    rows = []

    for generated_file in metadata["generated_files"]:
        generated_path = resolve_generated_path(
            path_text=generated_file["path"],
            metadata_path=metadata_path,
        )
        generated_text = generated_path.read_text(encoding="utf-8")
        nearest = find_nearest_training_record(
            generated_text=generated_text,
            training_records=training_records,
            ngram_size=ngram_size,
        )
        rows.append({
            "prompt_id": generated_file["prompt_id"],
            "prompt_text": generated_file["prompt_text"],
            "path": str(generated_path),
            "seed": generated_file["seed"],
            "generated_character_count": len(generated_text),
            **nearest,
        })

    return rows


def markdown_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", r"\|")


def markdown_memorization_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Prompt",
        "Chars",
        "Nearest Training Poem",
        "Author",
        "Containment",
        "LCS Chars",
        "Risk",
        "Seed",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        values = [
            row["prompt_id"],
            row["generated_character_count"],
            row["nearest_title_or_first_line"],
            row["nearest_author"],
            f"{row['ngram_containment']:.4f}",
            row["longest_common_substring_chars"],
            row["risk_level"],
            row["seed"],
        ]
        lines.append(
            "| " + " | ".join(markdown_cell(value) for value in values) + " |"
        )

    return "\n".join(lines)


def build_memorization_report(
    generation_dir: Path,
    dataset: str,
    split: str,
    ngram_size: int,
    rows: list[dict[str, Any]],
) -> str:
    return "\n\n".join([
        "# Memorization Checks",
        f"Generation directory: `{generation_dir}`",
        f"Comparison dataset: `{dataset}`",
        f"Comparison split: `{split}`",
        f"Character n-gram size: `{ngram_size}`",
        markdown_memorization_table(rows),
        "## Notes",
        "- Text is lowercased and whitespace-normalized before comparison.",
        "- Punctuation is preserved because copied punctuation is useful evidence.",
        "- `Containment` is the fraction of generated character n-grams also found in the nearest training poem.",
        "- `LCS Chars` is the longest contiguous copied character span after normalization.",
        "- Risk labels are heuristic surface-copying checks, not proof of memorization.",
        f"- `medium`: containment >= {MEDIUM_CONTAINMENT_THRESHOLD:.2f} or LCS >= {MEDIUM_SUBSTRING_THRESHOLD} chars.",
        f"- `high`: containment >= {HIGH_CONTAINMENT_THRESHOLD:.2f} or LCS >= {HIGH_SUBSTRING_THRESHOLD} chars.",
    ]) + "\n"


def write_memorization_report(
    generation_dir: Path,
    manifest_path: Path,
    repo_root: Path,
    dataset: str,
    split: str,
    output_path: Path,
    ngram_size: int = DEFAULT_NGRAM_SIZE,
) -> list[dict[str, Any]]:
    training_records = load_training_records(
        manifest_path=manifest_path,
        repo_root=repo_root,
        dataset=dataset,
        split=split,
    )
    rows = score_generation_memorization(
        generation_dir=generation_dir,
        training_records=training_records,
        ngram_size=ngram_size,
    )
    report = build_memorization_report(
        generation_dir=generation_dir,
        dataset=dataset,
        split=split,
        ngram_size=ngram_size,
        rows=rows,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    return rows
