import json
from pathlib import Path
from typing import Any


POEM_SEPARATOR = "<|poem_end|>"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def non_empty_line_count(text: str) -> int:
    return len([
        line
        for line in text.splitlines()
        if line.strip()
    ])


def unique_character_ratio(text: str) -> float:
    if text == "":
        return 0.0

    return len(set(text)) / len(text)


def repeated_ngram_ratio(text: str, ngram_size: int = 4) -> float:
    if ngram_size <= 0:
        raise ValueError("ngram_size must be greater than 0")

    if len(text) < ngram_size:
        return 0.0

    ngrams = [
        text[index:index + ngram_size]
        for index in range(len(text) - ngram_size + 1)
    ]

    return 1.0 - (len(set(ngrams)) / len(ngrams))


def resolve_generated_path(path_text: str, metadata_path: Path) -> Path:
    path = Path(path_text)

    if path.is_absolute() or path.is_file():
        return path

    candidate = metadata_path.parent / path.name

    if candidate.is_file():
        return candidate

    return path


def score_generated_text(
    text: str,
    prompt_text: str,
    ngram_size: int = 4,
) -> dict[str, Any]:
    return {
        "character_count": len(text),
        "non_empty_line_count": non_empty_line_count(text),
        "poem_separator_count": text.count(POEM_SEPARATOR),
        "unique_character_ratio": unique_character_ratio(text),
        "repetition_ratio": repeated_ngram_ratio(
            text=text,
            ngram_size=ngram_size,
        ),
        "prompt_preserved": text.startswith(prompt_text),
    }


def score_generation_directory(
    generation_dir: Path,
    ngram_size: int = 4,
) -> list[dict[str, Any]]:
    metadata_path = generation_dir / "metadata.json"
    metadata = read_json(metadata_path)
    rows = []

    for generated_file in metadata["generated_files"]:
        generated_path = resolve_generated_path(
            path_text=generated_file["path"],
            metadata_path=metadata_path,
        )
        text = generated_path.read_text(encoding="utf-8")
        metrics = score_generated_text(
            text=text,
            prompt_text=generated_file["prompt_text"],
            ngram_size=ngram_size,
        )
        rows.append({
            "prompt_id": generated_file["prompt_id"],
            "prompt_text": generated_file["prompt_text"],
            "path": str(generated_path),
            "seed": generated_file["seed"],
            **metrics,
        })

    return rows


def markdown_metrics_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Prompt",
        "Chars",
        "Lines",
        "Separators",
        "Unique Chars",
        "Repeat Ratio",
        "Prompt Kept",
        "Seed",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for row in rows:
        values = [
            row["prompt_id"],
            str(row["character_count"]),
            str(row["non_empty_line_count"]),
            str(row["poem_separator_count"]),
            f"{row['unique_character_ratio']:.4f}",
            f"{row['repetition_ratio']:.4f}",
            "yes" if row["prompt_preserved"] else "no",
            str(row["seed"]),
        ]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def build_generation_metrics_report(
    generation_dir: Path,
    rows: list[dict[str, Any]],
) -> str:
    return "\n\n".join([
        "# Generation Metrics",
        f"Generation directory: `{generation_dir}`",
        markdown_metrics_table(rows),
        "## Notes",
        "- `Lines` counts non-empty lines.",
        f"- `Separators` counts `{POEM_SEPARATOR}` occurrences.",
        "- `Repeat Ratio` is based on repeated character 4-grams by default.",
        "- These are basic automatic checks, not a full quality evaluation.",
        "",
    ])


def write_generation_metrics_report(
    generation_dir: Path,
    output_path: Path,
    ngram_size: int = 4,
) -> list[dict[str, Any]]:
    rows = score_generation_directory(
        generation_dir=generation_dir,
        ngram_size=ngram_size,
    )
    report = build_generation_metrics_report(
        generation_dir=generation_dir,
        rows=rows,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    return rows
