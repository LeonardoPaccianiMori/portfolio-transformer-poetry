import json
from pathlib import Path
from typing import Any

from sonnet_evaluation.metrics import resolve_generated_path


RATING_PLACEHOLDER = "TODO: low / medium / high"
NOTES_PLACEHOLDER = "TODO"


def read_generation_metadata(generation_dir: Path) -> dict[str, Any]:
    metadata_path = generation_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if not isinstance(metadata, dict):
        raise ValueError("generation metadata must contain a JSON object")

    if "generated_files" not in metadata:
        raise ValueError("generation metadata must contain generated_files")

    return metadata


def load_generated_reviews(generation_dir: Path) -> list[dict[str, Any]]:
    metadata_path = generation_dir / "metadata.json"
    metadata = read_generation_metadata(generation_dir)
    reviews = []

    for generated_file in metadata["generated_files"]:
        generated_path = resolve_generated_path(
            path_text=generated_file["path"],
            metadata_path=metadata_path,
        )
        generated_text = generated_path.read_text(encoding="utf-8")
        reviews.append({
            "prompt_id": generated_file["prompt_id"],
            "prompt_text": generated_file["prompt_text"],
            "path": str(generated_path),
            "seed": generated_file["seed"],
            "generated_text": generated_text,
        })

    return reviews


def markdown_heading_text(text: str) -> str:
    return text.replace("\n", " ").strip()


def fenced_text_block(text: str) -> str:
    safe_text = text.replace("```", "'''")
    return f"```text\n{safe_text.rstrip()}\n```"


def markdown_review_section(review: dict[str, Any]) -> str:
    return "\n\n".join([
        f"## Prompt: {markdown_heading_text(review['prompt_id'])}",
        f"- Prompt text: `{review['prompt_text']}`",
        f"- Seed: `{review['seed']}`",
        f"- Generated file: `{review['path']}`",
        "### Human Review",
        f"- Sonnet-like structure: {RATING_PLACEHOLDER}",
        f"- Language/style plausibility: {RATING_PLACEHOLDER}",
        f"- Coherence: {RATING_PLACEHOLDER}",
        f"- Repetition problems: {RATING_PLACEHOLDER}",
        f"- Memorization concern: {RATING_PLACEHOLDER}",
        f"- Strongest failure mode: {NOTES_PLACEHOLDER}",
        f"- Notes: {NOTES_PLACEHOLDER}",
        "### Generated Text",
        fenced_text_block(review["generated_text"]),
    ])


def build_qualitative_review_report(
    generation_dir: Path,
    reviews: list[dict[str, Any]],
) -> str:
    sections = [
        "# Qualitative Generation Review",
        f"Generation directory: `{generation_dir}`",
        "## Review Instructions",
        "- Fill in each `TODO` field after reading the generated text.",
        "- Use `low`, `medium`, or `high` consistently within this report.",
        "- Judge the generated text as model output, not as a polished poem.",
        "- Keep weak and failed samples in the report.",
    ]

    sections.extend(markdown_review_section(review) for review in reviews)
    sections.append("")

    return "\n\n".join(sections)


def write_qualitative_review_report(
    generation_dir: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    reviews = load_generated_reviews(generation_dir)
    report = build_qualitative_review_report(
        generation_dir=generation_dir,
        reviews=reviews,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    return reviews
