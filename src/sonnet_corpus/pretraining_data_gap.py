"""Quantify corpus scale and training exposure for pretraining decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PretrainingDataGapConfig:
    """Inputs and target budget for one reproducible corpus-scale report."""

    run_config_path: Path
    encoding_report_path: Path
    tokenizer_report_path: Path
    balance_report_path: Path
    markdown_report_path: Path
    target_unique_tokens: int = 75_000_000
    max_train_steps: int = 650_000
    heuristic_tokens_per_parameter: int = 20


def build_pretraining_data_gap_report(
    config: PretrainingDataGapConfig,
) -> dict[str, Any]:
    """Read local artifacts and write a public Markdown data-gap report."""

    _validate_positive(config.target_unique_tokens, "target_unique_tokens")
    _validate_positive(config.max_train_steps, "max_train_steps")
    _validate_positive(
        config.heuristic_tokens_per_parameter,
        "heuristic_tokens_per_parameter",
    )

    run_config = _read_json(config.run_config_path)
    encoding = _read_json(config.encoding_report_path)
    tokenizer = _read_json(config.tokenizer_report_path)
    balance = _read_json(config.balance_report_path)

    current_total_tokens = _required_int(encoding, "total_tokens")
    current_train_tokens = _required_int(encoding, "train_tokens")
    current_validation_tokens = _required_int(encoding, "validation_tokens")
    batch_size = _required_int(run_config, "batch_size")
    context_length = _required_int(run_config, "context_length")
    completed_steps = _required_int(run_config, "completed_steps")
    parameter_count = _required_int(run_config, "parameter_count")
    characters_per_token = _required_number(tokenizer, "characters_per_token")
    total_characters = _required_int(balance, "total_cleaned_characters")

    tokens_per_step = batch_size * context_length
    completed_exposures = completed_steps * tokens_per_step
    max_exposures = config.max_train_steps * tokens_per_step
    report = {
        "current_total_tokens": current_total_tokens,
        "current_train_tokens": current_train_tokens,
        "current_validation_tokens": current_validation_tokens,
        "target_unique_tokens": config.target_unique_tokens,
        "additional_unique_tokens_needed": max(
            config.target_unique_tokens - current_total_tokens,
            0,
        ),
        "target_completion_fraction": current_total_tokens / config.target_unique_tokens,
        "batch_size": batch_size,
        "context_length": context_length,
        "tokens_per_step": tokens_per_step,
        "completed_steps": completed_steps,
        "completed_exposures": completed_exposures,
        "completed_passes_over_train_stream": completed_exposures / current_train_tokens,
        "max_train_steps": config.max_train_steps,
        "max_exposures": max_exposures,
        "max_passes_over_target_corpus": max_exposures / config.target_unique_tokens,
        "parameter_count": parameter_count,
        "heuristic_tokens_per_parameter": config.heuristic_tokens_per_parameter,
        "heuristic_token_budget": parameter_count
        * config.heuristic_tokens_per_parameter,
        "target_to_heuristic_budget_fraction": config.target_unique_tokens
        / (parameter_count * config.heuristic_tokens_per_parameter),
        "cleaned_character_count": total_characters,
        "characters_per_token": characters_per_token,
        "source_count": _required_int(encoding, "source_count"),
        "largest_source": _largest_entry(balance, "source_entries"),
        "largest_author": _largest_entry(balance, "author_entries"),
    }
    config.markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.markdown_report_path.write_text(
        render_pretraining_data_gap_markdown(report),
        encoding="utf-8",
    )
    return report


def render_pretraining_data_gap_markdown(report: dict[str, Any]) -> str:
    """Render the stable public interpretation of a scale calculation."""

    largest_source = report["largest_source"]
    largest_author = report["largest_author"]
    lines = [
        "# Pretraining Data-Gap Report",
        "",
        "This report fixes the scale target for the next broader-Italian corpus "
        "revision. It measures encoded BPE tokens, not whitespace words or raw "
        "characters, because BPE tokens are the units consumed by the model.",
        "",
        "## Corpus Scale",
        "",
        "| Measurement | Value |",
        "| --- | ---: |",
        f"| Active sources | {report['source_count']:,} |",
        f"| Cleaned characters | {report['cleaned_character_count']:,} |",
        f"| Current encoded corpus tokens | {report['current_total_tokens']:,} |",
        f"| Current training tokens | {report['current_train_tokens']:,} |",
        f"| Current validation tokens | {report['current_validation_tokens']:,} |",
        f"| Target unique corpus tokens | {report['target_unique_tokens']:,} |",
        f"| Additional unique tokens needed | {report['additional_unique_tokens_needed']:,} |",
        f"| Target currently assembled | {report['target_completion_fraction']:.1%} |",
        f"| Current tokenizer fertility | {report['characters_per_token']:.3f} characters/token |",
        "",
        "The 75M figure is a corpus-assembly target before applying the deterministic "
        "training/validation split. It is not a claim that 75M tokens alone guarantee "
        "coherent generation.",
        "",
        "## Training Exposure Budget",
        "",
        "| Measurement | Value |",
        "| --- | ---: |",
        f"| Model parameters | {report['parameter_count']:,} |",
        f"| Batch size | {report['batch_size']:,} |",
        f"| Context length | {report['context_length']:,} |",
        f"| Tokens processed per step | {report['tokens_per_step']:,} |",
        f"| Completed pretraining steps | {report['completed_steps']:,} |",
        f"| Completed token exposures | {report['completed_exposures']:,} |",
        f"| Completed passes over current train stream | {report['completed_passes_over_train_stream']:.2f} |",
        f"| Proposed maximum pretraining steps | {report['max_train_steps']:,} |",
        f"| Proposed maximum token exposures | {report['max_exposures']:,} |",
        f"| Proposed passes over 75M-token corpus | {report['max_passes_over_target_corpus']:.2f} |",
        "",
        "For orientation only, a commonly cited compute-optimal heuristic uses roughly "
        f"{report['heuristic_tokens_per_parameter']}:1 training tokens per parameter. "
        f"For this model that is {report['heuristic_token_budget']:,} exposures; the "
        f"75M-token corpus target is {report['target_to_heuristic_budget_fraction']:.1%} "
        "of that rough exposure budget. This heuristic is not a training prescription, "
        "especially for a small historical corpus with repeated passes.",
        "",
        "## Existing Composition",
        "",
        f"- Largest work: `{largest_source['name']}` at "
        f"{largest_source['character_share']:.2%} of cleaned characters.",
        f"- Largest author: `{largest_author['name']}` at "
        f"{largest_author['character_share']:.2%} of cleaned characters.",
        "- The project decision is to retain the complete Ramusio compilation; this "
        "report records its share but does not propose capping it.",
        "",
        "## Next Gate",
        "",
        "Before page-level extraction, each candidate source must pass the documented "
        "metadata, license, composition, and representative-text gate. Only selected "
        "core-compatible prose sources proceed to the expensive revision-pinned audit "
        "and builder pipeline.",
        "",
    ]
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required report artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in: {path}")
    return payload


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"required positive integer is missing or invalid: {key}")
    return value


def _required_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"required positive number is missing or invalid: {key}")
    return float(value)


def _largest_entry(payload: dict[str, Any], key: str) -> dict[str, Any]:
    entries = payload.get(key)
    if not isinstance(entries, list) or not entries or not isinstance(entries[0], dict):
        raise ValueError(f"required non-empty entry list is missing or invalid: {key}")
    entry = entries[0]
    if not isinstance(entry.get("name"), str) or not isinstance(
        entry.get("character_share"),
        (int, float),
    ):
        raise ValueError(f"first entry is missing required fields: {key}")
    return entry


def _validate_positive(value: int, label: str) -> None:
    if value <= 0:
        raise ValueError(f"{label} must be greater than zero")
