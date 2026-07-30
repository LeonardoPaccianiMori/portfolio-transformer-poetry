"""Generate user-visible sonnet continuations from task-format checkpoints."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from pathlib import Path
from typing import Any

import torch

from sonnet_corpus.bpe import BytePairEncodingTokenizer
from sonnet_corpus.dataset_text import (
    load_poem_text,
    read_manifest_rows,
    select_manifest_rows,
)
from sonnet_corpus.task_format import (
    SONNET_CONTINUATION_TOKEN,
    SONNET_LINE_COUNT,
    SONNET_OPENING_TOKEN,
    TASK_FORMAT_SPECIAL_TOKENS,
    build_task_prompt,
    split_sonnet_for_continuation,
)
from sonnet_evaluation.generation import (
    completed_non_empty_line_count,
    load_tokenizer,
    load_transformer_from_checkpoint,
    safe_prompt_filename,
)
from sonnet_model.transformer import CausalTransformerLanguageModel


TASK_FORMAT_VERSION = "opening_line_continuation_v1"
TASK_CONTINUATION_LINE_TARGET = SONNET_LINE_COUNT - 1
END_OF_TEXT_TOKEN = "<|endoftext|>"
ACCEPTANCE_PROMPT_COUNT = 10
ACCEPTANCE_SEEDS = (1337, 1338)
ACCEPTANCE_TEMPERATURE = 0.8
ACCEPTANCE_TOP_K = 50


def load_task_format_prompts(path: Path) -> list[dict[str, str]]:
    """Read fixed opening-line prompts for task-format generation."""
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, list):
        raise ValueError("task-format prompts file must contain a JSON list")

    required_fields = {"id", "poem_id", "opening_line"}
    prompts: list[dict[str, str]] = []

    for prompt in payload:
        if not isinstance(prompt, dict):
            raise ValueError("each task-format prompt must be a JSON object")

        missing_fields = sorted(required_fields - prompt.keys())
        if missing_fields:
            raise ValueError(
                "each task-format prompt is missing fields: "
                + ", ".join(missing_fields)
            )

        normalized_prompt = {
            key: value
            for key, value in prompt.items()
            if isinstance(key, str) and isinstance(value, str)
        }
        if set(normalized_prompt) != set(prompt):
            raise ValueError("task-format prompt fields must be strings")

        build_task_prompt(normalized_prompt["opening_line"])
        prompts.append(normalized_prompt)

    prompt_ids = [prompt["id"] for prompt in prompts]
    poem_ids = [prompt["poem_id"] for prompt in prompts]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("task-format prompt ids must be unique")
    if len(set(poem_ids)) != len(poem_ids):
        raise ValueError("task-format prompt poem_ids must be unique")

    return prompts


def validate_task_format_prompts_against_manifest(
    prompts: Sequence[dict[str, str]],
    manifest_path: Path,
    repo_root: Path,
    dataset: str,
    split: str,
) -> None:
    """Require every fixed prompt to be the exact first line of one held-out poem."""
    rows = read_manifest_rows(manifest_path)
    split_rows = select_manifest_rows(rows=rows, dataset=dataset, split=split)
    rows_by_poem_id = {
        row["poem_id"]: row
        for row in split_rows
    }

    for prompt in prompts:
        poem_id = prompt["poem_id"]
        row = rows_by_poem_id.get(poem_id)
        if row is None:
            raise ValueError(
                "task-format prompt poem_id is not in the requested split: "
                f"{poem_id}"
            )

        actual_opening_line, _ = split_sonnet_for_continuation(
            load_poem_text(row=row, repo_root=repo_root)
        )
        if prompt["opening_line"] != actual_opening_line:
            raise ValueError(
                "task-format prompt opening_line does not match its processed poem: "
                f"{poem_id}"
            )


def validate_task_format_acceptance_configuration(
    prompts: Sequence[dict[str, str]],
    seeds: Sequence[int],
    temperature: float,
    top_k: int | None,
    continuation_line_target: int,
) -> None:
    """Reject altered settings for the predeclared 20-output acceptance set."""
    if len(prompts) != ACCEPTANCE_PROMPT_COUNT:
        raise ValueError(
            "task-format acceptance requires exactly "
            f"{ACCEPTANCE_PROMPT_COUNT} prompts"
        )
    if tuple(seeds) != ACCEPTANCE_SEEDS:
        raise ValueError(
            "task-format acceptance requires seeds "
            f"{list(ACCEPTANCE_SEEDS)}"
        )
    if temperature != ACCEPTANCE_TEMPERATURE:
        raise ValueError(
            "task-format acceptance requires temperature "
            f"{ACCEPTANCE_TEMPERATURE}"
        )
    if top_k != ACCEPTANCE_TOP_K:
        raise ValueError(
            "task-format acceptance requires top_k "
            f"{ACCEPTANCE_TOP_K}"
        )
    if continuation_line_target != TASK_CONTINUATION_LINE_TARGET:
        raise ValueError(
            "task-format acceptance requires continuation_line_target "
            f"{TASK_CONTINUATION_LINE_TARGET}"
        )


def generate_task_format_continuation(
    model: CausalTransformerLanguageModel,
    tokenizer: BytePairEncodingTokenizer,
    opening_line: str,
    max_new_tokens: int,
    device: torch.device | str,
    seed: int,
    temperature: float = 0.8,
    top_k: int | None = 50,
    continuation_line_target: int = TASK_CONTINUATION_LINE_TARGET,
) -> dict[str, Any]:
    """Generate a visible sonnet from one opening line and hidden control tokens."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be greater than or equal to 0")
    if continuation_line_target <= 0:
        raise ValueError("continuation_line_target must be greater than 0")

    control_token_ids = _task_control_token_ids(tokenizer)
    prompt = build_task_prompt(opening_line)
    generator = torch.Generator(device=device).manual_seed(seed)
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device,
    )
    generated_ids = input_ids
    continuation_text = ""
    stop_reason: str | None = None

    for _ in range(max_new_tokens):
        if (
            completed_non_empty_line_count(continuation_text)
            >= continuation_line_target
        ):
            stop_reason = "target_lines"
            break

        # Control tokens belong to the prompt protocol, never to visible poetry.
        generated_ids = model.generate(
            input_ids=generated_ids,
            max_new_tokens=1,
            generator=generator,
            temperature=temperature,
            top_k=top_k,
            forbidden_token_ids=control_token_ids,
        )
        decoded_text = tokenizer.decode(generated_ids[0].cpu().tolist())
        if not decoded_text.startswith(prompt):
            raise RuntimeError("generated task-format text lost its prompt prefix")

        continuation_text = decoded_text[len(prompt):]

    if stop_reason is None:
        if (
            completed_non_empty_line_count(continuation_text)
            >= continuation_line_target
        ):
            stop_reason = "target_lines"
        else:
            stop_reason = "max_new_tokens"

    return {
        "text": f"{opening_line}\n{continuation_text}",
        "opening_line": opening_line,
        "task_prompt": prompt,
        "stop_reason": stop_reason,
        "generated_new_tokens": generated_ids.shape[1] - input_ids.shape[1],
        "completed_continuation_lines": completed_non_empty_line_count(
            continuation_text
        ),
    }


def generate_task_format_for_prompts(
    run_dir: Path,
    prompts: Sequence[dict[str, str]],
    output_dir: Path,
    max_new_tokens: int,
    seeds: Sequence[int],
    device: torch.device | str,
    temperature: float = 0.8,
    top_k: int | None = 50,
    continuation_line_target: int = TASK_CONTINUATION_LINE_TARGET,
    checkpoint_path: Path | None = None,
    model_config_path: Path | None = None,
    prompt_config_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate every prompt/seed pair and persist reproducibility metadata."""
    if not prompts:
        raise ValueError("prompts must contain at least one prompt")
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")

    _report_progress(progress, f"loading tokenizer from {run_dir / 'tokenizer.json'}")
    tokenizer = load_tokenizer(run_dir / "tokenizer.json")
    if not isinstance(tokenizer, BytePairEncodingTokenizer):
        raise ValueError("task-format generation requires a Unicode BPE tokenizer")
    _task_control_token_ids(tokenizer)

    checkpoint_path = checkpoint_path or run_dir / "model.pt"
    model_config_path = model_config_path or run_dir / "config.json"
    _report_progress(progress, f"loading checkpoint from {checkpoint_path}")
    model = load_transformer_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config_path=model_config_path,
        device=device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = []
    total_outputs = len(prompts) * len(seeds)

    for prompt_index, prompt in enumerate(prompts):
        for seed_index, seed in enumerate(seeds):
            output_index = prompt_index * len(seeds) + seed_index + 1
            generation_id = f"{prompt['id']}__seed_{seed}"
            _report_progress(
                progress,
                f"generating output {output_index}/{total_outputs}: {generation_id}",
            )
            result = generate_task_format_continuation(
                model=model,
                tokenizer=tokenizer,
                opening_line=prompt["opening_line"],
                max_new_tokens=max_new_tokens,
                device=device,
                seed=seed,
                temperature=temperature,
                top_k=top_k,
                continuation_line_target=continuation_line_target,
            )
            output_path = output_dir / safe_prompt_filename(generation_id)
            output_path.write_text(result["text"], encoding="utf-8")
            generated_files.append({
                "prompt_id": generation_id,
                "source_prompt_id": prompt["id"],
                "poem_id": prompt["poem_id"],
                "author": prompt.get("author", ""),
                "prompt_text": prompt["opening_line"],
                "opening_line": prompt["opening_line"],
                "path": str(output_path),
                "seed": seed,
                "stop_reason": result["stop_reason"],
                "generated_new_tokens": result["generated_new_tokens"],
                "completed_continuation_lines": result[
                    "completed_continuation_lines"
                ],
            })
            _report_progress(progress, f"wrote output {output_index}/{total_outputs}: {output_path}")

    metadata = {
        "generation_format": "task_format_opening_line_continuation",
        "task_format_version": TASK_FORMAT_VERSION,
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "model_config_path": str(model_config_path),
        "prompt_config_path": str(prompt_config_path) if prompt_config_path else None,
        "output_dir": str(output_dir),
        "max_new_tokens": max_new_tokens,
        "seeds": list(seeds),
        "device": str(device),
        "temperature": temperature,
        "top_k": top_k,
        "stop_text": END_OF_TEXT_TOKEN,
        "continuation_line_target": continuation_line_target,
        "total_line_target": continuation_line_target + 1,
        "suppressed_control_tokens": [
            END_OF_TEXT_TOKEN,
            *TASK_FORMAT_SPECIAL_TOKENS,
        ],
        "generated_files": generated_files,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return metadata


def _task_control_token_ids(tokenizer: BytePairEncodingTokenizer) -> set[int]:
    required_tokens = (
        END_OF_TEXT_TOKEN,
        SONNET_OPENING_TOKEN,
        SONNET_CONTINUATION_TOKEN,
    )
    missing_tokens = [
        token
        for token in required_tokens
        if token not in tokenizer.special_tokens
    ]
    if missing_tokens:
        raise ValueError(
            "task-format tokenizer is missing special tokens: "
            + ", ".join(missing_tokens)
        )

    token_ids = set()
    for token in required_tokens:
        encoded = tokenizer.encode(token)
        if len(encoded) != 1:
            raise ValueError(f"task control token must encode to one ID: {token}")
        token_ids.add(encoded[0])

    return token_ids


def _report_progress(
    progress: Callable[[str], None] | None,
    message: str,
) -> None:
    if progress is not None:
        progress(message)
