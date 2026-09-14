"""Few-shot context A/B for the constrained plan pipeline.

Conditions differ only in the number of complete plan-plus-sonnet examples
placed in the chat context: zero, one, or three. All conditions use the same
anchored plans, openings, seeds, prompt text, and recipe.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sonnet_analysis.minerva_v7_high_volume_generation import generate_batch
from sonnet_analysis.minerva_v7_prompt_intervention import intervention_user_content
from sonnet_analysis.plan_then_poem_validation import PROMPT_ARM, plan_instruction
from sonnet_evaluation.corpus_memorization import (  # noqa: F401
    memorization_screen,
    screen_normalize,
)
from sonnet_evaluation.rhyme_lexicon import line_final_word
from sonnet_training.form_targeted_data import read_sonnet_text

GENERATION_VERSION = "few_shot_context_validation_v1"
CONDITIONS = {"zeroshot": 0, "fewshot1": 1, "fewshot3": 3}
Progress = Callable[[str], None]


def output_name(
    condition: str,
    prompt_id: str,
    seed: int,
    *,
    version: str = GENERATION_VERSION,
) -> str:
    key = f"{version}|{condition}|{prompt_id}|{seed}"
    return hashlib.sha256(key.encode()).hexdigest()[:20] + ".json"


def build_context_examples(
    rows: Sequence[Mapping[str, str]],
    root: Path,
    *,
    limit: int = 3,
    excluded_logical_sha256: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded = excluded_logical_sha256 or set()
    examples: list[dict[str, Any]] = []
    for row in rows:
        if row["logical_sha256"] in excluded:
            continue
        text = read_sonnet_text(root, row)
        raw_lines = text.splitlines()
        lines = [line.strip() for line in raw_lines if line.strip()]
        if len(lines) != 14:
            continue
        words = [line_final_word(line) for line in lines]
        if any(word is None for word in words):
            continue
        examples.append(
            {
                "unit_id": row["unit_id"],
                "opening_line": lines[0],
                "words": [str(word) for word in words],
                "text": "\n".join(raw_lines).strip(),
            }
        )
        if len(examples) >= limit:
            break
    if len(examples) < limit:
        raise ValueError("not enough context examples could be built")
    return examples


def context_prompt(
    tokenizer: Any,
    opening_line: str,
    words: Sequence[str],
    examples: Sequence[Mapping[str, Any]],
) -> str:
    messages = []
    for example in examples:
        messages.append(
            {
                "role": "user",
                "content": intervention_user_content(
                    str(example["opening_line"]),
                    PROMPT_ARM,
                    extra_instruction=plan_instruction(example["words"]),
                ),
            }
        )
        messages.append({"role": "assistant", "content": str(example["text"])})
    messages.append(
        {
            "role": "user",
            "content": intervention_user_content(
                opening_line,
                PROMPT_ARM,
                extra_instruction=plan_instruction(words),
            ),
        }
    )
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return f"{rendered}{opening_line}\n"


def generate_context_validation(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Mapping[str, Any]],
    plans: Mapping[str, Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    recipe: Mapping[str, Any],
    output_dir: Path,
    device: Any,
    batch_size: int,
    model_identity: str | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for condition, shot_count in CONDITIONS.items():
        jobs = []
        for prompt in prompts:
            words = plans[str(prompt["id"])]["words"]
            rendered = context_prompt(
                tokenizer,
                str(prompt["opening_line"]),
                words,
                examples[:shot_count],
            )
            for seed in seeds:
                path = output_dir / output_name(condition, str(prompt["id"]), int(seed))
                if path.is_file():
                    continue
                jobs.append(
                    {
                        "condition": condition,
                        "shots": shot_count,
                        "prompt": dict(prompt),
                        "seed": int(seed),
                        "rendered_prompt": rendered,
                        "planned_words": list(words),
                    }
                )
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start : start + batch_size]
            results = generate_batch(
                model=model,
                tokenizer=tokenizer,
                jobs=batch,
                recipe=recipe,
                device=device,
            )
            for job, result in zip(batch, results, strict=True):
                payload = {
                    "generation_version": GENERATION_VERSION,
                    "analysis_role": "few_shot_context_validation",
                    "condition": job["condition"],
                    "shots": job["shots"],
                    "prompt": job["prompt"],
                    "planned_words": job["planned_words"],
                    "recipe": dict(recipe),
                    "model_identity_sha256": model_identity,
                    **result,
                    "v7_test_accessed": False,
                }
                path = output_dir / output_name(
                    job["condition"], str(job["prompt"]["id"]), job["seed"]
                )
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, path)
            if progress is not None:
                completed = len(
                    [p for p in output_dir.glob("*.json") if p.name != "complete.json"]
                )
                progress(
                    f"condition={condition} completed={completed} "
                    f"elapsed={time.monotonic() - started:.1f}s"
                )
    outputs = []
    for condition in CONDITIONS:
        for prompt in prompts:
            for seed in seeds:
                path = output_dir / output_name(condition, str(prompt["id"]), int(seed))
                if not path.is_file():
                    raise FileNotFoundError(path)
                outputs.append(
                    {
                        "path": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "condition": condition,
                        "prompt_id": str(prompt["id"]),
                        "seed": int(seed),
                    }
                )
    result = {
        "generation_version": GENERATION_VERSION,
        "analysis_role": "few_shot_context_validation",
        "conditions": list(CONDITIONS),
        "prompt_count": len(prompts),
        "seeds": [int(seed) for seed in seeds],
        "completed_output_count": len(outputs),
        "outputs": sorted(outputs, key=lambda row: row["path"]),
        "v7_test_accessed": False,
    }
    complete = output_dir / "complete.json"
    temporary = complete.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, complete)
    return result


def load_context_records(generation_dir: Path) -> list[dict[str, Any]]:
    complete = json.loads((generation_dir / "complete.json").read_text(encoding="utf-8"))
    records = []
    for row in complete["outputs"]:
        path = generation_dir / row["path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("condition") != row["condition"]:
            raise ValueError(f"condition mismatch in {row['path']}")
        records.append(
            {
                "path": row["path"],
                "prompt_id": row["prompt_id"],
                "seed": int(row["seed"]),
                "system_id": str(row["condition"]),
                "shots": int(payload.get("shots", 0)),
                "opening_line": payload.get("opening_line"),
                "text": payload["text"],
                "planned_words": payload["planned_words"],
            }
        )
    if len(records) != int(complete["completed_output_count"]):
        raise ValueError("context generation count does not match complete.json")
    return records
