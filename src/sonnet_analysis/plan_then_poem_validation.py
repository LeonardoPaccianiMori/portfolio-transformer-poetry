"""Inference-only plan-then-poem validation for the rhyme-planning pilot.

The planned condition appends a numbered list of pre-committed line endings to
the frozen prompt. The format control appends the same list shape with
non-rhyming placeholder endings. Both are scored against each other and
against the verifier-DPO no-plan baseline.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sonnet_analysis.minerva_v7_high_volume_generation import generate_batch
from sonnet_analysis.minerva_v7_prompt_intervention import build_intervention_prompt
from sonnet_evaluation.rhyme_lexicon import choose_endings, line_final_word
from sonnet_evaluation.sonnet_prosody import rhyme_key

GENERATION_VERSION = "plan_then_poem_validation_v1"
CONDITIONS = ("planned", "control")
DEFAULT_SCHEME = "ABBAABBACDECDE"
PROMPT_ARM = "explicit_no_labels_or_prose"


def output_name(
    condition: str,
    prompt_id: str,
    seed: int,
    *,
    version: str = GENERATION_VERSION,
) -> str:
    key = f"{version}|{condition}|{prompt_id}|{seed}"
    return hashlib.sha256(key.encode()).hexdigest()[:20] + ".json"


def build_plan_words(
    lexicon: Mapping[str, Any],
    *,
    scheme: str = DEFAULT_SCHEME,
    seed: int = 6200,
) -> dict[str, Any]:
    lines = choose_endings(scheme, lexicon, seed=seed)
    return {
        "scheme": scheme,
        "seed": seed,
        "lines": lines,
        "words": [row["word"] for row in lines],
    }


def build_control_words(
    lexicon: Mapping[str, Any], *, seed: int = 6200, count: int = 14
) -> list[str]:
    entries = [
        entry
        for entry in lexicon["entries"]
        if entry["count"] >= 5 and len(entry["words"]) >= 2
    ]
    rng = random.Random(seed)
    rng.shuffle(entries)
    words: list[str] = []
    used_keys: set[str] = set()
    for entry in entries:
        if entry["key"] in used_keys:
            continue
        used_keys.add(entry["key"])
        words.append(entry["words"][1])
        if len(words) == count:
            break
    if len(words) != count:
        raise ValueError("rhyme lexicon has too few distinct keys for the control")
    return words


def build_plans(
    prompts: Sequence[Mapping[str, Any]],
    lexicon: Mapping[str, Any],
    *,
    scheme: str = DEFAULT_SCHEME,
    seed: int = 6200,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    plans = {
        str(prompt["id"]): build_plan_words(lexicon, scheme=scheme, seed=seed + index)
        for index, prompt in enumerate(prompts)
    }
    return plans, build_control_words(lexicon, seed=seed)


def plan_instruction(words: Sequence[str]) -> str:
    numbered = "\n".join(f"{index + 1}. {word}" for index, word in enumerate(words))
    return (
        "Termina ogni verso con la parola indicata, nello stesso ordine:\n" + numbered
    )


def planned_prompt(tokenizer: Any, opening_line: str, words: Sequence[str]) -> str:
    return build_intervention_prompt(
        tokenizer,
        opening_line,
        PROMPT_ARM,
        extra_instruction=plan_instruction(words),
    )


def adherence(
    text: str, planned_words: Sequence[str]
) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    word_matches = 0
    key_matches = 0
    missing = 0
    for index, planned in enumerate(planned_words):
        if index >= len(lines):
            missing += 1
            continue
        final = line_final_word(lines[index])
        if final is None:
            missing += 1
            continue
        if final == planned.lower():
            word_matches += 1
        if rhyme_key(final) == rhyme_key(planned):
            key_matches += 1
    return {
        "planned_count": len(planned_words),
        "word_matches": word_matches,
        "key_matches": key_matches,
        "missing": missing,
        "word_match_rate": word_matches / len(planned_words) if planned_words else None,
        "key_match_rate": key_matches / len(planned_words) if planned_words else None,
    }


def generate_plan_validation(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Mapping[str, Any]],
    plans: Mapping[str, Mapping[str, Any]],
    control_words: Sequence[str],
    seeds: Sequence[int],
    recipe: Mapping[str, Any],
    output_dir: Path,
    device: Any,
    batch_size: int,
    adapter_identity: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for condition in CONDITIONS:
        jobs = []
        for prompt in prompts:
            words = (
                plans[str(prompt["id"])]["words"]
                if condition == "planned"
                else list(control_words)
            )
            rendered = planned_prompt(tokenizer, str(prompt["opening_line"]), words)
            for seed in seeds:
                path = output_dir / output_name(condition, str(prompt["id"]), int(seed))
                if path.is_file():
                    continue
                jobs.append(
                    {
                        "condition": condition,
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
                    "analysis_role": "plan_then_poem_inference_validation",
                    "condition": job["condition"],
                    "prompt": job["prompt"],
                    "planned_words": job["planned_words"],
                    "recipe": dict(recipe),
                    "adapter_identity_sha256": adapter_identity,
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
                completed = len(list(output_dir.glob("*.json")))
                elapsed = time.monotonic() - started
                progress(
                    f"condition={condition} completed={completed} "
                    f"elapsed={elapsed:.1f}s"
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
        "analysis_role": "plan_then_poem_inference_validation",
        "conditions": list(CONDITIONS),
        "prompt_count": len(prompts),
        "seeds": [int(seed) for seed in seeds],
        "planned_output_count": len(prompts) * len(seeds) * len(CONDITIONS),
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


def load_plan_records(generation_dir: Path) -> list[dict[str, Any]]:
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
                "opening_line": payload.get("opening_line"),
                "text": payload["text"],
                "planned_words": payload["planned_words"],
            }
        )
    if len(records) != int(complete["completed_output_count"]):
        raise ValueError("plan generation count does not match complete.json")
    return records
