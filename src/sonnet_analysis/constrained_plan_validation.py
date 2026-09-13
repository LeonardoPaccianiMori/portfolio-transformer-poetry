"""Opening-anchored plans and ending repair for the constrained plan test.

Line 1 is the forced opening, so a feasible plan anchors rhyme class A to the
opening's own final word. The repair condition replaces each line-final word
with the planned word when it differs, which is the cheap equivalent of
constrained ending decoding for this pilot.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sonnet_analysis.minerva_v7_high_volume_generation import generate_batch
from sonnet_analysis.plan_then_poem_validation import (
    DEFAULT_SCHEME,
    PROMPT_ARM,
    echo_detected,
    planned_prompt,
)
from sonnet_evaluation.rhyme_lexicon import line_final_word
from sonnet_evaluation.sonnet_prosody import rhyme_key

GENERATION_VERSION = "constrained_plan_validation_v1"
CONDITIONS = ("anchored", "anchored_repaired")
WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÿ']+")
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


def plan_from_anchored(
    opening_line: str,
    lexicon: Mapping[str, Any],
    *,
    scheme: str = DEFAULT_SCHEME,
    seed: int = 6200,
) -> dict[str, Any]:
    opening_word = line_final_word(opening_line)
    if not opening_word:
        raise ValueError("opening line has no final word")
    key = rhyme_key(opening_word)
    if not key:
        raise ValueError("opening line has no rhyme key")
    entries = {entry["key"]: entry for entry in lexicon["entries"]}
    if key not in entries or len(entries[key]["words"]) < 2:
        raise ValueError("opening rhyme key is absent from the lexicon")
    rng = random.Random(seed)
    class_words: dict[str, list[str]] = {
        "A": [word for word in entries[key]["words"] if word != opening_word]
    }
    used_words = {opening_word}
    used_keys = {key}
    candidates = [
        entry
        for entry in lexicon["entries"]
        if entry["key"] != key
        and entry["count"] >= 5
        and len(entry["words"]) >= 5
    ]
    rng.shuffle(candidates)
    classes: dict[str, dict[str, Any]] = {"A": {"key": key}}
    for letter in sorted(set(scheme)):
        if letter == "A":
            continue
        chosen = None
        for entry in candidates:
            if entry["key"] in used_keys:
                continue
            pool = [word for word in entry["words"] if word not in used_words]
            if len(pool) < 2:
                continue
            chosen = entry
            break
        if chosen is None:
            raise ValueError(f"no lexicon candidates for class {letter}")
        used_keys.add(chosen["key"])
        class_words[letter] = [
            word for word in chosen["words"] if word not in used_words
        ]
        classes[letter] = {"key": chosen["key"]}
    lines = []
    counters: dict[str, int] = {}
    for index, letter in enumerate(scheme):
        if letter == "A" and index == 0:
            word = opening_word
        else:
            pool = class_words[letter]
            position = counters.get(letter, 0)
            if position >= len(pool):
                raise ValueError(f"not enough words for class {letter}")
            word = pool[position]
            counters[letter] = position + 1
        used_words.add(word)
        lines.append(
            {
                "line": index + 1,
                "class": letter,
                "key": classes[letter]["key"],
                "word": word,
            }
        )
    words = [row["word"] for row in lines]
    if len(set(words)) != len(words):
        raise ValueError("plan words are not distinct")
    return {"scheme": scheme, "seed": seed, "lines": lines, "words": words}


def repair_endings(
    text: str, planned_words: Sequence[str]
) -> tuple[str, dict[str, Any]]:
    raw_lines = text.splitlines()
    nonempty = [index for index, line in enumerate(raw_lines) if line.strip()]
    repaired_lines = 0
    flags: list[bool] = []
    for position, line_index in enumerate(nonempty):
        if position >= len(planned_words):
            break
        line = raw_lines[line_index]
        matches = list(WORD_PATTERN.finditer(line))
        if not matches:
            raw_lines[line_index] = f"{line.rstrip()} {planned_words[position]}"
            repaired_lines += 1
            flags.append(True)
            continue
        match = matches[-1]
        if match.group(0).lower() == planned_words[position].lower():
            flags.append(False)
            continue
        raw_lines[line_index] = (
            line[: match.start()] + planned_words[position] + line[match.end() :]
        )
        repaired_lines += 1
        flags.append(True)
    return "\n".join(raw_lines), {
        "repaired_lines": repaired_lines,
        "repair_flags": flags,
        "repair_distance": repaired_lines,
    }


def generate_anchored_validation(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Mapping[str, Any]],
    anchored_plans: Mapping[str, Mapping[str, Any]],
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
    jobs = []
    for prompt in prompts:
        words = anchored_plans[str(prompt["id"])]["words"]
        rendered = planned_prompt(tokenizer, str(prompt["opening_line"]), words)
        for seed in seeds:
            path = output_dir / output_name("anchored", str(prompt["id"]), int(seed))
            if path.is_file():
                continue
            jobs.append(
                {
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
                "analysis_role": "constrained_plan_validation",
                "condition": "anchored",
                "prompt": job["prompt"],
                "planned_words": job["planned_words"],
                "recipe": dict(recipe),
                "model_identity_sha256": model_identity,
                **result,
                "v7_test_accessed": False,
            }
            path = output_dir / output_name(
                "anchored", str(job["prompt"]["id"]), job["seed"]
            )
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        if progress is not None:
            completed = len([p for p in output_dir.glob("*.json") if p.name != "complete.json"])
            progress(
                f"completed={completed}/{len(jobs)} elapsed={time.monotonic() - started:.1f}s"
            )
    for prompt in prompts:
        for seed in seeds:
            anchored_path = output_dir / output_name(
                "anchored", str(prompt["id"]), int(seed)
            )
            anchored = json.loads(anchored_path.read_text(encoding="utf-8"))
            repaired_text, repair = repair_endings(
                str(anchored["text"]), anchored["planned_words"]
            )
            payload = {
                **anchored,
                "condition": "anchored_repaired",
                "text": repaired_text,
                "repair": repair,
            }
            repaired_path = output_dir / output_name(
                "anchored_repaired", str(prompt["id"]), int(seed)
            )
            temporary = repaired_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, repaired_path)
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
        "analysis_role": "constrained_plan_validation",
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


def load_anchored_records(generation_dir: Path) -> list[dict[str, Any]]:
    complete = json.loads((generation_dir / "complete.json").read_text(encoding="utf-8"))
    records = []
    for row in complete["outputs"]:
        path = generation_dir / row["path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("condition") != row["condition"]:
            raise ValueError(f"condition mismatch in {row['path']}")
        record = {
            "path": row["path"],
            "prompt_id": row["prompt_id"],
            "seed": int(row["seed"]),
            "system_id": str(row["condition"]),
            "opening_line": payload.get("opening_line"),
            "text": payload["text"],
            "planned_words": payload["planned_words"],
            "echo": echo_detected(str(payload["text"])),
        }
        if "repair" in payload:
            record["repair_distance"] = payload["repair"]["repair_distance"]
        records.append(record)
    if len(records) != int(complete["completed_output_count"]):
        raise ValueError("anchored generation count does not match complete.json")
    return records
