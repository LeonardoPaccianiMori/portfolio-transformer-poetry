"""Self-play rejection fine-tuning for autonomous rhyme generation.

The route keeps anchored candidates that already satisfy the target scheme
without repair, then trains a no-plan objective so the model must choose its
own endings. Generation and scoring helpers share the same prompt function so
training and evaluation stay aligned.
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
from sonnet_analysis.minerva_v7_prompt_intervention import build_intervention_prompt
from sonnet_analysis.plan_then_poem_validation import DEFAULT_SCHEME, PROMPT_ARM
from sonnet_evaluation.corpus_memorization import copied_continuation_lines

GENERATION_VERSION = "self_play_rft_v1"
ARMS = ("autonomous_baseline", "autonomous_rft")
MIN_ACCEPTED_LINES = 7
MAX_FAILED_LINES = 2
MIN_TYPE_TOKEN_RATIO = 0.60
MAX_REPEATED_BIGRAM_RATIO = 0.05
MAX_FINAL_WORD_REPETITION = 0.05
Progress = Callable[[str], None]


def autonomous_prompt(tokenizer: Any, opening_line: str) -> str:
    """Render the no-plan instruction with the exact opening prefill."""

    return build_intervention_prompt(tokenizer, opening_line, PROMPT_ARM)


def output_name(arm: str, prompt_id: str, seed: int) -> str:
    key = f"{GENERATION_VERSION}|{arm}|{prompt_id}|{seed}"
    return hashlib.sha256(key.encode()).hexdigest()[:20] + ".json"


def candidate_keep(
    card: Mapping[str, Any],
    *,
    min_accepted_lines: int = MIN_ACCEPTED_LINES,
    max_failed_lines: int = MAX_FAILED_LINES,
    min_type_token_ratio: float = MIN_TYPE_TOKEN_RATIO,
    max_repeated_bigram_ratio: float = MAX_REPEATED_BIGRAM_RATIO,
    max_final_word_repetition: float = MAX_FINAL_WORD_REPETITION,
) -> bool:
    """Return True when a candidate is scheme-valid and not degenerate."""

    return (
        bool(card.get("plan_scheme_ok"))
        and int(card.get("copied_lines", 0)) == 0
        and int(card.get("hendecasyllable_lines", 0)) >= min_accepted_lines
        and int(card.get("failed_lines", 0)) <= max_failed_lines
        and float(card.get("type_token_ratio", 0.0)) >= min_type_token_ratio
        and float(card.get("repeated_line_ratio", 1.0)) == 0.0
        and float(card.get("repeated_bigram_ratio", 1.0)) <= max_repeated_bigram_ratio
        and float(card.get("final_word_repetition", 1.0))
        <= max_final_word_repetition
    )


def build_no_plan_examples(
    cards: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_sequence_tokens: int = 1024,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build response-masked examples without the ending list in the prompt."""

    examples: list[dict[str, Any]] = []
    skipped = {"malformed": 0, "too_long": 0}
    for card in cards:
        raw_lines = [line for line in str(card["text"]).splitlines()]
        nonempty = [
            (index, line.strip())
            for index, line in enumerate(raw_lines)
            if line.strip()
        ]
        opening = str(card["opening_line"]).strip()
        if len(nonempty) != 14 or nonempty[0][1] != opening:
            skipped["malformed"] += 1
            continue
        first_index = nonempty[0][0]
        continuation = "\n".join(raw_lines[first_index + 1 :]).strip("\n") + "\n"
        prompt = autonomous_prompt(tokenizer, nonempty[0][1])
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
        target_ids = list(target_ids) + [int(tokenizer.eos_token_id)]
        if len(prompt_ids) + len(target_ids) > max_sequence_tokens:
            skipped["too_long"] += 1
            continue
        examples.append(
            {
                "unit_id": str(card["path"]),
                "prompt_ids": list(prompt_ids),
                "target_ids": target_ids,
                "opening_line": nonempty[0][1],
            }
        )
    if not examples:
        raise ValueError("no self-play examples could be built")
    return examples, skipped


def collect_candidates(
    records: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    proxies: Mapping[str, Mapping[str, Any]],
    line_index: set[str],
) -> list[dict[str, Any]]:
    """Merge records, checker scores, proxies, and copied-line counts."""

    score_by_path = {str(row["path"]): row for row in scores}
    cards: list[dict[str, Any]] = []
    for record in records:
        path = str(record["path"])
        row = score_by_path.get(path)
        if row is None:
            raise ValueError(f"missing score row for {path}")
        proxy = proxies[path]
        cards.append(
            {
                "path": path,
                "prompt_id": str(record["prompt_id"]),
                "seed": int(record["seed"]),
                "opening_line": str(record["prompt"]["opening_line"]),
                "text": str(record["text"]),
                "planned_words": [str(word) for word in record["planned_words"]],
                "plan_scheme_ok": bool(
                    row["rhyme_scheme"] == DEFAULT_SCHEME
                    and row["quatrain_ok"]
                    and row["tercet_ok"]
                ),
                "hendecasyllable_lines": int(row["hendecasyllable_lines"]),
                "failed_lines": int(row["failed_lines"]),
                "type_token_ratio": float(proxy["type_token_ratio"]),
                "repeated_line_ratio": float(proxy["repeated_line_ratio"]),
                "repeated_bigram_ratio": float(proxy["repeated_bigram_ratio"]),
                "final_word_repetition": float(proxy["final_word_repetition"]),
                "copied_lines": copied_continuation_lines(
                    str(record["text"]), line_index
                ),
            }
        )
    return cards


def load_anchored_candidates(anchored_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(anchored_dir.glob("*.json")):
        if path.name == "complete.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("condition") != "anchored":
            continue
        records.append(
            {
                "path": path.name,
                "prompt_id": str(payload["prompt"]["id"]),
                "seed": int(payload["seed"]),
                "prompt": dict(payload["prompt"]),
                "text": str(payload["text"]),
                "planned_words": [str(word) for word in payload["planned_words"]],
            }
        )
    if not records:
        raise ValueError("no anchored candidates were found")
    return records


def filter_cards(cards: Sequence[Mapping[str, Any]]) -> tuple[list[dict], dict[str, int]]:
    kept: list[dict] = []
    rejected = {
        "scheme": 0,
        "copied": 0,
        "accepted_lines": 0,
        "failed_lines": 0,
        "degenerate": 0,
    }
    for card in cards:
        if not bool(card.get("plan_scheme_ok")):
            rejected["scheme"] += 1
            continue
        if int(card.get("copied_lines", 0)) > 0:
            rejected["copied"] += 1
            continue
        if int(card.get("hendecasyllable_lines", 0)) < MIN_ACCEPTED_LINES:
            rejected["accepted_lines"] += 1
            continue
        if int(card.get("failed_lines", 0)) > MAX_FAILED_LINES:
            rejected["failed_lines"] += 1
            continue
        if not candidate_keep(card):
            rejected["degenerate"] += 1
            continue
        kept.append(dict(card))
    return kept, rejected


def generate_autonomous_validation(
    *,
    model: Any,
    tokenizer: Any,
    prompts: Sequence[Mapping[str, Any]],
    arm: str,
    seeds: Sequence[int],
    recipe: Mapping[str, Any],
    output_dir: Path,
    device: Any,
    batch_size: int,
    model_identity: str | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    jobs = []
    for prompt in prompts:
        rendered = autonomous_prompt(tokenizer, str(prompt["opening_line"]))
        for seed in seeds:
            path = output_dir / output_name(arm, str(prompt["id"]), int(seed))
            if path.is_file():
                continue
            jobs.append(
                {
                    "prompt": dict(prompt),
                    "seed": int(seed),
                    "rendered_prompt": rendered,
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
                "analysis_role": "self_play_rft_validation",
                "condition": arm,
                "prompt": job["prompt"],
                "planned_words": [],
                "recipe": dict(recipe),
                "model_identity_sha256": model_identity,
                **result,
                "v7_test_accessed": False,
            }
            path = output_dir / output_name(
                arm, str(job["prompt"]["id"]), int(job["seed"])
            )
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        if progress is not None:
            progress(
                f"arm={arm} completed={min(start + batch_size, len(jobs))}"
                f"/{len(jobs)} elapsed={time.monotonic() - started:.1f}s"
            )
    outputs = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name == "complete.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        outputs.append(
            {
                "path": path.name,
                "condition": str(payload["condition"]),
                "prompt_id": str(payload["prompt"]["id"]),
                "seed": int(payload["seed"]),
            }
        )
    result = {
        "generation_version": GENERATION_VERSION,
        "analysis_role": "self_play_rft_validation",
        "arms": sorted({row["condition"] for row in outputs}),
        "seeds": [int(seed) for seed in seeds],
        "completed_output_count": len(outputs),
        "outputs": outputs,
        "v7_test_accessed": False,
    }
    complete = output_dir / "complete.json"
    temporary = complete.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, complete)
    return result


def self_play_reading(
    *,
    baseline_valid: float,
    rft_valid: float,
    baseline_accepted: float,
    rft_accepted: float,
    judge_gain: float | None,
    validity_gate: float = 0.30,
    max_accepted_loss: float = 0.5,
    min_judge_gain: float = -0.3,
) -> dict[str, Any]:
    accepted_gap = rft_accepted - baseline_accepted
    checks = {
        "validity_gate": rft_valid >= validity_gate,
        "accepted_loss": accepted_gap >= -max_accepted_loss,
        "judge_loss": judge_gain is None or judge_gain >= min_judge_gain,
    }
    result = "SELF_PLAY_SIGNAL" if all(checks.values()) else "SELF_PLAY_NULL"
    reasons = [
        f"Autonomous scheme validity {rft_valid:.4f} against baseline "
        f"{baseline_valid:.4f} and gate {validity_gate:.2f}.",
        f"Accepted-line gap {accepted_gap:+.3f} (minimum {-max_accepted_loss}).",
        f"Judge gain {judge_gain} (minimum {min_judge_gain}).",
    ]
    return {"result": result, "reasons": reasons, "checks": checks}
