#!/usr/bin/env python3
"""Analyze and freeze a blinded sample from matched DPO validation outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_dpo_validation import VALIDATION_VERSION
from sonnet_analysis.minerva_v7_memorization import (
    load_verified_sonnet_train_reference, score_texts_against_reference,
)
from sonnet_analysis.minerva_v7_quality import generated_sonnet_surface_diagnostics
from sonnet_evaluation.metrics import score_generated_text


METRICS = (
    "fourteen_line", "prompt_preserved", "repetition_ratio",
    "unique_character_ratio", "character_count", "meta_text_free",
    "ends_with_terminal_punctuation", "no_line_at_or_above_120_characters",
    "surface_screen_pass",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--memorization-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--blind-prompts", type=int, default=40)
    args = parser.parse_args()
    completion = json.loads(
        (args.generation_dir / "complete.json").read_text(encoding="utf-8")
    )
    if (
        completion.get("validation_version") != VALIDATION_VERSION
        or completion.get("completed_output_count") != 960
        or completion.get("v7_test_accessed") is not False
    ):
        raise ValueError("DPO validation completion contract mismatch")
    rows = []
    for declared in completion["outputs"]:
        path = args.generation_dir / declared["path"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != declared["sha256"]:
            raise ValueError("DPO validation output hash mismatch")
        payload = json.loads(raw)
        metrics = score_generated_text(payload["text"], payload["opening_line"])
        surface = generated_sonnet_surface_diagnostics(
            payload["text"], non_empty_line_count=metrics["non_empty_line_count"],
            repetition_ratio=metrics["repetition_ratio"],
        )
        rows.append({
            "system_id": payload["system_id"],
            "prompt_id": payload["prompt"]["id"], "seed": payload["seed"],
            "opening_line": payload["opening_line"], "text": payload["text"],
            "fourteen_line": metrics["non_empty_line_count"] == 14,
            "prompt_preserved": metrics["prompt_preserved"],
            "repetition_ratio": metrics["repetition_ratio"],
            "unique_character_ratio": metrics["unique_character_ratio"],
            "character_count": metrics["character_count"],
            "meta_text_free": surface["meta_text_free"],
            "ends_with_terminal_punctuation": surface["ends_with_terminal_punctuation"],
            "no_line_at_or_above_120_characters": surface["no_line_at_or_above_120_characters"],
            "surface_screen_pass": surface["surface_screen_pass"],
        })
    references, reference_manifest = load_verified_sonnet_train_reference(
        args.memorization_reference
    )
    scores = score_texts_against_reference(
        [row["text"] for row in rows], references,
        progress=lambda message: print(f"minerva-v7-dpo | memorization {message}", flush=True),
    )
    for row, score in zip(rows, scores, strict=True):
        row["memorization"] = score
    summaries = [_summary(rows, system, args.bootstrap_resamples, 12201 + index)
                 for index, system in enumerate(("stage_3", "dpo"))]
    comparison = _comparison(rows, args.bootstrap_resamples, 12211)
    analysis = {
        "validation_version": VALIDATION_VERSION,
        "rows": rows, "summaries": summaries, "paired_comparison": comparison,
        "memorization_reference_manifest_sha256": reference_manifest["manifest_sha256"],
        "v7_test_accessed": False,
    }
    blind = _blind_sample(rows, count=args.blind_prompts, seed=12217)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "blind_mapping.private.json").write_text(
        json.dumps(blind, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "blinded_review.md").write_text(
        _review_markdown(blind), encoding="utf-8"
    )
    print(
        f"minerva-v7-dpo | complete job=validation_analysis rows={len(rows)} "
        f"blind_rows={len(blind['mapping'])} v7_test_accessed=False", flush=True,
    )


def _summary(rows, system, resamples, seed):
    selected = [row for row in rows if row["system_id"] == system]
    by_prompt = defaultdict(list)
    for row in selected: by_prompt[row["prompt_id"]].append(row)
    result = {"system_id": system, "outputs": len(selected), "prompt_clusters": len(by_prompt)}
    for index, metric in enumerate(METRICS):
        means = {key: statistics.fmean(float(x[metric]) for x in values) for key, values in by_prompt.items()}
        result[metric] = _bootstrap(means, resamples, seed + index, paired=False)
    result["high_memorization_risk_count"] = sum(
        row["memorization"]["risk_level"] == "high" for row in selected
    )
    return result


def _comparison(rows, resamples, seed):
    lookup = {(row["system_id"], row["prompt_id"], row["seed"]): row for row in rows}
    prompts = sorted({row["prompt_id"] for row in rows})
    by_prompt = defaultdict(list)
    for prompt in prompts:
        for current_seed in (5200, 5201, 5202, 5203):
            left, right = lookup[("stage_3", prompt, current_seed)], lookup[("dpo", prompt, current_seed)]
            by_prompt[prompt].append({metric: float(right[metric]) - float(left[metric]) for metric in METRICS})
    result = {"left_system_id": "stage_3", "right_system_id": "dpo", "paired_outputs": 480}
    for index, metric in enumerate(METRICS):
        means = {key: statistics.fmean(x[metric] for x in values) for key, values in by_prompt.items()}
        result[metric] = _bootstrap(means, resamples, seed + index, paired=True)
    return result


def _bootstrap(means, resamples, seed, paired):
    rng, keys = random.Random(seed), sorted(means)
    distribution = sorted(statistics.fmean(means[rng.choice(keys)] for _ in keys) for _ in range(resamples))
    label = "mean_paired_change" if paired else "mean"
    return {label: statistics.fmean(means.values()), "ci_low": distribution[int(.025*(resamples-1))], "ci_high": distribution[int(.975*(resamples-1))]}


def _blind_sample(
    rows, count, seed, version=VALIDATION_VERSION,
    seeds=(5200, 5201, 5202, 5203),
):
    prompts = sorted({row["prompt_id"] for row in rows})
    rng = random.Random(seed); selected = sorted(rng.sample(prompts, count))
    chosen_seed = {prompt: rng.choice(tuple(seeds)) for prompt in selected}
    mapping = []
    for row in rows:
        if row["prompt_id"] in chosen_seed and row["seed"] == chosen_seed[row["prompt_id"]]:
            blind_id = hashlib.sha256(
                f"{version}|{row['system_id']}|{row['prompt_id']}|{row['seed']}".encode()
            ).hexdigest()[:16]
            mapping.append({"blind_id": blind_id, **row})
    if len(mapping) != count * 2: raise ValueError("DPO blind sample is incomplete")
    return {
        "selection_seed": seed,
        "selected_prompt_count": count,
        "eligible_seeds": list(seeds),
        "mapping": sorted(mapping, key=lambda x: x["blind_id"]),
        "v7_test_accessed": False,
    }


def _review_markdown(blind, title="DPO Validation Blinded Review"):
    lines = [f"# {title}", "", "Score 1–5 without consulting the private mapping.", "", "| Blind ID | Grammar | Historical Register | Poetic Quality | Sonnet/Form | Volta/Argument | Complete | Meta-text | Collapse | Evidence |", "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |"]
    for row in blind["mapping"]: lines.append(f"| `{row['blind_id']}` | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |")
    lines += ["", "## Outputs", ""]
    for row in blind["mapping"]: lines += [f"### `{row['blind_id']}`", "", "```text", row["text"], "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
