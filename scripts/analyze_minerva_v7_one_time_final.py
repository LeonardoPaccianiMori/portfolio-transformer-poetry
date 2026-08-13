#!/usr/bin/env python3
"""Analyze and freeze the blinded sample from the one-time V7 final outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_minerva_v7_dpo_validation import (
    METRICS, _blind_sample, _comparison, _review_markdown, _summary,
)
from sonnet_analysis.minerva_v7_final_evaluation import (
    EXPECTED_DOCUMENTS, EXPECTED_SEEDS, FINAL_GENERATION_VERSION,
    load_frozen_final_protocol,
)
from sonnet_analysis.minerva_v7_memorization import (
    load_verified_sonnet_train_reference, score_texts_against_reference,
)
from sonnet_analysis.minerva_v7_quality import generated_sonnet_surface_diagnostics
from sonnet_evaluation.metrics import score_generated_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--memorization-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args()
    protocol = load_frozen_final_protocol(args.protocol)
    protocol_sha = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    completion_path = args.generation_dir / "complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    expected_outputs = EXPECTED_DOCUMENTS * len(EXPECTED_SEEDS) * 2
    expected = {
        "validation_version": FINAL_GENERATION_VERSION,
        "analysis_role": "one_time_final_test_no_retuning",
        "completed_output_count": expected_outputs,
        "planned_output_count": expected_outputs,
        "v7_test_accessed": True,
        "retuning_after_test_forbidden": True,
        "final_protocol_sha256": protocol_sha,
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            raise ValueError(f"one-time final completion contract mismatch: {key}")
    rows = []
    for declared in completion["outputs"]:
        path = args.generation_dir / declared["path"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != declared["sha256"]:
            raise ValueError("one-time final output hash mismatch")
        payload = json.loads(raw)
        if (
            payload.get("validation_version") != FINAL_GENERATION_VERSION
            or payload.get("analysis_role") != "one_time_final_test_no_retuning"
            or payload.get("v7_test_accessed") is not True
        ):
            raise ValueError("one-time final output lineage mismatch")
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
        progress=lambda message: print(f"minerva-v7-final | memorization {message}", flush=True),
    )
    for row, score in zip(rows, scores, strict=True):
        row["memorization"] = score
    summaries = [
        _summary(rows, system, args.bootstrap_resamples, 16201 + index)
        for index, system in enumerate(("stage_3", "dpo"))
    ]
    comparison = _comparison_final(rows, args.bootstrap_resamples, 16211)
    analysis = {
        "analysis_version": "minerva_7b_v7_one_time_final_analysis_v1",
        "final_protocol_sha256": protocol_sha,
        "completion_manifest_sha256": hashlib.sha256(completion_path.read_bytes()).hexdigest(),
        "rows": rows, "summaries": summaries, "paired_comparison": comparison,
        "memorization_reference_manifest_sha256": reference_manifest["manifest_sha256"],
        "v7_test_accessed": True, "retuning_after_test_forbidden": True,
    }
    blind = _blind_sample(
        rows, count=int(protocol["blinded_review"]["prompt_count"]),
        seed=int(protocol["blinded_review"]["selection_seed"]),
        version=FINAL_GENERATION_VERSION,
        seeds=EXPECTED_SEEDS,
    )
    blind["v7_test_accessed"] = True
    blind["retuning_after_test_forbidden"] = True
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
        _review_markdown(blind, title="One-Time V7 Final Blinded Review"),
        encoding="utf-8",
    )
    print(
        f"minerva-v7-final | complete job=final_analysis rows={len(rows)} "
        f"blind_rows={len(blind['mapping'])} v7_test_accessed=True no_retuning=True",
        flush=True,
    )


def _comparison_final(rows, resamples, seed):
    """Use the protocol's final seeds instead of validation's fixed seed list."""
    from collections import defaultdict
    import statistics
    from analyze_minerva_v7_dpo_validation import _bootstrap

    lookup = {(row["system_id"], row["prompt_id"], row["seed"]): row for row in rows}
    prompts = sorted({row["prompt_id"] for row in rows})
    by_prompt = defaultdict(list)
    for prompt in prompts:
        for current_seed in EXPECTED_SEEDS:
            left = lookup[("stage_3", prompt, current_seed)]
            right = lookup[("dpo", prompt, current_seed)]
            by_prompt[prompt].append({
                metric: float(right[metric]) - float(left[metric]) for metric in METRICS
            })
    result = {
        "left_system_id": "stage_3", "right_system_id": "dpo",
        "paired_outputs": len(prompts) * len(EXPECTED_SEEDS),
    }
    for index, metric in enumerate(METRICS):
        means = {
            key: statistics.fmean(item[metric] for item in values)
            for key, values in by_prompt.items()
        }
        result[metric] = _bootstrap(means, resamples, seed + index, paired=True)
    return result


if __name__ == "__main__":
    main()
