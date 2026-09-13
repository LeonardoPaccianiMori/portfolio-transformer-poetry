#!/usr/bin/env python3
"""Train the plan-following LoRA SFT arm on the V8 train sonnets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.plan_then_poem_validation import planned_prompt
from sonnet_analysis.minerva_v7_exploratory_prompts import (
    validate_exploratory_prompt_manifest,
)
from sonnet_analysis.minerva_v7_runtime import load_verified_state
from sonnet_evaluation.rhyme_lexicon import load_lexicon
from sonnet_training.form_targeted_data import load_train_rows
from sonnet_training.plan_following_sft import (
    build_examples,
    split_examples,
    train_plan_following_sft,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/plan_following_sft.json",
    )
    parser.add_argument(
        "--state-audit",
        type=Path,
        default=ROOT / "artifacts/local/verifier_labelled_dpo/state_audit.json",
    )
    parser.add_argument(
        "--verifier-adapter",
        type=Path,
        default=ROOT / "artifacts/local/verifier_labelled_dpo/training/best_adapter.pt",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=ROOT / "data/metadata/rhyme_lexicon_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/local/plan_following_sft/training",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    config = json.loads(args.config.read_text(encoding="utf-8"))
    state = load_verified_state(args.state_audit, "stage_3_selected")
    model_dir = Path(str(state["model_dir"]))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = validate_exploratory_prompt_manifest(
        ROOT / "configs/minerva_7b_v7_exploratory_prompts.json",
        expected_sha256="2f33aa518aa61c11193831e53b07fd3bd861a72bf68bb23c0e0e5b1a13b1d0c7",
    )["prompts"]
    lexicon = load_lexicon(args.lexicon)
    from sonnet_analysis.plan_then_poem_validation import build_plans

    plans, _ = build_plans(prompts, lexicon)
    prompt_hashes = {
        str(prompt.get("source_logical_sha256", "")) for prompt in prompts
    }
    rows = load_train_rows(
        ROOT / "data/processed/sonnets_expanded_v8/sonnets_manifest.csv"
    )
    excluded = {
        row["unit_id"] for row in rows if row["logical_sha256"] in prompt_hashes
    }
    if excluded:
        print(
            f"plan-following | excluding {len(excluded)} train rows that overlap "
            "the evaluation prompts",
            flush=True,
        )
    examples, skipped = build_examples(
        rows,
        ROOT,
        tokenizer=tokenizer,
        max_sequence_tokens=int(config["data"]["max_sequence_tokens"]),
        limit=args.limit,
        excluded_unit_ids=excluded,
        progress=lambda message: print(f"plan-following | {message}", flush=True),
    )
    train, validation = split_examples(
        examples,
        validation_fraction=float(config["data"]["validation_fraction"]),
        seed=int(config["training"]["seed"]),
    )
    probe_prompts = prompts[: int(config["probe"]["prompt_count"])]
    probe_jobs = []
    for prompt in probe_prompts:
        words = plans[str(prompt["id"])]["words"]
        probe_jobs.append(
            {
                "prompt": dict(prompt),
                "seed": 5200,
                "rendered_prompt": planned_prompt(
                    tokenizer, str(prompt["opening_line"]), words
                ),
                "planned_words": list(words),
            }
        )
    print(
        "plan-following | start "
        f"examples={len(train)} validation={len(validation)} "
        f"skipped={skipped} batch={config['training']['batch_size']} "
        f"accumulation={config['training']['gradient_accumulation_steps']}",
        flush=True,
    )
    report = train_plan_following_sft(
        state=state,
        verifier_adapter_path=args.verifier_adapter,
        examples=train,
        validation_examples=validation,
        probe_jobs=probe_jobs,
        output_dir=args.output_dir,
        config=config,
        progress=lambda message: print(f"plan-following | {message}", flush=True),
    )
    print(
        "plan-following | complete "
        f"steps={report['steps']}/{report['planned_steps']} "
        f"probe_key={report['probe']['mean_key_match_rate'] if report['probe'] else None} "
        f"cost_usd={report['estimated_cost_usd']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
