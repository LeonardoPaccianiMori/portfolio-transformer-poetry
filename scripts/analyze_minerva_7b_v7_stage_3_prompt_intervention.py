#!/usr/bin/env python3
"""Verify and analyze the bounded Stage-3 prompt/stopping experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_memorization import (
    load_verified_sonnet_train_reference,
)
from sonnet_analysis.minerva_v7_prompt_intervention import (
    load_prompt_intervention_config,
)
from sonnet_analysis.minerva_v7_prompt_intervention_analysis import (
    analyze_prompt_intervention,
    build_prompt_intervention_blinded_sample,
    prompt_intervention_review_markdown,
)
from sonnet_analysis.minerva_v7_runtime import load_verified_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument(
        "--generation-dir",
        type=Path,
        default=Path(
            "artifacts/local/minerva_7b_v7_stage_3_prompt_intervention/authoritative"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/minerva_7b_v7_stage_3_prompt_intervention.json"),
    )
    parser.add_argument("--memorization-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=9321)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--blinded-prompts", type=int, default=30)
    parser.add_argument("--blinded-selection-seed", type=int, default=9327)
    args = parser.parse_args()

    config = load_prompt_intervention_config(args.config)
    state = load_verified_state(args.state_audit, str(config["state_id"]))
    records, reference = load_verified_sonnet_train_reference(
        args.memorization_reference
    )
    print(
        "minerva-v7-research | start job=stage_3_prompt_intervention_analysis "
        "outputs=3840 bootstrap_resamples="
        f"{args.bootstrap_resamples}",
        flush=True,
    )
    analysis = analyze_prompt_intervention(
        output_dir=args.generation_dir,
        expected_state_identity=str(state["state_identity_sha256"]),
        memorization_records=records,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
        progress=lambda message: print(
            f"minerva-v7-research | memorization {message}", flush=True
        ),
    )
    analysis["memorization_reference_manifest_sha256"] = reference["manifest_sha256"]
    blind = build_prompt_intervention_blinded_sample(
        analysis=analysis,
        selected_prompt_count=args.blinded_prompts,
        selection_seed=args.blinded_selection_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prompt_intervention_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "prompt_intervention_blind_mapping.private.json").write_text(
        json.dumps(blind, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "prompt_intervention_blinded_review.md").write_text(
        prompt_intervention_review_markdown(blind), encoding="utf-8"
    )
    print(
        "minerva-v7-research | complete outputs=3840 summaries=4 "
        f"blinded_rows={blind['sample_rows']} output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
