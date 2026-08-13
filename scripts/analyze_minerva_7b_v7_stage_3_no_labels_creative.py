#!/usr/bin/env python3
"""Verify and analyze the Stage-3 no-labels plus creative-decoding cell."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_no_labels_creative_analysis import (
    analyze_no_labels_creative,
    build_no_labels_creative_blinded_sample,
    no_labels_creative_review_markdown,
)
from sonnet_analysis.minerva_v7_memorization import (
    load_verified_sonnet_train_reference,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-cell-dir", type=Path, required=True)
    parser.add_argument("--high-volume-stage-3-dir", type=Path, required=True)
    parser.add_argument("--prompt-intervention-dir", type=Path, required=True)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--memorization-reference", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=10321)
    parser.add_argument("--blind-prompts", type=int, default=40)
    parser.add_argument("--blind-seed", type=int, default=10327)
    args = parser.parse_args()

    audit = json.loads(args.state_audit.read_text(encoding="utf-8"))
    matches = [row for row in audit["states"] if row["state_id"] == "stage_3_selected"]
    if len(matches) != 1 or matches[0].get("status") != "complete":
        raise ValueError("state audit lacks one complete Stage-3 selected state")
    identity = str(matches[0]["state_identity_sha256"])
    memorization_records, memorization_manifest = (
        load_verified_sonnet_train_reference(args.memorization_reference)
    )
    print(
        "minerva-v7-research | start job=no_labels_creative_analysis "
        f"bootstrap_resamples={args.bootstrap_resamples} blind_prompts={args.blind_prompts}",
        flush=True,
    )
    analysis = analyze_no_labels_creative(
        new_cell_dir=args.new_cell_dir,
        high_volume_stage_3_dir=args.high_volume_stage_3_dir,
        prompt_intervention_dir=args.prompt_intervention_dir,
        expected_state_identity=identity,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=0.95,
        memorization_records=memorization_records,
        progress=lambda message: print(
            f"minerva-v7-research | memorization {message}", flush=True
        ),
    )
    analysis["memorization_reference_manifest_sha256"] = memorization_manifest[
        "manifest_sha256"
    ]
    blind = build_no_labels_creative_blinded_sample(
        analysis=analysis, prompt_count=args.blind_prompts,
        selection_seed=args.blind_seed,
    )
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
        no_labels_creative_review_markdown(blind), encoding="utf-8"
    )
    print(
        "minerva-v7-research | complete rows=2880 paired_comparisons=2 "
        f"blinded_rows={blind['sample_rows']} v7_test_accessed=False",
        flush=True,
    )


if __name__ == "__main__":
    main()
