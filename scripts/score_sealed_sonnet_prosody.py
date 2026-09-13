#!/usr/bin/env python3
"""Score the sealed Stage-3 and DPO outputs with the sonnet prosody checker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_evaluation.sonnet_prosody_sealed import (
    aggregate,
    format_report,
    load_sealed_outputs,
    mcnemar,
    paired_comparison,
    pair_records,
    score_records,
)

CONTINUOUS_FIELDS = (
    "hendecasyllable_lines",
    "failed_lines",
    "uncertain_lines",
    "perfect_rhyme_pairs",
    "soft_rhyme_pairs",
    "rhyme_score",
)
BINARY_FIELDS = (
    "structure_ok",
    "stanza_pattern_ok",
    "all_lines_valid",
    "quatrain_ok",
    "tercet_ok",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-dir",
        type=Path,
        default=ROOT / "artifacts/local/minerva_7b_v7_dpo/final_test/generation",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=ROOT / "reports/sonnet_prosody_retroactive_v1.md",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=ROOT / "reports/sonnet_prosody_retroactive_v1.json",
    )
    parser.add_argument(
        "--scores-jsonl",
        type=Path,
        default=ROOT / "artifacts/local/sonnet_prosody/sealed_scores_v1.jsonl",
    )
    parser.add_argument("--skip-hash-check", action="store_true")
    return parser.parse_args()


def repository_relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    complete = json.loads((args.generation_dir / "complete.json").read_text(encoding="utf-8"))
    outputs = load_sealed_outputs(
        args.generation_dir, verify_hashes=not args.skip_hash_check
    )
    scored = score_records(outputs)
    pairs = pair_records(scored)
    stage_3 = aggregate(scored, "stage_3")
    dpo = aggregate(scored, "dpo")
    comparisons = [paired_comparison(pairs, field) for field in CONTINUOUS_FIELDS]
    mcnemar_results = [mcnemar(pairs, field) for field in BINARY_FIELDS]

    args.scores_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.scores_jsonl.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report_md = format_report(
        output_count=len(scored),
        pair_count=len(pairs),
        protocol_sha256=str(complete.get("final_protocol_sha256", "unknown")),
        adapter_sha256=str(complete.get("dpo_adapter_identity_sha256", "unknown")),
        stage_3=stage_3,
        dpo=dpo,
        comparisons=comparisons,
        mcnemar_results=mcnemar_results,
    )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(report_md, encoding="utf-8")

    report = {
        "schema_version": 1,
        "generation_dir": repository_relative(args.generation_dir),
        "protocol_sha256": complete.get("final_protocol_sha256"),
        "adapter_sha256": complete.get("dpo_adapter_identity_sha256"),
        "output_count": len(scored),
        "pair_count": len(pairs),
        "system_metrics": {"stage_3": stage_3, "dpo": dpo},
        "paired_comparisons": comparisons,
        "mcnemar": mcnemar_results,
        "scores_jsonl": repository_relative(args.scores_jsonl),
        "checker_status": "reviewed_conservative_2026-09-13",
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"outputs: {len(scored)} pairs: {len(pairs)}")
    print(
        "all-lines-valid rate: "
        f"stage_3={stage_3['all_lines_valid_rate']:.4f} dpo={dpo['all_lines_valid_rate']:.4f}"
    )
    print(
        "mean rhyme score: "
        f"stage_3={stage_3['mean_rhyme_score']:.4f} dpo={dpo['mean_rhyme_score']:.4f}"
    )
    print(f"wrote report: {repository_relative(args.report_md)}")


if __name__ == "__main__":
    main()
