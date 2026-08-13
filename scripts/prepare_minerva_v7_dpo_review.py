#!/usr/bin/env python3
"""Freeze three-judge assignments and a 20-pair human calibration packet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sonnet_analysis.minerva_v7_dpo_preferences import (
    build_judge_assignments,
    build_user_calibration_packet,
    write_json_atomic,
)


RUBRIC = (
    "Score A and B from 1--5 for grammar, coherence/progression, historical "
    "register, poetic force, sonnet form, and volta/closure. Then choose A, B, "
    "or tie and cite concrete textual evidence. Penalize meta-text, collapse, "
    "and incomplete syntax. For completion contrasts, require genuine closure; "
    "do not reward punctuation alone."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--judge-ids", nargs=3,
        default=("ai_judge_1", "ai_judge_2", "ai_judge_3"),
    )
    parser.add_argument("--calibration-count", type=int, default=20)
    args = parser.parse_args()
    print(
        "minerva-v7-dpo | start job=prepare_blind_review device=cpu "
        "total_steps=2 progress_interval=one_packet",
        flush=True,
    )
    ordinary = _load(args.analysis_dir / "blinded_pairs.json")
    completion = _load(
        args.analysis_dir / "completion_contrast_blinded_pairs.json"
    )
    combined = _combine_pair_packets(ordinary, completion)
    assignments = build_judge_assignments(
        combined, judge_ids=tuple(args.judge_ids)
    )
    calibration = build_user_calibration_packet(
        combined, count=args.calibration_count
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignment_hash = write_json_atomic(
        args.output_dir / "judge_assignments.json", assignments
    )
    calibration_hash = write_json_atomic(
        args.output_dir / "user_calibration_packet.json", calibration
    )
    (args.output_dir / "user_calibration_packet.md").write_text(
        _calibration_markdown(calibration), encoding="utf-8"
    )
    (args.output_dir / "judge_rubric.md").write_text(
        f"# Minerva V7 DPO Blinded Judge Rubric\n\n{RUBRIC}\n",
        encoding="utf-8",
    )
    print(
        "minerva-v7-dpo | complete "
        f"pairs={combined['pair_count']} assignments={assignments['assignment_count']} "
        f"calibration_pairs={calibration['pair_count']} "
        f"assignment_sha256={assignment_hash} calibration_sha256={calibration_hash} "
        "v7_test_accessed=False training_performed=False",
        flush=True,
    )


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _combine_pair_packets(ordinary: dict, completion: dict) -> dict:
    rows = [
        {**row, "pair_type": row.get("pair_type", "clean_literary_comparison")}
        for row in ordinary.get("pairs", [])
    ] + list(completion.get("pairs", []))
    identities = [str(row["pair_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("ordinary and completion DPO pairs overlap")
    return {
        "preference_version": ordinary["preference_version"],
        "pair_count": len(rows),
        "pairs": sorted(rows, key=lambda row: str(row["pair_id"])),
        "component_pair_counts": {
            "clean_literary_comparison": int(ordinary["pair_count"]),
            "terminal_completion_contrast": int(completion["pair_count"]),
        },
        "v7_test_accessed": False,
    }


def _calibration_markdown(packet: dict) -> str:
    lines = [
        "# Human--AI Calibration Packet",
        "",
        RUBRIC,
        "",
        "For every pair, record `A`, `B`, or `tie` and a short reason. The AI "
        "majority answers are intentionally absent.",
        "",
    ]
    for index, pair in enumerate(packet["pairs"], start=1):
        lines.extend(
            [
                f"## Pair {index}: `{pair['pair_id']}`",
                "",
                f"Pair type: `{pair.get('pair_type', 'clean_literary_comparison')}`",
                "",
                "### Candidate A",
                "",
                str(pair["candidate_a"]),
                "",
                "### Candidate B",
                "",
                str(pair["candidate_b"]),
                "",
                "Scores for A (1--5):",
                "",
                "- Grammar:",
                "- Coherence/progression:",
                "- Historical register:",
                "- Poetic force:",
                "- Sonnet form:",
                "- Volta/closure:",
                "- Terminal syntax genuinely complete (yes/no):",
                "",
                "Scores for B (1--5):",
                "",
                "- Grammar:",
                "- Coherence/progression:",
                "- Historical register:",
                "- Poetic force:",
                "- Sonnet form:",
                "- Volta/closure:",
                "- Terminal syntax genuinely complete (yes/no):",
                "",
                "Preference (A/B/tie):",
                "",
                "Reason:",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
